from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/KuaiRand-1K/data"
PROCESSED = ROOT / "data/processed"
OUTPUT = PROCESSED / "phase5"

TRAIN_FILE = PROCESSED / "train.parquet"
VALIDATION_FILE = PROCESSED / "validation.parquet"

STANDARD_FILES = [
    RAW / "log_standard_4_08_to_4_21_1k.csv",
    RAW / "log_standard_4_22_to_5_08_1k.csv",
]


# =============================================================================
# PHASE 5 DESIGN
# =============================================================================

# Candidate temporal history windows from the proposal.
WINDOWS_DAYS = [1, 3, 7, 14]

# Candidate maximum history lengths.
#
# These values are used to study coverage only.
# Final L_max is selected after inspecting development EDA.
L_MAX_CANDIDATES = [
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
]


# =============================================================================
# LOAD DEVELOPMENT EVENTS
# =============================================================================

def load_development_events():
    """
    Development data = Train + Validation.

    Test is intentionally never loaded.
    """

    print("Loading development evaluation events...")

    columns = [
        "eval_idx",
        "user_id",
        "time_ms",
    ]

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=columns,
    )

    validation = pd.read_parquet(
        VALIDATION_FILE,
        columns=columns,
    )

    development = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    development = development.sort_values(
        ["user_id", "time_ms"]
    ).reset_index(drop=True)

    print(f"  Train events:        {len(train):,}")
    print(f"  Validation events:   {len(validation):,}")
    print(f"  Development events:  {len(development):,}")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if development["eval_idx"].duplicated().any():
        raise ValueError(
            "Duplicate eval_idx found in development events."
        )

    if development["user_id"].isna().any():
        raise ValueError(
            "Missing user_id found in development events."
        )

    if development["time_ms"].isna().any():
        raise ValueError(
            "Missing time_ms found in development events."
        )

    return development


# =============================================================================
# LOAD STANDARD INTERACTIONS
# =============================================================================

def load_standard_interactions():
    """
    Load only user_id and time_ms from standard interaction logs.

    We do not need video IDs or feedback features for Phase 5 because the
    objective is only to measure the number of historical interactions inside
    each temporal window.
    """

    print("\nLoading standard interactions...")

    user_times = {}

    total_rows = 0

    for path in STANDARD_FILES:

        print(f"  Loading: {path.name}")

        file_rows = 0

        for chunk in pd.read_csv(
            path,
            usecols=[
                "user_id",
                "time_ms",
            ],
            dtype={
                "user_id": "int32",
                "time_ms": "int64",
            },
            chunksize=500_000,
        ):

            file_rows += len(chunk)
            total_rows += len(chunk)

            for user_id, times in chunk.groupby("user_id")["time_ms"]:

                arr = times.to_numpy(
                    dtype=np.int64,
                    copy=True,
                )

                if user_id in user_times:
                    user_times[user_id].append(arr)
                else:
                    user_times[user_id] = [arr]

        print(f"    Rows: {file_rows:,}")

    print(f"  Total standard interactions: {total_rows:,}")
    print(f"  Users with interactions:      {len(user_times):,}")

    # -------------------------------------------------------------------------
    # Merge chunks and sort timestamps for every user.
    # -------------------------------------------------------------------------

    print("  Sorting interaction timestamps...")

    for user_id in user_times:

        arr = np.concatenate(
            user_times[user_id]
        )

        arr.sort()

        user_times[user_id] = arr

    return user_times


# =============================================================================
# COMPUTE HISTORY LENGTHS
# =============================================================================

def compute_history_lengths(
    events,
    user_times,
):
    """
    Compute |H_t^W| for every development evaluation event.

    For an evaluation event at timestamp t:

        H_t^W = {h : t - W <= timestamp(h) < t}

    The upper boundary is strictly before the evaluation event.
    """

    print("\nComputing history lengths...")

    result = events[
        [
            "eval_idx",
            "user_id",
            "time_ms",
        ]
    ].copy()

    for days in WINDOWS_DAYS:

        print(f"  W = {days} day(s)")

        window_ms = (
            days
            * 24
            * 60
            * 60
            * 1000
        )

        lengths = np.zeros(
            len(events),
            dtype=np.int32,
        )

        for i, row in enumerate(
            events.itertuples(index=False)
        ):

            times = user_times.get(
                row.user_id
            )

            if times is None:
                continue

            # First interaction >= evaluation timestamp.
            #
            # Therefore interactions at exactly t are excluded.
            right = np.searchsorted(
                times,
                row.time_ms,
                side="left",
            )

            # First interaction >= t - W.
            left = np.searchsorted(
                times,
                row.time_ms - window_ms,
                side="left",
            )

            lengths[i] = right - left

        result[
            f"history_{days}d"
        ] = lengths

    return result


# =============================================================================
# DISTRIBUTION STATISTICS
# =============================================================================

def build_distribution_table(
    history_lengths,
):
    """
    Compute the statistics required by the proposal.
    """

    rows = []

    for days in WINDOWS_DAYS:

        values = history_lengths[
            f"history_{days}d"
        ].to_numpy()

        rows.append(
            {
                "W": f"{days}d",

                "median": np.percentile(
                    values,
                    50,
                ),

                "P75": np.percentile(
                    values,
                    75,
                ),

                "P90": np.percentile(
                    values,
                    90,
                ),

                "P95": np.percentile(
                    values,
                    95,
                ),

                "P99": np.percentile(
                    values,
                    99,
                ),

                "max": int(
                    values.max()
                ),

                "mean": values.mean(),

                "zero_history": int(
                    np.sum(values == 0)
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# L_MAX COVERAGE
# =============================================================================

def build_lmax_coverage_table(
    history_lengths,
):
    """
    Measure the percentage of development events whose complete history fits
    inside each candidate L_max.

        coverage = P(|H_t^W| <= L_max)

    This allows L_max to be selected from empirical development evidence.
    """

    rows = []

    for days in WINDOWS_DAYS:

        values = history_lengths[
            f"history_{days}d"
        ].to_numpy()

        for l_max in L_MAX_CANDIDATES:

            covered = values <= l_max

            coverage = covered.mean()

            rows.append(
                {
                    "W": f"{days}d",
                    "L_max": l_max,
                    "coverage": coverage,
                    "truncated_events": int(
                        np.sum(values > l_max)
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(
    distribution,
    coverage,
):
    """
    Print the main Phase 5 results.
    """

    print("\n" + "=" * 80)
    print("HISTORY LENGTH DISTRIBUTION")
    print("=" * 80)

    display = distribution.copy()

    for column in [
        "median",
        "P75",
        "P90",
        "P95",
        "P99",
        "mean",
    ]:
        display[column] = display[
            column
        ].round(2)

    print(
        display.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("L_MAX COVERAGE")
    print("=" * 80)

    display = coverage.copy()

    display["coverage"] = (
        display["coverage"] * 100
    ).round(2).astype(str) + "%"

    print(
        display.to_string(
            index=False
        )
    )


# =============================================================================
# SAVE PROCESSED ARTIFACTS
# =============================================================================

def save_outputs(
    development_events,
    history_lengths,
    distribution,
    coverage,
):
    """
    Save all reusable Phase 5 artifacts under data/processed/phase5/.
    """

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_file = (
        OUTPUT
        / "development_events.parquet"
    )

    history_file = (
        OUTPUT
        / "development_history_lengths.parquet"
    )

    distribution_file = (
        OUTPUT
        / "history_window_stats.csv"
    )

    coverage_file = (
        OUTPUT
        / "history_lmax_coverage.csv"
    )

    # Development population used for selection.
    development_events.to_parquet(
        development_file,
        index=False,
    )

    # Per-event history lengths for every candidate W.
    history_lengths.to_parquet(
        history_file,
        index=False,
    )

    # Proposal-required statistics.
    distribution.to_csv(
        distribution_file,
        index=False,
    )

    # Coverage of candidate L_max values.
    coverage.to_csv(
        coverage_file,
        index=False,
    )

    print("\nSaved Phase 5 artifacts:")

    print(
        f"  {development_file}"
    )

    print(
        f"  {history_file}"
    )

    print(
        f"  {distribution_file}"
    )

    print(
        f"  {coverage_file}"
    )


# =============================================================================
# FINAL AUDIT
# =============================================================================

def run_final_audit(
    development_events,
    history_lengths,
):
    """
    Basic sanity checks before declaring the development EDA complete.
    """

    print("\n" + "=" * 80)
    print("PHASE 5 DATA AUDIT")
    print("=" * 80)

    # Expected development size.
    expected = 30_118 + 6_454

    if len(development_events) == expected:
        print(
            f"Development event count: PASS ({len(development_events):,})"
        )
    else:
        print(
            f"Development event count: FAIL "
            f"({len(development_events):,}, expected {expected:,})"
        )

    # Unique evaluation events.
    if not development_events["eval_idx"].duplicated().any():
        print("Development eval_idx uniqueness: PASS")
    else:
        print("Development eval_idx uniqueness: FAIL")

    # Check all history lengths are non-negative.
    history_columns = [
        f"history_{days}d"
        for days in WINDOWS_DAYS
    ]

    negative_count = 0

    for column in history_columns:
        negative_count += int(
            (history_lengths[column] < 0).sum()
        )

    if negative_count == 0:
        print("History length non-negative: PASS")
    else:
        print(
            f"History length non-negative: FAIL "
            f"({negative_count} violations)"
        )

    # Check maximum does not exceed int32.
    max_value = max(
        history_lengths[column].max()
        for column in history_columns
    )

    if max_value < np.iinfo(np.int32).max:
        print(
            f"History length integer range: PASS (max={max_value:,})"
        )
    else:
        print("History length integer range: FAIL")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("PHASE 5 — CHỌN W, L_MAX")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Development population
    # -------------------------------------------------------------------------

    development_events = load_development_events()

    # -------------------------------------------------------------------------
    # 2. Standard interaction history
    # -------------------------------------------------------------------------

    user_times = load_standard_interactions()

    # -------------------------------------------------------------------------
    # 3. Compute |H_t^W|
    # -------------------------------------------------------------------------

    history_lengths = compute_history_lengths(
        development_events,
        user_times,
    )

    # -------------------------------------------------------------------------
    # 4. Distribution statistics
    # -------------------------------------------------------------------------

    distribution = build_distribution_table(
        history_lengths
    )

    # -------------------------------------------------------------------------
    # 5. L_max coverage
    # -------------------------------------------------------------------------

    coverage = build_lmax_coverage_table(
        history_lengths
    )

    # -------------------------------------------------------------------------
    # 6. Audit
    # -------------------------------------------------------------------------

    run_final_audit(
        development_events,
        history_lengths,
    )

    # -------------------------------------------------------------------------
    # 7. Print
    # -------------------------------------------------------------------------

    print_results(
        distribution,
        coverage,
    )

    # -------------------------------------------------------------------------
    # 8. Save
    # -------------------------------------------------------------------------

    save_outputs(
        development_events,
        history_lengths,
        distribution,
        coverage,
    )

    # -------------------------------------------------------------------------
    # 9. Final status
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 5 DEVELOPMENT EDA COMPLETE")
    print("=" * 80)

    print(
        "\nSelection status:"
        "\n  W*      : NOT YET FROZEN"
        "\n  L_max*  : NOT YET FROZEN"
    )

    print(
        "\nProtocol:"
        "\n  Development data = Train + Validation"
        "\n  Test data        = NOT USED"
        "\n  Candidate W      = {1d, 3d, 7d, 14d}"
    )

    print(
        "\nNext step:"
        "\n  Use the saved development statistics to select and"
        "\n  freeze W* and L_max* before the final test experiment."
    )


if __name__ == "__main__":
    main()

