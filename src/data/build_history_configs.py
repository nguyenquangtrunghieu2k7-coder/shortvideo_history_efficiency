
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = (
    ROOT / "data/processed/evaluation_events.parquet"
)

STANDARD_FILES = [
    ROOT
    / "data/raw/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv",
    ROOT
    / "data/raw/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv",
]

OUTPUT_DIR = (
    ROOT / "data/processed/phase7"
)


# =============================================================================
# FROZEN CONFIGURATION — FROM PHASE 5
# =============================================================================

W_DAYS = 1
L_MAX = 2048

W_MS = W_DAYS * 24 * 60 * 60 * 1000

K_VALUES = [
    0,
    1,
    5,
    10,
    20,
    50,
    100,
    "Full",
]


# =============================================================================
# LOAD EVALUATION EVENTS
# =============================================================================

def load_evaluation_events():

    print("Loading evaluation events...")

    df = pd.read_parquet(
        EVAL_FILE,
        columns=[
            "evaluation_id",
            "user_id",
            "video_id",
            "time_ms",
            "long_view",
        ],
    )

    df = df.sort_values(
        ["time_ms", "evaluation_id"]
    ).reset_index(drop=True)

    print(
        f"  Evaluation events: {len(df):,}"
    )

    if not df["evaluation_id"].is_unique:
        raise ValueError(
            "evaluation_id is not unique."
        )

    return df


# =============================================================================
# LOAD STANDARD INTERACTIONS
# =============================================================================

def load_standard_interactions():

    print("\nLoading standard interactions...")

    frames = []

    for path in STANDARD_FILES:

        print(f"  Loading: {path.name}")

        df = pd.read_csv(
            path,
            usecols=[
                "user_id",
                "video_id",
                "time_ms",
            ],
        )

        print(
            f"    Rows: {len(df):,}"
        )

        frames.append(df)

    interactions = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"  Total standard interactions: "
        f"{len(interactions):,}"
    )

    return interactions


# =============================================================================
# BUILD ONE FULL 1-DAY HISTORY
# =============================================================================

def build_one_day_histories(
    eval_df,
    interactions,
):

    print("\nBuilding H_t^(1 day) once...")

    # Only interactions strictly before the evaluation event
    # are eligible for history.
    #
    # We process users independently and keep only the recent
    # 1-day window needed by the evaluation events.

    interactions = interactions.sort_values(
        ["user_id", "time_ms"]
    ).reset_index(drop=True)

    eval_by_user = defaultdict(list)

    for idx, row in eval_df.iterrows():
        eval_by_user[row["user_id"]].append(
            (idx, row["time_ms"])
        )

    histories = [None] * len(eval_df)

    grouped = interactions.groupby(
        "user_id",
        sort=False,
    )

    for user_id, user_events in grouped:

        if user_id not in eval_by_user:
            continue

        user_events = user_events[
            ["video_id", "time_ms"]
        ].to_numpy()

        eval_events = eval_by_user[user_id]

        left = 0
        right = 0

        n = len(user_events)

        # Events are already sorted by time.
        for eval_idx, eval_time in eval_events:

            # Add all interactions strictly before t.
            while (
                right < n
                and user_events[right][1] < eval_time
            ):
                right += 1

            # Remove events outside [t-W, t).
            lower_bound = eval_time - W_MS

            while (
                left < right
                and user_events[left][1] < lower_bound
            ):
                left += 1

            window = user_events[left:right]

            histories[eval_idx] = [
                int(video_id)
                for video_id, _ in window
            ]

    # Evaluation users with no interactions.
    for i in range(len(histories)):
        if histories[i] is None:
            histories[i] = []

    return histories


# =============================================================================
# BUILD K CONFIGURATION
# =============================================================================

def make_k_history(
    full_history,
    k,
):

    if k == 0:
        return []

    if k == "Full":
        return full_history[-L_MAX:]

    return full_history[-min(k, len(full_history)):]


# =============================================================================
# AUDIT ONE CONFIGURATION
# =============================================================================

def audit_config(
    histories,
    k,
):

    lengths = [
        len(history)
        for history in histories
    ]

    max_allowed = (
        L_MAX
        if k == "Full"
        else k
    )

    if max(lengths, default=0) > max_allowed:
        raise ValueError(
            f"K={k}: history exceeds allowed length."
        )

    if k == 0:
        if any(lengths):
            raise ValueError(
                "K=0 must contain only empty histories."
            )

    return {
        "K": k,
        "evaluation_events": len(histories),
        "mean_history_len": sum(lengths) / len(lengths),
        "median_history_len": float(
            pd.Series(lengths).median()
        ),
        "max_history_len": max(lengths, default=0),
        "zero_history": sum(
            length == 0
            for length in lengths
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("PHASE 7 — BUILD HISTORY CONFIGURATIONS")
    print("=" * 80)

    print("\nFrozen configuration:")
    print(f"  W_days : {W_DAYS}")
    print(f"  L_max  : {L_MAX}")
    print(
        "  K      : "
        "{0, 1, 5, 10, 20, 50, 100, Full}"
    )

    if L_MAX < 100:
        raise ValueError(
            "L_max must be >= 100."
        )

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    eval_df = load_evaluation_events()

    interactions = load_standard_interactions()

    # -------------------------------------------------------------------------
    # BUILD ONE H^(W)
    # -------------------------------------------------------------------------

    full_histories = build_one_day_histories(
        eval_df,
        interactions,
    )

    print(
        "\nH_t^(1 day) reconstruction complete."
    )

    full_lengths = [
        len(history)
        for history in full_histories
    ]

    print(
        f"  Mean length:   "
        f"{sum(full_lengths) / len(full_lengths):.2f}"
    )

    print(
        f"  Median length: "
        f"{pd.Series(full_lengths).median():.2f}"
    )

    print(
        f"  Max length:    "
        f"{max(full_lengths):,}"
    )

    print(
        f"  Zero history:  "
        f"{sum(x == 0 for x in full_lengths):,}"
    )

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    # -------------------------------------------------------------------------
    # BUILD K VALUES BY SLICING
    # -------------------------------------------------------------------------

    for k in K_VALUES:

        print("\n" + "-" * 80)
        print(f"Building K = {k}")
        print("-" * 80)

        histories = [
            make_k_history(
                history,
                k,
            )
            for history in full_histories
        ]

        audit = audit_config(
            histories,
            k,
        )

        summary.append(audit)

        output = eval_df[
            [
                "evaluation_id",
                "user_id",
                "video_id",
                "time_ms",
                "long_view",
            ]
        ].copy()

        output["history"] = histories

        output["history_len"] = [
            len(history)
            for history in histories
        ]

        # Availability mask:
        #
        # K=0 -> no history requested
        # K>0 -> 1 iff at least one valid history event exists

        if k == 0:
            output["history_available"] = 0
        else:
            output["history_available"] = (
                output["history_len"] > 0
            ).astype("int8")

        filename = (
            "history_KFull.parquet"
            if k == "Full"
            else f"history_K{k}.parquet"
        )

        output_path = OUTPUT_DIR / filename

        output.to_parquet(
            output_path,
            index=False,
        )

        print(
            f"  Events:              "
            f"{audit['evaluation_events']:,}"
        )

        print(
            f"  Mean history length:  "
            f"{audit['mean_history_len']:.2f}"
        )

        print(
            f"  Median history length:"
            f" {audit['median_history_len']:.2f}"
        )

        print(
            f"  Max history length:   "
            f"{audit['max_history_len']:,}"
        )

        print(
            f"  Zero-history events:  "
            f"{audit['zero_history']:,}"
        )

        print(
            f"  Saved: {output_path}"
        )

    # -------------------------------------------------------------------------
    # SAVE SUMMARY
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(summary)

    summary_path = (
        OUTPUT_DIR / "history_config_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SAVE FROZEN CONFIG
    # -------------------------------------------------------------------------

    config_path = (
        OUTPUT_DIR / "history_config.yaml"
    )

    config_path.write_text(
        """# Phase 7 frozen history configuration

W_days: 1
L_max: 2048

K_values:
  - 0
  - 1
  - 5
  - 10
  - 20
  - 50
  - 100
  - Full

protocol:
  history_definition: "[t-W, t)"
  strict_temporal_order: true
  fixed_evaluation_set: true
  test_data_used: false
  reconstruct_history_once: true
  derive_K_by_slicing: true
""",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 7 HISTORY BUILD COMPLETE")
    print("=" * 80)

    print("\nSaved artifacts:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path}")

    print("\nProtocol:")
    print("  W = 1 day:             FROZEN")
    print("  L_max = 2048:          FROZEN")
    print("  Fixed evaluation set:  PASS")
    print("  Test data used:        NO")
    print("  H^(W) built once:      PASS")
    print("  K derived by slicing:  PASS")


if __name__ == "__main__":
    main()

