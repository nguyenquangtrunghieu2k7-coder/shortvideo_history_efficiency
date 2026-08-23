# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"

RANDOM_FILE = DATA_DIR / "log_random_4_22_to_5_08_1k.csv"
STANDARD_1_FILE = DATA_DIR / "log_standard_4_08_to_4_21_1k.csv"
STANDARD_2_FILE = DATA_DIR / "log_standard_4_22_to_5_08_1k.csv"

USER_COL = "user_id"
VIDEO_COL = "video_id"
TIME_COL = "time_ms"


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fmt_ts(ts):
    return pd.to_datetime(ts, unit="ms")


def load_logs():
    print("Loading logs...")

    random_log = pd.read_csv(RANDOM_FILE)
    standard_1 = pd.read_csv(STANDARD_1_FILE)
    standard_2 = pd.read_csv(STANDARD_2_FILE)

    standard = pd.concat(
        [standard_1, standard_2],
        ignore_index=True,
    )

    print(f"Random log:   {len(random_log):,} rows")
    print(f"Standard log: {len(standard):,} rows")

    return random_log, standard


# ============================================================
# 1. TEMPORAL COVERAGE
# ============================================================

def audit_temporal_coverage(random_log, standard):
    header("1. TEMPORAL COVERAGE")

    for name, df in [
        ("STANDARD", standard),
        ("RANDOM", random_log),
    ]:
        min_ts = df[TIME_COL].min()
        max_ts = df[TIME_COL].max()

        print(f"\n{name}")
        print(f"  Min timestamp: {min_ts} -> {fmt_ts(min_ts)}")
        print(f"  Max timestamp: {max_ts} -> {fmt_ts(max_ts)}")
        print(f"  Duration: {fmt_ts(max_ts) - fmt_ts(min_ts)}")


# ============================================================
# 2. INTERACTIONS PER USER
# ============================================================

def audit_interactions_per_user(standard):
    header("2. INTERACTIONS PER USER")

    counts = standard.groupby(USER_COL).size()

    print(f"Users: {len(counts):,}")
    print(f"Min:   {counts.min():,}")
    print(f"Mean:  {counts.mean():,.2f}")
    print(f"Median:{counts.median():,.2f}")
    print(f"P90:   {counts.quantile(0.90):,.2f}")
    print(f"P95:   {counts.quantile(0.95):,.2f}")
    print(f"P99:   {counts.quantile(0.99):,.2f}")
    print(f"Max:   {counts.max():,}")


# ============================================================
# 3. RANDOM EXPOSURES PER USER
# ============================================================

def audit_random_exposures_per_user(random_log):
    header("3. RANDOM EXPOSURES PER USER")

    counts = random_log.groupby(USER_COL).size()

    print(f"Users with random exposure: {len(counts):,}")
    print(f"Min:    {counts.min():,}")
    print(f"Mean:   {counts.mean():,.2f}")
    print(f"Median: {counts.median():,.2f}")
    print(f"P90:    {counts.quantile(0.90):,.2f}")
    print(f"P95:    {counts.quantile(0.95):,.2f}")
    print(f"P99:    {counts.quantile(0.99):,.2f}")
    print(f"Max:    {counts.max():,}")


# ============================================================
# 4. RANDOM EXPOSURE DATE DISTRIBUTION
# ============================================================

def audit_random_date_distribution(random_log):
    header("4. RANDOM EXPOSURE DATE DISTRIBUTION")

    df = random_log.copy()

    df["datetime"] = pd.to_datetime(df[TIME_COL], unit="ms")
    df["date_only"] = df["datetime"].dt.date

    print(df["date_only"].value_counts().sort_index().to_string())

    print("\nUnique dates:", df["date_only"].nunique())


# ============================================================
# 5. DUPLICATE EVALUATION EVENTS
# ============================================================

def audit_duplicate_evaluation_events(random_log):
    header("5. DUPLICATE EVALUATION EVENTS")

    keys = [USER_COL, VIDEO_COL, TIME_COL]

    duplicates = random_log.duplicated(subset=keys, keep=False)

    print(f"Duplicate rows: {duplicates.sum():,}")
    print(f"Duplicate groups: {random_log.loc[duplicates, keys].drop_duplicates().shape[0]:,}")

    if duplicates.any():
        print("\nExample duplicates:")
        print(
            random_log.loc[duplicates, keys]
            .sort_values(keys)
            .head(20)
            .to_string(index=False)
        )


# ============================================================
# 6. SAME USER / SAME TIMESTAMP
# ============================================================

def audit_same_user_same_timestamp(random_log):
    header("6. SAME-USER / SAME-TIMESTAMP BEHAVIOR")

    grouped = (
        random_log
        .groupby([USER_COL, TIME_COL])
        .size()
    )

    multi = grouped[grouped > 1]

    print(f"User-timestamp groups: {len(grouped):,}")
    print(f"Groups with >1 video: {len(multi):,}")
    print(f"Maximum videos at same user/timestamp: {grouped.max()}")

    if len(multi):
        print("\nDistribution:")
        print(multi.value_counts().sort_index().to_string())

        print("\nExamples:")
        print(
            random_log[
                random_log.set_index([USER_COL, TIME_COL]).index.isin(multi.index)
            ][[USER_COL, VIDEO_COL, TIME_COL]]
            .sort_values([USER_COL, TIME_COL])
            .head(30)
            .to_string(index=False)
        )


# ============================================================
# 7. WHETHER RANDOM VIDEOS APPEARED PREVIOUSLY
# ============================================================

def audit_previous_video_exposure(random_log, standard):
    print("\n" + "=" * 80)
    print("7. WHETHER RANDOM VIDEOS APPEARED PREVIOUSLY")
    print("=" * 80)

    # Only keep columns needed for temporal lookup
    history = standard[["user_id", "video_id", "time_ms"]].copy()
    history = history.rename(columns={"time_ms": "history_time_ms"})

    random_events = random_log[["user_id", "video_id", "time_ms"]].copy()
    random_events = random_events.rename(columns={"time_ms": "random_time_ms"})

    # Match random exposure with same-user + same-video history
    merged = random_events.merge(
        history,
        on=["user_id", "video_id"],
        how="left"
    )

    # Only history strictly before the random exposure
    merged = merged[
        merged["history_time_ms"] < merged["random_time_ms"]
    ]

    # For each random exposure, find the latest prior interaction
    latest_prior = (
        merged
        .groupby(
            ["user_id", "video_id", "random_time_ms"],
            as_index=False
        )["history_time_ms"]
        .max()
    )

    random_events = random_events.merge(
        latest_prior,
        on=["user_id", "video_id", "random_time_ms"],
        how="left"
    )

    appeared_before = random_events["history_time_ms"].notna()

    print(f"Random exposures: {len(random_events):,}")
    print(
        f"Previously appeared: "
        f"{appeared_before.sum():,} "
        f"({appeared_before.mean() * 100:.2f}%)"
    )
    print(
        f"Never appeared before: "
        f"{(~appeared_before).sum():,} "
        f"({(~appeared_before).mean() * 100:.2f}%)"
    )

    if appeared_before.any():
        gap = (
            random_events.loc[appeared_before, "random_time_ms"]
            - random_events.loc[appeared_before, "history_time_ms"]
        )

        print("\nTime since previous appearance:")
        print(f"  Median: {gap.median() / 3600000:.2f} hours")
        print(f"  Mean:   {gap.mean() / 3600000:.2f} hours")
        print(f"  Min:    {gap.min() / 3600000:.2f} hours")
        print(f"  Max:    {gap.max() / 3600000:.2f} hours")


# ============================================================
# 8. PRIOR INTERACTION COUNT
# ============================================================

def audit_prior_interaction_count(random_log, standard):
    header("8. PRIOR INTERACTION COUNT BEFORE EACH RANDOM EXPOSURE")

    standard_sorted = standard.sort_values(TIME_COL)
    random_sorted = random_log.sort_values(TIME_COL)

    # Number of standard interactions for the same user
    # strictly before each random exposure.
    history = standard_sorted[[USER_COL, TIME_COL]]

    random_with_count = pd.merge_asof(
        random_sorted,
        history,
        on=TIME_COL,
        by=USER_COL,
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_history"),
    )

    # Count interactions using cumulative user index.
    standard_sorted = standard_sorted.copy()
    standard_sorted["_user_order"] = (
        standard_sorted.groupby(USER_COL).cumcount() + 1
    )

    history_counts = standard_sorted[[USER_COL, TIME_COL, "_user_order"]]

    random_with_count = pd.merge_asof(
        random_sorted,
        history_counts,
        on=TIME_COL,
        by=USER_COL,
        direction="backward",
        allow_exact_matches=False,
    )

    counts = random_with_count["_user_order"].fillna(0)

    print(f"Random exposures: {len(counts):,}")
    print(f"Min prior interactions:    {counts.min():,.0f}")
    print(f"Mean prior interactions:   {counts.mean():,.2f}")
    print(f"Median prior interactions: {counts.median():,.0f}")
    print(f"P25:                       {counts.quantile(.25):,.0f}")
    print(f"P75:                       {counts.quantile(.75):,.0f}")
    print(f"P90:                       {counts.quantile(.90):,.0f}")
    print(f"P95:                       {counts.quantile(.95):,.0f}")
    print(f"P99:                       {counts.quantile(.99):,.0f}")
    print(f"Max prior interactions:    {counts.max():,.0f}")


# ============================================================
# 9. TEMPORAL ORDERING / LEAKAGE SANITY
# ============================================================

def audit_temporal_leakage(random_log, standard):
    header("9. TEMPORAL ORDERING / LEAKAGE SANITY CHECKS")

    random_min = random_log[TIME_COL].min()
    random_max = random_log[TIME_COL].max()

    before = (standard[TIME_COL] < random_min).sum()
    inside = (
        (standard[TIME_COL] >= random_min)
        & (standard[TIME_COL] <= random_max)
    ).sum()
    after = (standard[TIME_COL] > random_max).sum()

    print("Standard interactions relative to RANDOM period:")
    print(f"  Before random period: {before:,}")
    print(f"  Inside random period: {inside:,}")
    print(f"  After random period:  {after:,}")

    print("\nRandom log ordering:")

    random_sorted = random_log.sort_values([USER_COL, TIME_COL])

    time_diff = (
        random_sorted
        .groupby(USER_COL)[TIME_COL]
        .diff()
        .dropna()
    )

    print(f"Negative time differences: {(time_diff < 0).sum():,}")
    print(f"Zero time differences:     {(time_diff == 0).sum():,}")
    print(f"Positive time differences: {(time_diff > 0).sum():,}")

    print("\nGlobal random timestamp monotonicity:")
    print(
        "  Sorted globally:",
        random_log[TIME_COL].is_monotonic_increasing
    )


# ============================================================
# 10. COVERAGE BETWEEN STANDARD AND RANDOM LOGS
# ============================================================

def audit_standard_random_coverage(random_log, standard):
    header("10. COVERAGE BETWEEN STANDARD AND RANDOM LOGS")

    standard_users = set(standard[USER_COL].unique())
    random_users = set(random_log[USER_COL].unique())

    standard_videos = set(standard[VIDEO_COL].unique())
    random_videos = set(random_log[VIDEO_COL].unique())

    user_overlap = standard_users & random_users
    video_overlap = standard_videos & random_videos

    print("USERS")
    print(f"  Standard users:       {len(standard_users):,}")
    print(f"  Random users:         {len(random_users):,}")
    print(f"  Overlap:              {len(user_overlap):,}")
    print(f"  Random users covered: {len(user_overlap) / len(random_users):.4%}")

    print("\nVIDEOS")
    print(f"  Standard videos:       {len(standard_videos):,}")
    print(f"  Random videos:         {len(random_videos):,}")
    print(f"  Overlap:               {len(video_overlap):,}")
    print(f"  Random videos covered: {len(video_overlap) / len(random_videos):.4%}")

    # Pair coverage: (user, video)
    standard_pairs = set(
        zip(
            standard[USER_COL],
            standard[VIDEO_COL],
        )
    )

    random_pairs = list(
        zip(
            random_log[USER_COL],
            random_log[VIDEO_COL],
        )
    )

    pair_overlap = sum(pair in standard_pairs for pair in random_pairs)

    print("\nUSER-VIDEO PAIRS")
    print(f"  Random exposures:             {len(random_pairs):,}")
    print(f"  Pairs seen in standard:      {pair_overlap:,}")
    print(
        f"  Pair overlap rate:           "
        f"{pair_overlap / len(random_pairs):.4%}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("KUAIRAND-1K DATA AUDIT")
    print("=" * 80)
    print(f"Data directory: {DATA_DIR}")

    random_log, standard = load_logs()

    audit_temporal_coverage(random_log, standard)
    audit_interactions_per_user(standard)
    audit_random_exposures_per_user(random_log)
    audit_random_date_distribution(random_log)
    audit_duplicate_evaluation_events(random_log)
    audit_same_user_same_timestamp(random_log)
    audit_previous_video_exposure(random_log, standard)
    audit_prior_interaction_count(random_log, standard)
    audit_temporal_leakage(random_log, standard)
    audit_standard_random_coverage(random_log, standard)

    header("AUDIT COMPLETE")


if __name__ == "__main__":
    main()