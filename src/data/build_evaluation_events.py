# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"
OUTPUT_DIR = ROOT / "data" / "processed"

RANDOM_FILE = DATA_DIR / "log_random_4_22_to_5_08_1k.csv"
STANDARD_FILES = [
    DATA_DIR / "log_standard_4_08_to_4_21_1k.csv",
    DATA_DIR / "log_standard_4_22_to_5_08_1k.csv",
]

OUTPUT_FILE = OUTPUT_DIR / "evaluation_events.parquet"


# ============================================================
# COLUMNS
# ============================================================

EVENT_COLS = [
    "user_id",
    "video_id",
    "time_ms",
    "date",
    "hourmin",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_random_log():
    print("Loading random log...")

    df = pd.read_csv(
        RANDOM_FILE,
        usecols=EVENT_COLS,
    )

    print(f"  Random events: {len(df):,}")

    return df


def load_standard_logs():
    print("Loading standard logs...")

    dfs = []

    for path in STANDARD_FILES:
        print(f"  Loading: {path.name}")

        df = pd.read_csv(
            path,
            usecols=EVENT_COLS,
        )

        print(f"    Rows: {len(df):,}")

        dfs.append(df)

    standard = pd.concat(
        dfs,
        ignore_index=True,
    )

    print(f"  Total standard events: {len(standard):,}")

    return standard


# ============================================================
# CLEAN / SORT
# ============================================================

def prepare_logs(random_log, standard):
    print("\nPreparing logs...")

    random_log = random_log.drop_duplicates(
        subset=["user_id", "video_id", "time_ms"]
    ).copy()

    standard = standard.drop_duplicates(
        subset=["user_id", "video_id", "time_ms"]
    ).copy()

    random_log = random_log.sort_values(
        ["user_id", "time_ms", "video_id"]
    ).reset_index(drop=True)

    standard = standard.sort_values(
        ["user_id", "time_ms", "video_id"]
    ).reset_index(drop=True)

    print(f"  Random events after dedup:   {len(random_log):,}")
    print(f"  Standard events after dedup: {len(standard):,}")

    return random_log, standard


# ============================================================
# BUILD EVALUATION EVENTS
# ============================================================

def build_evaluation_events(random_log, standard):
    print("\nBuilding evaluation events...")

    # Only information needed to identify prior user-video exposure.
    prior = standard[
        ["user_id", "video_id", "time_ms"]
    ].copy()

    prior = prior.rename(
        columns={
            "time_ms": "prior_time_ms"
        }
    )

    # Every random exposure is an evaluation event.
    eval_events = random_log.copy()

    # --------------------------------------------------------
    # Find the latest previous occurrence of the same
    # user-video pair in standard history.
    # --------------------------------------------------------

    merged = eval_events[
        ["user_id", "video_id", "time_ms"]
    ].merge(
        prior,
        on=["user_id", "video_id"],
        how="left",
    )

    merged = merged[
        merged["prior_time_ms"] < merged["time_ms"]
    ]

    latest_prior = (
        merged
        .groupby(
            ["user_id", "video_id", "time_ms"],
            as_index=False,
        )["prior_time_ms"]
        .max()
    )

    latest_prior = latest_prior.rename(
        columns={
            "prior_time_ms": "previous_same_video_time_ms"
        }
    )

    eval_events = eval_events.merge(
        latest_prior,
        on=["user_id", "video_id", "time_ms"],
        how="left",
    )

    # --------------------------------------------------------
    # Basic evaluation metadata
    # --------------------------------------------------------

    eval_events["evaluation_id"] = range(len(eval_events))

    eval_events["has_previous_same_video"] = (
        eval_events["previous_same_video_time_ms"].notna()
    )

    eval_events["time_since_previous_same_video_ms"] = (
        eval_events["time_ms"]
        - eval_events["previous_same_video_time_ms"]
    )

    # --------------------------------------------------------
    # Keep evaluation events in strict chronological order.
    # --------------------------------------------------------

    eval_events = eval_events.sort_values(
        ["time_ms", "user_id", "video_id"]
    ).reset_index(drop=True)

    # Reassign IDs after final ordering.
    eval_events["evaluation_id"] = range(len(eval_events))

    return eval_events


# ============================================================
# VALIDATION
# ============================================================

def validate_evaluation_events(eval_events):
    print("\n" + "=" * 80)
    print("EVALUATION EVENT VALIDATION")
    print("=" * 80)

    print(f"Evaluation events: {len(eval_events):,}")
    print(f"Users:             {eval_events['user_id'].nunique():,}")
    print(f"Videos:            {eval_events['video_id'].nunique():,}")

    print("\nDuplicate evaluation IDs:")
    print(
        eval_events["evaluation_id"]
        .duplicated()
        .sum()
    )

    print("\nDuplicate user-video-time events:")
    print(
        eval_events[
            ["user_id", "video_id", "time_ms"]
        ]
        .duplicated()
        .sum()
    )

    print("\nPreviously seen same video:")
    n_seen = eval_events["has_previous_same_video"].sum()

    print(f"  Yes: {n_seen:,}")
    print(
        f"  No:  {len(eval_events) - n_seen:,}"
    )

    print("\nEvaluation time range:")

    min_time = pd.to_datetime(
        eval_events["time_ms"],
        unit="ms",
    )

    print(f"  Min: {min_time.min()}")
    print(f"  Max: {min_time.max()}")

    # Ensure every evaluation event is actually random.
    if not (eval_events["is_rand"] == 1).all():
        raise ValueError(
            "Evaluation table contains non-random events."
        )

    # Ensure no duplicate evaluation events remain.
    if eval_events[
        ["user_id", "video_id", "time_ms"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate evaluation events remain."
        )

    # Previous same-video timestamps must be strictly earlier.
    valid_previous = eval_events[
        "previous_same_video_time_ms"
    ].notna()

    if valid_previous.any():
        if not (
            eval_events.loc[
                valid_previous,
                "previous_same_video_time_ms",
            ]
            <
            eval_events.loc[
                valid_previous,
                "time_ms",
            ]
        ).all():
            raise ValueError(
                "Temporal ordering violation detected."
            )

    print("\nValidation: PASS")


# ============================================================
# SAVE
# ============================================================

def save_evaluation_events(eval_events):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_events.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(f"  {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("BUILD KUAIRAND EVALUATION EVENTS")
    print("=" * 80)

    random_log = load_random_log()
    standard = load_standard_logs()

    random_log, standard = prepare_logs(
        random_log,
        standard,
    )

    eval_events = build_evaluation_events(
        random_log,
        standard,
    )

    validate_evaluation_events(
        eval_events,
    )

    save_evaluation_events(
        eval_events,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()