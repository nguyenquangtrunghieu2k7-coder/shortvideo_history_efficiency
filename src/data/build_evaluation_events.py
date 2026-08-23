# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"
OUTPUT_DIR = ROOT / "data" / "processed"

RANDOM_FILE = DATA_DIR / "log_random_4_22_to_5_08_1k.csv"

STANDARD_FILES = [
    DATA_DIR / "log_standard_4_08_to_4_21_1k.csv",
    DATA_DIR / "log_standard_4_22_to_5_08_1k.csv",
]

OUTPUT_FILE = OUTPUT_DIR / "evaluation_events.parquet"

COLS = [
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

CHUNK_SIZE = 500_000


def load_random():
    print("Loading random log...")

    df = pd.read_csv(
        RANDOM_FILE,
        usecols=COLS,
    )

    df = df.drop_duplicates(
        subset=["user_id", "video_id", "time_ms"]
    )

    df = df.sort_values(
        ["time_ms", "user_id", "video_id"]
    ).reset_index(drop=True)

    print(f"  Random events: {len(df):,}")

    return df


def find_previous_same_video(random_events):
    """
    Find the latest standard interaction of the same
    user-video pair strictly before each random exposure.
    """

    print("\nFinding previous same-video interactions...")

    # Only columns needed for this operation.
    target = random_events[
        ["user_id", "video_id", "time_ms"]
    ].copy()

    target = target.rename(
        columns={"time_ms": "random_time_ms"}
    )

    # Keep latest previous occurrence for each evaluation event.
    best = {}

    total_rows = 0

    for path in STANDARD_FILES:

        print(f"\nProcessing: {path.name}")

        for chunk_no, chunk in enumerate(
            pd.read_csv(
                path,
                usecols=["user_id", "video_id", "time_ms"],
                chunksize=CHUNK_SIZE,
            ),
            start=1,
        ):
            total_rows += len(chunk)

            # Only standard events that could possibly match.
            merged = chunk.merge(
                target,
                on=["user_id", "video_id"],
                how="inner",
            )

            if len(merged) == 0:
                print(
                    f"  chunk {chunk_no}: "
                    f"{len(chunk):,} rows -> no matches"
                )
                continue

            merged = merged[
                merged["time_ms"] < merged["random_time_ms"]
            ]

            if len(merged) == 0:
                continue

            grouped = (
                merged
                .groupby(
                    [
                        "user_id",
                        "video_id",
                        "random_time_ms",
                    ],
                    as_index=False,
                )["time_ms"]
                .max()
            )

            for row in grouped.itertuples(index=False):
                key = (
                    row.user_id,
                    row.video_id,
                    row.random_time_ms,
                )

                previous = best.get(key)

                if previous is None or row.time_ms > previous:
                    best[key] = row.time_ms

            if chunk_no % 2 == 0:
                print(
                    f"  chunk {chunk_no}: "
                    f"{total_rows:,} standard rows processed"
                )

    print(
        f"\nTotal standard rows processed: "
        f"{total_rows:,}"
    )

    result = target.copy()

    result["previous_same_video_time_ms"] = [
        best.get(
            (
                row.user_id,
                row.video_id,
                row.random_time_ms,
            )
        )
        for row in result.itertuples(index=False)
    ]

    return result


def build_evaluation_events(random_events):
    print("\nBuilding evaluation table...")

    previous = find_previous_same_video(
        random_events
    )

    events = random_events.merge(
        previous,
        left_on=[
            "user_id",
            "video_id",
            "time_ms",
        ],
        right_on=[
            "user_id",
            "video_id",
            "random_time_ms",
        ],
        how="left",
    )

    events = events.drop(
        columns=["random_time_ms"]
    )

    events["has_previous_same_video"] = (
        events["previous_same_video_time_ms"]
        .notna()
    )

    events["time_since_previous_same_video_ms"] = (
        events["time_ms"]
        - events["previous_same_video_time_ms"]
    )

    events = events.sort_values(
        ["time_ms", "user_id", "video_id"]
    ).reset_index(drop=True)

    events.insert(
        0,
        "evaluation_id",
        range(len(events)),
    )

    return events


def validate(events):
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(f"Evaluation events: {len(events):,}")
    print(f"Users:              {events.user_id.nunique():,}")
    print(f"Videos:             {events.video_id.nunique():,}")

    dup = events[
        ["user_id", "video_id", "time_ms"]
    ].duplicated().sum()

    print(f"Duplicate events:   {dup:,}")

    previous = events["has_previous_same_video"].sum()

    print(
        f"Previously seen:    "
        f"{previous:,} "
        f"({previous / len(events) * 100:.4f}%)"
    )

    print(
        f"Never seen:         "
        f"{len(events) - previous:,} "
        f"({(len(events) - previous) / len(events) * 100:.4f}%)"
    )

    if not (events["is_rand"] == 1).all():
        raise ValueError(
            "Non-random event found."
        )

    if dup != 0:
        raise ValueError(
            "Duplicate evaluation events remain."
        )

    mask = events["previous_same_video_time_ms"].notna()

    if mask.any():
        if not (
            events.loc[
                mask,
                "previous_same_video_time_ms",
            ]
            <
            events.loc[
                mask,
                "time_ms",
            ]
        ).all():
            raise ValueError(
                "Temporal ordering violation."
            )

    print("\nValidation: PASS")


def main():

    print("=" * 80)
    print("BUILD KUAIRAND EVALUATION EVENTS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    random_events = load_random()

    events = build_evaluation_events(
        random_events
    )

    validate(events)

    events.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(f"  {OUTPUT_FILE}")

    print("\nDone.")


if __name__ == "__main__":
    main()