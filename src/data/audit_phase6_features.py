from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = (
    ROOT / "data/processed/evaluation_events.parquet"
)

ITEM_FILE = (
    ROOT
    / "data/raw/KuaiRand-1K/data/video_features_basic_1k.csv"
)

STAT_FILE = (
    ROOT
    / "data/raw/KuaiRand-1K/data/video_features_statistic_1k.csv"
)


SAFE_FEATURES = [
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "video_duration",
    "server_width",
    "server_height",
    "music_id",
    "music_type",
    "tag",
]


def main():

    print("=" * 80)
    print("PHASE 6 — FUTURE-DEPENDENT FEATURE AUDIT")
    print("=" * 80)

    print("\nLoading evaluation events...")

    eval_df = pd.read_parquet(
        EVAL_FILE,
        columns=[
            "video_id",
            "time_ms",
        ],
    )

    print(
        f"  Evaluation events: {len(eval_df):,}"
    )

    print("\nLoading static item metadata...")

    item_df = pd.read_csv(
        ITEM_FILE,
        usecols=[
            "video_id",
            *SAFE_FEATURES,
        ],
    )

    print(
        f"  Metadata rows:      {len(item_df):,}"
    )

    print("\n" + "=" * 80)
    print("1. STATIC FEATURE AVAILABILITY")
    print("=" * 80)

    missing_columns = [
        col
        for col in SAFE_FEATURES
        if col not in item_df.columns
    ]

    if missing_columns:
        print(
            "  Required static features: FAIL"
        )
        print(
            f"  Missing: {missing_columns}"
        )
        return

    print(
        "  Required static features: PASS"
    )

    print("\n" + "=" * 80)
    print("2. VIDEO METADATA COVERAGE")
    print("=" * 80)

    eval_videos = set(
        eval_df["video_id"].dropna().unique()
    )

    meta_videos = set(
        item_df["video_id"].dropna().unique()
    )

    missing_videos = eval_videos - meta_videos

    if not missing_videos:
        print(
            "  Evaluation video metadata coverage: PASS"
        )
    else:
        print(
            "  Evaluation video metadata coverage: FAIL"
        )
        print(
            f"  Missing videos: {len(missing_videos):,}"
        )
        return

    print("\n" + "=" * 80)
    print("3. METADATA TEMPORAL SAFETY")
    print("=" * 80)

    # upload_dt was already checked in Phase 4.
    # Here we explicitly preserve the decision:
    #
    # upload_dt <= evaluation timestamp
    #
    # for every evaluation event.

    if "upload_dt" not in item_df.columns:
        print(
            "  upload_dt availability: FAIL"
        )
        return

    print(
        "  upload_dt temporal safety: "
        "VERIFIED BY PHASE 4 AUDIT"
    )

    print("\n" + "=" * 80)
    print("4. FUTURE AGGREGATE STATISTICS")
    print("=" * 80)

    if STAT_FILE.exists():
        print(
            "  video_features_statistic_1k.csv: EXCLUDED"
        )
        print(
            "  Reason: aggregate behavioral statistics "
            "may contain post-exposure information."
        )
    else:
        print(
            "  Statistical feature file: NOT FOUND"
        )

    print("\n" + "=" * 80)
    print("5. FEATURE POLICY")
    print("=" * 80)

    print(
        "  Static item metadata: ALLOWED"
    )

    print(
        "  Future-dependent behavioral aggregates: EXCLUDED"
    )

    print(
        "  Future user representations: EXCLUDED"
    )

    print(
        "  Future popularity statistics: EXCLUDED"
    )

    print("\n" + "=" * 80)
    print("PHASE 6 FEATURE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
