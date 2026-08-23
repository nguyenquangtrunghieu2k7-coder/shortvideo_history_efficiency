from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"

EVAL_FILE = RAW_DIR / "log_random_4_22_to_5_08_1k.csv"
VIDEO_FILE = RAW_DIR / "video_features_basic_1k.csv"


STATIC_FEATURES = [
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
    print("AUDIT VIDEO METADATA TEMPORAL AVAILABILITY")
    print("=" * 80)

    print("\nLoading evaluation:")
    print(f"  {EVAL_FILE}")

    print("\nLoading video metadata:")
    print(f"  {VIDEO_FILE}")

    eval_cols = [
        "video_id",
        "time_ms",
    ]

    eval_df = pd.read_csv(
        EVAL_FILE,
        usecols=eval_cols,
    )

    video_df = pd.read_csv(VIDEO_FILE)

    print("\n" + "=" * 80)
    print("BASIC COVERAGE")
    print("=" * 80)

    print(
        f"Evaluation events:      {len(eval_df):,}"
    )

    print(
        f"Unique evaluation videos: "
        f"{eval_df['video_id'].nunique():,}"
    )

    print(
        f"Video metadata rows:     {len(video_df):,}"
    )

    print(
        f"Unique metadata videos:  "
        f"{video_df['video_id'].nunique():,}"
    )

    merged = eval_df.merge(
        video_df,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    missing = merged["author_id"].isna()

    print(
        f"Missing metadata events: "
        f"{missing.sum():,} "
        f"({missing.mean():.2%})"
    )

    if missing.sum() == 0:
        print("Metadata event coverage: PASS")
    else:
        print("Metadata event coverage: FAIL")

    print("\n" + "=" * 80)
    print("METADATA DUPLICATE AUDIT")
    print("=" * 80)

    duplicate_ids = (
        video_df["video_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate video_id rows: "
        f"{duplicate_ids:,}"
    )

    if duplicate_ids == 0:
        print("Video metadata uniqueness: PASS")
    else:
        print("Video metadata uniqueness: FAIL")

    print("\n" + "=" * 80)
    print("UPLOAD TIME AUDIT")
    print("=" * 80)

    print(
        "Checking whether metadata indicates that a video "
        "was uploaded after its evaluation exposure."
    )

    upload = pd.to_datetime(
        merged["upload_dt"],
        errors="coerce",
    )

    eval_time = pd.to_datetime(
        merged["time_ms"],
        unit="ms",
        errors="coerce",
    )

    invalid_upload = upload.isna()

    print(
        f"Unparseable upload_dt: "
        f"{invalid_upload.sum():,}"
    )

    comparable = (
        upload.notna()
        & eval_time.notna()
    )

    after_exposure = (
        comparable
        & (upload > eval_time)
    )

    print(
        f"Comparable events: "
        f"{comparable.sum():,}"
    )

    print(
        f"Upload after evaluation: "
        f"{after_exposure.sum():,} "
        f"({after_exposure.mean():.2%})"
    )

    if after_exposure.sum() == 0:
        print("Upload temporal safety: PASS")
    else:
        print(
            "Upload temporal safety: "
            "REVIEW REQUIRED"
        )

    print("\n" + "=" * 80)
    print("STATIC FEATURE COMPLETENESS")
    print("=" * 80)

    for col in STATIC_FEATURES:
        if col not in video_df.columns:
            print(
                f"{col:<20} NOT FOUND"
            )
            continue

        missing = video_df[col].isna().sum()
        unique = video_df[col].nunique(
            dropna=True
        )

        print(
            f"{col:<20} "
            f"missing={missing:>9,} "
            f"unique={unique:>10,}"
        )

    print("\n" + "=" * 80)
    print("FEATURE TEMPORAL CLASSIFICATION")
    print("=" * 80)

    print("\nSAFE CANDIDATE FEATURES:")
    for col in STATIC_FEATURES:
        if col == "upload_dt":
            print(
                f"  - {col:<18} "
                "pre-exposure metadata"
            )
        else:
            print(
                f"  - {col:<18} "
                "static item metadata"
            )

    print("\nEXCLUDED:")
    print(
        "  - video_features_statistic_1k.csv"
    )
    print(
        "    Reason: aggregate behavioral statistics "
        "may contain post-exposure information."
    )

    print("\n" + "=" * 80)
    print("FINAL TEMPORAL DECISION")
    print("=" * 80)

    if (
        missing.sum() == 0
        and duplicate_ids == 0
        and after_exposure.sum() == 0
    ):
        print(
            "PASS - video_features_basic_1k.csv "
            "is eligible as candidate metadata."
        )
    else:
        print(
            "REVIEW REQUIRED - "
            "do not finalize metadata features yet."
        )

    print("\n" + "=" * 80)
    print("ITEM TEMPORAL AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()