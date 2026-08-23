from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"
HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"


ID_COLUMNS = {
    "user_id",
    "video_id",
    "time_ms",
}

POST_EXPOSURE_COLUMNS = {
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
}

TEMPORAL_COLUMNS = {
    "date",
    "hourmin",
}

CONTROL_COLUMNS = {
    "is_rand",
    "tab",
}


def inspect_csv_files():
    print("\n" + "=" * 80)
    print("KUairAND FILE INVENTORY")
    print("=" * 80)

    files = sorted(RAW_DIR.glob("*.csv"))

    print(f"CSV files found: {len(files):,}")

    results = []

    for path in files:
        print(f"\n{'-' * 80}")
        print(f"FILE: {path.name}")
        print(f"{'-' * 80}")

        try:
            df = pd.read_csv(path, nrows=5)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        columns = list(df.columns)

        print(f"Columns: {len(columns):,}")

        for col in columns:
            print(f"  - {col}")

        has_video = "video_id" in columns

        results.append(
            {
                "file": path.name,
                "has_video_id": has_video,
                "columns": columns,
            }
        )

    return results


def audit_video_files(results):
    print("\n" + "=" * 80)
    print("VIDEO-ID FILE AUDIT")
    print("=" * 80)

    video_files = [
        r for r in results
        if r["has_video_id"]
    ]

    print(
        f"Files containing video_id: "
        f"{len(video_files):,}"
    )

    for result in video_files:
        print(f"  - {result['file']}")


def classify_metadata_columns(results):
    print("\n" + "=" * 80)
    print("POTENTIAL ITEM METADATA")
    print("=" * 80)

    found = False

    for result in results:
        columns = set(result["columns"])

        excluded = (
            ID_COLUMNS
            | POST_EXPOSURE_COLUMNS
            | TEMPORAL_COLUMNS
            | CONTROL_COLUMNS
        )

        candidates = columns - excluded

        if not candidates:
            continue

        found = True

        print(f"\nFILE: {result['file']}")

        for col in sorted(candidates):
            print(f"  - {col}")

    if not found:
        print(
            "No potential static item metadata "
            "found in scanned CSV schemas."
        )


def audit_evaluation_candidate_coverage(results):
    print("\n" + "=" * 80)
    print("EVALUATION CANDIDATE COVERAGE")
    print("=" * 80)

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    history = pd.read_parquet(HISTORY_PATH)

    candidate_videos = set(
        history["video_id"].unique()
    )

    print(
        f"Evaluation candidate videos: "
        f"{len(candidate_videos):,}"
    )

    for path in sorted(RAW_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(
                path,
                usecols=["video_id"],
            )
        except (ValueError, FileNotFoundError):
            continue

        videos = set(
            df["video_id"].dropna().unique()
        )

        overlap = candidate_videos & videos

        print(
            f"\n{path.name}"
        )
        print(
            f"  unique videos: {len(videos):,}"
        )
        print(
            f"  candidate coverage: "
            f"{len(overlap):,} / "
            f"{len(candidate_videos):,} "
            f"({len(overlap) / len(candidate_videos):.2%})"
        )


def audit_static_candidate_sources(results):
    print("\n" + "=" * 80)
    print("STATIC CANDIDATE SOURCE AUDIT")
    print("=" * 80)

    static_sources = []

    for result in results:
        columns = set(result["columns"])

        if "video_id" not in columns:
            continue

        excluded = (
            ID_COLUMNS
            | POST_EXPOSURE_COLUMNS
            | TEMPORAL_COLUMNS
            | CONTROL_COLUMNS
        )

        candidate_features = columns - excluded

        if candidate_features:
            static_sources.append(
                (
                    result["file"],
                    sorted(candidate_features),
                )
            )

    if not static_sources:
        print("Static candidate metadata: NOT FOUND")
        print(
            "\nCurrent representation must therefore rely "
            "on behavioral history / IDs."
        )
        return

    print("Potential static sources:")

    for file_name, features in static_sources:
        print(f"\nFILE: {file_name}")

        for feature in features:
            print(f"  - {feature}")


def main():
    print("=" * 80)
    print("AUDIT KUAIRAND ITEM METADATA")
    print("=" * 80)

    print(f"\nRaw directory:")
    print(f"  {RAW_DIR}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(RAW_DIR)

    results = inspect_csv_files()

    audit_video_files(results)
    classify_metadata_columns(results)
    audit_evaluation_candidate_coverage(results)
    audit_static_candidate_sources(results)

    print("\n" + "=" * 80)
    print("ITEM METADATA AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()