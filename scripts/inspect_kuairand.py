from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/KuaiRand-1K/data")

FILES = [
    "log_random_4_22_to_5_08_1k.csv",
    "log_standard_4_08_to_4_21_1k.csv",
    "log_standard_4_22_to_5_08_1k.csv",
    "user_features_1k.csv",
    "video_features_basic_1k.csv",
    "video_features_statistic_1k.csv",
]


def inspect_file(path: Path, nrows: int = 5):
    print("\n" + "=" * 100)
    print(path.name)
    print("=" * 100)

    # Only read a small sample.
    df = pd.read_csv(path, nrows=nrows)

    print(f"Sample rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nSchema:")
    for col in df.columns:
        print(f"  {col:<35} {str(df[col].dtype):<12} null={df[col].isna().sum()}")

    print("\nSample:")
    print(df.to_string(index=False))

def inspect_log_stats(path: Path):
    print("\n" + "-" * 100)
    print(f"LOG STATS: {path.name}")
    print("-" * 100)

    users = set()
    videos = set()

    min_time = None
    max_time = None

    rand_counts = {}
    tab_counts = {}

    total_rows = 0

    feedback_cols = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
    ]

    feedback_counts = {col: 0 for col in feedback_cols}

    for chunk in pd.read_csv(path, chunksize=100_000):
        total_rows += len(chunk)

        users.update(chunk["user_id"].unique())
        videos.update(chunk["video_id"].unique())

        chunk_min = chunk["time_ms"].min()
        chunk_max = chunk["time_ms"].max()

        min_time = chunk_min if min_time is None else min(min_time, chunk_min)
        max_time = chunk_max if max_time is None else max(max_time, chunk_max)

        for value, count in chunk["is_rand"].value_counts().items():
            rand_counts[value] = rand_counts.get(value, 0) + count

        for value, count in chunk["tab"].value_counts().items():
            tab_counts[value] = tab_counts.get(value, 0) + count

        for col in feedback_cols:
            feedback_counts[col] += chunk[col].sum()

    print(f"Rows: {total_rows:,}")
    print(f"Unique users: {len(users):,}")
    print(f"Unique videos: {len(videos):,}")

    print(f"\nTime range:")
    print(f"  min: {min_time}")
    print(f"  max: {max_time}")

    print("\nis_rand:")
    print(rand_counts)

    print("\ntab:")
    print(tab_counts)

    print("\nPositive feedback counts:")
    for col, count in feedback_counts.items():
        print(f"  {col:<20} {int(count):,}")


def main():
    print(f"Dataset directory: {DATA_DIR}")

    for filename in FILES:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"\n[MISSING] {path}")
            continue

        inspect_file(path)

        if filename.startswith("log_"):
            inspect_log_stats(path)



if __name__ == "__main__":
    main()