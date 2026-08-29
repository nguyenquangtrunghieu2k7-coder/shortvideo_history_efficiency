from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

TRAIN_LOG = (
    ROOT
    / "data/raw/KuaiRand-1K/KuaiRand-1K/data"
    / "log_standard_4_08_to_4_21_1k.csv"
)

OUT_DIR = ROOT / "data/processed/phase8"
OUT_FILE = OUT_DIR / "frozen_popularity.parquet"


def build_popularity():
    print("=" * 80)
    print("BUILD FROZEN POPULARITY BASELINE")
    print("=" * 80)

    print(f"Loading:\n  {TRAIN_LOG}")

    df = pd.read_csv(
        TRAIN_LOG,
        usecols=["video_id"],
    )

    counts = (
        df["video_id"]
        .value_counts()
        .rename("interaction_count")
        .reset_index()
        .rename(columns={"index": "video_id"})
    )

    counts["video_id"] = counts["video_id"].astype(np.int64)

    # Log transform reduces domination by extremely popular videos.
    counts["popularity_score"] = np.log1p(
        counts["interaction_count"]
    ).astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(OUT_FILE, index=False)

    print(f"Videos: {len(counts):,}")
    print(f"Saved: {OUT_FILE}")

    return counts


if __name__ == "__main__":
    build_popularity()