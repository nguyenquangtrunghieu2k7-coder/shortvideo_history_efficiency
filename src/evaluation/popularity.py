from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RAW_STANDARD_FILE = (
    Path("data/raw/KuaiRand-1K/KuaiRand-1K/data")
    / "log_standard_4_08_to_4_21_1k.csv"
)


def compute_popularity_from_training_interactions(
    train_df: pd.DataFrame,
    raw_path: Path = RAW_STANDARD_FILE,
) -> dict[int, float]:
    """
    Build a frozen popularity lookup table from raw standard interactions
    that fall inside the temporal range of the training evaluation split.

    Popularity score:
        long_view_rate = sum(long_view) / number_of_interactions

    The returned dictionary is static and must not be updated during
    validation/test evaluation.
    """

    train_start = int(train_df["time_ms"].min())
    train_end = int(train_df["time_ms"].max())

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw standard interaction log not found: {raw_path}"
        )

    raw = pd.read_csv(raw_path)

    required = {"video_id", "time_ms", "long_view"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Raw interaction log is missing columns: {sorted(missing)}"
        )

    raw["time_ms"] = pd.to_numeric(raw["time_ms"], errors="coerce")
    raw["video_id"] = pd.to_numeric(raw["video_id"], errors="coerce")
    raw["long_view"] = pd.to_numeric(raw["long_view"], errors="coerce")

    raw = raw.dropna(subset=["video_id", "time_ms", "long_view"])

    # Only information available during the training period.
    raw = raw[
        (raw["time_ms"] >= train_start)
        & (raw["time_ms"] <= train_end)
    ].copy()

    if raw.empty:
        raise ValueError(
            "No raw standard interactions fall inside the training "
            f"timestamp range [{train_start}, {train_end}]."
        )

    stats = (
        raw.groupby("video_id", sort=False)
        .agg(
            impression_count=("video_id", "size"),
            long_view_count=("long_view", "sum"),
        )
        .reset_index()
    )

    stats["long_view_rate"] = (
        stats["long_view_count"] / stats["impression_count"]
    )

    # Freeze as a static video_id -> score lookup.
    popularity = {
        int(row.video_id): float(row.long_view_rate)
        for row in stats.itertuples(index=False)
    }

    return popularity


def score_popularity(
    df: pd.DataFrame,
    popularity: dict[int, float],
) -> np.ndarray:
    """
    Lookup frozen popularity scores for evaluation candidates.

    No statistics are computed or updated here.
    """

    return np.asarray(
        [popularity.get(int(video_id), 0.0) for video_id in df["video_id"]],
        dtype=np.float64,
    )


def evaluate_popularity(
    df: pd.DataFrame,
    popularity: dict[int, float],
) -> dict:
    """
    Evaluate the frozen popularity baseline on an evaluation split.
    """

    y = df["long_view"].to_numpy()
    scores = score_popularity(df, popularity)

    auc = (
        float(roc_auc_score(y, scores))
        if np.unique(y).size > 1
        else float("nan")
    )

    return {
        "auc": auc,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "unique_training_videos": int(len(popularity)),
    }


def save_popularity(
    popularity: dict[int, float],
    path: Path,
) -> None:
    """
    Persist the frozen popularity lookup table.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "video_id": list(popularity.keys()),
            "popularity_score": list(popularity.values()),
        }
    ).sort_values("video_id").to_parquet(
        path,
        index=False,
    )


def load_popularity(
    path: Path,
) -> dict[int, float]:
    """
    Load the frozen popularity lookup table.
    """

    df = pd.read_parquet(path)

    required = {"video_id", "popularity_score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Popularity artifact is missing columns: {sorted(missing)}"
        )

    return dict(
        zip(
            df["video_id"].astype(int),
            df["popularity_score"].astype(float),
        )
    )