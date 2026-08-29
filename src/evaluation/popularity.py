from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score


def compute_popularity_from_training_interactions(train_df):
    counts = train_df["video_id"].value_counts()
    return {int(k): int(v) for k, v in counts.items()}


def score_popularity(df, popularity):
    return np.asarray(
        [popularity.get(int(v), 0) for v in df["video_id"]],
        dtype=np.float64,
    )


def evaluate_popularity(df, popularity):
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


def save_popularity(popularity, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "video_id": list(popularity.keys()),
            "popularity_count": list(popularity.values()),
        }
    ).sort_values("video_id").to_parquet(path, index=False)


def load_popularity(path: Path):
    df = pd.read_parquet(path)

    return dict(
        zip(
            df["video_id"].astype(int),
            df["popularity_count"].astype(int),
        )
    )
