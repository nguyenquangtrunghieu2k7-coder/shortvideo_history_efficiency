from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
PHASE7_DIR = ROOT / "data/processed/phase7"


K_VALUES = [0, 1, 5, 10, 20, 50, 100, "Full"]


def load_history(k):
    name = f"history_K{k}.parquet"

    path = PHASE7_DIR / name

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_parquet(path)


class HistoryDataset(Dataset):

    def __init__(self, df):
        self.video_id = torch.tensor(
            df["video_id"].values,
            dtype=torch.long,
        )

        self.label = torch.tensor(
            df["long_view"].values,
            dtype=torch.float32,
        )

        self.history = df["history"].tolist()

    def __len__(self):
        return len(self.video_id)

    def __getitem__(self, idx):
        history = self.history[idx]

        return {
            "video_id": self.video_id[idx],
            "history": torch.tensor(
                history,
                dtype=torch.long,
            ),
            "label": self.label[idx],
        }


def load_split(path):
    columns = [
        "evaluation_id",
        "user_id",
        "video_id",
        "time_ms",
        "long_view",
    ]

    return pd.read_parquet(
        path,
        columns=columns,
    )


def load_phase7_dataset(k):
    df = load_history(k)

    # Phase 7 artifacts must contain these fields.
    required = [
        "evaluation_id",
        "video_id",
        "history",
        "history_len",
        "history_available",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in history_K{k}: {missing}"
        )

    return df