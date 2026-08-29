from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


K_TO_COL = {
    0: None,
    1: "history_1",
    5: "history_5",
    10: "history_10",
    20: "history_20",
    50: "history_50",
    100: "history_100",
    "Full": "history_full",
}


@dataclass
class Phase8Paths:
    root: Path

    @property
    def processed(self):
        return self.root / "data" / "processed"

    @property
    def evaluation_events(self):
        return self.processed / "evaluation_events.parquet"

    @property
    def train(self):
        return self.processed / "train.parquet"

    @property
    def validation(self):
        return self.processed / "validation.parquet"

    @property
    def test(self):
        return self.processed / "test.parquet"


def load_split_with_labels(split_path: Path, events_path: Path) -> pd.DataFrame:
    split = pd.read_parquet(split_path)
    events = pd.read_parquet(events_path, columns=["evaluation_id", "long_view"])
    events = events.reset_index(drop=True)

    if "eval_idx" not in split.columns:
        raise ValueError(f"{split_path} is missing eval_idx.")

    if (
        events["evaluation_id"].to_numpy() == np.arange(len(events))
    ).all():
        idx = split["eval_idx"].to_numpy()
        if idx.min(initial=0) < 0 or idx.max(initial=-1) >= len(events):
            raise ValueError("eval_idx is outside evaluation_events.")
        out = split.copy()
        out["long_view"] = events.iloc[idx]["long_view"].to_numpy()
    else:
        label_map = events.set_index("evaluation_id")["long_view"]
        out = split.copy()
        out["long_view"] = out["eval_idx"].map(label_map)

    if out["long_view"].isna().any():
        raise ValueError("Some split rows could not be mapped to long_view.")

    out["long_view"] = out["long_view"].astype(np.float32)
    return out


def build_vocab(train_df: pd.DataFrame) -> dict[int, int]:
    ids: set[int] = set(train_df["video_id"].astype(np.int64).tolist())

    for col in K_TO_COL.values():
        if col is None or col not in train_df.columns:
            continue
        for arr in train_df[col]:
            if arr is not None:
                ids.update(np.asarray(arr, dtype=np.int64).tolist())

    return {vid: i + 1 for i, vid in enumerate(sorted(ids))}


def _encode_video(video_id: int, vocab: dict[int, int]) -> int:
    return vocab.get(int(video_id), 0)


class Phase8Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: dict[int, int],
        condition: str,
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.condition = condition

        if condition not in {"video_only", "history_only", "history_video"}:
            raise ValueError(condition)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video = _encode_video(row["video_id"], self.vocab)
        label = float(row["long_view"])

        if self.condition == "video_only":
            history = torch.empty(0, dtype=torch.long)
        else:
            arr = row["history"]
            if arr is None:
                history = torch.empty(0, dtype=torch.long)
            else:
                x = np.asarray(arr, dtype=np.int64)
                history = torch.tensor(
                    [self.vocab.get(int(v), 0) for v in x],
                    dtype=torch.long,
                )

        return {
            "video_id": torch.tensor(video, dtype=torch.long),
            "history": history,
            "history_len": torch.tensor(len(history), dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def phase8_collate(batch):
    videos = torch.stack([x["video_id"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    lengths = torch.stack([x["history_len"] for x in batch])

    histories = [x["history"] for x in batch]
    history = pad_sequence(histories, batch_first=True, padding_value=0)

    return {
        "video_id": videos,
        "history": history,
        "history_len": lengths,
        "label": labels,
    }


def prepare_split(split_path, events_path, history_k):
    df = load_split_with_labels(split_path, events_path)

    if history_k == 0:
        df = df.copy()
        df["history"] = [None] * len(df)
        return df

    col = K_TO_COL[history_k]
    if col not in df.columns:
        raise ValueError(f"{split_path} is missing {col}.")

    df = df.copy()
    df["history"] = df[col]
    return df
