from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .dataset import HistoryDataset
from ..models.recommender import HistoryVideoRecommender


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/phase8"


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def make_loader(df, batch_size=512):

    dataset = HistoryDataset(df)

    def collate(batch):

        max_len = max(
            len(x["history"])
            for x in batch
        )

        histories = []
        masks = []

        for x in batch:

            h = x["history"].tolist()

            histories.append(
                h + [0] * (max_len - len(h))
            )

            masks.append(
                [1] * len(h)
                + [0] * (max_len - len(h))
            )

        return {
            "video_id": torch.stack(
                [x["video_id"] for x in batch]
            ),
            "history": torch.tensor(
                histories,
                dtype=torch.long,
            ),
            "mask": torch.tensor(
                masks,
                dtype=torch.bool,
            ),
            "label": torch.stack(
                [x["label"] for x in batch]
            ),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )


@torch.no_grad()
def evaluate_history_model(
    model,
    df,
    device,
):

    loader = make_loader(df)

    model.eval()

    ys = []
    preds = []

    for batch in loader:

        video_id = batch["video_id"].to(device)
        history = batch["history"].to(device)
        mask = batch["mask"].to(device)

        logits = model(
            video_id,
            history,
            mask,
        )

        preds.extend(
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

        ys.extend(
            batch["label"]
            .numpy()
        )

    auc = roc_auc_score(
        ys,
        preds,
    )

    return {
        "n": len(ys),
        "positive": int(np.sum(ys)),
        "negative": int(len(ys) - np.sum(ys)),
        "roc_auc": float(auc),
    }