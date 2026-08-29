from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .metrics import classification_metrics


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    ys = []
    ps = []

    for batch in loader:
        video = batch["video_id"].to(device)
        history = batch["history"].to(device)
        history_len = batch["history_len"].to(device)

        logits = model(video, history, history_len)
        probs = torch.sigmoid(logits)

        ys.append(batch["label"].cpu().numpy())
        ps.append(probs.cpu().numpy())

    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def evaluate_model(model, loader, device):
    y, p = predict(model, loader, device)
    return classification_metrics(y, p)


def save_predictions(model, loader, device, df, path: Path):
    y, p = predict(model, loader, device)

    out = df[
        ["eval_idx", "user_id", "video_id", "time_ms"]
    ].copy()

    out["long_view"] = y.astype(np.int8)
    out["prediction"] = p.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
