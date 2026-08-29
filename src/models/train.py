from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.evaluate import evaluate_model


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred="auto"):
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")

    if preferred == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps")

    if preferred == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    return torch.device("cpu")


def _epoch_path(run_dir, epoch):
    return run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"


def save_checkpoint(path, model, optimizer, epoch, best_auc, history):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_auc": best_auc,
            "history": history,
        },
        path,
    )


def find_latest_checkpoint(run_dir):
    paths = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    return paths[-1] if paths else None


def load_best_checkpoint(run_dir, model, device):
    path = run_dir / "checkpoints" / "best.pt"
    if not path.exists():
        return model, None

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    run_dir,
    epochs=10,
    lr=1e-3,
    resume=True,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 1
    best_auc = -float("inf")
    history = []

    latest = find_latest_checkpoint(run_dir) if resume else None

    if latest is not None:
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_auc = float(ckpt["best_auc"])
        history = ckpt.get("history", [])
        print(f"Resume: {latest} -> epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for batch in train_loader:
            video = batch["video_id"].to(device)
            history_x = batch["history"].to(device)
            history_len = batch["history_len"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(video, history_x, history_len)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            bs = y.numel()
            total_loss += float(loss.item()) * bs
            n += bs

        train_loss = total_loss / max(n, 1)
        val_metrics = evaluate_model(model, val_loader, device)
        val_auc = val_metrics["auc"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
        }
        history.append(record)

        # Save the epoch checkpoint immediately.
        epoch_path = _epoch_path(run_dir, epoch)
        save_checkpoint(
            epoch_path,
            model,
            optimizer,
            epoch,
            best_auc,
            history,
        )

        # Also maintain an explicit last.pt.
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            best_auc,
            history,
        )

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = float(val_auc)
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                best_auc,
                history,
            )

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "device": str(device),
                    "best_validation_auc": best_auc,
                    "history": history,
                },
                f,
                indent=2,
            )

        print(
            f"epoch {epoch:03d}/{epochs} | "
            f"loss={train_loss:.5f} | val_auc={val_auc:.6f}"
        )

    # Always return the best validation model, not merely the last epoch.
    model, best_ckpt = load_best_checkpoint(run_dir, model, device)
    return model, history, best_auc
