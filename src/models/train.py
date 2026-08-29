from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .recommender import HistoryVideoRecommender


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/phase8"

K_VALUES = [0, 1, 5, 10, 20, 50, 100, "Full"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrainDataset(Dataset):

    def __init__(self, frames):
        self.rows = []

        for k, df in frames.items():
            for row in df.itertuples():
                self.rows.append(
                    (
                        row.video_id,
                        list(row.history),
                        row.long_view,
                    )
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        video_id, history, label = self.rows[idx]

        return {
            "video_id": torch.tensor(
                video_id,
                dtype=torch.long,
            ),
            "history": history,
            "label": torch.tensor(
                label,
                dtype=torch.float32,
            ),
        }


def collate_fn(batch):

    max_len = max(
        len(x["history"])
        for x in batch
    )

    histories = []
    masks = []

    for x in batch:
        h = x["history"]

        pad = max_len - len(h)

        histories.append(
            h + [0] * pad
        )

        masks.append(
            [1] * len(h) + [0] * pad
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


def train_model(
    train_frames,
    num_videos,
    epochs=5,
    batch_size=512,
    lr=1e-3,
):

    seed_everything()

    device = get_device()

    print(f"Training device: {device}")

    dataset = TrainDataset(train_frames)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = HistoryVideoRecommender(
        num_videos=num_videos,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch in loader:

            video_id = batch["video_id"].to(device)
            history = batch["history"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(
                video_id,
                history,
                mask,
            )

            loss = loss_fn(
                logits,
                label,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss={mean_loss:.5f}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = OUT_DIR / "history_video_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_videos": num_videos,
            "embedding_dim": 64,
            "hidden_dim": 128,
        },
        checkpoint,
    )

    print(f"Saved checkpoint:\n  {checkpoint}")

    return model