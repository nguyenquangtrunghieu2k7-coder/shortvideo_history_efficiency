from __future__ import annotations

from torch import nn

from .encoder import HistoryEncoder


class HistoryOnlyModel(nn.Module):
    def __init__(self, n_videos, dim=64, hidden=128):
        super().__init__()

        self.history = HistoryEncoder(n_videos, dim)

        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, video_id, history, history_len):
        history_x = self.history(history, history_len)
        return self.head(history_x).squeeze(-1)
