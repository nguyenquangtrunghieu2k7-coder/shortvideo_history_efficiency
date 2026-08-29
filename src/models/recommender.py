from __future__ import annotations

import torch
from torch import nn

from .encoder import HistoryEncoder, VideoEncoder


class VideoOnlyModel(nn.Module):
    def __init__(self, n_videos, dim=64, hidden=128):
        super().__init__()
        self.video = VideoEncoder(n_videos, dim)

        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, video_id, history=None, history_len=None):
        return self.head(self.video(video_id)).squeeze(-1)


class HistoryVideoModel(nn.Module):
    def __init__(self, n_videos, dim=64, hidden=128):
        super().__init__()

        self.video = VideoEncoder(n_videos, dim)
        self.history = HistoryEncoder(n_videos, dim)

        self.head = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, video_id, history, history_len):
        video_x = self.video(video_id)
        history_x = self.history(history, history_len)

        x = torch.cat([video_x, history_x], dim=-1)

        return self.head(x).squeeze(-1)
