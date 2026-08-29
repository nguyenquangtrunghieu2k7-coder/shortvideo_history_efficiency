from __future__ import annotations

import torch
from torch import nn


class VideoEncoder(nn.Module):
    def __init__(self, n_videos, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(
            n_videos + 1,
            dim,
            padding_idx=0,
        )

    def forward(self, video_id):
        return self.embedding(video_id)


class HistoryEncoder(nn.Module):
    def __init__(self, n_videos, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(
            n_videos + 1,
            dim,
            padding_idx=0,
        )

    def forward(self, history, history_len):
        x = self.embedding(history)

        mask = (history != 0).unsqueeze(-1)
        x = x * mask

        denom = history_len.clamp_min(1).to(x.dtype).unsqueeze(-1)

        return x.sum(dim=1) / denom
