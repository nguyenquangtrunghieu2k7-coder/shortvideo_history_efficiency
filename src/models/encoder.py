import torch
import torch.nn as nn


class VideoEncoder(nn.Module):

    def __init__(
        self,
        num_videos,
        embedding_dim=64,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_videos + 1,
            embedding_dim,
            padding_idx=0,
        )

    def forward(self, video_ids):
        return self.embedding(video_ids)


class HistoryEncoder(nn.Module):

    def __init__(
        self,
        num_videos,
        embedding_dim=64,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_videos + 1,
            embedding_dim,
            padding_idx=0,
        )

    def forward(self, history_ids, mask):
        emb = self.embedding(history_ids)

        mask = mask.unsqueeze(-1).float()

        emb = emb * mask

        length = mask.sum(dim=1).clamp(min=1.0)

        return emb.sum(dim=1) / length