import torch
import torch.nn as nn

from .encoder import HistoryEncoder


class HistoryOnlyRecommender(nn.Module):

    def __init__(
        self,
        num_videos,
        embedding_dim=64,
        hidden_dim=128,
    ):
        super().__init__()

        self.history_encoder = HistoryEncoder(
            num_videos,
            embedding_dim,
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history,
        history_mask,
    ):
        x = self.history_encoder(
            history,
            history_mask,
        )

        return self.mlp(x).squeeze(-1)