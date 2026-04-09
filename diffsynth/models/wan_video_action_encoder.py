import torch
import torch.nn as nn
from typing import Optional


class WanVideoActionEncoder(torch.nn.Module):
    def __init__(
        self,
        action_dim: int = 14,
        dim: int = 1536,
        num_action_per_chunk: Optional[int] = None,
        in_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.dim = dim

        if in_features is None:
            in_features = action_dim if num_action_per_chunk is None else action_dim * num_action_per_chunk
        if hidden_features is None:
            hidden_features = dim * 4 if num_action_per_chunk is not None else dim
            
        self.action_embedding = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_features, dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_embedding(action)
