import torch
import torch.nn as nn


def _choose_group_count(num_channels: int, max_groups: int = 32) -> int:
    for group_count in range(min(max_groups, num_channels), 0, -1):
        if num_channels % group_count == 0:
            return group_count
    return 1


class TrackContextBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        group_count = _choose_group_count(dim)
        self.norm1 = nn.GroupNorm(group_count, dim)
        self.norm2 = nn.GroupNorm(group_count, dim)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + residual


class WanTrackResidualAdapter(nn.Module):
    def __init__(
        self,
        track_in_dim: int = 4,
        hidden_dim: int = 512,
        out_dim: int = 1536,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_layers: int = 15,
    ):
        super().__init__()
        self.track_in_dim = track_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.patch_size = tuple(int(value) for value in patch_size)
        self.num_layers = int(num_layers)

        stem_group_count = _choose_group_count(hidden_dim)
        self.stem = nn.Sequential(
            nn.Conv3d(track_in_dim, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GroupNorm(stem_group_count, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(stem_group_count, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([TrackContextBlock(hidden_dim) for _ in range(self.num_layers)])
        self.output_projections = nn.ModuleList([nn.Conv3d(hidden_dim, out_dim, kernel_size=1) for _ in range(self.num_layers)])
        self.gates = nn.Parameter(torch.ones(self.num_layers))

        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def forward(self, track_feature_map: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.stem(track_feature_map)
        residuals = []
        for gate, block, projection in zip(self.gates, self.blocks, self.output_projections):
            hidden = block(hidden)
            delta = projection(hidden) * gate.view(1, 1, 1, 1, 1)
            residuals.append(delta.flatten(2).transpose(1, 2).contiguous())
        return residuals
