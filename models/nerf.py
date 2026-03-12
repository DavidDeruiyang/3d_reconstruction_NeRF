from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    """
    Baseline NeRF MLP.

    Inputs:
        x: encoded 3D points, shape (..., input_ch)
        d: encoded view directions, shape (..., input_ch_dir)

    Outputs:
        rgb: (..., 3) in [0, 1]
        sigma: (..., 1), non-negative density
    """

    def __init__(
        self,
        input_ch: int = 63,
        input_ch_dir: int = 27,
        depth: int = 8,
        width: int = 256,
        skips: list[int] | None = None,
    ):
        super().__init__()

        if skips is None:
            skips = [4]

        self.input_ch = input_ch
        self.input_ch_dir = input_ch_dir
        self.depth = depth
        self.width = width
        self.skips = skips

        # Point-processing trunk
        self.pts_linears = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                in_ch = input_ch
            elif i in skips:
                in_ch = width + input_ch
            else:
                in_ch = width
            self.pts_linears.append(nn.Linear(in_ch, width))

        # Density head
        self.sigma_linear = nn.Linear(width, 1)

        # Feature branch for color
        self.feature_linear = nn.Linear(width, width)

        # View-direction branch for color
        self.view_linear = nn.Linear(width + input_ch_dir, width // 2)
        self.rgb_linear = nn.Linear(width // 2, 3)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: encoded 3D points, shape (..., input_ch)
            d: encoded directions, shape (..., input_ch_dir)

        Returns:
            rgb: (..., 3)
            sigma: (..., 1)
        """
        input_x = x
        h = x

        for i, layer in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([h, input_x], dim=-1)
            h = F.relu(layer(h))

        sigma = F.relu(self.sigma_linear(h))  # non-negative density

        feature = self.feature_linear(h)
        h_dir = torch.cat([feature, d], dim=-1)
        h_dir = F.relu(self.view_linear(h_dir))

        rgb = torch.sigmoid(self.rgb_linear(h_dir))  # map to [0, 1]

        return rgb, sigma