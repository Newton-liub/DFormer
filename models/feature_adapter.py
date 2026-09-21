"""Low-capacity feature adapters used by MMFR E1 screening.

The E1 F-lite candidate applies independent residual bottlenecks to DFormerv2
stages 1, 2 and 3 after the backbone has produced its feature tuple and before
the decoder consumes it.  Zero-initialized up projections make construction a
strict no-op while preserving a two-step gradient startup path.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class FeatureLiteResidualAdapter(nn.Module):
    """One frozen-contract ``1x1 down -> GELU -> zero-init 1x1 up`` adapter."""

    def __init__(self, channels: int, bottleneck_ratio: int = 4) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if bottleneck_ratio <= 0 or channels % bottleneck_ratio != 0:
            raise ValueError(
                f"channels={channels} must be divisible by bottleneck_ratio={bottleneck_ratio}"
            )
        bottleneck_channels = channels // bottleneck_ratio
        self.down = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=True)
        self.activation = nn.GELU()
        self.up = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=True)
        trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(features)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.residual(features)


class FeatureLiteAdapter(nn.Module):
    """Apply independent F-lite adapters to stages 1, 2 and 3 only."""

    STAGE_INDICES = (1, 2, 3)

    def __init__(self, channels: Sequence[int], bottleneck_ratio: int = 4) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"F-lite expects four backbone stages, got {len(channels)}")
        self.adapters = nn.ModuleDict(
            {
                str(index): FeatureLiteResidualAdapter(
                    int(channels[index]), bottleneck_ratio=bottleneck_ratio
                )
                for index in self.STAGE_INDICES
            }
        )

    def forward(self, features: Sequence[torch.Tensor]):
        if len(features) != 4:
            raise ValueError(f"F-lite expects four feature tensors, got {len(features)}")
        adapted = list(features)
        for index in self.STAGE_INDICES:
            adapted[index] = self.adapters[str(index)](adapted[index])
        if isinstance(features, tuple):
            return tuple(adapted)
        if isinstance(features, list):
            return adapted
        return type(features)(adapted)
