"""Frozen R-OE-lite observable-empty RGB-to-Depth substitute.

The module is deliberately narrow: it consumes normalized RGB and a geometry-valid
support mask only for the final padding mask, and emits a floating-point uint8-like
Depth plane.  The caller is responsible for observable-empty detection and for
normalizing/repeating the output for the existing Depth path.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class ObservableEmptyGeometrySubstitute(nn.Module):
    """The frozen seven-convolution R-OE-lite substitute."""

    EXPECTED_TRAINABLE_PARAMETERS = 3_302_785
    OUTPUT_MIN = 0.0
    OUTPUT_MAX = 255.0

    def __init__(self) -> None:
        super().__init__()
        self.e1 = nn.Conv2d(3, 122, kernel_size=3, padding=1, bias=True)
        self.e2 = nn.Conv2d(122, 398, kernel_size=3, padding=1, bias=True)
        self.b1 = nn.Conv2d(398, 256, kernel_size=3, padding=1, bias=True)
        self.b2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.d1 = nn.Conv2d(256, 398, kernel_size=3, padding=1, bias=True)
        self.d2 = nn.Conv2d(398, 122, kernel_size=3, padding=1, bias=True)
        self.head = nn.Conv2d(122, 1, kernel_size=1, padding=0, bias=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.activation = nn.GELU()
        self._initialize_frozen_contract()

        actual = sum(int(parameter.numel()) for parameter in self.parameters() if parameter.requires_grad)
        if actual != self.EXPECTED_TRAINABLE_PARAMETERS:
            raise ValueError(
                "R-OE-lite substitute parameter count mismatch: "
                f"expected {self.EXPECTED_TRAINABLE_PARAMETERS}, got {actual}"
            )

    def _initialize_frozen_contract(self) -> None:
        for layer in (self.e1, self.e2, self.b1, self.b2, self.d1, self.d2, self.head):
            trunc_normal_(layer.weight, std=0.02)
        for layer in (self.e1, self.e2, self.b1, self.b2, self.d1, self.d2):
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.head.bias, 127.5)

    @staticmethod
    def _validate_inputs(rgb: torch.Tensor, geometry_mask: torch.Tensor) -> Tuple[int, int]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"R-OE-lite expects RGB [B,3,H,W], got {tuple(rgb.shape)}")
        if geometry_mask.ndim != 4 or geometry_mask.shape[1] != 1:
            raise ValueError(
                "R-OE-lite expects geometry mask [B,1,H,W], "
                f"got {tuple(geometry_mask.shape)}"
            )
        if geometry_mask.shape[0] != rgb.shape[0] or geometry_mask.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("R-OE-lite geometry mask must match RGB batch and spatial shape")
        if not torch.isfinite(rgb).all():
            raise ValueError("R-OE-lite RGB input contains non-finite values")
        return int(rgb.shape[-2]), int(rgb.shape[-1])

    def forward(self, rgb: torch.Tensor, geometry_mask: torch.Tensor) -> torch.Tensor:
        """Return floating-point uint8-like Depth with exact zero padding."""

        input_height, input_width = self._validate_inputs(rgb, geometry_mask)
        geometry_mask = geometry_mask.to(dtype=torch.bool)

        stage1 = self.activation(self.e1(self.pool(rgb)))
        stage2 = self.activation(self.e2(self.pool(stage1)))
        features = self.activation(self.b1(stage2))
        features = self.activation(self.b2(features))
        features = F.interpolate(
            features,
            size=stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = self.activation(self.d1(features))
        features = self.activation(self.d2(features))
        logits = self.head(features)
        logits = F.interpolate(
            logits,
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )

        clipped = logits.clamp(self.OUTPUT_MIN, self.OUTPUT_MAX)
        straight_through = logits + (clipped - logits).detach()
        return straight_through * geometry_mask.to(dtype=straight_through.dtype)
