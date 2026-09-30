"""Action-utility residual module for the MMFR A-v1 proposal."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def depth_stats(
    observed_depth: Tensor,
    geometry_mask: Tensor | None = None,
) -> Tensor:
    """Return detached FP32 per-image depth statistics with shape ``[B, 4]``.

    Statistics are zero ratio, nonzero-depth population mean and standard
    deviation, and the mean absolute horizontal/vertical difference over pairs
    whose two pixels are valid and nonzero. ``geometry_mask=True`` marks pixels
    inside the legal crop/pad region.
    """
    if not isinstance(observed_depth, Tensor):
        raise TypeError("observed_depth must be a torch.Tensor")
    if observed_depth.ndim != 4 or observed_depth.shape[1] != 1:
        raise ValueError("observed_depth must have shape [B, 1, H, W]")
    if observed_depth.shape[0] <= 0 or observed_depth.shape[2] <= 0 or observed_depth.shape[3] <= 0:
        raise ValueError("observed_depth must have non-empty batch and spatial dimensions")
    if not observed_depth.is_floating_point():
        raise TypeError("observed_depth must have a floating-point dtype")

    detached_depth = observed_depth.detach()
    if not bool(torch.isfinite(detached_depth).all()):
        raise ValueError("observed_depth must contain only finite values")
    if not bool(((detached_depth >= 0) & (detached_depth <= 1)).all()):
        raise ValueError("observed_depth values must be in [0, 1]")

    if geometry_mask is None:
        valid = torch.ones_like(detached_depth, dtype=torch.bool)
    else:
        if not isinstance(geometry_mask, Tensor):
            raise TypeError("geometry_mask must be a torch.Tensor or None")
        if geometry_mask.dtype != torch.bool:
            raise TypeError("geometry_mask must have bool dtype")
        if geometry_mask.shape != detached_depth.shape:
            raise ValueError("geometry_mask must have the same shape as observed_depth")
        if geometry_mask.device != detached_depth.device:
            raise ValueError("geometry_mask and observed_depth must be on the same device")
        valid = geometry_mask.detach()

    depth = detached_depth.to(dtype=torch.float32)
    dims = (1, 2, 3)
    valid_count = valid.sum(dim=dims, dtype=torch.float32)
    zero_count = ((depth == 0) & valid).sum(dim=dims, dtype=torch.float32)
    zero_ratio = zero_count / valid_count.clamp_min(1)

    nonzero = (depth > 0) & valid
    nonzero_count = nonzero.sum(dim=dims, dtype=torch.float32)
    nonzero_depth = torch.where(nonzero, depth, torch.zeros_like(depth))
    nonzero_mean = nonzero_depth.sum(dim=dims) / nonzero_count.clamp_min(1)
    centered = depth - nonzero_mean[:, None, None, None]
    squared_deviation = torch.where(nonzero, centered.square(), torch.zeros_like(depth))
    nonzero_std = torch.sqrt(
        squared_deviation.sum(dim=dims) / nonzero_count.clamp_min(1)
    )

    horizontal_pair = (
        valid[..., :, 1:]
        & valid[..., :, :-1]
        & (depth[..., :, 1:] > 0)
        & (depth[..., :, :-1] > 0)
    )
    vertical_pair = (
        valid[..., 1:, :]
        & valid[..., :-1, :]
        & (depth[..., 1:, :] > 0)
        & (depth[..., :-1, :] > 0)
    )
    horizontal_diff = torch.where(
        horizontal_pair,
        (depth[..., :, 1:] - depth[..., :, :-1]).abs(),
        torch.zeros_like(depth[..., :, 1:]),
    )
    vertical_diff = torch.where(
        vertical_pair,
        (depth[..., 1:, :] - depth[..., :-1, :]).abs(),
        torch.zeros_like(depth[..., 1:, :]),
    )
    pair_count = horizontal_pair.sum(dim=dims, dtype=torch.float32)
    pair_count = pair_count + vertical_pair.sum(dim=dims, dtype=torch.float32)
    difference_sum = horizontal_diff.sum(dim=dims) + vertical_diff.sum(dim=dims)
    neighbor_diff_mean = difference_sum / pair_count.clamp_min(1)

    return torch.stack(
        (zero_ratio, nonzero_mean, nonzero_std, neighbor_diff_mean), dim=1
    ).detach()


class ActionUtilityResidual(nn.Module):
    """Stage-2 residual with an optional per-image observed-depth utility gate."""

    stage_index = 2
    modes = ("off", "full", "learned")

    def __init__(self) -> None:
        super().__init__()
        self.proposal = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 256, kernel_size=1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Linear(260, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.proposal[-1].weight)
        nn.init.zeros_(self.proposal[-1].bias)

    @staticmethod
    def _validate_stage2(feature: Tensor) -> None:
        if not isinstance(feature, Tensor):
            raise TypeError("stage-2 feature must be a torch.Tensor")
        if feature.ndim != 4 or feature.shape[1] != 256:
            raise ValueError("stage-2 feature must have shape [B, 256, H, W]")
        if feature.shape[0] <= 0 or feature.shape[2] <= 0 or feature.shape[3] <= 0:
            raise ValueError("stage-2 feature must have non-empty batch and spatial dimensions")
        if not feature.is_floating_point():
            raise TypeError("stage-2 feature must have a floating-point dtype")
        if not bool(torch.isfinite(feature.detach()).all()):
            raise ValueError("stage-2 feature must contain only finite values")

    @classmethod
    def _validate_features(cls, features: Sequence[Tensor]) -> Tensor:
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise ValueError("features must be a four-level tuple or list")
        batch_size = None
        for index, feature in enumerate(features):
            if not isinstance(feature, Tensor) or feature.ndim != 4:
                raise ValueError("each feature level must be a four-dimensional tensor")
            if batch_size is None:
                batch_size = feature.shape[0]
            elif feature.shape[0] != batch_size:
                raise ValueError("all feature levels must have the same batch size")
            if index == cls.stage_index:
                cls._validate_stage2(feature)
        return features[cls.stage_index]

    def gate_value(
        self,
        feature: Tensor,
        observed_depth: Tensor,
        geometry_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one continuous sigmoid gate value per image, shaped ``[B]``."""
        self._validate_stage2(feature)
        stats = depth_stats(observed_depth, geometry_mask)
        if feature.device != observed_depth.device:
            raise ValueError("stage-2 feature and observed_depth must be on the same device")
        if feature.shape[0] != observed_depth.shape[0]:
            raise ValueError("stage-2 feature and observed_depth must have the same batch size")

        pooled_feature = feature.detach().to(dtype=torch.float32).mean(dim=(2, 3))
        gate_input = torch.cat((pooled_feature, stats), dim=1)
        gate_dtype = self.gate[0].weight.dtype
        gate_value = self.gate(gate_input.to(dtype=gate_dtype)).squeeze(-1)
        return gate_value

    def forward(
        self,
        features: Sequence[Tensor],
        observed_depth: Tensor | None = None,
        geometry_mask: Tensor | None = None,
        *,
        mode: str = "learned",
    ) -> Sequence[Tensor]:
        if mode == "off":
            return features
        if mode not in self.modes:
            raise ValueError(f"mode must be one of {self.modes}, got {mode!r}")

        stage2 = self._validate_features(features)
        if mode == "full":
            residual = self.proposal(stage2)
        else:
            if observed_depth is None:
                raise ValueError("observed_depth is required in learned mode")
            gate = self.gate_value(stage2, observed_depth, geometry_mask)
            residual = self.proposal(stage2) * gate[:, None, None, None]

        updated = list(features)
        updated[self.stage_index] = stage2 + residual
        return tuple(updated) if isinstance(features, tuple) else updated
