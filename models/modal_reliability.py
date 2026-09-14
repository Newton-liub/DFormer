#!/usr/bin/env python3
"""Signal-feature reliability head and no-op geometry adapter for MMFR (MUSeg).

Standalone scaffold for the ``MMFR-A1-train-corruption-basis-v1`` plan. It is
deliberately not wired into ``DFormerv2``, the training loop, the evaluator or any
checkpoint path, and it declares no compatibility with the Quick-B0 checkpoint:
the reliability head and the adapter carry new parameters.

Contract
--------
* ``rgb`` is ``[B, 3, H, W]`` and ``depth`` is ``[B, 1, H, W]`` or ``[B, 3, H, W]``;
  both are floating point and inside ``[0, 1]``.
* Signal features are fixed differentiable operators and carry no learnable
  parameters. They are learned inputs, never hard gates.
* Reliability logits are ``[B, 2, H, W]`` ordered ``rgb`` then ``depth``.
* The reliability pyramid uses ``torch.nn.functional.interpolate(mode="area")`` and
  keeps ``[0, 1]`` semantics. It is not an OpenCV ``INTER_AREA`` reimplementation and
  makes no DVG-B1 Oracle protocol claim.
* The four-level geometry adapter predicts spatial/depth scales through
  ``2 * sigmoid(raw)`` contracted around ``1`` by ``ADAPTER_SCALE_MARGIN`` with a
  zeroed output layer: the scales stay strictly inside ``(0, 2)`` even where float32
  ``sigmoid`` saturates and return exactly ``1`` at initialization for any legal
  input, i.e. an exact no-op on the existing prior.

Feature ranges are all ``[0, 1]``: intensities, local mean, local standard
deviation, Sobel magnitude, Laplacian magnitude, Gaussian high-frequency residual,
depth validity and the cosine-based cross-modal gradient alignment
``0.5 * (cos + 1)`` (where depth is invalid the alignment is ``0``).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITY_ORDER: Tuple[str, ...] = ("rgb", "depth")
RELIABILITY_CHANNELS: int = len(MODALITY_ORDER)
PYRAMID_SCALES: Tuple[int, ...] = (4, 8, 16, 32)

FEATURE_NAMES: Tuple[str, ...] = (
    "rgb_intensity",
    "rgb_local_mean",
    "rgb_local_std",
    "rgb_sobel",
    "rgb_laplacian",
    "rgb_high_freq",
    "depth_intensity",
    "depth_local_mean",
    "depth_local_std",
    "depth_sobel",
    "depth_laplacian",
    "depth_high_freq",
    "depth_validity",
    "cross_grad_alignment",
)
LOCAL_WINDOW = 3
GAUSSIAN_WINDOW = 5
FEATURE_EPS = 1e-6
#: Adapter scales come from ``ADAPTER_SCALE_GAIN * sigmoid(raw)``, contracted around
#: the no-op scale ``1`` by ``ADAPTER_SCALE_MARGIN``: strictly inside
#: ``(ADAPTER_SCALE_MIN, ADAPTER_SCALE_MAX)`` and exactly ``1`` when ``raw == 0``.
ADAPTER_SCALE_GAIN = 2.0
ADAPTER_SCALE_MIN = 0.0
ADAPTER_SCALE_MAX = 2.0
#: Inward margin that keeps the realized scale strictly inside the open interval.
#: float32 ``ADAPTER_SCALE_GAIN * sigmoid(raw)`` saturates to exactly ``0`` / ``2``
#: for large ``|raw|``, so the plain sigmoid value cannot serve as the bound. The
#: margin costs at most this much of the scale range above ``1`` and below ``1`` and
#: leaves the exact ``1`` at ``raw == 0`` untouched.
ADAPTER_SCALE_MARGIN = 1.0e-3


def _require_image_tensor(image: torch.Tensor, name: str, channels: Sequence[int]) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not image.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor, got {image.dtype}")
    if image.dim() != 4:
        raise ValueError(f"{name} must be [B, C, H, W], got shape {tuple(image.shape)}")
    if image.shape[1] not in tuple(channels):
        raise ValueError(f"{name} must have channels in {tuple(channels)}, got {image.shape[1]}")
    if image.shape[0] == 0 or image.shape[2] == 0 or image.shape[3] == 0:
        raise ValueError(f"{name} must have non-empty batch and spatial dims, got shape {tuple(image.shape)}")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} contains non-finite values")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(f"{name} must be inside [0, 1]")
    return image


def _require_reliability(reliability: torch.Tensor, name: str = "reliability") -> torch.Tensor:
    if not torch.is_tensor(reliability):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not reliability.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor, got {reliability.dtype}")
    if reliability.dim() != 4 or reliability.shape[1] != RELIABILITY_CHANNELS:
        raise ValueError(f"{name} must be [B, {RELIABILITY_CHANNELS}, H, W], got shape {tuple(reliability.shape)}")
    if reliability.shape[0] == 0 or reliability.shape[2] == 0 or reliability.shape[3] == 0:
        raise ValueError(f"{name} must have non-empty batch and spatial dims, got shape {tuple(reliability.shape)}")
    if not torch.isfinite(reliability).all():
        raise ValueError(f"{name} contains non-finite values")
    if reliability.min() < 0.0 or reliability.max() > 1.0:
        raise ValueError(f"{name} must be inside [0, 1]")
    return reliability


def _require_feature_map(features: torch.Tensor, in_channels: int) -> torch.Tensor:
    if not torch.is_tensor(features):
        raise TypeError("features must be a torch.Tensor")
    if not features.is_floating_point():
        raise TypeError(f"features must be a floating point tensor, got {features.dtype}")
    if features.dim() != 4:
        raise ValueError(f"features must be [B, C, H, W], got shape {tuple(features.shape)}")
    if features.shape[1] != in_channels:
        raise ValueError(f"features must have {in_channels} channels, got {features.shape[1]}")
    if features.shape[0] == 0 or features.shape[2] == 0 or features.shape[3] == 0:
        raise ValueError(f"features must have non-empty batch and spatial dims, got shape {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if features.min() < 0.0 or features.max() > 1.0:
        raise ValueError("features must be inside [0, 1]")
    return features


def _luminance(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    return image.mean(dim=1, keepdim=True)


def split_features(features: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split a stacked feature tensor into named ``[B, 1, H, W]`` maps."""
    _require_feature_map(features, len(FEATURE_NAMES))
    return {name: features[:, index : index + 1] for index, name in enumerate(FEATURE_NAMES)}


class SignalFeatureExtractor(nn.Module):
    """Fixed differentiable signal features; buffers only, no learnable parameters."""

    def __init__(self, eps: float = FEATURE_EPS) -> None:
        super().__init__()
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError(f"eps must be a positive finite float, got {eps!r}")
        self.eps = float(eps)
        gaussian = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        gaussian = gaussian / gaussian.sum()
        self.register_buffer("gaussian_kernel", (gaussian[:, None] * gaussian[None, :]).view(1, 1, 5, 5))
        self.register_buffer(
            "sobel_x", torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        )
        self.register_buffer(
            "sobel_y", torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        )
        self.register_buffer(
            "laplacian", torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3) / 8.0
        )

    def _depth_validity(self, depth: torch.Tensor, depth_valid: torch.Tensor | None) -> torch.Tensor:
        if depth_valid is None:
            valid = (depth > 0).amax(dim=1, keepdim=True)
        else:
            if not torch.is_tensor(depth_valid):
                raise TypeError("depth_valid must be a torch.Tensor")
            if not depth_valid.is_floating_point():
                raise TypeError(f"depth_valid must be a floating point tensor, got {depth_valid.dtype}")
            if depth_valid.dim() != 4 or depth_valid.shape[1] != 1:
                raise ValueError(f"depth_valid must be [B, 1, H, W], got shape {tuple(depth_valid.shape)}")
            if depth_valid.shape[0] != depth.shape[0] or depth_valid.shape[-2:] != depth.shape[-2:]:
                raise ValueError("depth_valid must match the depth batch and spatial shape")
            if not torch.isfinite(depth_valid).all():
                raise ValueError("depth_valid contains non-finite values")
            if depth_valid.min() < 0.0 or depth_valid.max() > 1.0:
                raise ValueError("depth_valid must be inside [0, 1]")
            valid = depth_valid > 0.5
        return valid.to(depth.dtype)

    def _modality_features(self, gray: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        intensity = gray
        mean = F.avg_pool2d(gray, LOCAL_WINDOW, stride=1, padding=1, count_include_pad=False)
        mean_square = F.avg_pool2d(gray * gray, LOCAL_WINDOW, stride=1, padding=1, count_include_pad=False)
        local_std = torch.sqrt((mean_square - mean * mean).clamp_min(0.0) + self.eps)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        sobel = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        laplacian = torch.abs(F.conv2d(gray, self.laplacian, padding=1))
        high_frequency = torch.abs(gray - F.conv2d(gray, self.gaussian_kernel, padding=GAUSSIAN_WINDOW // 2))
        return intensity, mean, local_std, sobel, laplacian, high_frequency

    def _cross_modal_alignment(
        self, rgb_gray: torch.Tensor, depth_gray: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        grad_x_rgb = F.conv2d(rgb_gray, self.sobel_x, padding=1)
        grad_y_rgb = F.conv2d(rgb_gray, self.sobel_y, padding=1)
        grad_x_depth = F.conv2d(depth_gray, self.sobel_x, padding=1)
        grad_y_depth = F.conv2d(depth_gray, self.sobel_y, padding=1)
        dot = grad_x_rgb * grad_x_depth + grad_y_rgb * grad_y_depth
        magnitude = torch.sqrt(
            (grad_x_rgb * grad_x_rgb + grad_y_rgb * grad_y_rgb + self.eps)
            * (grad_x_depth * grad_x_depth + grad_y_depth * grad_y_depth + self.eps)
        )
        cosine = (dot / magnitude).clamp(-1.0, 1.0)
        return 0.5 * (cosine + 1.0) * validity

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the stacked ``[B, 14, H, W]`` feature map.

        When ``depth_valid`` is omitted, validity follows the corruption module's
        encoding convention ``depth > 0``; pass the explicit mask when available,
        because an all-zero depth map is not distinguishable from a dark clean one.
        """
        rgb = _require_image_tensor(rgb, "rgb", (3,))
        depth = _require_image_tensor(depth, "depth", (1, 3))
        if rgb.shape[0] != depth.shape[0]:
            raise ValueError(f"rgb and depth batch sizes differ: {rgb.shape[0]} vs {depth.shape[0]}")
        if rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError(f"rgb and depth spatial shapes differ: {tuple(rgb.shape[-2:])} vs {tuple(depth.shape[-2:])}")

        validity = self._depth_validity(depth, depth_valid)
        rgb_gray = _luminance(rgb)
        depth_gray = _luminance(depth)
        alignment = self._cross_modal_alignment(rgb_gray, depth_gray, validity)
        features = torch.cat((*self._modality_features(rgb_gray), *self._modality_features(depth_gray), validity, alignment), dim=1)
        if features.shape[1] != len(FEATURE_NAMES):
            raise RuntimeError(f"expected {len(FEATURE_NAMES)} features, built {features.shape[1]}")
        if not torch.isfinite(features).all():
            raise RuntimeError("signal features contain non-finite values")
        if features.min() < 0.0 or features.max() > 1.0:
            raise RuntimeError("signal features left the declared [0, 1] range")
        return features


class ModalReliabilityHead(nn.Module):
    """Small convolutional head producing one reliability logit per modality."""

    def __init__(
        self,
        in_channels: int = len(FEATURE_NAMES),
        hidden_channels: int = 16,
        num_modalities: int = RELIABILITY_CHANNELS,
    ) -> None:
        super().__init__()
        if num_modalities != RELIABILITY_CHANNELS:
            raise ValueError(f"num_modalities must be {RELIABILITY_CHANNELS}, got {num_modalities}")
        if in_channels < 1 or hidden_channels < 1:
            raise ValueError("in_channels and hidden_channels must be positive")
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_modalities = int(num_modalities)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.num_modalities, 1, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _require_feature_map(features, self.in_channels)
        logits = self.net(features)
        if logits.shape[1] != self.num_modalities or logits.shape[-2:] != features.shape[-2:]:
            raise RuntimeError(f"unexpected reliability logits shape {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("reliability head produced non-finite logits")
        return logits


class ReliabilityPyramid(nn.Module):
    """PyTorch ``area`` reliability pyramid over the four geometry stages."""

    def __init__(self, scales: Sequence[int] = PYRAMID_SCALES) -> None:
        super().__init__()
        scale_list = tuple(int(scale) for scale in scales)
        if not scale_list:
            raise ValueError("scales must not be empty")
        if any(scale < 1 for scale in scale_list):
            raise ValueError(f"scales must be positive, got {scale_list}")
        if len(set(scale_list)) != len(scale_list):
            raise ValueError(f"scales must be distinct, got {scale_list}")
        self.scales = scale_list

    def forward(self, reliability: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        _require_reliability(reliability, "reliability")
        height, width = reliability.shape[-2:]
        levels = []
        for scale in self.scales:
            if height % scale or width % scale:
                raise ValueError(f"reliability shape {(height, width)} is not divisible by pyramid scale {scale}")
            level = F.interpolate(reliability, size=(height // scale, width // scale), mode="area")
            if not torch.isfinite(level).all():
                raise RuntimeError(f"area pyramid level {scale} contains non-finite values")
            if level.min() < 0.0 or level.max() > 1.0:
                raise RuntimeError(f"area pyramid level {scale} left the [0, 1] range")
            levels.append(level)
        return tuple(levels)


class GeometryContributionAdapter(nn.Module):
    """Learns per-level spatial/depth scales inside ``(0, 2)`` from a reliability pyramid.

    The scales are ``2 * sigmoid(raw)`` contracted around ``1`` by
    ``ADAPTER_SCALE_MARGIN`` with a zero-initialized output layer, so the untrained
    adapter returns exactly ``1`` for every input and is an exact no-op on the existing
    geometry prior, while the learned scale stays strictly inside
    ``(ADAPTER_SCALE_MIN, ADAPTER_SCALE_MAX)`` even when float32 ``sigmoid`` saturates.
    The adapter does not consume or modify ``GeoPriorGen``; it only defines the future
    interface.
    """

    def __init__(self, num_levels: int = len(PYRAMID_SCALES), hidden_channels: int = 8) -> None:
        super().__init__()
        if not isinstance(num_levels, int) or num_levels < 1:
            raise ValueError(f"num_levels must be a positive int, got {num_levels!r}")
        if not isinstance(hidden_channels, int) or hidden_channels < 1:
            raise ValueError(f"hidden_channels must be a positive int, got {hidden_channels!r}")
        self.num_levels = num_levels
        self.hidden_channels = hidden_channels
        self.trunk = nn.Sequential(
            nn.Linear(4 * self.num_levels, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, 2 * self.num_levels),
        )
        output = self.trunk[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def level_statistics(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        """Per-level mean and standard deviation of both modalities, ``[B, 4L]``."""
        level_list = list(levels)
        if len(level_list) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} pyramid levels, got {len(level_list)}")
        batch = None
        statistics = []
        for index, level in enumerate(level_list):
            _require_reliability(level, f"pyramid level {index}")
            if batch is None:
                batch = level.shape[0]
            elif level.shape[0] != batch:
                raise ValueError(f"pyramid level {index} batch {level.shape[0]} differs from {batch}")
            flattened = level.flatten(2)
            statistics.append(flattened.mean(dim=2))
            statistics.append(flattened.std(dim=2, correction=0))
        return torch.cat(statistics, dim=1)

    def forward(self, levels: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(spatial_scales, depth_scales)``, each ``[B, num_levels]`` inside ``(0, 2)``."""
        statistics = self.level_statistics(levels)
        raw = self.trunk(statistics).view(statistics.shape[0], 2, self.num_levels)
        # ``centered`` is the float32 image of ``2 * sigmoid(raw) - 1`` in [-1, 1] and is
        # exactly ``0`` at ``raw == 0``; contracting it by ``ADAPTER_SCALE_MARGIN`` keeps
        # the realized scale strictly inside the open interval even at saturation.
        half_gain = 0.5 * ADAPTER_SCALE_GAIN
        centered = ADAPTER_SCALE_GAIN * torch.sigmoid(raw) - half_gain
        scales = half_gain + (half_gain - ADAPTER_SCALE_MARGIN) * centered
        if not torch.isfinite(scales).all():
            raise RuntimeError("geometry adapter produced non-finite scales")
        if scales.min() <= ADAPTER_SCALE_MIN or scales.max() >= ADAPTER_SCALE_MAX:
            raise RuntimeError(
                f"geometry adapter scales left the open interval ({ADAPTER_SCALE_MIN}, {ADAPTER_SCALE_MAX})"
            )
        return scales[:, 0], scales[:, 1]


def combine_geometry_contributions(
    spatial_contribution: torch.Tensor,
    depth_contribution: torch.Tensor,
    spatial_scale: torch.Tensor,
    depth_scale: torch.Tensor,
) -> torch.Tensor:
    """Combine one geometry level as ``spatial_scale * G_s + depth_scale * G_d``.

    Scales are ``[B]`` tensors and broadcast over the trailing geometry dimensions.
    Adapter outputs already satisfy ``0 < scale < 2``; this helper only requires
    strictly positive scales so other callers can pass their own weighting.
    """
    if not torch.is_tensor(spatial_contribution) or not torch.is_tensor(depth_contribution):
        raise TypeError("geometry contributions must be torch tensors")
    if spatial_contribution.shape != depth_contribution.shape:
        raise ValueError(
            f"spatial and depth contributions must share a shape, got {tuple(spatial_contribution.shape)} "
            f"and {tuple(depth_contribution.shape)}"
        )
    if spatial_contribution.dim() < 2:
        raise ValueError(f"geometry contributions must be at least [B, X], got {tuple(spatial_contribution.shape)}")
    for name, scale in (("spatial_scale", spatial_scale), ("depth_scale", depth_scale)):
        if not torch.is_tensor(scale):
            raise TypeError(f"{name} must be a torch.Tensor")
        if scale.dim() != 1 or scale.shape[0] != spatial_contribution.shape[0]:
            raise ValueError(f"{name} must be [B] matching batch {spatial_contribution.shape[0]}")
        if not torch.isfinite(scale).all():
            raise ValueError(f"{name} contains non-finite values")
        if scale.min() <= 0.0:
            raise ValueError(f"{name} must be strictly positive")
    view = (-1,) + (1,) * (spatial_contribution.dim() - 1)
    return spatial_scale.view(view) * spatial_contribution + depth_scale.view(view) * depth_contribution


class ModalReliabilityEstimator(nn.Module):
    """Feature extractor, reliability head and area pyramid in one frozen-interface call."""

    def __init__(
        self,
        hidden_channels: int = 16,
        scales: Sequence[int] = PYRAMID_SCALES,
        eps: float = FEATURE_EPS,
    ) -> None:
        super().__init__()
        self.features = SignalFeatureExtractor(eps=eps)
        self.head = ModalReliabilityHead(in_channels=len(FEATURE_NAMES), hidden_channels=hidden_channels)
        self.pyramid = ReliabilityPyramid(scales=scales)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid: torch.Tensor | None = None,
    ) -> Dict[str, object]:
        features = self.features(rgb, depth, depth_valid)
        logits = self.head(features)
        reliability = torch.sigmoid(logits)
        return {"logits": logits, "reliability": reliability, "pyramid": self.pyramid(reliability)}


def continuous_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Stable BCE on continuous ``[0, 1]`` reliability targets.

    ``valid_mask`` may be broadcastable to ``logits``. It accepts a boolean mask (the
    usual training-loop validity mask, cast to the loss dtype before broadcasting) or
    a floating point mask inside ``[0, 1]`` for soft weighting. With
    ``reduction="mean"`` the loss is averaged over the masked elements only.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    if not torch.is_tensor(logits) or not torch.is_tensor(target):
        raise TypeError("logits and target must be torch tensors")
    if logits.shape != target.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} does not match target shape {tuple(target.shape)}")
    if not logits.is_floating_point() or not target.is_floating_point():
        raise TypeError("logits and target must be floating point tensors")
    if logits.numel() == 0:
        raise ValueError("logits and target must not be empty")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    if not torch.isfinite(target).all():
        raise ValueError("target contains non-finite values")
    if target.min() < 0.0 or target.max() > 1.0:
        raise ValueError("target must be inside [0, 1]")

    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if not torch.isfinite(loss).all():
        raise RuntimeError("continuous BCE produced non-finite values")

    if valid_mask is None:
        if reduction == "none":
            return loss
        return loss.mean() if reduction == "mean" else loss.sum()

    if not torch.is_tensor(valid_mask):
        raise TypeError("valid_mask must be a torch.Tensor")
    if valid_mask.dim() != logits.dim():
        raise ValueError(f"valid_mask must have {logits.dim()} dims to match logits, got {valid_mask.dim()}")
    if valid_mask.dtype == torch.bool:
        mask = valid_mask.to(loss.dtype)
    elif valid_mask.is_floating_point():
        if not torch.isfinite(valid_mask).all():
            raise ValueError("valid_mask contains non-finite values")
        if valid_mask.min() < 0.0 or valid_mask.max() > 1.0:
            raise ValueError("valid_mask must be inside [0, 1]")
        mask = valid_mask
    else:
        raise TypeError(f"valid_mask must be bool or floating point, got {valid_mask.dtype}")
    try:
        mask = mask.expand_as(loss)
    except RuntimeError as error:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} cannot broadcast to logits") from error

    weighted = loss * mask
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    return weighted.sum() / mask.sum().clamp_min(1.0)
