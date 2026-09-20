#!/usr/bin/env python3
"""Shared Oracle-A definitions: variants, per-view validity injection, unit results.

``MMFR-Oracle-A`` is an inference-only go/no-go experiment.  It never trains anything and
never adds parameters.  Every variant differs from the reference only in how the
Depth-derived geometry contribution of DFormerv2's GSA is treated:

``original``
    The checkpoint's own code path, no oracle argument at all.  This is the reference.
``strict``
    Binary pair reliability from the **final** Depth validity state.  A stage token is
    valid only when every raw Depth pixel that contributes to its interpolated Depth
    value is valid; a pair keeps its Depth-derived geometry contribution only when both
    endpoints are valid.
``aggregated``
    Continuous pair reliability from the same final validity state.  A stage token keeps
    the weighted valid fraction ``c_i`` of exactly the raw pixels that contribute to its
    Depth value, and the pair gate is ``c_i * c_j``.
``geometry_off``
    ``Depth-Geometry-Off`` control: the whole ``weight[1] * mask_d`` term is removed, so
    only the position/spatial prior remains.  No mask is involved.

Definitions of the two no-op variants used by the identity tests live in
:data:`ORACLE_VARIANTS` as well; they drive the *same* oracle code path with an
all-valid ("invalidity == 0 everywhere") map, which must reproduce ``original`` bitwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from utils.dataloader.oracle_a_validity import build_view_invalidity_plan, raw_invalidity

GEOMETRY_ORACLE_IMPLEMENTATION = (
    "models.encoders.DFormerv2.GeoPriorGen: gate only self.weight[1] * mask_d; "
    "invalidity downsampled with F.interpolate(bilinear, align_corners=False), the same "
    "operator GeoPriorGen applies to the Depth patches"
)


@dataclass(frozen=True)
class OracleVariant:
    """One inference-only Oracle-A variant."""

    name: str
    mode: str | None
    depth_geometry_off: bool
    all_valid_mask: bool
    description: str

    @property
    def geometry_implementation(self) -> str:
        if self.depth_geometry_off:
            return "depth-geometry-off: self.weight[1] * mask_d removed entirely"
        if self.all_valid_mask:
            return f"{self.mode} oracle path driven by an all-valid (invalidity == 0) map"
        return GEOMETRY_ORACLE_IMPLEMENTATION

    def requires_mask(self) -> bool:
        return self.mode is not None and not self.depth_geometry_off

    def uses_historical_path(self) -> bool:
        return self.mode is None and not self.depth_geometry_off


ORACLE_VARIANTS: dict[str, OracleVariant] = {
    "original": OracleVariant(
        name="original",
        mode=None,
        depth_geometry_off=False,
        all_valid_mask=False,
        description="A2 epoch-420 reference: no oracle argument, historical geometry path",
    ),
    "strict": OracleVariant(
        name="strict",
        mode="strict",
        depth_geometry_off=False,
        all_valid_mask=False,
        description="Strict Validity Pair Oracle rho_ij = r_i * r_j from the final validity state",
    ),
    "aggregated": OracleVariant(
        name="aggregated",
        mode="continuous",
        depth_geometry_off=False,
        all_valid_mask=False,
        description="Aggregated Validity Oracle rho_ij = c_i * c_j with weighted valid fractions",
    ),
    "geometry_off": OracleVariant(
        name="geometry_off",
        mode=None,
        depth_geometry_off=True,
        all_valid_mask=False,
        description="Depth-Geometry-Off control: remove weight[1]*mask_d, keep position prior",
    ),
    "noop_continuous": OracleVariant(
        name="noop_continuous",
        mode="continuous",
        depth_geometry_off=False,
        all_valid_mask=True,
        description="Identity test: continuous path with an all-valid mask",
    ),
    "noop_strict": OracleVariant(
        name="noop_strict",
        mode="strict",
        depth_geometry_off=False,
        all_valid_mask=True,
        description="Identity test: strict path with an all-valid mask",
    ),
}

REQUIRED_VARIANTS: tuple[str, ...] = ("original", "strict", "aggregated", "geometry_off")
IDENTITY_VARIANTS: tuple[str, ...] = ("noop_continuous", "noop_strict")


def resolve_variants(spec: str | None) -> list[OracleVariant]:
    """Resolve a comma-separated variant list, preserving the requested order."""
    if spec is None:
        names = list(REQUIRED_VARIANTS)
    else:
        names = [token.strip() for token in str(spec).split(",") if token.strip()]
    if not names:
        raise ValueError("variant list must not be empty")
    unknown = [name for name in names if name not in ORACLE_VARIANTS]
    if unknown:
        raise ValueError(f"unknown Oracle-A variants: {unknown}; known: {sorted(ORACLE_VARIANTS)}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate Oracle-A variant in {names}")
    return [ORACLE_VARIANTS[name] for name in names]


def unit_invalidity_plan(
    validity_state: np.ndarray,
    views: Sequence[Mapping[str, Any]],
    variant: OracleVariant,
) -> list[np.ndarray]:
    """Per-view invalidity maps for one ``(sample, condition)`` unit and one variant."""
    plan = build_view_invalidity_plan(raw_invalidity(validity_state), views)
    if variant.all_valid_mask:
        return [np.zeros_like(entry) for entry in plan]
    return plan


def unit_invalidity_summary(validity_state: np.ndarray, views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Masks and per-view invalid fractions for one unit, independent of the variant."""
    plan = build_view_invalidity_plan(raw_invalidity(validity_state), views)
    fractions = [float(entry.mean()) for entry in plan]
    totals = [int(entry.size) for entry in plan]
    return {
        "view_invalid_fraction_mean": float(np.mean(fractions)),
        "view_invalid_fraction_min": float(np.min(fractions)),
        "view_invalid_fraction_max": float(np.max(fractions)),
        "view_invalid_pixels_mean": float(np.mean([fraction * total for fraction, total in zip(fractions, totals)])),
    }


class GeometryOracleModel(nn.Module):
    """Inject one pre-built invalidity map per view into the wrapped segmentation model.

    ``FAST.ForwardTimer`` calls its inner module as ``inner(rgb, depth)`` once per view or
    per same-scale view pair, in the frozen view order.  This wrapper therefore consumes
    the queue that the caller filled for exactly that unit, and fails closed if the queue
    does not cover the incoming batch.  It never touches the Depth tensor itself.
    """

    def __init__(self, inner: nn.Module, variant: OracleVariant) -> None:
        super().__init__()
        self.inner = inner
        self.variant = variant
        self._queue: list[np.ndarray] = []
        self.oracle_calls = 0

    def push_unit(self, per_view_invalidity: Sequence[np.ndarray] | None) -> None:
        if not self.variant.requires_mask():
            if per_view_invalidity:
                raise ValueError(f"variant {self.variant.name} does not take a validity mask")
            self._queue = []
            return
        if per_view_invalidity is None:
            raise ValueError(f"variant {self.variant.name} requires a per-view validity mask")
        self._queue = list(per_view_invalidity)

    def pending(self) -> int:
        return len(self._queue)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if self.variant.uses_historical_path():
            # The reference variant must not pass an oracle argument at all, so the
            # checkpoint's own code path stays byte-identical to the frozen evaluator.
            return self.inner(rgb, depth)
        if not self.variant.requires_mask():
            return self.inner(
                rgb,
                depth,
                geometry_oracle={"mode": "continuous", "invalidity": None, "depth_geometry_off": True},
            )
        count = int(rgb.shape[0])
        if len(self._queue) < count:
            raise RuntimeError(
                f"geometry oracle queue holds {len(self._queue)} views but the evaluator forwarded {count}"
            )
        batch = self._queue[:count]
        self._queue = self._queue[count:]
        stacked = np.ascontiguousarray(np.concatenate(batch, axis=0), dtype=np.float32)
        invalidity = torch.from_numpy(stacked).to(device=rgb.device, dtype=torch.float32)
        self.oracle_calls += 1
        return self.inner(
            rgb,
            depth,
            geometry_oracle={
                "mode": str(self.variant.mode),
                "invalidity": invalidity,
                "depth_geometry_off": bool(self.variant.depth_geometry_off),
            },
        )
