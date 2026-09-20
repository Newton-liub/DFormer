#!/usr/bin/env python3
"""MMFR ``Oracle-A`` Depth-validity transport to the frozen multi-scale flip views.

This module converts the **final** Depth validity state produced by the frozen
``MMFR-A1-corruption-basis-v3`` pipeline into a per-view *invalidity* map that the
DFormerv2 geometry path can consume.

Definitions used throughout Oracle-A
------------------------------------
* ``validity_state`` is the boolean final validity of the corrupted Depth on the raw
  aligned grid, i.e. exactly ``utils.dataloader.multimodal_failure_v3``'s explicit
  sequential state.  It is the only accepted Oracle source: it is not re-derived from the
  Depth values, not inferred from predictions and not taken from segmentation labels.
* ``invalidity = 1 - validity`` is float32 with values in ``{0.0, 1.0}`` on the raw grid.
* A view is one frozen ``(scale, flipped)`` combination.  The invalidity map is pushed
  through exactly the geometry of ``tools.evaluate_museg_checkpoint.build_msflip_views``:
  ``cv2.resize(..., INTER_LINEAR)`` -> horizontal flip -> right/bottom zero padding.

Padding convention (deliberate, documented)
-------------------------------------------
Right/bottom padding is filled with ``invalidity = 0`` (i.e. "valid").  The frozen
pipeline defines validity on the raw aligned grid and treats padding as a separate
concept, and the earlier frozen ``DVG-B1`` preregistration used the same convention for
its view-level reliability.  Consequences that must be reported, not hidden:

* padding keeps exactly its original Depth-derived geometry, so ``Oracle-A`` never
  changes the padded region;
* with ``invalidity == 0`` everywhere the gate degenerates to the identity, which is what
  makes the all-valid no-op test exact;
* under ``entire_missing@1.0`` all real tokens become invalid while pad tokens stay
  untouched, so ``Strict`` is *not* bit-identical to ``Depth-Geometry-Off`` on the pad
  strip.  The pad token fraction is measured and reported.

This is a pure NumPy/OpenCV transform: no torch and no model code is imported here, so it
can be qualified on CPU.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

PAD_INVALIDITY = 0.0
VIEW_PAD_DIVISOR = 32


def raw_invalidity(validity_state: np.ndarray) -> np.ndarray:
    """Return the float32 ``1 - validity`` map of a raw-grid boolean validity state."""
    if not isinstance(validity_state, np.ndarray):
        raise TypeError("validity_state must be a numpy.ndarray")
    if validity_state.dtype != np.bool_:
        raise ValueError(f"validity_state must be bool, got {validity_state.dtype}")
    if validity_state.ndim != 2:
        raise ValueError(f"validity_state must be HxW, got shape {validity_state.shape}")
    if validity_state.shape[0] == 0 or validity_state.shape[1] == 0:
        raise ValueError("validity_state spatial shape must be non-empty")
    return np.ascontiguousarray((~validity_state).astype(np.float32))


def build_view_invalidity(
    invalidity_raw: np.ndarray,
    scaled_size_hw: Sequence[int],
    padded_size_hw: Sequence[int],
    flipped: bool,
) -> np.ndarray:
    """Build one view's ``[1, 1, H_pad, W_pad]`` float32 invalidity map.

    ``scaled_size_hw`` / ``padded_size_hw`` must be the frozen view metadata of the same
    view, so the map stays aligned with the Depth tensor the model consumes.
    """
    if invalidity_raw.ndim != 2:
        raise ValueError(f"invalidity_raw must be HxW, got shape {invalidity_raw.shape}")
    if invalidity_raw.dtype != np.float32:
        raise ValueError(f"invalidity_raw must be float32, got {invalidity_raw.dtype}")
    scaled_height, scaled_width = int(scaled_size_hw[0]), int(scaled_size_hw[1])
    padded_height, padded_width = int(padded_size_hw[0]), int(padded_size_hw[1])
    if scaled_height <= 0 or scaled_width <= 0 or padded_height <= 0 or padded_width <= 0:
        raise ValueError("view geometry must be positive")
    if padded_height < scaled_height or padded_width < scaled_width:
        raise ValueError("padded view geometry must not be smaller than the scaled geometry")
    if padded_height % VIEW_PAD_DIVISOR or padded_width % VIEW_PAD_DIVISOR:
        raise ValueError("padded view geometry must be divisible by 32")
    scaled = cv2.resize(
        invalidity_raw,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LINEAR,
    )
    if scaled.shape != (scaled_height, scaled_width):
        raise RuntimeError("cv2.resize returned an unexpected invalidity geometry")
    if flipped:
        scaled = np.flip(scaled, axis=1).copy()
    padded = np.pad(
        scaled,
        ((0, padded_height - scaled_height), (0, padded_width - scaled_width)),
        mode="constant",
        constant_values=PAD_INVALIDITY,
    )
    if padded.shape != (padded_height, padded_width):
        raise RuntimeError("invalidity padding produced an unexpected geometry")
    if not np.isfinite(padded).all():
        raise RuntimeError("view invalidity contains non-finite values")
    if padded.min() < 0.0 or padded.max() > 1.0:
        raise RuntimeError("view invalidity left [0,1]")
    return np.ascontiguousarray(padded[None, None, :, :])


def build_view_invalidity_plan(
    invalidity_raw: np.ndarray,
    views: Sequence[Mapping[str, Any]],
) -> list[np.ndarray]:
    """Build one invalidity map per frozen view, in the frozen view order."""
    return [
        build_view_invalidity(
            invalidity_raw,
            view["scaled_size_hw"],
            view["padded_size_hw"],
            bool(view["flipped"]),
        )
        for view in views
    ]


def raw_grid_validity_from_depth(depth_uint8: np.ndarray) -> np.ndarray:
    """The ``clean`` condition's validity state: ``raw Depth uint8 > 0``."""
    if depth_uint8.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {depth_uint8.shape}")
    return np.ascontiguousarray(depth_uint8 > 0)


def validity_summary(validity_state: np.ndarray) -> dict[str, Any]:
    """Raw-grid validity statistics for evidence bookkeeping."""
    if not isinstance(validity_state, np.ndarray) or validity_state.dtype != np.bool_:
        raise TypeError("validity_state must be a boolean numpy.ndarray")
    total = int(validity_state.size)
    valid = int(np.count_nonzero(validity_state))
    return {
        "raw_grid_hw": [int(validity_state.shape[0]), int(validity_state.shape[1])],
        "raw_pixels": total,
        "valid_pixels": valid,
        "invalid_pixels": total - valid,
        "valid_fraction": float(valid) / float(total),
        "invalid_fraction": float(total - valid) / float(total),
    }
