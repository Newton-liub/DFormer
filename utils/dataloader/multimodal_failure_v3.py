#!/usr/bin/env python3
"""MMFR-A1 v3 corruption basis with explicit sequential validity state.

This module starts a new protocol identity from the audited v2 basis.  The v1 and v2
modules remain frozen and are not imported here.  The v3 semantic change is A + MID-A:
missing Depth is an absorbing measurement state for intensity corruptions, while an
integer misalignment transports Depth values and their validity state together.

The six corruption kinds, single severity encoding, relative spatial scales, burden
composition and three-stage curriculum remain compatible with v2.  The validity state
is explicit rather than inferred after corruption:

* ``gaussian_noise``, ``blur`` and ``quantization`` preserve the current state and act
  only on pixels whose current state is valid;
* ``spatial_dropout`` applies a keep mask to the current state;
* ``entire_missing`` clears the state;
* ``misalignment`` applies one identical integer translation to the Depth array and the
  state, followed by the fixed geometry-valid mask.

For Depth, every final state-valid pixel is encoded by a uint8 value in ``[1, 255]``;
state-invalid pixels are encoded as exact zero.  The returned ``validity_state`` is the
final explicit state used by the v3 training helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

import cv2
import numpy as np

PROTOCOL_ID = "MMFR-A1-corruption-basis-v3"
SUPERSEDES = "MMFR-A1-corruption-basis-v2"
SEVERITY_ENCODING = "single"

MODALITIES: Tuple[str, ...] = ("rgb", "depth")
MODALITY_INDEX: Mapping[str, int] = {"rgb": 0, "depth": 1}
RELIABILITY_CHANNELS: int = len(MODALITIES)

FAILURE_KINDS: Tuple[str, ...] = (
    "entire_missing",
    "spatial_dropout",
    "gaussian_noise",
    "blur",
    "quantization",
    "misalignment",
)
DEPTH_ONLY_KINDS: Tuple[str, ...] = ("quantization", "misalignment")

SEVERITY_MIN_EXCLUSIVE = 0.0
SEVERITY_MAX_INCLUSIVE = 1.0
MISSING_BURDEN = 1.0e4
DAMAGE_REFERENCE = 0.25
NOISE_SIGMA_MAX = 48.0
BLUR_SIGMA_FRACTION = 1.0 / 80.0
MISALIGN_MAX_SHIFT_FRACTION = 1.0 / 30.0
MIN_DROPOUT_CELL = 8
DROPOUT_GRID_TARGET = 16

CURRICULUM_SEVERITY_FLOOR = 0.05
CURRICULUM_LIGHT_MAX_SEVERITY = 0.30
CURRICULUM_MODERATE_MAX_SEVERITY = 0.60
ENTIRE_MISSING_HEAVY_SEVERITY = 1.0

if not 0.0 < BLUR_SIGMA_FRACTION < 1.0:
    raise RuntimeError("BLUR_SIGMA_FRACTION must be a fraction in (0, 1)")
if not 0.0 < MISALIGN_MAX_SHIFT_FRACTION < 0.5:
    raise RuntimeError("MISALIGN_MAX_SHIFT_FRACTION must be a fraction in (0, 0.5)")


@dataclass(frozen=True)
class FailureSpec:
    """One explicit failure request; it carries no randomness of its own."""

    modality: str
    kind: str
    severity: float


@dataclass(frozen=True)
class FailureResult:
    """Corrupted arrays, synthetic reliability and the final explicit state."""

    rgb: np.ndarray
    depth: np.ndarray
    reliability: np.ndarray
    validity_state: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _check_rng(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator supplied by the caller")
    return rng


def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a numpy.ndarray")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb dtype must be uint8, got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3 uint8, got shape {rgb.shape}")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError(f"rgb spatial shape must be non-empty, got {rgb.shape}")
    return rgb


def _check_depth(depth: np.ndarray, rgb_shape: Tuple[int, int]) -> np.ndarray:
    if not isinstance(depth, np.ndarray):
        raise TypeError("depth must be a numpy.ndarray")
    if depth.dtype != np.uint8:
        raise ValueError(f"depth dtype must be uint8, got {depth.dtype}")
    if depth.ndim == 2:
        pass
    elif depth.ndim == 3 and depth.shape[2] == 3:
        if not np.array_equal(depth[:, :, 0], depth[:, :, 1]) or not np.array_equal(depth[:, :, 0], depth[:, :, 2]):
            raise ValueError("depth HxWx3 input must have three identical channels")
    else:
        raise ValueError(f"depth must be HW or HxWx3 uint8, got shape {depth.shape}")
    if depth.shape[:2] != tuple(rgb_shape):
        raise ValueError(f"depth spatial shape {depth.shape[:2]} does not match rgb shape {tuple(rgb_shape)}")
    if depth.shape[0] == 0 or depth.shape[1] == 0:
        raise ValueError(f"depth spatial shape must be non-empty, got {depth.shape}")
    return depth


def _check_spec(spec: FailureSpec) -> FailureSpec:
    if not isinstance(spec, FailureSpec):
        raise TypeError(f"specs must contain FailureSpec instances, got {type(spec).__name__}")
    if spec.modality not in MODALITY_INDEX:
        raise ValueError(f"unsupported modality {spec.modality!r}; expected one of {MODALITIES}")
    if spec.kind not in FAILURE_KINDS:
        raise ValueError(f"unsupported failure kind {spec.kind!r}; expected one of {FAILURE_KINDS}")
    severity = float(spec.severity)
    if not math.isfinite(severity):
        raise ValueError(f"severity must be finite, got {spec.severity!r}")
    if not (SEVERITY_MIN_EXCLUSIVE < severity <= SEVERITY_MAX_INCLUSIVE):
        raise ValueError(f"severity must lie in ({SEVERITY_MIN_EXCLUSIVE}, {SEVERITY_MAX_INCLUSIVE}], got {severity}")
    if spec.kind in DEPTH_ONLY_KINDS and spec.modality != "depth":
        raise ValueError(f"failure kind {spec.kind!r} is depth-only, got modality {spec.modality!r}")
    return spec


def _check_state(value: Any, shape: Tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        actual = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        raise ValueError(f"{name} must be a bool array with shape {shape}, got {actual} / {dtype}")
    return value


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def _channel_map(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        return np.max(np.abs(values), axis=2)
    return np.abs(values)


def _normalized_damage(delta: np.ndarray) -> np.ndarray:
    return np.clip(_channel_map(delta).astype(np.float32) / (255.0 * DAMAGE_REFERENCE), 0.0, 1.0)


def _structural_burden(shape: Tuple[int, int], mask: np.ndarray) -> np.ndarray:
    burden = np.zeros(shape, dtype=np.float32)
    burden[mask] = np.float32(MISSING_BURDEN)
    return burden


def _damage_statistics(delta: np.ndarray) -> Mapping[str, Any]:
    damage = _normalized_damage(delta)
    return {
        "realized_damage_mean": float(damage.mean()),
        "realized_damage_max": float(damage.max()),
        "realized_damage_nonzero_fraction": float(np.count_nonzero(damage) / float(damage.size)),
    }


def _dropout_mask(height: int, width: int, severity: float, rng: np.random.Generator):
    cell = max(MIN_DROPOUT_CELL, min(height, width) // DROPOUT_GRID_TARGET)
    grid_h = -(-height // cell)
    grid_w = -(-width // cell)
    total = grid_h * grid_w
    count = min(total, max(1, int(round(float(severity) * total))))
    chosen = rng.choice(total, size=count, replace=False)
    cells = np.zeros(total, dtype=bool)
    cells[np.asarray(chosen, dtype=np.int64)] = True
    cells = cells.reshape(grid_h, grid_w)
    mask = np.repeat(np.repeat(cells, cell, axis=0), cell, axis=1)[:height, :width]
    return mask, {"cell_size": cell, "grid_shape": (grid_h, grid_w), "dropped_cells": count, "grid_cells": total}


def _depth_finalize(image: np.ndarray, state: np.ndarray, geometry_valid: np.ndarray) -> np.ndarray:
    state = state & geometry_valid
    output = np.array(image, dtype=np.uint8, copy=True)
    if output.ndim == 2:
        output[~state] = np.uint8(0)
        output[state] = np.maximum(output[state], np.uint8(1))
    else:
        output[~state, :] = np.uint8(0)
        output[state, :] = np.maximum(output[state, :], np.uint8(1))
    return output


def _masked_assign(original: np.ndarray, candidate: np.ndarray, state: np.ndarray) -> np.ndarray:
    output = np.array(original, dtype=np.uint8, copy=True)
    if output.ndim == 2:
        output[state] = candidate[state]
    else:
        output[state, :] = candidate[state, :]
    return output


def _entire_missing(image, state, geometry_valid, rng, severity, modality):
    corrupted = np.zeros_like(image)
    next_state = np.zeros_like(state, dtype=bool)
    burden = np.full(image.shape[:2], np.float32(MISSING_BURDEN), dtype=np.float32)
    return corrupted, burden, next_state, {
        "corruption_parameter": {"missing_fraction": 1.0},
        "missing_fraction": 1.0,
        "graded": False,
        "validity_transition": "state=0",
    }


def _spatial_dropout(image, state, geometry_valid, rng, severity, modality):
    height, width = image.shape[:2]
    mask, info = _dropout_mask(height, width, severity, rng)
    next_state = state & ~mask
    candidate = np.array(image, copy=True)
    if candidate.ndim == 2:
        candidate[mask] = np.uint8(0)
    else:
        candidate[mask, :] = np.uint8(0)
    burden = _structural_burden((height, width), mask)
    corrupted = _masked_assign(image, candidate, next_state)
    if modality == "depth":
        corrupted = _depth_finalize(corrupted, next_state, geometry_valid)
    damage = _damage_statistics(corrupted.astype(np.float32) - image.astype(np.float32))
    info.update(damage)
    info.update(
        {
            "corruption_parameter": {
                "cell_size": info["cell_size"],
                "grid_shape": info["grid_shape"],
                "dropped_cells": info["dropped_cells"],
                "grid_cells": info["grid_cells"],
            },
            "missing_fraction": float(np.count_nonzero(mask)) / float(height * width),
            "dropped_pixels": int(np.count_nonzero(mask)),
            "graded": False,
            "validity_transition": "state &= keep_mask",
        }
    )
    return corrupted, burden, next_state & geometry_valid, info


def _gaussian_noise(image, state, geometry_valid, rng, severity, modality):
    sigma = float(severity) * NOISE_SIGMA_MAX
    if modality == "depth" and image.ndim == 3:
        shape = image.shape[:2] + (1,)
    else:
        shape = image.shape
    unit = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    candidate = _to_uint8(image.astype(np.float32) + unit * np.float32(sigma))
    corrupted = _masked_assign(image, candidate, state)
    if modality == "depth":
        corrupted = _depth_finalize(corrupted, state, geometry_valid)
    delta = corrupted.astype(np.float32) - image.astype(np.float32)
    burden = _normalized_damage(delta)
    info = {
        "corruption_parameter": {"sigma_uint8": sigma},
        "sigma": sigma,
        "graded": True,
        "validity_transition": "state unchanged",
    }
    info.update(_damage_statistics(delta))
    return corrupted, burden, state & geometry_valid, info


def _blur(image, state, geometry_valid, rng, severity, modality):
    height, width = image.shape[:2]
    reference = min(height, width)
    sigma = float(severity) * BLUR_SIGMA_FRACTION * float(reference)
    ksize = max(3, 2 * int(math.ceil(2.0 * sigma)) + 1)
    state_float = state.astype(np.float32)
    if image.ndim == 3:
        numerator = cv2.GaussianBlur(
            image.astype(np.float32) * state_float[:, :, None],
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        denominator = cv2.GaussianBlur(
            state_float,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                numerator,
                denominator[:, :, None],
                out=np.zeros_like(numerator),
                where=denominator[:, :, None] > 1.0e-6,
            )
    else:
        numerator = cv2.GaussianBlur(
            image.astype(np.float32) * state_float,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        denominator = cv2.GaussianBlur(
            state_float,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 1.0e-6,
            )
    candidate = _to_uint8(normalized)
    corrupted = _masked_assign(image, candidate, state)
    if modality == "depth":
        corrupted = _depth_finalize(corrupted, state, geometry_valid)
    delta = corrupted.astype(np.float32) - image.astype(np.float32)
    burden = _normalized_damage(delta)
    info = {
        "corruption_parameter": {
            "sigma_px": sigma,
            "sigma_fraction_of_min_side": BLUR_SIGMA_FRACTION * float(severity),
            "reference_min_side_px": float(reference),
            "ksize": ksize,
            "mask_normalized": True,
            "epsilon": 1.0e-6,
        },
        "sigma": sigma,
        "ksize": ksize,
        "graded": True,
        "validity_transition": "state unchanged",
    }
    info.update(_damage_statistics(delta))
    return corrupted, burden, state & geometry_valid, info


def _quantization(image, state, geometry_valid, rng, severity, modality):
    levels = int(np.clip(round(256.0 * (1.0 - float(severity))), 2, 256))
    if levels >= 256:
        candidate = image.copy()
    else:
        step = 255.0 / float(levels - 1)
        candidate = _to_uint8(np.rint(image.astype(np.float32) / step) * step)
    corrupted = _masked_assign(image, candidate, state)
    if modality == "depth":
        corrupted = _depth_finalize(corrupted, state, geometry_valid)
    delta = corrupted.astype(np.float32) - image.astype(np.float32)
    burden = _normalized_damage(delta)
    info = {
        "corruption_parameter": {"levels": levels, "step_uint8": 255.0 / float(max(1, levels - 1))},
        "levels": levels,
        "graded": True,
        "validity_transition": "state unchanged",
    }
    info.update(_damage_statistics(delta))
    return corrupted, burden, state & geometry_valid, info


def misalignment_reference_displacement(height: int, width: int) -> float:
    return math.hypot(
        MISALIGN_MAX_SHIFT_FRACTION * float(height),
        MISALIGN_MAX_SHIFT_FRACTION * float(width),
    )


def _translate(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = array.shape[:2]
    src_y = slice(max(0, -dy), height - max(0, dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    translated = np.zeros_like(array)
    if array.ndim == 2:
        translated[dst_y, dst_x] = array[src_y, src_x]
    else:
        translated[dst_y, dst_x, :] = array[src_y, src_x, :]
    return translated


def _misalignment(image, state, geometry_valid, rng, severity, modality):
    height, width = image.shape[:2]
    max_shift_y = float(severity) * MISALIGN_MAX_SHIFT_FRACTION * float(height)
    max_shift_x = float(severity) * MISALIGN_MAX_SHIFT_FRACTION * float(width)
    unit_y = float(rng.random()) * 2.0 - 1.0
    unit_x = float(rng.random()) * 2.0 - 1.0
    dy = int(round(unit_y * max_shift_y))
    dx = int(round(unit_x * max_shift_x))
    if dy == 0 and dx == 0:
        if max_shift_x >= max_shift_y and width > 1:
            dx = 1
        else:
            dy = 1
    dy = int(np.clip(dy, -(height - 1), height - 1))
    dx = int(np.clip(dx, -(width - 1), width - 1))

    translated = _translate(image, dy, dx)
    translated_state = _translate(state, dy, dx)
    next_state = translated_state & geometry_valid
    if modality == "depth":
        corrupted = _depth_finalize(translated, next_state, geometry_valid)
    else:
        corrupted = translated.copy()
        if corrupted.ndim == 2:
            corrupted[~next_state] = np.uint8(0)
        else:
            corrupted[~next_state, :] = np.uint8(0)

    invalid = np.ones((height, width), dtype=bool)
    src_y = slice(max(0, -dy), height - max(0, dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    invalid[dst_y, dst_x] = False
    displacement = math.hypot(float(dy), float(dx))
    reference = misalignment_reference_displacement(height, width)
    in_bounds = float(min(1.0, displacement / reference)) if reference > 0.0 else 0.0
    burden = np.full((height, width), np.float32(in_bounds), dtype=np.float32)
    burden[invalid] = np.float32(MISSING_BURDEN)
    info = {
        "corruption_parameter": {
            "dy_px": dy,
            "dx_px": dx,
            "max_shift_y_px": max_shift_y,
            "max_shift_x_px": max_shift_x,
            "shift_fraction_of_axis": MISALIGN_MAX_SHIFT_FRACTION * float(severity),
            "displacement_px": displacement,
            "reference_displacement_px": reference,
            "transport": "integer-translate-depth-and-validity-together",
        },
        "dy": dy,
        "dx": dx,
        "displacement_px": displacement,
        "reference_displacement_px": reference,
        "normalized_burden": in_bounds,
        "invalid_pixels": int(np.count_nonzero(invalid)),
        "invalid_fraction": float(np.count_nonzero(invalid)) / float(height * width),
        "graded": True,
        "validity_transition": "same integer translation as Depth values",
    }
    info.update(_damage_statistics(corrupted.astype(np.float32) - image.astype(np.float32)))
    return corrupted, burden, next_state, info


_KERNELS = {
    "entire_missing": _entire_missing,
    "spatial_dropout": _spatial_dropout,
    "gaussian_noise": _gaussian_noise,
    "blur": _blur,
    "quantization": _quantization,
    "misalignment": _misalignment,
}


def apply_failures(
    rgb: np.ndarray,
    depth: np.ndarray,
    specs: Sequence[FailureSpec] = (),
    rng: np.random.Generator | None = None,
    *,
    depth_validity: np.ndarray | None = None,
    validity_mask: np.ndarray | None = None,
) -> FailureResult:
    """Apply v3 specs and return arrays, burden-derived reliability and final state.

    ``depth_validity`` is the initial explicit Depth state, normally
    ``(depth_uint8 > 0) & valid_mask`` from the training helper.  ``validity_mask`` is
    the fixed post-crop geometry mask.  Both are required to be bool arrays when given.
    The two masks are deliberately separate so MID-A can transport the observation and
    then re-apply the geometry boundary without allowing padding to become a sample.
    """
    _check_rng(rng)
    rgb = _check_rgb(rgb)
    depth = _check_depth(depth, rgb.shape[:2])
    spec_list = tuple(specs)
    for spec in spec_list:
        _check_spec(spec)

    height, width = rgb.shape[:2]
    geometry = np.ones((height, width), dtype=bool) if validity_mask is None else _check_state(validity_mask, (height, width), "validity_mask")
    inferred_depth_state = (depth if depth.ndim == 2 else depth[:, :, 0]) > 0
    initial_depth_state = inferred_depth_state if depth_validity is None else _check_state(depth_validity, (height, width), "depth_validity")
    initial_depth_state = initial_depth_state & geometry
    expected_initial_state = inferred_depth_state & geometry
    if not np.array_equal(initial_depth_state, expected_initial_state):
        raise ValueError("depth_validity must equal (depth_uint8 > 0) & validity_mask at v3 entry")
    reliability = np.ones((RELIABILITY_CHANNELS, height, width), dtype=np.float32)
    if not spec_list:
        depth_plane = depth if depth.ndim == 2 else depth[:, :, 0]
        if np.any(depth_plane[initial_depth_state] == 0) or np.any(depth_plane[~initial_depth_state] != 0):
            raise RuntimeError("v3 Depth validity state and uint8 sentinel encoding disagree")
        return FailureResult(
            rgb=rgb,
            depth=depth,
            reliability=reliability,
            validity_state=initial_depth_state,
            metadata={},
        )

    rgb_shape, rgb_dtype = rgb.shape, rgb.dtype
    depth_shape, depth_dtype = depth.shape, depth.dtype
    images = {"rgb": rgb, "depth": depth}
    states = {"rgb": np.ones((height, width), dtype=bool), "depth": initial_depth_state.copy()}
    records = []
    for spec in spec_list:
        modality = spec.modality
        severity = float(spec.severity)
        previous_state = states[modality]
        corrupted, burden, next_state, info = _KERNELS[spec.kind](
            images[modality], previous_state, geometry, rng, severity, modality
        )
        next_state = next_state & geometry
        if modality == "depth":
            corrupted = _depth_finalize(corrupted, next_state, geometry)
        images[modality] = corrupted
        states[modality] = next_state
        with np.errstate(under="ignore"):
            decay = np.exp(-burden).astype(np.float32)
        reliability[MODALITY_INDEX[modality]] *= decay
        record = {
            "modality": modality,
            "kind": spec.kind,
            "severity": severity,
            "severity_encoding": SEVERITY_ENCODING,
            "validity_state_before_pixels": int(np.count_nonzero(previous_state)),
            "validity_state_after_pixels": int(np.count_nonzero(next_state)),
            "validity_invalidated_pixels": int(np.count_nonzero(previous_state & ~next_state)),
            "validity_newly_valid_pixels": int(np.count_nonzero(~previous_state & next_state)),
        }
        record.update(info)
        record.update(
            {
                "burden_min": float(burden.min()),
                "burden_max": float(burden.max()),
                "burden_mean": float(burden.mean()),
                "reliability_min": float(reliability[MODALITY_INDEX[modality]].min()),
                "reliability_mean": float(reliability[MODALITY_INDEX[modality]].mean()),
            }
        )
        records.append(record)

    depth_output = _depth_finalize(images["depth"], states["depth"], geometry)
    images["depth"] = depth_output
    if images["rgb"].shape != rgb_shape or images["rgb"].dtype != rgb_dtype:
        raise RuntimeError("failure synthesis changed the rgb shape or dtype")
    if images["depth"].shape != depth_shape or images["depth"].dtype != depth_dtype:
        raise RuntimeError("failure synthesis changed the depth shape or dtype")
    state = states["depth"] & geometry
    depth_plane = images["depth"] if images["depth"].ndim == 2 else images["depth"][:, :, 0]
    if np.any(depth_plane[state] == 0) or np.any(depth_plane[~state] != 0):
        raise RuntimeError("v3 Depth validity state and uint8 sentinel encoding disagree")
    if not np.isfinite(reliability).all():
        raise RuntimeError("failure synthesis produced a non-finite reliability target")
    if reliability.min() < 0.0 or reliability.max() > 1.0:
        raise RuntimeError("failure synthesis produced a reliability target outside [0, 1]")

    metadata = {
        "protocol": PROTOCOL_ID,
        "severity_encoding": SEVERITY_ENCODING,
        "validity_state_semantics": "sequential explicit state; intensity-preserving and MID-A transport",
        "num_specs": len(spec_list),
        "specs": tuple(records),
        "validity_state_initial_pixels": int(np.count_nonzero(initial_depth_state)),
        "validity_state_final_pixels": int(np.count_nonzero(state)),
        "reliability_min": float(reliability.min()),
        "reliability_mean": float(reliability.mean()),
    }
    return FailureResult(
        rgb=images["rgb"],
        depth=images["depth"],
        reliability=reliability,
        validity_state=state,
        metadata=metadata,
    )


class FailureCurriculumSampler:
    """The unchanged six-kind, three-stage v2-compatible curriculum."""

    def __init__(
        self,
        rng: np.random.Generator,
        max_specs: int = 2,
        kinds: Sequence[str] = FAILURE_KINDS,
    ) -> None:
        _check_rng(rng)
        if not isinstance(max_specs, int) or max_specs < 1:
            raise ValueError(f"max_specs must be a positive int, got {max_specs!r}")
        kind_list = tuple(kinds)
        if not kind_list:
            raise ValueError("kinds must not be empty")
        for kind in kind_list:
            if kind not in FAILURE_KINDS:
                raise ValueError(f"unsupported failure kind {kind!r}; expected one of {FAILURE_KINDS}")
        self.rng = rng
        self.max_specs = max_specs
        self.kinds = kind_list

    def candidates(self, include_entire_missing: bool = True) -> Tuple[Tuple[str, str], ...]:
        pairs = []
        for modality in MODALITIES:
            for kind in self.kinds:
                if kind in DEPTH_ONLY_KINDS and modality != "depth":
                    continue
                if kind == "entire_missing" and not include_entire_missing:
                    continue
                pairs.append((modality, kind))
        if not pairs:
            raise ValueError("no (modality, kind) candidate remains under the current kinds")
        return tuple(pairs)

    def sample(self, progress: float) -> Tuple[FailureSpec, ...]:
        value = float(progress)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"progress must be a finite value in [0, 1], got {progress!r}")
        if value < 1.0 / 3.0:
            max_severity, max_count, heavy = CURRICULUM_LIGHT_MAX_SEVERITY, 1, False
        elif value < 2.0 / 3.0:
            max_severity, max_count, heavy = CURRICULUM_MODERATE_MAX_SEVERITY, min(2, self.max_specs), False
        else:
            max_severity, max_count, heavy = 1.0, self.max_specs, True
        pairs = self.candidates(include_entire_missing=heavy)
        count = min(max_count, len(pairs))
        if count > 1:
            count = int(self.rng.integers(1, count + 1))
        chosen = self.rng.choice(len(pairs), size=count, replace=False)
        specs = []
        for index in np.asarray(chosen, dtype=np.int64):
            modality, kind = pairs[int(index)]
            severity = float(self.rng.uniform(CURRICULUM_SEVERITY_FLOOR, max_severity))
            if kind == "entire_missing":
                severity = ENTIRE_MISSING_HEAVY_SEVERITY
            specs.append(FailureSpec(modality=modality, kind=kind, severity=severity))
        return tuple(specs)


def sample_failure_specs(
    progress: float,
    rng: np.random.Generator,
    max_specs: int = 2,
    kinds: Sequence[str] = FAILURE_KINDS,
) -> Tuple[FailureSpec, ...]:
    """Draw one v3 curriculum spec tuple without persistent sampler state."""
    return FailureCurriculumSampler(rng=rng, max_specs=max_specs, kinds=kinds).sample(progress)
