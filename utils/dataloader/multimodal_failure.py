#!/usr/bin/env python3
"""Controlled multi-modal failure synthesis for the MMFR (MUSeg) scaffolding.

Standalone helper for the ``MMFR-A1-train-corruption-basis-v1`` plan. It produces
corrupted uint8 RGB/Depth arrays, a continuous per-pixel reliability target and
auditable metadata. It never reads or writes process-wide random state, never
mutates the caller's arrays, and is not wired into the training main chain.

Contract
--------
* ``rgb``: uint8 ``H x W x 3``.
* ``depth``: uint8 ``H x W`` or uint8 ``H x W x 3`` whose channels are identical.
  Outputs keep the input shape and dtype.
* ``reliability``: float32 ``[2, H, W]`` inside ``[0, 1]``, channel order ``rgb``
  then ``depth``.
* Empty spec: arrays element-wise identical, all-ones reliability, empty metadata.
* All randomness comes from a caller supplied ``numpy.random.Generator``.

Reliability semantics
---------------------
Every spec contributes a non-negative local burden ``b_k(p)`` and the target is
the multiplicative composition ``R_m(p) = prod_k exp(-b_k(p)) = exp(-sum_k b_k(p))``.

* ``entire_missing`` is a whole-modality heavy failure: it always covers the entire
  modality and its burden is the constant ``MISSING_BURDEN``, so the affected
  modality reliability is exactly ``0`` for every legal severity. ``severity`` is
  accepted and recorded for bookkeeping but does not change the result.
* ``spatial_dropout`` is also *non-graded*: ``severity`` scales the dropped extent
  and dropped pixels receive ``b = MISSING_BURDEN * severity``, so their target
  reliability underflows to exactly ``0``.
* ``gaussian_noise``, ``blur`` and ``quantization`` are *graded*: ``severity`` sets
  the corruption intensity and ``b = severity * local damage``, where the local
  damage is the realized per-pixel change (maximum absolute channel change)
  normalized so that a change of ``DAMAGE_REFERENCE`` of the uint8 range saturates
  the burden. ``quantization`` (depth bit-depth / stair loss) and ``misalignment``
  are depth-only.
* ``misalignment`` uses a *displacement-driven* burden that is uniform over all
  in-bounds pixels and independent of image texture: the known integer shift gives
  ``b = sqrt(dx^2 + dy^2) / (sqrt(2) * MISALIGN_MAX_PX)`` clipped to ``[0, 1]``, so a
  flat Depth image shifted by the same amount is not reported as reliable.
  ``severity`` only caps the drawn shift magnitude, so this kind still reports
  ``graded=True``, but the burden is never derived from the realized pixel delta.
  Out-of-bounds pixels are explicitly invalid and receive ``MISSING_BURDEN``.

First version deliberately excludes haze, dust synthesis, non-rigid warp, Poisson
shot noise and camera response models; those need a separate protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

import cv2
import numpy as np

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

#: Burden of a structurally missing pixel; ``exp(-MISSING_BURDEN)`` underflows to 0.
MISSING_BURDEN = 1.0e4
#: Fraction of the uint8 dynamic range at which graded local damage saturates.
DAMAGE_REFERENCE = 0.25
NOISE_SIGMA_MAX = 48.0
BLUR_SIGMA_MAX = 6.0
MISALIGN_MAX_PX = 16
MIN_DROPOUT_CELL = 8
DROPOUT_GRID_TARGET = 16

CURRICULUM_SEVERITY_FLOOR = 0.05
CURRICULUM_LIGHT_MAX_SEVERITY = 0.30
CURRICULUM_MODERATE_MAX_SEVERITY = 0.60
#: ``entire_missing`` is a heavy-only failure and is always sampled at full severity.
ENTIRE_MISSING_HEAVY_SEVERITY = 1.0


@dataclass(frozen=True)
class FailureSpec:
    """One explicit failure request; it carries no randomness of its own."""

    modality: str
    kind: str
    severity: float


@dataclass(frozen=True)
class FailureResult:
    """Corrupted arrays, continuous reliability target and audit metadata."""

    rgb: np.ndarray
    depth: np.ndarray
    reliability: np.ndarray
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


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def _channel_map(values: np.ndarray) -> np.ndarray:
    """Reduce a per-pixel array to a two-dimensional map (max over channels)."""
    if values.ndim == 3:
        return np.max(np.abs(values), axis=2)
    return np.abs(values)


def _graded_burden(delta: np.ndarray, severity: float) -> np.ndarray:
    damage = np.clip(_channel_map(delta).astype(np.float32) / (255.0 * DAMAGE_REFERENCE), 0.0, 1.0)
    return (float(severity) * damage).astype(np.float32)


def _structural_burden(shape: Tuple[int, int], severity: float, mask: np.ndarray) -> np.ndarray:
    burden = np.zeros(shape, dtype=np.float32)
    burden[mask] = np.float32(MISSING_BURDEN * float(severity))
    return burden


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
    return mask, {"cell_size": cell, "grid_shape": (grid_h, grid_w), "dropped_cells": count}


def _entire_missing(image, rng, severity, ctx):
    corrupted = np.zeros_like(image)
    burden = np.full(image.shape[:2], np.float32(MISSING_BURDEN), dtype=np.float32)
    return corrupted, burden, {"missing_fraction": 1.0, "graded": False}


def _spatial_dropout(image, rng, severity, ctx):
    height, width = image.shape[:2]
    mask, info = _dropout_mask(height, width, severity, rng)
    corrupted = image.copy()
    corrupted[mask] = 0
    burden = _structural_burden((height, width), severity, mask)
    info.update(
        {
            "missing_fraction": float(np.count_nonzero(mask)) / float(height * width),
            "dropped_pixels": int(np.count_nonzero(mask)),
            "graded": False,
        }
    )
    return corrupted, burden, info


def _gaussian_noise(image, rng, severity, ctx):
    sigma = float(severity) * NOISE_SIGMA_MAX
    if ctx["modality"] == "depth" and image.ndim == 3:
        shape = image.shape[:2] + (1,)
    else:
        shape = image.shape
    noise = rng.normal(0.0, sigma, size=shape).astype(np.float32)
    corrupted = _to_uint8(image.astype(np.float32) + noise)
    burden = _graded_burden(corrupted.astype(np.float32) - image.astype(np.float32), severity)
    return corrupted, burden, {"sigma": sigma, "graded": True}


def _blur(image, rng, severity, ctx):
    sigma = float(severity) * BLUR_SIGMA_MAX
    ksize = max(3, 2 * int(math.ceil(2.0 * sigma)) + 1)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    burden = _graded_burden(blurred.astype(np.float32) - image.astype(np.float32), severity)
    return blurred, burden, {"sigma": sigma, "ksize": ksize, "graded": True}


def _quantization(image, rng, severity, ctx):
    levels = int(np.clip(round(256.0 * (1.0 - float(severity))), 2, 256))
    if levels >= 256:
        quantized = image.copy()
    else:
        step = 255.0 / float(levels - 1)
        quantized = _to_uint8(np.rint(image.astype(np.float32) / step) * step)
    burden = _graded_burden(quantized.astype(np.float32) - image.astype(np.float32), severity)
    return quantized, burden, {"levels": levels, "graded": True}


def _displacement_burden(dy: int, dx: int) -> float:
    """Uniform in-bounds burden from the known shift magnitude, clipped to ``[0, 1]``."""
    displacement = math.hypot(float(dy), float(dx))
    return min(1.0, displacement / (math.sqrt(2.0) * MISALIGN_MAX_PX))


def _misalignment(image, rng, severity, ctx):
    height, width = image.shape[:2]
    max_shift = max(1, int(round(float(severity) * MISALIGN_MAX_PX)))
    offsets = rng.integers(-max_shift, max_shift + 1, size=2)
    dy, dx = int(offsets[0]), int(offsets[1])
    if dy == 0 and dx == 0:
        dx = 1
    src_y = slice(max(0, -dy), height - max(0, dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    corrupted = np.zeros_like(image)
    corrupted[dst_y, dst_x] = image[src_y, src_x]
    invalid = np.ones((height, width), dtype=bool)
    invalid[dst_y, dst_x] = False
    displacement = math.hypot(float(dy), float(dx))
    in_bounds = _displacement_burden(dy, dx)
    burden = np.full((height, width), np.float32(in_bounds), dtype=np.float32)
    burden[invalid] = np.float32(MISSING_BURDEN)
    info = {
        "dy": dy,
        "dx": dx,
        "displacement_px": displacement,
        "normalized_burden": in_bounds,
        "invalid_pixels": int(np.count_nonzero(invalid)),
        "invalid_fraction": float(np.count_nonzero(invalid)) / float(height * width),
        "graded": True,
    }
    return corrupted, burden, info


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
) -> FailureResult:
    """Apply failure specs in order and return corrupted arrays plus reliability target.

    Specs acting on the same modality compose sequentially, so the second spec sees
    the output of the first. Reliability is the multiplicative composition of all
    per-spec burdens, therefore adding specs can only lower the target.

    An empty ``specs`` returns the caller's arrays unchanged (no copy) with an
    all-ones target and empty metadata; copy first if an independent buffer is needed.
    """
    _check_rng(rng)
    rgb = _check_rgb(rgb)
    depth = _check_depth(depth, rgb.shape[:2])
    spec_list = tuple(specs)
    for spec in spec_list:
        _check_spec(spec)

    height, width = rgb.shape[:2]
    reliability = np.ones((RELIABILITY_CHANNELS, height, width), dtype=np.float32)
    if not spec_list:
        return FailureResult(rgb=rgb, depth=depth, reliability=reliability, metadata={})

    rgb_shape, rgb_dtype = rgb.shape, rgb.dtype
    depth_shape, depth_dtype = depth.shape, depth.dtype
    images = {"rgb": rgb, "depth": depth}
    records = []
    for spec in spec_list:
        modality = spec.modality
        severity = float(spec.severity)
        ctx = {"modality": modality}
        corrupted, burden, info = _KERNELS[spec.kind](images[modality], rng, severity, ctx)
        images[modality] = corrupted
        with np.errstate(under="ignore"):
            decay = np.exp(-burden).astype(np.float32)
        reliability[MODALITY_INDEX[modality]] *= decay
        record = {"modality": modality, "kind": spec.kind, "severity": severity}
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

    if images["rgb"].shape != rgb_shape or images["rgb"].dtype != rgb_dtype:
        raise RuntimeError("failure synthesis changed the rgb shape or dtype")
    if images["depth"].shape != depth_shape or images["depth"].dtype != depth_dtype:
        raise RuntimeError("failure synthesis changed the depth shape or dtype")
    if not np.isfinite(reliability).all():
        raise RuntimeError("failure synthesis produced a non-finite reliability target")
    if reliability.min() < 0.0 or reliability.max() > 1.0:
        raise RuntimeError("failure synthesis produced a reliability target outside [0, 1]")

    metadata = {
        "num_specs": len(spec_list),
        "specs": tuple(records),
        "reliability_min": float(reliability.min()),
        "reliability_mean": float(reliability.mean()),
    }
    return FailureResult(rgb=images["rgb"], depth=images["depth"], reliability=reliability, metadata=metadata)


class FailureCurriculumSampler:
    """Severity/composition curriculum that only emits failure specs.

    ``progress in [0, 1]`` selects a stage: single mild failures first, then
    moderate single or paired failures, then heavy failures up to ``max_specs``.
    Each stage caps both the severity and the number of specs, and the spec count
    is drawn uniformly from ``1..stage_cap``. ``entire_missing`` is a whole-modality
    heavy failure, so it is never offered in the light or moderate stage and is
    always sampled at ``ENTIRE_MISSING_HEAVY_SEVERITY`` when it is offered. The
    sampler never touches image data; all draws use the caller supplied generator.
    """

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
    """Draw one curriculum spec tuple without constructing a persistent sampler."""
    return FailureCurriculumSampler(rng=rng, max_specs=max_specs, kinds=kinds).sample(progress)
