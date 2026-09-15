#!/usr/bin/env python3
"""Controlled multi-modal failure synthesis, revision ``MMFR-A1-corruption-basis-v2``.

This module is the v2 revision of ``utils/dataloader/multimodal_failure.py``. The v1
module is frozen and keeps its historical synthetic-burden semantics; nothing here
imports from, or mutates, it. Revision is required because two properties of v1 are not
auditable under the v2 protocol:

1. **Severity was encoded twice.** ``gaussian_noise``, ``blur`` and ``quantization``
   used ``severity`` to set the corruption intensity *and* multiplied the realized
   damage burden by ``severity`` a second time, so the reported burden did not equal
   the realized normalized damage. In v2 ``severity`` only selects the corruption
   parameter and the continuous burden is the realized normalized damage itself.
2. **``misalignment`` and ``blur`` severities were absolute pixel quantities.** The
   same severity therefore meant different degradations on the ``480 x 640`` training
   crop and on the original ``932 x 1082`` aligned grid. In v2 both are defined as
   fractions of the actual grid and converted to pixels per resolution.

Frozen semantics that are *not* changed by this revision:

* the three-stage curriculum (mild / moderate / heavy severity caps ``0.30`` / ``0.60``
  / ``1.0``, spec counts ``1`` / ``1-2`` / up to ``max_specs``, ``entire_missing``
  heavy-only at severity ``1.0``, severity floor ``0.05``);
* the multiplicative reliability composition ``R_m(p) = prod_k exp(-b_{m,k}(p))``;
* the ``entire_missing`` / ``spatial_dropout`` structural encodings as whole-modality
  and block-wise zeroing;
* ``gaussian_noise`` and ``quantization`` as intensity-space corruptions, whose
  severity keeps its v1 intensity meaning (``sigma`` up to ``48`` uint8 units, bit-depth
  steps down to two levels);
* the output contract (uint8 RGB/Depth of unchanged shape and dtype, float32
  ``[2, H, W]`` reliability target in ``[0, 1]``, channel order RGB then Depth) and the
  fail-closed validation of unsupported specs, shapes, dtypes and non-finite targets.

These corruptions are model-input / representation-level synthetic corruptions. They are
not a Kinect physical noise model and must not be reported as one.

Contract
--------
* ``rgb``: uint8 ``H x W x 3``.
* ``depth``: uint8 ``H x W`` or uint8 ``H x W x 3`` whose channels are identical.
  Outputs keep the input shape and dtype.
* ``reliability``: float32 ``[2, H, W]`` inside ``[0, 1]``, channel order ``rgb`` then
  ``depth``.
* Empty spec: arrays element-wise identical, all-ones reliability, empty metadata.
* All randomness comes from a caller supplied ``numpy.random.Generator``; process-wide
  Python, NumPy and PyTorch random state is never read or written.

Burden semantics per kind
-------------------------
* ``entire_missing``: whole-modality zeroing, constant burden ``MISSING_BURDEN``.
  ``severity`` is recorded for bookkeeping only.
* ``spatial_dropout``: block-wise zeroing whose extent is selected by ``severity``;
  dropped pixels receive ``MISSING_BURDEN``.
* ``gaussian_noise``: ``sigma = severity * NOISE_SIGMA_MAX`` uint8 units, and the burden
  is the realized per-pixel maximum absolute channel change normalized by
  ``255 * DAMAGE_REFERENCE``, clipped to ``[0, 1]``.
* ``blur``: ``sigma = severity * BLUR_SIGMA_FRACTION * min(H, W)`` pixels (so severity is
  a fraction of the grid rather than an absolute pixel count), and the burden is the
  same realized normalized damage.
* ``quantization``: ``levels = round(256 * (1 - severity))`` clipped to ``[2, 256]``, and
  the burden is the same realized normalized damage.
* ``misalignment``: the per-axis maximum displacement is
  ``severity * MISALIGN_MAX_SHIFT_FRACTION * H`` and ``... * W``; the drawn displacement
  is reported in pixels, and the in-bounds burden is
  ``hypot(dy, dx) / hypot(frac * H, frac * W)``, i.e. the realized displacement divided
  by the displacement the protocol permits at this input scale and severity ``1.0``.
  Out-of-bounds pixels are explicitly invalid and receive ``MISSING_BURDEN``.

Monotonicity intent
-------------------
For every kind, increasing ``severity`` can only increase the corruption parameter, and
the realized damage / burden is monotone non-decreasing while the reliability target is
monotone non-increasing. The noise and misalignment kernels realize this exactly for a
fixed RNG stream: they draw the *unit* deviate once and scale it by the severity-driven
parameter, so the same random draw at a higher severity produces the same pattern with a
larger amplitude or displacement. ``tools/mmfr/severity_burden_audit.py`` checks this
property on real and synthetic fixtures and records the mapping
``severity -> corruption parameter -> realized damage -> burden -> target``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

import cv2
import numpy as np

PROTOCOL_ID = "MMFR-A1-corruption-basis-v2"
SUPERSEDES = "MMFR-A1-corruption-basis-v1"
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

#: Burden of a structurally missing pixel; ``exp(-MISSING_BURDEN)`` underflows to 0.
MISSING_BURDEN = 1.0e4
#: Fraction of the uint8 dynamic range at which graded local damage saturates.
DAMAGE_REFERENCE = 0.25
#: Intensity-space corruption caps, unchanged from v1 (uint8 units).
NOISE_SIGMA_MAX = 48.0
#: ``blur`` sigma at severity ``1.0`` as a fraction of ``min(H, W)``. The fraction equals
#: the v1 absolute cap ``6.0`` px on the frozen ``480``-pixel training crop.
BLUR_SIGMA_FRACTION = 1.0 / 80.0
#: ``misalignment`` maximum displacement at severity ``1.0`` as a fraction of each axis.
#: The fraction equals the v1 absolute cap ``16`` px on the frozen ``480``-pixel crop side.
MISALIGN_MAX_SHIFT_FRACTION = 1.0 / 30.0
MIN_DROPOUT_CELL = 8
DROPOUT_GRID_TARGET = 16

CURRICULUM_SEVERITY_FLOOR = 0.05
CURRICULUM_LIGHT_MAX_SEVERITY = 0.30
CURRICULUM_MODERATE_MAX_SEVERITY = 0.60
#: ``entire_missing`` is a heavy-only failure and is always sampled at full severity.
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


def _normalized_damage(delta: np.ndarray) -> np.ndarray:
    """Realized normalized damage in ``[0, 1]``: the single severity encoding of v2.

    ``severity`` already chose the corruption parameter that produced ``delta``, so the
    damage is *not* scaled by severity again.
    """
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


def _entire_missing(image, rng, severity, ctx):
    corrupted = np.zeros_like(image)
    burden = np.full(image.shape[:2], np.float32(MISSING_BURDEN), dtype=np.float32)
    return corrupted, burden, {
        "corruption_parameter": {"missing_fraction": 1.0},
        "missing_fraction": 1.0,
        "graded": False,
    }


def _spatial_dropout(image, rng, severity, ctx):
    height, width = image.shape[:2]
    mask, info = _dropout_mask(height, width, severity, rng)
    corrupted = image.copy()
    corrupted[mask] = 0
    burden = _structural_burden((height, width), mask)
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
        }
    )
    return corrupted, burden, info


def _gaussian_noise(image, rng, severity, ctx):
    sigma = float(severity) * NOISE_SIGMA_MAX
    if ctx["modality"] == "depth" and image.ndim == 3:
        shape = image.shape[:2] + (1,)
    else:
        shape = image.shape
    # Draw the unit deviate and scale it, so a fixed RNG stream yields exactly
    # proportional realizations across severities (monotone realized damage).
    unit = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    noise = unit * np.float32(sigma)
    corrupted = _to_uint8(image.astype(np.float32) + noise)
    delta = corrupted.astype(np.float32) - image.astype(np.float32)
    burden = _normalized_damage(delta)
    info = {"corruption_parameter": {"sigma_uint8": sigma}, "sigma": sigma, "graded": True}
    info.update(_damage_statistics(delta))
    return corrupted, burden, info


def _blur(image, rng, severity, ctx):
    height, width = image.shape[:2]
    reference = min(height, width)
    sigma = float(severity) * BLUR_SIGMA_FRACTION * float(reference)
    ksize = max(3, 2 * int(math.ceil(2.0 * sigma)) + 1)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    delta = blurred.astype(np.float32) - image.astype(np.float32)
    burden = _normalized_damage(delta)
    info = {
        "corruption_parameter": {
            "sigma_px": sigma,
            "sigma_fraction_of_min_side": BLUR_SIGMA_FRACTION * float(severity),
            "reference_min_side_px": float(reference),
            "ksize": ksize,
        },
        "sigma": sigma,
        "ksize": ksize,
        "graded": True,
    }
    info.update(_damage_statistics(delta))
    return blurred, burden, info


def _quantization(image, rng, severity, ctx):
    levels = int(np.clip(round(256.0 * (1.0 - float(severity))), 2, 256))
    if levels >= 256:
        quantized = image.copy()
    else:
        step = 255.0 / float(levels - 1)
        quantized = _to_uint8(np.rint(image.astype(np.float32) / step) * step)
    delta = quantized.astype(np.float32) - image.astype(np.float32)
    burden = _normalized_damage(delta)
    info = {
        "corruption_parameter": {"levels": levels, "step_uint8": 255.0 / float(max(1, levels - 1))},
        "levels": levels,
        "graded": True,
    }
    info.update(_damage_statistics(delta))
    return quantized, burden, info


def misalignment_reference_displacement(height: int, width: int) -> float:
    """Displacement the protocol permits at severity ``1.0`` for this input scale."""
    return math.hypot(
        MISALIGN_MAX_SHIFT_FRACTION * float(height),
        MISALIGN_MAX_SHIFT_FRACTION * float(width),
    )


def _misalignment(image, rng, severity, ctx):
    height, width = image.shape[:2]
    max_shift_y = float(severity) * MISALIGN_MAX_SHIFT_FRACTION * float(height)
    max_shift_x = float(severity) * MISALIGN_MAX_SHIFT_FRACTION * float(width)
    # Unit draws are taken once and scaled by the severity-driven caps, so a fixed RNG
    # stream gives a monotone non-decreasing realized displacement as severity grows.
    unit_y = float(rng.random()) * 2.0 - 1.0
    unit_x = float(rng.random()) * 2.0 - 1.0
    dy = int(round(unit_y * max_shift_y))
    dx = int(round(unit_x * max_shift_x))
    if dy == 0 and dx == 0:
        # Keep a strictly non-trivial displacement: fall back to a single pixel along the
        # axis with the larger permitted displacement.
        if max_shift_x >= max_shift_y and width > 1:
            dx = 1
        else:
            dy = 1
    dy = int(np.clip(dy, -(height - 1), height - 1))
    dx = int(np.clip(dx, -(width - 1), width - 1))

    src_y = slice(max(0, -dy), height - max(0, dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    corrupted = np.zeros_like(image)
    corrupted[dst_y, dst_x] = image[src_y, src_x]
    invalid = np.ones((height, width), dtype=bool)
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
        },
        "dy": dy,
        "dx": dx,
        "displacement_px": displacement,
        "reference_displacement_px": reference,
        "normalized_burden": in_bounds,
        "invalid_pixels": int(np.count_nonzero(invalid)),
        "invalid_fraction": float(np.count_nonzero(invalid)) / float(height * width),
        "graded": True,
    }
    info.update(_damage_statistics(corrupted.astype(np.float32) - image.astype(np.float32)))
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
    """Apply v2 failure specs in order and return corrupted arrays plus reliability target.

    Specs acting on the same modality compose sequentially, so the second spec sees the
    output of the first. Reliability is the multiplicative composition of all per-spec
    burdens, therefore adding specs can only lower the target.

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
        record = {
            "modality": modality,
            "kind": spec.kind,
            "severity": severity,
            "severity_encoding": SEVERITY_ENCODING,
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

    if images["rgb"].shape != rgb_shape or images["rgb"].dtype != rgb_dtype:
        raise RuntimeError("failure synthesis changed the rgb shape or dtype")
    if images["depth"].shape != depth_shape or images["depth"].dtype != depth_dtype:
        raise RuntimeError("failure synthesis changed the depth shape or dtype")
    if not np.isfinite(reliability).all():
        raise RuntimeError("failure synthesis produced a non-finite reliability target")
    if reliability.min() < 0.0 or reliability.max() > 1.0:
        raise RuntimeError("failure synthesis produced a reliability target outside [0, 1]")

    metadata = {
        "protocol": PROTOCOL_ID,
        "severity_encoding": SEVERITY_ENCODING,
        "num_specs": len(spec_list),
        "specs": tuple(records),
        "reliability_min": float(reliability.min()),
        "reliability_mean": float(reliability.mean()),
    }
    return FailureResult(rgb=images["rgb"], depth=images["depth"], reliability=reliability, metadata=metadata)


class FailureCurriculumSampler:
    """Three-stage severity/composition curriculum that only emits failure specs.

    ``progress in [0, 1]`` selects a stage: single mild failures first, then moderate
    single or paired failures, then heavy failures up to ``max_specs``. Each stage caps
    both severity and spec count, and the spec count is drawn uniformly from
    ``1..stage_cap``. ``entire_missing`` is a whole-modality heavy failure, so it is
    never offered earlier and is always sampled at ``ENTIRE_MISSING_HEAVY_SEVERITY``.
    The sampler never touches image data; all draws use the caller supplied generator.
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
