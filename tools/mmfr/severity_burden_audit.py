#!/usr/bin/env python3
"""CPU-only, machine-checkable audit of the ``MMFR-A1-corruption-basis-v2`` chain.

The audited chain is::

    severity -> corruption parameter -> realized damage -> burden -> Depth target

for the six failure kinds implemented in ``utils/dataloader/multimodal_failure_v2.py``.

What this tool decides
----------------------
1. ``no_double_severity_encoding``: for the graded kinds (``gaussian_noise``, ``blur``,
   ``quantization``) the per-pixel burden equals the realized normalized damage
   ``clip(max_channel_abs_delta / (255 * DAMAGE_REFERENCE), 0, 1)`` recomputed here from
   the fixture and the returned corrupted array -- with *no* second multiplication by
   ``severity``.  For ``misalignment`` the in-bounds burden must equal
   ``min(1, displacement_px / reference_displacement_px)`` and out-of-bounds pixels must
   carry ``MISSING_BURDEN``.  Every corruption parameter must also equal the closed-form
   function of ``severity`` given in the module docstring.
2. ``monotonic_burden`` / ``monotonic_target``: per kind, the burden mean is monotone
   non-decreasing and the Depth reliability mean is monotone non-increasing in severity.
3. ``relative_scale_consistent``: ``blur`` and ``misalignment`` severities are fractions
   of the actual grid, so the same severity means the same *relative* corruption on a
   ``480 x 640`` training crop and on a ``932 x 1082`` aligned grid.

Method notes
------------
* Exactly one spec is applied per run, and a fresh RNG is rebuilt for every
  ``(kind, severity)`` run from ``FIXED_SEED`` (declared below), so severities are
  directly comparable: v2 draws the *unit* deviate once and scales it by the
  severity-driven parameter.
* The expected burden array is recomputed here, independently of the module, from the
  returned corrupted array plus the fixture.  The module returns no burden array, but for
  a single spec ``reliability[m] == float32(exp(-burden))``; the check therefore
  reconstructs the target from the independently computed burden and requires a
  *bitwise* match, which is exactness -- not a tolerance-based proxy.  A numeric
  round-trip readout (``-log(reliability)`` vs the recomputed burden) is reported as well
  and is only limited by float32 log rounding.
* Fixtures are deterministic synthetic uint8 images generated from recorded closed-form
  rules (no RNG), and deliberately contain no 0 and no 255 pixel values, so structural
  zeroing and out-of-bounds regions can be recovered from the data alone.
* This tool never writes to, or imports state from, the audited module beyond its public
  API.  It performs no GPU, training, evaluation or network work.

Exit code is non-zero if any check fails; violations are listed on stdout and in the JSON
report under ``conclusions.violations``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloader.multimodal_failure_v2 import (  # noqa: E402
    BLUR_SIGMA_FRACTION,
    DAMAGE_REFERENCE,
    DROPOUT_GRID_TARGET,
    MISALIGN_MAX_SHIFT_FRACTION,
    MIN_DROPOUT_CELL,
    MISSING_BURDEN,
    NOISE_SIGMA_MAX,
    PROTOCOL_ID,
    SEVERITY_ENCODING,
    FailureSpec,
    apply_failures,
    misalignment_reference_displacement,
)
import utils.dataloader.multimodal_failure_v2 as mmfr_module  # noqa: E402

# --------------------------------------------------------------------------------------
# Audit configuration (all values that affect the result are declared here and recorded)
# --------------------------------------------------------------------------------------

#: Every ``(kind, severity)`` run rebuilds its RNG from this seed.
FIXED_SEED = 20260914

#: Requirement: at least ``0.05, 0.25, 0.50, 0.75, 1.0``; the extra points are denser
#: monotonicity probes and cost nothing on CPU.
SEVERITY_GRID: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.30, 0.50, 0.60, 0.75, 0.90, 1.0)
REQUIRED_SEVERITIES: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 1.0)
SUMMARY_SEVERITIES: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)

GRADED_KINDS: Tuple[str, ...] = ("gaussian_noise", "blur", "quantization")
STRUCTURAL_KINDS: Tuple[str, ...] = ("entire_missing", "spatial_dropout")
DEPTH_CHAIN_KINDS: Tuple[str, ...] = (
    "entire_missing",
    "spatial_dropout",
    "gaussian_noise",
    "blur",
    "quantization",
    "misalignment",
)
RGB_SPOT_KINDS: Tuple[str, ...] = ("gaussian_noise", "blur")

#: Round-trip tolerance for ``-log(float32(exp(-b)))`` vs ``b``.  float32 log round-trip
#: noise is <= ~3e-7 for the burden range [0, 1]; a second severity multiplication would
#: shift the value by O(0.1-1.0), i.e. five to seven orders of magnitude more.
RECOVERY_ATOL = 1.0e-5
#: Agreement bound between the float32 mirrored damage and its float64 reference.  Pure
#: float32 rounding is ~6e-8 for damage in [0, 1]; a double severity encoding would
#: deviate by O(0.1-1.0).
FLOAT32_PRECISION_ATOL = 1.0e-6
#: Degenerate guarded comparison for values that are expected to be bit-identical.
EXACT_REL_TOL = 1.0e-12
EXACT_ABS_TOL = 1.0e-15

OUTPUT_DIR = REPO_ROOT / "outputs" / "mmfr-a1-v2-severity-audit"
REPORT_PATH = OUTPUT_DIR / "severity-burden-audit.json"

FIXTURE_SPECS: Tuple[Mapping[str, Any], ...] = (
    {
        "name": "train_crop_480x640",
        "height": 480,
        "width": 640,
        "role": "frozen training crop scale",
    },
    {
        "name": "aligned_grid_932x1082",
        "height": 932,
        "width": 1082,
        "role": "original aligned grid scale",
    },
)

DEPTH_FIXTURE_RULE = (
    "uint8 Depth = clip(rint(base + texture)) where u = y/(H-1), v = x/(W-1), "
    "base = 40 + 150*(0.55*u + 0.45*v), "
    "texture = 25*sin(2*pi*7*v)*cos(2*pi*5*u) + 10*sin(2*pi*23*v + 1.3)*sin(2*pi*19*u); "
    "smooth ramp plus two low-frequency texture terms, deterministic (no RNG), "
    "value range strictly inside (0, 255)"
)
RGB_FIXTURE_RULE = (
    "uint8 RGB = clip(rint(channel)) where u = y/(H-1), v = x/(W-1), "
    "base = 30 + 150*(0.5*u + 0.5*v), texture = 30*sin(2*pi*9*v + 0.4)*cos(2*pi*6*u), "
    "channel0 = base + texture, channel1 = 0.85*base + 20 + texture, "
    "channel2 = 200 - 0.7*base + texture; deterministic (no RNG), "
    "value range strictly inside (0, 255)"
)


def make_rng() -> np.random.Generator:
    """Fresh, reproducible generator for one audit run (equivalent to PCG64(SeedSequence))."""
    return np.random.default_rng(np.random.SeedSequence([FIXED_SEED]))


# --------------------------------------------------------------------------------------
# Deterministic fixtures
# --------------------------------------------------------------------------------------


def _grids(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    u = (np.arange(height, dtype=np.float64) / float(height - 1))[:, None]
    v = (np.arange(width, dtype=np.float64) / float(width - 1))[None, :]
    return u, v


def build_depth_fixture(height: int, width: int) -> np.ndarray:
    u, v = _grids(height, width)
    base = 40.0 + 150.0 * (0.55 * u + 0.45 * v)
    texture = 25.0 * np.sin(2.0 * np.pi * 7.0 * v) * np.cos(2.0 * np.pi * 5.0 * u) + 10.0 * np.sin(
        2.0 * np.pi * 23.0 * v + 1.3
    ) * np.sin(2.0 * np.pi * 19.0 * u)
    return np.clip(np.rint(base + texture), 0.0, 255.0).astype(np.uint8)


def build_rgb_fixture(height: int, width: int) -> np.ndarray:
    u, v = _grids(height, width)
    base = 30.0 + 150.0 * (0.5 * u + 0.5 * v)
    texture = 30.0 * np.sin(2.0 * np.pi * 9.0 * v + 0.4) * np.cos(2.0 * np.pi * 6.0 * u)
    channels = [base + texture, 0.85 * base + 20.0 + texture, 200.0 - 0.7 * base + texture]
    stacked = np.stack([np.clip(np.rint(channel), 0.0, 255.0) for channel in channels], axis=-1)
    return stacked.astype(np.uint8)


def build_fixture(spec: Mapping[str, Any]) -> Dict[str, Any]:
    height = int(spec["height"])
    width = int(spec["width"])
    depth = build_depth_fixture(height, width)
    rgb = build_rgb_fixture(height, width)
    for name, array in (("depth", depth), ("rgb", rgb)):
        if int(array.min()) <= 0 or int(array.max()) >= 255:
            raise RuntimeError(f"fixture {spec['name']} {name} must stay inside (0, 255)")
    return {
        "name": spec["name"],
        "height": height,
        "width": width,
        "role": spec["role"],
        "depth": depth,
        "rgb": rgb,
        "record": {
            "name": spec["name"],
            "height": height,
            "width": width,
            "role": spec["role"],
            "depth_generation_rule": DEPTH_FIXTURE_RULE,
            "rgb_generation_rule": RGB_FIXTURE_RULE,
            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
            "depth_dtype": str(depth.dtype),
            "depth_min": int(depth.min()),
            "depth_max": int(depth.max()),
            "has_zero_pixels": bool((depth == 0).any()),
            "has_255_pixels": bool((depth == 255).any()),
        },
    }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def close_exact(observed: Any, expected: Any) -> bool:
    if isinstance(observed, bool) or isinstance(expected, bool):
        return bool(observed) == bool(expected)
    if isinstance(observed, (int, np.integer)) and isinstance(expected, (int, np.integer)):
        return int(observed) == int(expected)
    try:
        return math.isclose(float(observed), float(expected), rel_tol=EXACT_REL_TOL, abs_tol=EXACT_ABS_TOL)
    except (TypeError, ValueError):
        return observed == expected


def parameter_brief(kind: str, parameter: Mapping[str, Any]) -> str:
    if kind == "gaussian_noise":
        return f"sigma={parameter['sigma_uint8']:.3f}u8"
    if kind == "blur":
        return f"sigma={parameter['sigma_px']:.4f}px k={parameter['ksize']} frac={parameter['sigma_fraction_of_min_side']:.6f}"
    if kind == "quantization":
        return f"levels={parameter['levels']} step={parameter['step_uint8']:.4f}"
    if kind == "misalignment":
        return (
            f"dy={parameter['dy_px']} dx={parameter['dx_px']} "
            f"disp={parameter['displacement_px']:.3f} ref={parameter['reference_displacement_px']:.3f}"
        )
    if kind == "spatial_dropout":
        return f"cell={parameter['cell_size']} cells={parameter['dropped_cells']}/{parameter['grid_cells']}"
    if kind == "entire_missing":
        return "whole-modality zeroing"
    return str(dict(parameter))


def expected_corruption_parameter(
    kind: str, modality: str, severity: float, height: int, width: int
) -> Dict[str, Any]:
    """Closed-form corruption parameter for ``severity``, recomputed from the docstring.

    Nothing here reads the module's metadata; the returned mapping is compared against it.
    """
    severity = float(severity)
    if kind == "gaussian_noise":
        return {"sigma_uint8": severity * NOISE_SIGMA_MAX}
    if kind == "blur":
        reference = float(min(height, width))
        sigma = severity * BLUR_SIGMA_FRACTION * reference
        return {
            "sigma_px": sigma,
            "sigma_fraction_of_min_side": BLUR_SIGMA_FRACTION * severity,
            "reference_min_side_px": reference,
            "ksize": max(3, 2 * int(math.ceil(2.0 * sigma)) + 1),
        }
    if kind == "quantization":
        levels = int(np.clip(round(256.0 * (1.0 - severity)), 2, 256))
        return {"levels": levels, "step_uint8": 255.0 / float(max(1, levels - 1))}
    if kind == "misalignment":
        rng = make_rng()
        unit_y = float(rng.random()) * 2.0 - 1.0
        unit_x = float(rng.random()) * 2.0 - 1.0
        max_shift_y = severity * MISALIGN_MAX_SHIFT_FRACTION * float(height)
        max_shift_x = severity * MISALIGN_MAX_SHIFT_FRACTION * float(width)
        dy = int(round(unit_y * max_shift_y))
        dx = int(round(unit_x * max_shift_x))
        if dy == 0 and dx == 0:
            if max_shift_x >= max_shift_y and width > 1:
                dx = 1
            else:
                dy = 1
        dy = int(np.clip(dy, -(height - 1), height - 1))
        dx = int(np.clip(dx, -(width - 1), width - 1))
        return {
            "dy_px": dy,
            "dx_px": dx,
            "max_shift_y_px": max_shift_y,
            "max_shift_x_px": max_shift_x,
            "shift_fraction_of_axis": MISALIGN_MAX_SHIFT_FRACTION * severity,
            "displacement_px": math.hypot(float(dy), float(dx)),
            "reference_displacement_px": misalignment_reference_displacement(height, width),
        }
    if kind == "spatial_dropout":
        cell = max(MIN_DROPOUT_CELL, min(height, width) // DROPOUT_GRID_TARGET)
        grid_h = -(-height // cell)
        grid_w = -(-width // cell)
        total = grid_h * grid_w
        count = min(total, max(1, int(round(severity * total))))
        return {
            "cell_size": cell,
            "grid_shape": (grid_h, grid_w),
            "dropped_cells": count,
            "grid_cells": total,
        }
    if kind == "entire_missing":
        return {"missing_fraction": 1.0}
    raise ValueError(f"unsupported kind {kind!r}")


# --------------------------------------------------------------------------------------
# One audit run
# --------------------------------------------------------------------------------------


def run_job(fixture: Mapping[str, Any], kind: str, modality: str, severity: float, section: str) -> Dict[str, Any]:
    depth: np.ndarray = fixture["depth"]
    rgb: np.ndarray = fixture["rgb"]
    height, width = depth.shape[:2]
    severity = float(severity)

    rng = make_rng()
    result = apply_failures(
        rgb,
        depth,
        [FailureSpec(modality=modality, kind=kind, severity=severity)],
        rng,
    )
    module_record = dict(result.metadata["specs"][0])
    parameter = dict(module_record["corruption_parameter"])

    channel_index = 1 if modality == "depth" else 0
    target = result.reliability[channel_index]
    other_channel = result.reliability[1 - channel_index]
    original = depth if modality == "depth" else rgb
    corrupted = result.depth if modality == "depth" else result.rgb

    # ---- independently recomputed realized damage ------------------------------------
    delta = corrupted.astype(np.float32) - original.astype(np.float32)
    channel_abs_delta = np.max(np.abs(delta), axis=2) if delta.ndim == 3 else np.abs(delta)
    # The module's _normalized_damage divides a float32 array by a python float, which
    # under NEP 50 stays float32.  Mirror that precision for the exactness check and keep
    # a float64 variant as the higher-precision reference (their difference is pure
    # float32 rounding and is bounded by FLOAT32_PRECISION_ATOL).
    damage_scalar = np.float32(255.0 * float(DAMAGE_REFERENCE))
    damage = np.clip(channel_abs_delta / damage_scalar, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    damage_float64 = np.clip(
        channel_abs_delta.astype(np.float64) / (255.0 * float(DAMAGE_REFERENCE)), 0.0, 1.0
    )
    damage_precision_diff = float(np.max(np.abs(damage.astype(np.float64) - damage_float64)))

    # ---- independently reconstructed burden ------------------------------------------
    if kind in GRADED_KINDS:
        expected_burden = damage  # float64, identical expression to _normalized_damage
        burden_class = "graded"
        valid_from_data = np.ones(damage.shape, dtype=bool)
    elif kind == "misalignment":
        expected_parameter = expected_corruption_parameter(kind, modality, severity, height, width)
        displacement = float(expected_parameter["displacement_px"])
        reference = float(expected_parameter["reference_displacement_px"])
        in_bounds = float(min(1.0, displacement / reference)) if reference > 0.0 else 0.0
        valid_from_data = corrupted != 0
        expected_burden = np.full((height, width), np.float32(MISSING_BURDEN), dtype=np.float32)
        expected_burden[valid_from_data] = np.float32(in_bounds)
        burden_class = "misalignment"
    else:
        valid_from_data = corrupted != 0
        expected_burden = np.zeros((height, width), dtype=np.float32)
        expected_burden[~valid_from_data] = np.float32(MISSING_BURDEN)
        burden_class = "structural"

    missing_expected = expected_burden >= np.float32(MISSING_BURDEN)
    present_expected = ~missing_expected

    # ---- exact target reconstruction (the exactness evidence) -------------------------
    reconstructed_target = np.exp(-expected_burden).astype(np.float32)
    target_bitwise_equal = bool(np.array_equal(reconstructed_target, target))
    target_max_abs_diff = float(np.max(np.abs(reconstructed_target.astype(np.float64) - target.astype(np.float64))))

    # ---- float32 round-trip readout ---------------------------------------------------
    missing_observed = target == 0.0
    missing_mask_agrees = bool(np.array_equal(missing_expected, missing_observed))
    if present_expected.any():
        recovered = -np.log(target[present_expected].astype(np.float64))
        roundtrip_diff = float(
            np.max(np.abs(recovered - expected_burden[present_expected].astype(np.float64)))
        )
    else:
        roundtrip_diff = 0.0

    # ---- module metadata agreement ----------------------------------------------------
    my_damage = {
        "realized_damage_mean": float(damage.mean()),
        "realized_damage_max": float(damage.max()),
        "realized_damage_nonzero_fraction": float(np.count_nonzero(damage) / float(damage.size)),
    }
    my_burden = summarize(expected_burden)
    metadata_checks = {}
    metadata_missing_keys: List[str] = []
    for key, mine in my_damage.items():
        if key in module_record:
            metadata_checks[key] = {
                "module": float(module_record[key]),
                "independent": mine,
                "abs_diff": abs(float(module_record[key]) - mine),
            }
        else:
            metadata_missing_keys.append(key)
    metadata_checks["burden_mean"] = {
        "module": float(module_record["burden_mean"]),
        "independent": my_burden["mean"],
        "abs_diff": abs(float(module_record["burden_mean"]) - my_burden["mean"]),
    }
    metadata_checks["burden_min"] = {
        "module": float(module_record["burden_min"]),
        "independent": my_burden["min"],
        "abs_diff": abs(float(module_record["burden_min"]) - my_burden["min"]),
    }
    metadata_checks["burden_max"] = {
        "module": float(module_record["burden_max"]),
        "independent": my_burden["max"],
        "abs_diff": abs(float(module_record["burden_max"]) - my_burden["max"]),
    }
    metadata_checks["reliability_min"] = {
        "module": float(module_record["reliability_min"]),
        "independent": float(target.min()),
        "abs_diff": abs(float(module_record["reliability_min"]) - float(target.min())),
    }
    metadata_checks["reliability_mean"] = {
        "module": float(module_record["reliability_mean"]),
        "independent": float(target.mean()),
        "abs_diff": abs(float(module_record["reliability_mean"]) - float(target.mean())),
    }
    metadata_agreement = all(
        entry["abs_diff"] <= max(RECOVERY_ATOL, 1.0e-9 * max(1.0, abs(entry["module"])))
        for entry in metadata_checks.values()
    )
    # ---- corruption parameter formula check -------------------------------------------
    expected_parameter = expected_corruption_parameter(kind, modality, severity, height, width)
    parameter_mismatches: List[str] = []
    for key, expected_value in expected_parameter.items():
        if key not in parameter or not close_exact(parameter[key], expected_value):
            parameter_mismatches.append(
                f"{key}: module={parameter.get(key)!r} expected={expected_value!r}"
            )
    for key in parameter:
        if key not in expected_parameter:
            parameter_mismatches.append(f"unexpected corruption parameter key {key!r}")

    # ---- derived structural / misalignment geometry checks ---------------------------
    derived: Dict[str, Any] = {}
    if kind == "spatial_dropout":
        dropped_pixels_independent = int(np.count_nonzero(~valid_from_data))
        derived = {
            "dropped_pixels_module": int(module_record["dropped_pixels"]),
            "dropped_pixels_independent": dropped_pixels_independent,
            "dropped_pixels_agree": dropped_pixels_independent == int(module_record["dropped_pixels"]),
            "missing_fraction_module": float(module_record["missing_fraction"]),
            "missing_fraction_independent": float(dropped_pixels_independent) / float(height * width),
        }
        derived["missing_fraction_agree"] = math.isclose(
            derived["missing_fraction_module"],
            derived["missing_fraction_independent"],
            rel_tol=EXACT_REL_TOL,
            abs_tol=EXACT_ABS_TOL,
        )
    if kind == "entire_missing":
        derived = {
            "dropped_pixels_independent": int(np.count_nonzero(~valid_from_data)),
            "missing_fraction_module": float(module_record["missing_fraction"]),
            "all_pixels_missing": bool(not valid_from_data.any()),
        }
    if kind == "misalignment":
        expected_parameter = expected_corruption_parameter(kind, modality, severity, height, width)
        geometry_invalid = np.ones((height, width), dtype=bool)
        dy = int(parameter["dy_px"])
        dx = int(parameter["dx_px"])
        dst_y = slice(max(0, dy), height - max(0, -dy))
        dst_x = slice(max(0, dx), width - max(0, -dx))
        geometry_invalid[dst_y, dst_x] = False
        derived = {
            "invalid_pixels_module": int(module_record["invalid_pixels"]),
            "invalid_pixels_from_data": int(np.count_nonzero(~valid_from_data)),
            "invalid_pixels_from_geometry": int(np.count_nonzero(geometry_invalid)),
            "data_mask_equals_geometry_mask": bool(np.array_equal(~valid_from_data, geometry_invalid)),
            "in_bounds_burden_float64": float(
                min(
                    1.0,
                    float(expected_parameter["displacement_px"])
                    / float(expected_parameter["reference_displacement_px"]),
                )
            ),
            "in_bounds_burden_float32": float(
                np.float32(
                    min(
                        1.0,
                        float(expected_parameter["displacement_px"])
                        / float(expected_parameter["reference_displacement_px"]),
                    )
                )
            ),
            "module_normalized_burden": float(module_record["normalized_burden"]),
        }
        derived["in_bounds_burden_agrees"] = close_exact(
            derived["in_bounds_burden_float64"], derived["module_normalized_burden"]
        )

    # ---- target statistics ------------------------------------------------------------
    target_unique = sorted({float(value) for value in np.unique(target)})
    record: Dict[str, Any] = {
        "section": section,
        "fixture": fixture["name"],
        "shape": [height, width],
        "modality": modality,
        "kind": kind,
        "severity": severity,
        "burden_class": burden_class,
        "corruption_parameter": parameter,
        "expected_corruption_parameter": to_jsonable(expected_parameter),
        "corruption_parameter_formula_match": not parameter_mismatches,
        "corruption_parameter_mismatches": parameter_mismatches,
        "realized_damage": my_damage,
        "realized_damage_float64_mean": float(damage_float64.mean()),
        "realized_damage_float32_vs_float64_max_abs_diff": damage_precision_diff,
        "burden_all_pixels": my_burden,
        "burden_non_missing_pixels": (
            summarize(expected_burden[present_expected]) if present_expected.any() else None
        ),
        "burden_value_set": sorted({float(value) for value in np.unique(expected_burden)}),
        "target_depth": {"min": float(result.reliability[1].min()), "mean": float(result.reliability[1].mean())},
        "target_rgb": {"min": float(result.reliability[0].min()), "mean": float(result.reliability[0].mean())},
        "target_applied_channel": {"min": float(target.min()), "mean": float(target.mean())},
        "target_other_channel": {
            "min": float(other_channel.min()),
            "mean": float(other_channel.mean()),
        },
        "target_unique_values": target_unique,
        "target_bitwise_reconstruction_equal": target_bitwise_equal,
        "target_reconstruction_max_abs_diff": target_max_abs_diff,
        "burden_roundtrip_max_abs_diff": roundtrip_diff,
        "missing_mask_agrees": missing_mask_agrees,
        "metadata_checks": metadata_checks,
        "module_metadata_missing_keys": metadata_missing_keys,
        "metadata_agreement": bool(metadata_agreement),
        "derived": derived,
        "module_record": to_jsonable(module_record),
    }
    return record


# --------------------------------------------------------------------------------------
# Aggregate checks
# --------------------------------------------------------------------------------------


def check_graded_burden(records: Sequence[Mapping[str, Any]], violations: List[str]) -> Dict[str, Any]:
    graded = [record for record in records if record["kind"] in GRADED_KINDS]
    worst_roundtrip = 0.0
    worst_target_diff = 0.0
    worst_precision = 0.0
    failures: List[str] = []
    for record in graded:
        label = f"{record['fixture']}/{record['modality']}/{record['kind']}@{record['severity']}"
        worst_roundtrip = max(worst_roundtrip, record["burden_roundtrip_max_abs_diff"])
        worst_target_diff = max(worst_target_diff, record["target_reconstruction_max_abs_diff"])
        worst_precision = max(worst_precision, record["realized_damage_float32_vs_float64_max_abs_diff"])
        if not record["target_bitwise_reconstruction_equal"]:
            failures.append(f"{label}: target is not the bitwise exp(-independent_burden)")
        if record["burden_roundtrip_max_abs_diff"] > RECOVERY_ATOL:
            failures.append(
                f"{label}: |(-log target) - independent burden| = {record['burden_roundtrip_max_abs_diff']:.3e}"
            )
        if record["realized_damage_float32_vs_float64_max_abs_diff"] > FLOAT32_PRECISION_ATOL:
            failures.append(
                f"{label}: float32 mirrored damage deviates from the float64 reference by "
                f"{record['realized_damage_float32_vs_float64_max_abs_diff']:.3e}"
            )
        if not record["missing_mask_agrees"]:
            failures.append(f"{label}: missing-pixel mask disagrees with independent burden support")
        if not record["metadata_agreement"]:
            failures.append(f"{label}: module metadata disagrees with independent recomputation")
    violations.extend(failures)
    return {
        "pass": not failures and bool(graded),
        "scope": "single-spec runs of gaussian_noise / blur / quantization (depth chain and rgb spot check)",
        "runs_checked": len(graded),
        "requirement": "per-pixel burden == clip(max_channel_abs_delta / (255 * DAMAGE_REFERENCE), 0, 1), error 0",
        "exactness_evidence": (
            "float32(exp(-independent_burden)) reproduces the module's reliability target bitwise, "
            "so the burden is exactly the realized damage recomputed here -- no residual factor of severity"
        ),
        "target_reconstruction_is_bitexact_for_all_runs": all(
            record["target_bitwise_reconstruction_equal"] for record in graded
        ),
        "max_target_reconstruction_abs_diff": worst_target_diff,
        "max_float32_log_roundtrip_abs_diff": worst_roundtrip,
        "roundtrip_tolerance": RECOVERY_ATOL,
        "max_float32_vs_float64_damage_abs_diff": worst_precision,
        "float32_precision_tolerance": FLOAT32_PRECISION_ATOL,
        "failures": failures,
    }


def check_misalignment(records: Sequence[Mapping[str, Any]], violations: List[str]) -> Dict[str, Any]:
    runs = [record for record in records if record["kind"] == "misalignment"]
    failures: List[str] = []
    max_seam_error = 0.0
    max_seam_deviation = 0.0
    for record in runs:
        label = f"{record['fixture']}/{record['modality']}/misalignment@{record['severity']}"
        derived = record["derived"]
        if not record["target_bitwise_reconstruction_equal"]:
            failures.append(f"{label}: target is not the bitwise exp(-expected burden array)")
        if not derived["in_bounds_burden_agrees"]:
            failures.append(
                f"{label}: in-bounds burden {derived['in_bounds_burden_float64']!r} != module "
                f"{derived['module_normalized_burden']!r}"
            )
        if not derived["data_mask_equals_geometry_mask"]:
            failures.append(f"{label}: zeroed region does not match the dx/dy geometry")
        if derived["invalid_pixels_module"] != derived["invalid_pixels_from_data"]:
            failures.append(f"{label}: invalid pixel count mismatch")
        if record["burden_roundtrip_max_abs_diff"] > RECOVERY_ATOL:
            failures.append(
                f"{label}: in-bounds burden is not exp(-b) of the target "
                f"(diff {record['burden_roundtrip_max_abs_diff']:.3e})"
            )
        max_seam_error = max(max_seam_error, record["burden_roundtrip_max_abs_diff"])
        expected = derived["in_bounds_burden_float64"]
        max_seam_deviation = max(
            max_seam_deviation,
            abs(float(record["burden_non_missing_pixels"]["mean"]) - expected)
            if record["burden_non_missing_pixels"]
            else 0.0,
        )
    violations.extend(failures)
    return {
        "pass": not failures and bool(runs),
        "runs_checked": len(runs),
        "requirement": "in-bounds burden == min(1, displacement_px/reference_displacement_px); out-of-bounds == MISSING_BURDEN",
        "max_in_bounds_roundtrip_abs_diff": max_seam_error,
        "max_in_bounds_mean_deviation": max_seam_deviation,
        "failures": failures,
    }


def check_monotonicity(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    direction: str,
    key: str,
    violations: List[str],
    scope: str = "depth_chain",
) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for record in records:
        if record["section"] != scope:
            continue
        grouped.setdefault((record["fixture"], record["kind"]), []).append(record)
    failures: List[str] = []
    observations: List[Dict[str, Any]] = []
    skipped: List[str] = []
    checked_series = 0
    for (fixture, kind), group in sorted(grouped.items()):
        series = sorted(group, key=lambda item: item["severity"])
        if any(item[key] is None for item in series):
            skipped.append(f"{fixture}/{kind} (empty non-missing pixel set)")
            continue
        values = [float(item[key]) for item in series]
        severities = [float(item["severity"]) for item in series]
        checked_series += 1
        observations.append(
            {"fixture": fixture, "kind": kind, "severities": severities, "values": values}
        )
        for previous, current, value_previous, value_current in zip(
            severities, severities[1:], values, values[1:]
        ):
            if direction == "non_decreasing" and value_current < value_previous - 1.0e-12:
                failures.append(
                    f"{fixture}/{kind} ({metric}): severity {previous} -> {current} "
                    f"gives {value_previous:.6f} -> {value_current:.6f} (decrease)"
                )
            if direction == "non_increasing" and value_current > value_previous + 1.0e-12:
                failures.append(
                    f"{fixture}/{kind} ({metric}): severity {previous} -> {current} "
                    f"gives {value_previous:.6f} -> {value_current:.6f} (increase)"
                )
    violations.extend(failures)
    return {
        "pass": not failures,
        "metric": metric,
        "direction": direction,
        "series_checked": checked_series,
        "series_skipped": skipped,
        "observations": observations,
        "failures": failures,
    }


def check_dropout_extent(records: Sequence[Mapping[str, Any]], violations: List[str]) -> Dict[str, Any]:
    failures: List[str] = []
    observations: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        if record["kind"] == "spatial_dropout" and record["section"] == "depth_chain":
            grouped.setdefault(record["fixture"], []).append(record)
    for fixture, group in sorted(grouped.items()):
        series = sorted(group, key=lambda item: item["severity"])
        severities = [float(item["severity"]) for item in series]
        cells = [int(item["corruption_parameter"]["dropped_cells"]) for item in series]
        pixels = [int(item["derived"]["dropped_pixels_independent"]) for item in series]
        observations.append(
            {"fixture": fixture, "severities": severities, "dropped_cells": cells, "dropped_pixels": pixels}
        )
        for index in range(1, len(series)):
            if cells[index] < cells[index - 1]:
                failures.append(
                    f"{fixture}/spatial_dropout: dropped_cells decreased "
                    f"{cells[index - 1]} -> {cells[index]} at severity {severities[index]}"
                )
            if pixels[index] < pixels[index - 1]:
                failures.append(
                    f"{fixture}/spatial_dropout: dropped_pixels decreased "
                    f"{pixels[index - 1]} -> {pixels[index]} at severity {severities[index]}"
                )
    violations.extend(failures)
    return {
        "pass": not failures,
        "observations": observations,
        "failures": failures,
    }


def check_structural_burden(records: Sequence[Mapping[str, Any]], violations: List[str]) -> Dict[str, Any]:
    runs = [record for record in records if record["kind"] in STRUCTURAL_KINDS]
    failures: List[str] = []
    allowed = {0.0, float(MISSING_BURDEN)}
    for record in runs:
        label = f"{record['fixture']}/{record['kind']}@{record['severity']}"
        observed = {float(value) for value in record["target_unique_values"]}
        if not observed.issubset({0.0, 1.0}):
            failures.append(f"{label}: reliability target takes values outside {{0, 1}}: {sorted(observed)[:5]}")
        if not {float(value) for value in record["burden_value_set"]}.issubset(allowed):
            failures.append(f"{label}: burden value set {record['burden_value_set']} is not {{0, MISSING_BURDEN}}")
        if not record["target_bitwise_reconstruction_equal"]:
            failures.append(f"{label}: target is not the bitwise exp(-expected burden)")
    violations.extend(failures)
    return {
        "pass": not failures and bool(runs),
        "runs_checked": len(runs),
        "allowed_values": sorted(allowed),
        "failures": failures,
    }


def check_empty_spec(fixtures: Mapping[str, Mapping[str, Any]], violations: List[str]) -> Dict[str, Any]:
    observations: List[Dict[str, Any]] = []
    failures: List[str] = []
    for name, fixture in fixtures.items():
        depth = fixture["depth"]
        rgb = fixture["rgb"]
        height, width = depth.shape[:2]
        result = apply_failures(rgb, depth, (), make_rng())
        observation = {
            "fixture": name,
            "rgb_elementwise_equal": bool(np.array_equal(result.rgb, rgb)),
            "depth_elementwise_equal": bool(np.array_equal(result.depth, depth)),
            "returns_same_rgb_object": bool(result.rgb is rgb),
            "returns_same_depth_object": bool(result.depth is depth),
            "rgb_shape_dtype_unchanged": bool(result.rgb.shape == rgb.shape and result.rgb.dtype == rgb.dtype),
            "depth_shape_dtype_unchanged": bool(result.depth.shape == depth.shape and result.depth.dtype == depth.dtype),
            "target_shape": list(result.reliability.shape),
            "target_dtype": str(result.reliability.dtype),
            "target_all_ones": bool(np.all(result.reliability == 1.0)),
            "target_min": float(result.reliability.min()),
            "target_max": float(result.reliability.max()),
            "metadata_empty": bool(dict(result.metadata) == {}),
        }
        observations.append(observation)
        for key in (
            "rgb_elementwise_equal",
            "depth_elementwise_equal",
            "rgb_shape_dtype_unchanged",
            "depth_shape_dtype_unchanged",
            "target_all_ones",
            "metadata_empty",
        ):
            if not observation[key]:
                failures.append(f"{name}: empty spec failed {key}")
        if observation["target_shape"] != [len(("rgb", "depth")), height, width]:
            failures.append(f"{name}: empty spec target shape {observation['target_shape']}")
    violations.extend(failures)
    return {"pass": not failures, "observations": observations, "failures": failures}


def build_cross_resolution(
    fixtures: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    violations: List[str],
) -> Dict[str, Any]:
    by_key = {
        (record["kind"], record["fixture"], record["severity"]): record
        for record in records
        if record["section"] == "depth_chain"
    }
    blur_rows: List[Dict[str, Any]] = []
    alignment_rows: List[Dict[str, Any]] = []
    blur_failures: List[str] = []
    alignment_failures: List[str] = []

    for severity in SEVERITY_GRID:
        for name, fixture in fixtures.items():
            height, width = fixture["height"], fixture["width"]
            blur = by_key[("blur", name, severity)]["corruption_parameter"]
            ratio = float(blur["sigma_px"]) / float(min(height, width))
            expected_ratio = BLUR_SIGMA_FRACTION * float(severity)
            blur_rows.append(
                {
                    "severity": severity,
                    "fixture": name,
                    "sigma_px": float(blur["sigma_px"]),
                    "ksize": int(blur["ksize"]),
                    "sigma_px_over_min_side": ratio,
                    "expected_fraction": expected_ratio,
                    "abs_error": abs(ratio - expected_ratio),
                    "reference_min_side_px": float(min(height, width)),
                }
            )
            if abs(ratio - expected_ratio) > 1.0e-12:
                blur_failures.append(
                    f"blur {name}@{severity}: sigma/min(H,W)={ratio} != {expected_ratio}"
                )

            misalign = by_key[("misalignment", name, severity)]["corruption_parameter"]
            reference = float(misalign["reference_displacement_px"])
            permitted = math.hypot(float(misalign["max_shift_y_px"]), float(misalign["max_shift_x_px"]))
            alignment_rows.append(
                {
                    "severity": severity,
                    "fixture": name,
                    "max_shift_y_px": float(misalign["max_shift_y_px"]),
                    "max_shift_x_px": float(misalign["max_shift_x_px"]),
                    "max_shift_y_px_over_H": float(misalign["max_shift_y_px"]) / float(height),
                    "max_shift_x_px_over_W": float(misalign["max_shift_x_px"]) / float(width),
                    "shift_fraction_of_axis": float(misalign["shift_fraction_of_axis"]),
                    "permitted_displacement_over_reference": permitted / reference,
                    "displacement_px": float(misalign["displacement_px"]),
                    "displacement_px_over_reference": float(misalign["displacement_px"]) / reference,
                    "displacement_px_over_min_side": float(misalign["displacement_px"]) / float(min(height, width)),
                    "reference_displacement_px": reference,
                }
            )
            expected_fraction = MISALIGN_MAX_SHIFT_FRACTION * float(severity)
            for key, value in (
                ("max_shift_y_px_over_H", float(misalign["max_shift_y_px"]) / float(height)),
                ("max_shift_x_px_over_W", float(misalign["max_shift_x_px"]) / float(width)),
                ("shift_fraction_of_axis", float(misalign["shift_fraction_of_axis"])),
            ):
                if abs(value - expected_fraction) > 1.0e-12:
                    alignment_failures.append(f"misalignment {name}@{severity}: {key}={value} != {expected_fraction}")
            if abs(permitted / reference - float(severity)) > 1.0e-12:
                alignment_failures.append(
                    f"misalignment {name}@{severity}: permitted/reference={permitted / reference} != severity"
                )

    def spread(key: str) -> Dict[str, float]:
        by_severity: Dict[float, List[float]] = {}
        for row in alignment_rows:
            by_severity.setdefault(row["severity"], []).append(row[key])
        worst = 0.0
        for _severity, group in by_severity.items():
            finite = [value for value in group if math.isfinite(value)]
            if len(finite) == 2 and max(finite) > 0:
                worst = max(worst, (max(finite) - min(finite)) / max(finite))
        return {"max_relative_spread_across_fixtures": worst}

    violations.extend(blur_failures)
    violations.extend(alignment_failures)
    return {
        "pass": not blur_failures and not alignment_failures,
        "blur": {
            "pass": not blur_failures,
            "metric": "sigma_px / min(H, W)",
            "rows": blur_rows,
            "max_abs_error_vs_severity_times_fraction": max(row["abs_error"] for row in blur_rows),
            "failures": blur_failures,
        },
        "misalignment": {
            "pass": not alignment_failures,
            "metric": "per-axis permitted shift / axis length, and permitted displacement / reference_displacement_px",
            "rows": alignment_rows,
            "realized_ratio_spread": spread("displacement_px_over_reference"),
            "note": (
                "The permitted displacement fractions are resolution-invariant and equal severity "
                "on both fixtures, while the absolute pixel displacement scales with the grid; the "
                "small spread of the realized displacement/reference ratio is integer pixel rounding "
                "of dy/dx (the unit draws are identical because the RNG is rebuilt from FIXED_SEED)."
            ),
            "failures": alignment_failures,
        },
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def print_depth_table(fixture_name: str, records: Sequence[Mapping[str, Any]]) -> None:
    print()
    print(f"[depth chain] fixture={fixture_name}")
    header = (
        f"{'kind':<16}{'sev':>5}  {'corruption parameter':<52}"
        f"{'dmg_mean':>10}{'burden_mean':>13}{'tgt_depth_mean':>16}{'bitexact':>9}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        if record["fixture"] != fixture_name or record["section"] != "depth_chain":
            continue
        if float(record["severity"]) not in REQUIRED_SEVERITIES:
            continue
        print(
            f"{record['kind']:<16}{record['severity']:>5.2f}  "
            f"{parameter_brief(record['kind'], record['corruption_parameter']):<52}"
            f"{record['realized_damage']['realized_damage_mean']:>10.6f}"
            f"{record['burden_all_pixels']['mean']:>13.6f}"
            f"{record['target_depth']['mean']:>16.6f}"
            f"{str(record['target_bitwise_reconstruction_equal']):>9}"
        )


def print_summary(fixtures: Mapping[str, Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> None:
    print()
    print("[required severities] fixture=train_crop_480x640")
    for kind in DEPTH_CHAIN_KINDS:
        for record in records:
            if (
                record["fixture"] != "train_crop_480x640"
                or record["section"] != "depth_chain"
                or record["kind"] != kind
                or float(record["severity"]) not in SUMMARY_SEVERITIES
            ):
                continue
            print(
                f"  {kind:<16} sev={record['severity']:.2f}  "
                f"{parameter_brief(kind, record['corruption_parameter']):<50} "
                f"damage_mean={record['realized_damage']['realized_damage_mean']:.6f} "
                f"burden_mean={record['burden_all_pixels']['mean']:.6f} "
                f"target_depth_mean={record['target_depth']['mean']:.6f}"
            )


def print_cross_resolution(cross: Mapping[str, Any]) -> None:
    print()
    print("[cross-resolution] blur sigma_px / min(H, W) must equal severity * BLUR_SIGMA_FRACTION")
    header = f"{'severity':>9}{'fixture':>20}{'sigma_px':>11}{'ksize':>7}{'sigma/min':>13}{'expected':>13}{'abs_err':>10}"
    print(header)
    print("-" * len(header))
    for row in cross["blur"]["rows"]:
        print(
            f"{row['severity']:>9.2f}{row['fixture']:>20}{row['sigma_px']:>11.4f}{row['ksize']:>7}"
            f"{row['sigma_px_over_min_side']:>13.8f}{row['expected_fraction']:>13.8f}{row['abs_error']:>10.2e}"
        )
    print()
    print("[cross-resolution] misalignment displacement, relative to grid and reference")
    header = (
        f"{'severity':>9}{'fixture':>20}{'max_shift_y/H':>15}{'max_shift_x/W':>15}"
        f"{'permitted/ref':>15}{'disp_px':>10}{'disp/ref':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in cross["misalignment"]["rows"]:
        print(
            f"{row['severity']:>9.2f}{row['fixture']:>20}{row['max_shift_y_px_over_H']:>15.8f}"
            f"{row['max_shift_x_px_over_W']:>15.8f}{row['permitted_displacement_over_reference']:>15.8f}"
            f"{row['displacement_px']:>10.3f}{row['displacement_px_over_reference']:>12.8f}"
        )


def print_checks(checks: Mapping[str, Mapping[str, Any]]) -> None:
    print()
    print("[assertions]")
    for name in sorted(checks):
        entry = checks[name]
        print(f"  {name}: {'PASS' if entry['pass'] else 'FAIL'}")
        for key in (
            "runs_checked",
            "series_checked",
            "scope",
            "requirement",
            "metric",
            "direction",
            "max_target_reconstruction_abs_diff",
            "max_float32_log_roundtrip_abs_diff",
            "max_float32_vs_float64_damage_abs_diff",
            "target_reconstruction_is_bitexact_for_all_runs",
            "max_in_bounds_roundtrip_abs_diff",
            "target_reconstruction_is_bitexact_for_all_runs",
            "allowed_values",
        ):
            if key in entry:
                print(f"      {key}: {entry[key]}")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> int:
    fixtures = {spec["name"]: build_fixture(spec) for spec in FIXTURE_SPECS}
    fixture_records = [fixture["record"] for fixture in fixtures.values()]

    records: List[Dict[str, Any]] = []
    for name in fixtures:
        for kind in DEPTH_CHAIN_KINDS:
            for severity in SEVERITY_GRID:
                records.append(run_job(fixtures[name], kind, "depth", severity, "depth_chain"))
    for name in ("train_crop_480x640",):
        for kind in RGB_SPOT_KINDS:
            for severity in SUMMARY_SEVERITIES:
                records.append(run_job(fixtures[name], kind, "rgb", severity, "rgb_spot_check"))

    violations: List[str] = []
    # Convenience flattened fields consumed by the aggregate checks.
    for record in records:
        record["burden_all_pixels_mean"] = record["burden_all_pixels"]["mean"]
        record["burden_non_missing_pixels_mean"] = (
            record["burden_non_missing_pixels"]["mean"] if record["burden_non_missing_pixels"] else None
        )
        record["target_depth_mean"] = record["target_depth"]["mean"]

    checks: Dict[str, Dict[str, Any]] = {}
    checks["a_graded_burden_equals_realized_damage"] = check_graded_burden(records, violations)
    checks["b_misalignment_burden_geometry"] = check_misalignment(records, violations)
    checks["c_burden_mean_monotone_nondecreasing"] = check_monotonicity(
        records, "burden mean (all pixels)", "non_decreasing", "burden_all_pixels_mean", violations
    )
    checks["c2_burden_mean_non_missing_monotone_nondecreasing"] = check_monotonicity(
        records,
        "burden mean (non-missing pixels)",
        "non_decreasing",
        "burden_non_missing_pixels_mean",
        violations,
    )
    checks["d_depth_target_mean_monotone_nonincreasing"] = check_monotonicity(
        records,
        "Depth target mean",
        "non_increasing",
        "target_depth_mean",
        violations,
    )
    checks["e_dropout_extent_monotone_nondecreasing"] = check_dropout_extent(records, violations)
    checks["f_structural_burden_is_binary"] = check_structural_burden(records, violations)
    checks["g_empty_spec_strict_noop"] = check_empty_spec(fixtures, violations)

    parameter_violations: List[str] = []
    for record in records:
        if not record["corruption_parameter_formula_match"]:
            parameter_violations.append(
                f"{record['fixture']}/{record['modality']}/{record['kind']}@{record['severity']}: "
                + "; ".join(record["corruption_parameter_mismatches"])
            )
    for record in records:
        derived = record["derived"]
        for key in ("dropped_pixels_agree", "missing_fraction_agree"):
            if key in derived and not derived[key]:
                parameter_violations.append(
                    f"{record['fixture']}/{record['kind']}@{record['severity']}: {key} is False"
                )
        if record["kind"] == "entire_missing" and not derived.get("all_pixels_missing", False):
            parameter_violations.append(
                f"{record['fixture']}/entire_missing@{record['severity']}: not all pixels are missing"
            )
    violations.extend(parameter_violations)
    checks["s_corruption_parameter_matches_closed_form"] = {
        "pass": not parameter_violations,
        "runs_checked": len(records),
        "requirement": "every corruption parameter equals the closed-form function of severity",
        "failures": parameter_violations,
    }

    cross = build_cross_resolution(fixtures, records, violations)

    conclusions = {
        "no_double_severity_encoding": bool(
            checks["a_graded_burden_equals_realized_damage"]["pass"]
            and checks["b_misalignment_burden_geometry"]["pass"]
            and checks["f_structural_burden_is_binary"]["pass"]
            and checks["s_corruption_parameter_matches_closed_form"]["pass"]
        ),
        "monotonic_burden": bool(checks["c_burden_mean_monotone_nondecreasing"]["pass"]),
        "monotonic_target": bool(checks["d_depth_target_mean_monotone_nonincreasing"]["pass"]),
        "relative_scale_consistent": bool(cross["pass"]),
        "dropout_extent_monotone": bool(checks["e_dropout_extent_monotone_nondecreasing"]["pass"]),
        "empty_spec_strict_noop": bool(checks["g_empty_spec_strict_noop"]["pass"]),
        "violations": violations,
        "violation_count": len(violations),
    }
    exit_code = 0 if not violations else 1

    report = {
        "audit": "MMFR-A1-corruption-basis-v2 severity -> corruption -> realized damage -> burden -> target",
        "protocol_id": PROTOCOL_ID,
        "module_severity_encoding_constant": SEVERITY_ENCODING,
        "audit_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "module_under_test": {
            "path": str(Path(mmfr_module.__file__).resolve()),
            "sha256": sha256_file(Path(mmfr_module.__file__).resolve()),
            "protocol_id": PROTOCOL_ID,
            "severity_encoding": SEVERITY_ENCODING,
        },
        "fixed_seed": FIXED_SEED,
        "rng_construction": "numpy.random.default_rng(numpy.random.SeedSequence([FIXED_SEED])), rebuilt for every (kind, severity) run",
        "recovery_tolerance": {
            "value": RECOVERY_ATOL,
            "rationale": (
                "the module returns no burden array, so burden is re-derived as -log(reliability) for a "
                "single spec; that float32 round trip is accurate to ~3e-7 for burden in [0, 1], whereas a "
                "second severity multiplication would deviate by O(0.1-1.0). Exactness itself is asserted "
                "separately by requiring float32(exp(-independent_burden)) == reliability bitwise."
            ),
        },
        "severity_grid": list(SEVERITY_GRID),
        "required_severities": list(REQUIRED_SEVERITIES),
        "summary_severities": list(SUMMARY_SEVERITIES),
        "fixtures": fixture_records,
        "constants": {
            "MISSING_BURDEN": MISSING_BURDEN,
            "DAMAGE_REFERENCE": DAMAGE_REFERENCE,
            "NOISE_SIGMA_MAX": NOISE_SIGMA_MAX,
            "BLUR_SIGMA_FRACTION": BLUR_SIGMA_FRACTION,
            "MISALIGN_MAX_SHIFT_FRACTION": MISALIGN_MAX_SHIFT_FRACTION,
            "MIN_DROPOUT_CELL": MIN_DROPOUT_CELL,
            "DROPOUT_GRID_TARGET": DROPOUT_GRID_TARGET,
        },
        "records": records,
        "cross_resolution": cross,
        "checks": checks,
        "conclusions": conclusions,
        "exit_code": exit_code,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    report_sha256 = sha256_file(REPORT_PATH)
    sidecar = REPORT_PATH.with_suffix(".json.sha256")
    sidecar.write_text(f"{report_sha256}  {REPORT_PATH.name}\n", encoding="utf-8")

    print("=" * 118)
    print("MMFR-A1-corruption-basis-v2 severity/burden audit (CPU only)")
    print("=" * 118)
    print(f"protocol_id       : {PROTOCOL_ID}   (module severity_encoding={SEVERITY_ENCODING!r})")
    print(f"FIXED_SEED        : {FIXED_SEED}   (fresh RNG per (kind, severity) run)")
    print(f"python            : {sys.version.split()[0]}   numpy {np.__version__}")
    print(f"module under test : {Path(mmfr_module.__file__).resolve()}")
    print(f"                    sha256 {sha256_file(Path(mmfr_module.__file__).resolve())}")
    print(f"severity grid     : {list(SEVERITY_GRID)}")
    for fixture in fixture_records:
        print(
            f"fixture           : {fixture['name']:<20} {fixture['height']}x{fixture['width']}"
            f"  depth range [{fixture['depth_min']}, {fixture['depth_max']}]"
            f"  sha256 {fixture['depth_sha256'][:16]}..."
        )

    for name in fixtures:
        print_depth_table(name, records)
    print_summary(fixtures, records)
    print_cross_resolution(cross)
    print_checks(checks)

    print()
    print("[conclusions]")
    for key in (
        "no_double_severity_encoding",
        "monotonic_burden",
        "monotonic_target",
        "relative_scale_consistent",
        "dropout_extent_monotone",
        "empty_spec_strict_noop",
    ):
        print(f"  {key}: {conclusions[key]}")
    print(f"  violation_count: {conclusions['violation_count']}")
    for violation in violations:
        print(f"    - {violation}")
    print()
    print(f"JSON report : {REPORT_PATH}")
    print(f"SHA-256     : {report_sha256}")
    print()
    print("OVERALL: " + ("PASS" if exit_code == 0 else "FAIL"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
