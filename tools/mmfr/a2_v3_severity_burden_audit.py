#!/usr/bin/env python3
"""CPU-only severity/burden audit for MMFR-A1 corruption basis v3.

The audit covers the six Depth failure kinds with one shared fixture per resolution:

    severity -> corruption parameter -> realized burden -> synthetic target

The v3 fixture deliberately contains native-invalid Depth pixels and an explicit
geometry-valid mask.  Intensity corruptions must leave the invalid state untouched;
spatial dropout and entire-missing must update it; MID-A misalignment must transport
Depth and validity with the same integer translation.  The audit never reads data,
checkpoints, the evaluator or the official test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloader import multimodal_failure_v3 as mmfr_module  # noqa: E402
from utils.dataloader.multimodal_failure_v3 import (  # noqa: E402
    BLUR_SIGMA_FRACTION,
    DAMAGE_REFERENCE,
    DROPOUT_GRID_TARGET,
    FAILURE_KINDS,
    MISALIGN_MAX_SHIFT_FRACTION,
    MIN_DROPOUT_CELL,
    MISSING_BURDEN,
    NOISE_SIGMA_MAX,
    SEVERITY_ENCODING,
    FailureSpec,
    apply_failures,
    misalignment_reference_displacement,
)

FIXED_SEED = 20260915
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
RECOVERY_ATOL = 1.0e-5
FLOAT32_PRECISION_ATOL = 1.0e-6
PARAMETER_ATOL = 1.0e-12
METADATA_ATOL = 2.0e-6

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "mmfr-a1-v3-severity-audit" / "severity-burden-audit.json"

DEPTH_FIXTURE_RULE = (
    "uint8 Depth = clip(rint(base + texture)); base = 40 + 150*(0.55*u + 0.45*v); "
    "texture = 25*sin(2*pi*7*v)*cos(2*pi*5*u) + 10*sin(2*pi*23*v + 1.3)*sin(2*pi*19*u); "
    "native-invalid pixels are set to 0 by a deterministic mask"
)
INVALID_FIXTURE_RULE = (
    "native_invalid = (((37*y + 19*x) mod 257) < 7) plus one deterministic interior rectangle; "
    "geometry_valid excludes the bottom and right border; depth_valid_pre=(depth>0)&geometry_valid"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_rng() -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([FIXED_SEED]))


def _grids(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    u = (np.arange(height, dtype=np.float64) / float(height - 1))[:, None]
    v = (np.arange(width, dtype=np.float64) / float(width - 1))[None, :]
    return u, v


def build_geometry_valid(height: int, width: int) -> np.ndarray:
    geometry = np.ones((height, width), dtype=bool)
    border_y = max(2, height // 32)
    border_x = max(2, width // 32)
    geometry[-border_y:, :] = False
    geometry[:, -border_x:] = False
    return geometry


def build_native_invalid(height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=np.int64)[:, None]
    cols = np.arange(width, dtype=np.int64)[None, :]
    invalid = ((37 * rows + 19 * cols) % 257) < 7
    y0, y1 = height // 3, height // 3 + max(4, height // 18)
    x0, x1 = width // 4, width // 4 + max(5, width // 16)
    invalid[y0:y1, x0:x1] = True
    return invalid


def build_depth_fixture(height: int, width: int) -> Dict[str, Any]:
    u, v = _grids(height, width)
    base = 40.0 + 150.0 * (0.55 * u + 0.45 * v)
    texture = 25.0 * np.sin(2.0 * np.pi * 7.0 * v) * np.cos(2.0 * np.pi * 5.0 * u) + 10.0 * np.sin(
        2.0 * np.pi * 23.0 * v + 1.3
    ) * np.sin(2.0 * np.pi * 19.0 * u)
    depth = np.clip(np.rint(base + texture), 1.0, 254.0).astype(np.uint8)
    geometry_valid = build_geometry_valid(height, width)
    native_invalid = build_native_invalid(height, width) & geometry_valid
    depth[~geometry_valid] = np.uint8(0)
    depth[native_invalid] = np.uint8(0)
    depth_valid_pre = (depth > 0) & geometry_valid
    if not depth_valid_pre.any() or int(depth[depth_valid_pre].min()) <= 0:
        raise RuntimeError("fixture must contain valid nonzero Depth pixels")
    return {
        "name": f"fixture_{height}x{width}",
        "height": height,
        "width": width,
        "depth": depth,
        "geometry_valid": geometry_valid,
        "native_invalid": native_invalid,
        "depth_valid_pre": depth_valid_pre,
        "record": {
            "name": f"fixture_{height}x{width}",
            "height": height,
            "width": width,
            "depth_generation_rule": DEPTH_FIXTURE_RULE,
            "invalid_state_rule": INVALID_FIXTURE_RULE,
            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
            "geometry_valid_sha256": hashlib.sha256(geometry_valid.tobytes()).hexdigest(),
            "depth_valid_pre_sha256": hashlib.sha256(depth_valid_pre.tobytes()).hexdigest(),
            "depth_min": int(depth.min()),
            "depth_max": int(depth.max()),
            "geometry_invalid_pixels": int(np.count_nonzero(~geometry_valid)),
            "native_invalid_pixels": int(np.count_nonzero(native_invalid)),
            "depth_valid_pre_pixels": int(np.count_nonzero(depth_valid_pre)),
            "depth_valid_pre_fraction": float(depth_valid_pre.mean()),
        },
    }


def build_fixtures() -> Dict[str, Dict[str, Any]]:
    return {
        "train_crop_480x640": build_depth_fixture(480, 640),
        "aligned_grid_932x1082": build_depth_fixture(932, 1082),
    }


def translate_state(state: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = state.shape[:2]
    src_y = slice(max(0, -dy), height - max(0, dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    output = np.zeros_like(state)
    output[dst_y, dst_x] = state[src_y, src_x]
    return output


def expected_dropout_mask(height: int, width: int, severity: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    cell = max(MIN_DROPOUT_CELL, min(height, width) // DROPOUT_GRID_TARGET)
    grid_h = -(-height // cell)
    grid_w = -(-width // cell)
    total = grid_h * grid_w
    count = min(total, max(1, int(round(float(severity) * total))))
    rng = make_rng()
    chosen = rng.choice(total, size=count, replace=False)
    cells = np.zeros(total, dtype=bool)
    cells[np.asarray(chosen, dtype=np.int64)] = True
    cells = cells.reshape(grid_h, grid_w)
    mask = np.repeat(np.repeat(cells, cell, axis=0), cell, axis=1)[:height, :width]
    return mask, {
        "cell_size": cell,
        "grid_shape": (grid_h, grid_w),
        "dropped_cells": count,
        "grid_cells": total,
    }


def expected_parameter(kind: str, severity: float, height: int, width: int) -> Dict[str, Any]:
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
            "mask_normalized": True,
            "epsilon": 1.0e-6,
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
        displacement = math.hypot(float(dy), float(dx))
        reference = misalignment_reference_displacement(height, width)
        return {
            "dy_px": dy,
            "dx_px": dx,
            "max_shift_y_px": max_shift_y,
            "max_shift_x_px": max_shift_x,
            "shift_fraction_of_axis": MISALIGN_MAX_SHIFT_FRACTION * severity,
            "displacement_px": displacement,
            "reference_displacement_px": reference,
            "transport": "integer-translate-depth-and-validity-together",
        }
    if kind == "spatial_dropout":
        _mask, parameter = expected_dropout_mask(height, width, severity)
        return parameter
    if kind == "entire_missing":
        return {"missing_fraction": 1.0}
    raise ValueError(f"unsupported failure kind {kind!r}")


def value_equal(observed: Any, expected: Any) -> bool:
    if isinstance(observed, (list, tuple)) or isinstance(expected, (list, tuple)):
        return list(observed) == list(expected)
    if isinstance(observed, bool) or isinstance(expected, bool):
        return bool(observed) == bool(expected)
    if isinstance(observed, (int, np.integer)) and isinstance(expected, (int, np.integer)):
        return int(observed) == int(expected)
    if isinstance(observed, str) or isinstance(expected, str):
        return str(observed) == str(expected)
    try:
        return math.isclose(float(observed), float(expected), rel_tol=PARAMETER_ATOL, abs_tol=PARAMETER_ATOL)
    except (TypeError, ValueError):
        return observed == expected


def stats(array: np.ndarray) -> Dict[str, float]:
    value = np.asarray(array, dtype=np.float64)
    return {"min": float(value.min()), "mean": float(value.mean()), "max": float(value.max())}


def expected_state_and_burden(
    fixture: Mapping[str, Any], kind: str, severity: float, parameter: Mapping[str, Any], damage: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    height, width = int(fixture["height"]), int(fixture["width"])
    initial = np.asarray(fixture["depth_valid_pre"], dtype=bool)
    geometry = np.asarray(fixture["geometry_valid"], dtype=bool)
    if kind in GRADED_KINDS:
        return initial, damage.astype(np.float32), None
    if kind == "spatial_dropout":
        mask, _ = expected_dropout_mask(height, width, severity)
        state = initial & ~mask & geometry
        burden = np.zeros((height, width), dtype=np.float32)
        burden[mask] = np.float32(MISSING_BURDEN)
        return state, burden, mask
    if kind == "entire_missing":
        return np.zeros((height, width), dtype=bool), np.full(
            (height, width), np.float32(MISSING_BURDEN), dtype=np.float32
        ), np.ones((height, width), dtype=bool)
    if kind == "misalignment":
        dy, dx = int(parameter["dy_px"]), int(parameter["dx_px"])
        state = translate_state(initial, dy, dx) & geometry
        invalid = np.ones((height, width), dtype=bool)
        dst_y = slice(max(0, dy), height - max(0, -dy))
        dst_x = slice(max(0, dx), width - max(0, -dx))
        invalid[dst_y, dst_x] = False
        displacement = float(parameter["displacement_px"])
        reference = float(parameter["reference_displacement_px"])
        in_bounds = min(1.0, displacement / reference) if reference > 0.0 else 0.0
        burden = np.full((height, width), np.float32(in_bounds), dtype=np.float32)
        burden[invalid] = np.float32(MISSING_BURDEN)
        return state, burden, invalid
    raise ValueError(kind)


def run_job(fixture: Mapping[str, Any], kind: str, severity: float) -> Dict[str, Any]:
    depth = np.asarray(fixture["depth"], dtype=np.uint8)
    height, width = depth.shape
    geometry = np.asarray(fixture["geometry_valid"], dtype=bool)
    initial = np.asarray(fixture["depth_valid_pre"], dtype=bool)
    result = apply_failures(
        depth[:, :, None].repeat(3, axis=2),
        depth,
        [FailureSpec(modality="depth", kind=kind, severity=float(severity))],
        make_rng(),
        depth_validity=initial,
        validity_mask=geometry,
    )
    module_record = dict(result.metadata["specs"][0])
    parameter = dict(module_record["corruption_parameter"])
    corrupted = result.depth if result.depth.ndim == 2 else result.depth[:, :, 0]
    delta = corrupted.astype(np.float32) - depth.astype(np.float32)
    damage = np.clip(
        np.abs(delta).astype(np.float32) / np.float32(255.0 * DAMAGE_REFERENCE),
        np.float32(0.0),
        np.float32(1.0),
    ).astype(np.float32)
    damage_float64 = np.clip(
        np.abs(delta).astype(np.float64) / (255.0 * float(DAMAGE_REFERENCE)), 0.0, 1.0
    )
    expected_param = expected_parameter(kind, severity, height, width)
    parameter_mismatches = [
        f"{key}: observed={parameter.get(key)!r} expected={value!r}"
        for key, value in expected_param.items()
        if key not in parameter or not value_equal(parameter[key], value)
    ]
    parameter_mismatches.extend(
        f"unexpected key: {key!r}" for key in parameter if key not in expected_param
    )
    expected_state, expected_burden, structural_mask = expected_state_and_burden(
        fixture, kind, severity, parameter, damage
    )
    target = np.asarray(result.reliability[1], dtype=np.float32)
    with np.errstate(under="ignore"):
        reconstructed = np.exp(-expected_burden).astype(np.float32)
    missing_expected = expected_burden >= np.float32(MISSING_BURDEN)
    state = np.asarray(result.validity_state, dtype=bool)
    supervised_reconstruction = (expected_state.astype(np.float32) * reconstructed).astype(np.float32)
    supervised_reconstruction[~geometry] = np.float32(1.0)
    metadata_checks: Dict[str, Any] = {}
    metadata_missing_keys: List[str] = []
    independent_damage = {
        "realized_damage_mean": float(damage.mean()),
        "realized_damage_max": float(damage.max()),
        "realized_damage_nonzero_fraction": float(np.count_nonzero(damage) / float(damage.size)),
    }
    for key, value in independent_damage.items():
        if key not in module_record:
            metadata_missing_keys.append(key)
            continue
        observed = float(module_record[key])
        metadata_checks[key] = {"module": observed, "independent": value, "abs_diff": abs(observed - value)}
    for key, value in stats(expected_burden).items():
        metadata_key = f"burden_{key}"
        if metadata_key not in module_record:
            metadata_missing_keys.append(metadata_key)
            continue
        observed = float(module_record[metadata_key])
        metadata_checks[metadata_key] = {"module": observed, "independent": value, "abs_diff": abs(observed - value)}
    for metadata_key, independent in (
        ("reliability_mean", float(target.mean())),
        ("reliability_min", float(target.min())),
    ):
        if metadata_key not in module_record:
            metadata_missing_keys.append(metadata_key)
            continue
        observed = float(module_record[metadata_key])
        metadata_checks[metadata_key] = {
            "module": observed,
            "independent": independent,
            "abs_diff": abs(observed - independent),
        }
    metadata_required_missing = [
        key for key in metadata_missing_keys if key in independent_damage
    ]
    metadata_agreement = not metadata_required_missing and all(
        item["abs_diff"] <= METADATA_ATOL for item in metadata_checks.values()
    )
    invalid_region = ~initial
    invalid_delta_max = float(np.max(np.abs(delta[invalid_region]))) if invalid_region.any() else 0.0
    target_supervised_stats = stats(supervised_reconstruction)
    record: Dict[str, Any] = {
        "fixture": str(fixture["name"]),
        "shape": [height, width],
        "kind": kind,
        "severity": float(severity),
        "modality": "depth",
        "protocol_id": "MMFR-A1-corruption-basis-v3",
        "severity_encoding": SEVERITY_ENCODING,
        "corruption_parameter": parameter,
        "expected_corruption_parameter": expected_param,
        "corruption_parameter_formula_match": not parameter_mismatches,
        "corruption_parameter_mismatches": parameter_mismatches,
        "realized_damage": independent_damage,
        "realized_damage_float64_mean": float(damage_float64.mean()),
        "realized_damage_float32_vs_float64_max_abs_diff": float(
            np.max(np.abs(damage.astype(np.float64) - damage_float64))
        ),
        "burden_all_pixels": stats(expected_burden),
        "burden_non_missing_pixels": stats(expected_burden[~missing_expected]) if (~missing_expected).any() else None,
        "burden_value_set": sorted(float(value) for value in np.unique(expected_burden)),
        "target_synthetic": stats(target),
        "target_synthetic_unique_values": sorted(float(value) for value in np.unique(target)),
        "target_bitwise_reconstruction_equal": bool(np.array_equal(target, reconstructed)),
        "target_reconstruction_max_abs_diff": float(np.max(np.abs(target.astype(np.float64) - reconstructed.astype(np.float64)))),
        "missing_mask_agrees": bool(np.array_equal(target == 0.0, missing_expected)),
        "supervised_target_reconstruction": target_supervised_stats,
        "initial_state_pixels": int(np.count_nonzero(initial)),
        "final_state_pixels": int(np.count_nonzero(state)),
        "expected_final_state_pixels": int(np.count_nonzero(expected_state)),
        "state_matches_expected": bool(np.array_equal(state, expected_state)),
        "invalid_fixture": {
            "native_invalid_pixels": int(np.count_nonzero(fixture["native_invalid"])),
            "geometry_invalid_pixels": int(np.count_nonzero(~geometry)),
            "initial_invalid_pixels_inside_geometry": int(np.count_nonzero(geometry & ~initial)),
            "invalid_region_delta_max_abs": invalid_delta_max,
            "intensity_invalid_region_unchanged": bool(invalid_delta_max == 0.0) if kind in GRADED_KINDS else None,
            "structural_mask_pixels": int(np.count_nonzero(structural_mask)) if structural_mask is not None else None,
        },
        "metadata_checks": metadata_checks,
        "metadata_missing_keys": metadata_missing_keys,
        "metadata_required_missing_keys": metadata_required_missing,
        "metadata_agreement": bool(metadata_agreement),
        "module_record": module_record,
    }
    if kind == "misalignment":
        displacement = float(parameter["displacement_px"])
        reference = float(parameter["reference_displacement_px"])
        record["misalignment"] = {
            "dy_px": int(parameter["dy_px"]),
            "dx_px": int(parameter["dx_px"]),
            "in_bounds_burden": float(min(1.0, displacement / reference)) if reference > 0 else 0.0,
            "module_normalized_burden": float(module_record["normalized_burden"]),
            "geometry_invalid_pixels": int(np.count_nonzero(structural_mask)) if structural_mask is not None else None,
            "validity_transport": "same integer translation of Depth and state, then geometry mask",
        }
    return record


def monotonic_check(
    records: Sequence[Mapping[str, Any]], key: str, direction: str, skip_misalignment_state: bool = False
) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for record in records:
        kind = str(record["kind"])
        if skip_misalignment_state and kind == "misalignment":
            continue
        grouped.setdefault((str(record["fixture"]), kind), []).append(record)
    failures: List[str] = []
    observations: List[Dict[str, Any]] = []
    for (fixture, kind), group in sorted(grouped.items()):
        series = sorted(group, key=lambda item: float(item["severity"]))
        values = [None if item[key] is None else float(item[key]) for item in series]
        present = [(float(item["severity"]), value) for item, value in zip(series, values) if value is not None]
        observations.append({"fixture": fixture, "kind": kind, "severities": [item[0] for item in present], "values": [item[1] for item in present]})
        for (previous_severity, previous), (current_severity, current) in zip(present, present[1:]):
            if direction == "non_decreasing" and current < previous - 1.0e-12:
                failures.append(f"{fixture}/{kind}: {key} decreased {previous} -> {current} at {previous_severity}->{current_severity}")
            if direction == "non_increasing" and current > previous + 1.0e-12:
                failures.append(f"{fixture}/{kind}: {key} increased {previous} -> {current} at {previous_severity}->{current_severity}")
    return {"pass": not failures, "series_checked": len(observations), "observations": observations, "failures": failures}


def check_graded(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    worst_roundtrip = 0.0
    worst_target_diff = 0.0
    worst_precision = 0.0
    for record in records:
        if record["kind"] not in GRADED_KINDS:
            continue
        label = f"{record['fixture']}/{record['kind']}@{record['severity']}"
        worst_roundtrip = max(worst_roundtrip, float(record["target_reconstruction_max_abs_diff"]))
        worst_target_diff = max(worst_target_diff, float(record["target_reconstruction_max_abs_diff"]))
        worst_precision = max(worst_precision, float(record["realized_damage_float32_vs_float64_max_abs_diff"]))
        if not record["target_bitwise_reconstruction_equal"]:
            failures.append(f"{label}: target != float32(exp(-independent realized burden))")
        if not record["corruption_parameter_formula_match"]:
            failures.append(f"{label}: parameter formula mismatch: {record['corruption_parameter_mismatches']}")
        if not record["metadata_agreement"]:
            failures.append(f"{label}: metadata mismatch")
        if not record["state_matches_expected"]:
            failures.append(f"{label}: invalid-state transition changed unexpectedly")
        if not record["invalid_fixture"]["intensity_invalid_region_unchanged"]:
            failures.append(f"{label}: intensity corruption changed native-invalid/padding Depth")
        if worst_precision > FLOAT32_PRECISION_ATOL:
            failures.append(f"{label}: float32/float64 damage precision exceeded tolerance")
    return {
        "pass": not failures,
        "runs_checked": sum(1 for record in records if record["kind"] in GRADED_KINDS),
        "requirement": "single-encoded graded burden equals independently realized normalized damage",
        "max_target_reconstruction_abs_diff": worst_target_diff,
        "max_float32_log_roundtrip_abs_diff": worst_roundtrip,
        "max_float32_vs_float64_damage_abs_diff": worst_precision,
        "float32_precision_tolerance": FLOAT32_PRECISION_ATOL,
        "failures": failures,
    }


def check_structural(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    structural = [record for record in records if record["kind"] in STRUCTURAL_KINDS]
    for record in structural:
        label = f"{record['fixture']}/{record['kind']}@{record['severity']}"
        if not set(record["burden_value_set"]).issubset({0.0, float(MISSING_BURDEN)}):
            failures.append(f"{label}: burden is not binary structural burden")
        if not set(record["target_synthetic_unique_values"]).issubset({0.0, 1.0}):
            failures.append(f"{label}: target is not binary structural target")
        if not record["target_bitwise_reconstruction_equal"]:
            failures.append(f"{label}: structural target reconstruction mismatch")
        if not record["state_matches_expected"]:
            failures.append(f"{label}: structural validity state mismatch")
    return {"pass": not failures, "runs_checked": len(structural), "failures": failures}


def check_misalignment(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    runs = [record for record in records if record["kind"] == "misalignment"]
    for record in runs:
        label = f"{record['fixture']}/misalignment@{record['severity']}"
        parameter = record["corruption_parameter"]
        expected = float(min(1.0, float(parameter["displacement_px"]) / float(parameter["reference_displacement_px"])))
        observed = float(record["module_record"]["normalized_burden"])
        if not math.isclose(expected, observed, rel_tol=1.0e-12, abs_tol=1.0e-12):
            failures.append(f"{label}: in-bounds burden {observed} != {expected}")
        if not record["target_bitwise_reconstruction_equal"]:
            failures.append(f"{label}: target reconstruction mismatch")
        if not record["missing_mask_agrees"]:
            failures.append(f"{label}: translated out-of-bounds mask mismatch")
        if not record["state_matches_expected"]:
            failures.append(f"{label}: MID-A state transport mismatch")
        if record["corruption_parameter"].get("transport") != "integer-translate-depth-and-validity-together":
            failures.append(f"{label}: transport marker mismatch")
    return {
        "pass": not failures,
        "runs_checked": len(runs),
        "requirement": "in-bounds burden follows actual integer displacement/reference; validity follows the same MID-A translation",
        "failures": failures,
    }


def check_dropout_extent(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    observations: List[Dict[str, Any]] = []
    for fixture in sorted({str(record["fixture"]) for record in records}):
        series = sorted(
            [record for record in records if record["fixture"] == fixture and record["kind"] == "spatial_dropout"],
            key=lambda item: float(item["severity"]),
        )
        cells = [int(record["corruption_parameter"]["dropped_cells"]) for record in series]
        pixels = [int(record["invalid_fixture"]["structural_mask_pixels"]) for record in series]
        observations.append({"fixture": fixture, "severities": [float(record["severity"]) for record in series], "dropped_cells": cells, "dropped_pixels": pixels})
        for left, right in zip(cells, cells[1:]):
            if right < left:
                failures.append(f"{fixture}/spatial_dropout: dropped_cells decreased {left}->{right}")
        for left, right in zip(pixels, pixels[1:]):
            if right < left:
                failures.append(f"{fixture}/spatial_dropout: dropped_pixels decreased {left}->{right}")
    return {"pass": not failures, "observations": observations, "failures": failures}


def check_relative_scale(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    rows: List[Dict[str, Any]] = []
    for record in records:
        if record["kind"] not in ("blur", "misalignment"):
            continue
        severity = float(record["severity"])
        parameter = record["corruption_parameter"]
        height, width = record["shape"]
        if record["kind"] == "blur":
            ratio = float(parameter["sigma_px"]) / float(min(height, width))
            expected = BLUR_SIGMA_FRACTION * severity
            error = abs(ratio - expected)
            rows.append({"kind": "blur", "fixture": record["fixture"], "severity": severity, "sigma_over_min_side": ratio, "expected": expected, "abs_error": error})
            if error > 1.0e-12:
                failures.append(f"{record['fixture']}/blur@{severity}: relative sigma {ratio} != {expected}")
        else:
            y_ratio = float(parameter["max_shift_y_px"]) / float(height)
            x_ratio = float(parameter["max_shift_x_px"]) / float(width)
            permitted = math.hypot(float(parameter["max_shift_y_px"]), float(parameter["max_shift_x_px"]))
            reference = float(parameter["reference_displacement_px"])
            expected = MISALIGN_MAX_SHIFT_FRACTION * severity
            row = {
                "kind": "misalignment",
                "fixture": record["fixture"],
                "severity": severity,
                "max_shift_y_over_height": y_ratio,
                "max_shift_x_over_width": x_ratio,
                "expected_axis_fraction": expected,
                "permitted_displacement_over_reference": permitted / reference,
                "realized_displacement_over_reference": float(parameter["displacement_px"]) / reference,
            }
            rows.append(row)
            if abs(y_ratio - expected) > 1.0e-12 or abs(x_ratio - expected) > 1.0e-12:
                failures.append(f"{record['fixture']}/misalignment@{severity}: axis fraction mismatch")
            if abs(permitted / reference - severity) > 1.0e-12:
                failures.append(f"{record['fixture']}/misalignment@{severity}: permitted/reference != severity")
    by_kind_severity: Dict[Tuple[str, float], List[float]] = {}
    for row in rows:
        if row["kind"] == "misalignment":
            by_kind_severity.setdefault((row["kind"], row["severity"]), []).append(float(row["realized_displacement_over_reference"]))
    spreads = []
    for (kind, severity), values in sorted(by_kind_severity.items()):
        spread = (max(values) - min(values)) / max(values) if len(values) == 2 and max(values) > 0 else 0.0
        spreads.append({"kind": kind, "severity": severity, "max_relative_spread": spread})
    return {
        "pass": not failures,
        "metric": "blur sigma/min-side and misalignment permitted axis/displacement fractions",
        "rows": rows,
        "realized_misalignment_ratio_spreads": spreads,
        "note": "MID-A may move which invalid pixels are present; misalignment strength is gated by the actual permitted/reference burden, not by pixel identity.",
        "failures": failures,
    }


def check_empty_spec(fixtures: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    observations: List[Dict[str, Any]] = []
    for name, fixture in fixtures.items():
        depth = fixture["depth"]
        rgb = depth[:, :, None].repeat(3, axis=2)
        result = apply_failures(
            rgb,
            depth,
            (),
            make_rng(),
            depth_validity=fixture["depth_valid_pre"],
            validity_mask=fixture["geometry_valid"],
        )
        observation = {
            "fixture": name,
            "rgb_elementwise_equal": bool(np.array_equal(result.rgb, rgb)),
            "depth_elementwise_equal": bool(np.array_equal(result.depth, depth)),
            "same_rgb_object": bool(result.rgb is rgb),
            "same_depth_object": bool(result.depth is depth),
            "state_equals_initial": bool(np.array_equal(result.validity_state, fixture["depth_valid_pre"])),
            "target_all_ones": bool(np.all(result.reliability == 1.0)),
            "target_shape": list(result.reliability.shape),
            "metadata_empty": bool(dict(result.metadata) == {}),
        }
        observations.append(observation)
        for key in ("rgb_elementwise_equal", "depth_elementwise_equal", "same_rgb_object", "same_depth_object", "state_equals_initial", "target_all_ones", "metadata_empty"):
            if not observation[key]:
                failures.append(f"{name}: empty spec {key} failed")
        if observation["target_shape"] != [2, fixture["height"], fixture["width"]]:
            failures.append(f"{name}: empty spec target shape mismatch")
    return {"pass": not failures, "observations": observations, "failures": failures}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="structured JSON evidence path")
    args = parser.parse_args(argv)
    fixtures = build_fixtures()
    records = [
        run_job(fixtures[fixture_name], kind, severity)
        for fixture_name in fixtures
        for kind in DEPTH_CHAIN_KINDS
        for severity in SEVERITY_GRID
    ]
    for record in records:
        record["burden_all_pixels_mean"] = record["burden_all_pixels"]["mean"]
        record["burden_non_missing_pixels_mean"] = (
            record["burden_non_missing_pixels"]["mean"] if record["burden_non_missing_pixels"] else None
        )
        record["target_synthetic_mean"] = record["target_synthetic"]["mean"]
        record["final_state_pixels_value"] = record["final_state_pixels"]
    checks: Dict[str, Any] = {
        "graded_single_encoding_and_realized_burden": check_graded(records),
        "structural_burden_and_target": check_structural(records),
        "misalignment_actual_strength_and_mid_a_transport": check_misalignment(records),
        "burden_mean_monotone_nondecreasing": monotonic_check(records, "burden_all_pixels_mean", "non_decreasing"),
        "burden_non_missing_monotone_nondecreasing": monotonic_check(records, "burden_non_missing_pixels_mean", "non_decreasing"),
        "synthetic_target_mean_monotone_nonincreasing": monotonic_check(records, "target_synthetic_mean", "non_increasing"),
        "state_extent_monotone_for_nontransport_kinds": monotonic_check(records, "final_state_pixels_value", "non_increasing", skip_misalignment_state=True),
        "dropout_extent_monotone": check_dropout_extent(records),
        "relative_scale_consistent": check_relative_scale(records),
        "empty_spec_strict_noop": check_empty_spec(fixtures),
    }
    violations: List[str] = []
    for name, check in checks.items():
        violations.extend(f"{name}: {failure}" for failure in check.get("failures", []))
    conclusions = {
        "single_severity_encoding": SEVERITY_ENCODING == "single",
        "no_double_severity_encoding": bool(checks["graded_single_encoding_and_realized_burden"]["pass"] and checks["structural_burden_and_target"]["pass"] and checks["misalignment_actual_strength_and_mid_a_transport"]["pass"]),
        "monotonic_burden": bool(checks["burden_mean_monotone_nondecreasing"]["pass"]),
        "monotonic_target": bool(checks["synthetic_target_mean_monotone_nonincreasing"]["pass"]),
        "relative_scale_consistent": bool(checks["relative_scale_consistent"]["pass"]),
        "dropout_extent_monotone": bool(checks["dropout_extent_monotone"]["pass"]),
        "invalid_state_consistent": bool(checks["graded_single_encoding_and_realized_burden"]["pass"] and checks["structural_burden_and_target"]["pass"] and checks["misalignment_actual_strength_and_mid_a_transport"]["pass"]),
        "empty_spec_strict_noop": bool(checks["empty_spec_strict_noop"]["pass"]),
        "violations": violations,
        "violation_count": len(violations),
    }
    report = {
        "schema_version": "mmfr-a1-v3-severity-burden-audit-v1",
        "audit": "MMFR-A1-corruption-basis-v3 severity -> corruption parameter -> realized burden -> target",
        "protocol_id": "MMFR-A1-corruption-basis-v3",
        "official_test_included": False,
        "checkpoint_read": False,
        "training_run": False,
        "gpu_used": False,
        "network_used": False,
        "audit_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "python": sys.version.split()[0], "numpy": np.__version__},
        "module_under_test": {"path": str(Path(mmfr_module.__file__).resolve()), "sha256": sha256_file(Path(mmfr_module.__file__).resolve()), "severity_encoding": SEVERITY_ENCODING},
        "fixed_seed": FIXED_SEED,
        "rng_construction": "numpy.random.default_rng(numpy.random.SeedSequence([FIXED_SEED])) rebuilt for every kind/severity run",
        "fixture_semantics": "both resolutions use the same deterministic native-invalid pattern and relative geometry border rule; every run passes the same fixture's depth_valid_pre and geometry_valid explicitly",
        "severity_grid": list(SEVERITY_GRID),
        "required_severities": list(REQUIRED_SEVERITIES),
        "summary_severities": list(SUMMARY_SEVERITIES),
        "failure_kinds": list(FAILURE_KINDS),
        "fixtures": [fixture["record"] for fixture in fixtures.values()],
        "constants": {"MISSING_BURDEN": MISSING_BURDEN, "DAMAGE_REFERENCE": DAMAGE_REFERENCE, "NOISE_SIGMA_MAX": NOISE_SIGMA_MAX, "BLUR_SIGMA_FRACTION": BLUR_SIGMA_FRACTION, "MISALIGN_MAX_SHIFT_FRACTION": MISALIGN_MAX_SHIFT_FRACTION, "MIN_DROPOUT_CELL": MIN_DROPOUT_CELL, "DROPOUT_GRID_TARGET": DROPOUT_GRID_TARGET},
        "records": records,
        "checks": checks,
        "conclusions": conclusions,
        "status": "PASS" if not violations else "FAIL",
        "exit_code": 0 if not violations else 1,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    report_sha = sha256_file(output)
    print("MMFR-A1-corruption-basis-v3 severity/burden audit (CPU only)")
    print(f"protocol_id={report['protocol_id']} severity_encoding={SEVERITY_ENCODING!r} fixtures={len(fixtures)} records={len(records)}")
    for fixture_name in fixtures:
        selected = [record for record in records if record["fixture"] == fixtures[fixture_name]["name"]]
        print(f"fixture={fixture_name} shape={fixtures[fixture_name]['height']}x{fixtures[fixture_name]['width']} depth_valid_pre_pixels={fixtures[fixture_name]['record']['depth_valid_pre_pixels']}")
        for kind in DEPTH_CHAIN_KINDS:
            rows = [record for record in selected if record["kind"] == kind and float(record["severity"]) in SUMMARY_SEVERITIES]
            row = rows[-1]
            print(f"  {kind:<16} sev={row['severity']:.2f} burden_mean={row['burden_all_pixels']['mean']:.6f} target_mean={row['target_synthetic']['mean']:.6f} final_state={row['final_state_pixels']}")
    print("checks:")
    for name, check in checks.items():
        print(f"  {name}: {'PASS' if check['pass'] else 'FAIL'}")
    print(f"conclusions={json.dumps(conclusions, ensure_ascii=False, sort_keys=True)}")
    print(f"JSON report: {output}")
    print(f"SHA-256: {report_sha}")
    print("OVERALL: " + report["status"])
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
