#!/usr/bin/env python3
"""Pure DVG-B1 Oracle-mask geometry operators and CPU qualification."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from models.encoders.DFormerv2 import (
    build_dvg_b1_pairwise_gates,
    build_dvg_b1_stage_reliability,
    normalize_dvg_b1_corruption_mask,
)

DVG_B1_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
DVG_B1_PAD_DIVISOR = 32


def build_view_corruption_mask(
    raw_corruption_mask: np.ndarray,
    *,
    scaled_size_hw: tuple[int, int],
    padded_size_hw: tuple[int, int],
    flipped: bool,
) -> np.ndarray:
    """Map one raw corruption mask to a padded evaluator view."""
    raw = np.asarray(raw_corruption_mask)
    if raw.ndim != 2 or raw.dtype != np.bool_:
        raise ValueError("raw corruption mask must be a two-dimensional boolean array")
    scaled_height, scaled_width = (int(value) for value in scaled_size_hw)
    padded_height, padded_width = (int(value) for value in padded_size_hw)
    if min(scaled_height, scaled_width) <= 0:
        raise ValueError("scaled view geometry must be positive")
    if padded_height < scaled_height or padded_width < scaled_width:
        raise ValueError("padded view cannot be smaller than the scaled view")
    resized = cv2.resize(
        raw.astype(np.float32),
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)
    if flipped:
        resized = np.flip(resized, axis=1).copy()
    padded = np.pad(
        resized,
        ((0, padded_height - scaled_height), (0, padded_width - scaled_width)),
        mode="constant",
        constant_values=0.0,
    )
    return np.ascontiguousarray(padded, dtype=np.float32)


def stage_sizes_from_padded_view(padded_size_hw: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return the four DFormerv2 stage grids for a divisor-32 padded view."""
    padded_height, padded_width = (int(value) for value in padded_size_hw)
    if padded_height <= 0 or padded_width <= 0:
        raise ValueError("padded view geometry must be positive")
    if padded_height % DVG_B1_PAD_DIVISOR or padded_width % DVG_B1_PAD_DIVISOR:
        raise ValueError("DVG-B1 padded view geometry must be divisible by 32")
    return tuple((padded_height // divisor, padded_width // divisor) for divisor in (4, 8, 16, 32))


def _view_geometry(raw_size_hw: tuple[int, int], scale: float) -> tuple[tuple[int, int], tuple[int, int]]:
    raw_height, raw_width = raw_size_hw
    scaled_height = max(1, int(round(raw_height * float(scale))))
    scaled_width = max(1, int(round(raw_width * float(scale))))
    padded_height = math.ceil(scaled_height / DVG_B1_PAD_DIVISOR) * DVG_B1_PAD_DIVISOR
    padded_width = math.ceil(scaled_width / DVG_B1_PAD_DIVISOR) * DVG_B1_PAD_DIVISOR
    return (scaled_height, scaled_width), (padded_height, padded_width)


def _fixture_masks(raw_size_hw: tuple[int, int]) -> Mapping[str, np.ndarray]:
    height, width = raw_size_hw
    all_trusted = np.zeros((height, width), dtype=np.bool_)
    all_corrupted = np.ones((height, width), dtype=np.bool_)
    single_pixel = np.zeros((height, width), dtype=np.bool_)
    single_pixel[height // 2, width // 2] = True
    cross_token_boundary = np.zeros((height, width), dtype=np.bool_)
    row, column = height // 2, width // 2
    cross_token_boundary[max(0, row - 2) : min(height, row + 3), :] = True
    cross_token_boundary[:, max(0, column - 2) : min(width, column + 3)] = True
    return {
        "all-trusted": all_trusted,
        "all-corrupted": all_corrupted,
        "single-pixel": single_pixel,
        "cross-token-boundary": cross_token_boundary,
    }


def _geometry_guard_qualification() -> dict[str, bool]:
    checks: dict[str, bool] = {}

    invalid_padded = torch.zeros((1, 1, 33, 64), dtype=torch.float32)
    try:
        normalize_dvg_b1_corruption_mask(invalid_padded, invalid_padded)
    except ValueError:
        checks["reject_non_divisor_32_padded_view"] = True
    else:
        checks["reject_non_divisor_32_padded_view"] = False

    padded = torch.zeros((1, 1, 64, 96), dtype=torch.float32)
    for name, stage_size in {
        "reject_non_integer_stage_reduction": (16, 11),
        "reject_anisotropic_stage_reduction": (16, 12),
        "reject_unregistered_stage_reduction": (32, 48),
    }.items():
        try:
            build_dvg_b1_stage_reliability(padded, stage_size)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    return checks


def _gate_qualification() -> dict[str, Any]:
    reliability = torch.tensor(
        [[[[1.0, 0.75, 0.25, 0.0], [0.5, 1.0, 0.0, 0.25], [0.0, 0.5, 1.0, 0.75]]]],
        dtype=torch.float32,
    )
    batch, _, height, width = reliability.shape
    full = build_dvg_b1_pairwise_gates(reliability, False)
    gate_h, gate_w = build_dvg_b1_pairwise_gates(reliability, True)
    flat = reliability[:, 0].flatten(1)
    expected_full = flat[:, None, :, None] * flat[:, None, None, :]
    expected_h = reliability[:, 0].transpose(1, 2)[:, None, :, :, None] * reliability[:, 0].transpose(
        1, 2
    )[:, None, :, None, :]
    expected_w = reliability[:, 0, :, :, None] * reliability[:, 0, :, None, :]
    expected_w = expected_w[:, None]
    checks = {
        "full_shape": list(full.shape) == [batch, 1, height * width, height * width],
        "h_shape": list(gate_h.shape) == [batch, 1, width, height, height],
        "w_shape": list(gate_w.shape) == [batch, 1, height, width, width],
        "full_expected": torch.equal(full, expected_full),
        "h_expected": torch.equal(gate_h, expected_h),
        "w_expected": torch.equal(gate_w, expected_w),
        "full_symmetric": torch.equal(full, full.transpose(-1, -2)),
        "h_symmetric": torch.equal(gate_h, gate_h.transpose(-1, -2)),
        "w_symmetric": torch.equal(gate_w, gate_w.transpose(-1, -2)),
        "range": bool(
            (full.min() >= 0)
            and (full.max() <= 1)
            and (gate_h.min() >= 0)
            and (gate_h.max() <= 1)
            and (gate_w.min() >= 0)
            and (gate_w.max() <= 1)
        ),
    }
    ones = torch.ones((1, 1, 3, 4), dtype=torch.float32)
    ones_full = build_dvg_b1_pairwise_gates(ones, False)
    ones_h, ones_w = build_dvg_b1_pairwise_gates(ones, True)
    checks["all_one_identity"] = bool(
        torch.equal(ones_full, torch.ones_like(ones_full))
        and torch.equal(ones_h, torch.ones_like(ones_h))
        and torch.equal(ones_w, torch.ones_like(ones_w))
    )
    zero_index = 3
    checks["zero_endpoint_closes_pairs"] = bool(
        torch.equal(full[0, 0, zero_index], torch.zeros_like(full[0, 0, zero_index]))
        and torch.equal(full[0, 0, :, zero_index], torch.zeros_like(full[0, 0, :, zero_index]))
    )
    return {
        "status": "passed" if all(checks.values()) else "protocol-blocked",
        "checks": checks,
        "sample_shapes": {
            "stage_reliability": list(reliability.shape),
            "full": list(full.shape),
            "h": list(gate_h.shape),
            "w": list(gate_w.shape),
        },
    }


def run_cpu_qualification(
    *,
    scales: Sequence[float] = DVG_B1_SCALES,
    raw_size_hw: tuple[int, int] = (37, 53),
) -> dict[str, Any]:
    """Exercise A/B on deterministic synthetic masks without model forward."""
    if tuple(float(value) for value in scales) != DVG_B1_SCALES:
        raise ValueError("qualification scales must match the frozen five-scale evaluator")
    fixtures = _fixture_masks(raw_size_hw)
    view_records: list[dict[str, Any]] = []
    stage_cache: dict[tuple[str, float, bool, int], np.ndarray] = {}
    view_mask_cache: dict[tuple[str, float, bool], np.ndarray] = {}
    failures: list[str] = []
    partial_observed = False

    for fixture_name, raw_mask in fixtures.items():
        for scale in DVG_B1_SCALES:
            scaled_size, padded_size = _view_geometry(raw_size_hw, scale)
            stage_sizes = stage_sizes_from_padded_view(padded_size)
            for flipped in (False, True):
                view_mask = build_view_corruption_mask(
                    raw_mask,
                    scaled_size_hw=scaled_size,
                    padded_size_hw=padded_size,
                    flipped=flipped,
                )
                view_mask_cache[(fixture_name, scale, flipped)] = view_mask
                pad_height = padded_size[0] - scaled_size[0]
                pad_width = padded_size[1] - scaled_size[1]
                if pad_height and not np.array_equal(view_mask[-pad_height:, :], np.zeros_like(view_mask[-pad_height:, :])):
                    failures.append(f"{fixture_name}/{scale}/{flipped}: bottom padding is not trusted")
                if pad_width and not np.array_equal(view_mask[:, -pad_width:], np.zeros_like(view_mask[:, -pad_width:])):
                    failures.append(f"{fixture_name}/{scale}/{flipped}: right padding is not trusted")
                view_tensor = torch.from_numpy(view_mask).unsqueeze(0).unsqueeze(0)
                for stage_index, stage_size in enumerate(stage_sizes):
                    stage = build_dvg_b1_stage_reliability(view_tensor, stage_size)
                    stage_array = stage[0, 0].numpy()
                    height_divisor = padded_size[0] // stage_size[0]
                    width_divisor = padded_size[1] // stage_size[1]
                    expected = (
                        (1.0 - view_mask)
                        .reshape(
                            stage_size[0],
                            height_divisor,
                            stage_size[1],
                            width_divisor,
                        )
                        .mean(axis=(1, 3), dtype=np.float32)
                    )
                    max_abs_error = float(np.max(np.abs(stage_array - expected)))
                    if max_abs_error > 1e-6:
                        failures.append(
                            f"{fixture_name}/{scale}/{flipped}/stage-{stage_index}: "
                            f"OpenCV INTER_AREA differs from exact block mean by {max_abs_error}"
                        )
                    if not np.isfinite(stage_array).all() or float(stage_array.min()) < 0 or float(stage_array.max()) > 1:
                        failures.append(f"{fixture_name}/{scale}/{flipped}/stage-{stage_index}: invalid reliability range")
                    if fixture_name == "all-trusted" and not np.array_equal(stage_array, np.ones_like(stage_array)):
                        failures.append(f"{fixture_name}/{scale}/{flipped}/stage-{stage_index}: all-one identity failed")
                    partial_count = int(np.count_nonzero((stage_array > 0) & (stage_array < 1)))
                    partial_observed = partial_observed or (
                        fixture_name in {"single-pixel", "cross-token-boundary"} and partial_count > 0
                    )
                    stage_cache[(fixture_name, scale, flipped, stage_index)] = stage_array
                    view_records.append(
                        {
                            "fixture": fixture_name,
                            "scale": scale,
                            "flipped": flipped,
                            "scaled_size_hw": list(scaled_size),
                            "padded_size_hw": list(padded_size),
                            "stage_index": stage_index,
                            "stage_size_hw": list(stage_size),
                            "minimum": float(stage_array.min()),
                            "maximum": float(stage_array.max()),
                            "partial_token_count": partial_count,
                            "opencv_inter_area_vs_exact_block_mean_max_abs_error": max_abs_error,
                        }
                    )

    flip_records: list[dict[str, Any]] = []
    for fixture_name in ("single-pixel", "cross-token-boundary"):
        for scale in DVG_B1_SCALES:
            scaled_size, padded_size = _view_geometry(raw_size_hw, scale)
            nonflipped_view = view_mask_cache[(fixture_name, scale, False)]
            flipped_view = view_mask_cache[(fixture_name, scale, True)]
            expected_flipped_content = np.flip(
                nonflipped_view[: scaled_size[0], : scaled_size[1]],
                axis=1,
            )
            content_error = float(
                np.max(
                    np.abs(
                        flipped_view[: scaled_size[0], : scaled_size[1]]
                        - expected_flipped_content
                    )
                )
            )
            if content_error != 0.0:
                failures.append(f"{fixture_name}/{scale}: resize-then-flip view content is misaligned")
            for stage_index, _stage_size in enumerate(stage_sizes_from_padded_view(padded_size)):
                original = stage_cache[(fixture_name, scale, False, stage_index)]
                inverse_flip = np.flip(stage_cache[(fixture_name, scale, True, stage_index)], axis=1)
                flip_records.append(
                    {
                        "fixture": fixture_name,
                        "scale": scale,
                        "stage_index": stage_index,
                        "padding_width_pixels": padded_size[1] - scaled_size[1],
                        "scaled_content_flip_max_abs_error": content_error,
                        "full_grid_inverse_flip_max_abs_error": float(
                            np.max(np.abs(original - inverse_flip))
                        ),
                        "full_grid_difference_role": "descriptive-single-sided-padding-and-token-phase-effect",
                    }
                )

    if not partial_observed:
        failures.append("qualification fixtures did not produce any partial token")
    geometry_guards = _geometry_guard_qualification()
    if not all(geometry_guards.values()):
        failures.append("padded-view or stage geometry fail-closed qualification failed")
    gate = _gate_qualification()
    if gate["status"] != "passed":
        failures.append("query/key product gate qualification failed")

    return {
        "schema_version": "museg-dvg-b1-cpu-qualification-v1",
        "protocol_id": "DVG-B1-oracle-gsa-v1",
        "status": "passed" if not failures else "protocol-blocked",
        "device": "cpu",
        "checkpoint_loaded": False,
        "model_forward_executed": False,
        "gpu_used": False,
        "official_test_included": False,
        "raw_fixture_size_hw": list(raw_size_hw),
        "scales": list(DVG_B1_SCALES),
        "fixtures": list(fixtures),
        "view_stage_record_count": len(view_records),
        "view_stage_records": view_records,
        "flip_padding_records": flip_records,
        "geometry_guard_checks": geometry_guards,
        "pairwise_gate": gate,
        "partial_token_observed": partial_observed,
        "failures": failures,
    }
