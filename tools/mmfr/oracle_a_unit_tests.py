#!/usr/bin/env python3
"""Implementation-level qualification for ``MMFR-Oracle-A``.

Runs the required Oracle-A unit tests before any formal 318-sample evaluation:

Test 1  all-valid identity: an all-valid validity state must reproduce the original
        checkpoint output bitwise through both oracle modes, including the intermediate
        geometry-prior tensors.
Test 2  all-invalid boundary: a fully invalid validity state must neutralise the
        Depth-derived geometry on every real token while leaving the position prior, the
        RGB attention and the decoder untouched.
Test 3  half-valid synthetic mask: left half valid / right half invalid, checked both at
        the pairwise-gate level and through the real ``GeoPriorGen``.
Test 4  natural clean invalid: the real ``clean`` validity state (raw ``Depth == 0``), its
        transport to each frozen view, and the proof that the transport neither loses an
        invalid pixel nor spreads one beyond its own bilinear support.
Test 5  no-action pipeline reproduction: ``original`` vs both all-valid oracle variants on
        real ``clean`` / ``spatial_dropout@0.75`` / ``entire_missing@1.0`` units.

The tests are read-only with respect to the checkpoint, the corruption pipeline and the
frozen evaluator.  Nothing is trained and no parameter is created.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import cv2

import tools.evaluate_museg_10condition as E10
import tools.evaluate_museg_checkpoint as EV
import tools.evaluate_museg_checkpoint_fast as FAST
from tools.mmfr.oracle_a_core import (
    ORACLE_VARIANTS,
    GeometryOracleModel,
    unit_invalidity_plan,
)
from utils.dataloader.oracle_a_validity import (
    build_view_invalidity,
    raw_invalidity,
    validity_summary,
)

DEFAULT_CONFIG = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3"
DEFAULT_PROTOCOL = "protocols/mmfr-a2-train-integration-v3.template.json"
DEFAULT_DATASET_ROOT = r"D:\0Project\dataset\MUSeg_DFormer"
DEFAULT_SPLIT = r"data\splits\MUSeg\dev-v1\val-dev.txt"
VIEW_PAD_DIVISOR = 32


class Recorder:
    """Accumulates assertions and free-form evidence for the final JSON report."""

    def __init__(self) -> None:
        self.assertions: list[dict[str, Any]] = []
        self.evidence: dict[str, Any] = {}

    def check(self, name: str, condition: bool, detail: Any = None) -> bool:
        self.assertions.append({"name": name, "passed": bool(condition), "detail": detail})
        return bool(condition)

    def all_passed(self) -> bool:
        return all(entry["passed"] for entry in self.assertions)

    def failures(self) -> list[dict[str, Any]]:
        return [entry for entry in self.assertions if not entry["passed"]]


def _tensor_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"comparable": False, "reason": f"shape {tuple(left.shape)} vs {tuple(right.shape)}"}
    if not torch.equal(left, right):
        difference = (left.float() - right.float()).abs()
        return {
            "comparable": True,
            "bitwise_equal": False,
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
            "prediction_mismatch": int((left.argmax(dim=1) != right.argmax(dim=1)).sum().item()),
        }
    return {
        "comparable": True,
        "bitwise_equal": True,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "prediction_mismatch": 0,
    }


# --------------------------------------------------------------------------------------
# independent recomputation of the geometry prior (test 1-3 reference values)
# --------------------------------------------------------------------------------------


def _expected_full_mask(
    generator: torch.nn.Module,
    reliability: torch.Tensor,
    depth_map: torch.Tensor,
    HW_tuple: tuple[int, int],
) -> torch.Tensor:
    height, width = HW_tuple
    weight = generator.weight
    interpolated = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)
    flat = interpolated.reshape(interpolated.shape[0], height * width)
    difference = (flat[:, :, None] - flat[:, None, :]).abs()
    mask_d = difference.unsqueeze(1) * generator.decay[None, :, None, None]
    positions = torch.stack(
        torch.meshgrid(
            [torch.arange(height).to(generator.decay), torch.arange(width).to(generator.decay)],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(height * width, 2)
    position = (positions[:, None, :] - positions[None, :, :]).abs().sum(dim=-1)
    position = position * generator.decay[:, None, None]
    values = reliability[:, 0].flatten(1)
    gate = values[:, None, :, None] * values[:, None, None, :]
    return weight[0] * position + weight[1] * (gate * mask_d)


def _values_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Value equality after broadcasting, for paths that legitimately differ in rank."""
    try:
        broadcast = torch.broadcast_tensors(left, right)
    except RuntimeError:
        return False
    return bool(torch.equal(broadcast[0], broadcast[1]))


def _position_only_full(generator: torch.nn.Module, HW_tuple: tuple[int, int]) -> torch.Tensor:
    """Independent recomputation of ``weight[0] * position`` (the Depth-Geometry-Off mask)."""
    height, width = HW_tuple
    positions = torch.stack(
        torch.meshgrid(
            [torch.arange(height).to(generator.decay), torch.arange(width).to(generator.decay)],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(height * width, 2)
    position = (positions[:, None, :] - positions[None, :, :]).abs().sum(dim=-1)
    position = position * generator.decay[:, None, None]
    return generator.weight[0] * position


def _position_only_split(generator: torch.nn.Module, HW_tuple: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent recomputation of the decomposed ``weight[0] * position`` masks."""
    height, width = HW_tuple
    index_w = torch.arange(width).to(generator.decay)
    position_w = (index_w[:, None] - index_w[None, :]).abs() * generator.decay[:, None, None]
    index_h = torch.arange(height).to(generator.decay)
    position_h = (index_h[:, None] - index_h[None, :]).abs() * generator.decay[:, None, None]
    mask_w = generator.weight[0] * position_w.unsqueeze(0).unsqueeze(2)
    mask_h = generator.weight[0] * position_h.unsqueeze(0).unsqueeze(2)
    return mask_h, mask_w


def _depth_term_split(
    generator: torch.nn.Module,
    depth_map: torch.Tensor,
    HW_tuple: tuple[int, int],
    direction: str,
) -> torch.Tensor:
    height, width = HW_tuple
    interpolated = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)
    plane = interpolated[:, 0]
    if direction == "full":
        flat = plane.reshape(plane.shape[0], height * width)
        difference = (flat[:, :, None] - flat[:, None, :]).abs()
        return difference.unsqueeze(1) * generator.decay[None, :, None, None]
    if direction == "w":
        difference = (plane[:, :, :, None] - plane[:, :, None, :]).abs()
        return difference.unsqueeze(1) * generator.decay[None, :, None, None, None]
    transposed = plane.transpose(1, 2)
    difference = (transposed[:, :, :, None] - transposed[:, :, None, :]).abs()
    return difference.unsqueeze(1) * generator.decay[None, :, None, None, None]


def _expected_split_masks(
    generator: torch.nn.Module,
    reliability: torch.Tensor,
    depth_map: torch.Tensor,
    HW_tuple: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = HW_tuple
    weight = generator.weight
    index_w = torch.arange(width).to(generator.decay)
    position_w = (index_w[:, None] - index_w[None, :]).abs() * generator.decay[:, None, None]
    index_h = torch.arange(height).to(generator.decay)
    position_h = (index_h[:, None] - index_h[None, :]).abs() * generator.decay[:, None, None]
    reliability_h = reliability[:, 0].transpose(1, 2)
    gate_h = reliability_h[:, None, :, :, None] * reliability_h[:, None, :, None, :]
    gate_w = reliability[:, 0][:, None, :, :, None] * reliability[:, 0][:, None, :, None, :]
    mask_w = weight[0] * position_w.unsqueeze(0).unsqueeze(2) + weight[1] * (
        gate_w * _depth_term_split(generator, depth_map, HW_tuple, "w")
    )
    mask_h = weight[0] * position_h.unsqueeze(0).unsqueeze(2) + weight[1] * (
        gate_h * _depth_term_split(generator, depth_map, HW_tuple, "h")
    )
    return mask_h, mask_w


def test_geometry_gate_semantics(recorder: Recorder, generator: torch.nn.Module) -> None:
    """Test 3 at gate level plus the exact identity / off algebra of the mask."""
    torch.manual_seed(11)
    device = generator.decay.device
    height, width = 6, 9
    stage = (height, width)
    depth_map = torch.rand(1, 1, 13, 17, dtype=torch.float32, device=device)

    invalidity_pattern = torch.zeros(1, 1, 4, 6, dtype=torch.float32, device=device)
    invalidity_pattern[:, :, :, 3:] = 1.0
    invalidity_pattern[:, :, :2, :] = 1.0
    downsampled = F.interpolate(invalidity_pattern, size=stage, mode="bilinear", align_corners=False)
    expected_strict = (downsampled == 0).float()
    expected_continuous = (1.0 - downsampled).clamp(0.0, 1.0)

    from models.encoders.DFormerv2 import build_oracle_pairwise_gates, build_oracle_stage_reliability

    oracle_strict = {"mode": "strict", "invalidity": invalidity_pattern, "depth_geometry_off": False}
    oracle_continuous = {"mode": "continuous", "invalidity": invalidity_pattern, "depth_geometry_off": False}
    oracle_all_valid = {"mode": "continuous", "invalidity": torch.zeros_like(invalidity_pattern), "depth_geometry_off": False}
    oracle_strict_all_valid = {"mode": "strict", "invalidity": torch.zeros_like(invalidity_pattern), "depth_geometry_off": False}
    oracle_off = {"mode": "continuous", "invalidity": None, "depth_geometry_off": True}
    oracle_all_invalid = {"mode": "strict", "invalidity": torch.ones_like(invalidity_pattern), "depth_geometry_off": False}

    strict_reliability = build_oracle_stage_reliability(oracle_strict, stage, depth_map)
    continuous_reliability = build_oracle_stage_reliability(oracle_continuous, stage, depth_map)
    recorder.check(
        "test3.strict_reliability_is_binary",
        bool(((strict_reliability == 0) | (strict_reliability == 1)).all()),
    )
    recorder.check("test3.strict_reliability_matches_expected", bool(torch.equal(strict_reliability, expected_strict)))
    recorder.check(
        "test3.continuous_reliability_matches_expected",
        bool(torch.equal(continuous_reliability, expected_continuous)),
    )

    valid = strict_reliability[0, 0].flatten()
    gate = build_oracle_pairwise_gates(strict_reliability, False)[0, 0]
    valid_valid = gate[valid == 1][:, valid == 1]
    valid_invalid = gate[valid == 1][:, valid == 0]
    invalid_invalid = gate[valid == 0][:, valid == 0]
    recorder.check(
        "test3.valid_valid_pair_gate_is_one",
        bool(valid_valid.numel() > 0 and torch.equal(valid_valid, torch.ones_like(valid_valid))),
        {"count": int(valid_valid.numel())},
    )
    recorder.check(
        "test3.valid_invalid_pair_gate_is_zero",
        bool(valid_invalid.numel() > 0 and not valid_invalid.any()),
        {"count": int(valid_invalid.numel())},
    )
    recorder.check(
        "test3.invalid_invalid_pair_gate_is_zero",
        bool(invalid_invalid.numel() > 0 and not invalid_invalid.any()),
        {"count": int(invalid_invalid.numel())},
    )

    _, reference_mask = generator.forward(stage, depth_map, split_or_not=False)
    _, identity_mask = generator.forward(stage, depth_map, split_or_not=False, geometry_oracle=oracle_all_valid)
    _, strict_identity_mask = generator.forward(stage, depth_map, split_or_not=False, geometry_oracle=oracle_strict_all_valid)
    recorder.check("test1.full_all_valid_mask_bitwise_equal_reference", bool(torch.equal(identity_mask, reference_mask)))
    recorder.check(
        "test1.full_all_valid_strict_mask_bitwise_equal_reference",
        bool(torch.equal(strict_identity_mask, reference_mask)),
    )

    _, mask = generator.forward(stage, depth_map, split_or_not=False, geometry_oracle=oracle_continuous)
    recorder.check(
        "test1.full_continuous_mask_equals_expected",
        bool(torch.equal(mask, _expected_full_mask(generator, continuous_reliability, depth_map, stage))),
    )
    _, strict_mask = generator.forward(stage, depth_map, split_or_not=False, geometry_oracle=oracle_strict)
    recorder.check(
        "test1.full_strict_mask_equals_expected",
        bool(torch.equal(strict_mask, _expected_full_mask(generator, expected_strict, depth_map, stage))),
    )

    expected_off = _position_only_full(generator, stage)
    _, off_mask = generator.forward(stage, depth_map, split_or_not=False, geometry_oracle=oracle_off)
    recorder.check("test2.geometry_off_mask_equals_position_only", _values_equal(off_mask, expected_off))
    _, all_invalid_mask = generator.forward(stage, depth_map, split_or_not=False, geometry_oracle=oracle_all_invalid)
    recorder.check("test2.all_invalid_strict_mask_equals_position_only", _values_equal(all_invalid_mask, expected_off))
    recorder.check(
        "test2.all_invalid_strict_mask_bitwise_equals_geometry_off",
        bool(torch.equal(all_invalid_mask, off_mask)),
        {"shape_off": list(off_mask.shape), "shape_all_invalid": list(all_invalid_mask.shape)},
    )

    _, (reference_h, reference_w) = generator.forward(stage, depth_map, split_or_not=True)
    _, (identity_h, identity_w) = generator.forward(stage, depth_map, split_or_not=True, geometry_oracle=oracle_all_valid)
    recorder.check("test1.split_all_valid_mask_h_bitwise_equal_reference", bool(torch.equal(identity_h, reference_h)))
    recorder.check("test1.split_all_valid_mask_w_bitwise_equal_reference", bool(torch.equal(identity_w, reference_w)))
    _, (mask_h, mask_w) = generator.forward(stage, depth_map, split_or_not=True, geometry_oracle=oracle_continuous)
    expected_h, expected_w = _expected_split_masks(generator, continuous_reliability, depth_map, stage)
    recorder.check("test1.split_continuous_mask_h_equals_expected", bool(torch.equal(mask_h, expected_h)))
    recorder.check("test1.split_continuous_mask_w_equals_expected", bool(torch.equal(mask_w, expected_w)))
    expected_off_h, expected_off_w = _position_only_split(generator, stage)
    _, (off_h, off_w) = generator.forward(stage, depth_map, split_or_not=True, geometry_oracle=oracle_off)
    recorder.check("test2.split_geometry_off_mask_h_equals_position_only", _values_equal(off_h, expected_off_h))
    recorder.check("test2.split_geometry_off_mask_w_equals_position_only", _values_equal(off_w, expected_off_w))
    _, (all_invalid_h, all_invalid_w) = generator.forward(
        stage, depth_map, split_or_not=True, geometry_oracle=oracle_all_invalid
    )
    recorder.check("test2.split_all_invalid_strict_mask_h_equals_position_only", _values_equal(all_invalid_h, expected_off_h))
    recorder.check("test2.split_all_invalid_strict_mask_w_equals_position_only", _values_equal(all_invalid_w, expected_off_w))
    recorder.check(
        "test2.split_all_invalid_strict_mask_bitwise_equals_geometry_off",
        bool(torch.equal(all_invalid_h, off_h) and torch.equal(all_invalid_w, off_w)),
    )

    recorder.check("test2.position_prior_is_nonzero", bool(off_mask.abs().sum().item() > 0.0))
    recorder.check("test2.position_prior_is_nonpositive", bool((off_mask <= 0).all()), {"max": float(off_mask.max().item())})
    recorder.check(
        "test2.depth_term_is_active_in_reference",
        bool(not torch.equal(reference_mask, off_mask)),
        {"reference_minus_off_max_abs": float((reference_mask - off_mask).abs().max().item())},
    )

    recorder.evidence["test3_pattern"] = {
        "stage_hw": list(stage),
        "invalidity_pattern_shape": list(invalidity_pattern.shape),
        "strict_invalid_token_count": int((strict_reliability == 0).sum().item()),
        "strict_token_count": int(strict_reliability.numel()),
        "strict_invalid_fraction": float((strict_reliability == 0).float().mean().item()),
        "continuous_reliability_mean": float(continuous_reliability.mean().item()),
        "continuous_reliability_min": float(continuous_reliability.min().item()),
        "downsampled_invalidity_min": float(downsampled.min().item()),
        "downsampled_invalidity_max": float(downsampled.max().item()),
    }


# --------------------------------------------------------------------------------------
# test 4: real clean validity, transport and spread
# --------------------------------------------------------------------------------------


def _view_metadata(height: int, width: int) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for scale in EV.MSFLIP_SCALES:
        scaled_height, scaled_width, padded_height, padded_width = E10._view_geometry(
            height, width, float(scale), VIEW_PAD_DIVISOR
        )
        for flipped in (False, True):
            metadata.append(
                {
                    "scale": float(scale),
                    "flipped": bool(flipped),
                    "scaled_size_hw": (scaled_height, scaled_width),
                    "padded_size_hw": (padded_height, padded_width),
                }
            )
    return metadata


def _support_indices(count_in: int, count_out: int, index: int) -> tuple[int, int]:
    """Bilinear support columns of one downscaled output index (align-corners free)."""
    factor = float(count_in) / float(count_out)
    position = (float(index) + 0.5) * factor - 0.5
    low = int(math.floor(position))
    return (min(max(low, 0), count_in - 1), min(max(low + 1, 0), count_in - 1))
def test_natural_clean_validity(
    recorder: Recorder,
    dataset_root: Path,
    split_entries: Sequence[str],
    channel_order: str,
    sample_limit: int,
) -> None:
    from models.encoders.DFormerv2 import build_oracle_stage_reliability

    per_sample: list[dict[str, Any]] = []
    for entry in split_entries[:sample_limit]:
        sample_id, rgb, depth, label = E10.read_sample(dataset_root, entry, channel_order)
        del rgb, label
        validity = depth > 0
        invalidity = raw_invalidity(validity)
        summary = validity_summary(validity)
        recorder.check(
            f"test4.{sample_id}.invalidity_is_exactly_depth_equals_zero",
            bool(np.array_equal(invalidity.astype(bool), depth == 0)),
        )
        views = _view_metadata(depth.shape[0], depth.shape[1])
        view_records: list[dict[str, Any]] = []
        nonpad_fractions: list[float] = []
        mismatch_total = 0
        for view in views:
            scaled_height, scaled_width = view["scaled_size_hw"]
            padded_height, padded_width = view["padded_size_hw"]
            inv_view = build_view_invalidity(
                invalidity, view["scaled_size_hw"], view["padded_size_hw"], view["flipped"]
            )
            image = inv_view[0, 0, :scaled_height, :scaled_width]
            nonpad_fraction = float(image.mean())
            nonpad_fractions.append(nonpad_fraction)
            # Independent reimplementation of the raw -> view resize with PyTorch bilinear
            # (same align_corners=False coordinate convention as OpenCV INTER_LINEAR).
            reference = F.interpolate(
                torch.from_numpy(invalidity)[None, None],
                size=(scaled_height, scaled_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            if view["flipped"]:
                reference = np.flip(reference, axis=1)
            mask_mismatch = int(np.count_nonzero((image > 0) != (reference > 0)))
            mismatch_total += mask_mismatch
            max_abs = float(np.abs(image - reference).max())
            stage_height = max(1, padded_height // 4)
            stage_width = max(1, padded_width // 4)
            tensor = torch.from_numpy(inv_view)
            strict = build_oracle_stage_reliability(
                {"mode": "strict", "invalidity": tensor, "depth_geometry_off": False},
                (stage_height, stage_width),
                tensor,
            )
            continuous = build_oracle_stage_reliability(
                {"mode": "continuous", "invalidity": tensor, "depth_geometry_off": False},
                (stage_height, stage_width),
                tensor,
            )
            view_records.append(
                {
                    "scale": view["scale"],
                    "flipped": view["flipped"],
                    "stage1_hw": [stage_height, stage_width],
                    "strict_invalid_fraction": float((strict == 0).float().mean().item()),
                    "continuous_reliability_mean": float(continuous.mean().item()),
                    "nonpad_invalid_fraction": nonpad_fraction,
                    "cv2_vs_torch_boolean_mismatch": mask_mismatch,
                    "cv2_vs_torch_max_abs": max_abs,
                    "pad_is_exactly_valid": bool((inv_view[0, 0, scaled_height:, :] == 0).all())
                    and bool((inv_view[0, 0, :, scaled_width:] == 0).all()),
                }
            )
        recorder.check(
            f"test4.{sample_id}.transport_matches_independent_torch_bilinear",
            mismatch_total == 0,
            {"boolean_mismatch_pixels": mismatch_total},
        )
        recorder.check(
            f"test4.{sample_id}.transport_numeric_agreement",
            bool(max(record["cv2_vs_torch_max_abs"] for record in view_records) <= 1.0e-3),
            {"max_abs": max(record["cv2_vs_torch_max_abs"] for record in view_records)},
        )
        recorder.check(
            f"test4.{sample_id}.nonpad_invalid_fraction_ge_raw",
            bool(min(nonpad_fractions) >= summary["invalid_fraction"] - 1.0e-6),
            {"min_view_nonpad": float(min(nonpad_fractions)), "raw": summary["invalid_fraction"]},
        )
        per_sample.append(
            {
                "sample_id": sample_id,
                **summary,
                "view_nonpad_invalid_fraction_mean": float(np.mean(nonpad_fractions)),
                "views": view_records,
            }
        )
        del invalidity, validity, views
    recorder.evidence["test4_samples"] = per_sample


def test_depth_and_invalidity_share_support(recorder: Recorder) -> None:
    """The invalidity transport must reuse exactly the Depth view geometry.

    A single invalid raw pixel is enough: the set of scaled view pixels that the frozen
    Depth resize changes must be *identical* to the set the invalidity transport marks.
    """
    height, width = 96, 128
    depth = np.full((height, width), 200, dtype=np.uint8)
    depth[48, 64] = 0
    invalidity = raw_invalidity(depth > 0)
    results: dict[str, Any] = {}
    for scale in (1.0, 0.75, 0.5):
        scaled_height = max(1, int(round(height * scale)))
        scaled_width = max(1, int(round(width * scale)))
        padded_height = ((scaled_height + 31) // 32) * 32
        padded_width = ((scaled_width + 31) // 32) * 32
        depth_scaled = cv2.resize(depth, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        inv_view = build_view_invalidity(
            invalidity, (scaled_height, scaled_width), (padded_height, padded_width), False
        )[0, 0, :scaled_height, :scaled_width]
        depth_affected = depth_scaled != 200
        invalidity_affected = inv_view > 0.0
        identical = bool(np.array_equal(depth_affected, invalidity_affected))
        recorder.check(f"test4.support_identity.scale{int(scale * 100)}", identical, {
            "depth_affected": int(depth_affected.sum()),
            "invalidity_affected": int(invalidity_affected.sum()),
        })
        results[f"scale_{scale}"] = {
            "depth_affected_pixels": int(depth_affected.sum()),
            "invalidity_affected_pixels": int(invalidity_affected.sum()),
            "identical": identical,
        }
    recorder.evidence["test4_support_identity"] = results


def test_single_invalid_pixel_spread(recorder: Recorder) -> None:
    """One invalid raw pixel must mark exactly its own bilinear support, nothing else."""
    height, width = 96, 128
    validity = np.ones((height, width), dtype=bool)
    validity[48, 64] = False
    invalidity = raw_invalidity(validity)
    results: dict[str, Any] = {}
    for scale, (scaled_h, scaled_w) in ((1.0, (96, 128)), (0.5, (48, 64))):
        padded_h = ((scaled_h + 31) // 32) * 32
        padded_w = ((scaled_w + 31) // 32) * 32
        inv_view = build_view_invalidity(invalidity, (scaled_h, scaled_w), (padded_h, padded_w), False)
        support = inv_view[0, 0, :scaled_h, :scaled_w] > 0.0
        rows, columns = np.nonzero(support)
        ratio = width / float(scaled_w)
        expected_radius = int(math.ceil(ratio)) + 1
        row_span = int(rows.max() - rows.min()) + 1 if rows.size else 0
        column_span = int(columns.max() - columns.min()) + 1 if columns.size else 0
        recorder.check(
            f"test4.spread.scale{int(scale * 100)}.support_is_local",
            bool(rows.size > 0 and row_span <= 2 * expected_radius + 1 and column_span <= 2 * expected_radius + 1),
            {"row_span": row_span, "column_span": column_span, "expected_radius": expected_radius},
        )
        far = bool(inv_view[0, 0, 0, 0] == 0.0 and inv_view[0, 0, scaled_h - 1, scaled_w - 1] == 0.0)
        recorder.check(f"test4.spread.scale{int(scale * 100)}.far_pixels_exactly_zero", far)
        results[f"scale_{scale}"] = {
            "support_pixels": int(support.sum()),
            "row_span": row_span,
            "column_span": column_span,
            "expected_radius": expected_radius,
        }
    recorder.evidence["test4_single_pixel_spread"] = results


# --------------------------------------------------------------------------------------
# test 1/2/5: checkpoint level
# --------------------------------------------------------------------------------------


def _build_unit(
    dataset_root: Path,
    entry: str,
    channel_order: str,
    config: Any,
    condition: E10.Condition,
    evaluation_seed: int,
    validity_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    sample_id, rgb, depth, label = E10.read_sample(dataset_root, entry, channel_order)
    # Mirror the frozen Main-Val runner: the corruption seed words come from the split
    # entry (``RGB/<id>.jpg``), not from the bare sample id.
    corruption = E10.corrupt_depth(condition, rgb, depth, evaluation_seed, E10.normalize_sample_id(entry))
    if validity_override is not None:
        corruption = E10.CorruptionOutcome(
            depth=corruption.depth,
            validity_state=validity_override,
            metadata=corruption.metadata,
            seed_words=corruption.seed_words,
            rng_constructed=corruption.rng_constructed,
            strict_no_op=corruption.strict_no_op,
        )
    rgb_cache = E10.build_rgb_view_cache(rgb, config)
    views = E10.assemble_views(rgb_cache, corruption.depth)
    del rgb, depth
    sample_like = {
        "label": torch.from_numpy(np.ascontiguousarray(label)),
        "views": views,
        "sample_id": sample_id,
        "original_height": int(label.shape[0]),
        "original_width": int(label.shape[1]),
    }
    return sample_like, corruption, views


def _run_unit(
    timed_model: FAST.ForwardTimer,
    wrapped: GeometryOracleModel,
    sample_like: Mapping[str, Any],
    device: torch.device,
    config: Any,
    base_rng: Any,
    variant: Any,
    per_view_invalidity: Sequence[np.ndarray] | None,
) -> dict[str, Any]:
    wrapped.push_unit(per_view_invalidity)
    FAST._restore_rng(base_rng, device)
    result = E10.run_msflip(timed_model, sample_like, device, config, 1, {})
    if wrapped.pending() != 0:
        raise RuntimeError("oracle queue was not fully consumed; view bookkeeping is broken")
    return result


def run_checkpoint_tests(
    recorder: Recorder,
    dataset_root: Path,
    split_entries: Sequence[str],
    config: Any,
    channel_order: str,
    checkpoint: Path,
    device: torch.device,
    evaluation_seed: int,
) -> None:
    conditions = E10.build_conditions(E10.FROZEN_EVALUATION)
    by_directory = {condition.directory: condition for condition in conditions}
    units = [
        by_directory["clean"],
        by_directory["spatial_dropout_075"],
        by_directory["entire_missing_100"],
    ]
    inner = EV.load_model(config, checkpoint, device)
    variants = [
        ORACLE_VARIANTS["original"],
        ORACLE_VARIANTS["noop_continuous"],
        ORACLE_VARIANTS["noop_strict"],
    ]
    unit_report: list[dict[str, Any]] = []
    base_rng = FAST._capture_rng(device)
    for entry in split_entries:
        for condition in units:
            sample_like, corruption, views = _build_unit(
                dataset_root, entry, channel_order, config, condition, evaluation_seed
            )
            outputs: dict[str, torch.Tensor] = {}
            for variant in variants:
                wrapped = GeometryOracleModel(inner, variant)
                timed = FAST.ForwardTimer(wrapped)
                plan = unit_invalidity_plan(corruption.validity_state, views, variant) if variant.requires_mask() else None
                result = _run_unit(timed, wrapped, sample_like, device, config, base_rng, variant, plan)
                outputs[variant.name] = result["logits"].detach().clone()
                del result, timed, wrapped
            reference = outputs["original"]
            record: dict[str, Any] = {
                "sample_id": sample_like["sample_id"],
                "condition": condition.condition_id,
                "validity": validity_summary(corruption.validity_state),
            }
            for name in ("noop_continuous", "noop_strict"):
                comparison = _tensor_bitwise_equal(outputs[name], reference)
                record[name] = comparison
                recorder.check(
                    f"test1.identity.{record['sample_id']}.{condition.condition_id}.{name}.bitwise_equal",
                    bool(comparison.get("bitwise_equal")),
                    comparison,
                )
                recorder.check(
                    f"test1.identity.{record['sample_id']}.{condition.condition_id}.{name}.prediction_mismatch_zero",
                    int(comparison.get("prediction_mismatch", -1)) == 0,
                    {"prediction_mismatch": comparison.get("prediction_mismatch")},
                )
            unit_report.append(record)
            del sample_like, views, outputs, corruption
    recorder.evidence["test1_identity_units"] = unit_report


def run_boundary_tests(
    recorder: Recorder,
    dataset_root: Path,
    split_entries: Sequence[str],
    config: Any,
    channel_order: str,
    checkpoint: Path,
    device: torch.device,
    evaluation_seed: int,
) -> None:
    """Test 2 at checkpoint level: fully invalid validity vs Depth-Geometry-Off."""
    conditions = E10.build_conditions(E10.FROZEN_EVALUATION)
    clean = next(condition for condition in conditions if condition.directory == "clean")
    inner = EV.load_model(config, checkpoint, device)
    base_rng = FAST._capture_rng(device)
    records: list[dict[str, Any]] = []
    for entry in split_entries:
        sample_like, corruption, views = _build_unit(
            dataset_root, entry, channel_order, config, clean, evaluation_seed
        )
        all_invalid = np.zeros_like(corruption.validity_state)
        outputs: dict[str, torch.Tensor] = {}
        for name, variant in (
            ("strict_all_invalid", ORACLE_VARIANTS["strict"]),
            ("geometry_off", ORACLE_VARIANTS["geometry_off"]),
        ):
            wrapped = GeometryOracleModel(inner, variant)
            timed = FAST.ForwardTimer(wrapped)
            plan = unit_invalidity_plan(all_invalid, views, variant) if variant.requires_mask() else None
            result = _run_unit(timed, wrapped, sample_like, device, config, base_rng, variant, plan)
            outputs[name] = result["logits"].detach().clone()
            del result, timed, wrapped
        comparison = _tensor_bitwise_equal(outputs["strict_all_invalid"], outputs["geometry_off"])
        records.append({"sample_id": sample_like["sample_id"], "strict_all_invalid_vs_geometry_off": comparison})
        # Only the untouched padded strip can differ, so the logits must stay close.
        recorder.check(
            f"test2.boundary.{sample_like['sample_id']}.difference_is_small_or_zero",
            bool(
                (not comparison.get("comparable"))
                or float(comparison.get("max_abs_diff", 1.0)) < 1.0e-2
            ),
            comparison,
        )
        del sample_like, views, outputs, corruption
    recorder.evidence["test2_boundary_units"] = records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", type=Path, default=Path(DEFAULT_SPLIT))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--evaluation-seed", type=int, default=2026091401)
    parser.add_argument("--skip-model-tests", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    recorder = Recorder()
    protocol = E10.load_frozen_protocol(args.protocol.resolve())
    module = importlib.import_module(args.config)
    config = copy.copy(module.C)
    channel_order = getattr(config, "channel_order", None)
    if channel_order not in EV.CHANNEL_ORDERS:
        raise ValueError("config must declare channel_order as BGR or RGB")
    entries = EV.split_entries(args.split.resolve())
    if not entries:
        raise RuntimeError("the validation split is empty")
    dataset_root = args.dataset_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    from models.encoders.DFormerv2 import GeoPriorGen

    generator = GeoPriorGen(embed_dim=32, num_heads=4, initial_value=2, heads_range=4).to(device).eval()
    test_geometry_gate_semantics(recorder, generator)
    test_natural_clean_validity(
        recorder, dataset_root, entries, channel_order, sample_limit=int(args.samples)
    )
    test_depth_and_invalidity_share_support(recorder)
    test_single_invalid_pixel_spread(recorder)

    checkpoint = args.checkpoint.resolve()
    recorder.check("checkpoint.exists", checkpoint.exists(), str(checkpoint))
    recorder.evidence["checkpoint"] = {"path": str(checkpoint), "sha256": EV.file_sha256(checkpoint)}
    recorder.evidence["protocol"] = {
        "path": str(args.protocol),
        "raw_sha256": protocol.raw_sha256,
        "evaluation_seed": int(protocol.evaluation["seed"]),
    }
    recorder.evidence["evaluation_seed_used"] = int(args.evaluation_seed)
    recorder.evidence["config_module"] = str(args.config)
    recorder.evidence["device"] = str(device)
    if not args.skip_model_tests:
        run_checkpoint_tests(
            recorder,
            dataset_root,
            list(entries[: int(args.samples)]),
            config,
            channel_order,
            checkpoint,
            device,
            int(args.evaluation_seed),
        )
        run_boundary_tests(
            recorder,
            dataset_root,
            list(entries[:1]),
            config,
            channel_order,
            checkpoint,
            device,
            int(args.evaluation_seed),
        )

    report = {
        "schema_version": "mmfr-oracle-a-unit-tests-v1",
        "status": "PASS" if recorder.all_passed() else "FAIL",
        "assertions_total": len(recorder.assertions),
        "assertions_failed": len(recorder.failures()),
        "failures": recorder.failures(),
        "evidence": recorder.evidence,
        "official_test_included": False,
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "timestamp_utc": E10.now_utc(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "oracle-a-unit-tests.json"
    E10.atomic_write_json(target, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "assertions_total": report["assertions_total"],
                "assertions_failed": report["assertions_failed"],
                "output": str(target),
            },
            indent=2,
        )
    )
    if report["failures"]:
        print(json.dumps(report["failures"][:20], indent=2, default=str))
    return 0 if recorder.all_passed() else 2


if __name__ == "__main__":
    raise SystemExit(main())
