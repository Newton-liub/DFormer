#!/usr/bin/env python3
"""Unified, single-view S1 evaluator for natural MUSeg depth missingness.

This is a new evaluator identity; it is not the legacy Quick-Val protocol. The
entry point scores only the allowlisted val-dev split and its three fixed
conditions. Raw corrupted uint8 Depth observations are persisted once per
sample/condition and reused across every checkpoint in one invocation (and by
later invocations that point at the same output directory).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tools import evaluate_museg_10condition as E10
from tools import evaluate_museg_checkpoint as EV
from tools.mmfr import audit_museg_natural_missing as NMF
from tools.mmfr import e1_quickval as QV

EVALUATOR_IDENTITY = "mmfr-natural-missing-s1-single-view-v1"
SCHEMA_VERSION = "mmfr-natural-missing-s1-results-v1"
PAD_DIVISOR = 32
EVALUATION_SEED = 20261003
P25_MISSING_RATIO = 0.10761048923865359
P75_MISSING_RATIO = 0.4944140559923207
DEFAULT_CONFIG = "local_configs.MUSeg.DFormerv2_S_NaturalMissing"
DEFAULT_SPLIT_ROOT = Path(__file__).resolve().parents[2] / "data" / "splits" / "MUSeg" / "dev-v1"
CONDITION_IDS = (
    "natural_original",
    "rectangle_add50_current_valid",
    "entire_missing",
)
RECTANGLE_FRACTION = 0.5


def _condition_definition(condition_id: str) -> dict[str, Any]:
    definitions = {
        "natural_original": {
            "input": "unaltered original uint8 Depth",
            "added_missing_fraction_target": 0.0,
            "target_basis": "none",
        },
        "rectangle_add50_current_valid": {
            "input": "one axis-aligned rectangle zeroed on original uint8 Depth",
            "added_missing_fraction_target": RECTANGLE_FRACTION,
            "target_basis": "original current-valid Depth pixels (Depth > 0)",
            "rectangle_policy": (
                "choose one deterministic HxW geometry whose expected area is the target pixel "
                "count divided by the observed valid density; scan translations of this declared "
                "fixed geometry and select the placement nearest the target count, with a stable-key tie break"
            ),
            "unreachable_policy": (
                "unreachable means unreachable only among translations of the declared fixed HxW "
                "geometry, not among all possible rectangle geometries; keep the closest placement "
                "and record target/actual counts and target_reachable=false"
            ),
        },
        "entire_missing": {
            "input": "all-zero original uint8 Depth",
            "added_missing_fraction_target": 1.0,
            "target_basis": "entire original image",
        },
    }
    try:
        return definitions[condition_id]
    except KeyError as exc:
        raise ValueError(f"unsupported S1 condition: {condition_id!r}") from exc


def _stable_seed(sample_id: str, condition_id: str) -> int:
    identity = f"{EVALUATOR_IDENTITY}\0{EVALUATION_SEED}\0{sample_id}\0{condition_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


@contextlib.contextmanager
def _fixed_forward_rng(sample_id: str, condition_id: str, device: torch.device) -> Iterator[int]:
    """Restore one checkpoint-independent Python/NumPy/Torch RNG identity per unit."""
    seed = _stable_seed(sample_id, condition_id)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices: list[int] = []
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA forward RNG requested but CUDA is unavailable")
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            torch.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield seed
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def pad_view_right_bottom(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    divisor: int = PAD_DIVISOR,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Zero-pad normalized BxCxHxW inputs on the right and bottom only."""
    if rgb.ndim != 4 or depth.ndim != 4 or rgb.shape[0] != 1 or depth.shape[0] != 1:
        raise ValueError("S1 requires batch-one 4-D RGB and Depth tensors")
    if rgb.shape[-2:] != depth.shape[-2:]:
        raise ValueError("RGB and Depth must share the original spatial grid")
    if divisor <= 0:
        raise ValueError("padding divisor must be positive")
    height, width = (int(value) for value in rgb.shape[-2:])
    padded_height = math.ceil(height / divisor) * divisor
    padded_width = math.ceil(width / divisor) * divisor
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    padding = (0, pad_right, 0, pad_bottom)
    rgb_padded = F.pad(rgb, padding, mode="constant", value=0.0)
    depth_padded = F.pad(depth, padding, mode="constant", value=0.0)
    return rgb_padded, depth_padded, {
        "original_height": height,
        "original_width": width,
        "padded_height": padded_height,
        "padded_width": padded_width,
        "pad_bottom": pad_bottom,
        "pad_right": pad_right,
    }


def crop_logits_to_original_grid(
    logits: torch.Tensor,
    padding: Mapping[str, int],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Restore logits to the padded grid, remove pad, then align to original labels."""
    if logits.ndim != 4:
        raise ValueError("S1 model forward must return a BxClassesxHxW logits tensor")
    padded_size = (int(padding["padded_height"]), int(padding["padded_width"]))
    original_size = (int(padding["original_height"]), int(padding["original_width"]))
    if tuple(int(value) for value in labels.shape[-2:]) != original_size:
        raise ValueError("labels are not on the original unpadded grid")
    padded_labels = torch.empty(
        (labels.shape[0], padded_size[0], padded_size[1]),
        dtype=labels.dtype,
        device=labels.device,
    )
    logits_on_padded_grid = EV.restore_logits_to_metric_grid(logits, padded_labels)
    logits_cropped = logits_on_padded_grid[..., : original_size[0], : original_size[1]].contiguous()
    return EV.restore_logits_to_metric_grid(logits_cropped, labels)


def make_rectangle_observation(
    original_depth: np.ndarray,
    sample_id: str,
    fraction: float = RECTANGLE_FRACTION,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Zero one deterministic rectangle to remove the requested share of valid Depth."""
    if original_depth.ndim != 2 or original_depth.dtype != np.uint8:
        raise ValueError("rectangle condition requires a 2-D uint8 original Depth image")
    if not math.isfinite(float(fraction)) or not 0.0 <= fraction <= 1.0:
        raise ValueError("rectangle deletion fraction must lie in [0, 1]")
    height, width = (int(value) for value in original_depth.shape)
    current_valid = original_depth > 0
    valid_count = int(np.count_nonzero(current_valid))
    target_count = int(math.floor(valid_count * float(fraction) + 0.5))
    base: dict[str, Any] = {
        "target_fraction_of_current_valid": float(fraction),
        "current_valid_pixels": valid_count,
        "target_added_missing_pixels": target_count,
        "rectangle_bounds_yxyx": None,
        "rectangle_geometry_hw": None,
        "actual_added_missing_pixels": 0,
        "actual_added_fraction_of_current_valid": 0.0,
        "target_fraction_defined": valid_count > 0,
        "actual_fraction_defined": valid_count > 0,
        "target_reachable": valid_count > 0 and target_count == 0,
        "unreachable_semantics": None,
    }
    corrupted = original_depth.copy()
    if valid_count == 0:
        base["target_reachable"] = False
        base["unreachable_semantics"] = (
            "no_current_valid_pixels; the requested added fraction is undefined and the rectangle is an explicit no-op"
        )
        return corrupted, base
    if target_count == 0:
        base["unreachable_semantics"] = "rounded target count is zero; the rectangle is an explicit no-op"
        return corrupted, base

    valid_density = valid_count / float(height * width)
    desired_area = target_count / valid_density
    rect_height = max(1, min(height, int(math.floor(math.sqrt(desired_area * height / width) + 0.5))))
    rect_width = max(1, int(math.floor(desired_area / rect_height + 0.5)))
    if rect_width > width:
        rect_width = width
        rect_height = max(1, min(height, int(math.floor(desired_area / rect_width + 0.5))))
    if rect_height > height:
        rect_height = height
        rect_width = max(1, min(width, int(math.floor(desired_area / rect_height + 0.5))))
    rect_width = min(rect_width, width)

    integral = np.pad(
        current_valid.astype(np.int64).cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    counts = (
        integral[rect_height:, rect_width:]
        - integral[:-rect_height, rect_width:]
        - integral[rect_height:, :-rect_width]
        + integral[:-rect_height, :-rect_width]
    )
    errors = np.abs(counts - target_count)
    minimum_error = int(errors.min())
    candidate_indices = np.flatnonzero(errors.ravel() == minimum_error)
    tie_index = _stable_seed(sample_id, "rectangle_add50_current_valid") % int(candidate_indices.size)
    flat_index = int(candidate_indices[tie_index])
    candidates_per_row = int(errors.shape[1])
    top = flat_index // candidates_per_row
    left = flat_index % candidates_per_row
    actual_count = int(counts[top, left])
    bottom = top + rect_height
    right = left + rect_width
    corrupted[top:bottom, left:right] = 0
    base.update(
        {
            "rectangle_bounds_yxyx": [int(top), int(bottom), int(left), int(right)],
            "rectangle_geometry_hw": [rect_height, rect_width],
            "actual_added_missing_pixels": actual_count,
            "actual_added_fraction_of_current_valid": actual_count / float(valid_count),
            "target_reachable": actual_count == target_count,
            "unreachable_semantics": (
                None
                if actual_count == target_count
                else (
                    "exact rounded target count is not reachable by any translation of this declared fixed HxW "
                    "rectangle geometry; this does not claim all possible rectangles are unreachable; "
                    "select the closest valid-count placement and record error"
                )
            ),
        }
    )
    return corrupted, base


def build_condition_observation(
    original_depth: np.ndarray,
    sample_id: str,
    condition_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create one raw uint8 Depth observation and its deletion record."""
    if original_depth.ndim != 2 or original_depth.dtype != np.uint8:
        raise ValueError("S1 conditions require a 2-D uint8 original Depth image")
    valid_count = int(np.count_nonzero(original_depth > 0))
    pixels = int(original_depth.size)
    if condition_id == "natural_original":
        depth = original_depth.copy()
        detail: dict[str, Any] = {
            "input_valid_pixels": valid_count,
            "added_missing_pixels": 0,
            "added_fraction_of_original_current_valid": 0.0,
            "target_added_fraction": 0.0,
            "target_reachable": True,
            "rectangle_bounds_yxyx": None,
        }
    elif condition_id == "rectangle_add50_current_valid":
        depth, detail = make_rectangle_observation(original_depth, sample_id, RECTANGLE_FRACTION)
        detail["input_valid_pixels"] = valid_count
        detail["added_missing_pixels"] = detail["actual_added_missing_pixels"]
        detail["added_fraction_of_original_current_valid"] = detail[
            "actual_added_fraction_of_current_valid"
        ]
        detail["added_fraction_defined"] = detail["actual_fraction_defined"]
        detail["target_added_fraction"] = RECTANGLE_FRACTION
    elif condition_id == "entire_missing":
        depth = np.zeros_like(original_depth)
        detail = {
            "input_valid_pixels": valid_count,
            "added_missing_pixels": valid_count,
            "added_fraction_of_original_current_valid": 1.0 if valid_count else 0.0,
            "added_fraction_defined": valid_count > 0,
            "target_added_fraction": 1.0,
            "target_reachable": True,
            "rectangle_bounds_yxyx": None,
        }
    else:
        raise ValueError(f"unsupported S1 condition: {condition_id!r}")
    detail.update(
        {
            "condition_id": condition_id,
            "original_missing_pixels": pixels - valid_count,
            "original_missing_ratio": (pixels - valid_count) / float(pixels),
        }
    )
    return depth, detail


def _load_allowlisted_val_dev() -> tuple[list[tuple[str, str, str]], Path, str]:
    """Read exactly the registered val-dev allowlist; no split path is user-selectable."""
    split_root = DEFAULT_SPLIT_ROOT.resolve()
    allowlist_source = split_root / "val-dev.txt"
    allowlist = _validate_split_path(allowlist_source)
    if allowlist.parent != split_root:
        raise ValueError("refusing a non-canonical val-dev allowlist path")
    entries = NMF._read_allowlist(split_root, "val-dev")
    if len(entries) != int(NMF.EXPECTED_IMAGES["val-dev"]):
        raise ValueError("val-dev allowlist count differs from the registered 318-sample identity")
    return entries, allowlist, EV.file_sha256(allowlist)


def _read_original_depth(dataset_root: Path, sample_id: str) -> np.ndarray:
    path = dataset_root / "Depth" / f"{sample_id}.png"
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"cannot read allowlisted val-dev Depth image: {path}")
    if depth.dtype != np.uint8 or depth.ndim != 2:
        raise ValueError(f"expected original 2-D uint8 Depth for {sample_id}, got {depth.shape}/{depth.dtype}")
    if tuple(depth.shape) != tuple(NMF.EXPECTED_SHAPE_HW):
        raise ValueError(
            f"expected original MUSeg grid {NMF.EXPECTED_SHAPE_HW} for {sample_id}, got {depth.shape}"
        )
    return depth


def _prepare_observations(
    selected_entries: Sequence[tuple[str, str, str]],
    dataset_root: Path,
) -> dict[str, dict[str, tuple[np.ndarray, dict[str, Any]]]]:
    """Build raw observations once and retain them in memory across checkpoints."""
    observations: dict[str, dict[str, tuple[np.ndarray, dict[str, Any]]]] = {}
    for filename, _mine, _group_id in selected_entries:
        sample_id = Path(filename).stem
        original_depth = _read_original_depth(dataset_root, sample_id)
        observations[sample_id] = {
            condition_id: build_condition_observation(original_depth, sample_id, condition_id)
            for condition_id in CONDITION_IDS
        }
    return observations


def _pad_and_forward(
    model: torch.nn.Module,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_id: str,
    condition_id: str,
    device: torch.device,
    rng_counts: dict[str, int],
) -> tuple[torch.Tensor, dict[str, int], int]:
    rgb_padded, depth_padded, padding = pad_view_right_bottom(rgb, depth)
    rng_counts["forward_calls"] = rng_counts.get("forward_calls", 0) + 1
    rng_counts["rng_state_resets_before_forward"] = rng_counts.get("rng_state_resets_before_forward", 0) + 1
    seed = _stable_seed(sample_id, condition_id)
    with _fixed_forward_rng(sample_id, condition_id, device):
        with torch.inference_mode():
            logits = model(rgb_padded, depth_padded)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("segmentation model forward must return one logits tensor")
            logits = crop_logits_to_original_grid(logits, padding, labels)
    if tuple(int(value) for value in logits.shape[-2:]) != tuple(int(value) for value in labels.shape[-2:]):
        raise RuntimeError("S1 logits did not return to the original label grid")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError(f"non-finite S1 logits for {sample_id}/{condition_id}")
    return logits, padding, seed


def _natural_missing_profile(
    original_depth: np.ndarray,
    label: np.ndarray,
    num_classes: int,
) -> tuple[float | None, int, int, str]:
    """Keep frozen full-image strata; label-region missingness is diagnostic only."""
    if original_depth.shape != label.shape:
        raise ValueError("original Depth and labels must share one unpadded spatial grid")
    full_image_ratio = float(np.count_nonzero(original_depth == 0)) / original_depth.size
    stratum = NMF._stratum(full_image_ratio, P25_MISSING_RATIO, P75_MISSING_RATIO)
    valid_label = (label >= 0) & (label < num_classes)
    valid_label_pixels = int(np.count_nonzero(valid_label))
    missing_pixels = int(np.count_nonzero((original_depth == 0) & valid_label))
    ratio = missing_pixels / float(valid_label_pixels) if valid_label_pixels else None
    return ratio, valid_label_pixels, missing_pixels, stratum


def _empty_histograms(num_classes: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        condition_id: {
            stratum: np.zeros((num_classes, num_classes), dtype=np.int64)
            for stratum in ("all", "low", "high")
        }
        for condition_id in CONDITION_IDS
    }


def _run_checkpoint(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    config: Any,
    config_name: str,
    runtime: Any,
    selected_entries: Sequence[tuple[str, str, str]],
    dataset_root: Path,
    observations: Mapping[str, Mapping[str, tuple[np.ndarray, dict[str, Any]]]],
    channel_order: str,
    split_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    num_classes = int(config.num_classes)
    class_names = list(config.class_names)
    ignore_label = int(config.background)
    if num_classes <= 0 or len(class_names) != num_classes:
        raise ValueError("NaturalMissing config must define matching num_classes and class_names")
    if ignore_label < 0 or ignore_label < num_classes:
        raise ValueError("NaturalMissing config background must be a valid ignore label")

    model = runtime.build_model(config, device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("utils.natural_missing_runtime.build_model must return a torch.nn.Module")
    restore_result = runtime.load_segmentation_checkpoint(model, checkpoint)
    if not isinstance(restore_result, Mapping):
        raise TypeError("load_segmentation_checkpoint must return its restore summary mapping")
    required_restore_fields = {"loaded", "intentionally_dropped", "missing", "unexpected"}
    if not required_restore_fields.issubset(restore_result):
        raise ValueError("checkpoint restore summary omitted a required strict-load field")
    if restore_result.get("missing") or restore_result.get("unexpected"):
        raise RuntimeError("segmentation checkpoint restore reported missing or unexpected state keys")
    model.to(device=device, dtype=torch.float32).eval()

    histograms = _empty_histograms(num_classes)
    group_histograms: dict[str, dict[str, dict[str, Any]]] = {
        condition_id: {} for condition_id in CONDITION_IDS
    }
    sample_observations: dict[str, list[dict[str, Any]]] = {condition_id: [] for condition_id in CONDITION_IDS}
    stratum_counts = {name: 0 for name in ("all", "low", "medium", "high", "unsupported")}
    rng_counts = {"forward_calls": 0, "rng_state_resets_before_forward": 0}
    padding_examples: list[dict[str, Any]] = []

    for filename, _mine, group_id in selected_entries:
        entry = f"RGB/{filename}"
        sample_id, rgb_u8, original_depth, label = E10.read_sample(dataset_root, entry, channel_order)
        if sample_id != Path(filename).stem:
            raise RuntimeError("sample reader returned a different identity than the val-dev allowlist")
        cached_original = observations.get(sample_id, {}).get("natural_original")
        if cached_original is None or not np.array_equal(cached_original[0], original_depth):
            raise RuntimeError(f"original Depth changed during this invocation for {sample_id}")
        natural_missing_ratio, valid_label_pixels, missing_in_valid_labels, natural_stratum = (
            _natural_missing_profile(original_depth, label, num_classes)
        )
        stratum_counts["all"] += 1
        stratum_counts[natural_stratum] += 1
        sample_conditions = observations.get(sample_id)
        if sample_conditions is None:
            raise RuntimeError(f"in-memory observations are missing for {sample_id}")

        for condition_id in CONDITION_IDS:
            group_record = group_histograms[condition_id].setdefault(
                group_id,
                {
                    "sample_count_by_stratum": {name: 0 for name in ("all", "low", "medium", "high", "unsupported")},
                    "valid_label_support": {name: 0 for name in ("all", "low", "medium", "high", "unsupported")},
                    "confusion": {
                        stratum: np.zeros((num_classes, num_classes), dtype=np.int64)
                        for stratum in ("all", "low", "high")
                    },
                },
            )
            group_record["sample_count_by_stratum"]["all"] += 1
            group_record["sample_count_by_stratum"][natural_stratum] += 1
            group_record["valid_label_support"]["all"] += valid_label_pixels
            group_record["valid_label_support"][natural_stratum] += valid_label_pixels
            depth_observed, observation_detail = sample_conditions[condition_id]
            rgb_tensor, depth_tensor, label_tensor = QV.build_original_full_input(
                rgb_u8,
                depth_observed,
                label,
                config.norm_mean,
                config.norm_std,
                device,
            )
            logits, padding, seed = _pad_and_forward(
                model,
                rgb_tensor,
                depth_tensor,
                label_tensor,
                sample_id=sample_id,
                condition_id=condition_id,
                device=device,
                rng_counts=rng_counts,
            )
            EV.update_confusion(histograms[condition_id]["all"], logits, label_tensor, num_classes, ignore_label)
            EV.update_confusion(group_record["confusion"]["all"], logits, label_tensor, num_classes, ignore_label)
            if natural_stratum in ("low", "high"):
                EV.update_confusion(
                    histograms[condition_id][natural_stratum], logits, label_tensor, num_classes, ignore_label
                )
                EV.update_confusion(
                    group_record["confusion"][natural_stratum], logits, label_tensor, num_classes, ignore_label
                )
            if len(padding_examples) < 3:
                padding_examples.append({"sample_id": sample_id, "condition_id": condition_id, **padding})
            sample_observations[condition_id].append(
                {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "natural_missing_ratio_full_image": float(np.count_nonzero(original_depth == 0)) / original_depth.size,
                    "natural_missing_ratio_within_valid_labels": natural_missing_ratio,
                    "natural_stratum": natural_stratum,
                    "valid_label_pixels": valid_label_pixels,
                    "missing_depth_pixels_within_valid_labels": missing_in_valid_labels,
                    "forward_rng_seed": seed,
                    "added_missing_pixels": int(observation_detail["added_missing_pixels"]),
                    "added_fraction_of_original_current_valid": float(
                        observation_detail["added_fraction_of_original_current_valid"]
                    ),
                    "added_fraction_defined": bool(observation_detail.get("added_fraction_defined", True)),
                    "target_reachable": bool(observation_detail["target_reachable"]),
                    "rectangle_bounds_yxyx": observation_detail.get("rectangle_bounds_yxyx"),
                    "rectangle_geometry_hw": observation_detail.get("rectangle_geometry_hw"),
                    "unreachable_semantics": observation_detail.get("unreachable_semantics"),
                }
            )
            del rgb_tensor, depth_tensor, label_tensor, logits

    condition_results: dict[str, Any] = {}
    for condition_id in CONDITION_IDS:
        condition_results[condition_id] = {
            "definition": _condition_definition(condition_id),
            "sample_count": len(sample_observations[condition_id]),
            "metrics_percent_by_stratum": {
                stratum: EV.metrics_from_confusion(histograms[condition_id][stratum], class_names)
                for stratum in ("all", "low", "high")
            },
            "sample_observations": sample_observations[condition_id],
        }

    group_results: dict[str, Any] = {}
    for condition_id in CONDITION_IDS:
        condition_groups: dict[str, Any] = {}
        for group_id in sorted(group_histograms[condition_id]):
            group = group_histograms[condition_id][group_id]
            condition_groups[group_id] = {
                "sample_count_by_stratum": group["sample_count_by_stratum"],
                "valid_label_support_pixels_by_stratum": group["valid_label_support"],
                "confusion_matrix_by_stratum": {
                    stratum: group["confusion"][stratum].tolist() for stratum in ("all", "low", "high")
                },
                "class_target_pixels_by_stratum": {
                    stratum: group["confusion"][stratum].sum(axis=1).astype(np.int64).tolist()
                    for stratum in ("all", "low", "high")
                },
            }
        group_results[condition_id] = condition_groups

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "status": "completed",
        "official_test_included": False,
        "identity": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "config_module": config_name,
            "split_role": "val_dev",
            "split_path": str((DEFAULT_SPLIT_ROOT / "val-dev.txt").resolve()),
            "split_sha256": split_sha256,
            "dataset_root": str(dataset_root.resolve()),
            "sample_count": len(selected_entries),
            "channel_order": channel_order,
            "normalization_identity": str(getattr(config, "normalization_identity", "unspecified")),
            "depth_normalization": "uint8 Depth / 255, then (value - 0.48) / 0.28; replicate to three channels",
            "forward_precision": "FP32",
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "batch_size": 1,
            "scale": 1.0,
            "flip": False,
            "geometry": "whole-image; right/bottom zero pad to multiple of 32; crop before metric",
            "padding_contract": {
                "divisor": PAD_DIVISOR,
                "sides": ["right", "bottom"],
                "normalized_fill_value": 0.0,
                "crop_before_metric": True,
            },
            "metric_grid": "original label grid",
        },
        "checkpoint_restore": dict(restore_result),
        "class_policy": {
            "num_classes": num_classes,
            "class_names": class_names,
            "ignore_label": ignore_label,
            "unsupported_class_policy": (
                "all configured classes remain in the mean; shared EV.metrics_from_confusion assigns "
                "zero IoU when the class union is zero and does not drop classes without GT support"
            ),
        },
        "natural_missing_strata": {
            "definition": (
                "frozen strata use count(Depth == 0) / original unpadded image size, matching the train audit; "
                "missingness within valid labels is an additional diagnostic and never changes strata"
            ),
            "low_inclusive_max": P25_MISSING_RATIO,
            "high_strict_min": P75_MISSING_RATIO,
            "classify": "shared audit _stratum: low <= train P25; high > train P75; medium otherwise",
            "sample_count": stratum_counts,
            "collection_group_count_by_stratum": {
                stratum: sum(
                    group["sample_count_by_stratum"][stratum] > 0
                    for group in group_histograms[CONDITION_IDS[0]].values()
                )
                for stratum in ("all", "low", "medium", "high", "unsupported")
            },
            "unsupported_policy": "no valid-label pixels means diagnostic ratio undefined; frozen full-image stratum is retained",
        },
        "s1_conditions": condition_results,
        "collection_group_statistics": group_results,
        "raw_observations": {
            "storage": "in-memory only",
            "shared_across_checkpoints_in_this_invocation": True,
            "persisted_observation_cache": False,
        },
        "forward_rng_policy": {
            "policy": "stable sample_id × condition_id SHA-256-derived seed restored before every forward",
            "seed_base": EVALUATION_SEED,
            "checkpoint_independent": True,
            "independent_of_checkpoint_order_and_filesystem_order": True,
            "unit": "sample × condition",
            "counts": {
                **rng_counts,
                "unique_unit_keys": len(selected_entries) * len(CONDITION_IDS),
            },
        },
        "padding_examples": padding_examples,
    }


def _load_natural_config(config_name: str) -> tuple[Any, Any]:
    config_module = importlib.import_module(config_name)
    factory = getattr(config_module, "make_config", None)
    if not callable(factory):
        raise AttributeError(f"{config_name} must expose make_config(strategy='Natural')")
    config = factory(strategy="Natural")
    runtime = importlib.import_module("utils.natural_missing_runtime")
    if not callable(getattr(runtime, "build_model", None)):
        raise AttributeError("utils.natural_missing_runtime must expose build_model(config, device)")
    if not callable(getattr(runtime, "load_segmentation_checkpoint", None)):
        raise AttributeError(
            "utils.natural_missing_runtime must expose load_segmentation_checkpoint(model, checkpoint_path)"
        )
    return config, runtime


def run_evaluation(
    *,
    dataset_root: Path,
    checkpoints: Sequence[Path],
    output_dir: Path,
    config_name: str = DEFAULT_CONFIG,
    sample_limit: int = 2,
    formal_dev_authorization: bool = False,
    device_name: str = "cuda",
    verified_checkpoint_sha256: Mapping[Path, str] | None = None,
) -> list[Path]:
    """Evaluate an explicitly limited val-dev prefix; full runs require authorization."""
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")
    if sample_limit > 2 and not formal_dev_authorization:
        raise ValueError("evaluating more than the tiny default requires --formal-dev-authorization")
    allowlisted, allowlist_path, allowlist_sha256 = _load_allowlisted_val_dev()
    if sample_limit > len(allowlisted):
        raise ValueError(f"sample_limit {sample_limit} exceeds the val-dev allowlist size {len(allowlisted)}")
    selected_entries = allowlisted[:sample_limit]
    # Recovery owners may supply an already verified identity to avoid re-hashing
    # the same immutable checkpoint. Other checkpoints retain direct hashing.
    verified_identities = {
        Path(path).resolve(): str(sha256).lower()
        for path, sha256 in (verified_checkpoint_sha256 or {}).items()
    }
    checkpoint_paths = {Path(path).resolve() for path in checkpoints}
    if not set(verified_identities).issubset(checkpoint_paths):
        raise ValueError("verified identity refers to a checkpoint outside this invocation")
    if any(re.fullmatch(r"[0-9a-f]{64}", sha256) is None for sha256 in verified_identities.values()):
        raise ValueError("externally verified checkpoint identity must be a SHA-256 hex digest")
    dataset_root = dataset_root.resolve()
    if any("official" in part.casefold() or "test" in part.casefold() for part in dataset_root.parts):
        raise ValueError("refusing a dataset root identified as official-test/test data")
    output_dir.mkdir(parents=True, exist_ok=True)

    config, runtime = _load_natural_config(config_name)
    configured_splits = getattr(config, "expected_split_sha256", {})
    expected_val_sha256 = configured_splits.get("val") if isinstance(configured_splits, Mapping) else None
    if expected_val_sha256 and str(expected_val_sha256).lower() != allowlist_sha256.lower():
        raise ValueError("NaturalMissing config val-dev split identity differs from the canonical allowlist")
    channel_order = str(getattr(config, "channel_order", "RGB"))
    if channel_order not in EV.CHANNEL_ORDERS:
        raise ValueError("NaturalMissing config channel_order must be RGB or BGR")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    EV.configure_fp32_forward(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    observations = _prepare_observations(selected_entries, dataset_root)
    result_paths: list[Path] = []
    for checkpoint in checkpoints:
        checkpoint = checkpoint.resolve()
        externally_verified = checkpoint in verified_identities
        checkpoint_sha256 = verified_identities[checkpoint] if externally_verified else EV.file_sha256(checkpoint)
        report = _run_checkpoint(
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            config=config,
            config_name=config_name,
            runtime=runtime,
            selected_entries=selected_entries,
            dataset_root=dataset_root,
            observations=observations,
            channel_order=channel_order,
            split_sha256=allowlist_sha256,
            device=device,
        )
        report["identity"]["checkpoint_sha256_verification"] = (
            "externally verified by recovery owner; not re-hashed in this invocation"
            if externally_verified else "computed from checkpoint in this invocation"
        )
        report["identity"]["allowlist_path"] = str(allowlist_path)
        report["identity"]["allowlist_sha256"] = allowlist_sha256
        report["identity"]["authorization_scope"] = (
            "formal val-dev subset explicitly authorized" if formal_dev_authorization else "tiny CPU/GPU preflight scope"
        )
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint.stem)
        report_path = output_dir / "checkpoints" / f"{safe_stem}-{checkpoint_sha256[:12]}.json"
        E10.atomic_write_json(report_path, report)
        result_paths.append(report_path)
    return result_paths


def _run_cpu_self_check() -> dict[str, Any]:
    """Exercise S1 contracts with synthetic CPU inputs only."""
    synthetic_depth = np.full((10, 10), 100, dtype=np.uint8)
    synthetic_depth[:2, :4] = 0
    rect_a, rect_record_a = make_rectangle_observation(synthetic_depth, "01-01-01-01-001-001-001")
    rect_b, rect_record_b = make_rectangle_observation(synthetic_depth, "01-01-01-01-001-001-001")
    if not np.array_equal(rect_a, rect_b) or rect_record_a != rect_record_b:
        raise AssertionError("rectangle observation is not deterministic")
    valid_before = int(np.count_nonzero(synthetic_depth > 0))
    actual_added = int(np.count_nonzero((synthetic_depth > 0) & (rect_a == 0)))
    if actual_added != int(rect_record_a["actual_added_missing_pixels"]):
        raise AssertionError("rectangle telemetry does not match the raw observation")
    if rect_record_a["target_reachable"] != (actual_added == rect_record_a["target_added_missing_pixels"]):
        raise AssertionError("rectangle reachability telemetry is inconsistent")
    if rect_record_a["unreachable_semantics"] is not None and "fixed HxW" not in rect_record_a["unreachable_semantics"]:
        raise AssertionError("rectangle unreachable scope is not limited to its declared fixed geometry")
    _, fixed_geometry_unreachable = make_rectangle_observation(
        np.full((10, 10), 100, dtype=np.uint8), "01-01-01-01-001-001-001"
    )
    if fixed_geometry_unreachable["target_reachable"] or "fixed HxW" not in fixed_geometry_unreachable["unreachable_semantics"]:
        raise AssertionError("unreachable target was not scoped to the declared fixed HxW geometry")
    empty_depth = np.zeros((10, 10), dtype=np.uint8)
    _, empty_record = make_rectangle_observation(empty_depth, "01-01-01-01-001-001-001")
    if empty_record["unreachable_semantics"] is None:
        raise AssertionError("zero-current-valid rectangle semantics were not recorded")
    natural, natural_record = build_condition_observation(
        synthetic_depth, "01-01-01-01-001-001-001", "natural_original"
    )
    if natural_record["original_missing_ratio"] != float(np.count_nonzero(synthetic_depth == 0)) / synthetic_depth.size:
        raise AssertionError("natural missing ratio included any padded pixels")
    entire, entire_record = build_condition_observation(
        synthetic_depth, "01-01-01-01-001-001-001", "entire_missing"
    )
    if not np.array_equal(natural, synthetic_depth) or np.count_nonzero(entire) != 0:
        raise AssertionError("natural or entire-missing condition changed semantics")
    if entire_record["added_missing_pixels"] != valid_before:
        raise AssertionError("entire-missing telemetry does not equal current-valid pixels")

    valid_label_grid = np.array([[0, 1, 255], [2, 0, 255]], dtype=np.uint8)
    depth_grid = np.array([[0, 0, 100], [100, 0, 0]], dtype=np.uint8)
    ratio, support, missing_support, stratum = _natural_missing_profile(depth_grid, valid_label_grid, 3)
    if (ratio, support, missing_support, stratum) != (0.75, 4, 3, "high"):
        raise AssertionError("natural missingness was not measured within valid-label support")

    sample_id = "01-01-01-01-001-001-001"
    in_memory = {
        sample_id: {
            condition_id: build_condition_observation(synthetic_depth, sample_id, condition_id)
            for condition_id in CONDITION_IDS
        }
    }
    checkpoint_one = in_memory[sample_id]["rectangle_add50_current_valid"][0]
    checkpoint_two = in_memory[sample_id]["rectangle_add50_current_valid"][0]
    if checkpoint_one is not checkpoint_two or len(in_memory[sample_id]) != len(CONDITION_IDS):
        raise AssertionError("raw observations were not retained in memory for checkpoint reuse")

    rgb = torch.zeros((1, 3, 37, 65), dtype=torch.float32)
    depth = torch.zeros_like(rgb)
    label = torch.zeros((1, 37, 65), dtype=torch.int64)
    rgb_padded, depth_padded, padding = pad_view_right_bottom(rgb, depth)
    if tuple(rgb_padded.shape[-2:]) != (64, 96) or tuple(depth_padded.shape[-2:]) != (64, 96):
        raise AssertionError("S1 right/bottom pad-to-32 geometry is incorrect")

    class _SyntheticModel(torch.nn.Module):
        def forward(self, input_rgb: torch.Tensor, input_depth: torch.Tensor) -> torch.Tensor:
            if input_rgb.shape[-2:] != (64, 96) or input_depth.shape[-2:] != (64, 96):
                raise AssertionError("dummy model did not receive the padded original-scale grid")
            return torch.rand((1, 15, 32, 48), dtype=torch.float32, device=input_rgb.device)

    model = _SyntheticModel().eval()
    rng_counts: dict[str, int] = {}
    logits_a, _, seed_a = _pad_and_forward(
        model,
        rgb,
        depth,
        label,
        sample_id=sample_id,
        condition_id="natural_original",
        device=torch.device("cpu"),
        rng_counts=rng_counts,
    )
    logits_b, _, seed_b = _pad_and_forward(
        model,
        rgb,
        depth,
        label,
        sample_id=sample_id,
        condition_id="natural_original",
        device=torch.device("cpu"),
        rng_counts=rng_counts,
    )
    if logits_a.shape != (1, 15, 37, 65) or not torch.equal(logits_a, logits_b) or seed_a != seed_b:
        raise AssertionError("S1 crop-to-label-grid or keyed RNG replay failed")
    if rng_counts != {"forward_calls": 2, "rng_state_resets_before_forward": 2}:
        raise AssertionError("RNG policy counters do not match the dummy forward count")

    metric_logits = torch.full((1, 3, 1, 3), -5.0, dtype=torch.float32)
    metric_logits[0, 0, 0, 0] = 5.0
    metric_logits[0, 1, 0, 1] = 5.0
    metric_logits[0, 2, 0, 2] = 5.0
    metric_labels = torch.tensor([[[0, 1, 255]]], dtype=torch.int64)
    metric_histogram = np.zeros((3, 3), dtype=np.int64)
    EV.update_confusion(metric_histogram, metric_logits, metric_labels, 3, 255)
    shared_metrics = EV.metrics_from_confusion(metric_histogram, ["a", "b", "unsupported"])
    if shared_metrics["confusion_matrix"] != [[1, 0, 0], [0, 1, 0], [0, 0, 0]]:
        raise AssertionError("shared dataset confusion helper did not ignore label 255")
    if len(shared_metrics["per_class"]) != 3 or shared_metrics["per_class"][2]["target_pixels"] != 0:
        raise AssertionError("shared mIoU helper did not preserve unsupported classes")
    if shared_metrics["per_class"][2]["iou"] != 0.0 or shared_metrics["miou"] != 66.67:
        raise AssertionError("shared mIoU helper's unsupported-class inclusion policy changed")

    rejected_test_split = False
    try:
        _validate_split_path(Path("data/splits/MUSeg/test.txt"))
    except ValueError:
        rejected_test_split = True
    if not rejected_test_split:
        raise AssertionError("official-test split path was not rejected")
    return {
        "status": "passed",
        "checks": [
            "deterministic rectangle observation and reachability limited to declared fixed HxW geometry",
            "natural and entire-missing observation definitions and valid-label-only missingness",

            "in-memory observation reuse across checkpoint forwards without persistence",
            "right/bottom pad-to-32, padded-grid logits restore, pad crop, original-label-grid shape",
            "stable sample-condition RNG replay and policy counters",
            "shared dataset confusion/mIoU and unsupported-class convention",
            "test split identity refusal without reading a split file",
        ],
        "synthetic_original_hw": [37, 65],
        "synthetic_padded_hw": [padding["padded_height"], padding["padded_width"]],
        "synthetic_rectangle_target_pixels": rect_record_a["target_added_missing_pixels"],
        "synthetic_rectangle_actual_pixels": actual_added,
        "synthetic_rectangle_target_reachable": rect_record_a["target_reachable"],
    }


def _validate_split_path(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("symlinked split allowlists are not permitted")
    resolved = path.resolve()
    expected = (DEFAULT_SPLIT_ROOT / "val-dev.txt").resolve()
    if resolved != expected or resolved.name != "val-dev.txt":
        raise ValueError("only the canonical data/splits/MUSeg/dev-v1/val-dev.txt allowlist is permitted")
    return expected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", help="repeat to score matched checkpoints")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--verified-c0-sha256",
        help="exact C0 identity already verified by recovery owner; one checkpoint only, no re-hash",
    )
    parser.add_argument("--sample-limit", type=int, default=2, help="defaults to a tiny allowlisted val-dev subset")
    parser.add_argument(
        "--formal-dev-authorization",
        action="store_true",
        help="explicitly authorize more than the default two val-dev samples; never enables official test",
    )
    parser.add_argument("--self-check", action="store_true", help="run only CPU synthetic checks; no real model/data")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        print(json.dumps(_run_cpu_self_check(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    missing = [name for name in ("dataset_root", "output_dir") if getattr(args, name) is None]
    if not args.checkpoint:
        missing.append("checkpoint")
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    verified_identities = None
    if args.verified_c0_sha256 is not None:
        if len(args.checkpoint) != 1 or args.verified_c0_sha256.lower() != (
            "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a"
        ):
            raise SystemExit("--verified-c0-sha256 requires one checkpoint and the frozen exact C0 identity")
        verified_identities = {args.checkpoint[0]: args.verified_c0_sha256}
    try:
        paths = run_evaluation(
            dataset_root=args.dataset_root,
            checkpoints=args.checkpoint,
            output_dir=args.output_dir,
            config_name=args.config,
            sample_limit=args.sample_limit,
            formal_dev_authorization=args.formal_dev_authorization,
            device_name=args.device,
            verified_checkpoint_sha256=verified_identities,
        )
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError, ModuleNotFoundError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "completed", "result_files": [str(path.resolve()) for path in paths]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
