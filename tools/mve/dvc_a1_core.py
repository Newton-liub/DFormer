#!/usr/bin/env python3
"""Pure DVC-A1 corruption, Boundary IoU, aggregation, and adjudication logic."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from tools.prepare_museg import quantize_depth

CONDITION_IDS = (
    "clean",
    "boundary-q25",
    "boundary-q50",
    "boundary-q75",
    "nonboundary-q50",
)
DOSES = {
    "clean": 0.0,
    "boundary-q25": 0.25,
    "boundary-q50": 0.50,
    "boundary-q75": 0.75,
    "nonboundary-q50": 0.50,
}
_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class CorruptionMasks:
    """Reconstructable masks and audit metadata for one raw depth image."""

    masks: Mapping[str, np.ndarray]
    boundary_candidate_count: int
    nonboundary_candidate_count: int
    condition_counts: Mapping[str, int]
    condition_sha256: Mapping[str, str]


def location_group(sample_id: str) -> str:
    parts = sample_id.split("-")
    if len(parts) < 4 or any(not part for part in parts[:4]):
        raise ValueError(f"sample id cannot form a location group: {sample_id!r}")
    return "-".join(parts[:4])


def mine_id(sample_id: str) -> str:
    mine = sample_id.split("-", 1)[0]
    if mine not in {"01", "02", "03", "04", "05", "06"}:
        raise ValueError(f"unsupported MUSeg mine id in sample {sample_id!r}")
    return mine


def _mask_sha256(mask: np.ndarray) -> str:
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("mask hash requires a two-dimensional boolean array")
    digest = hashlib.sha256()
    digest.update(f"{mask.shape[0]}x{mask.shape[1]}:bool-c-order\n".encode("ascii"))
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _domain_key(seed: int, sample_id: str, domain: str) -> np.uint64:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    payload = f"DVC-A1/splitmix64-v1/{seed}/{sample_id}/{domain}".encode("utf-8")
    return np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized unsigned SplitMix64 permutation with fixed wraparound semantics."""
    with np.errstate(over="ignore"):
        z = (values.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _UINT64_MASK
        return (z ^ (z >> np.uint64(31))) & _UINT64_MASK


def _ranked_prefix(candidate: np.ndarray, count: int, *, seed: int, sample_id: str, domain: str) -> np.ndarray:
    flat_indices = np.flatnonzero(candidate).astype(np.uint64, copy=False)
    if count < 0 or count > flat_indices.size:
        raise ValueError(f"requested {count} pixels from {flat_indices.size} candidates")
    selected = np.zeros(candidate.size, dtype=np.bool_)
    if count:
        scores = _splitmix64(flat_indices ^ _domain_key(seed, sample_id, domain))
        order = np.argsort(scores, kind="stable")
        selected[flat_indices[order[:count]].astype(np.intp)] = True
    return selected.reshape(candidate.shape)


def depth_edge_candidates(depth16: np.ndarray, relative_jump_threshold: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    if depth16.dtype != np.uint16 or depth16.ndim != 2:
        raise ValueError(f"expected two-dimensional uint16 Depth16, got {depth16.shape} {depth16.dtype}")
    if not math.isfinite(relative_jump_threshold) or relative_jump_threshold <= 0:
        raise ValueError("relative jump threshold must be finite and positive")
    valid = depth16 > 0
    seed = np.zeros(depth16.shape, dtype=np.bool_)
    depth = depth16.astype(np.float64)

    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    horizontal_ratio = np.zeros(horizontal_valid.shape, dtype=np.float64)
    np.divide(
        np.abs(depth[:, :-1] - depth[:, 1:]),
        np.maximum(np.maximum(depth[:, :-1], depth[:, 1:]), 1.0),
        out=horizontal_ratio,
        where=horizontal_valid,
    )
    horizontal_edge = horizontal_valid & (horizontal_ratio >= relative_jump_threshold)
    seed[:, :-1] |= horizontal_edge
    seed[:, 1:] |= horizontal_edge

    vertical_valid = valid[:-1, :] & valid[1:, :]
    vertical_ratio = np.zeros(vertical_valid.shape, dtype=np.float64)
    np.divide(
        np.abs(depth[:-1, :] - depth[1:, :]),
        np.maximum(np.maximum(depth[:-1, :], depth[1:, :]), 1.0),
        out=vertical_ratio,
        where=vertical_valid,
    )
    vertical_edge = vertical_valid & (vertical_ratio >= relative_jump_threshold)
    seed[:-1, :] |= vertical_edge
    seed[1:, :] |= vertical_edge

    boundary = cv2.dilate(seed.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    boundary &= valid
    guard = cv2.dilate(boundary.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
    nonboundary = valid & ~guard
    return boundary, nonboundary


def build_corruption_masks(
    depth16: np.ndarray,
    sample_id: str,
    *,
    seed: int = 20260908,
    relative_jump_threshold: float = 0.05,
) -> CorruptionMasks:
    boundary, nonboundary = depth_edge_candidates(depth16, relative_jump_threshold)
    boundary_count = int(boundary.sum())
    boundary_counts = {
        condition: int(math.floor(boundary_count * DOSES[condition]))
        for condition in ("clean", "boundary-q25", "boundary-q50", "boundary-q75")
    }
    masks: dict[str, np.ndarray] = {}
    for condition in ("clean", "boundary-q25", "boundary-q50", "boundary-q75"):
        masks[condition] = _ranked_prefix(
            boundary,
            boundary_counts[condition],
            seed=seed,
            sample_id=sample_id,
            domain="boundary",
        )
    matched_count = boundary_counts["boundary-q50"]
    if int(nonboundary.sum()) < matched_count:
        raise ValueError(
            f"nonboundary-q50 shortage for {sample_id}: need {matched_count}, have {int(nonboundary.sum())}"
        )
    masks["nonboundary-q50"] = _ranked_prefix(
        nonboundary,
        matched_count,
        seed=seed,
        sample_id=sample_id,
        domain="nonboundary-matched",
    )
    if not (
        np.all(masks["boundary-q25"] <= masks["boundary-q50"])
        and np.all(masks["boundary-q50"] <= masks["boundary-q75"])
    ):
        raise RuntimeError("boundary dose masks are not nested")
    counts = {condition: int(mask.sum()) for condition, mask in masks.items()}
    hashes = {condition: _mask_sha256(mask) for condition, mask in masks.items()}
    return CorruptionMasks(
        masks=masks,
        boundary_candidate_count=boundary_count,
        nonboundary_candidate_count=int(nonboundary.sum()),
        condition_counts=counts,
        condition_sha256=hashes,
    )


def quantized_condition_depth(depth16: np.ndarray, mask: np.ndarray, depth_max_raw: int = 13932) -> np.ndarray:
    if mask.dtype != np.bool_ or mask.shape != depth16.shape:
        raise ValueError("corruption mask must be boolean and aligned with Depth16")
    corrupted = depth16.copy()
    corrupted[mask] = 0
    return quantize_depth(corrupted, depth_max_raw)


def boundary_distance_pixels(height: int, width: int, ratio: float = 0.02) -> int:
    if height <= 0 or width <= 0 or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("boundary geometry and ratio must be positive")
    return max(1, int(round(ratio * math.hypot(height, width))))


def _inside_boundary(mask: np.ndarray, distance: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.bool_)
    padded = np.pad(binary.astype(np.uint8), 1, mode="constant", constant_values=0)
    transform = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return binary & (transform <= float(distance))


def _ignore_safe_region(ignore: np.ndarray, distance: int) -> np.ndarray:
    if not np.any(ignore):
        return np.ones(ignore.shape, dtype=np.bool_)
    distance_to_ignore = cv2.distanceTransform((~ignore).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (~ignore) & (distance_to_ignore > float(distance))


def semantic_boundary_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int = 15,
    ignore_label: int = 255,
    distance_ratio: float = 0.02,
) -> dict[str, Any]:
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must be aligned two-dimensional arrays")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    distance = boundary_distance_pixels(*target.shape, ratio=distance_ratio)
    safe = _ignore_safe_region(target == ignore_label, distance)
    class_values: list[float | None] = []
    for class_id in range(num_classes):
        ground_truth = (target == class_id) & safe
        predicted = (prediction == class_id) & safe
        ground_truth_present = bool(np.any(ground_truth))
        predicted_present = bool(np.any(predicted))
        if not ground_truth_present and not predicted_present:
            class_values.append(None)
            continue
        if not ground_truth_present or not predicted_present:
            class_values.append(0.0)
            continue
        gt_boundary = _inside_boundary(ground_truth, distance) & safe
        pred_boundary = _inside_boundary(predicted, distance) & safe
        union = int(np.count_nonzero(gt_boundary | pred_boundary))
        intersection = int(np.count_nonzero(gt_boundary & pred_boundary))
        class_values.append(float(intersection / union) if union else 0.0)
    defined = [value for value in class_values if value is not None]
    return {
        "distance_pixels": distance,
        "value": float(np.mean(defined)) if defined else None,
        "per_class": class_values,
        "defined_class_count": len(defined),
    }


def confusion_matrix(prediction: np.ndarray, target: np.ndarray, num_classes: int = 15, ignore_label: int = 255) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must be aligned")
    valid = (target != ignore_label) & (target >= 0) & (target < num_classes)
    encoded = target[valid].astype(np.int64) * num_classes + prediction[valid].astype(np.int64)
    if np.any((prediction[valid] < 0) | (prediction[valid] >= num_classes)):
        raise ValueError("prediction contains an out-of-range class")
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def image_miou_from_confusion(hist: np.ndarray) -> float | None:
    denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    defined = denominator > 0
    if not np.any(defined):
        return None
    return float(np.mean(np.diag(hist)[defined] / denominator[defined]))


def aggregate_condition_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty condition records")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["location_group"]), []).append(record)
    per_group: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        boundary_values = [float(row["boundary_iou"]) for row in rows if row["boundary_iou"] is not None]
        miou_values = [float(row["image_miou"]) for row in rows if row["image_miou"] is not None]
        per_group.append(
            {
                "location_group": group,
                "mine": str(rows[0]["mine"]),
                "image_count": len(rows),
                "boundary_iou": float(np.mean(boundary_values)) if boundary_values else None,
                "image_miou": float(np.mean(miou_values)) if miou_values else None,
            }
        )
    return {"group_count": len(per_group), "per_group": per_group}


def paired_effect(
    group_values: Mapping[str, Mapping[str, float | None]],
    left: str,
    right: str,
    *,
    bootstrap_seed: int = 20260908,
    bootstrap_replicates: int = 10000,
) -> dict[str, Any]:
    pairs: list[tuple[str, float]] = []
    for group, values in sorted(group_values.items()):
        left_value, right_value = values.get(left), values.get(right)
        if left_value is not None and right_value is not None:
            pairs.append((group, float(left_value) - float(right_value)))
    if not pairs:
        raise ValueError(f"no defined paired groups for {left} minus {right}")
    differences = np.asarray([value for _, value in pairs], dtype=np.float64)
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    indices = rng.integers(0, len(differences), size=(bootstrap_replicates, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5], method="linear")
    return {
        "left": left,
        "right": right,
        "paired_group_count": len(pairs),
        "point_percentage_points": float(differences.mean() * 100.0),
        "interval95_percentage_points": [float(lower * 100.0), float(upper * 100.0)],
        "per_group_percentage_points": {group: float(value * 100.0) for group, value in pairs},
    }


def adjudicate(dose_effect: Mapping[str, Any], specificity_effect: Mapping[str, Any], group_mines: Mapping[str, str]) -> dict[str, Any]:
    dose_point = float(dose_effect["point_percentage_points"])
    dose_upper = float(dose_effect["interval95_percentage_points"][1])
    specificity_point = float(specificity_effect["point_percentage_points"])
    mine_values: dict[str, list[float]] = {mine: [] for mine in ("01", "02", "03", "04", "05", "06")}
    for group, value in specificity_effect["per_group_percentage_points"].items():
        mine_values[group_mines[group]].append(float(value))
    mine_means = {
        mine: (float(np.mean(values)) if values else None)
        for mine, values in mine_values.items()
    }
    negative_mines = sum(value is not None and value < 0.0 for value in mine_means.values())
    if dose_point <= -2.0 and dose_upper < 0.0 and specificity_point < 0.0 and negative_mines >= 4:
        decision = "supported"
    elif dose_point > -1.0 or specificity_point >= 0.0:
        decision = "not-supported"
    else:
        decision = "inconclusive"
    return {
        "decision": decision,
        "dose_effect": dict(dose_effect),
        "specificity_effect": dict(specificity_effect),
        "specificity_mine_means_percentage_points": mine_means,
        "negative_mine_count": negative_mines,
    }
