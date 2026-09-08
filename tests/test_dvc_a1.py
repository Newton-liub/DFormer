from __future__ import annotations

import numpy as np

from tools.mve.dvc_a1_core import (
    adjudicate,
    build_corruption_masks,
    paired_effect,
    quantized_condition_depth,
    semantic_boundary_iou,
)
from tools.mve.dvc_a1_protocol import DEFAULT_TEMPLATE, load_dvc_protocol
from tools.prepare_museg import quantize_depth


def _depth_with_edge() -> np.ndarray:
    depth = np.full((32, 32), 1000, dtype=np.uint16)
    depth[:, 16:] = 2000
    return depth


def test_template_has_frozen_dvc_a1_identity() -> None:
    protocol = load_dvc_protocol(DEFAULT_TEMPLATE, allow_placeholders=True)
    assert protocol.raw["protocol_id"] == "DVC-A1-valdev-boundary-zero-v1"
    assert protocol.raw["official_test"]["included"] is False


def test_masks_are_deterministic_nested_and_area_matched() -> None:
    depth = _depth_with_edge()
    first = build_corruption_masks(depth, "01-02-03-0004-sample")
    second = build_corruption_masks(depth, "01-02-03-0004-sample")

    assert first.condition_sha256 == second.condition_sha256
    assert np.all(first.masks["boundary-q25"] <= first.masks["boundary-q50"])
    assert np.all(first.masks["boundary-q50"] <= first.masks["boundary-q75"])
    assert first.condition_counts["boundary-q50"] == first.condition_counts["nonboundary-q50"]
    assert not np.any(first.masks["boundary-q50"] & first.masks["nonboundary-q50"])


def test_clean_depth_quantization_reuses_production_rounding() -> None:
    depth = _depth_with_edge()
    clean = np.zeros(depth.shape, dtype=np.bool_)
    assert np.array_equal(quantized_condition_depth(depth, clean), quantize_depth(depth, 13932))


def test_boundary_iou_identical_one_empty_both_empty_and_ignore() -> None:
    target = np.full((32, 32), 255, dtype=np.int64)
    target[8:24, 8:24] = 0
    prediction = np.zeros((32, 32), dtype=np.int64)

    identical = semantic_boundary_iou(prediction, target, num_classes=2)
    assert identical["per_class"][0] == 1.0
    assert identical["per_class"][1] is None
    assert identical["value"] == 1.0

    absent_prediction = np.ones((32, 32), dtype=np.int64)
    one_empty = semantic_boundary_iou(absent_prediction, target, num_classes=2)
    assert one_empty["per_class"][0] == 0.0

    all_ignore = np.full((32, 32), 255, dtype=np.int64)
    both_empty = semantic_boundary_iou(prediction, all_ignore, num_classes=2)
    assert both_empty["value"] is None
    assert both_empty["defined_class_count"] == 0


def test_bootstrap_is_reproducible_and_adjudication_thresholds_are_frozen() -> None:
    values = {
        "01-a": {"left": 0.70, "right": 0.74},
        "02-a": {"left": 0.60, "right": 0.63},
        "03-a": {"left": 0.80, "right": 0.82},
        "04-a": {"left": 0.50, "right": 0.53},
        "05-a": {"left": 0.65, "right": 0.67},
        "06-a": {"left": 0.72, "right": 0.75},
    }
    first = paired_effect(values, "left", "right", bootstrap_replicates=200)
    second = paired_effect(values, "left", "right", bootstrap_replicates=200)
    assert first == second

    mines = {group: group[:2] for group in values}
    decision = adjudicate(first, first, mines)
    assert decision["decision"] == "supported"
    assert decision["negative_mine_count"] == 6
