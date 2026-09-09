from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.mve.dvc_a1_core import (
    adjudicate,
    build_corruption_masks,
    paired_effect,
    quantized_condition_depth,
    semantic_boundary_iou,
)
from tools.mve.dvc_a1_protocol import (
    DEFAULT_TEMPLATE,
    DEFAULT_V2_TEMPLATE,
    DEFAULT_V3_TEMPLATE,
    DvcProtocolError,
    PROTOCOL_V3_ID,
    load_dvc_protocol,
)
from tools.mve.run_dvc_a1 import (
    _assert_mask_hashes_match_frozen_v1,
    _assert_paired_group_count,
    _boundary_target,
    _frozen_v1_condition_hashes,
)
from tools.prepare_museg import quantize_depth


def _depth_with_edge() -> np.ndarray:
    depth = np.full((32, 32), 1000, dtype=np.uint16)
    depth[:, 16:] = 2000
    return depth


def test_templates_have_frozen_dvc_a1_identities() -> None:
    v1 = load_dvc_protocol(DEFAULT_TEMPLATE, allow_placeholders=True)
    v2 = load_dvc_protocol(DEFAULT_V2_TEMPLATE, allow_placeholders=True)
    v3 = load_dvc_protocol(DEFAULT_V3_TEMPLATE, allow_placeholders=True)
    assert v1.raw["protocol_id"] == "DVC-A1-valdev-boundary-zero-v1"
    assert v2.raw["protocol_id"] == "DVC-A1-valdev-boundary-zero-v2"
    assert v3.raw["protocol_id"] == PROTOCOL_V3_ID
    assert v1.raw["official_test"]["included"] is False
    assert v2.raw["official_test"]["included"] is False
    assert v3.raw["official_test"]["included"] is False
    assert v3.raw["boundary_iou"]["background_context"] == "valid-one-vs-rest-context"
    assert v3.raw["boundary_iou"]["true_ignore_value"] == 255
    scope = v2.raw["evaluation_scope"]
    assert scope["included_location_group_count"] == 138
    assert scope["included_sample_count"] == 218
    assert scope["q75_zero_sample_count"] == 31
    assert scope["fully_constructable_location_group_count"] == 123
    assert scope["partially_constructable_location_group_count"] == 15


def test_v2_mask_hashes_are_checked_against_frozen_v1_manifest(tmp_path) -> None:
    hashes = {condition: (str(index + 1) * 64)[:64] for index, condition in enumerate((
        "clean", "boundary-q25", "boundary-q50", "boundary-q75", "nonboundary-q50"
    ))}
    manifest_path = tmp_path / "mask-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "museg-dvc-a1-mask-manifest-v1",
                "samples": [{"sample_id": "sample", "condition_mask_sha256": hashes}],
            }
        ),
        encoding="utf-8",
    )
    protocol = SimpleNamespace(source_mask_manifest_path=manifest_path)
    frozen = _frozen_v1_condition_hashes(protocol, ["RGB/sample.jpg"])
    _assert_mask_hashes_match_frozen_v1(frozen, frozen)

    changed = {"sample": {**hashes, "boundary-q75": "f" * 64}}
    with pytest.raises(DvcProtocolError, match="frozen v1 manifest"):
        _assert_mask_hashes_match_frozen_v1(changed, frozen)


def test_v2_analysis_requires_all_expected_paired_groups() -> None:
    analysis = {
        "location_group_count": 138,
        "dose_effect": {"paired_group_count": 138},
        "specificity_effect": {"paired_group_count": 138},
    }
    _assert_paired_group_count(analysis, 138, "primary")
    with pytest.raises(DvcProtocolError, match="valid paired groups"):
        _assert_paired_group_count({**analysis, "dose_effect": {"paired_group_count": 137}}, 138, "primary")


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


def test_boundary_iou_background_context_preserves_thin_foreground() -> None:
    background = 2
    target = np.full((32, 32), background, dtype=np.int64)
    target[15:17, 4:28] = 0
    prediction = target.copy()
    prediction[prediction == background] = 1

    result = semantic_boundary_iou(
        prediction,
        target,
        num_classes=2,
        background_label=background,
    )
    assert result["per_class"][0] == 1.0
    assert result["per_class"][1] == 0.0
    assert result["defined_class_count"] == 2


def test_v3_boundary_target_separates_background_from_true_ignore() -> None:
    protocol = SimpleNamespace(
        raw={
            "protocol_id": PROTOCOL_V3_ID,
            "input_contract": {
                "label": {
                    "foreground_ids": list(range(1, 16)),
                    "evaluator_ignore": 255,
                }
            },
        }
    )
    raw = np.asarray([[0, 1, 15]], dtype=np.int64)
    training = np.asarray([[255, 0, 14]], dtype=np.int64)
    metric = _boundary_target(protocol, training, raw)
    assert metric.tolist() == [[15, 0, 14]]


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
