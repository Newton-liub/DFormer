#!/usr/bin/env python3
"""CPU-only qualification for MMFR-A2 v3.

The fixture is synthetic and local to this process.  No checkpoint, official split,
training loop or GPU is accessed.  The checks target the v3 validity-state contract,
its v2-compatible burden/curriculum, and the CPU batch helper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import cv2
import numpy as np
import torch

from utils.dataloader import multimodal_failure_v3 as basis
from utils.dataloader.mmfr_training import (
    DEPTH_PLANE_MEAN,
    DEPTH_PLANE_STD,
    make_sample_generator,
    normalized_to_uint8,
    sample_id_words,
    build_seed_words,
    uint8_to_normalized,
)
from utils.dataloader.mmfr_training_v3 import (
    CORRUPTION_BASIS,
    PROTOCOL_ID,
    TARGET_COMPOSITION,
    build_mmfr_training_batch_v3,
    sample_depth_failure_specs_v3,
)


class Checks:
    def __init__(self) -> None:
        self.assertions = 0
        self.failures: list[Dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.assertions += 1
        if not condition:
            self.failures.append({"name": name, "detail": detail})


def build_fixture() -> Dict[str, Any]:
    height, width = 32, 40
    pad_rows, pad_cols = 4, 5
    valid_mask = np.ones((height, width), dtype=bool)
    valid_mask[-pad_rows:, :] = False
    valid_mask[:, -pad_cols:] = False

    rgb_uint8 = np.empty((height, width, 3), dtype=np.uint8)
    rows = np.arange(height, dtype=np.uint16)[:, None]
    cols = np.arange(width, dtype=np.uint16)[None, :]
    rgb_uint8[:, :, 0] = ((rows * 7 + cols * 3 + 11) % 251 + 1).astype(np.uint8)
    rgb_uint8[:, :, 1] = ((rows * 5 + cols * 11 + 31) % 251 + 1).astype(np.uint8)
    rgb_uint8[:, :, 2] = ((rows * 13 + cols * 2 + 61) % 251 + 1).astype(np.uint8)
    depth_uint8 = ((rows * 9 + cols * 5 + 17) % 220 + 20).astype(np.uint8)
    depth_uint8[6:12, 10:17] = np.uint8(0)
    depth_uint8[~valid_mask] = np.uint8(0)
    rgb_uint8[~valid_mask] = np.uint8(0)

    rgb_normalized = uint8_to_normalized(
        rgb_uint8,
        np.asarray([0.485, 0.456, 0.406], dtype=np.float64),
        np.asarray([0.229, 0.224, 0.225], dtype=np.float64),
    ).transpose(2, 0, 1)
    depth_normalized = uint8_to_normalized(
        depth_uint8[:, :, None],
        np.asarray([DEPTH_PLANE_MEAN], dtype=np.float64),
        np.asarray([DEPTH_PLANE_STD], dtype=np.float64),
    )[:, :, 0][None]
    rgb_normalized[:, ~valid_mask] = np.float32(0.0)
    depth_normalized[:, ~valid_mask] = np.float32(0.0)
    return {
        "rgb_uint8": rgb_uint8,
        "depth_uint8": depth_uint8,
        "valid_mask": valid_mask,
        "depth_valid_pre": (depth_uint8 > 0) & valid_mask,
        "rgb": torch.from_numpy(rgb_normalized[None].astype(np.float32)),
        "depth": torch.from_numpy(depth_normalized[None].astype(np.float32)),
        "sample_id": "fixture/mmfr-v3/qualification.png",
    }


def translate(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return basis._translate(array, dy, dx)


def call_helper(fixture: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    values = {
        "rgb": fixture["rgb"],
        "depth": fixture["depth"],
        "sample_ids": [fixture["sample_id"]],
        "epoch": 1,
        "iteration": 0,
        "niters_per_epoch": 5,
        "nepochs": 3,
        "rgb_mean": [0.485, 0.456, 0.406],
        "rgb_std": [0.229, 0.224, 0.225],
        "corruption_seed": 2026091402,
        "global_rank": 0,
        "max_specs": 2,
        "sample_id_root": None,
    }
    values.update(kwargs)
    return build_mmfr_training_batch_v3(**values)


def record_spec(result: basis.FailureResult) -> Mapping[str, Any]:
    return result.metadata["specs"][0]


def direct_result(fixture: Mapping[str, Any], kind: str, severity: float, seed: int) -> basis.FailureResult:
    return basis.apply_failures(
        fixture["rgb_uint8"],
        fixture["depth_uint8"],
        [basis.FailureSpec("depth", kind, severity)],
        np.random.default_rng(seed),
        depth_validity=fixture["depth_valid_pre"],
        validity_mask=fixture["valid_mask"],
    )


def run_checks() -> Dict[str, Any]:
    c = Checks()
    fixture = build_fixture()
    c.check("protocol_id", PROTOCOL_ID == "MMFR-A2-train-integration-v3", PROTOCOL_ID)
    c.check("corruption_basis", CORRUPTION_BASIS == "MMFR-A1-corruption-basis-v3", CORRUPTION_BASIS)
    c.check("target_composition", TARGET_COMPOSITION == "R_depth_sup_v3(p) = V_state_final(p) * R_depth_synthetic(p)")
    c.check("six_kinds_unchanged", basis.FAILURE_KINDS == ("entire_missing", "spatial_dropout", "gaussian_noise", "blur", "quantization", "misalignment"))
    c.check("single_severity_encoding", basis.SEVERITY_ENCODING == "single")

    clean = call_helper(fixture, p_clean=1.0)
    c.check("clean_rgb_exact_noop", torch.equal(clean["rgb"], fixture["rgb"]))
    c.check("clean_depth_exact_noop", torch.equal(clean["depth"], fixture["depth"]))
    c.check("clean_protocol", clean["protocol"] == PROTOCOL_ID)
    expected_clean_target = fixture["depth_valid_pre"].astype(np.float32).copy()
    expected_clean_target[~fixture["valid_mask"]] = np.float32(1.0)
    c.check("clean_target_native_invalid", torch.equal(clean["reliability_target"][0, 1], torch.from_numpy(expected_clean_target)))
    c.check("clean_pre_state", torch.equal(clean["depth_valid_pre"][0, 0], torch.from_numpy(fixture["depth_valid_pre"])))
    c.check("clean_post_state", torch.equal(clean["depth_valid_post"][0, 0], torch.from_numpy(fixture["depth_valid_pre"])))
    c.check("clean_rgb_scaffold", bool(torch.all(clean["reliability_target"][:, 0] == 1.0)))

    specs = {
        "entire_missing": 1.0,
        "spatial_dropout": 0.5,
        "gaussian_noise": 0.5,
        "blur": 0.5,
        "quantization": 0.5,
        "misalignment": 0.75,
    }
    direct: Dict[str, basis.FailureResult] = {}
    for index, (kind, severity) in enumerate(specs.items()):
        result = direct_result(fixture, kind, severity, 100 + index)
        direct[kind] = result
        state = result.validity_state
        out_depth = result.depth if result.depth.ndim == 2 else result.depth[:, :, 0]
        expected_post = (out_depth > 0) & fixture["valid_mask"]
        c.check(f"{kind}_shape", result.depth.shape == fixture["depth_uint8"].shape)
        c.check(f"{kind}_dtype", result.depth.dtype == np.uint8 and result.reliability.dtype == np.float32)
        c.check(f"{kind}_finite", bool(np.isfinite(result.reliability).all()))
        c.check(f"{kind}_reliability_range", float(result.reliability.min()) >= 0.0 and float(result.reliability.max()) <= 1.0)
        c.check(f"{kind}_state_dtype_shape", state.dtype == np.bool_ and state.shape == fixture["valid_mask"].shape)
        c.check(f"{kind}_state_equals_raw", np.array_equal(state, expected_post))
        c.check(f"{kind}_rgb_unchanged", np.array_equal(result.rgb, fixture["rgb_uint8"]))
        c.check(f"{kind}_valid_uint8_nonzero", bool(np.all(out_depth[state] >= 1)) if state.any() else True)
        c.check(f"{kind}_invalid_uint8_zero", bool(np.all(out_depth[~state] == 0)))
        c.check(f"{kind}_geometry_subset", not bool(np.any(state & ~fixture["valid_mask"])))

    for kind in ("gaussian_noise", "blur", "quantization"):
        result = direct[kind]
        out_depth = result.depth if result.depth.ndim == 2 else result.depth[:, :, 0]
        resurrected = (~fixture["depth_valid_pre"]) & (out_depth > 0)
        c.check(f"{kind}_invalid_never_resurrected", int(np.count_nonzero(resurrected)) == 0)
        c.check(f"{kind}_state_never_resurrected", int(np.count_nonzero((~fixture["depth_valid_pre"]) & result.validity_state)) == 0)
        c.check(f"{kind}_validity_transition_unchanged", np.array_equal(result.validity_state, fixture["depth_valid_pre"]))

    dropout = direct["spatial_dropout"]
    c.check("dropout_removes_state", bool(np.any(fixture["depth_valid_pre"] & ~dropout.validity_state)))
    c.check("dropout_state_subset", not bool(np.any(dropout.validity_state & ~fixture["depth_valid_pre"])))
    entire = direct["entire_missing"]
    c.check("entire_missing_clears_state", int(np.count_nonzero(entire.validity_state)) == 0)
    c.check("entire_missing_zero_depth", int(np.count_nonzero(entire.depth)) == 0)

    # The masked blur must equal the independently reconstructed normalized convolution.
    blur_height, blur_width = 81, 81
    blur_depth = np.full((blur_height, blur_width), np.uint8(200), dtype=np.uint8)
    blur_depth[40, 40] = np.uint8(0)
    blur_rgb = np.full((blur_height, blur_width, 3), np.uint8(100), dtype=np.uint8)
    blur_state = blur_depth > 0
    blur_result = basis.apply_failures(
        blur_rgb,
        blur_depth,
        [basis.FailureSpec("depth", "blur", 1.0)],
        np.random.default_rng(999),
        depth_validity=blur_state,
        validity_mask=np.ones_like(blur_state),
    )
    blur_record = record_spec(blur_result)
    sigma = float(blur_record["corruption_parameter"]["sigma_px"])
    ksize = int(blur_record["corruption_parameter"]["ksize"])
    numerator = cv2.GaussianBlur(blur_depth.astype(np.float32) * blur_state.astype(np.float32), (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    denominator = cv2.GaussianBlur(blur_state.astype(np.float32), (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    expected_blur = np.clip(np.rint(numerator / denominator), 0.0, 255.0).astype(np.uint8)
    expected_blur[~blur_state] = np.uint8(0)
    expected_blur[blur_state] = np.maximum(expected_blur[blur_state], np.uint8(1))
    c.check("blur_mask_normalized_formula", np.array_equal(blur_result.depth, expected_blur))
    naive_blur = cv2.GaussianBlur(blur_depth, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    c.check("blur_invalid_sentinel_not_polluting_neighbor", int(blur_result.depth[40, 39]) > int(naive_blur[40, 39]))
    c.check("blur_invalid_sentinel_stays_zero", int(blur_result.depth[40, 40]) == 0)

    # MID-A: reconstruct both transports from the recorded integer shift.
    alignment = direct["misalignment"]
    alignment_record = record_spec(alignment)
    dy, dx = int(alignment_record["dy"]), int(alignment_record["dx"])
    expected_alignment_depth = translate(fixture["depth_uint8"], dy, dx)
    expected_alignment_state = translate(fixture["depth_valid_pre"], dy, dx) & fixture["valid_mask"]
    expected_alignment_depth = basis._depth_finalize(expected_alignment_depth, expected_alignment_state, fixture["valid_mask"])
    c.check("mid_a_depth_integer_translation", np.array_equal(alignment.depth, expected_alignment_depth))
    c.check("mid_a_state_integer_translation", np.array_equal(alignment.validity_state, expected_alignment_state))
    c.check("mid_a_same_shift_metadata", alignment_record["corruption_parameter"]["transport"] == "integer-translate-depth-and-validity-together")

    # Reconstruct one helper call from its recorded seed words and spec draw to verify target composition.
    target_batch = call_helper(fixture, p_clean=0.0)
    target_record = target_batch["metadata"][0]
    words = tuple(int(value) for value in target_record["seed_words"])
    target_rng = make_sample_generator(words)
    target_rng.random()
    expected_specs, _ = sample_depth_failure_specs_v3(float(target_record["curriculum_progress"]), target_rng, max_specs=2)
    expected_result = basis.apply_failures(
        fixture["rgb_uint8"],
        fixture["depth_uint8"],
        expected_specs,
        target_rng,
        depth_validity=fixture["depth_valid_pre"],
        validity_mask=fixture["valid_mask"],
    )
    expected_target = np.ones((2, fixture["valid_mask"].shape[0], fixture["valid_mask"].shape[1]), dtype=np.float32)
    expected_target[1] = expected_result.validity_state.astype(np.float32) * expected_result.reliability[1]
    expected_target[:, ~fixture["valid_mask"]] = np.float32(1.0)
    c.check("target_formula_v_state_times_r_syn", torch.equal(target_batch["reliability_target"][0], torch.from_numpy(expected_target)))
    c.check("target_formula_metadata", target_record["target_composition"] == TARGET_COMPOSITION)
    c.check("target_post_state_exact", torch.equal(target_batch["depth_valid_post"][0, 0], torch.from_numpy(expected_result.validity_state)))

    # Determinism, pad contract and four telemetry categories are checked on the helper.
    repeat = call_helper(fixture, p_clean=0.0)
    for key in ("rgb", "depth", "raw_rgb", "raw_depth", "reliability_target", "valid_mask", "depth_valid_pre", "depth_valid_post"):
        c.check(f"deterministic_{key}", torch.equal(target_batch[key], repeat[key]))
    c.check("deterministic_metadata", repr(target_batch["metadata"]) == repr(repeat["metadata"]))
    pad = torch.from_numpy((~fixture["valid_mask"])[None, None])
    for key in ("rgb", "depth"):
        expanded_pad = pad.expand_as(target_batch[key])
        c.check(f"pad_{key}_exact_zero", bool(torch.all(torch.masked_select(target_batch[key], expanded_pad) == 0.0)))
    for key in ("raw_rgb", "raw_depth"):
        expanded_pad = pad.expand_as(target_batch[key])
        c.check(f"pad_{key}_exact_zero", bool(torch.all(torch.masked_select(target_batch[key], expanded_pad) == 0.0)))
    telemetry = target_record
    for key in ("natural_invalid_pixels", "synthetic_invalid_pixels", "implicit_quality_pixels", "valid_clean_pixels"):
        c.check(f"telemetry_{key}", key in telemetry and int(telemetry[key]) >= 0)
    c.check("telemetry_does_not_change_target", target_record["target_composition"] == TARGET_COMPOSITION)

    # Same RNG stream across severity values must preserve the frozen monotone burden intent.
    monotone_kinds = ("gaussian_noise", "blur", "quantization", "misalignment")
    for kind_index, kind in enumerate(monotone_kinds):
        burdens = []
        reliabilities = []
        for severity in (0.2, 0.5, 1.0):
            result = direct_result(fixture, kind, severity, 300 + kind_index)
            info = record_spec(result)
            burdens.append(float(info["burden_mean"]))
            reliabilities.append(float(info["reliability_mean"]))
        c.check(f"monotone_burden_{kind}", all(left <= right + 1.0e-7 for left, right in zip(burdens, burdens[1:])), repr(burdens))
        c.check(f"monotone_reliability_{kind}", all(left + 1.0e-7 >= right for left, right in zip(reliabilities, reliabilities[1:])), repr(reliabilities))
    dropout_counts = []
    for severity in (0.2, 0.5, 1.0):
        result = direct_result(fixture, "spatial_dropout", severity, 700)
        dropout_counts.append(int(record_spec(result)["dropped_pixels"]))
    c.check("monotone_dropout_extent", all(left <= right for left, right in zip(dropout_counts, dropout_counts[1:])), repr(dropout_counts))

    return {
        "schema_version": "mmfr-a2-v3-cpu-qualification-v1",
        "protocol_id": PROTOCOL_ID,
        "corruption_basis": CORRUPTION_BASIS,
        "assertions": c.assertions,
        "failed": len(c.failures),
        "status": "PASS" if not c.failures else "FAIL",
        "failures": c.failures,
        "fixture": {"height": 32, "width": 40, "pad_rows": 4, "pad_cols": 5},
        "official_test_included": False,
        "checkpoint_read": False,
        "training_run": False,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = run_checks()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
