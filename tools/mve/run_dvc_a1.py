#!/usr/bin/env python3
"""Run the frozen MUSeg DVC-A1 preflight or complete val-dev evaluation."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from tools.evaluate_museg_checkpoint import (
    apply_channel_order,
    build_msflip_views,
    configure_fp32_forward,
    load_model,
    metrics_from_confusion,
    msflip_whole_logits,
    split_entries,
)
from tools.mve.dvc_a1_core import (
    CONDITION_IDS,
    adjudicate,
    aggregate_condition_records,
    build_corruption_masks,
    confusion_matrix,
    image_miou_from_confusion,
    location_group,
    mine_id,
    paired_effect,
    quantized_condition_depth,
    semantic_boundary_iou,
)
from tools.mve.dvc_a1_protocol import DvcProtocol, DvcProtocolError, PROTOCOL_V3_ID, file_sha256, load_dvc_protocol


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sample_paths(protocol: DvcProtocol, sample_id: str) -> dict[str, Path]:
    dataset = protocol.raw["dataset"]
    root = protocol.dataset_root
    return {
        "rgb": root / str(dataset["rgb_directory"]) / f"{sample_id}.jpg",
        "depth16": root / str(dataset["depth16_directory"]) / f"{sample_id}.png",
        "depth": root / str(dataset["quantized_depth_directory"]) / f"{sample_id}.png",
        "label": root / str(dataset["label_directory"]) / f"{sample_id}.png",
    }


def _read_sample(protocol: DvcProtocol, sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = _sample_paths(protocol, sample_id)
    rgb_bgr = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
    depth16 = cv2.imread(str(paths["depth16"]), cv2.IMREAD_UNCHANGED)
    depth8 = cv2.imread(str(paths["depth"]), cv2.IMREAD_GRAYSCALE)
    label_raw = cv2.imread(str(paths["label"]), cv2.IMREAD_GRAYSCALE)
    if any(value is None for value in (rgb_bgr, depth16, depth8, label_raw)):
        raise FileNotFoundError(f"missing or undecodable val-dev modality for {sample_id}")
    assert rgb_bgr is not None and depth16 is not None and depth8 is not None and label_raw is not None
    if depth16.dtype != np.uint16 or depth16.ndim != 2:
        raise ValueError(f"Depth16 for {sample_id} is not two-dimensional uint16")
    if rgb_bgr.shape[:2] != depth16.shape or depth8.shape != depth16.shape or label_raw.shape != depth16.shape:
        raise ValueError(f"modality geometry mismatch for {sample_id}")
    if int(depth16.max()) > 13932:
        raise ValueError(f"Depth16 exceeds 13932 for {sample_id}")
    unexpected_labels = sorted(set(int(value) for value in np.unique(label_raw)) - set(range(16)))
    if unexpected_labels:
        raise ValueError(f"Label contains values outside 0..15 for {sample_id}: {unexpected_labels}")
    rgb = apply_channel_order(rgb_bgr, "RGB")
    label = label_raw.astype(np.int64) - 1
    label[label < 0] = 255
    return rgb, depth16, depth8, label, label_raw.astype(np.int64)


def _boundary_target(protocol: DvcProtocol, label: np.ndarray, label_raw: np.ndarray) -> np.ndarray:
    if protocol.raw["protocol_id"] == PROTOCOL_V3_ID:
        # Raw background (0) is a valid one-vs-rest context for Boundary IoU;
        # only a true ignore label would be encoded as 255.
        metric_label = label_raw - 1
        metric_label[label_raw == 0] = len(protocol.raw["input_contract"]["label"]["foreground_ids"])
        metric_label[label_raw == 255] = int(protocol.raw["input_contract"]["label"]["evaluator_ignore"])
        return metric_label
    return label


def _protocol_version(protocol: DvcProtocol) -> str:
    if protocol.raw["protocol_id"] == PROTOCOL_V3_ID:
        return "v3"
    if protocol.raw["protocol_id"] == "DVC-A1-valdev-boundary-zero-v2":
        return "v2"
    return "v1"


def _config(protocol: DvcProtocol) -> Any:
    config = copy.copy(importlib.import_module(str(protocol.raw["model"]["config_module"])).C)
    rgb = protocol.raw["input_contract"]["rgb"]
    config.channel_order = str(rgb["channel_order"])
    config.normalization_identity = str(rgb["normalization_identity"])
    config.norm_mean = np.asarray(rgb["mean"], dtype=np.float32)
    config.norm_std = np.asarray(rgb["std"], dtype=np.float32)
    if config.channel_order != "RGB" or config.normalization_identity != "rgb-imagenet-rgb-order-v1":
        raise ValueError("runtime RGB contract differs from DVC-A1")
    return config


def _frozen_v1_condition_hashes(protocol: DvcProtocol, entries: Sequence[str]) -> dict[str, dict[str, str]]:
    source_manifest = protocol.source_mask_manifest_path
    if source_manifest is None:
        raise DvcProtocolError("frozen v1 mask manifest is required for this protocol")
    try:
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DvcProtocolError(f"cannot read frozen v1 mask manifest: {exc}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "museg-dvc-a1-mask-manifest-v1":
        raise DvcProtocolError("frozen v1 mask manifest has an unexpected schema")
    rows = manifest.get("samples")
    if not isinstance(rows, list):
        raise DvcProtocolError("frozen v1 mask manifest lacks sample rows")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DvcProtocolError("frozen v1 mask manifest contains a non-object sample row")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in rows_by_id:
            raise DvcProtocolError("frozen v1 mask manifest sample identities are missing or duplicated")
        rows_by_id[sample_id] = row
    frozen: dict[str, dict[str, str]] = {}
    for entry in entries:
        sample_id = Path(entry).stem
        row = rows_by_id.get(sample_id)
        if row is None:
            raise DvcProtocolError(f"frozen v1 mask manifest lacks allowlisted sample {sample_id}")
        hashes = row.get("condition_mask_sha256")
        if not isinstance(hashes, Mapping) or set(hashes) != set(CONDITION_IDS):
            raise DvcProtocolError(f"frozen v1 mask manifest has incomplete condition hashes for {sample_id}")
        frozen[sample_id] = {condition: str(hashes[condition]) for condition in CONDITION_IDS}
    return frozen


def _assert_mask_hashes_match_frozen_v1(
    computed: Mapping[str, Mapping[str, str]],
    frozen: Mapping[str, Mapping[str, str]],
) -> None:
    if set(computed) != set(frozen):
        raise DvcProtocolError("computed condition mask samples differ from the frozen v1 manifest")
    for sample_id in sorted(frozen):
        if set(computed[sample_id]) != set(CONDITION_IDS):
            raise DvcProtocolError(f"computed condition mask hashes are incomplete for {sample_id}")
        for condition in CONDITION_IDS:
            if computed[sample_id][condition] != frozen[sample_id][condition]:
                raise DvcProtocolError(
                    f"condition mask hash differs from the frozen v1 manifest for {sample_id} in {condition}"
                )


def _assert_paired_group_count(analysis: Mapping[str, Any], expected: int, name: str) -> None:
    if int(analysis.get("location_group_count", -1)) != expected:
        raise DvcProtocolError(f"{name} analysis group count differs from the frozen scope")
    for effect_name in ("dose_effect", "specificity_effect"):
        effect = analysis.get(effect_name)
        if not isinstance(effect, Mapping) or int(effect.get("paired_group_count", -1)) != expected:
            raise DvcProtocolError(f"{name} {effect_name} does not contain exactly {expected} valid paired groups")


def _verify_protocol_assets(protocol: DvcProtocol) -> list[str]:
    if file_sha256(protocol.checkpoint_path) != str(protocol.raw["model"]["checkpoint_sha256"]):
        raise DvcProtocolError("checkpoint hash no longer matches the materialized protocol")
    if file_sha256(protocol.split_path) != str(protocol.raw["split"]["sha256"]):
        raise DvcProtocolError("split hash no longer matches the materialized protocol")
    code_identity = protocol.raw["code_identity"]
    for relative_path, expected_sha in code_identity["source_sha256"].items():
        source = Path(__file__).resolve().parents[2] / relative_path
        if file_sha256(source) != expected_sha:
            raise DvcProtocolError(f"source changed after protocol materialization: {relative_path}")
    frozen_entries = split_entries(protocol.split_path)
    if len(frozen_entries) != int(protocol.raw["split"]["sample_count"]):
        raise DvcProtocolError("split sample count differs from protocol")
    frozen_groups = {location_group(Path(entry).stem) for entry in frozen_entries}
    if len(frozen_groups) != int(protocol.raw["split"]["location_group_count"]):
        raise DvcProtocolError("split location-group count differs from protocol")
    if protocol.raw["protocol_id"] == "DVC-A1-valdev-boundary-zero-v1":
        return frozen_entries
    scope = protocol.raw["evaluation_scope"]
    source_manifest = protocol.source_mask_manifest_path
    if source_manifest is None or file_sha256(source_manifest) != str(scope["source_mask_manifest_sha256"]):
        raise DvcProtocolError("v2 source mask manifest hash no longer matches protocol")
    if not protocol.evaluation_path.is_file() or file_sha256(protocol.evaluation_path) != str(scope["allowlist_sha256"]):
        raise DvcProtocolError("v2 evaluation allowlist hash no longer matches protocol")
    allowlist_summary_path = protocol.evidence_root / "allowlist-summary.json"
    if not allowlist_summary_path.is_file() or file_sha256(allowlist_summary_path) != str(scope["allowlist_summary_sha256"]):
        raise DvcProtocolError("v2 allowlist summary hash no longer matches protocol")
    entries = split_entries(protocol.evaluation_path)
    if len(entries) != int(scope["included_sample_count"]) or len(set(entries)) != len(entries):
        raise DvcProtocolError("v2 allowlist sample count or uniqueness differs from protocol")
    if not set(entries).issubset(set(frozen_entries)):
        raise DvcProtocolError("v2 allowlist contains samples outside frozen val-dev")
    groups = {location_group(Path(entry).stem) for entry in entries}
    if len(groups) != int(scope["included_location_group_count"]):
        raise DvcProtocolError("v2 allowlist location-group count differs from protocol")
    if any("official" in entry.lower() or "test" in entry.lower() for entry in entries):
        raise DvcProtocolError("v2 allowlist violates the official-test exclusion")
    return entries


def _mask_scan(protocol: DvcProtocol, entries: Sequence[str]) -> dict[str, Any]:
    group_constructable: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    nonboundary_shortages: list[dict[str, str]] = []
    for entry in entries:
        sample_id = Path(entry).stem
        group = location_group(sample_id)
        group_constructable.setdefault(group, False)
        depth16 = cv2.imread(str(_sample_paths(protocol, sample_id)["depth16"]), cv2.IMREAD_UNCHANGED)
        if depth16 is None or depth16.dtype != np.uint16 or depth16.ndim != 2:
            raise ValueError(f"invalid Depth16 during mask scan for {sample_id}")
        try:
            masks = build_corruption_masks(
                depth16,
                sample_id,
                seed=int(protocol.raw["corruption"]["seed"]),
                relative_jump_threshold=float(protocol.raw["corruption"]["relative_jump_threshold"]),
            )
        except ValueError as exc:
            if "nonboundary-q50 shortage" not in str(exc):
                raise
            nonboundary_shortages.append(
                {"sample_id": sample_id, "location_group": group, "error": str(exc)}
            )
            continue
        q75_constructable = masks.condition_counts["boundary-q75"] > 0
        group_constructable[group] = group_constructable[group] or q75_constructable
        rows.append(
            {
                "sample_id": sample_id,
                "location_group": group,
                "boundary_candidate_count": masks.boundary_candidate_count,
                "nonboundary_candidate_count": masks.nonboundary_candidate_count,
                "condition_counts": dict(masks.condition_counts),
                "condition_mask_sha256": dict(masks.condition_sha256),
            }
        )
    unconstructable = sorted(group for group, constructable in group_constructable.items() if not constructable)
    fraction = len(unconstructable) / len(group_constructable)
    limit = float(protocol.raw["corruption"]["q75_unconstructable_location_group_fraction_limit"])
    gate_passed = not nonboundary_shortages and fraction <= limit
    return {
        "schema_version": f"museg-dvc-a1-mask-manifest-{_protocol_version(protocol)}",
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_sha256": protocol.sha256,
        "official_test_included": False,
        "status": "passed" if gate_passed else "protocol-blocked",
        "sample_count": len(entries),
        "mask_complete_sample_count": len(rows),
        "location_group_count": len(group_constructable),
        "q75_unconstructable_location_groups": unconstructable,
        "q75_unconstructable_location_group_fraction": fraction,
        "q75_unconstructable_location_group_fraction_limit": limit,
        "nonboundary_shortages": nonboundary_shortages,
        "gate_passed": gate_passed,
        "samples": rows,
    }


def _infer_condition(
    *,
    protocol: DvcProtocol,
    config: Any,
    model: torch.nn.Module,
    device: torch.device,
    entries: Sequence[str],
    condition: str,
    expected_mask_sha256: Mapping[str, str],
    progress_started: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    hist = np.zeros((int(config.num_classes), int(config.num_classes)), dtype=np.int64)
    total_samples = len(entries)
    for sample_index, entry in enumerate(entries):
        sample_id = Path(entry).stem
        rgb, depth16, _, label, label_raw = _read_sample(protocol, sample_id)
        masks = build_corruption_masks(
            depth16,
            sample_id,
            seed=int(protocol.raw["corruption"]["seed"]),
            relative_jump_threshold=float(protocol.raw["corruption"]["relative_jump_threshold"]),
        )
        if masks.condition_sha256[condition] != expected_mask_sha256[sample_id]:
            raise RuntimeError(f"mask hash differs from the frozen run manifest for {sample_id} in {condition}")
        depth = quantized_condition_depth(depth16, masks.masks[condition], 13932)
        views = build_msflip_views(rgb, depth, config)
        views = [
            {
                **view,
                "rgb": view["rgb"].to(device, non_blocking=True),
                "depth": view["depth"].to(device, non_blocking=True),
            }
            for view in views
        ]
        logits, _ = msflip_whole_logits(
            model,
            views,
            original_size_hw=label.shape,
            amp=False,
        )
        if tuple(logits.shape[-2:]) != label.shape or not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError(f"non-finite or wrong-grid logits for {sample_id} in {condition}")
        prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
        sample_hist = confusion_matrix(prediction, label, int(config.num_classes), int(config.background))
        hist += sample_hist
        boundary = semantic_boundary_iou(
            prediction,
            _boundary_target(protocol, label, label_raw),
            num_classes=int(config.num_classes),
            ignore_label=int(config.background),
            background_label=int(config.num_classes) if protocol.raw["protocol_id"] == PROTOCOL_V3_ID else None,
            distance_ratio=float(protocol.raw["boundary_iou"]["distance_ratio_of_image_diagonal"]),
        )
        records.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "location_group": location_group(sample_id),
                "mine": mine_id(sample_id),
                "height": int(label.shape[0]),
                "width": int(label.shape[1]),
                "boundary_iou": boundary["value"],
                "boundary_distance_pixels": boundary["distance_pixels"],
                "boundary_iou_per_class": boundary["per_class"],
                "image_miou": image_miou_from_confusion(sample_hist),
                "valid_depth_ratio_before": float(np.mean(depth16 > 0)),
                "valid_depth_ratio_after": float(np.mean(depth > 0)),
                "zeroed_pixel_count": masks.condition_counts[condition],
                "mask_sha256": masks.condition_sha256[condition],
            }
        )
        if progress_started is not None:
            completed = sample_index + 1
            condition_index = CONDITION_IDS.index(condition) + 1
            total_conditions = len(CONDITION_IDS)
            overall_completed = (condition_index - 1) * total_samples + completed
            overall_total = total_conditions * total_samples
            elapsed = max(time.monotonic() - progress_started, 1e-9)
            rate = overall_completed / elapsed
            eta_seconds = (overall_total - overall_completed) / rate
            print(
                f"\r[v3] condition {condition_index}/{total_conditions} {condition}: "
                f"{completed}/{total_samples} samples | overall {overall_completed}/{overall_total} "
                f"({overall_completed / overall_total:.1%}) | elapsed {elapsed / 60:.1f} min | "
                f"ETA {eta_seconds / 60:.1f} min",
                end="",
                flush=True,
            )
            if overall_completed == overall_total:
                print()
    aggregate = aggregate_condition_records(records)
    return {
        "schema_version": f"museg-dvc-a1-condition-result-{_protocol_version(protocol)}",
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_sha256": protocol.sha256,
        "condition": condition,
        "status": "completed",
        "generated_at_utc": _utc_now(),
        "official_test_included": False,
        "sample_count": len(records),
        "location_group_count": aggregate["group_count"],
        "metrics_percent": metrics_from_confusion(hist, config.class_names),
        "per_image": records,
        "per_group": aggregate["per_group"],
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def _select_preflight_entries(protocol: DvcProtocol, entries: Sequence[str], sample_count: int) -> list[str]:
    if protocol.raw["protocol_id"] == "DVC-A1-valdev-boundary-zero-v1":
        return list(entries[:sample_count])
    if sample_count != 2:
        raise DvcProtocolError("v2 preflight requires exactly two samples")
    summary_path = protocol.evidence_root / "allowlist-summary.json"
    if not summary_path.is_file():
        raise DvcProtocolError("v2 preflight requires the materialized allowlist summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("allowlist_sha256") != protocol.raw["evaluation_scope"]["allowlist_sha256"]:
        raise DvcProtocolError("v2 allowlist summary does not match protocol")
    entry_by_id = {Path(entry).stem: entry for entry in entries}
    zero_ids = [str(value) for value in summary.get("q75_zero_sample_ids", [])]
    preferred_zero_id = "06-01-01-0346-230921160051-12-99" if protocol.raw["protocol_id"] == PROTOCOL_V3_ID else None
    zero_id = (
        preferred_zero_id
        if preferred_zero_id in entry_by_id and preferred_zero_id in zero_ids
        else next((sample_id for sample_id in zero_ids if sample_id in entry_by_id), None)
    )
    if zero_id is None:
        raise DvcProtocolError("v2 preflight cannot locate a q75-empty sample")
    zero_group = location_group(zero_id)
    source_manifest = protocol.source_mask_manifest_path
    if source_manifest is None:
        raise DvcProtocolError("scoped preflight lacks its source mask manifest")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest_rows = {str(row["sample_id"]): row for row in manifest["samples"]}
    preferred_nonempty_id = "06-01-01-0346-230921160052-12-99" if protocol.raw["protocol_id"] == PROTOCOL_V3_ID else None
    nonempty_id = (
        preferred_nonempty_id
        if preferred_nonempty_id in entry_by_id
        and preferred_nonempty_id in manifest_rows
        and location_group(preferred_nonempty_id) != zero_group
        and int(manifest_rows[preferred_nonempty_id]["condition_counts"]["boundary-q75"]) > 0
        else next(
            (
                str(row["sample_id"])
                for row in manifest["samples"]
                if str(row["sample_id"]) in entry_by_id
                and location_group(str(row["sample_id"])) != zero_group
                and int(row["condition_counts"]["boundary-q75"]) > 0
            ),
            None,
        )
    )
    if nonempty_id is None:
        raise DvcProtocolError("v2 preflight cannot locate a q75-nonempty sample from another group")
    selected = [entry_by_id[zero_id], entry_by_id[nonempty_id]]
    return selected[:sample_count]


def run_preflight(protocol: DvcProtocol, *, device: torch.device, sample_count: int = 2) -> dict[str, Any]:
    entries = _verify_protocol_assets(protocol)
    config = _config(protocol)
    selected = _select_preflight_entries(protocol, entries, sample_count)
    frozen_hashes = (
        _frozen_v1_condition_hashes(protocol, selected)
        if protocol.raw["protocol_id"] != "DVC-A1-valdev-boundary-zero-v1"
        else None
    )
    checks: list[dict[str, Any]] = []
    for entry in selected:
        sample_id = Path(entry).stem
        _, depth16, depth8, _, _ = _read_sample(protocol, sample_id)
        first = build_corruption_masks(
            depth16,
            sample_id,
            seed=int(protocol.raw["corruption"]["seed"]),
            relative_jump_threshold=float(protocol.raw["corruption"]["relative_jump_threshold"]),
        )
        second = build_corruption_masks(
            depth16,
            sample_id,
            seed=int(protocol.raw["corruption"]["seed"]),
            relative_jump_threshold=float(protocol.raw["corruption"]["relative_jump_threshold"]),
        )
        if frozen_hashes is not None:
            _assert_mask_hashes_match_frozen_v1(
                {sample_id: dict(first.condition_sha256)},
                {sample_id: frozen_hashes[sample_id]},
            )
        checks.append(
            {
                "sample_id": sample_id,
                "location_group": location_group(sample_id),
                "boundary_q75_constructable": first.condition_counts["boundary-q75"] > 0,
                "condition_counts": dict(first.condition_counts),
                "condition_mask_sha256": dict(first.condition_sha256),
                "mask_hash_stable": first.condition_sha256 == second.condition_sha256,
                "mask_hash_matches_frozen_v1": frozen_hashes is None
                or first.condition_sha256 == frozen_hashes[sample_id],
                "nested": bool(
                    np.all(first.masks["boundary-q25"] <= first.masks["boundary-q50"])
                    and np.all(first.masks["boundary-q50"] <= first.masks["boundary-q75"])
                ),
                "matched_q50_count": first.condition_counts["boundary-q50"] == first.condition_counts["nonboundary-q50"],
                "q0_decoded_array_equal": bool(np.array_equal(quantized_condition_depth(depth16, first.masks["clean"]), depth8)),
            }
        )
    boolean_keys = (
        "mask_hash_stable",
        "mask_hash_matches_frozen_v1",
        "nested",
        "matched_q50_count",
        "q0_decoded_array_equal",
    )
    if not all(all(bool(row[key]) for key in boolean_keys) for row in checks):
        raise ValueError("one or more deterministic mask or q=0 checks failed")
    if protocol.raw["protocol_id"] != "DVC-A1-valdev-boundary-zero-v1":
        if len({str(row["location_group"]) for row in checks}) != 2:
            raise ValueError("v2 preflight samples must come from two location groups")
        if {bool(row["boundary_q75_constructable"]) for row in checks} != {False, True}:
            raise ValueError("v2 preflight must cover one q75-empty and one q75-nonempty sample")
    configure_fp32_forward(device)
    model = load_model(config, protocol.checkpoint_path, device)
    inference: list[dict[str, Any]] = []
    for entry in selected:
        sample_id = Path(entry).stem
        rgb, depth16, _, label, label_raw = _read_sample(protocol, sample_id)
        masks = build_corruption_masks(
            depth16,
            sample_id,
            seed=int(protocol.raw["corruption"]["seed"]),
            relative_jump_threshold=float(protocol.raw["corruption"]["relative_jump_threshold"]),
        )
        for condition in ("clean", "boundary-q50"):
            depth = quantized_condition_depth(depth16, masks.masks[condition])
            views = [
                {
                    **view,
                    "rgb": view["rgb"].to(device),
                    "depth": view["depth"].to(device),
                }
                for view in build_msflip_views(rgb, depth, config)
            ]
            logits, _ = msflip_whole_logits(model, views, original_size_hw=label.shape, amp=False)
            prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
            boundary = semantic_boundary_iou(
                prediction,
                _boundary_target(protocol, label, label_raw),
                num_classes=int(config.num_classes),
                ignore_label=int(config.background),
                background_label=int(config.num_classes) if protocol.raw["protocol_id"] == PROTOCOL_V3_ID else None,
                distance_ratio=float(protocol.raw["boundary_iou"]["distance_ratio_of_image_diagonal"]),
            )
            inference.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "finite": bool(torch.isfinite(logits).all().item()),
                    "output_size_hw": list(logits.shape[-2:]),
                    "expected_size_hw": list(label.shape),
                    "boundary_iou": boundary["value"],
                    "boundary_iou_defined_class_count": boundary["defined_class_count"],
                }
            )
    if not all(row["finite"] and row["output_size_hw"] == row["expected_size_hw"] for row in inference):
        raise RuntimeError("preflight inference produced non-finite or wrong-grid output")
    if protocol.raw["protocol_id"] == PROTOCOL_V3_ID and not all(
        int(row["boundary_iou_defined_class_count"]) > 0 and row["boundary_iou"] is not None
        for row in inference
    ):
        raise DvcProtocolError("v3 background-context Boundary IoU is undefined in preflight")
    version = _protocol_version(protocol)
    report = {
        "schema_version": f"museg-dvc-a1-preflight-{version}",
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_sha256": protocol.sha256,
        "status": "passed",
        "generated_at_utc": _utc_now(),
        "official_test_included": False,
        "device": str(device),
        "evaluation_sample_count": len(entries),
        "evaluation_location_group_count": len({location_group(Path(entry).stem) for entry in entries}),
        "checks": checks,
        "inference": inference,
    }
    _atomic_write_json(protocol.evidence_root / "preflight.json", report)
    return report


def run_full(protocol: DvcProtocol, *, device: torch.device) -> dict[str, Any]:
    entries = _verify_protocol_assets(protocol)
    preflight_path = protocol.evidence_root / "preflight.json"
    if not preflight_path.is_file():
        raise DvcProtocolError("full evaluation requires a completed preflight report")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("protocol_sha256") != protocol.sha256:
        raise DvcProtocolError("preflight did not pass for this exact protocol")
    is_v2 = protocol.raw["protocol_id"] != "DVC-A1-valdev-boundary-zero-v1"
    scan = _mask_scan(protocol, entries)
    _atomic_write_json(protocol.evidence_root / "mask-manifest.json", scan)
    if not scan["gate_passed"]:
        raise ValueError(
            "mask protocol gate failed: "
            f"q75 unconstructable for {len(scan['q75_unconstructable_location_groups'])}/"
            f"{scan['location_group_count']} groups (limit {scan['q75_unconstructable_location_group_fraction_limit']:.3f}); "
            f"nonboundary shortages={len(scan['nonboundary_shortages'])}"
        )
    scan_hashes = {
        str(row["sample_id"]): dict(row["condition_mask_sha256"])
        for row in scan["samples"]
    }
    frozen_hashes = _frozen_v1_condition_hashes(protocol, entries) if is_v2 else None
    if frozen_hashes is not None:
        _assert_mask_hashes_match_frozen_v1(scan_hashes, frozen_hashes)
    config = _config(protocol)
    configure_fp32_forward(device)
    model = load_model(config, protocol.checkpoint_path, device)
    condition_results: dict[str, dict[str, Any]] = {}
    full_started = time.monotonic()
    for condition in CONDITION_IDS:
        result = _infer_condition(
            protocol=protocol,
            config=config,
            model=model,
            device=device,
            entries=entries,
            condition=condition,
            expected_mask_sha256={sample_id: hashes[condition] for sample_id, hashes in scan_hashes.items()},
            progress_started=full_started,
        )
        condition_path = protocol.evidence_root / "conditions" / f"{condition}.json"
        _atomic_write_json(condition_path, result)
        condition_results[condition] = result
    group_values: dict[str, dict[str, float | None]] = {}
    group_mines: dict[str, str] = {}
    for condition, result in condition_results.items():
        for row in result["per_group"]:
            group = str(row["location_group"])
            group_values.setdefault(group, {})[condition] = row["boundary_iou"]
            group_mines[group] = str(row["mine"])
    statistics = protocol.raw["statistics"]
    fully: set[str] | None = None
    partial: set[str] | None = None
    if is_v2:
        scope = protocol.raw["evaluation_scope"]
        allowlist_summary = json.loads((protocol.evidence_root / "allowlist-summary.json").read_text(encoding="utf-8"))
        fully = {str(value) for value in allowlist_summary["fully_constructable_location_groups"]}
        partial = {str(value) for value in allowlist_summary["partially_constructable_location_groups"]}
        expected_primary_count = int(scope["included_location_group_count"])
        if len(group_values) != expected_primary_count:
            raise DvcProtocolError("v2 primary analysis does not contain exactly 138 location groups")
        if (
            len(fully) != int(scope["sensitivity_location_group_count"])
            or len(partial) != int(scope["partially_constructable_location_group_count"])
            or not fully.isdisjoint(partial)
            or fully | partial != set(group_values)
        ):
            raise DvcProtocolError("v2 constructability analysis groups differ from the frozen evaluation scope")
    else:
        expected_primary_count = None

    def effects_for(groups: set[str], *, expected_paired_group_count: int | None = None) -> dict[str, Any]:
        values = {group: group_values[group] for group in sorted(groups)}
        dose = paired_effect(
            values,
            "boundary-q75",
            "clean",
            bootstrap_seed=int(statistics["bootstrap_seed"]),
            bootstrap_replicates=int(statistics["bootstrap_replicates"]),
        )
        specificity = paired_effect(
            values,
            "boundary-q50",
            "nonboundary-q50",
            bootstrap_seed=int(statistics["bootstrap_seed"]),
            bootstrap_replicates=int(statistics["bootstrap_replicates"]),
        )
        analysis = {"location_group_count": len(groups), "dose_effect": dose, "specificity_effect": specificity}
        if expected_paired_group_count is not None:
            _assert_paired_group_count(analysis, expected_paired_group_count, "v2 analysis")
        return analysis

    primary = effects_for(set(group_values), expected_paired_group_count=expected_primary_count)
    decision = adjudicate(primary["dose_effect"], primary["specificity_effect"], group_mines)
    analyses: dict[str, Any] = {"primary": {**primary, "adjudication": decision}}
    if is_v2:
        assert fully is not None and partial is not None
        analyses["fully_constructable_sensitivity"] = effects_for(
            fully,
            expected_paired_group_count=int(protocol.raw["evaluation_scope"]["sensitivity_location_group_count"]),
        )
        analyses["partially_constructable_descriptive"] = effects_for(partial)
    version = _protocol_version(protocol)
    summary = {
        "schema_version": f"museg-dvc-a1-summary-{version}",
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol.sha256,
        "status": "completed",
        "generated_at_utc": _utc_now(),
        "official_test_included": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "tf32_enabled": False if device.type == "cuda" else None,
        },
        "condition_files": {
            condition: {
                "path": str(protocol.evidence_root / "conditions" / f"{condition}.json"),
                "sha256": file_sha256(protocol.evidence_root / "conditions" / f"{condition}.json"),
                "metrics_percent": result["metrics_percent"],
            }
            for condition, result in condition_results.items()
        },
        "mask_manifest": {
            "path": str(protocol.evidence_root / "mask-manifest.json"),
            "sha256": file_sha256(protocol.evidence_root / "mask-manifest.json"),
        },
        "analyses": analyses,
        "adjudication": decision,
    }
    _atomic_write_json(protocol.evidence_root / "summary.json", summary)
    return summary


def _write_execution_record(
    protocol: DvcProtocol,
    *,
    mode: str,
    argv: Sequence[str],
    started_at_utc: str,
    duration_seconds: float,
    exit_code: int,
    status: str,
    error: Exception | None = None,
) -> None:
    timestamp = started_at_utc.replace(":", "").replace("-", "").replace("+00:00", "Z").replace(".", "")
    record: dict[str, Any] = {
        "schema_version": f"museg-dvc-a1-execution-{_protocol_version(protocol)}",
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_sha256": protocol.sha256,
        "mode": mode,
        "argv": list(argv),
        "started_at_utc": started_at_utc,
        "duration_seconds": round(duration_seconds, 3),
        "exit_code": exit_code,
        "status": status,
        "official_test_included": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    if error is not None:
        record.update({"error_type": type(error).__name__, "error": str(error)})
    _atomic_write_json(protocol.evidence_root / "executions" / f"{timestamp}-{mode}.json", record)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-samples", type=int, choices=(1, 2), default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(argv)
    started_at_utc = _utc_now()
    started = time.monotonic()
    protocol: DvcProtocol | None = None
    try:
        protocol = load_dvc_protocol(args.protocol)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        result = (
            run_preflight(protocol, device=device, sample_count=args.preflight_samples)
            if args.mode == "preflight"
            else run_full(protocol, device=device)
        )
        duration = time.monotonic() - started
        _write_execution_record(
            protocol,
            mode=args.mode,
            argv=effective_argv,
            started_at_utc=started_at_utc,
            duration_seconds=duration,
            exit_code=0,
            status=result["status"],
        )
        print(json.dumps({"status": result["status"], "mode": args.mode, "duration_seconds": round(duration, 3)}, ensure_ascii=False))
        return 0
    except (DvcProtocolError, FileNotFoundError, OSError, RuntimeError, ValueError, ModuleNotFoundError) as exc:
        if protocol is None:
            try:
                protocol = load_dvc_protocol(args.protocol)
            except Exception:
                protocol = None
        if protocol is not None:
            failure_status = "protocol-blocked" if isinstance(exc, (DvcProtocolError, FileNotFoundError, ValueError)) else "stop"
            output = protocol.evidence_root / f"{args.mode}-failure.json"
            _atomic_write_json(
                output,
                {
                    "schema_version": f"museg-dvc-a1-failure-{_protocol_version(protocol)}",
                    "protocol_id": protocol.raw["protocol_id"],
                    "protocol_sha256": protocol.sha256,
                    "mode": args.mode,
                    "status": failure_status,
                    "generated_at_utc": _utc_now(),
                    "official_test_included": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            _write_execution_record(
                protocol,
                mode=args.mode,
                argv=effective_argv,
                started_at_utc=started_at_utc,
                duration_seconds=time.monotonic() - started,
                exit_code=2,
                status=failure_status,
                error=exc,
            )
        print(json.dumps({"status": "failed", "mode": args.mode, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
