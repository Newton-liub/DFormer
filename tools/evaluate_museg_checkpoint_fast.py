#!/usr/bin/env python3
"""Protocol-equivalent FP32 clean evaluator with optional view micro-batching.

The frozen evaluator remains the authority for view construction, single-view
post-processing, metric restoration, and confusion updates.  This module imports it
as ``EV`` and only changes scheduling: in optimized mode two views from one scale can
share one model forward.  No AMP, TF32, scale reduction, interpolation change, fusion
change, or metric change is permitted here.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import importlib
import json
import math
import os
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tools.evaluate_museg_checkpoint as EV
from utils.dataloader.multimodal_failure_v3 import FailureSpec, apply_failures


DEFAULT_CONFIG = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3"
DEFAULT_MAX_SAMPLES = 8
EVALUATION_SEED = 2026091401
LOGIT_ABS_TOLERANCE = 1.0e-4
METRIC_NOISE_TOLERANCE = 1.0e-6

# The values below are copied from the frozen local technical-check artifact.  They
# are evidence anchors, not runtime assumptions or protocol overrides.
OFFICIAL_TECHNICAL_ANCHOR = {
    "sample_id": "06-01-01-0352-230920140646-10-99",
    "original_size_hw": [932, 1082],
    "scale": 1.5,
    "scaled_size_hw": [1398, 1623],
    "padded_size_hw": [1408, 1632],
    "elapsed_seconds": {"original": 1.894933, "horizontal_flip": 0.975172},
    "peak_allocated_bytes": 4977043456,
}

CORRUPTION_CONDITIONS: tuple[tuple[str, tuple[FailureSpec, ...]], ...] = (
    ("spatial_dropout@0.75", (FailureSpec("depth", "spatial_dropout", 0.75),)),
    ("gaussian_noise@0.75", (FailureSpec("depth", "gaussian_noise", 0.75),)),
    ("blur@0.75", (FailureSpec("depth", "blur", 0.75),)),
    ("quantization@0.75", (FailureSpec("depth", "quantization", 0.75),)),
    ("misalignment@0.75", (FailureSpec("depth", "misalignment", 0.75),)),
    ("entire_missing@1.0", (FailureSpec("depth", "entire_missing", 1.0),)),
    (
        "spatial_dropout@0.5+gaussian_noise@0.5",
        (
            FailureSpec("depth", "spatial_dropout", 0.5),
            FailureSpec("depth", "gaussian_noise", 0.5),
        ),
    ),
    (
        "blur@0.5+misalignment@0.5",
        (
            FailureSpec("depth", "blur", 0.5),
            FailureSpec("depth", "misalignment", 0.5),
        ),
    ),
    (
        "quantization@0.5+misalignment@0.5",
        (
            FailureSpec("depth", "quantization", 0.5),
            FailureSpec("depth", "misalignment", 0.5),
        ),
    ),
)


def identity_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Return one raw dataset item without default-collate metadata conversion."""
    if len(batch) != 1:
        raise ValueError("the clean evaluator is frozen to one sample per DataLoader batch")
    return batch[0]


class ForwardTimer(nn.Module):
    """Transparent model proxy that records forward-only CUDA-event durations."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self._records: list[tuple[Any, Any, int, float | None, float | None]] = []

    def reset(self) -> None:
        self._records.clear()

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        batch_size = int(rgb.shape[0])
        if rgb.device.type == "cuda":
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            logits = self.inner(rgb, depth)
            ended.record()
            self._records.append((started, ended, batch_size, None, None))
            return logits
        started_cpu = time.perf_counter()
        logits = self.inner(rgb, depth)
        ended_cpu = time.perf_counter()
        self._records.append((None, None, batch_size, started_cpu, ended_cpu))
        return logits

    def durations_seconds(self) -> list[float]:
        durations: list[float] = []
        for started, ended, _batch_size, started_cpu, ended_cpu in self._records:
            if started is not None and ended is not None:
                durations.append(float(started.elapsed_time(ended)) / 1000.0)
            else:
                assert started_cpu is not None and ended_cpu is not None
                durations.append(float(ended_cpu - started_cpu))
        return durations

    def batch_sizes(self) -> list[int]:
        return [record[2] for record in self._records]


def _command_line(argv: Sequence[str] | None) -> str:
    values = list(sys.argv[1:] if argv is None else argv)
    return "python -m tools.evaluate_museg_checkpoint_fast" + (" " + shlex.join(values) if values else "")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    )


def _restore_rng(
    state: tuple[torch.Tensor, list[torch.Tensor] | None],
    device: torch.device,
) -> None:
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _peak_memory(device: torch.device) -> dict[str, int | float | None]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "allocated_mib": None,
            "reserved_mib": None,
        }
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "allocated_mib": round(allocated / (1024.0 * 1024.0), 3),
        "reserved_mib": round(reserved / (1024.0 * 1024.0), 3),
    }


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _device_memory_info(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"total_mib": None, "free_mib": None}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "total_mib": round(float(total_bytes) / (1024.0 * 1024.0), 3),
        "free_mib": round(float(free_bytes) / (1024.0 * 1024.0), 3),
    }


def _memory_margin(device: torch.device, peak: dict[str, int | float | None]) -> dict[str, float | None]:
    info = _device_memory_info(device)
    total = info["total_mib"]
    if total is None:
        return {"total_mib": None, "peak_allocated_margin_mib": None, "peak_reserved_margin_mib": None}
    allocated = peak.get("allocated_mib")
    reserved = peak.get("reserved_mib")
    return {
        "total_mib": total,
        "peak_allocated_margin_mib": round(total - float(allocated), 3) if allocated is not None else None,
        "peak_reserved_margin_mib": round(total - float(reserved), 3) if reserved is not None else None,
    }


def _select_entries(
    dataset_root: Path,
    entries: Sequence[str],
    selection: str,
    max_samples: int,
) -> list[str]:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not entries:
        raise ValueError("the validation split is empty")
    limit = min(int(max_samples), len(entries))
    if selection == "first":
        return list(entries[:limit])
    if selection == "stratified":
        positions = np.linspace(0, len(entries) - 1, num=limit, dtype=np.int64)
        selected: list[str] = []
        for position in positions.tolist():
            entry = entries[int(position)]
            if entry not in selected:
                selected.append(entry)
        return selected
    if selection == "largest":
        ranked: list[tuple[int, int, int, str]] = []
        for entry in entries:
            sample_id = Path(entry).stem
            label_path = dataset_root / "Label" / f"{sample_id}.png"
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise FileNotFoundError(f"missing label for sample {sample_id}")
            height, width = label.shape
            ranked.append((height * width, height, width, entry))
        ranked.sort(reverse=True)
        return [entry for _pixels, _height, _width, entry in ranked[:limit]]
    raise ValueError(f"unsupported sample selection: {selection}")


def _loader_options(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    options: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "collate_fn": identity_collate,
    }
    if args.num_workers > 0:
        options["persistent_workers"] = bool(args.persistent_workers)
        options["prefetch_factor"] = int(args.prefetch_factor)
    return options


def _sample_size(sample: dict[str, object]) -> tuple[int, int]:
    label = sample["label"]
    if not isinstance(label, torch.Tensor):
        raise TypeError("dataset label must be a Tensor")
    if label.ndim != 2:
        raise ValueError(f"dataset label must be HxW, got shape {tuple(label.shape)}")
    return int(label.shape[0]), int(label.shape[1])


def _transfer_reference_sample(
    sample: dict[str, object],
    device: torch.device,
    non_blocking: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
    label = sample["label"]
    views = sample["views"]
    if not isinstance(label, torch.Tensor) or not isinstance(views, list):
        raise TypeError("msflip sample has an invalid structure")
    _sync(device)
    started = time.perf_counter()
    labels = label.unsqueeze(0).to(device, non_blocking=non_blocking)
    device_views: list[dict[str, Any]] = []
    for view in views:
        device_views.append(
            {
                **view,
                "rgb": view["rgb"].to(device, non_blocking=non_blocking),
                "depth": view["depth"].to(device, non_blocking=non_blocking),
            }
        )
    _sync(device)
    return labels, device_views, time.perf_counter() - started


def _run_reference_sample(
    model: ForwardTimer,
    sample: dict[str, object],
    device: torch.device,
    num_classes: int,
    ignore_label: int,
    class_names: Sequence[str],
    *,
    non_blocking: bool,
) -> dict[str, Any]:
    """Run the frozen path one view at a time to keep the reference memory bound."""
    label = sample["label"]
    views = sample["views"]
    if not isinstance(label, torch.Tensor) or not isinstance(views, list):
        raise TypeError("msflip sample has an invalid structure")
    original_size = _sample_size(sample)
    _sync(device)
    label_started = time.perf_counter()
    labels = label.unsqueeze(0).to(device, non_blocking=non_blocking)
    _sync(device)
    h2d_seconds = time.perf_counter() - label_started
    model.reset()
    fused: torch.Tensor | None = None
    enriched_records: list[dict[str, Any]] = []
    view_started_wall = time.perf_counter()
    previous_scale: float | None = None
    for view in views:
        current_scale = float(view["scale"])
        if device.type == "cuda" and current_scale != previous_scale:
            # Release allocator blocks between scales without changing any tensor math.
            torch.cuda.empty_cache()
        previous_scale = current_scale
        _sync(device)
        transfer_started = time.perf_counter()
        device_view = {
            **view,
            "rgb": view["rgb"].to(device, non_blocking=non_blocking),
            "depth": view["depth"].to(device, non_blocking=non_blocking),
        }
        _sync(device)
        h2d_seconds += time.perf_counter() - transfer_started
        call_started = time.perf_counter()
        one_view_logits, one_view_records = EV.msflip_whole_logits(
            model,
            [device_view],
            original_size_hw=original_size,
            amp=False,
            measure_timing=True,
        )
        _ = time.perf_counter() - call_started
        if len(one_view_records) != 1:
            raise RuntimeError("frozen one-view reference call returned an unexpected record count")
        forward_seconds = model.durations_seconds()[-1]
        record = dict(one_view_records[0])
        elapsed = float(record.get("elapsed_seconds", 0.0))
        record["forward_seconds"] = round(forward_seconds, 6)
        record["postprocess_seconds"] = round(max(0.0, elapsed - forward_seconds), 6)
        record["timing_scope"] = "single_view"
        enriched_records.append(record)
        fused = one_view_logits if fused is None else fused + one_view_logits
        del device_view, one_view_logits
    _sync(device)
    view_wall_seconds = time.perf_counter() - view_started_wall
    assert fused is not None
    fused = fused / float(len(views))

    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    metric_started = time.perf_counter()
    restored = EV.restore_logits_to_metric_grid(fused, labels)
    EV.update_confusion(hist, restored, labels, num_classes, ignore_label)
    metric_seconds = time.perf_counter() - metric_started
    metrics = EV.metrics_from_confusion(hist, class_names)
    ten_views_seconds = float(sum(float(record.get("elapsed_seconds", 0.0)) for record in enriched_records))
    return {
        "logits": restored,
        "hist": hist,
        "metrics": metrics,
        "view_records": enriched_records,
        "timings": {
            "h2d_seconds": round(h2d_seconds, 6),
            "ten_views_seconds": round(ten_views_seconds, 6),
            "view_wall_seconds": round(view_wall_seconds, 6),
            "forward_seconds": round(float(sum(model.durations_seconds())), 6),
            "postprocess_seconds": round(max(0.0, ten_views_seconds - sum(model.durations_seconds())), 6),
            "metric_seconds": round(metric_seconds, 6),
        },
    }


def _postprocess_group(
    logits: torch.Tensor,
    group: Sequence[dict[str, Any]],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Apply the frozen per-view geometry steps to one scale group.

    This deliberately follows EV.msflip_whole_logits in order: interpolate to padded,
    crop to scaled, unflip, interpolate to original, and return FP32 logits for fusion.
    """
    if not group:
        raise ValueError("optimized view group cannot be empty")
    scaled_size = tuple(int(value) for value in group[0]["scaled_size_hw"])
    padded_size = tuple(int(value) for value in group[0]["padded_size_hw"])
    if tuple(logits.shape[-2:]) != padded_size:
        logits = F.interpolate(logits, size=padded_size, mode="bilinear", align_corners=False)
    logits = logits[..., : scaled_size[0], : scaled_size[1]]
    fused_group: torch.Tensor | None = None
    for local_index, view in enumerate(group):
        view_logits = logits[local_index : local_index + 1]
        if bool(view["flipped"]):
            view_logits = torch.flip(view_logits, dims=(-1,))
        view_logits = F.interpolate(
            view_logits, size=original_size, mode="bilinear", align_corners=False
        )
        fused_group = view_logits if fused_group is None else fused_group + view_logits
    assert fused_group is not None
    return fused_group


def _run_optimized_sample(
    model: ForwardTimer,
    sample: dict[str, object],
    device: torch.device,
    num_classes: int,
    ignore_label: int,
    class_names: Sequence[str],
    *,
    view_batching: int,
    non_blocking: bool,
    batching_strategy: dict[float, int] | None = None,
) -> dict[str, Any]:
    if view_batching not in (1, 2):
        raise ValueError("optimized view batching must be 1 or 2")
    label = sample["label"]
    views = sample["views"]
    if not isinstance(label, torch.Tensor) or not isinstance(views, list):
        raise TypeError("msflip sample has an invalid structure")
    original_size = _sample_size(sample)
    groups: list[list[dict[str, Any]]] = []
    offset = 0
    while offset < len(views):
        scale = float(views[offset]["scale"])
        requested_grouping = int((batching_strategy or {}).get(scale, view_batching))
        group_size = 2 if requested_grouping == 2 and offset + 1 < len(views) else 1
        if group_size == 2 and float(views[offset + 1]["scale"]) != scale:
            group_size = 1
        groups.append(views[offset : offset + group_size])
        offset += group_size

    model.reset()
    h2d_cuda_events: list[tuple[Any, Any]] = []
    h2d_cpu_seconds: list[float] = []
    group_cuda_events: list[tuple[Any, Any]] = []
    group_cpu_seconds: list[float] = []
    group_metadata: list[dict[str, Any]] = []
    fused: torch.Tensor | None = None

    with torch.inference_mode():
        if device.type == "cuda":
            label_start = torch.cuda.Event(enable_timing=True)
            label_end = torch.cuda.Event(enable_timing=True)
            label_start.record()
            labels = label.unsqueeze(0).to(device, non_blocking=non_blocking)
            label_end.record()
            h2d_cuda_events.append((label_start, label_end))
        else:
            label_started = time.perf_counter()
            labels = label.unsqueeze(0).to(device, non_blocking=non_blocking)
            h2d_cpu_seconds.append(time.perf_counter() - label_started)

        for group in groups:
            rgb_cpu = torch.stack([view["rgb"] for view in group], dim=0)
            depth_cpu = torch.stack([view["depth"] for view in group], dim=0)
            if device.type == "cuda":
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                h2d_start.record()
                rgb = rgb_cpu.to(device, non_blocking=non_blocking)
                depth = depth_cpu.to(device, non_blocking=non_blocking)
                h2d_end.record()
                h2d_cuda_events.append((h2d_start, h2d_end))
                group_start = torch.cuda.Event(enable_timing=True)
                group_end = torch.cuda.Event(enable_timing=True)
                group_start.record()
                logits = model(rgb, depth)
                transformed = _postprocess_group(logits, group, original_size)
                group_end.record()
                group_cuda_events.append((group_start, group_end))
            else:
                h2d_started = time.perf_counter()
                rgb = rgb_cpu.to(device, non_blocking=non_blocking)
                depth = depth_cpu.to(device, non_blocking=non_blocking)
                h2d_cpu_seconds.append(time.perf_counter() - h2d_started)
                group_started = time.perf_counter()
                logits = model(rgb, depth)
                transformed = _postprocess_group(logits, group, original_size)
                group_cpu_seconds.append(time.perf_counter() - group_started)
            fused = transformed if fused is None else fused + transformed
            group_metadata.append(
                {
                    "scale": float(group[0]["scale"]),
                    "view_indices": [
                        int(index)
                        for index in range(len(group_metadata) * (2 if view_batching == 2 else 1),
                                           len(group_metadata) * (2 if view_batching == 2 else 1) + len(group))
                    ],
                    "flipped": [bool(view["flipped"]) for view in group],
                    "scaled_size_hw": list(group[0]["scaled_size_hw"]),
                    "padded_size_hw": list(group[0]["padded_size_hw"]),
                    "batch_size": len(group),
                }
            )
            del rgb_cpu, depth_cpu, rgb, depth, logits, transformed

        _sync(device)

    assert fused is not None
    h2d_seconds = (
        float(sum(start.elapsed_time(end) for start, end in h2d_cuda_events)) / 1000.0
        if device.type == "cuda"
        else float(sum(h2d_cpu_seconds))
    )
    group_seconds = (
        float(sum(start.elapsed_time(end) for start, end in group_cuda_events)) / 1000.0
        if device.type == "cuda"
        else float(sum(group_cpu_seconds))
    )
    forward_seconds = model.durations_seconds()
    if len(forward_seconds) != len(groups):
        raise RuntimeError("optimized forward timing count differs from scale-group count")
    postprocess_seconds = max(0.0, group_seconds - float(sum(forward_seconds)))

    view_records: list[dict[str, Any]] = []
    forward_offset = 0
    for group_index, (group, group_meta) in enumerate(zip(groups, group_metadata)):
        group_elapsed = (
            float(group_cuda_events[group_index][0].elapsed_time(group_cuda_events[group_index][1])) / 1000.0
            if device.type == "cuda"
            else float(group_cpu_seconds[group_index])
        )
        group_forward = float(forward_seconds[group_index])
        for local_index, view in enumerate(group):
            view_records.append(
                {
                    "scale": float(view["scale"]),
                    "flipped": bool(view["flipped"]),
                    "scaled_size_hw": list(view["scaled_size_hw"]),
                    "padded_size_hw": list(view["padded_size_hw"]),
                    "restored_size_hw": list(original_size),
                    "batch_size": len(group),
                    "batch_group_index": group_index,
                    "batch_group_elapsed_seconds": round(group_elapsed, 6),
                    "forward_group_seconds": round(group_forward, 6),
                    "postprocess_group_seconds": round(max(0.0, group_elapsed - group_forward), 6),
                    "elapsed_seconds": round(group_elapsed / len(group), 6),
                    "timing_scope": "batch_group_amortized",
                    "local_index": local_index,
                    "forward_record_index": forward_offset,
                }
            )
        forward_offset += 1

    fused = fused / float(len(views))
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    metric_started = time.perf_counter()
    restored = EV.restore_logits_to_metric_grid(fused, labels)
    EV.update_confusion(hist, restored, labels, num_classes, ignore_label)
    metric_seconds = time.perf_counter() - metric_started
    metrics = EV.metrics_from_confusion(hist, class_names)
    return {
        "logits": restored,
        "hist": hist,
        "metrics": metrics,
        "view_records": view_records,
        "timings": {
            "h2d_seconds": round(h2d_seconds, 6),
            "ten_views_seconds": round(group_seconds, 6),
            "view_wall_seconds": round(group_seconds, 6),
            "forward_seconds": round(float(sum(forward_seconds)), 6),
            "postprocess_seconds": round(postprocess_seconds, 6),
            "metric_seconds": round(metric_seconds, 6),
        },
    }


def _compare_results(
    reference: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    reference_logits = reference["logits"]
    optimized_logits = optimized["logits"]
    if reference_logits.shape != optimized_logits.shape:
        return {
            "status": "FAIL",
            "reason": "logit_shape_mismatch",
            "reference_shape": list(reference_logits.shape),
            "optimized_shape": list(optimized_logits.shape),
        }
    difference = (reference_logits - optimized_logits).abs()
    max_abs_diff = float(difference.max().detach().cpu().numpy())
    mean_abs_diff = float(difference.mean().detach().cpu().numpy())
    reference_prediction = reference_logits.argmax(dim=1)
    optimized_prediction = optimized_logits.argmax(dim=1)
    mismatched_pixels = int((reference_prediction != optimized_prediction).sum().detach().cpu().numpy())
    total_pixels = int(reference_prediction.numel())
    prediction_rate = 1.0 - (float(mismatched_pixels) / float(total_pixels)) if total_pixels else 1.0
    reference_metrics = reference["metrics"]
    optimized_metrics = optimized["metrics"]
    metric_differences = {
        name: float(optimized_metrics[name]) - float(reference_metrics[name])
        for name in ("miou", "macc", "mf1")
    }
    confusion_equal = bool(np.array_equal(reference["hist"], optimized["hist"]))
    bitwise_same = bool(torch.equal(reference_logits, optimized_logits))
    status = "PASS"
    failure_reasons: list[str] = []
    if mismatched_pixels:
        status = "FAIL"
        failure_reasons.append("argmax_prediction_mismatch")
    if not confusion_equal:
        status = "FAIL"
        failure_reasons.append("confusion_matrix_mismatch")
    if any(abs(value) > METRIC_NOISE_TOLERANCE for value in metric_differences.values()):
        status = "FAIL"
        failure_reasons.append("metric_difference_above_tolerance")
    if max_abs_diff > LOGIT_ABS_TOLERANCE and not mismatched_pixels:
        failure_reasons.append("logit_difference_above_advisory_tolerance")
    return {
        "status": status,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        "bitwise_same": bitwise_same,
        "argmax_prediction_consistency_rate": prediction_rate,
        "mismatched_prediction_pixels": mismatched_pixels,
        "total_prediction_pixels": total_pixels,
        "confusion_matrix_equal": confusion_equal,
        "metric_noise_tolerance": METRIC_NOISE_TOLERANCE,
        "metric_differences_optimized_minus_reference": metric_differences,
        "failure_reasons": failure_reasons,
    }


def _metrics_with_subset_label(hist: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    metrics = EV.metrics_from_confusion(hist, class_names)
    metrics["scope"] = "selected subset; not full val-dev"
    return metrics


def _format_peak(peak: dict[str, int | float | None]) -> dict[str, Any]:
    return dict(peak)


def _raw_arrays(dataset_root: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(dataset_root / "RGB" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(dataset_root / "Depth" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or depth is None:
        raise FileNotFoundError(f"missing RGB/Depth input for sample {sample_id}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"raw RGB/Depth geometry mismatch for sample {sample_id}")
    return rgb, depth


def _measure_corruption(
    dataset_root: Path,
    selected_entries: Sequence[str],
    *,
    evaluation_seed: int = EVALUATION_SEED,
    repeats: int = 3,
) -> dict[str, Any]:
    if repeats != 3:
        raise ValueError("the corruption cost measurement is fixed to three repeats")
    samples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for entry in selected_entries:
        sample_id = Path(entry).stem
        rgb, depth = _raw_arrays(dataset_root, sample_id)
        samples.append((sample_id, rgb, depth))
    condition_reports: list[dict[str, Any]] = []
    for condition_index, (condition_name, specs) in enumerate(CORRUPTION_CONDITIONS):
        repeat_ms: list[float] = []
        sample_ms: list[list[float]] = []
        for repeat in range(repeats):
            per_sample: list[float] = []
            for sample_index, (sample_id, rgb, depth) in enumerate(samples):
                rng_seed = int(evaluation_seed + condition_index * 100_000 + repeat * 1_000 + sample_index)
                rng = np.random.default_rng(rng_seed)
                depth_validity = depth > 0
                validity_mask = np.ones(depth.shape, dtype=bool)
                started = time.perf_counter()
                result = apply_failures(
                    rgb,
                    depth,
                    specs,
                    rng,
                    depth_validity=depth_validity,
                    validity_mask=validity_mask,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                # Keep the operation live until timing has ended, then release its arrays.
                if result.depth.shape != depth.shape:
                    raise RuntimeError("corruption result changed raw image geometry")
                per_sample.append(elapsed_ms)
                del result
            repeat_ms.append(float(np.mean(per_sample)))
            sample_ms.append(per_sample)
        all_values = [value for row in sample_ms for value in row]
        condition_reports.append(
            {
                "condition": condition_name,
                "specs": [
                    {"modality": spec.modality, "kind": spec.kind, "severity": spec.severity}
                    for spec in specs
                ],
                "repeat_mean_ms_per_sample": [round(value, 6) for value in repeat_ms],
                "mean_ms_per_sample": round(float(np.mean(all_values)), 6),
                "std_ms_per_sample": round(float(np.std(all_values, ddof=0)), 6),
                "sample_ms": [
                    {
                        "sample_id": samples[index][0],
                        "raw_size_hw": [int(samples[index][1].shape[0]), int(samples[index][1].shape[1])],
                        "repeat_ms": [round(row[index], 6) for row in sample_ms],
                    }
                    for index in range(len(samples))
                ],
            }
        )
    return {
        "status": "completed",
        "evaluation_seed": evaluation_seed,
        "repeats": repeats,
        "sample_count": len(samples),
        "basis": "utils.dataloader.multimodal_failure_v3.apply_failures",
        "validity_mask": "all raw pixels true",
        "depth_validity": "(raw uint8 Depth > 0) & validity_mask",
        "conditions": condition_reports,
    }


def _microbatch_probe(
    model: ForwardTimer,
    dataset: Any,
    selected_entry: str,
    device: torch.device,
    config: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if selected_entry in dataset.entries:
        sample = dataset[dataset.entries.index(selected_entry)]
    else:
        probe_dataset = EV.MUSegPostEvalDataset(
            dataset.dataset_root,
            [selected_entry],
            EV.MSFLIP_GEOMETRY,
            dataset.config,
            dataset.channel_order,
        )
        sample = probe_dataset[0]
    if not isinstance(sample, dict) or not isinstance(sample.get("views"), list):
        raise TypeError("microbatch probe requires an msflip sample")
    scales = list(EV.MSFLIP_SCALES)
    anchor_by_scale: dict[float, list[dict[str, Any]]] = {float(scale): [] for scale in scales}
    for view in sample["views"]:
        anchor_by_scale[float(view["scale"])].append(view)
    num_classes = int(getattr(args, "_num_classes"))
    class_names = list(getattr(args, "_class_names"))
    ignore_label = int(getattr(args, "_ignore_label"))
    measurements: dict[str, dict[str, Any]] = {"1": {}, "2": {}}
    stopped_after_oom = False
    for batch_size in (1, 2):
        for scale in scales:
            scale_key = str(scale)
            views = anchor_by_scale[float(scale)]
            if len(views) != 2:
                raise RuntimeError(f"scale {scale} does not have original+flip views")
            if batch_size == 2 and stopped_after_oom:
                measurements[str(batch_size)][scale_key] = {
                    "status": "skipped_after_prior_oom",
                    "scale": scale,
                    "padded_size_hw": list(views[0]["padded_size_hw"]),
                    "attempts": 0,
                    "empty_cache_before_each_repeat": device.type == "cuda",
                }
                continue
            repeat_values: list[dict[str, Any]] = []
            oom_error: str | None = None
            for repeat in range(int(args.microbatch_repeats)):
                # Keep the reset immediately outside the timed path.  The explicit
                # empty_cache call is required by the measurement protocol; peak
                # statistics are reset only after the allocator has been released.
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                cache_reset_performed = device.type == "cuda"
                memory_before_measurement = _device_memory_info(device)
                rng_state = _capture_rng(device)
                probe_sample = {**sample, "views": views}
                try:
                    result = _run_optimized_sample(
                        model,
                        probe_sample,
                        device,
                        num_classes,
                        ignore_label,
                        class_names,
                        view_batching=batch_size,
                        non_blocking=bool(args.pin_memory),
                    )
                    peak = _peak_memory(device)
                    repeat_values.append(
                        {
                            "repeat": repeat,
                            "empty_cache_before_measurement": cache_reset_performed,
                            "memory_before_measurement": memory_before_measurement,
                            "ten_views_seconds": result["timings"]["ten_views_seconds"],
                            "forward_seconds": result["timings"]["forward_seconds"],
                            "postprocess_seconds": result["timings"]["postprocess_seconds"],
                            "peak_memory": peak,
                            "memory_margin": _memory_margin(device, peak),
                        }
                    )
                    del result
                except torch.OutOfMemoryError as exc:
                    oom_error = str(exc)
                    stopped_after_oom = batch_size == 2
                    _sync(device)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    oom_peak = _peak_memory(device)
                    repeat_values.append(
                        {
                            "repeat": repeat,
                            "status": "OOM",
                            "empty_cache_before_measurement": cache_reset_performed,
                            "memory_before_measurement": memory_before_measurement,
                            "error": oom_error,
                            "peak_memory": oom_peak,
                            "memory_margin": _memory_margin(device, oom_peak),
                        }
                    )
                    _restore_rng(rng_state, device)
                    break
                finally:
                    _restore_rng(rng_state, device)
                    gc.collect()
            successful = [item for item in repeat_values if item.get("status", "completed") == "completed"]
            if oom_error is not None:
                measurements[str(batch_size)][scale_key] = {
                    "status": "OOM",
                    "scale": scale,
                    "padded_size_hw": list(views[0]["padded_size_hw"]),
                    "attempts": len(repeat_values),
                    "repeats": repeat_values,
                    "oom_error": oom_error,
                    "fallback": "batch1" if batch_size == 2 else None,
                }
            else:
                measurements[str(batch_size)][scale_key] = {
                    "status": "completed",
                    "scale": scale,
                    "padded_size_hw": list(views[0]["padded_size_hw"]),
                    "attempts": len(repeat_values),
                    "repeats": repeat_values,
                    "mean_ten_views_seconds": round(
                        float(np.mean([item["ten_views_seconds"] for item in successful])), 6
                    ),
                    "mean_peak_allocated_mib": round(
                        float(np.mean([item["peak_memory"]["allocated_mib"] for item in successful])), 3
                    ),
                    "mean_peak_reserved_mib": round(
                        float(np.mean([item["peak_memory"]["reserved_mib"] for item in successful])), 3
                    ),
                }
    strategy: dict[str, Any] = {}
    for scale in scales:
        key = str(scale)
        batch1 = measurements["1"].get(key, {})
        batch2 = measurements["2"].get(key, {})
        use_batch2 = (
            batch2.get("status") == "completed"
            and (
                batch1.get("status") != "completed"
                or float(batch2["mean_ten_views_seconds"]) < float(batch1["mean_ten_views_seconds"])
            )
            and float(batch2.get("mean_peak_reserved_mib", float("inf")))
            < float(_device_memory_info(device).get("total_mib", float("inf"))) * 0.90
        )
        strategy[key] = {
            "scale": scale,
            "padded_size_hw": batch2.get("padded_size_hw", batch1.get("padded_size_hw")),
            "selected_view_batching": 2 if use_batch2 else 1,
            "reason": (
                "batch2 completed, faster, and peak reserved memory stayed below 90% of device capacity"
                if use_batch2
                else "fallback to batch1 because batch2 was unavailable, slower, or lacked memory margin"
            ),
        }
    return {
        "status": "completed",
        "sample_id": Path(selected_entry).stem,
        "sample_entry": selected_entry,
        "repeats_requested": int(args.microbatch_repeats),
        "order": "batch1 scales small-to-large, then batch2 scales small-to-large",
        "cache_reset_policy": "torch.cuda.empty_cache() immediately before every repeat, followed by reset_peak_memory_stats",
        "empty_cache_before_each_repeat": device.type == "cuda",
        "oom_policy": "one OOM is recorded; later batch2 scales are skipped after the first OOM",
        "measurements": measurements,
        "fixed_strategy": strategy,
    }


def _geometry_anchor(
    model: ForwardTimer,
    dataset_root: Path,
    entries: Sequence[str],
    config: Any,
    channel_order: str,
    device: torch.device,
    num_classes: int,
    ignore_label: int,
    class_names: Sequence[str],
    *,
    non_blocking: bool,
) -> dict[str, Any]:
    entry, original_size = EV.select_largest_val_sample(dataset_root, entries)
    anchor_dataset = EV.MUSegPostEvalDataset(
        dataset_root,
        [entry],
        EV.MSFLIP_GEOMETRY,
        config,
        channel_order,
        msflip_scales=(1.5,),
    )
    sample = anchor_dataset[0]
    _reset_peak_memory(device)
    result = _run_reference_sample(
        model,
        sample,
        device,
        num_classes,
        ignore_label,
        class_names,
        non_blocking=non_blocking,
    )
    peak = _peak_memory(device)
    records = result["view_records"]
    by_flip = {"horizontal_flip" if record["flipped"] else "original": record for record in records}
    self_geometry = {
        "sample_id": Path(entry).stem,
        "original_size_hw": list(original_size),
        "scale": 1.5,
        "views": {
            key: {
                "scaled_size_hw": value["scaled_size_hw"],
                "padded_size_hw": value["padded_size_hw"],
                "elapsed_seconds": value.get("elapsed_seconds"),
            }
            for key, value in by_flip.items()
        },
        "peak_memory": peak,
    }
    geometry_match = (
        self_geometry["sample_id"] == OFFICIAL_TECHNICAL_ANCHOR["sample_id"]
        and self_geometry["original_size_hw"] == OFFICIAL_TECHNICAL_ANCHOR["original_size_hw"]
        and all(
            value["scaled_size_hw"] == OFFICIAL_TECHNICAL_ANCHOR["scaled_size_hw"]
            and value["padded_size_hw"] == OFFICIAL_TECHNICAL_ANCHOR["padded_size_hw"]
            for value in self_geometry["views"].values()
        )
    )
    return {
        "official_frozen_values": OFFICIAL_TECHNICAL_ANCHOR,
        "self_reference_values": self_geometry,
        "geometry_match": geometry_match,
        "timing_comparison_note": "elapsed values are local measurements; only geometry is an equivalence anchor",
    }


def _run_evaluation(
    args: argparse.Namespace,
    model: ForwardTimer,
    dataset: Any,
    selected_entries: Sequence[str],
    config: Any,
    device: torch.device,
) -> dict[str, Any]:
    options = _loader_options(args, device)
    warmup_ids: list[str] = []
    if args.warmup_samples > 0:
        warmup_loader = DataLoader(dataset, **options)
        warmup_iter = iter(warmup_loader)
        try:
            for _ in range(min(int(args.warmup_samples), len(selected_entries))):
                sample = next(warmup_iter)
                warmup_ids.append(str(sample["sample_id"]))
                if args.compare_reference:
                    state = _capture_rng(device)
                    _run_reference_sample(
                        model,
                        sample,
                        device,
                        int(config.num_classes),
                        int(config.background),
                        config.class_names,
                        non_blocking=bool(args.pin_memory),
                    )
                    _restore_rng(state, device)
                    _run_optimized_sample(
                        model,
                        sample,
                        device,
                        int(config.num_classes),
                        int(config.background),
                        config.class_names,
                        view_batching=int(args.view_batching),
                        non_blocking=bool(args.pin_memory),
                        batching_strategy=getattr(args, "_effective_strategy", None),
                    )
                elif args.mode == "reference":
                    _run_reference_sample(
                        model,
                        sample,
                        device,
                        int(config.num_classes),
                        int(config.background),
                        config.class_names,
                        non_blocking=bool(args.pin_memory),
                    )
                else:
                    _run_optimized_sample(
                        model,
                        sample,
                        device,
                        int(config.num_classes),
                        int(config.background),
                        config.class_names,
                        view_batching=int(args.view_batching),
                        non_blocking=bool(args.pin_memory),
                        batching_strategy=getattr(args, "_effective_strategy", None),
                    )
        finally:
            del warmup_iter, warmup_loader
            gc.collect()
    _reset_peak_memory(device)
    loader = DataLoader(dataset, **options)
    loader_init_started = time.perf_counter()
    iterator = iter(loader)
    loader_init_seconds = time.perf_counter() - loader_init_started
    reference_hist = np.zeros((int(config.num_classes), int(config.num_classes)), dtype=np.int64)
    optimized_hist = np.zeros_like(reference_hist)
    selected_results: list[dict[str, Any]] = []
    comparison_results: list[dict[str, Any]] = []
    reference_peak: dict[str, int | float | None] | None = None
    optimized_peak: dict[str, int | float | None] | None = None
    try:
        for sample_index in range(len(selected_entries)):
            sample_started = time.perf_counter()
            dataset_started = time.perf_counter()
            sample = next(iterator)
            dataset_seconds = time.perf_counter() - dataset_started
            sample_id = str(sample["sample_id"])
            if sample_id != Path(selected_entries[sample_index]).stem:
                raise RuntimeError("DataLoader sample order differs from selected split order")
            if args.compare_reference:
                # Run optimized first so the requested batch-2 path is tested before
                # the reference path can leave a large-scale allocator pattern behind.
                _reset_peak_memory(device)
                rng_state = _capture_rng(device)
                optimized = _run_optimized_sample(
                    model,
                    sample,
                    device,
                    int(config.num_classes),
                    int(config.background),
                    config.class_names,
                    view_batching=int(args.view_batching),
                    non_blocking=bool(args.pin_memory),
                    batching_strategy=getattr(args, "_effective_strategy", None),
                )
                optimized_peak_sample = _peak_memory(device)
                optimized["logits"] = optimized["logits"].detach().cpu()
                _restore_rng(rng_state, device)
                _reset_peak_memory(device)
                reference = _run_reference_sample(
                    model,
                    sample,
                    device,
                    int(config.num_classes),
                    int(config.background),
                    config.class_names,
                    non_blocking=bool(args.pin_memory),
                )
                reference_peak_sample = _peak_memory(device)
                reference["logits"] = reference["logits"].detach().cpu()
                comparison = _compare_results(reference, optimized)
                comparison["sample_id"] = sample_id
                comparison["reference_peak_memory"] = reference_peak_sample
                comparison["optimized_peak_memory"] = optimized_peak_sample
                comparison_results.append(comparison)
                reference_hist += reference["hist"]
                optimized_hist += optimized["hist"]
                reference_timing = reference["timings"]
                optimized_timing = optimized["timings"]
                selected_results.append(
                    {
                        "sample_id": sample_id,
                        "original_size_hw": list(_sample_size(sample)),
                        "dataset_seconds": round(dataset_seconds, 6),
                        "reference": {
                            "timings": reference_timing,
                            "view_records": reference["view_records"],
                            "metrics_percent": reference["metrics"],
                            "peak_memory": reference_peak_sample,
                        },
                        "optimized": {
                            "timings": optimized_timing,
                            "view_records": optimized["view_records"],
                            "metrics_percent": optimized["metrics"],
                            "peak_memory": optimized_peak_sample,
                        },
                        "total_seconds": round(time.perf_counter() - sample_started, 6),
                    }
                )
                del reference, optimized
            else:
                _reset_peak_memory(device)
                if args.mode == "reference":
                    result = _run_reference_sample(
                        model,
                        sample,
                        device,
                        int(config.num_classes),
                        int(config.background),
                        config.class_names,
                        non_blocking=bool(args.pin_memory),
                    )
                else:
                    result = _run_optimized_sample(
                        model,
                        sample,
                        device,
                        int(config.num_classes),
                        int(config.background),
                        config.class_names,
                        view_batching=int(args.view_batching),
                        non_blocking=bool(args.pin_memory),
                        batching_strategy=getattr(args, "_effective_strategy", None),
                    )
                peak = _peak_memory(device)
                if args.mode == "reference":
                    reference_hist += result["hist"]
                    reference_peak = peak if reference_peak is None else _max_peak(reference_peak, peak)
                else:
                    optimized_hist += result["hist"]
                    optimized_peak = peak if optimized_peak is None else _max_peak(optimized_peak, peak)
                selected_results.append(
                    {
                        "sample_id": sample_id,
                        "original_size_hw": list(_sample_size(sample)),
                        "dataset_seconds": round(dataset_seconds, 6),
                        "timings": result["timings"],
                        "view_records": result["view_records"],
                        "metrics_percent": result["metrics"],
                        "peak_memory": peak,
                        "total_seconds": round(time.perf_counter() - sample_started, 6),
                    }
                )
                del result
    finally:
        del iterator, loader
        gc.collect()
    if args.compare_reference:
        reference_metrics = _metrics_with_subset_label(reference_hist, config.class_names)
        optimized_metrics = _metrics_with_subset_label(optimized_hist, config.class_names)
        overall_status = "PASS" if all(item["status"] == "PASS" for item in comparison_results) and np.array_equal(reference_hist, optimized_hist) else "FAIL"
        return {
            "mode": "compare-reference-and-optimized",
            "effective_view_batching": int(args.view_batching),
            "sample_count": len(selected_results),
            "sample_ids": [item["sample_id"] for item in selected_results],
            "warmup_samples": warmup_ids,
            "loader": _loader_report(options, loader_init_seconds),
            "sample_results": selected_results,
            "equivalence": {
                "status": overall_status,
                "per_sample": comparison_results,
                "aggregate_confusion_matrix_equal": bool(np.array_equal(reference_hist, optimized_hist)),
                "aggregate_metrics_percent": {
                    "reference": reference_metrics,
                    "optimized": optimized_metrics,
                    "optimized_minus_reference": {
                        name: float(optimized_metrics[name]) - float(reference_metrics[name])
                        for name in ("miou", "macc", "mf1")
                    },
                },
                "pass_criteria": {
                    "all_argmax_predictions_equal": all(
                        item["mismatched_prediction_pixels"] == 0 for item in comparison_results
                    ),
                    "all_confusion_matrices_equal": all(
                        item["confusion_matrix_equal"] for item in comparison_results
                    ),
                    "all_metric_differences_within_noise": all(
                        all(abs(value) <= METRIC_NOISE_TOLERANCE for value in item["metric_differences_optimized_minus_reference"].values())
                        for item in comparison_results
                    ),
                },
            },
        }
    hist = reference_hist if args.mode == "reference" else optimized_hist
    metrics = _metrics_with_subset_label(hist, config.class_names)
    aggregate_timings = _sum_timings(selected_results)
    peak = reference_peak if args.mode == "reference" else optimized_peak
    return {
        "mode": args.mode,
        "effective_view_batching": 1 if args.mode == "reference" else int(args.view_batching),
        "sample_count": len(selected_results),
        "sample_ids": [item["sample_id"] for item in selected_results],
        "warmup_samples": warmup_ids,
        "loader": _loader_report(options, loader_init_seconds),
        "sample_results": selected_results,
        "aggregate": {
            "dataset_seconds": round(aggregate_timings["dataset_seconds"], 6),
            "h2d_seconds": round(aggregate_timings["h2d_seconds"], 6),
            "ten_views_seconds": round(aggregate_timings["ten_views_seconds"], 6),
            "forward_seconds": round(aggregate_timings["forward_seconds"], 6),
            "postprocess_seconds": round(aggregate_timings["postprocess_seconds"], 6),
            "metric_seconds": round(aggregate_timings["metric_seconds"], 6),
            "total_seconds": round(aggregate_timings["total_seconds"], 6),
            "seconds_per_sample": round(aggregate_timings["total_seconds"] / len(selected_results), 6),
            "ms_per_view": round(aggregate_timings["ten_views_seconds"] * 1000.0 / (10.0 * len(selected_results)), 6),
            "dataset_preprocess_percent": _percent(aggregate_timings["dataset_seconds"], aggregate_timings["total_seconds"]),
            "h2d_percent": _percent(aggregate_timings["h2d_seconds"], aggregate_timings["total_seconds"]),
            "forward_percent": _percent(aggregate_timings["forward_seconds"], aggregate_timings["total_seconds"]),
            "postprocess_percent": _percent(aggregate_timings["postprocess_seconds"], aggregate_timings["total_seconds"]),
            "metric_percent": _percent(aggregate_timings["metric_seconds"], aggregate_timings["total_seconds"]),
            "metrics_percent": metrics,
            "peak_memory": peak,
        },
    }


def _max_peak(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key in ("allocated_bytes", "reserved_bytes", "allocated_mib", "reserved_mib"):
        if right.get(key) is not None:
            result[key] = max(float(result.get(key, 0.0) or 0.0), float(right[key]))
            if key.endswith("bytes"):
                result[key] = int(result[key])
            else:
                result[key] = round(float(result[key]), 3)
    return result


def _sum_timings(selected_results: Sequence[dict[str, Any]]) -> dict[str, float]:
    total = {
        "dataset_seconds": 0.0,
        "h2d_seconds": 0.0,
        "ten_views_seconds": 0.0,
        "forward_seconds": 0.0,
        "postprocess_seconds": 0.0,
        "metric_seconds": 0.0,
        "total_seconds": 0.0,
    }
    for item in selected_results:
        total["dataset_seconds"] += float(item["dataset_seconds"])
        if "timings" in item:
            timings = item["timings"]
        else:
            timings = item["optimized"]["timings"]
        for key in ("h2d_seconds", "ten_views_seconds", "forward_seconds", "postprocess_seconds", "metric_seconds"):
            total[key] += float(timings[key])
        total["total_seconds"] += float(item["total_seconds"])
    return total


def _percent(value: float, total: float) -> float:
    return round(100.0 * value / total, 3) if total > 0 else 0.0


def _loader_report(options: dict[str, Any], loader_init_seconds: float) -> dict[str, Any]:
    return {
        "num_workers": int(options["num_workers"]),
        "pin_memory": bool(options["pin_memory"]),
        "persistent_workers": bool(options.get("persistent_workers", False)),
        "prefetch_factor": options.get("prefetch_factor"),
        "batch_size": int(options["batch_size"]),
        "loader_init_seconds": round(loader_init_seconds, 6),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--view-batching", choices=("1", "2"), default="1")
    parser.add_argument("--mode", choices=("reference", "optimized"), default="optimized")
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--measure-corruption", action="store_true")
    parser.add_argument("--sample-selection", choices=("first", "stratified", "largest"), default="first")
    parser.add_argument("--microbatch-probe", action="store_true", help="measure batch1/batch2 for all five scales")
    parser.add_argument("--microbatch-repeats", type=int, default=3)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-split-sha256")
    return parser.parse_args(argv)


def _build_report_context(args: argparse.Namespace) -> tuple[dict[str, Any], Any, Any, list[str], torch.device, ForwardTimer]:
    started = dt.datetime.now(dt.timezone.utc)
    report: dict[str, Any] = {
        "schema_version": "museg-checkpoint-fast-evidence-v1",
        "generated_at_utc": started.isoformat(),
        "status": "failed",
        "official_test_included": False,
        "command_line": _command_line(None),
        "mode": args.mode,
        "requested_view_batching": int(args.view_batching),
        "sample_selection": args.sample_selection,
        "max_samples": int(args.max_samples),
        "forward_precision": "fp32",
        "logits_fusion_precision": "fp32",
        "tf32_enabled": False,
        "frozen_evaluator_module": "tools.evaluate_museg_checkpoint as EV",
        "reused_frozen_functions": [
            "EV.configure_fp32_forward",
            "EV.load_model",
            "EV.MUSegPostEvalDataset",
            "EV.MSFLIP_GEOMETRY",
            "EV.build_msflip_views",
            "EV.msflip_whole_logits",
            "EV.restore_logits_to_metric_grid",
            "EV.update_confusion",
            "EV.metrics_from_confusion",
            "EV.apply_channel_order",
        ],
        "optimized_postprocess_contract": [
            "F.interpolate to padded_size_hw",
            "crop to scaled_size_hw",
            "torch.flip(dims=(-1,)) when flipped",
            "F.interpolate to original label grid",
            "FP32 sum then arithmetic mean over ten views",
        ],
    }
    if args.mode == "reference" and args.view_batching != "1":
        raise ValueError("reference mode is fixed to one forward per view; use --view-batching 1")
    if args.warmup_samples < 0:
        raise ValueError("--warmup-samples cannot be negative")
    if args.microbatch_repeats != 3:
        raise ValueError("--microbatch-repeats is fixed to 3")
    dataset_root = args.dataset_root.resolve()
    split = args.split.resolve()
    checkpoint = args.checkpoint.resolve()
    split_sha = EV.file_sha256(split)
    checkpoint_sha = EV.file_sha256(checkpoint)
    if args.expected_split_sha256 and split_sha.lower() != args.expected_split_sha256.lower():
        raise ValueError("split SHA-256 mismatch")
    if args.expected_checkpoint_sha256 and checkpoint_sha.lower() != args.expected_checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA-256 mismatch")
    entries = EV.split_entries(split)
    selected_entries = _select_entries(dataset_root, entries, args.sample_selection, args.max_samples)
    module = importlib.import_module(args.config)
    config = copy.copy(module.C)
    channel_order = getattr(config, "channel_order", None)
    if channel_order not in EV.CHANNEL_ORDERS:
        raise ValueError("config must provide channel_order=BGR or RGB")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    EV.configure_fp32_forward(device)
    model_load_started = time.perf_counter()
    model = ForwardTimer(EV.load_model(config, checkpoint, device))
    model.eval()
    model_load_seconds = time.perf_counter() - model_load_started
    dataset = EV.MUSegPostEvalDataset(
        dataset_root,
        selected_entries,
        EV.MSFLIP_GEOMETRY,
        config,
        channel_order,
    )
    report.update(
        {
            "dataset_root": str(dataset_root),
            "split": {"path": str(split), "sha256": split_sha, "sample_count": len(entries)},
            "selected_entries": list(selected_entries),
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
            "config_module": args.config,
            "model": str(config.backbone),
            "input_contract": {
                "channel_order": channel_order,
                "normalization": {
                    "identity": getattr(config, "normalization_identity", None),
                    "mean": [float(value) for value in config.norm_mean],
                    "std": [float(value) for value in config.norm_std],
                },
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "tf32_enabled": False if device.type == "cuda" else None,
                "os": platform.platform(),
                "pid": os.getpid(),
            },
            "model_load_seconds": round(model_load_seconds, 6),
            "data_loader_requested": {
                "num_workers": args.num_workers,
                "pin_memory": args.pin_memory,
                "persistent_workers": args.persistent_workers,
                "prefetch_factor": args.prefetch_factor,
            },
        }
    )
    return report, config, dataset, entries, device, model


def main(argv: Sequence[str] | None = None) -> int:
    started = time.monotonic()
    args = _parse_args(argv)
    report: dict[str, Any] = {}
    status_code = 2
    try:
        report, config, dataset, all_entries, device, model = _build_report_context(args)
        selected_entries = list(report["selected_entries"])
        if args.microbatch_probe:
            anchor_entry, _anchor_size = EV.select_largest_val_sample(
                report["dataset_root"] and Path(report["dataset_root"]), all_entries
            )
            args._num_classes = int(config.num_classes)
            args._class_names = list(config.class_names)
            args._ignore_label = int(config.background)
            report["microbatch_probe"] = _microbatch_probe(
                model,
                dataset,
                anchor_entry,
                device,
                config,
                args,
            )
        else:
            if args.view_batching == "2" and (args.mode == "optimized" or args.compare_reference):
                anchor_entry, _anchor_size = EV.select_largest_val_sample(
                    Path(report["dataset_root"]), all_entries
                )
                args._num_classes = int(config.num_classes)
                args._class_names = list(config.class_names)
                args._ignore_label = int(config.background)
                report["microbatch_probe"] = _microbatch_probe(
                    model,
                    dataset,
                    anchor_entry,
                    device,
                    config,
                    args,
                )
                args._effective_strategy = {
                    float(scale): int(values["selected_view_batching"])
                    for scale, values in report["microbatch_probe"]["fixed_strategy"].items()
                }
            report["evaluation"] = _run_evaluation(
                args,
                model,
                dataset,
                selected_entries,
                config,
                device,
            )
            if args.mode == "reference" or args.compare_reference:
                report["technical_anchor"] = _geometry_anchor(
                    model,
                    Path(report["dataset_root"]),
                    all_entries,
                    config,
                    report["input_contract"]["channel_order"],
                    device,
                    int(config.num_classes),
                    int(config.background),
                    config.class_names,
                    non_blocking=bool(args.pin_memory),
                )
            if args.measure_corruption:
                report["corruption_measurement"] = _measure_corruption(
                    Path(report["dataset_root"]),
                    selected_entries,
                    evaluation_seed=EVALUATION_SEED,
                    repeats=3,
                )
        report["status"] = "completed"
        status_code = 0
    except torch.OutOfMemoryError as exc:
        report.update(
            {
                "status": "environment_limit",
                "error_type": "cuda_out_of_memory",
                "error": str(exc),
            }
        )
        status_code = 3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (OSError, RuntimeError, ValueError, ModuleNotFoundError, KeyError, TypeError) as exc:
        report.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        status_code = 2
    finally:
        report["duration_seconds"] = round(time.monotonic() - started, 3)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": report.get("status", "failed"),
                    "output": str(output),
                    "duration_seconds": report["duration_seconds"],
                },
                ensure_ascii=False,
            )
        )
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
