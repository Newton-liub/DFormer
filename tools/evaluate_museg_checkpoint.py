#!/usr/bin/env python3
"""Evaluate one preserved MUSeg checkpoint on frozen val-dev with structured evidence."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.builder import EncoderDecoder

GEOMETRIES = ("original-full", "resize-480x640", "sliding-480x640")
CHANNEL_ORDERS = ("BGR", "RGB")
RESIZE_HEIGHT = 480
RESIZE_WIDTH = 640
SLIDING_STRIDE_RATE = 2 / 3


def apply_channel_order(image_bgr: np.ndarray, channel_order: str) -> np.ndarray:
    if channel_order == "BGR":
        return image_bgr
    if channel_order == "RGB":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"unsupported channel order: {channel_order!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_entries(path: Path) -> list[str]:
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("validation split must be non-empty and contain unique entries")
    if any(not entry.startswith("RGB/") for entry in entries):
        raise ValueError("validation split contains a non-RGB identity")
    if any("test" in entry.lower() or "official" in entry.lower() for entry in entries):
        raise ValueError("refusing a split whose identity suggests official test data")
    return entries


class MUSegPostEvalDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        entries: list[str],
        geometry: str,
        config: Any,
        channel_order: str,
    ):
        self.dataset_root = dataset_root
        self.entries = entries
        self.geometry = geometry
        self.config = config
        self.channel_order = channel_order

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        sample_id = Path(entry).stem
        rgb = cv2.imread(str(self.dataset_root / "RGB" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(self.dataset_root / "Depth" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(self.dataset_root / "Label" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or label is None:
            raise FileNotFoundError(f"missing RGB/Depth/Label input for val-dev sample {sample_id}")
        if rgb.shape[:2] != depth.shape or depth.shape != label.shape:
            raise ValueError(f"modality geometry mismatch for val-dev sample {sample_id}")
        original_height, original_width = label.shape
        rgb = apply_channel_order(rgb, self.channel_order)
        if self.geometry == "resize-480x640":
            rgb = cv2.resize(rgb, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        input_height, input_width = rgb.shape[:2]
        depth = cv2.merge([depth, depth, depth])
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.asarray(self.config.norm_mean, dtype=np.float32)) / np.asarray(
            self.config.norm_std, dtype=np.float32
        )
        depth = depth.astype(np.float32) / 255.0
        depth = (depth - 0.48) / 0.28
        label = label.astype(np.int64) - 1
        label[label < 0] = 255
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
            "depth": torch.from_numpy(np.ascontiguousarray(depth.transpose(2, 0, 1))),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "sample_id": sample_id,
            "original_height": original_height,
            "original_width": original_width,
            "input_height": input_height,
            "input_width": input_width,
        }


def load_model(config: Any, checkpoint_path: Path, device: torch.device) -> nn.Module:
    # A complete training checkpoint provides every model weight. Passing no
    # criterion prevents EncoderDecoder from trying to load a separate
    # pretrained backbone before the strict checkpoint restore below.
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must decode to a mapping")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model state mapping")
    normalized = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(normalized, strict=True)
    return model.to(device).eval()


def sliding_logits(model: nn.Module, rgb: torch.Tensor, depth: torch.Tensor, *, height: int = 480, width: int = 640, stride_rate: float = 2 / 3) -> torch.Tensor:
    batch, _, image_height, image_width = rgb.shape
    if batch != 1:
        raise ValueError("sliding-window post-evaluation requires batch size 1")
    stride_height = max(1, int(height * stride_rate))
    stride_width = max(1, int(width * stride_rate))
    rows = max(math.ceil(max(image_height - height, 0) / stride_height) + 1, 1)
    columns = max(math.ceil(max(image_width - width, 0) / stride_width) + 1, 1)
    logits_sum: torch.Tensor | None = None
    counts = rgb.new_zeros((1, 1, image_height, image_width))
    for row in range(rows):
        for column in range(columns):
            y2 = min(row * stride_height + height, image_height)
            x2 = min(column * stride_width + width, image_width)
            y1 = max(y2 - height, 0)
            x1 = max(x2 - width, 0)
            rgb_crop = rgb[:, :, y1:y2, x1:x2]
            depth_crop = depth[:, :, y1:y2, x1:x2]
            crop_height, crop_width = rgb_crop.shape[-2:]
            pad = (0, width - crop_width, 0, height - crop_height)
            if pad[1] or pad[3]:
                rgb_crop = F.pad(rgb_crop, pad)
                depth_crop = F.pad(depth_crop, pad)
            crop_logits = model(rgb_crop, depth_crop)[:, :, :crop_height, :crop_width]
            if logits_sum is None:
                logits_sum = crop_logits.new_zeros((1, crop_logits.shape[1], image_height, image_width))
            assert logits_sum is not None
            logits_sum[:, :, y1:y2, x1:x2] += crop_logits
            counts[:, :, y1:y2, x1:x2] += 1
    if logits_sum is None or bool((counts == 0).any().item()):
        raise RuntimeError("sliding-window coverage is incomplete")
    return logits_sum / counts


def restore_logits_to_metric_grid(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Restore model logits to the untouched original label grid."""
    target_size = tuple(int(value) for value in labels.shape[-2:])
    if tuple(logits.shape[-2:]) == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def geometry_contract(geometry: str, output_sizes_hw: Sequence[tuple[int, int]]) -> dict[str, Any]:
    input_contract: dict[str, Any] = {
        "name": geometry,
        "color_interpolation": "opencv-linear" if geometry == "resize-480x640" else "none",
        "depth_interpolation": "opencv-linear" if geometry == "resize-480x640" else "none",
        "resize_size_hw": [RESIZE_HEIGHT, RESIZE_WIDTH] if geometry == "resize-480x640" else None,
    }
    if geometry == "sliding-480x640":
        input_contract["sliding"] = {
            "crop_size_hw": [RESIZE_HEIGHT, RESIZE_WIDTH],
            "stride_rate": SLIDING_STRIDE_RATE,
            "stride_hw": [int(RESIZE_HEIGHT * SLIDING_STRIDE_RATE), int(RESIZE_WIDTH * SLIDING_STRIDE_RATE)],
            "padding": "right-and-bottom zero padding only for inputs smaller than crop",
            "overlap_reduction": "mean logits over full-coverage count map",
        }
    return {
        "input_geometry": input_contract,
        "metric_geometry": {
            "name": "original-label-grid",
            "label_resize": "none",
            "logits_restore_interpolation": "pytorch-bilinear-align_corners-false",
            "output_sizes_hw": [list(size) for size in sorted(set(output_sizes_hw))],
        },
    }


def update_confusion(hist: np.ndarray, logits: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_label: int = 255) -> None:
    predictions = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    targets = labels.detach().cpu().numpy().astype(np.int64)
    keep = (targets != ignore_label) & (targets >= 0) & (targets < num_classes)
    values = num_classes * targets[keep] + predictions[keep]
    hist += np.bincount(values, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(hist: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    true_positive = np.diag(hist).astype(np.float64)
    prediction_count = hist.sum(axis=0).astype(np.float64)
    target_count = hist.sum(axis=1).astype(np.float64)
    union = prediction_count + target_count - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0) * 100
    accuracy = np.divide(true_positive, target_count, out=np.zeros_like(true_positive), where=target_count > 0) * 100
    f1_denominator = prediction_count + target_count
    f1 = np.divide(2 * true_positive, f1_denominator, out=np.zeros_like(true_positive), where=f1_denominator > 0) * 100
    return {
        "miou": round(float(iou.mean()), 2),
        "macc": round(float(accuracy.mean()), 2),
        "mf1": round(float(f1.mean()), 2),
        "per_class": [
            {
                "index": index,
                "name": str(name),
                "iou": round(float(iou[index]), 2),
                "accuracy": round(float(accuracy[index]), 2),
                "f1": round(float(f1[index]), 2),
                "target_pixels": int(target_count[index]),
            }
            for index, name in enumerate(class_names)
        ],
        "confusion_matrix": hist.tolist(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="local_configs.MUSeg.DFormerv2_S_4090")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channel-order", "--rgb-order", dest="channel_order", choices=CHANNEL_ORDERS)
    parser.add_argument("--normalization-identity")
    parser.add_argument("--normalization-mean", nargs=3, type=float)
    parser.add_argument("--normalization-std", nargs=3, type=float)
    return parser.parse_args(argv)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc)
    start_clock = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": "museg-checkpoint-post-evaluation-v2",
        "generated_at_utc": started.isoformat(),
        "status": "failed",
        "official_test_included": False,
        "split_role": "val_dev",
        "batch_size": args.batch_size,
        "geometry": args.geometry,
        "validation_amp": args.amp,
    }
    try:
        if args.batch_size != 1:
            raise ValueError("post-evaluation is frozen to batch size 1")
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        checkpoint = args.checkpoint.resolve()
        split = args.split.resolve()
        dataset_root = args.dataset_root.resolve()
        checkpoint_sha = file_sha256(checkpoint)
        split_sha = file_sha256(split)
        if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256.lower():
            raise ValueError("checkpoint SHA-256 mismatch")
        if args.expected_split_sha256 and split_sha != args.expected_split_sha256.lower():
            raise ValueError("split SHA-256 mismatch")
        entries = split_entries(split)
        config = copy.copy(importlib.import_module(args.config).C)
        channel_order = args.channel_order or getattr(config, "channel_order", None)
        normalization_identity = args.normalization_identity or getattr(config, "normalization_identity", None)
        if channel_order not in CHANNEL_ORDERS:
            raise ValueError("config or CLI must explicitly provide channel_order=BGR or RGB")
        if not isinstance(normalization_identity, str) or not normalization_identity.strip():
            raise ValueError("config or CLI must explicitly provide a normalization identity")
        if (args.normalization_mean is None) != (args.normalization_std is None):
            raise ValueError("normalization mean and std overrides must be supplied together")
        if args.normalization_mean is not None and args.normalization_identity is None:
            raise ValueError("normalization overrides require an explicit normalization identity")
        if args.normalization_mean is not None:
            if any(value <= 0 for value in args.normalization_std):
                raise ValueError("normalization std values must be positive")
            config.norm_mean = np.asarray(args.normalization_mean, dtype=np.float32)
            config.norm_std = np.asarray(args.normalization_std, dtype=np.float32)
        report["input_contract"] = {
            "channel_order": channel_order,
            "normalization": {
                "identity": normalization_identity,
                "mean": [float(value) for value in config.norm_mean],
                "std": [float(value) for value in config.norm_std],
            },
        }
        device = torch.device(args.device)
        model = load_model(config, checkpoint, device)
        dataset = MUSegPostEvalDataset(dataset_root, entries, args.geometry, config, channel_order)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        hist = np.zeros((int(config.num_classes), int(config.num_classes)), dtype=np.int64)
        output_sizes_hw: list[tuple[int, int]] = []
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                if args.geometry == "sliding-480x640":
                    logits = sliding_logits(
                        model,
                        rgb,
                        depth,
                        height=RESIZE_HEIGHT,
                        width=RESIZE_WIDTH,
                        stride_rate=SLIDING_STRIDE_RATE,
                    )
                else:
                    logits = model(rgb, depth)
                logits = restore_logits_to_metric_grid(logits, labels)
            output_sizes_hw.append(tuple(int(value) for value in labels.shape[-2:]))
            update_confusion(hist, logits, labels, int(config.num_classes), int(config.background))
        report.update(
            {
                "status": "completed",
                "dataset_root": str(dataset_root),
                "sample_count": len(entries),
                "split": {"path": str(split), "sha256": split_sha},
                "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
                "config_module": args.config,
                "model": str(config.backbone),
                **geometry_contract(args.geometry, output_sizes_hw),
                "metrics_percent": metrics_from_confusion(hist, config.class_names),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                },
            }
        )
        status = 0
    except torch.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report.update({"status": "environment_limit", "error_type": "cuda_out_of_memory", "error": str(exc)})
        status = 3
    except (OSError, RuntimeError, ValueError, ModuleNotFoundError) as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        status = 2
    report["duration_seconds"] = round(time.monotonic() - start_clock, 3)
    _write_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
