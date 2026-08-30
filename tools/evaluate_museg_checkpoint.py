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
from tools.museg_protocol import load_protocol

MSFLIP_GEOMETRY = "msflip-whole-original-grid-v1"
MSFLIP_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
PAD_DIVISOR = 32
GEOMETRIES = ("original-full", "resize-480x640", "sliding-480x640", MSFLIP_GEOMETRY)
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


def select_largest_val_sample(
    dataset_root: Path,
    entries: Sequence[str],
) -> tuple[str, tuple[int, int]]:
    """Select the largest val-dev sample by label pixels without reading RGB/depth."""
    ranked: list[tuple[int, int, int, str]] = []
    for entry in entries:
        sample_id = Path(entry).stem
        label_path = dataset_root / "Label" / f"{sample_id}.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"missing val-dev label for sample {sample_id}")
        height, width = label.shape
        ranked.append((height * width, height, width, entry))
    if not ranked:
        raise ValueError("cannot select a largest sample from an empty split")
    _, height, width, entry = max(ranked)
    return entry, (height, width)


def configure_fp32_forward(device: torch.device) -> None:
    """Disable reduced-precision CUDA paths for the frozen FP32 evaluator."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False



def build_msflip_views(
    rgb: np.ndarray,
    depth: np.ndarray,
    config: Any,
    *,
    scales: Sequence[float] = MSFLIP_SCALES,
    pad_divisor: int = PAD_DIVISOR,
) -> list[dict[str, Any]]:
    """Build OpenCV-resized, normalized, right/bottom-padded multi-scale flip views."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError("msflip RGB/depth inputs must share one HxW geometry")
    if pad_divisor <= 0:
        raise ValueError("pad_divisor must be positive")
    original_height, original_width = depth.shape
    mean = np.asarray(config.norm_mean, dtype=np.float32)
    std = np.asarray(config.norm_std, dtype=np.float32)
    views: list[dict[str, Any]] = []
    for scale in scales:
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("msflip scales must be finite and positive")
        scaled_height = max(1, int(round(original_height * float(scale))))
        scaled_width = max(1, int(round(original_width * float(scale))))
        scaled_rgb = cv2.resize(
            rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
        )
        scaled_depth = cv2.resize(
            depth, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
        )
        rgb_float = scaled_rgb.astype(np.float32) / 255.0
        rgb_float = (rgb_float - mean) / std
        depth_float = scaled_depth.astype(np.float32) / 255.0
        depth_float = (depth_float - 0.48) / 0.28
        depth_float = np.repeat(depth_float[:, :, None], 3, axis=2)
        padded_height = math.ceil(scaled_height / pad_divisor) * pad_divisor
        padded_width = math.ceil(scaled_width / pad_divisor) * pad_divisor
        pad_spec = ((0, padded_height - scaled_height), (0, padded_width - scaled_width), (0, 0))
        for flipped in (False, True):
            view_rgb = np.flip(rgb_float, axis=1).copy() if flipped else rgb_float
            view_depth = np.flip(depth_float, axis=1).copy() if flipped else depth_float
            view_rgb = np.pad(view_rgb, pad_spec, mode="constant", constant_values=0)
            view_depth = np.pad(view_depth, pad_spec, mode="constant", constant_values=0)
            views.append(
                {
                    "rgb": torch.from_numpy(np.ascontiguousarray(view_rgb.transpose(2, 0, 1))),
                    "depth": torch.from_numpy(np.ascontiguousarray(view_depth.transpose(2, 0, 1))),
                    "scale": float(scale),
                    "flipped": flipped,
                    "scaled_size_hw": (scaled_height, scaled_width),
                    "padded_size_hw": (padded_height, padded_width),
                }
            )
    return views


def _metadata_scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("view metadata tensor must contain one value")
        return value.item()
    return value


def msflip_whole_logits(
    model: nn.Module,
    views: Sequence[dict[str, Any]],
    *,
    original_size_hw: tuple[int, int],
    amp: bool = False,
    measure_timing: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Average pre-softmax view logits in FP32 on the original label grid."""
    if amp:
        raise ValueError("msflip evaluator is frozen to FP32 forward; AMP is not allowed")
    if not views:
        raise ValueError("msflip inference requires at least one view")
    fused: torch.Tensor | None = None
    records: list[dict[str, Any]] = []
    for view in views:
        rgb = view["rgb"]
        depth = view["depth"]
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
            depth = depth.unsqueeze(0)
        if rgb.ndim != 4 or depth.shape != rgb.shape or rgb.shape[0] != 1:
            raise ValueError("msflip inference is frozen to one aligned RGB/depth sample")
        if rgb.dtype != torch.float32 or depth.dtype != torch.float32:
            raise ValueError("msflip evaluator inputs must be FP32")
        scaled_size = tuple(int(_metadata_scalar(value)) for value in view["scaled_size_hw"])
        padded_size = tuple(int(_metadata_scalar(value)) for value in view["padded_size_hw"])
        flipped = bool(_metadata_scalar(view["flipped"]))
        scale = float(_metadata_scalar(view["scale"]))
        if measure_timing and rgb.device.type == "cuda":
            torch.cuda.synchronize(rgb.device)
        view_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(rgb, depth)
        if logits.dtype != torch.float32:
            raise ValueError("msflip evaluator model output must be FP32")
        if tuple(logits.shape[-2:]) != padded_size:
            logits = F.interpolate(logits, size=padded_size, mode="bilinear", align_corners=False)
        logits = logits[..., : scaled_size[0], : scaled_size[1]]
        if flipped:
            logits = torch.flip(logits, dims=(-1,))
        logits = F.interpolate(
            logits, size=original_size_hw, mode="bilinear", align_corners=False
        )
        if measure_timing and rgb.device.type == "cuda":
            torch.cuda.synchronize(rgb.device)
        elapsed_seconds = time.perf_counter() - view_started
        fused = logits if fused is None else fused + logits
        record = {
            "scale": scale,
            "flipped": flipped,
            "scaled_size_hw": list(scaled_size),
            "padded_size_hw": list(padded_size),
            "restored_size_hw": list(original_size_hw),
        }
        if measure_timing:
            record["elapsed_seconds"] = round(elapsed_seconds, 6)
        records.append(record)
    assert fused is not None
    return fused / float(len(views)), records

class MUSegPostEvalDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        entries: list[str],
        geometry: str,
        config: Any,
        channel_order: str,
        *,
        msflip_scales: Sequence[float] = MSFLIP_SCALES,
    ):
        self.dataset_root = dataset_root
        self.entries = entries
        self.geometry = geometry
        self.config = config
        self.channel_order = channel_order
        self.msflip_scales = tuple(float(scale) for scale in msflip_scales)

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
        label = label.astype(np.int64) - 1
        label[label < 0] = 255
        common: dict[str, object] = {
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "sample_id": sample_id,
            "original_height": original_height,
            "original_width": original_width,
        }
        if self.geometry == MSFLIP_GEOMETRY:
            return {
                **common,
                "views": build_msflip_views(
                    rgb,
                    depth,
                    self.config,
                    scales=self.msflip_scales,
                ),
                "input_height": original_height,
                "input_width": original_width,
            }
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
        return {
            **common,
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
            "depth": torch.from_numpy(np.ascontiguousarray(depth.transpose(2, 0, 1))),
            "input_height": input_height,
            "input_width": input_width,
        }


def load_model(config: Any, checkpoint_path: Path, device: torch.device) -> nn.Module:
    # A complete training checkpoint provides every model weight. Passing no
    # criterion prevents EncoderDecoder from trying to load a separate
    # pretrained backbone before the strict checkpoint restore below.
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    # These trusted legacy checkpoints contain NumPy metadata; PyTorch 2.6+
    # defaults to weights_only=True, which cannot deserialize the full payload.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must decode to a mapping")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model state mapping")
    normalized = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(normalized, strict=True)
    return model.float().to(device).eval()


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


def geometry_contract(
    geometry: str,
    output_sizes_hw: Sequence[tuple[int, int]],
    view_geometry: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    uses_resize = geometry in {"resize-480x640", MSFLIP_GEOMETRY}
    input_contract: dict[str, Any] = {
        "name": geometry,
        "color_interpolation": "opencv-linear" if uses_resize else "none",
        "depth_interpolation": "opencv-linear" if uses_resize else "none",
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
    if geometry == MSFLIP_GEOMETRY:
        input_contract["multi_scale_flip"] = {
            "scales": list(MSFLIP_SCALES),
            "views_per_scale": ["original", "horizontal_flip"],
            "padding": {
                "divisor": PAD_DIVISOR,
                "sides": ["right", "bottom"],
                "value_after_normalization": 0,
            },
            "unflip_before_restore": True,
            "fusion": "fp32 arithmetic mean of pre-softmax logits",
            "view_geometry": list(view_geometry),
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
    parser.add_argument("--protocol-manifest")
    parser.add_argument("--config", default="local_configs.MUSeg.DFormerv2_S_4090")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-role", choices=("val_dev",))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--geometry", choices=GEOMETRIES)
    mode.add_argument(
        "--technical-check",
        action="store_true",
        help="run only the largest val-dev sample at scale 1.5 with original+flip views",
    )
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
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
    geometry = MSFLIP_GEOMETRY if args.technical_check else args.geometry
    report: dict[str, Any] = {
        "schema_version": (
            "museg-evaluator-technical-check-v1"
            if args.technical_check
            else "museg-checkpoint-post-evaluation-v2"
        ),
        "evaluator_identity": geometry,
        "mode": "max-sample-scale-1.5-two-view" if args.technical_check else "evaluation",
        "generated_at_utc": started.isoformat(),
        "status": "failed",
        "official_test_included": False,
        "split_role": args.split_role,
        "batch_size": args.batch_size,
        "geometry": geometry,
        "validation_amp": args.amp,
        "forward_precision": "fp32",
        "logits_fusion_precision": "fp32",
    }
    try:
        if args.batch_size != 1:
            raise ValueError("post-evaluation is frozen to batch size 1")
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative")
        if geometry == MSFLIP_GEOMETRY and args.amp:
            raise ValueError("msflip evaluator is frozen to FP32; --amp is not allowed")
        if geometry == MSFLIP_GEOMETRY and args.split_role != "val_dev":
            raise ValueError("msflip modes require the positive declaration --split-role val_dev")
        if args.technical_check and not args.expected_checkpoint_sha256:
            raise ValueError("technical check requires --expected-checkpoint-sha256")
        if args.technical_check and not args.expected_split_sha256:
            raise ValueError("technical check requires --expected-split-sha256")
        checkpoint = args.checkpoint.resolve()
        split = args.split.resolve()
        dataset_root = args.dataset_root.resolve()
        checkpoint_sha = file_sha256(checkpoint)
        split_sha = file_sha256(split)
        protocol = load_protocol(args.protocol_manifest) if args.protocol_manifest else None
        if geometry == MSFLIP_GEOMETRY and not args.technical_check and protocol is None:
            raise ValueError("msflip main evaluation requires --protocol-manifest")
        if protocol is not None:
            if protocol.phase != "development" or protocol.run_kind != "standard":
                raise ValueError("checkpoint evaluator protocol must be a development standard protocol")
            if protocol.config_module != args.config:
                raise ValueError("evaluator config does not match protocol config_module")
            if protocol.split_path("val_dev") != split:
                raise ValueError("evaluator split does not match protocol val_dev")
            if split_sha != str(protocol.splits["val_dev"]["sha256"]).lower():
                raise ValueError("evaluator split SHA-256 does not match protocol val_dev")
            report["protocol"] = {
                "path": str(protocol.path),
                "sha256": protocol.manifest_sha256,
                "protocol_id": protocol.protocol_id,
                "schedule_version": protocol.schedule_version,
            }
            expected_protocol_contract = protocol.input_contract
        else:
            expected_protocol_contract = None
        if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256.lower():
            raise ValueError("checkpoint SHA-256 mismatch")
        if args.expected_split_sha256 and split_sha != args.expected_split_sha256.lower():
            raise ValueError("split SHA-256 mismatch")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
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
        if expected_protocol_contract is not None:
            protocol_normalization = expected_protocol_contract["normalization"]
            if report["input_contract"] != {
                "channel_order": expected_protocol_contract["channel_order"],
                "normalization": {
                    "identity": protocol_normalization["identity"],
                    "mean": [float(value) for value in protocol_normalization["mean"]],
                    "std": [float(value) for value in protocol_normalization["std"]],
                },
            }:
                raise ValueError("evaluator input contract differs from protocol")
        device = torch.device(args.device)
        configure_fp32_forward(device)
        common_result = {
            "dataset_root": str(dataset_root),
            "split": {"path": str(split), "sha256": split_sha},
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
            "config_module": args.config,
            "model": str(config.backbone),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "tf32_enabled": False if device.type == "cuda" else None,
            },
        }
        report.update(common_result)
        if args.technical_check:
            selected_entry, selected_size = select_largest_val_sample(dataset_root, entries)
            sample_id = Path(selected_entry).stem
            report.update(
                {
                    "sample_count": 1,
                    "split_sample_count": len(entries),
                    "sample_selection": {
                        "criterion": (
                            "maximum original label pixel count; ties use height, width, "
                            "then lexicographically greatest entry"
                        ),
                        "entry": selected_entry,
                        "sample_id": sample_id,
                        "original_size_hw": list(selected_size),
                    },
                    "metrics_computed": False,
                }
            )
            dataset = MUSegPostEvalDataset(
                dataset_root,
                [selected_entry],
                MSFLIP_GEOMETRY,
                config,
                channel_order,
                msflip_scales=(1.5,),
            )
            sample = dataset[0]
            model = load_model(config, checkpoint, device)
            views = [
                {
                    **view,
                    "rgb": view["rgb"].to(device, non_blocking=True),
                    "depth": view["depth"].to(device, non_blocking=True),
                }
                for view in sample["views"]
            ]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            logits, view_geometry = msflip_whole_logits(
                model,
                views,
                original_size_hw=selected_size,
                amp=False,
                measure_timing=True,
            )
            if tuple(logits.shape[-2:]) != selected_size:
                raise RuntimeError("technical-check logits were not restored to the original label grid")
            peak_memory = {
                "allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
                "reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                ),
            }
            report.update(
                {
                    "status": "completed",
                    **common_result,
                    "views": [
                        {"sample_id": sample_id, **record} for record in view_geometry
                    ],
                    **geometry_contract(
                        MSFLIP_GEOMETRY,
                        [selected_size],
                        [{"sample_id": sample_id, **record} for record in view_geometry],
                    ),
                    "peak_device_memory": peak_memory,
                    "metrics_computed": False,
                }
            )
        else:
            model = load_model(config, checkpoint, device)
            dataset = MUSegPostEvalDataset(dataset_root, entries, geometry, config, channel_order)
            loader_options: dict[str, Any] = {
                "batch_size": 1,
                "shuffle": False,
                "num_workers": args.num_workers,
                "pin_memory": device.type == "cuda",
            }
            if args.num_workers > 0:
                loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
            loader = DataLoader(dataset, **loader_options)
            hist = np.zeros((int(config.num_classes), int(config.num_classes)), dtype=np.int64)
            output_sizes_hw: list[tuple[int, int]] = []
            view_geometry: list[dict[str, Any]] = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for batch in loader:
                labels = batch["label"].to(device, non_blocking=True)
                if geometry == MSFLIP_GEOMETRY:
                    views = [
                        {
                            **view,
                            "rgb": view["rgb"].to(device, non_blocking=True),
                            "depth": view["depth"].to(device, non_blocking=True),
                        }
                        for view in batch["views"]
                    ]
                    target_size = tuple(int(value) for value in labels.shape[-2:])
                    logits, sample_view_geometry = msflip_whole_logits(
                        model,
                        views,
                        original_size_hw=target_size,
                        amp=False,
                    )
                    sample_id = str(batch["sample_id"][0])
                    view_geometry.extend(
                        {"sample_id": sample_id, **record} for record in sample_view_geometry
                    )
                else:
                    rgb = batch["rgb"].to(device, non_blocking=True)
                    depth = batch["depth"].to(device, non_blocking=True)
                    with torch.inference_mode(), torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=args.amp and device.type == "cuda",
                    ):
                        if geometry == "sliding-480x640":
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
                update_confusion(
                    hist,
                    logits,
                    labels,
                    int(config.num_classes),
                    int(config.background),
                )
            peak_memory = {
                "allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
                "reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                ),
            }
            report.update(
                {
                    "status": "completed",
                    **common_result,
                    "sample_count": len(entries),
                    **geometry_contract(geometry, output_sizes_hw, view_geometry),
                    "metrics_percent": metrics_from_confusion(hist, config.class_names),
                    "peak_device_memory": peak_memory,
                    "metrics_computed": True,
                }
            )
        status = 0
    except torch.OutOfMemoryError as exc:
        peak_memory = None
        if torch.cuda.is_available():
            try:
                oom_device = torch.device(args.device)
                peak_memory = {
                    "allocated_bytes": torch.cuda.max_memory_allocated(oom_device),
                    "reserved_bytes": torch.cuda.max_memory_reserved(oom_device),
                }
            finally:
                torch.cuda.empty_cache()
        report.update(
            {
                "status": "environment_limit",
                "error_type": "cuda_out_of_memory",
                "error": str(exc),
                "peak_device_memory": peak_memory,
            }
        )
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
