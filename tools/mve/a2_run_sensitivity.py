"""Run the A2 corruption sweep on a CUDA-capable machine."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.builder import EncoderDecoder


class CorruptionDataset(Dataset):
    def __init__(self, dataset_root: Path, entries: list[tuple[dict, dict]], height: int, width: int, rgb_order: str):
        self.dataset_root = dataset_root
        self.entries = entries
        self.height = height
        self.width = width
        self.rgb_order = rgb_order

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample_entry, condition = self.entries[index]
        sample = sample_entry["stats"]
        sample_id = sample["sample_id"]
        rgb = cv2.imread(str(self.dataset_root / "RGB" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(self.dataset_root / "Depth" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(self.dataset_root / "Label" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(condition["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if any(image is None for image in (rgb, depth, label, mask)):
            raise FileNotFoundError(f"failed to read A2 inputs for {sample_id}")
        if rgb.shape[:2] != depth.shape or depth.shape != label.shape or depth.shape != mask.shape:
            raise ValueError(f"A2 input dimensions differ for {sample_id}")

        if self.rgb_order == "RGB":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = depth.copy()
        depth[mask > 0] = 0
        depth = cv2.merge([depth, depth, depth])

        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int64) - 1
        label[label < 0] = 255

        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        depth = depth.astype(np.float32) / 255.0
        depth = (depth - 0.48) / 0.28
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
            "depth": torch.from_numpy(np.ascontiguousarray(depth.transpose(2, 0, 1))),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask > 0)),
            "sample_id": sample_id,
            "injected_zero_pixels": int(condition["injected_zero_pixels"]),
            "q": float(condition["q"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="local_configs.MUSeg.DFormerv2_S_MVE")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--q", type=float, nargs="+", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rgb-order", choices=("BGR", "RGB"), default="BGR")
    return parser.parse_args()


def resolve_mask_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.is_file():
        return path
    candidates = [path, manifest_path.parent / path, manifest_path.parent.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"mask not found: {raw_path}")


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("samples"):
        raise ValueError(f"manifest has no samples: {path}")
    for sample in manifest["samples"]:
        for condition in sample["conditions"]:
            condition["mask_path"] = str(resolve_mask_path(path, condition["mask_path"]))
    return manifest


def load_config(config_name: str):
    return importlib.import_module(config_name).C


def load_model(config, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = EncoderDecoder(cfg=config, norm_layer=nn.BatchNorm2d)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"warning: unexpected checkpoint keys: {len(unexpected)}")
    return model.to(device).eval()


def confusion(pred: np.ndarray, target: np.ndarray, num_classes: int, valid: np.ndarray) -> np.ndarray:
    keep = valid & (target >= 0) & (target < num_classes)
    values = num_classes * target[keep].astype(np.int64) + pred[keep].astype(np.int64)
    return np.bincount(values, minlength=num_classes**2).reshape(num_classes, num_classes)


def mean_iou(hist: np.ndarray) -> float:
    union = hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist)
    valid = union > 0
    return float(np.mean(np.divide(np.diag(hist)[valid], union[valid]))) if valid.any() else 0.0


def boundary_iou(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, num_classes: int) -> float:
    if not mask.any():
        return float("nan")
    kernel = np.ones((11, 11), dtype=np.uint8)
    band = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
    return mean_iou(confusion(pred, target, num_classes, band))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition",
        "q",
        "sample_id",
        "injected_zero_pixels",
        "foreground_miou",
        "boundary_band_miou",
        "prediction_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_condition(model: nn.Module, loader: DataLoader, condition_name: str, output_root: Path, device: torch.device) -> tuple[list[dict], np.ndarray]:
    rows = []
    aggregate = np.zeros((15, 15), dtype=np.int64)
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        with torch.inference_mode():
            logits = model(rgb, depth)
        predictions = logits.argmax(dim=1).cpu().numpy()
        labels = batch["label"].numpy()
        masks = batch["mask"].numpy().astype(bool)
        for index, sample_id in enumerate(batch["sample_id"]):
            prediction = predictions[index]
            label = labels[index]
            mask = masks[index]
            image_hist = confusion(prediction, label, 15, label != 255)
            aggregate += image_hist
            prediction_path = output_root / condition_name / f"{sample_id}.png"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(prediction_path), prediction.astype(np.uint8))
            rows.append(
                {
                    "condition": condition_name,
                    "q": float(batch["q"][index]),
                    "sample_id": sample_id,
                    "injected_zero_pixels": int(batch["injected_zero_pixels"][index]),
                    "foreground_miou": mean_iou(image_hist),
                    "boundary_band_miou": boundary_iou(prediction, label, mask, 15),
                    "prediction_path": str(prediction_path),
                }
            )
    return rows, aggregate


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    manifest = load_manifest(args.manifest)
    config = load_config(args.config)
    model = load_model(config, args.checkpoint, device)
    wanted_q = None if args.q is None else {round(value, 6) for value in args.q}
    all_rows = []
    summaries = {}
    grouped: dict[str, list[tuple[dict, dict]]] = {}
    for sample in manifest["samples"]:
        for condition in sample["conditions"]:
            q = round(float(condition["q"]), 6)
            if wanted_q is not None and q not in wanted_q:
                continue
            condition_name = f"q_{q:g}_{condition['mask_type']}"
            grouped.setdefault(condition_name, []).append((sample, condition))

    for condition_name, entries in grouped.items():
        dataset = CorruptionDataset(args.dataset_root, entries, args.height, args.width, args.rgb_order)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        rows, aggregate = run_condition(model, loader, condition_name, args.output_root, device)
        all_rows.extend(rows)
        summaries[condition_name] = {
            "q": float(entries[0][1]["q"]),
            "foreground_miou": mean_iou(aggregate),
            "image_count": len(rows),
            "prediction_root": str(args.output_root / condition_name),
        }
    if not all_rows:
        raise ValueError("no conditions selected; check --q and manifest")
    write_csv(args.output_root / "per_image_metrics.csv", all_rows)
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()