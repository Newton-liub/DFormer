"""Recompute A2 per-image and aggregate metrics from saved predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


NUM_CLASSES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def resolve_mask_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [path, manifest_path.parent / path, manifest_path.parent.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"mask not found: {raw_path}")


def confusion(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    keep = valid & (target >= 0) & (target < NUM_CLASSES)
    values = NUM_CLASSES * target[keep].astype(np.int64) + pred[keep].astype(np.int64)
    return np.bincount(values, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def mean_iou(hist: np.ndarray) -> float:
    union = hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist)
    valid = union > 0
    return float(np.mean(np.divide(np.diag(hist)[valid], union[valid]))) if valid.any() else 0.0


def boundary_band_iou(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    kernel = np.ones((11, 11), dtype=np.uint8)
    band = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
    return mean_iou(confusion(pred, target, band))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "q", "sample_id", "foreground_miou", "boundary_band_miou", "prediction_path"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    aggregate_hists: dict[str, np.ndarray] = {}
    for sample_entry in manifest["samples"]:
        sample = sample_entry["stats"]
        sample_id = sample["sample_id"]
        label_path = args.dataset_root / "Label" / f"{sample_id}.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(label_path)
        for condition in sample_entry["conditions"]:
            q = float(condition["q"])
            condition_name = f"q_{q:g}_{condition['mask_type']}"
            prediction_path = args.prediction_root / condition_name / f"{sample_id}.png"
            prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(resolve_mask_path(args.manifest, condition["mask_path"])), cv2.IMREAD_GRAYSCALE)
            if prediction is None or mask is None:
                raise FileNotFoundError(f"missing prediction or mask for {condition_name}/{sample_id}")
            target = cv2.resize(label, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64) - 1
            target[target < 0] = 255
            resized_mask = cv2.resize(mask, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            image_hist = confusion(prediction, target, target != 255)
            aggregate_hists.setdefault(condition_name, np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
            aggregate_hists[condition_name] += image_hist
            rows.append(
                {
                    "condition": condition_name,
                    "q": q,
                    "sample_id": sample_id,
                    "foreground_miou": mean_iou(image_hist),
                    "boundary_band_miou": boundary_band_iou(prediction, target, resized_mask),
                    "prediction_path": str(prediction_path),
                }
            )

    for condition_name, hist in aggregate_hists.items():
        condition_rows = [row for row in rows if row["condition"] == condition_name]
        summaries[condition_name] = {
            "q": condition_rows[0]["q"],
            "foreground_miou": mean_iou(hist),
            "image_count": len(condition_rows),
            "prediction_root": str(args.prediction_root / condition_name),
        }
    if not rows:
        raise ValueError("manifest did not produce any metric rows")
    write_csv(args.output_root / "per_image_metrics.csv", rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()