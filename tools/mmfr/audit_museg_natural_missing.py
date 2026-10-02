#!/usr/bin/env python3
"""CPU-only audit of natural zero-depth patterns in frozen MUSeg dev splits.

The tool reads only the train-dev and (unless --skip-val is given) val-dev
allowlists. It never opens labels, official split files, or masks from disk; the
8-connected component mask is transient per image and is not written.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, cast

import cv2
import numpy as np


DEPTH_MAX_RAW = 13932
EXPECTED_IMAGES = {"train-dev": 1277, "val-dev": 318}
EXPECTED_GROUPS = {"train-dev": 762, "val-dev": 196}
EXPECTED_SHAPE_HW = (932, 1082)
EXPECTED_PIXELS = EXPECTED_SHAPE_HW[0] * EXPECTED_SHAPE_HW[1]
PERCENTILES = (10, 25, 50, 75, 90)
INVALID_BINS = (
    (0.0, 0.01, "[0,0.01)"),
    (0.01, 0.05, "[0.01,0.05)"),
    (0.05, 0.10, "[0.05,0.10)"),
    (0.10, 0.25, "[0.10,0.25)"),
    (0.25, 0.50, "[0.25,0.50)"),
    (0.50, 0.75, "[0.50,0.75)"),
    (0.75, 1.0, "[0.75,1)"),
    (1.0, 1.0, "[1,1]"),
)
CSV_FIELDS = (
    "split",
    "filename",
    "mine",
    "group_id",
    "height",
    "width",
    "pixels",
    "rgb_mean_luma_0_1",
    "depth16_zero_pixels",
    "depth16_zero_ratio",
    "depth8_zero_pixels",
    "depth8_invalid_ratio",
    "depth8_valid_pixels",
    "depth8_valid_ratio",
    "depth8_zero_from_positive_depth16_pixels",
    "depth8_zero_from_positive_depth16_ratio_image",
    "depth8_zero_from_positive_depth16_ratio_raw_valid",
    "invalid_cc8_count",
    "largest_invalid_cc_pixels",
    "largest_invalid_cc_image_ratio",
    "largest_invalid_cc_invalid_ratio",
    "cc_area_p50_pixels",
    "cc_area_p90_pixels",
    "invalid_ratio_stratum",
    "large_hole_flag",
    "scattered_flag",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_allowlist(split_root: Path, split_name: str) -> list[tuple[str, str, str]]:
    """Read one explicitly named dev allowlist and parse its sample identity."""
    allowlist = split_root / f"{split_name}.txt"
    text = allowlist.read_text(encoding="utf-8")
    entries: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        item = raw_line.strip()
        if not item:
            continue
        rel = PurePosixPath(item)
        if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != "RGB" or rel.suffix.lower() != ".jpg":
            raise ValueError(f"{allowlist}:{line_number}: not an allowlisted RGB/*.jpg path: {item!r}")
        filename = rel.name
        if filename in seen:
            raise ValueError(f"{allowlist}:{line_number}: duplicate sample {filename}")
        seen.add(filename)
        parts = Path(filename).stem.split("-")
        if len(parts) != 7 or not all(re.fullmatch(r"[0-9]+", value) for value in parts):
            raise ValueError(f"{allowlist}:{line_number}: unexpected MUSeg filename format: {filename}")
        mine = parts[0]
        if mine not in {"01", "02", "03", "04", "05", "06"}:
            raise ValueError(f"{allowlist}:{line_number}: invalid mine prefix in {filename}")
        group_id = "-".join(parts[:4])
        entries.append((filename, mine, group_id))

    if len(entries) != EXPECTED_IMAGES[split_name]:
        raise ValueError(
            f"{allowlist}: expected {EXPECTED_IMAGES[split_name]} entries, got {len(entries)}"
        )
    groups = {entry[2] for entry in entries}
    if len(groups) != EXPECTED_GROUPS[split_name]:
        raise ValueError(
            f"{allowlist}: expected {EXPECTED_GROUPS[split_name]} first-four-segment groups, got {len(groups)}"
        )
    return entries


def _percentiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {f"p{p}": None for p in PERCENTILES}
    qs = np.percentile(array, PERCENTILES, method="linear")
    return {f"p{p}": float(value) for p, value in zip(PERCENTILES, qs)}


def _distribution(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "median": None, **_percentiles(array)}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        **_percentiles(array),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or y.size != x.size or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _audit_image(
    dataset_root: Path,
    split_name: str,
    filename: str,
    mine: str,
    group_id: str,
) -> tuple[dict[str, Any], np.ndarray]:
    stem = Path(filename).stem
    rgb = cv2.imread(str(dataset_root / "RGB" / filename), cv2.IMREAD_COLOR)
    depth16 = cv2.imread(str(dataset_root / "Depth16" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    depth8 = cv2.imread(str(dataset_root / "Depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth16 is None or depth8 is None:
        missing = [
            name
            for name, image in (("RGB", rgb), ("Depth16", depth16), ("Depth", depth8))
            if image is None
        ]
        raise FileNotFoundError(f"{filename}: unreadable allowlisted input(s): {', '.join(missing)}")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"{filename}: expected 3-channel uint8 RGB decode, got {rgb.shape}/{rgb.dtype}")
    if depth16.dtype != np.uint16 or depth16.ndim != 2:
        raise ValueError(f"{filename}: expected 2-D uint16 Depth16, got {depth16.shape}/{depth16.dtype}")
    if depth8.dtype != np.uint8 or depth8.ndim != 2:
        raise ValueError(f"{filename}: expected 2-D uint8 Depth, got {depth8.shape}/{depth8.dtype}")
    if rgb.shape[:2] != depth16.shape or depth16.shape != depth8.shape:
        raise ValueError(
            f"{filename}: RGB/Depth16/Depth dimensions differ: {rgb.shape}/{depth16.shape}/{depth8.shape}"
        )
    if depth8.shape != EXPECTED_SHAPE_HW:
        raise ValueError(f"{filename}: expected fixed-cropped input shape {EXPECTED_SHAPE_HW}, got {depth8.shape}")

    height, width = depth8.shape
    pixels = height * width
    if int(depth16.max()) > DEPTH_MAX_RAW:
        raise ValueError(f"{filename}: raw depth exceeds fixed mapping maximum {DEPTH_MAX_RAW}")
    expected_depth8 = np.rint(depth16.astype(np.float64) * 255.0 / DEPTH_MAX_RAW).astype(np.uint8)
    if not np.array_equal(expected_depth8, depth8):
        raise ValueError(f"{filename}: Depth8 differs from the fixed Depth16 quantization")

    raw_invalid = depth16 == 0
    invalid = depth8 == 0
    quantization_zero = (depth16 > 0) & invalid
    raw_valid_pixels = int(np.count_nonzero(~raw_invalid))
    invalid_pixels = int(np.count_nonzero(invalid))
    quantization_zero_pixels = int(np.count_nonzero(quantization_zero))

    # OpenCV decodes color in BGR order. This is un-gamma-corrected ITU-R BT.601
    # luma from 8-bit channel code values, normalized to [0,1].
    y_num = 299 * rgb[:, :, 2].astype(np.int64) + 587 * rgb[:, :, 1].astype(np.int64) + 114 * rgb[:, :, 0].astype(np.int64)
    mean_luma = float(y_num.sum(dtype=np.int64)) / float(1000 * 255 * pixels)

    # The mask is transient only. Connected components use 8-neighbor adjacency;
    # background label 0 is excluded from the returned component areas.
    cc_count_with_background, _, stats, _ = cv2.connectedComponentsWithStats(
        invalid.astype(np.uint8), connectivity=8
    )
    component_areas = stats[1:cc_count_with_background, cv2.CC_STAT_AREA].astype(np.int64)
    cc_count = int(component_areas.size)
    largest_cc = int(component_areas.max()) if cc_count else 0
    largest_cc_ratio_image = largest_cc / pixels
    largest_cc_ratio_invalid = largest_cc / invalid_pixels if invalid_pixels else 0.0
    area_p50 = float(np.percentile(component_areas, 50, method="linear")) if cc_count else None
    area_p90 = float(np.percentile(component_areas, 90, method="linear")) if cc_count else None
    large_hole = bool(largest_cc_ratio_image >= 0.01 and largest_cc_ratio_invalid >= 0.50)
    scattered = bool(cc_count >= 20 and largest_cc_ratio_invalid <= 0.05 and invalid_pixels > 0)

    row: dict[str, Any] = {
        "split": split_name,
        "filename": filename,
        "mine": mine,
        "group_id": group_id,
        "height": int(height),
        "width": int(width),
        "pixels": int(pixels),
        "rgb_mean_luma_0_1": mean_luma,
        "depth16_zero_pixels": int(np.count_nonzero(raw_invalid)),
        "depth16_zero_ratio": float(np.mean(raw_invalid)),
        "depth8_zero_pixels": invalid_pixels,
        "depth8_invalid_ratio": invalid_pixels / pixels,
        "depth8_valid_pixels": pixels - invalid_pixels,
        "depth8_valid_ratio": (pixels - invalid_pixels) / pixels,
        "depth8_zero_from_positive_depth16_pixels": quantization_zero_pixels,
        "depth8_zero_from_positive_depth16_ratio_image": quantization_zero_pixels / pixels,
        "depth8_zero_from_positive_depth16_ratio_raw_valid": (
            quantization_zero_pixels / raw_valid_pixels if raw_valid_pixels else 0.0
        ),
        "invalid_cc8_count": cc_count,
        "largest_invalid_cc_pixels": largest_cc,
        "largest_invalid_cc_image_ratio": largest_cc_ratio_image,
        "largest_invalid_cc_invalid_ratio": largest_cc_ratio_invalid,
        "cc_area_p50_pixels": area_p50,
        "cc_area_p90_pixels": area_p90,
        "invalid_ratio_stratum": "",
        "large_hole_flag": large_hole,
        "scattered_flag": scattered,
    }
    return row, component_areas


def _stratum(value: float, p25: float, p75: float) -> str:
    if value <= p25:
        return "low"
    if value <= p75:
        return "medium"
    return "high"


def _choose_nearest(rows: list[dict[str, Any]], field: str, target: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: (abs(float(row[field]) - target), row["filename"]))


def _representative(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = (
        "filename",
        "mine",
        "group_id",
        "depth8_invalid_ratio",
        "depth16_zero_ratio",
        "depth8_zero_from_positive_depth16_ratio_image",
        "rgb_mean_luma_0_1",
        "invalid_cc8_count",
        "largest_invalid_cc_image_ratio",
        "largest_invalid_cc_invalid_ratio",
    )
    return {key: row[key] for key in keys}


def _summarize_split(
    split_name: str,
    rows: list[dict[str, Any]],
    component_areas_all: list[int],
    train_p25: float,
    train_p75: float,
) -> dict[str, Any]:
    for row in rows:
        row["invalid_ratio_stratum"] = _stratum(row["depth8_invalid_ratio"], train_p25, train_p75)

    invalid_ratios = [float(row["depth8_invalid_ratio"]) for row in rows]
    raw_invalid_ratios = [float(row["depth16_zero_ratio"]) for row in rows]
    valid_ratios = [float(row["depth8_valid_ratio"]) for row in rows]
    brightness = [float(row["rgb_mean_luma_0_1"]) for row in rows]
    global_pixels = sum(int(row["pixels"]) for row in rows)
    global_invalid_pixels = sum(int(row["depth8_zero_pixels"]) for row in rows)
    global_raw_invalid_pixels = sum(int(row["depth16_zero_pixels"]) for row in rows)
    global_quantization_zero_pixels = sum(
        int(row["depth8_zero_from_positive_depth16_pixels"]) for row in rows
    )

    hist = []
    for lower, upper, label in INVALID_BINS:
        if lower == upper == 1.0:
            count = sum(value == 1.0 for value in invalid_ratios)
        else:
            count = sum(lower <= value < upper for value in invalid_ratios)
        hist.append({"bin": label, "images": int(count)})

    by_mine: dict[str, Any] = {}
    for mine in sorted({row["mine"] for row in rows}):
        mine_rows = [row for row in rows if row["mine"] == mine]
        by_mine[mine] = {
            "images": len(mine_rows),
            "groups": len({row["group_id"] for row in mine_rows}),
            "depth8_invalid_ratio": _distribution([float(row["depth8_invalid_ratio"]) for row in mine_rows]),
            "depth16_zero_ratio": _distribution([float(row["depth16_zero_ratio"]) for row in mine_rows]),
            "rgb_mean_luma_0_1": _distribution([float(row["rgb_mean_luma_0_1"]) for row in mine_rows]),
            "invalid_cc8_count_mean": float(np.mean([row["invalid_cc8_count"] for row in mine_rows])),
        }

    strata: dict[str, Any] = {}
    for name in ("low", "medium", "high"):
        bucket = [row for row in rows if row["invalid_ratio_stratum"] == name]
        mean_ratio = float(np.mean([row["depth8_invalid_ratio"] for row in bucket])) if bucket else 0.0
        strata[name] = {
            "images": len(bucket),
            "invalid_ratio_mean": mean_ratio if bucket else None,
            "representative": _representative(_choose_nearest(bucket, "depth8_invalid_ratio", mean_ratio)),
        }

    area_values = np.asarray(component_areas_all, dtype=np.float64)
    largest = max(rows, key=lambda row: (row["largest_invalid_cc_image_ratio"], row["filename"]))
    high_scatter_pool = [row for row in rows if row["depth8_zero_pixels"] > 0 and row["depth8_invalid_ratio"] >= 0.01]
    if not high_scatter_pool:
        high_scatter_pool = [row for row in rows if row["depth8_zero_pixels"] > 0]
    scatter = max(
        high_scatter_pool,
        key=lambda row: (
            row["invalid_cc8_count"] / max(row["depth8_zero_pixels"], 1),
            row["invalid_cc8_count"],
            row["filename"],
        ),
        default=None,
    )
    p25 = float(np.percentile(invalid_ratios, 25, method="linear"))
    p75 = float(np.percentile(invalid_ratios, 75, method="linear"))
    median = float(np.percentile(invalid_ratios, 50, method="linear"))
    large_hole_rows = [row for row in rows if row["large_hole_flag"]]
    scattered_rows = [row for row in rows if row["scattered_flag"]]

    return {
        "images": len(rows),
        "unique_groups": len({row["group_id"] for row in rows}),
        "mine_counts": {mine: data["images"] for mine, data in by_mine.items()},
        "pixels_total": global_pixels,
        "pixel_weighted_depth8_invalid_ratio": global_invalid_pixels / global_pixels,
        "pixel_weighted_depth16_zero_ratio": global_raw_invalid_pixels / global_pixels,
        "pixel_weighted_depth8_zero_from_positive_depth16_ratio": global_quantization_zero_pixels / global_pixels,
        "per_image_depth8_invalid_ratio": _distribution(invalid_ratios),
        "per_image_depth16_zero_ratio": _distribution(raw_invalid_ratios),
        "per_image_depth8_valid_ratio": _distribution(valid_ratios),
        "per_image_rgb_mean_luma_0_1": _distribution(brightness),
        "pearson_rgb_luma_vs_depth8_invalid_ratio": _pearson(brightness, invalid_ratios),
        "pearson_n_images": len(rows),
        "invalid_ratio_histogram_images": hist,
        "by_mine": by_mine,
        "train_quartile_strata_thresholds": {"p25": train_p25, "p75": train_p75},
        "low_medium_high_strata": strata,
        "connected_components_8": {
            "total_components": int(area_values.size),
            "per_image_count": _distribution([float(row["invalid_cc8_count"]) for row in rows]),
            "largest_component_image_fraction": _distribution(
                [float(row["largest_invalid_cc_image_ratio"]) for row in rows]
            ),
            "component_area_pixels": _distribution(area_values),
            "component_area_image_fraction": _distribution(area_values / EXPECTED_PIXELS),
            "large_hole_definition": (
                "largest 8-connected zero-Depth8 component >=1% of image pixels and >=50% of that image's zero pixels"
            ),
            "large_hole_images": len(large_hole_rows),
            "scattered_definition": (
                "at least 20 8-connected zero-Depth8 components and largest component <=5% of that image's zero pixels"
            ),
            "scattered_images": len(scattered_rows),
        },
        "representatives": {
            "minimum_invalid_ratio": _representative(min(rows, key=lambda row: (row["depth8_invalid_ratio"], row["filename"]))),
            "median_nearest_invalid_ratio": _representative(_choose_nearest(rows, "depth8_invalid_ratio", median)),
            "maximum_invalid_ratio": _representative(max(rows, key=lambda row: (row["depth8_invalid_ratio"], row["filename"]))),
            "largest_connected_component": _representative(largest),
            "largest_hole_candidate": _representative(
                max(large_hole_rows, key=lambda row: row["largest_invalid_cc_image_ratio"], default=None)
            ),
            "most_scattered_candidate": _representative(scatter),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cast(Any, rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="audit only train-dev; val-dev is included by default",
    )
    args = parser.parse_args()

    project_root = _project_root()
    dataset_root = project_root.parent / "dataset" / "MUSeg_DFormer"
    split_root = project_root / "data" / "splits" / "MUSeg" / "dev-v1"
    output_root = project_root / "outputs" / "direction-audit-20261002"
    for directory in (dataset_root / "RGB", dataset_root / "Depth", dataset_root / "Depth16"):
        if not directory.is_dir():
            raise FileNotFoundError(f"required allowed input directory is unavailable: {directory}")

    split_names = ["train-dev"] + ([] if args.skip_val else ["val-dev"])
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    areas_by_split: dict[str, list[int]] = {}
    for split_name in split_names:
        allowlisted = _read_allowlist(split_root, split_name)
        rows: list[dict[str, Any]] = []
        areas: list[int] = []
        for filename, mine, group_id in allowlisted:
            row, component_areas = _audit_image(dataset_root, split_name, filename, mine, group_id)
            rows.append(row)
            areas.extend(int(value) for value in component_areas)
        rows_by_split[split_name] = rows
        areas_by_split[split_name] = areas

    train_ratios = [float(row["depth8_invalid_ratio"]) for row in rows_by_split["train-dev"]]
    train_p25 = float(np.percentile(train_ratios, 25, method="linear"))
    train_p75 = float(np.percentile(train_ratios, 75, method="linear"))
    summaries: dict[str, Any] = {}
    for split_name, rows in rows_by_split.items():
        summaries[split_name] = _summarize_split(
            split_name, rows, areas_by_split[split_name], train_p25, train_p75
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, rows in rows_by_split.items():
        _write_csv(output_root / f"{split_name}_per-image.csv", rows)

    summary = {
        "schema_version": "museg-natural-depth-missing-audit-v1",
        "scope": {
            "allowlists": [f"data/splits/MUSeg/dev-v1/{name}.txt" for name in split_names],
            "official_test_read": False,
            "label_files_read": False,
            "training_gpu_evaluation_or_model_forward": False,
            "raw_input_directories": ["RGB", "Depth16", "Depth"],
            "raw_resolution_expected": "1082x932 from dataset metadata and MUSeg paper fixed crop",
        },
        "definitions": {
            "Depth16_invalid_sentinel": "raw Depth16 == 0",
            "Depth8_invalid_sentinel": "quantized Depth8 == 0; valid iff Depth8 > 0",
            "quantization_created_zero": "Depth16 > 0 and Depth8 == 0",
            "fixed_quantization": "round(Depth16 * 255 / 13932), checked pixelwise against stored Depth8",
            "brightness": "mean BT.601 luma (0.299 R + 0.587 G + 0.114 B) over uint8 code values, divided by 255; no gamma/ICC correction",
            "group_parse": "mine = first filename stem segment; group_id = first four ASCII hyphen-separated stem segments",
            "connected_components": "8-connectivity over transient Depth8 == 0 masks; no mask is written",
            "percentiles": "NumPy percentile with linear interpolation; image-level distributions weight each image equally",
            "invalid_ratio_bins": [
                {"bin": name, "lower_inclusive": lower, "upper_exclusive": upper}
                for lower, upper, name in INVALID_BINS[:-1]
            ] + [{"bin": "[1,1]", "lower_inclusive": 1.0, "upper_inclusive": 1.0}],
            "low_medium_high": "train-dev P25/P75 thresholds applied to both splits: low <= P25, medium (P25,P75], high > P75",
        },
        "inputs": {
            "dataset_root": str(dataset_root),
            "split_root": str(split_root),
        },
        "train_dev_quartile_cutoffs": {"p25_depth8_invalid_ratio": train_p25, "p75_depth8_invalid_ratio": train_p75},
        "splits": summaries,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(output_root / f"{name}_per-image.csv") for name in split_names] + [str(summary_path)], "splits": {name: {"images": data["images"], "groups": data["unique_groups"], "invalid_ratio": data["per_image_depth8_invalid_ratio"], "pearson_luma_invalid": data["pearson_rgb_luma_vs_depth8_invalid_ratio"]} for name, data in summaries.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
