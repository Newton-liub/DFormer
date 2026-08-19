"""Convert the original MUSeg release into the DFormer directory layout.

The converter keeps the original 16-bit depth and creates the 8-bit model input
with one fixed dataset-wide quantization range. It never modifies the source
release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DEPTH_MAX_RAW = 13932
SUPPORTED_MINE_DIRS = tuple(f"{index:02d}-Mine" for index in range(1, 7))


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: Path
    depth16: Path
    label: Path


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    test: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    default_source = repository_root.parent / "dataset" / "MUSeg"
    default_output = repository_root.parent / "dataset" / "MUSeg_DFormer"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument(
        "--split-zip",
        type=Path,
        default=None,
        help="Official DatasetSplit.zip; defaults to SOURCE/Experiment/DatasetSplit.zip.",
    )
    parser.add_argument(
        "--depth-max-raw",
        type=int,
        default=DEFAULT_DEPTH_MAX_RAW,
        help=f"Fixed global raw-depth maximum (default: {DEFAULT_DEPTH_MAX_RAW}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory only after conversion succeeds.",
    )
    return parser.parse_args()


def canonical_label_id(path: Path) -> str:
    if not path.stem.endswith("_label"):
        raise ValueError(f"not a semantic label file: {path}")
    return path.stem.removesuffix("_label")


def collect_samples(source_root: Path) -> dict[str, Sample]:
    samples: dict[str, Sample] = {}

    for mine_name in SUPPORTED_MINE_DIRS:
        mine_root = source_root / mine_name
        image_root = mine_root / "Image"
        depth_root = mine_root / "Depth"
        label_root = mine_root / "Label"
        missing_dirs = [path for path in (image_root, depth_root, label_root) if not path.is_dir()]
        if missing_dirs:
            raise FileNotFoundError(
                f"missing MUSeg source directories: {', '.join(str(path) for path in missing_dirs)}"
            )

        images = {path.stem: path for path in image_root.glob("*.jpg")}
        depths = {path.stem: path for path in depth_root.glob("*.png")}
        labels = {
            canonical_label_id(path): path
            for path in label_root.glob("*_label.png")
        }
        modality_sets = {"Image": set(images), "Depth": set(depths), "Label": set(labels)}
        if len({frozenset(names) for names in modality_sets.values()}) != 1:
            details = ", ".join(f"{name}={len(names)}" for name, names in modality_sets.items())
            raise ValueError(f"unmatched modalities in {mine_name}: {details}")

        for sample_id in sorted(images):
            if sample_id in samples:
                raise ValueError(f"duplicate sample ID across mines: {sample_id}")
            samples[sample_id] = Sample(sample_id, images[sample_id], depths[sample_id], labels[sample_id])

    if not samples:
        raise ValueError(f"no MUSeg samples found under {source_root}")
    return samples


def read_split_lines(split_zip: Path, member: str) -> tuple[str, ...]:
    with zipfile.ZipFile(split_zip) as archive:
        try:
            content = archive.read(member).decode("utf-8")
        except KeyError as error:
            raise ValueError(f"{split_zip} does not contain {member}") from error

    sample_ids = tuple(line.strip() for line in content.splitlines() if line.strip())
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"duplicate entries in {member}")
    return sample_ids


def load_split(split_zip: Path, samples: dict[str, Sample]) -> Split:
    train = read_split_lines(split_zip, "train.txt")
    test = read_split_lines(split_zip, "val.txt")
    train_set, test_set = set(train), set(test)
    known_ids = set(samples)

    if train_set & test_set:
        raise ValueError("official train/test split contains overlapping samples")
    if train_set | test_set != known_ids:
        missing = sorted(known_ids - train_set - test_set)
        unknown = sorted(train_set | test_set - known_ids)
        raise ValueError(
            f"split does not cover source exactly: missing={len(missing)}, unknown={len(unknown)}"
        )
    return Split(train=train, test=test)


def quantize_depth(depth16: np.ndarray, depth_max_raw: int) -> np.ndarray:
    if depth16.dtype != np.uint16 or depth16.ndim != 2:
        raise ValueError(f"expected 2-D uint16 depth, got shape={depth16.shape}, dtype={depth16.dtype}")
    observed_max = int(depth16.max())
    if observed_max > depth_max_raw:
        raise ValueError(
            f"raw depth value {observed_max} exceeds fixed maximum {depth_max_raw}"
        )
    quantized = np.rint(depth16.astype(np.float64) * 255.0 / depth_max_raw)
    return quantized.astype(np.uint8)


def validate_source_sample(sample: Sample) -> tuple[tuple[int, int], int]:
    image = cv2.imread(str(sample.image), cv2.IMREAD_UNCHANGED)
    depth16 = cv2.imread(str(sample.depth16), cv2.IMREAD_UNCHANGED)
    label = cv2.imread(str(sample.label), cv2.IMREAD_UNCHANGED)
    if image is None or depth16 is None or label is None:
        raise ValueError(f"failed to read one or more files for {sample.sample_id}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"RGB must be uint8 with 3 channels: {sample.image}")
    if depth16.ndim != 2 or depth16.dtype != np.uint16:
        raise ValueError(f"Depth must be uint16 single-channel: {sample.depth16}")
    if label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError(f"Label must be uint8 single-channel: {sample.label}")
    if image.shape[:2] != depth16.shape or image.shape[:2] != label.shape:
        raise ValueError(f"modality dimensions differ for {sample.sample_id}")
    return (int(image.shape[1]), int(image.shape[0])), int(depth16.max())


def copy_sample(sample: Sample, output_root: Path, depth_max_raw: int) -> None:
    shutil.copy2(sample.image, output_root / "RGB" / f"{sample.sample_id}.jpg")
    shutil.copy2(sample.depth16, output_root / "Depth16" / f"{sample.sample_id}.png")
    shutil.copy2(sample.label, output_root / "Label" / f"{sample.sample_id}.png")

    depth16 = cv2.imread(str(sample.depth16), cv2.IMREAD_UNCHANGED)
    depth8 = quantize_depth(depth16, depth_max_raw)
    output_depth = output_root / "Depth" / f"{sample.sample_id}.png"
    if not cv2.imwrite(str(output_depth), depth8):
        raise OSError(f"failed to write converted depth: {output_depth}")


def write_split(path: Path, sample_ids: tuple[str, ...]) -> None:
    path.write_text(
        "".join(f"RGB/{sample_id}.jpg\n" for sample_id in sample_ids),
        encoding="utf-8",
    )


def write_metadata(
    output_root: Path,
    split_zip: Path,
    samples: dict[str, Sample],
    split: Split,
    image_size_wh: tuple[int, int],
    observed_depth_max: int,
    depth_max_raw: int,
) -> None:
    metadata = {
        "schema_version": 1,
        "dataset": "MUSeg",
        "source_layout": "original_six_mine_directories",
        "split_source": "Experiment/DatasetSplit.zip",
        "split_source_sha256": hashlib.sha256(split_zip.read_bytes()).hexdigest(),
        "sample_count": len(samples),
        "split_policy": "official_group_disjoint",
        "split_counts": {"train": len(split.train), "test": len(split.test)},
        "image_size_wh": list(image_size_wh),
        "modalities": {
            "rgb": {"directory": "RGB", "dtype": "uint8", "channels": 3},
            "depth16": {"directory": "Depth16", "dtype": "uint16", "channels": 1},
            "depth8": {"directory": "Depth", "dtype": "uint8", "channels": 1},
            "label": {"directory": "Label", "dtype": "uint8", "channels": 1},
        },
        "depth_quantization": {
            "method": "global_linear_round",
            "formula": "round(depth16 * 255 / depth_max_raw)",
            "min_raw": 0,
            "max_raw": depth_max_raw,
            "observed_source_max_raw": observed_depth_max,
            "invalid_raw": 0,
            "invalid_policy": "preserve_zero",
        },
        "label_storage": "source *_label.png copied without ID remapping",
        "generated_for": "DFormer RGBXDataset",
    }
    (output_root / "dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_output(
    output_root: Path,
    expected_ids: set[str],
    split: Split,
    image_size_wh: tuple[int, int],
    depth_max_raw: int,
) -> None:
    suffixes = {"RGB": ".jpg", "Depth": ".png", "Depth16": ".png", "Label": ".png"}
    for directory, suffix in suffixes.items():
        actual_ids = {path.stem for path in (output_root / directory).glob(f"*{suffix}")}
        if actual_ids != expected_ids:
            raise ValueError(
                f"output {directory} IDs differ: expected={len(expected_ids)}, actual={len(actual_ids)}"
            )

    expected_split_lines = {
        "train.txt": [f"RGB/{sample_id}.jpg" for sample_id in split.train],
        "test.txt": [f"RGB/{sample_id}.jpg" for sample_id in split.test],
    }
    for filename, expected_lines in expected_split_lines.items():
        actual_lines = (output_root / filename).read_text(encoding="utf-8").splitlines()
        if actual_lines != expected_lines:
            raise ValueError(f"output {filename} differs from the official split")

    expected_shape = (image_size_wh[1], image_size_wh[0])
    label_ids: set[int] = set()
    for sample_id in sorted(expected_ids):
        rgb = cv2.imread(str(output_root / "RGB" / f"{sample_id}.jpg"), cv2.IMREAD_UNCHANGED)
        depth8 = cv2.imread(str(output_root / "Depth" / f"{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        depth16 = cv2.imread(str(output_root / "Depth16" / f"{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(output_root / "Label" / f"{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        if any(value is None for value in (rgb, depth8, depth16, label)):
            raise ValueError(f"failed to read converted files for {sample_id}")
        if rgb.shape != (*expected_shape, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"invalid converted RGB for {sample_id}")
        if depth8.shape != expected_shape or depth8.dtype != np.uint8:
            raise ValueError(f"invalid converted 8-bit depth for {sample_id}")
        if depth16.shape != expected_shape or depth16.dtype != np.uint16:
            raise ValueError(f"invalid converted 16-bit depth for {sample_id}")
        if label.shape != expected_shape or label.dtype != np.uint8:
            raise ValueError(f"invalid converted label for {sample_id}")
        if not np.array_equal(depth8, quantize_depth(depth16, depth_max_raw)):
            raise ValueError(f"depth quantization mismatch for {sample_id}")
        label_ids.update(int(value) for value in np.unique(label))

    if label_ids != set(range(16)):
        raise ValueError(f"expected label IDs 0..15, got {sorted(label_ids)}")


def convert(
    source_root: Path,
    output_root: Path,
    split_zip: Path,
    depth_max_raw: int,
    overwrite: bool,
) -> None:
    if depth_max_raw <= 0 or depth_max_raw > np.iinfo(np.uint16).max:
        raise ValueError("depth-max-raw must be between 1 and 65535")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if not split_zip.is_file():
        raise FileNotFoundError(f"split archive does not exist: {split_zip}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output is not empty; use --overwrite: {output_root}")

    samples = collect_samples(source_root)
    split = load_split(split_zip, samples)
    image_size: tuple[int, int] | None = None
    observed_depth_max = 0
    for sample in samples.values():
        sample_size, sample_depth_max = validate_source_sample(sample)
        image_size = sample_size if image_size is None else image_size
        if sample_size != image_size:
            raise ValueError(f"inconsistent image size for {sample.sample_id}: {sample_size}")
        observed_depth_max = max(observed_depth_max, sample_depth_max)
    if observed_depth_max > depth_max_raw:
        raise ValueError(
            f"source maximum {observed_depth_max} exceeds fixed maximum {depth_max_raw}"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        for directory in ("RGB", "Depth", "Depth16", "Label"):
            (temporary_root / directory).mkdir()
        for sample in samples.values():
            copy_sample(sample, temporary_root, depth_max_raw)
        write_split(temporary_root / "train.txt", split.train)
        write_split(temporary_root / "test.txt", split.test)
        write_metadata(
            temporary_root,
            split_zip,
            samples,
            split,
            image_size,
            observed_depth_max,
            depth_max_raw,
        )
        verify_output(
            temporary_root,
            set(samples),
            split,
            image_size,
            depth_max_raw,
        )

        if output_root.exists():
            shutil.rmtree(output_root)
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(f"Converted {len(samples)} samples to {output_root}")
    print(f"Official split: train={len(split.train)}, test={len(split.test)}")
    print(f"Depth mapping: round(depth16 * 255 / {depth_max_raw})")
    print(f"Observed source depth maximum: {observed_depth_max}")


def main() -> None:
    args = parse_args()
    split_zip = args.split_zip or args.source_root / "Experiment" / "DatasetSplit.zip"
    convert(
        source_root=args.source_root.resolve(),
        output_root=args.output_root.resolve(),
        split_zip=split_zip.resolve(),
        depth_max_raw=args.depth_max_raw,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()