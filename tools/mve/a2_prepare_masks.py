"""Prepare deterministic Depth=0 corruption masks for the A2 screening run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SampleStats:
    sample_id: str
    depth16_path: str
    depth_path: str
    label_path: str
    height: int
    width: int
    valid_depth_ratio: float
    foreground_pixels: int


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    default_dataset = repository_root.parent / "dataset" / "MUSeg_DFormer"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--q", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument("--block-count", type=int, default=2)
    return parser.parse_args()


def read_test_ids(dataset_root: Path) -> list[str]:
    split_path = dataset_root / "test.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"missing official test split: {split_path}")
    return [Path(line.strip()).stem for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inspect_sample(dataset_root: Path, sample_id: str) -> SampleStats | None:
    depth16_path = dataset_root / "Depth16" / f"{sample_id}.png"
    depth_path = dataset_root / "Depth" / f"{sample_id}.png"
    label_path = dataset_root / "Label" / f"{sample_id}.png"
    depth16 = cv2.imread(str(depth16_path), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if depth16 is None or depth is None or label is None:
        return None
    if depth16.dtype != np.uint16 or depth16.ndim != 2:
        raise ValueError(f"expected uint16 Depth16 for {sample_id}, got {depth16.dtype} {depth16.shape}")
    if depth.ndim != 2 or label.ndim != 2 or depth.shape != depth16.shape or label.shape != depth16.shape:
        raise ValueError(f"modality shape mismatch for {sample_id}")

    valid = depth16 > 0
    foreground = label > 0
    return SampleStats(
        sample_id=sample_id,
        depth16_path=str(depth16_path),
        depth_path=str(depth_path),
        label_path=str(label_path),
        height=int(depth16.shape[0]),
        width=int(depth16.shape[1]),
        valid_depth_ratio=float(valid.mean()),
        foreground_pixels=int(foreground.sum()),
    )


def choose_samples(dataset_root: Path, sample_count: int) -> list[SampleStats]:
    candidates = [
        stats
        for sample_id in read_test_ids(dataset_root)
        if (stats := inspect_sample(dataset_root, sample_id)) is not None
        and stats.foreground_pixels > 0
    ]
    candidates.sort(key=lambda stats: (-stats.valid_depth_ratio, stats.sample_id))
    if len(candidates) < sample_count:
        raise ValueError(f"only {len(candidates)} eligible test samples; need {sample_count}")
    return candidates[:sample_count]


def select_block_mask(valid: np.ndarray, target_count: int, rng: np.random.Generator, block_count: int) -> np.ndarray:
    if target_count == 0:
        return np.zeros(valid.shape, dtype=np.uint8)

    height, width = valid.shape
    selected = np.zeros(valid.shape, dtype=bool)
    max_block_height = max(1, int(height * 0.35))
    max_block_width = max(1, int(width * 0.35))
    for _ in range(max(1, block_count * 8)):
        if selected.sum() >= target_count:
            break
        block_height = int(rng.integers(1, max_block_height + 1))
        block_width = int(rng.integers(1, max_block_width + 1))
        top = int(rng.integers(0, max(1, height - block_height + 1)))
        left = int(rng.integers(0, max(1, width - block_width + 1)))
        selected[top : top + block_height, left : left + block_width] |= valid[top : top + block_height, left : left + block_width]

    if selected.sum() < target_count:
        remaining = np.flatnonzero(valid & ~selected)
        remaining = remaining[rng.permutation(remaining.size)]
        selected.flat[remaining[: target_count - selected.sum()]] = True
    elif selected.sum() > target_count:
        chosen = np.flatnonzero(selected)
        chosen = chosen[rng.permutation(chosen.size)]
        selected.flat[chosen[target_count:]] = False
    return selected.astype(np.uint8)


def write_mask(mask_path: Path, mask: np.ndarray) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"failed to write mask: {mask_path}")


def build_masks(output_root: Path, samples: list[SampleStats], q_values: list[float], seed: int, block_count: int) -> dict:
    masks_root = output_root / "masks"
    manifest_samples = []
    for sample_index, stats in enumerate(samples):
        depth16 = cv2.imread(stats.depth16_path, cv2.IMREAD_UNCHANGED)
        valid = depth16 > 0
        valid_count = int(valid.sum())
        sample_conditions = []
        for q in q_values:
            target_count = int(round(valid_count * q))
            condition_seed = seed + sample_index * 1009 + int(round(q * 1000))
            rng = np.random.default_rng(condition_seed)
            mask = select_block_mask(valid, target_count, rng, block_count)
            mask_path = masks_root / f"q_{q:g}" / "block" / f"{stats.sample_id}.png"
            write_mask(mask_path, mask)
            sample_conditions.append(
                {
                    "q": q,
                    "mask_type": "block",
                    "seed": condition_seed,
                    "mask_path": str(mask_path.relative_to(output_root)),
                    "valid_depth_pixels": valid_count,
                    "injected_zero_pixels": int(mask.sum()),
                    "actual_added_fraction_of_valid": float(mask.sum() / valid_count) if valid_count else 0.0,
                }
            )
        manifest_samples.append({"stats": asdict(stats), "conditions": sample_conditions})

    manifest = {
        "schema_version": 1,
        "dataset": "MUSeg_DFormer",
        "split": "official_test",
        "mask_type": "block",
        "seed": seed,
        "selection": "foreground_present_then_highest_Depth16_nonzero_ratio",
        "injection_definition": "additional zeros are selected only from original Depth16 > 0 pixels",
        "samples": manifest_samples,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "a2_screening_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0 or args.block_count <= 0:
        raise ValueError("sample-count and block-count must be positive")
    if any(q < 0.0 or q > 1.0 for q in args.q):
        raise ValueError("q values must be in [0, 1]")
    samples = choose_samples(args.dataset_root, args.sample_count)
    manifest = build_masks(args.output_root, samples, args.q, args.seed, args.block_count)
    print(f"wrote {len(manifest['samples'])} samples to {args.output_root / 'a2_screening_manifest.json'}")


if __name__ == "__main__":
    main()