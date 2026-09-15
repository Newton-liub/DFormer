#!/usr/bin/env python3
"""CPU-only, read-only audit of Depth validity and reliability supervision on train-dev.

Why this tool exists
--------------------
The frozen A2 v2 supervision target is

    R_D^sup(p) = depth_valid_pre(p) * R_D^syn(p),

so every natively invalid MUSeg Depth pixel (``raw Depth == 0``) becomes a hard target
``0``. If MUSeg Depth is very sparse, the auxiliary continuous BCE could be dominated by
a "zero / non-zero detector" signal instead of a graded reliability signal. A single
observed sample was almost entirely ``Depth == 0``, which cannot be extrapolated. This
tool measures the real training distribution instead of guessing it.

What one run collects (facts only)
----------------------------------
1. **Raw aligned grid** (no geometry augmentation, deterministic): for every ``train-dev``
   sample, the fraction of pixels with ``Depth > 0``, read with ``cv2.IMREAD_UNCHANGED``
   from ``Depth/<stem><x_format>``; plus the full distribution summary, the sample counts
   below the 1/5/10/25/50 percent validity thresholds and the number of all-zero samples.
2. **Training input geometry** (post ``TrainPre`` mirror/scale/crop/pad): the same
   distribution of ``depth_valid_pre`` *inside* the geometry ``valid_mask``, produced by
   the repository's own ``RGBXDataset(setting, "train", TrainPre(...))`` pipeline, plus the
   per-sample ``valid_mask`` coverage.
3. **Location group summary** using the repository's frozen rule
   ``tools.mve.dvc_a1_core.location_group`` (first four ASCII-hyphen-separated stem
   segments, the same rule frozen as ``first-four-ascii-hyphen-separated-stem-segments``
   in the DVC-A1 / DVG-B1 protocols).
4. **Supervision pixel composition** at training progress early / mid / late: each
   supervised pixel of a deterministically drawn sample set is classified into exactly one
   of ``clean_valid`` / ``natural_invalid`` / ``newly_valid`` / ``synthetic_missing`` /
   ``implicit_quality`` by calling the frozen
   ``utils.dataloader.mmfr_training_v2.build_mmfr_training_batch_v2`` helper. The frozen
   helper's own (overlapping) ``natural_invalid_pixels`` / ``synthetic_missing_pixels`` /
   ``newly_valid_pixels`` / ``implicit_quality_pixels`` census is reported next to the
   disjoint split, never merged with it.
5. **``R_D^sup`` distribution** over the supervised pixels: the 11-bin histogram
   (``0``, ``(0, 0.1]`` ... ``(0.9, 1.0]``), mean, median and the exactly-zero share, per
   progress point.

What this tool does **not** do
------------------------------
It collects facts only. It does not modify any loss, any target definition, any
configuration, any dataset code, or any existing file; it does not "repair" the
supervision semantics; it never reads the sealed ``official-test`` split; it never starts
training, evaluation or any cloud job; and it never touches a GPU -- ``CUDA_VISIBLE_DEVICES``
is forced to ``-1`` *before* ``torch`` is imported, every helper output tensor is asserted to
be on the CPU, and both facts are recorded in the report.

Recommended run command::

    python tools/mmfr/train_dev_depth_validity_audit.py

Evidence
--------
``outputs/mmfr-a2-v2-depth-validity-audit/train-dev-depth-validity-audit.json``
(schema ``museg-mmfr-a2-v2-train-dev-depth-validity-audit-v1``).
"""

from __future__ import annotations

import os

# Must be set before ``torch`` is imported so that no CUDA device can ever be selected.
# ``-1`` is a device index that never exists; an *empty* value is ignored as "unset" by this
# CUDA build, so ``-1`` is the value that actually makes ``torch.cuda.is_available()`` False.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import copy  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import torch  # noqa: E402

from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.dataloader.dataloader import TrainPre  # noqa: E402
from utils.dataloader.mmfr_training import (  # noqa: E402
    CLEAN_PROBABILITY,
    CORRUPTION_SEED,
    DEPTH_PLANE_MEAN,
    DEPTH_PLANE_STD,
    MAX_SPECS,
    curriculum_progress,
    normalized_to_uint8,
)
from utils.dataloader.mmfr_training_v2 import (  # noqa: E402
    DEPTH_RELIABILITY_INDEX,
    build_mmfr_training_batch_v2,
)
from tools.mve.dvc_a1_core import location_group  # noqa: E402

SCHEMA = "museg-mmfr-a2-v2-train-dev-depth-validity-audit-v1"
CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v2"
TRAIN_SPLIT = REPO_ROOT / "data" / "splits" / "MUSeg" / "dev-v1" / "train-dev.txt"
EXPECTED_TRAIN_SPLIT_SHA256 = "a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470"
EXPECTED_TRAIN_SAMPLE_COUNT = 1277

#: Geometry seed for the part-2 whole-split scan. Set exactly once before the first
#: ``__getitem__`` and never reset, so the mirror/scale/crop sequence is one deterministic
#: stream over the split order. ``TrainPre`` draws its geometry from the process-wide
#: ``random`` module only (mirror, scale, crop position), so this seed fully determines it.
GEOMETRY_SEED = 2026091501
#: Geometry seed for the part-4 sample set: a *separate* fresh draw stream, set once before
#: the ascending 16-sample read sequence, so part 4 stays reproducible even if part 2 is
#: skipped. It is deliberately not the same stream as part 2.
SAMPLE_GEOMETRY_SEED = 2026091502
#: Number of location groups drawn for the part-4 supervision composition.
GROUP_SAMPLE_COUNT = 16
#: Training progress points ``(label, epoch, iteration)``; ``iteration = 0`` by request.
PROGRESS_POINTS: Tuple[Tuple[str, int, int], ...] = (
    ("early", 1, 0),
    ("mid", 250, 0),
    ("late", 500, 0),
)
#: Validity-fraction thresholds for the "how many samples are that sparse" counts.
VALIDITY_THRESHOLDS: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50)
PERCENTILES: Tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
#: Five-way partition of the geometry valid region used in part 4.
SUPERVISION_CLASSES: Tuple[str, ...] = (
    "clean_valid",
    "natural_invalid",
    "synthetic_missing",
    "newly_valid",
    "implicit_quality",
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "mmfr-a2-v2-depth-validity-audit"
REPORT_PATH = OUTPUT_DIR / "train-dev-depth-validity-audit.json"


# ---------------------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def script_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def load_config(module_name: str) -> Any:
    """Import the frozen v2 config and return a copy of ``C`` without mutating the module."""
    module = importlib.import_module(module_name)
    if not hasattr(module, "C"):
        raise RuntimeError(f"config module {module_name!r} does not expose C")
    config = copy.copy(getattr(module, "C"))
    mmfr_a2 = dict(getattr(config, "mmfr_a2", None) or {})
    if not mmfr_a2.get("corruption") or not mmfr_a2.get("reliability_head"):
        raise RuntimeError(
            f"{module_name!r} does not bind MMFR-A2 Depth corruption with a reliability head"
        )
    return config


def read_split_entries(split_path: Path) -> List[str]:
    with open(split_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def describe(values: Sequence[float]) -> Dict[str, Any]:
    """Distribution summary: count/mean/median/std/min/max plus the requested percentiles.

    ``std`` is the population standard deviation (``ddof = 0``, the numpy default).
    Percentiles use numpy's default linear interpolation between order statistics.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    summary: Dict[str, Any] = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std_population_ddof0": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "percentiles": {f"P{q}": float(np.percentile(array, q)) for q in PERCENTILES},
    }
    return summary


def validity_summary(values: Sequence[float]) -> Dict[str, Any]:
    """Distribution summary plus the sparse-sample counts for a batch of validity fractions."""
    array = np.asarray(values, dtype=np.float64)
    summary = describe(array)
    summary["below_threshold_sample_counts"] = {
        f"lt_{int(round(threshold * 100))}pct": int(np.count_nonzero(array < threshold))
        for threshold in VALIDITY_THRESHOLDS
    }
    summary["exact_zero_sample_count"] = int(np.count_nonzero(array == 0.0))
    summary["exact_one_sample_count"] = int(np.count_nonzero(array == 1.0))
    return summary


def target_histogram(values: np.ndarray) -> Dict[str, int]:
    """11-bin histogram: ``0``, ``(0, 0.1]`` ... ``(0.9, 1.0]``, plus an overflow guard."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.1, 1.0, 10)
    zero_count = int(np.count_nonzero(flat == 0.0))
    nonzero = flat[flat != 0.0]
    bins = np.digitize(nonzero, edges, right=True)
    histogram: Dict[str, int] = {"0": zero_count}
    for index, edge in enumerate(edges):
        lower_label = "0" if index == 0 else f"{float(edges[index - 1]):.1f}"
        histogram[f"({lower_label},{float(edge):.1f}]"] = int(np.count_nonzero(bins == index))
    histogram["greater_than_1.0"] = int(np.count_nonzero(bins > edges.size - 1))
    return histogram


def read_raw_depth_validity(depth_path: Path) -> Dict[str, Any]:
    """Read one raw ``Depth`` image and measure its ``> 0`` pixel fraction.

    The Depth plane is a single-channel uint8 PNG in this dataset; a three-channel array is
    accepted as well and channel 0 is used, matching the ``IMREAD_GRAYSCALE`` read of
    ``RGBXDataset``.
    """
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"could not read Depth image {depth_path}")
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    if raw.ndim != 2:
        raise RuntimeError(f"unexpected Depth array shape {raw.shape} for {depth_path}")
    if raw.dtype != np.uint8:
        raise RuntimeError(f"Depth must be uint8, got {raw.dtype} for {depth_path}")
    valid = raw > 0
    return {
        "shape": [int(raw.shape[0]), int(raw.shape[1])],
        "pixels": int(valid.size),
        "valid_pixels": int(np.count_nonzero(valid)),
        "valid_fraction": float(np.count_nonzero(valid)) / float(valid.size),
        "all_zero": bool(not valid.any()),
    }


def build_dataset_setting(config: Any) -> Dict[str, Any]:
    """Mirror ``get_train_loader``'s dataset setting exactly."""
    return {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "val_source": getattr(config, "val_source", getattr(config, "eval_source", None)),
        "test_source": getattr(config, "test_source", None),
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
        "channel_order": config.channel_order,
    }


def build_train_dataset(config: Any) -> RGBXDataset:
    return RGBXDataset(
        build_dataset_setting(config),
        "train",
        TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config),
    )


def corruption_kwargs(config: Any) -> Dict[str, Any]:
    """Frozen corruption parameters taken from the config, never re-declared here."""
    corruption = dict(config.mmfr_a2["corruption"])
    return {
        "corruption_seed": int(corruption.get("seed", CORRUPTION_SEED)),
        "p_clean": float(corruption.get("p_clean", CLEAN_PROBABILITY)),
        "max_specs": int(corruption.get("max_specs", MAX_SPECS)),
        "niters_per_epoch": int(config.niters_per_epoch),
        "nepochs": int(config.nepochs),
        "rgb_mean": np.asarray(config.norm_mean, dtype=np.float64),
        "rgb_std": np.asarray(config.norm_std, dtype=np.float64),
    }


def geometry_inputs(dataset: RGBXDataset, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """Load one preprocessed train-dev sample and return ``(rgb, depth, stem, entry)``."""
    entry = str(dataset._file_names[index])
    item = dataset[index]
    stem = Path(entry).stem
    return item["data"], item["modal_x"], stem, entry


def run_helper(
    config: Any,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    stem: str,
    epoch: int,
    iteration: int,
) -> Mapping[str, Any]:
    kwargs = corruption_kwargs(config)
    if rgb.is_cuda or depth.is_cuda:
        raise RuntimeError("this audit must stay on the CPU; a CUDA tensor was supplied")
    outputs = build_mmfr_training_batch_v2(
        rgb[None],
        depth[None],
        [stem],
        epoch=int(epoch),
        iteration=int(iteration),
        niters_per_epoch=int(kwargs["niters_per_epoch"]),
        nepochs=int(kwargs["nepochs"]),
        rgb_mean=kwargs["rgb_mean"],
        rgb_std=kwargs["rgb_std"],
        corruption_seed=int(kwargs["corruption_seed"]),
        p_clean=float(kwargs["p_clean"]),
        max_specs=int(kwargs["max_specs"]),
    )
    for key in ("rgb", "depth", "raw_rgb", "raw_depth", "reliability_target", "valid_mask", "depth_valid_pre", "depth_valid_post"):
        if outputs[key].is_cuda:
            raise RuntimeError(f"this audit must stay on the CPU; {key} landed on CUDA")
    return outputs


# ---------------------------------------------------------------------------------------
# Part 4 / 5: supervision composition of one helper output
# ---------------------------------------------------------------------------------------
def supervision_record(
    outputs: Mapping[str, Any],
    depth_input: torch.Tensor,
    index: int,
    stem: str,
    label: str,
) -> Dict[str, Any]:
    """Classify every supervised pixel of one batch-1 helper output.

    Two censuses are reported side by side and are never merged.

    1. ``partition_class_pixel_counts`` -- a **disjoint** five-way split of the geometry
       valid region, so that every supervised pixel belongs to exactly one class:

       * ``clean_valid``      ``valid & pre & post & (pre == post)``
       * ``natural_invalid``  ``valid & ~pre & ~post``  (natively invalid, still invalid after corruption)
       * ``newly_valid``      ``valid & ~pre & post``
       * ``synthetic_missing`` ``valid & pre & ~post``
       * ``implicit_quality`` ``valid & pre & post & (pre != post)``

       The five counts sum to ``valid_pixels`` exactly, and this is the census the report's
       "supervision composition" means.

    2. ``helper_frozen_census`` -- the frozen helper's own ``_pixel_statistics`` fields,
       reported verbatim because they are the protocol's published vocabulary. The helper's
       ``natural_invalid_pixels`` counts **all** natively invalid pixels, so it deliberately
       *overlaps* ``newly_valid_pixels``; the two censuses are related by
       ``helper.natural_invalid_pixels == partition.natural_invalid + partition.newly_valid``.
       The audit does not redefine the helper's field; it adds the disjoint split next to it.

    Every mask and every count is recomputed here from the returned tensors and from the
    reconstructed pre/post uint8 Depth planes, then compared against the helper metadata, so
    the audit does not take the helper's numbers on trust.
    """
    valid = outputs["valid_mask"][0, 0].detach().cpu().numpy().astype(bool)
    pre_reported = outputs["depth_valid_pre"][0, 0].detach().cpu().numpy().astype(bool)
    post_reported = outputs["depth_valid_post"][0, 0].detach().cpu().numpy().astype(bool)
    metadata = dict(outputs["metadata"][0])
    target = (
        outputs["reliability_target"][0, int(DEPTH_RELIABILITY_INDEX)].detach().cpu().numpy().astype(np.float64)
    )

    # Independent reconstruction. ``depth_input`` is the exact TrainPre-normalized tensor the
    # helper consumed, so the pre-corruption uint8 plane is reproduced through the same frozen
    # inverse; the post-corruption uint8 plane comes back from the returned ``raw_depth``,
    # which the helper builds as ``corrupted_uint8 / 255`` with padding zeroed.
    depth_mean64 = np.asarray((DEPTH_PLANE_MEAN,), dtype=np.float64)
    depth_std64 = np.asarray((DEPTH_PLANE_STD,), dtype=np.float64)
    pre_uint8 = normalized_to_uint8(depth_input[:1].detach().cpu().numpy(), depth_mean64, depth_std64)[0]
    pre_uint8 = np.where(valid, pre_uint8, np.uint8(0)).astype(np.uint8)
    raw_depth = outputs["raw_depth"][0, 0].detach().cpu().numpy().astype(np.float64)
    post_uint8 = np.rint(raw_depth * 255.0).astype(np.uint8)
    pre_mask = (pre_uint8 > 0) & valid
    post_mask = (post_uint8 > 0) & valid
    if not np.array_equal(pre_mask, pre_reported):
        raise RuntimeError(f"reconstructed depth_valid_pre disagrees with the helper for {stem} ({label})")
    if not np.array_equal(post_mask, post_reported):
        raise RuntimeError(f"reconstructed depth_valid_post disagrees with the helper for {stem} ({label})")

    valid_pixels = int(np.count_nonzero(valid))
    pre_and_post = valid & pre_mask & post_mask
    changed = pre_and_post & (pre_uint8 != post_uint8)
    class_masks = {
        "clean_valid": pre_and_post & ~changed,
        "natural_invalid": valid & ~pre_mask & ~post_mask,
        "synthetic_missing": valid & pre_mask & ~post_mask,
        "newly_valid": valid & ~pre_mask & post_mask,
        "implicit_quality": changed,
    }
    counts = {name: int(np.count_nonzero(class_masks[name])) for name in SUPERVISION_CLASSES}
    if sum(counts.values()) != valid_pixels:
        raise RuntimeError(f"the five-way partition does not cover the valid region for {stem} ({label})")

    helper_census = {
        "valid_pixels": int(metadata["valid_pixels"]),
        "natural_invalid_pixels": int(metadata["natural_invalid_pixels"]),
        "synthetic_missing_pixels": int(metadata["synthetic_missing_pixels"]),
        "newly_valid_pixels": int(metadata["newly_valid_pixels"]),
        "implicit_quality_pixels": int(metadata["implicit_quality_pixels"]),
        "post_corruption_invalid_pixels": int(metadata["post_corruption_invalid_pixels"]),
    }
    relations = {
        "helper_valid_pixels_equals_recomputed": helper_census["valid_pixels"] == valid_pixels,
        "helper_natural_invalid_equals_partition_natural_plus_newly_valid": (
            helper_census["natural_invalid_pixels"] == counts["natural_invalid"] + counts["newly_valid"]
        ),
        "helper_synthetic_missing_equals_partition": helper_census["synthetic_missing_pixels"] == counts["synthetic_missing"],
        "helper_newly_valid_equals_partition": helper_census["newly_valid_pixels"] == counts["newly_valid"],
        "helper_implicit_quality_equals_partition": helper_census["implicit_quality_pixels"] == counts["implicit_quality"],
        "helper_post_corruption_invalid_equals_partition_natural_plus_synthetic": (
            helper_census["post_corruption_invalid_pixels"] == counts["natural_invalid"] + counts["synthetic_missing"]
        ),
    }
    if not all(relations.values()):
        raise RuntimeError(f"helper metadata disagrees with the recomputed partition for {stem} ({label}): "
                           f"{[name for name, ok in relations.items() if not ok]}")

    supervised = target[valid]
    supervised_count = int(supervised.size)
    if supervised_count == 0:
        raise RuntimeError(f"no supervised pixel for {stem} ({label})")
    zero_map = target == 0.0
    target_zero_native = int(np.count_nonzero(valid & zero_map & ~pre_mask))
    target_zero_synthetic = int(np.count_nonzero(valid & zero_map & pre_mask))
    target_zero = target_zero_native + target_zero_synthetic
    if target_zero_native != counts["natural_invalid"] + counts["newly_valid"]:
        raise RuntimeError(
            f"not every natively invalid supervised pixel has target exactly 0 for {stem} ({label})"
        )
    if float(np.abs(target[valid & ~pre_mask]).max(initial=0.0)) != 0.0:
        raise RuntimeError(f"supervised Depth target is not exactly 0 outside native validity for {stem}")
    target_zero_by_class = {
        name: int(np.count_nonzero(valid & zero_map & class_masks[name])) for name in SUPERVISION_CLASSES
    }

    return {
        "label": label,
        "sample_index": int(index),
        "stem": stem,
        "location_group": location_group(stem),
        "clean_draw": bool(metadata["clean"]),
        "num_specs": int(metadata["num_specs"]),
        "specs": list(metadata["specs"]),
        "curriculum_progress": float(metadata["curriculum_progress"]),
        "seed_words": [int(word) for word in metadata["seed_words"]],
        "valid_pixels": valid_pixels,
        "valid_fraction_of_grid": float(metadata["valid_fraction"]),
        "depth_valid_pre_pixels": int(np.count_nonzero(pre_mask)),
        "depth_valid_post_pixels": int(np.count_nonzero(post_mask)),
        "post_corruption_invalid_pixels": helper_census["post_corruption_invalid_pixels"],
        "supervised_pixels": supervised_count,
        "class_pixel_counts": counts,
        "class_fractions_of_valid_region": {
            name: float(value) / float(valid_pixels) for name, value in counts.items()
        },
        "helper_frozen_census": helper_census,
        "helper_census_relations": relations,
        "supervised_fraction_of_valid_region": float(supervised_count) / float(valid_pixels),
        "target_zero_pixels": target_zero,
        "target_zero_fraction_of_supervised": float(target_zero) / float(supervised_count),
        "target_zero_native_pixels": target_zero_native,
        "target_zero_synthetic_pixels": target_zero_synthetic,
        "target_zero_by_partition_class": target_zero_by_class,
        "target_min": float(supervised.min()),
        "target_mean": float(supervised.mean()),
        "target_median": float(np.median(supervised)),
        "target_histogram": target_histogram(supervised),
    }


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Cross-sample mean/median for the composition and target statistics."""
    if not records:
        return {"count": 0}
    aggregate: Dict[str, Any] = {"count": len(records)}
    aggregate["pixel_counts"] = {
        "valid_pixels": int(sum(int(row["valid_pixels"]) for row in records)),
        "supervised_pixels": int(sum(int(row["supervised_pixels"]) for row in records)),
        "class_pixel_counts": {
            name: int(sum(int(row["class_pixel_counts"][name]) for row in records))
            for name in SUPERVISION_CLASSES
        },
    }
    pooled = aggregate["pixel_counts"]
    pooled_valid = max(pooled["valid_pixels"], 1)
    aggregate["pooled_class_fractions_of_valid_region"] = {
        name: float(pooled["class_pixel_counts"][name]) / float(pooled_valid) for name in SUPERVISION_CLASSES
    }
    pooled_zero_by_class = {
        name: int(sum(int(row["target_zero_by_partition_class"][name]) for row in records))
        for name in SUPERVISION_CLASSES
    }
    aggregate["pooled_target_zero_pixels_by_partition_class"] = pooled_zero_by_class
    aggregate["pooled_target_zero_fraction_by_partition_class"] = {
        name: float(pooled_zero_by_class[name]) / float(max(pooled["class_pixel_counts"][name], 1))
        for name in SUPERVISION_CLASSES
    }
    for name in SUPERVISION_CLASSES:
        fractions = [float(row["class_fractions_of_valid_region"][name]) for row in records]
        aggregate[f"per_sample_{name}_fraction"] = {
            "mean": float(np.mean(fractions)),
            "median": float(np.median(fractions)),
            "min": float(np.min(fractions)),
            "max": float(np.max(fractions)),
        }
    zero_fractions = [float(row["target_zero_fraction_of_supervised"]) for row in records]
    aggregate["per_sample_target_zero_fraction_of_supervised"] = {
        "mean": float(np.mean(zero_fractions)),
        "median": float(np.median(zero_fractions)),
        "min": float(np.min(zero_fractions)),
        "max": float(np.max(zero_fractions)),
    }
    target_values = np.asarray([float(row["target_mean"]) for row in records], dtype=np.float64)
    aggregate["per_sample_target_mean"] = {
        "mean": float(target_values.mean()),
        "median": float(np.median(target_values)),
        "min": float(target_values.min()),
        "max": float(target_values.max()),
    }
    aggregate["clean_draws"] = int(sum(1 for row in records if row["clean_draw"]))
    aggregate["corrupt_draws"] = int(sum(1 for row in records if not row["clean_draw"]))
    aggregate["pooled_target_zero_fraction_of_supervised"] = (
        float(sum(int(row["target_zero_pixels"]) for row in records)) / float(max(pooled["supervised_pixels"], 1))
    )
    aggregate["pooled_target_zero_native_fraction_of_supervised"] = (
        float(sum(int(row["target_zero_native_pixels"]) for row in records))
        / float(max(pooled["supervised_pixels"], 1))
    )
    aggregate["pooled_target_zero_synthetic_fraction_of_supervised"] = (
        float(sum(int(row["target_zero_synthetic_pixels"]) for row in records))
        / float(max(pooled["supervised_pixels"], 1))
    )
    return aggregate


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_MODULE, help="v2 train config module")
    parser.add_argument("--output", default=str(REPORT_PATH), help="JSON evidence path")
    parser.add_argument(
        "--skip-training-geometry",
        action="store_true",
        help="skip the part-2 whole-split TrainPre scan and mark it as not completed",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="print a progress line every N samples during the part-2 scan (0 disables)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    generated = dt.datetime.now(dt.timezone.utc).isoformat()

    config = load_config(args.config)
    config_file = Path(importlib.import_module(args.config).__file__).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = read_split_entries(TRAIN_SPLIT)
    stems = [Path(entry).stem for entry in entries]
    split_sha256 = file_sha256(TRAIN_SPLIT)
    if split_sha256 != EXPECTED_TRAIN_SPLIT_SHA256:
        raise RuntimeError(
            f"train-dev split SHA-256 mismatch: expected {EXPECTED_TRAIN_SPLIT_SHA256}, got {split_sha256}"
        )
    if len(stems) != EXPECTED_TRAIN_SAMPLE_COUNT:
        raise RuntimeError(f"train-dev must contain {EXPECTED_TRAIN_SAMPLE_COUNT} samples, got {len(stems)}")
    if len(set(stems)) != len(stems):
        raise RuntimeError("train-dev contains duplicate sample stems")

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": generated,
        "status": "running",
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": script_sha256(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cv2": cv2.__version__,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
            "device": "cpu",
        },
        "scope_statements": {
            "collects_facts_only": True,
            "modifies_loss": False,
            "modifies_target_definition": False,
            "modifies_supervision_semantics": False,
            "modifies_any_existing_file": False,
            "reads_official_test_split": False,
            "runs_training_or_evaluation": False,
            "uses_gpu": False,
            "note": (
                "This audit measures the frozen A2 v2 supervision distribution as-is. It does not "
                "change the loss, the target, the validity definition or any supervision semantics."
            ),
        },
        "frozen_inputs": {
            "config_module": args.config,
            "config_file": str(config_file),
            "config_sha256": file_sha256(config_file),
            "batch_helper": str(REPO_ROOT / "utils" / "dataloader" / "mmfr_training_v2.py"),
            "batch_helper_sha256": file_sha256(REPO_ROOT / "utils" / "dataloader" / "mmfr_training_v2.py"),
            "rgbx_dataset_sha256": file_sha256(REPO_ROOT / "utils" / "dataloader" / "RGBXDataset.py"),
            "train_pre_sha256": file_sha256(REPO_ROOT / "utils" / "dataloader" / "dataloader.py"),
            "location_group_rule_source": str(REPO_ROOT / "tools" / "mve" / "dvc_a1_core.py"),
            "train_split": {
                "path": str(TRAIN_SPLIT),
                "sha256": split_sha256,
                "expected_sha256": EXPECTED_TRAIN_SPLIT_SHA256,
                "sample_count": len(stems),
                "expected_sample_count": EXPECTED_TRAIN_SAMPLE_COUNT,
            },
            "data_root": str(config.dataset_path),
            "depth_dir": str(config.x_root_folder),
            "depth_format": str(config.x_format),
            "validity_definition": "depth_valid_pre = (raw_depth_uint8 > 0) AND valid_mask",
            "valid_mask_definition": "NOT(all RGB channels == normalized 0 AND all Depth channels == normalized 0)",
            "corruption": {
                "seed": int(config.mmfr_a2["corruption"]["seed"]),
                "p_clean": float(config.mmfr_a2["corruption"]["p_clean"]),
                "max_specs": int(config.mmfr_a2["corruption"]["max_specs"]),
                "modalities": list(config.mmfr_a2["corruption"]["modalities"]),
                "kinds": list(config.mmfr_a2["corruption"]["kinds"]),
            },
            "nepochs": int(config.nepochs),
            "niters_per_epoch": int(config.niters_per_epoch),
            "norm_mean": [float(value) for value in config.norm_mean],
            "norm_std": [float(value) for value in config.norm_std],
            "depth_normalization_mean": 0.48,
            "depth_normalization_std": 0.28,
            "target_composition": str(config.mmfr_a2["corruption"].get("target_composition", "")),
            "supervised_channels": list(config.mmfr_a2["reliability_head"].get("supervised_channels", [])),
            "reliability_target_depth_channel_index": int(DEPTH_RELIABILITY_INDEX),
        },
        "revealed_rules": {
            "location_group_rule": (
                "first-four-ascii-hyphen-separated-stem-segments -- imported verbatim as "
                "tools.mve.dvc_a1_core.location_group, the same rule frozen as "
                "'first-four-ascii-hyphen-separated-stem-segments' in the DVC-A1 / DVG-B1 protocols. "
                "No fallback degraded rule was needed."
            ),
            "part2_geometry_seed_policy": (
                f"random.seed({GEOMETRY_SEED}) is called exactly once, immediately before the first "
                "__getitem__ of the whole-split scan, and is never reset; TrainPre draws mirror, scale "
                "and crop position from the process-wide random module only, so the full 1277-sample "
                "geometry sequence is one deterministic, order-stable stream."
            ),
            "part4_sampling_rule": (
                f"take the {GROUP_SAMPLE_COUNT} distinct location groups at even stride over the sorted "
                "group list, and inside each group take its lowest train-dev index; the geometry is drawn "
                f"from a separate fresh random.seed({SAMPLE_GEOMETRY_SEED}) set once before reading those "
                "indices in ascending order, and that one geometry is reused for all three progress points "
                "so the only varying factor is the curriculum position."
            ),
            "part4_progress_points": [f"{label}: epoch={epoch}, iteration={iteration}" for label, epoch, iteration in PROGRESS_POINTS],
            "target_zero_definition": (
                "a supervised pixel has target exactly 0 when the frozen R_D^sup = depth_valid_pre * R_D^syn "
                "evaluates to 0, i.e. on every natively invalid pixel and on every corrupted pixel whose "
                "synthetic reliability surrogate is exactly 0"
            ),
        },
        "task_statement_notes": [
            (
                "The task statement called R_D^sup 'reliability_target[0]'. In this repository the "
                "reliability_target channel order is RGB then Depth (MODALITY_INDEX = {'rgb': 0, 'depth': 1}), "
                f"so index 0 is the all-ones RGB scaffold channel and the supervised Depth channel is index "
                f"{int(DEPTH_RELIABILITY_INDEX)}. This audit uses the supervised Depth channel, i.e. "
                f"reliability_target[:, {int(DEPTH_RELIABILITY_INDEX)}], read from "
                "mmfr_training_v2.DEPTH_RELIABILITY_INDEX."
            ),
            (
                "The raw Depth plane in this dataset is single-channel uint8 (not three-channel); the reader "
                "accepts both and uses channel 0, matching the IMREAD_GRAYSCALE read of RGBXDataset."
            ),
            (
                "The five supervision classes requested by the task are reported as a DISJOINT partition of the "
                "geometry valid region (each supervised pixel in exactly one class), so here "
                "'natural_invalid' means natively invalid AND still invalid after corruption. The frozen "
                "helper's own field of the same name counts ALL natively invalid pixels and therefore overlaps "
                "newly_valid; that original census is reported verbatim in "
                "part4_supervision_composition.<progress>.per_sample[*].helper_frozen_census and is related to "
                "the partition by helper.natural_invalid == partition.natural_invalid + partition.newly_valid."
            ),
        ],
        "warnings": [],
        "incomplete_or_unverified": [],
    }

    print(f"=== {SCHEMA} ===")
    print(f"config={args.config}  train-dev={len(stems)} samples  groups={len({location_group(s) for s in stems})}")
    print(f"cuda_visible_devices='{report['tool']['cuda_visible_devices']}'  torch.cuda.is_available={report['tool']['torch_cuda_is_available']}")
    if report["tool"]["torch_cuda_is_available"]:
        report["warnings"].append(
            "torch reports a visible CUDA device even though CUDA_VISIBLE_DEVICES was forced to -1; "
            "the audit still performs no GPU work and every helper output tensor is asserted to be on the CPU."
        )

    # -----------------------------------------------------------------------------------
    # Part 1: raw aligned grid
    # -----------------------------------------------------------------------------------
    part1_start = time.perf_counter()
    raw_fractions: List[float] = []
    raw_all_zero: List[str] = []
    shape_counts: Dict[str, int] = {}
    depth_dir = Path(str(config.x_root_folder))
    for position, stem in enumerate(stems):
        depth_path = depth_dir / f"{stem}{config.x_format}"
        stats = read_raw_depth_validity(depth_path)
        raw_fractions.append(float(stats["valid_fraction"]))
        if stats["all_zero"]:
            raw_all_zero.append(stem)
        key = f"{stats['shape'][0]}x{stats['shape'][1]}"
        shape_counts[key] = shape_counts.get(key, 0) + 1
        if args.progress_every and (position + 1) % args.progress_every == 0:
            print(f"  [part1] {position + 1}/{len(stems)} raw Depth images read")

    raw_array = np.asarray(raw_fractions, dtype=np.float64)
    report["part1_raw_aligned_grid"] = {
        "description": (
            "No geometry augmentation. Per train-dev sample, the fraction of raw uint8 Depth pixels > 0 "
            "on the original aligned grid, i.e. depth_valid_pre with an all-true valid_mask."
        ),
        "sample_count": len(raw_fractions),
        "depth_shapes": shape_counts,
        "valid_fraction": validity_summary(raw_fractions),
        "all_zero_sample_count": len(raw_all_zero),
        "all_zero_samples": raw_all_zero,
        "all_zero_count_matches_exact_zero": bool(len(raw_all_zero) == int(np.count_nonzero(raw_array == 0.0))),
        "seconds": float(time.perf_counter() - part1_start),
    }

    # -----------------------------------------------------------------------------------
    # Part 2: training input geometry (post TrainPre crop/pad)
    # -----------------------------------------------------------------------------------
    if args.skip_training_geometry:
        report["part2_training_geometry"] = {
            "status": "not_completed",
            "reason": "--skip-training-geometry was passed",
        }
        report["incomplete_or_unverified"].append(
            "part2_training_geometry was skipped by --skip-training-geometry; no training-geometry "
            "validity distribution is reported."
        )
        print("[part2] SKIPPED by --skip-training-geometry")
    else:
        part2_start = time.perf_counter()
        dataset = build_train_dataset(config)
        if len(dataset) != len(stems):
            raise RuntimeError(f"dataset length {len(dataset)} does not match train-dev ({len(stems)})")
        if [Path(entry).stem for entry in dataset._file_names] != stems:
            raise RuntimeError("dataset file order does not match the train-dev split order")

        random.seed(GEOMETRY_SEED)
        geometry_fractions: List[float] = []
        valid_fractions: List[float] = []
        geometry_zero_pre: List[str] = []
        geometry_meta: List[Dict[str, Any]] = []
        for index in range(len(stems)):
            rgb, depth, stem, entry = geometry_inputs(dataset, index)
            outputs = run_helper(config, rgb, depth, stem, epoch=1, iteration=0)
            valid = outputs["valid_mask"][0, 0].detach().cpu().numpy().astype(bool)
            pre = outputs["depth_valid_pre"][0, 0].detach().cpu().numpy().astype(bool)
            valid_pixels = int(np.count_nonzero(valid))
            pre_pixels = int(np.count_nonzero(pre))
            if valid_pixels == 0:
                raise RuntimeError(f"sample {stem} has an empty geometry valid mask")
            fraction = float(pre_pixels) / float(valid_pixels)
            geometry_fractions.append(fraction)
            valid_fractions.append(float(valid_pixels) / float(valid.size))
            if pre_pixels == 0:
                geometry_zero_pre.append(stem)
            metadata = dict(outputs["metadata"][0])
            geometry_meta.append(
                {
                    "sample_index": int(index),
                    "stem": stem,
                    "entry": entry,
                    "location_group": location_group(stem),
                    "valid_mask_fraction_of_grid": float(valid_pixels) / float(valid.size),
                    "depth_valid_pre_fraction_of_valid_mask": fraction,
                    "depth_valid_pre_fraction_of_grid": float(pre_pixels) / float(valid.size),
                    "clean_draw": bool(metadata["clean"]),
                    "specs": list(metadata["specs"]),
                }
            )
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"  [part2] {index + 1}/{len(stems)} samples through TrainPre + corruption helper")

        geometry_array = np.asarray(geometry_fractions, dtype=np.float64)
        report["part2_training_geometry"] = {
            "status": "completed",
            "description": (
                "Repository data pipeline RGBXDataset(setting, 'train', TrainPre(norm_mean, norm_std, "
                "x_is_single_channel, config)) over all train-dev samples. depth_valid_pre is read straight "
                "from the frozen build_mmfr_training_batch_v2 output, so the crop/pad valid_mask and the "
                "uint8 round trip are the production ones. Corruption does not affect depth_valid_pre."
            ),
            "geometry_seed": GEOMETRY_SEED,
            "corruption_epoch": 1,
            "corruption_iteration": 0,
            "sample_count": len(geometry_fractions),
            "depth_valid_pre_fraction_of_valid_mask": validity_summary(geometry_fractions),
            "valid_mask_fraction_of_grid": describe(valid_fractions),
            "samples_with_zero_valid_depth_in_valid_mask": len(geometry_zero_pre),
            "samples_with_zero_valid_depth_in_valid_mask_list": geometry_zero_pre,
            "scaled_grid": {
                "crop_size": [int(config.image_height), int(config.image_width)],
                "train_scale_array": [float(value) for value in config.train_scale_array],
            },
            "per_sample": geometry_meta,
            "seconds": float(time.perf_counter() - part2_start),
        }
        print(
            "  [part2] depth_valid_pre/valid_mask mean={:.6f} median={:.6f}  valid_mask/grid mean={:.6f}".format(
                float(np.mean(geometry_array)), float(np.median(geometry_array)), float(np.mean(valid_fractions))
            )
        )

    # -----------------------------------------------------------------------------------
    # Part 3: location group summary
    # -----------------------------------------------------------------------------------
    part3_start = time.perf_counter()
    group_members: Dict[str, List[int]] = {}
    for index, stem in enumerate(stems):
        group_members.setdefault(location_group(stem), []).append(index)
    group_keys = sorted(group_members)
    geometry_lookup: Dict[int, float] = {}
    valid_lookup: Dict[int, float] = {}
    part2 = report.get("part2_training_geometry") or {}
    if part2.get("status") == "completed":
        for row in part2["per_sample"]:
            geometry_lookup[int(row["sample_index"])] = float(row["depth_valid_pre_fraction_of_valid_mask"])
            valid_lookup[int(row["sample_index"])] = float(row["valid_mask_fraction_of_grid"])

    per_group: List[Dict[str, Any]] = []
    for key in group_keys:
        members = sorted(group_members[key])
        raw_values = [float(raw_fractions[index]) for index in members]
        row: Dict[str, Any] = {
            "location_group": key,
            "sample_count": len(members),
            "sample_stems": [stems[index] for index in members],
            "sample_indices": members,
            "raw_grid_valid_fraction": {
                "mean": float(np.mean(raw_values)),
                "median": float(np.median(raw_values)),
                "min": float(np.min(raw_values)),
                "max": float(np.max(raw_values)),
            },
            "raw_grid_all_zero_sample_count": int(sum(1 for value in raw_values if value == 0.0)),
        }
        if geometry_lookup:
            geometry_values = [geometry_lookup[index] for index in members]
            valid_values = [valid_lookup[index] for index in members]
            row["training_geometry_valid_fraction_of_valid_mask"] = {
                "mean": float(np.mean(geometry_values)),
                "median": float(np.median(geometry_values)),
                "min": float(np.min(geometry_values)),
                "max": float(np.max(geometry_values)),
            }
            row["training_geometry_valid_mask_fraction_of_grid"] = {
                "mean": float(np.mean(valid_values)),
                "median": float(np.median(valid_values)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
            }
        per_group.append(row)

    sizes = np.asarray([row["sample_count"] for row in per_group], dtype=np.float64)
    group_means_raw = np.asarray([row["raw_grid_valid_fraction"]["mean"] for row in per_group], dtype=np.float64)
    report["part3_location_groups"] = {
        "description": (
            "Location groups derived with the repository's frozen rule imported from "
            "tools.mve.dvc_a1_core.location_group (first four ASCII-hyphen-separated stem segments). "
            "Group statistics are computed over the group's train-dev samples."
        ),
        "group_rule": "first-four-ascii-hyphen-separated-stem-segments",
        "group_rule_source": str(REPO_ROOT / "tools" / "mve" / "dvc_a1_core.py"),
        "group_count": len(per_group),
        "sample_count": len(stems),
        "group_size": describe(sizes),
        "group_size_histogram": {str(int(size)): int(np.count_nonzero(sizes == size)) for size in np.unique(sizes)},
        "group_mean_raw_grid_valid_fraction": describe(group_means_raw),
        "groups_with_all_samples_all_zero_raw": int(sum(1 for row in per_group if row["raw_grid_all_zero_sample_count"] == row["sample_count"])),
        "per_group": per_group,
        "seconds": float(time.perf_counter() - part3_start),
    }

    # -----------------------------------------------------------------------------------
    # Part 4 / 5: supervision composition and R_D^sup at early / mid / late
    # -----------------------------------------------------------------------------------
    part45_start = time.perf_counter()
    stride_positions = [
        min(int((index + 0.5) * len(group_keys) / GROUP_SAMPLE_COUNT), len(group_keys) - 1)
        for index in range(GROUP_SAMPLE_COUNT)
    ]
    if len(set(stride_positions)) != GROUP_SAMPLE_COUNT:
        raise RuntimeError("group stride rule produced duplicate groups")
    sampled_index: List[int] = []
    sampled_group: List[str] = []
    for position in stride_positions:
        key = group_keys[position]
        sampled_index.append(min(group_members[key]))
        sampled_group.append(key)
    order = sorted(range(len(sampled_index)), key=lambda slot: sampled_index[slot])

    sample_dataset = build_train_dataset(config)
    random.seed(SAMPLE_GEOMETRY_SEED)
    loaded: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for slot in order:
        index = sampled_index[slot]
        rgb, depth, _stem, _entry = geometry_inputs(sample_dataset, index)
        loaded[index] = (rgb, depth)

    progress_records: Dict[str, List[Dict[str, Any]]] = {label: [] for label, _epoch, _iteration in PROGRESS_POINTS}
    progress_meta: Dict[str, Dict[str, Any]] = {}
    for label, epoch, iteration in PROGRESS_POINTS:
        progress = curriculum_progress(epoch, iteration, int(config.niters_per_epoch), int(config.nepochs))
        progress_meta[label] = {
            "epoch": int(epoch),
            "iteration": int(iteration),
            "curriculum_progress": float(progress),
        }
        for slot in order:
            index = sampled_index[slot]
            rgb, depth = loaded[index]
            stem = stems[index]
            outputs = run_helper(config, rgb, depth, stem, epoch=epoch, iteration=iteration)
            progress_records[label].append(supervision_record(outputs, depth, index, stem, label))

    part4: Dict[str, Any] = {
        "description": (
            "Each supervised pixel (geometry valid_mask, supervised Depth channel) is classified into exactly "
            "one of clean_valid / natural_invalid / newly_valid / synthetic_missing / implicit_quality, and the "
            "same pixels are histogrammed by their R_D^sup value. Composition facts only: no loss, target or "
            "validity semantics are changed by this audit."
        ),
        "partition_definition": {
            "clean_valid": "valid & depth_valid_pre & depth_valid_post & (pre_uint8 == post_uint8)",
            "natural_invalid": "valid & ~depth_valid_pre & ~depth_valid_post",
            "newly_valid": "valid & ~depth_valid_pre & depth_valid_post",
            "synthetic_missing": "valid & depth_valid_pre & ~depth_valid_post",
            "implicit_quality": "valid & depth_valid_pre & depth_valid_post & (pre_uint8 != post_uint8)",
            "coverage": "the five classes are disjoint and sum to valid_pixels exactly",
        },
        "helper_frozen_census_note": (
            "The frozen helper's own natural_invalid_pixels counts ALL natively invalid pixels, so it "
            "overlaps newly_valid_pixels and is NOT a class of the disjoint partition above. The two are "
            "related by helper.natural_invalid_pixels == partition.natural_invalid + partition.newly_valid. "
            "Both censuses are reported; neither is rewritten."
        ),
        "sampling_rule": report["revealed_rules"]["part4_sampling_rule"],
        "geometry_seed": SAMPLE_GEOMETRY_SEED,
        "per_group_stride_positions": stride_positions,
        "sampled_count": len(sampled_index),
        "sampled_sample_indices": [sampled_index[slot] for slot in order],
        "sampled_sample_stems": [stems[sampled_index[slot]] for slot in order],
        "sampled_location_groups": [sampled_group[slot] for slot in order],
        "progress_points": progress_meta,
        "classes": list(SUPERVISION_CLASSES),
    }
    part5: Dict[str, Any] = {
        "description": (
            f"R_D^sup distribution over the supervised pixels, using channel index "
            f"{int(DEPTH_RELIABILITY_INDEX)} of reliability_target (the supervised Depth channel)."
        ),
        "supervised_depth_channel_index": int(DEPTH_RELIABILITY_INDEX),
        "histogram_bins": ["0", "(0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]", "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]", "(0.8,0.9]", "(0.9,1.0]"],
        "progress_points": progress_meta,
    }
    for label, _epoch, _iteration in PROGRESS_POINTS:
        records = progress_records[label]
        part4[label] = aggregate_records(records)
        part4[label]["per_sample"] = [
            {
                "sample_index": row["sample_index"],
                "stem": row["stem"],
                "location_group": row["location_group"],
                "clean_draw": row["clean_draw"],
                "num_specs": row["num_specs"],
                "specs": row["specs"],
                "seed_words": row["seed_words"],
                "valid_pixels": row["valid_pixels"],
                "supervised_pixels": row["supervised_pixels"],
                "class_pixel_counts": row["class_pixel_counts"],
                "class_fractions_of_valid_region": row["class_fractions_of_valid_region"],
                "helper_frozen_census": row["helper_frozen_census"],
                "post_corruption_invalid_pixels": row["post_corruption_invalid_pixels"],
                "target_zero_pixels": row["target_zero_pixels"],
                "target_zero_native_pixels": row["target_zero_native_pixels"],
                "target_zero_synthetic_pixels": row["target_zero_synthetic_pixels"],
                "target_zero_by_partition_class": row["target_zero_by_partition_class"],
                "target_zero_fraction_of_supervised": row["target_zero_fraction_of_supervised"],
                "target_mean": row["target_mean"],
                "target_median": row["target_median"],
                "target_min": row["target_min"],
                "target_histogram": row["target_histogram"],
                "valid_mask_fraction_of_grid": row["valid_fraction_of_grid"],
            }
            for row in records
        ]
        histograms = [row["target_histogram"] for row in records]
        pooled_histogram = {
            key: int(sum(int(entry[key]) for entry in histograms)) for key in part5["histogram_bins"]
        }
        pooled_histogram["greater_than_1.0"] = int(sum(int(entry["greater_than_1.0"]) for entry in histograms))
        target_means = [float(row["target_mean"]) for row in records]
        target_medians = [float(row["target_median"]) for row in records]
        zero_fractions = [float(row["target_zero_fraction_of_supervised"]) for row in records]
        pooled_supervised = max(int(sum(int(row["supervised_pixels"]) for row in records)), 1)
        pooled_zero = int(sum(int(row["target_zero_pixels"]) for row in records))
        part5[label] = {
            "supervised_pixels": int(sum(int(row["supervised_pixels"]) for row in records)),
            "histogram_pooled_counts": pooled_histogram,
            "histogram_pooled_fractions": {
                key: float(value) / float(pooled_supervised) for key, value in pooled_histogram.items()
            },
            "exactly_zero_pixels": pooled_zero,
            "exactly_zero_fraction_of_supervised": float(pooled_zero) / float(pooled_supervised),
            "exactly_zero_native_pixels": int(sum(int(row["target_zero_native_pixels"]) for row in records)),
            "exactly_zero_synthetic_pixels": int(sum(int(row["target_zero_synthetic_pixels"]) for row in records)),
            "native_share_of_exactly_zero": float(sum(int(row["target_zero_native_pixels"]) for row in records)) / float(max(pooled_zero, 1)),
            "per_sample_target_mean": {
                "mean": float(np.mean(target_means)),
                "median": float(np.median(target_means)),
                "min": float(np.min(target_means)),
                "max": float(np.max(target_means)),
            },
            "per_sample_target_median": {
                "mean": float(np.mean(target_medians)),
                "median": float(np.median(target_medians)),
                "min": float(np.min(target_medians)),
                "max": float(np.max(target_medians)),
            },
            "per_sample_exactly_zero_fraction": {
                "mean": float(np.mean(zero_fractions)),
                "median": float(np.median(zero_fractions)),
                "min": float(np.min(zero_fractions)),
                "max": float(np.max(zero_fractions)),
            },
        }

    report["part4_supervision_composition"] = part4
    report["part5_reliability_target_distribution"] = part5

    # -----------------------------------------------------------------------------------
    # BCE domination statement
    # -----------------------------------------------------------------------------------
    bce: Dict[str, Any] = {
        "question": "Can the auxiliary continuous BCE be dominated by natively invalid pixels whose target is constantly 0?",
        "supervision_mask_definition": (
            "valid_mask * supervised-channel weight, so the supervised pixel set is exactly the geometry "
            "valid_mask plane of the Depth channel (models/builder.py reliability_auxiliary_loss)"
        ),
        "per_progress": {},
    }
    for label, _epoch, _iteration in PROGRESS_POINTS:
        entry = part5[label]
        agg = part4[label]
        denominator = float(max(entry["supervised_pixels"], 1))
        bce["per_progress"][label] = {
            "supervised_pixels": entry["supervised_pixels"],
            "target_exactly_zero_fraction": entry["exactly_zero_fraction_of_supervised"],
            "target_exactly_zero_native_fraction": float(entry["exactly_zero_native_pixels"]) / denominator,
            "target_exactly_zero_synthetic_fraction": float(entry["exactly_zero_synthetic_pixels"]) / denominator,
            "target_mean": entry["per_sample_target_mean"]["mean"],
            "target_median": entry["per_sample_target_median"]["mean"],
            "target_zero_fraction_within_partition_class": agg["pooled_target_zero_fraction_by_partition_class"],
        }
    if part2.get("status") == "completed":
        raw_zero_share = float(report["part1_raw_aligned_grid"]["all_zero_sample_count"]) / float(len(stems))
        bce["whole_split_context"] = {
            "samples_with_all_zero_depth": report["part1_raw_aligned_grid"]["all_zero_sample_count"],
            "share_of_train_dev": raw_zero_share,
            "raw_grid_mean_valid_fraction": report["part1_raw_aligned_grid"]["valid_fraction"]["mean"],
            "raw_grid_median_valid_fraction": report["part1_raw_aligned_grid"]["valid_fraction"]["median"],
        }
    statements = []
    for label, _epoch, _iteration in PROGRESS_POINTS:
        row = bce["per_progress"][label]
        statements.append(
            f"{label}: target == 0 on {100.0 * row['target_exactly_zero_fraction']:.2f}% of supervised pixels "
            f"({100.0 * row['target_exactly_zero_native_fraction']:.2f} pp from natively invalid Depth, "
            f"{100.0 * row['target_exactly_zero_synthetic_fraction']:.2f} pp from corrupted pixels whose "
            "synthetic reliability surrogate is exactly 0)"
        )
    bce["statement"] = (
        "Measured, not assumed -- " + "; ".join(statements) + ". "
        "This is a factual statement about the frozen target distribution. It does not change the loss and does "
        "not conclude that the current target is wrong. A share close to 100% would mean the continuous BCE is "
        "effectively driven by a zero / non-zero signal on this sample set, because the BCE-with-logits gradient "
        "vanishes on correctly predicted saturated targets."
    )
    report["bce_domination_statement"] = bce
    report["part45_seconds"] = float(time.perf_counter() - part45_start)

    report["status"] = "completed"
    report["runtime_seconds"] = float(time.perf_counter() - started)
    report["generated_utc_end"] = dt.datetime.now(dt.timezone.utc).isoformat()

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
    report_sha = file_sha256(output_path)
    report["report_sha256"] = report_sha

    # -----------------------------------------------------------------------------------
    # stdout summary
    # -----------------------------------------------------------------------------------
    p1 = report["part1_raw_aligned_grid"]["valid_fraction"]
    print("")
    print(f"[part1] raw aligned grid, {report['part1_raw_aligned_grid']['sample_count']} samples")
    print(
        "        mean={:.6f} median={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
            p1["mean"], p1["median"], p1["std_population_ddof0"], p1["min"], p1["max"]
        )
    )
    print("        " + " ".join(f"{k}={v:.6f}" for k, v in p1["percentiles"].items()))
    print("        " + " ".join(f"{k}={v}" for k, v in p1["below_threshold_sample_counts"].items()))
    print(f"        all-zero-depth samples={report['part1_raw_aligned_grid']['all_zero_sample_count']}")

    if part2.get("status") == "completed":
        p2 = report["part2_training_geometry"]["depth_valid_pre_fraction_of_valid_mask"]
        v2 = report["part2_training_geometry"]["valid_mask_fraction_of_grid"]
        print(f"[part2] training geometry (TrainPre), {report['part2_training_geometry']['sample_count']} samples")
        print(
            "        depth_valid_pre|valid_mask mean={:.6f} median={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
                p2["mean"], p2["median"], p2["std_population_ddof0"], p2["min"], p2["max"]
            )
        )
        print("        " + " ".join(f"{k}={v:.6f}" for k, v in p2["percentiles"].items()))
        print("        " + " ".join(f"{k}={v}" for k, v in p2["below_threshold_sample_counts"].items()))
        print(
            "        valid_mask/grid mean={:.6f} median={:.6f} min={:.6f} max={:.6f}".format(
                v2["mean"], v2["median"], v2["min"], v2["max"]
            )
        )
        print(
            "        samples with zero valid Depth inside valid_mask={}".format(
                report["part2_training_geometry"]["samples_with_zero_valid_depth_in_valid_mask"]
            )
        )
    else:
        print("[part2] NOT COMPLETED (--skip-training-geometry)")

    p3 = report["part3_location_groups"]
    print(f"[part3] location groups={p3['group_count']} rule={p3['group_rule']}")
    print(
        "        group size min={:.0f} median={:.0f} max={:.0f}".format(
            p3["group_size"]["min"], p3["group_size"]["median"], p3["group_size"]["max"]
        )
    )
    gm = p3["group_mean_raw_grid_valid_fraction"]
    print(
        "        group-mean raw valid fraction: mean={:.6f} median={:.6f} min={:.6f} max={:.6f}".format(
            gm["mean"], gm["median"], gm["min"], gm["max"]
        )
    )

    for label, _epoch, _iteration in PROGRESS_POINTS:
        agg = part4[label]
        p5 = part5[label]
        print(
            f"[part4] {label}: progress={progress_meta[label]['curriculum_progress']:.6f} "
            f"clean={agg['clean_draws']} corrupt={agg['corrupt_draws']} supervised={agg['pixel_counts']['supervised_pixels']}"
        )
        print(
            "        pooled class shares of valid region: "
            + " ".join(f"{name}={agg['pooled_class_fractions_of_valid_region'][name]:.6f}" for name in SUPERVISION_CLASSES)
        )
        print(
            "        per-sample class share mean: "
            + " ".join(f"{name}={agg[f'per_sample_{name}_fraction']['mean']:.6f}" for name in SUPERVISION_CLASSES)
        )
        print(
            "        per-sample class share median: "
            + " ".join(f"{name}={agg[f'per_sample_{name}_fraction']['median']:.6f}" for name in SUPERVISION_CLASSES)
        )
        print(
            f"[part5] {label}: R_D^sup exactly0={p5['exactly_zero_fraction_of_supervised']:.6f} "
            f"(native={float(p5['exactly_zero_native_pixels']) / max(p5['supervised_pixels'], 1):.6f}, "
            f"synthetic={float(p5['exactly_zero_synthetic_pixels']) / max(p5['supervised_pixels'], 1):.6f}) "
            f"mean={p5['per_sample_target_mean']['mean']:.6f} median={p5['per_sample_target_median']['mean']:.6f}"
        )
        print("        histogram shares: " + " ".join(f"{k}={v:.6f}" for k, v in p5["histogram_pooled_fractions"].items()))

    print("")
    print(f"JSON: {output_path}")
    print(f"JSON sha256: {report_sha}")
    print(f"script sha256: {report['tool']['sha256']}")
    print(f"runtime seconds: {report['runtime_seconds']:.3f}")
    print("status: completed; facts only, no loss/target/semantics change")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # write a partial report so a failure still leaves evidence
        failure_path = Path(parse_args().output)
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        partial = {
            "schema": SCHEMA,
            "status": "failed",
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tool": {"path": str(Path(__file__).resolve()), "sha256": script_sha256()},
            "error": f"{type(error).__name__}: {error}",
        }
        with open(failure_path, "w", encoding="utf-8") as handle:
            json.dump(partial, handle, indent=2)
        raise
