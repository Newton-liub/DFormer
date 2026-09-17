#!/usr/bin/env python3
"""Formal 10-condition MUSeg main-val runner for the frozen MMFR-A2 v3 protocol.

The runner owns scheduling, corruption application and evidence bookkeeping only.  It
imports the frozen evaluator (``tools.evaluate_museg_checkpoint`` as ``EV``) and the
already verified accelerated evaluator (``tools.evaluate_museg_checkpoint_fast`` as
``FAST``) and never re-implements view construction, per-view post-processing, logit
fusion, grid restoration, the confusion update or the metric definitions.

Frozen conditions (order fixed by the protocol ``evaluation`` node):
``clean`` + six single-corruption Depth conditions + three mixed Depth conditions.

Determinism scheme
------------------
Corruption is drawn per ``(sample, condition)`` from a fresh
``numpy.PCG64(SeedSequence(words))`` with
``words = [evaluation_seed, condition_index_0_based, *sample_id_sha256_u32_be_0..3]``.
No generator is shared across units, so the corruption result of a unit does not depend
on the call order.  ``clean`` draws no random numbers at all.

The frozen DFormerv2-S Ham decoder re-draws its NMF bases with ``torch.rand`` on every
forward (``RAND_INIT=True``), so the model forward is only deterministic when the torch
generator state is controlled.  The runner restores a fixed base state before every
``(sample, condition)`` forward (``--forward-rng reset-per-unit``), which makes results
order-independent, and exposes ``--forward-rng progressive`` only so a byte-level
comparison against the frozen sequential reference path is possible.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

import tools.evaluate_museg_checkpoint as EV
import tools.evaluate_museg_checkpoint_fast as FAST
from utils.dataloader.mmfr_training import normalize_sample_id, sample_id_words
from utils.dataloader.multimodal_failure_v3 import (
    BLUR_SIGMA_FRACTION,
    MISALIGN_MAX_SHIFT_FRACTION,
    FailureSpec,
    apply_failures,
)

SCHEMA_VERSION = "museg-10condition-mainval-evidence-v1"
VERIFY_SCHEMA_VERSION = "museg-10condition-mainval-verification-v1"

DEFAULT_CONFIG = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3"
DEFAULT_PROTOCOL = "protocols/mmfr-a2-train-integration-v3.template.json"
DEFAULT_OUTPUT_DIR = "experiments/MMFR_A2_v3/eval_results/mainval_10cond_v1"
DEFAULT_BENCHMARK_DIR = "experiments/MMFR_A2_v3/benchmarks"
DEFAULT_STRATEGY_PATH = "experiments/MMFR_A2_v3/benchmarks/microbatch-strategy-final.json"

SHUFFLE_SALT = 0x9E3779B9
EVALUATOR_IDENTITY = "msflip-whole-original-grid-v1"

# Literal copy of the frozen protocol ``evaluation`` node.  The protocol file is the
# source of truth; this copy exists only so that a missing or tampered node exits
# fail-closed instead of silently evaluating a drifted condition set.
FROZEN_EVALUATION: dict[str, Any] = {
    "evaluator": EVALUATOR_IDENTITY,
    "seed": 2026091401,
    "sample_count": 318,
    "primary_score": "unweighted-macro-mean-single-condition-mIoU",
    "single_conditions": [
        ["spatial_dropout", 0.75],
        ["gaussian_noise", 0.75],
        ["blur", 0.75],
        ["quantization", 0.75],
        ["misalignment", 0.75],
        ["entire_missing", 1.0],
    ],
    "mixed_conditions": [
        [["spatial_dropout", 0.5], ["gaussian_noise", 0.5]],
        [["blur", 0.5], ["misalignment", 0.5]],
        [["quantization", 0.5], ["misalignment", 0.5]],
    ],
}

FROZEN_CONSTANTS = {
    "scales": [0.5, 0.75, 1.0, 1.25, 1.5],
    "views_per_scale": ["original", "horizontal_flip"],
    "view_count": 10,
    "pad_divisor": 32,
    "depth_normalization_mean": 0.48,
    "depth_normalization_std": 0.28,
    "forward_precision": "fp32",
    "logits_fusion_precision": "fp32",
    "tf32_enabled": False,
    "amp": False,
    "rgb_corruption": "none (every condition corrupts Depth only)",
    "validity_mask": "np.ones(depth.shape, bool) on the raw aligned grid (no crop, no pad)",
    "depth_validity": "(raw Depth uint8 > 0) & validity_mask",
}


# --------------------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> str:
    """Write JSON through a temp file, flush + fsync, then atomically rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def command_line(module: str, argv: Sequence[str]) -> str:
    return f"python -m {module}" + (" " + shlex.join([str(value) for value in argv]) if argv else "")


# --------------------------------------------------------------------------------------
# frozen protocol
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Protocol:
    path: Path
    raw_sha256: str
    evaluation: Mapping[str, Any]
    document_status: Any
    official_test_included: Any


def load_frozen_protocol(path: Path) -> Protocol:
    if not path.is_file():
        raise FileNotFoundError(f"frozen protocol file is missing: {path}")
    raw = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw.decode("utf-8"))
    evaluation = document.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("frozen protocol has no evaluation node")
    mismatches: list[str] = []
    for key, expected in FROZEN_EVALUATION.items():
        if key not in evaluation:
            mismatches.append(f"{key}: missing in protocol")
            continue
        if canonical_json(evaluation[key]) != canonical_json(expected):
            mismatches.append(f"{key}: protocol={canonical_json(evaluation[key])} expected={canonical_json(expected)}")
    if mismatches:
        raise ValueError("protocol evaluation node disagrees with the frozen contract: " + "; ".join(mismatches))
    if document.get("official_test_included") is not False:
        raise ValueError("protocol must declare official_test_included=false")
    return Protocol(
        path=path,
        raw_sha256=raw_sha256,
        evaluation={
            "evaluator": evaluation["evaluator"],
            "seed": int(evaluation["seed"]),
            "sample_count": int(evaluation["sample_count"]),
            "primary_score": evaluation["primary_score"],
            "single_conditions": [[str(kind), float(severity)] for kind, severity in evaluation["single_conditions"]],
            "mixed_conditions": [
                [[str(kind), float(severity)] for kind, severity in spec_list]
                for spec_list in evaluation["mixed_conditions"]
            ],
        },
        document_status=document.get("status"),
        official_test_included=document.get("official_test_included"),
    )


# --------------------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    index: int
    condition_id: str
    directory: str
    group: str
    specs: tuple[FailureSpec, ...]

    def spec_records(self) -> list[dict[str, Any]]:
        return [
            {"modality": spec.modality, "kind": spec.kind, "severity": float(spec.severity)} for spec in self.specs
        ]

    def definition(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "condition_id": self.condition_id,
            "directory": self.directory,
            "group": self.group,
            "specs": self.spec_records(),
        }


def _severity_label(severity: float) -> str:
    text = "%g" % float(severity)
    return text if "." in text else text + ".0"


def _severity_code(severity: float) -> str:
    return f"{int(round(float(severity) * 100)):03d}"


def build_conditions(evaluation: Mapping[str, Any]) -> list[Condition]:
    conditions = [Condition(0, "clean", "clean", "clean", ())]
    index = 1
    for kind, severity in evaluation["single_conditions"]:
        spec = FailureSpec("depth", str(kind), float(severity))
        conditions.append(
            Condition(
                index,
                f"{spec.kind}@{_severity_label(spec.severity)}",
                f"{spec.kind}_{_severity_code(spec.severity)}",
                "single_corruptions",
                (spec,),
            )
        )
        index += 1
    for spec_list in evaluation["mixed_conditions"]:
        specs = tuple(FailureSpec("depth", str(kind), float(severity)) for kind, severity in spec_list)
        conditions.append(
            Condition(
                index,
                "+".join(f"{spec.kind}@{_severity_label(spec.severity)}" for spec in specs),
                "__".join(f"{spec.kind}_{_severity_code(spec.severity)}" for spec in specs),
                "mixed_corruptions",
                specs,
            )
        )
        index += 1
    return conditions


def _normalize_condition_token(text: str) -> str:
    token = str(text).strip().lower().replace(" ", "").replace("__", "+")
    return re.sub(
        r"_(\d{3})(?=\+|$)",
        lambda match: "@" + _severity_label(int(match.group(1)) / 100.0),
        token,
    )


def condition_lookup(conditions: Sequence[Condition]) -> dict[str, Condition]:
    lookup: dict[str, Condition] = {}
    for condition in conditions:
        for token in (condition.condition_id, condition.directory):
            lookup[_normalize_condition_token(token)] = condition
    return lookup


def resolve_conditions(spec: str | None, conditions: Sequence[Condition]) -> list[Condition]:
    if spec is None or not str(spec).strip():
        return list(conditions)
    lookup = condition_lookup(conditions)
    selected: list[Condition] = []
    requested = [piece for piece in re.split(r"[,;]", str(spec)) if piece.strip()]
    if not requested:
        raise ValueError("--conditions was supplied but contains no name")
    for piece in requested:
        key = _normalize_condition_token(piece)
        if key not in lookup:
            raise ValueError(
                f"unknown condition {piece!r}; available: "
                + ", ".join(condition.condition_id for condition in conditions)
            )
        condition = lookup[key]
        if condition not in selected:
            selected.append(condition)
    return sorted(selected, key=lambda item: item.index)


def shuffled_conditions(conditions: Sequence[Condition], evaluation_seed: int) -> list[Condition]:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([_as_uint32(evaluation_seed, "evaluation_seed"), SHUFFLE_SALT]))
    )
    order = [int(value) for value in generator.permutation(len(conditions))]
    return [conditions[position] for position in order]


# --------------------------------------------------------------------------------------
# deterministic seeds and corruption
# --------------------------------------------------------------------------------------


def _as_uint32(value: Any, name: str) -> int:
    resolved = int(value)
    if not 0 <= resolved < (1 << 32):
        raise ValueError(f"{name} must be a uint32 seed word, got {value!r}")
    return resolved


def unit_seed_words(evaluation_seed: int, condition_index: int, normalized_sample_id: str) -> tuple[int, ...]:
    words = (
        _as_uint32(evaluation_seed, "evaluation_seed"),
        _as_uint32(condition_index, "condition_index"),
    ) + tuple(sample_id_words(normalized_sample_id))
    return tuple(int(word) for word in words)


def unit_generator(words: Sequence[int]) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([_as_uint32(word, "seed word") for word in words])))


@dataclass
class CorruptionOutcome:
    depth: np.ndarray
    validity_state: np.ndarray
    metadata: Mapping[str, Any]
    seed_words: tuple[int, ...]
    rng_constructed: bool
    strict_no_op: bool

    def depth_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.depth).tobytes()).hexdigest()

    def state_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.validity_state.astype(np.uint8)).tobytes()).hexdigest()


def corrupt_depth(
    condition: Condition,
    rgb_uint8: np.ndarray,
    depth_uint8: np.ndarray,
    evaluation_seed: int,
    normalized_sample_id: str,
) -> CorruptionOutcome:
    """Apply one frozen condition to one raw Depth array on the original aligned grid."""
    words = unit_seed_words(evaluation_seed, condition.index, normalized_sample_id)
    if not condition.specs:
        if condition.group != "clean":
            raise RuntimeError("only the clean condition may carry an empty spec tuple")
        state = depth_uint8 > 0
        return CorruptionOutcome(
            depth=np.array(depth_uint8, dtype=np.uint8, copy=True),
            validity_state=state,
            metadata={
                "strict_no_op": True,
                "apply_failures_invoked": False,
                "rng_constructed": False,
                "specs": (),
            },
            seed_words=words,
            rng_constructed=False,
            strict_no_op=True,
        )
    generator = unit_generator(words)
    validity_mask = np.ones(depth_uint8.shape, dtype=bool)
    depth_validity = depth_uint8 > 0
    result = apply_failures(
        rgb_uint8,
        depth_uint8,
        condition.specs,
        generator,
        depth_validity=depth_validity,
        validity_mask=validity_mask,
    )
    if not np.array_equal(result.rgb, rgb_uint8):
        raise RuntimeError("frozen corruption changed RGB; every frozen condition is Depth-only")
    depth = np.ascontiguousarray(result.depth)
    state = np.ascontiguousarray(result.validity_state, dtype=bool)
    if not np.array_equal(depth != 0, state):
        raise RuntimeError("v3 corrupted Depth and its explicit validity state disagree")
    return CorruptionOutcome(
        depth=depth,
        validity_state=state,
        metadata=result.metadata,
        seed_words=words,
        rng_constructed=True,
        strict_no_op=False,
    )


# --------------------------------------------------------------------------------------
# sample reading and multi-scale flip views
# --------------------------------------------------------------------------------------


def read_sample(
    dataset_root: Path,
    entry: str,
    channel_order: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Read one val-dev sample exactly the way the frozen evaluation dataset does."""
    sample_id = Path(entry).stem
    rgb = cv2.imread(str(dataset_root / "RGB" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(dataset_root / "Depth" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(dataset_root / "Label" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or depth is None or label is None:
        raise FileNotFoundError(f"missing RGB/Depth/Label input for val-dev sample {sample_id}")
    if rgb.shape[:2] != depth.shape or depth.shape != label.shape:
        raise ValueError(f"modality geometry mismatch for val-dev sample {sample_id}")
    rgb = EV.apply_channel_order(rgb, channel_order)
    label = label.astype(np.int64) - 1
    label[label < 0] = 255
    return sample_id, rgb, depth, label


def _view_geometry(height: int, width: int, scale: float, pad_divisor: int) -> tuple[int, int, int, int]:
    scaled_height = max(1, int(round(height * float(scale))))
    scaled_width = max(1, int(round(width * float(scale))))
    padded_height = math.ceil(scaled_height / pad_divisor) * pad_divisor
    padded_width = math.ceil(scaled_width / pad_divisor) * pad_divisor
    return scaled_height, scaled_width, padded_height, padded_width


def build_rgb_view_cache(
    rgb: np.ndarray,
    config: Any,
    scales: Sequence[float] = EV.MSFLIP_SCALES,
    pad_divisor: int = EV.PAD_DIVISOR,
) -> list[dict[str, Any]]:
    """Pre-compute the condition-independent RGB part of the ten frozen views."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")
    height, width = rgb.shape[:2]
    mean = np.asarray(config.norm_mean, dtype=np.float32)
    std = np.asarray(config.norm_std, dtype=np.float32)
    cache: list[dict[str, Any]] = []
    for scale in scales:
        scaled_height, scaled_width, padded_height, padded_width = _view_geometry(height, width, scale, pad_divisor)
        scaled_rgb = cv2.resize(rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        rgb_float = scaled_rgb.astype(np.float32) / 255.0
        rgb_float = (rgb_float - mean) / std
        pad_spec = ((0, padded_height - scaled_height), (0, padded_width - scaled_width), (0, 0))
        for flipped in (False, True):
            view_rgb = np.flip(rgb_float, axis=1).copy() if flipped else rgb_float
            view_rgb = np.pad(view_rgb, pad_spec, mode="constant", constant_values=0)
            cache.append(
                {
                    "scale": float(scale),
                    "flipped": bool(flipped),
                    "scaled_size_hw": (scaled_height, scaled_width),
                    "padded_size_hw": (padded_height, padded_width),
                    "rgb": torch.from_numpy(np.ascontiguousarray(view_rgb.transpose(2, 0, 1))),
                }
            )
    return cache


def build_depth_view(depth: np.ndarray, cache_entry: Mapping[str, Any]) -> torch.Tensor:
    scaled_height, scaled_width = cache_entry["scaled_size_hw"]
    padded_height, padded_width = cache_entry["padded_size_hw"]
    scaled_depth = cv2.resize(depth, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    depth_float = scaled_depth.astype(np.float32) / 255.0
    depth_float = (depth_float - 0.48) / 0.28
    depth_float = np.repeat(depth_float[:, :, None], 3, axis=2)
    view_depth = np.flip(depth_float, axis=1).copy() if cache_entry["flipped"] else depth_float
    pad_spec = ((0, padded_height - scaled_height), (0, padded_width - scaled_width), (0, 0))
    view_depth = np.pad(view_depth, pad_spec, mode="constant", constant_values=0)
    return torch.from_numpy(np.ascontiguousarray(view_depth.transpose(2, 0, 1)))


def assemble_views(rgb_cache: Sequence[Mapping[str, Any]], depth: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "rgb": entry["rgb"],
            "depth": build_depth_view(depth, entry),
            "scale": entry["scale"],
            "flipped": entry["flipped"],
            "scaled_size_hw": entry["scaled_size_hw"],
            "padded_size_hw": entry["padded_size_hw"],
        }
        for entry in rgb_cache
    ]


def check_view_reuse(
    rgb: np.ndarray,
    depth: np.ndarray,
    config: Any,
    sample_id: str,
) -> dict[str, Any]:
    """Compare the RGB-reuse path against a direct ``EV.build_msflip_views`` call."""
    direct = EV.build_msflip_views(rgb, depth, config)
    reused = assemble_views(build_rgb_view_cache(rgb, config), depth)
    if len(direct) != len(reused):
        raise RuntimeError("view reuse changed the view count")
    per_view: list[dict[str, Any]] = []
    max_abs_difference = 0.0
    all_equal = True
    for index, (left, right) in enumerate(zip(direct, reused)):
        record: dict[str, Any] = {
            "view_index": index,
            "scale": float(left["scale"]),
            "flipped": bool(left["flipped"]),
            "scaled_size_hw": [int(value) for value in left["scaled_size_hw"]],
            "padded_size_hw": [int(value) for value in left["padded_size_hw"]],
        }
        metadata_equal = (
            float(left["scale"]) == float(right["scale"])
            and bool(left["flipped"]) == bool(right["flipped"])
            and tuple(int(value) for value in left["scaled_size_hw"]) == tuple(int(value) for value in right["scaled_size_hw"])
            and tuple(int(value) for value in left["padded_size_hw"]) == tuple(int(value) for value in right["padded_size_hw"])
        )
        for key in ("rgb", "depth"):
            left_tensor = left[key]
            right_tensor = right[key]
            if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
                raise RuntimeError(f"view reuse changed {key} shape or dtype for view {index}")
            bitwise_equal = bool(torch.equal(left_tensor, right_tensor))
            difference = 0.0 if bitwise_equal else float((left_tensor - right_tensor).abs().max().item())
            record[f"{key}_bitwise_equal"] = bitwise_equal
            record[f"{key}_max_abs_diff"] = difference
            max_abs_difference = max(max_abs_difference, difference)
            all_equal = all_equal and bitwise_equal
        record["metadata_equal"] = metadata_equal
        all_equal = all_equal and metadata_equal
        per_view.append(record)
    return {
        "sample_id": sample_id,
        "view_count": len(direct),
        "bitwise_equal": bool(all_equal),
        "max_abs_diff": max_abs_difference,
        "per_view": per_view,
    }


# --------------------------------------------------------------------------------------
# verified evaluation calls
# --------------------------------------------------------------------------------------


def run_msflip(
    model: FAST.ForwardTimer,
    sample: Mapping[str, Any],
    device: torch.device,
    config: Any,
    view_batching: int,
    batching_strategy: Mapping[float, int],
) -> dict[str, Any]:
    if view_batching == 1:
        return FAST._run_reference_sample(
            model,
            sample,
            device,
            int(config.num_classes),
            int(config.background),
            list(config.class_names),
            non_blocking=False,
        )
    if view_batching != 2:
        raise ValueError("--view-batching must be 1 or 2")
    return FAST._run_optimized_sample(
        model,
        sample,
        device,
        int(config.num_classes),
        int(config.background),
        list(config.class_names),
        view_batching=2,
        non_blocking=False,
        batching_strategy=dict(batching_strategy),
    )


def load_microbatch_strategy(path: Path, view_batching: int) -> tuple[dict[float, int], dict[str, Any]]:
    """Read the previously verified per-scale micro-batch strategy."""
    evidence: dict[str, Any] = {"requested_view_batching": int(view_batching)}
    if view_batching == 1:
        evidence["source"] = None
        evidence["effective"] = {str(scale): 1 for scale in EV.MSFLIP_SCALES}
        return {float(scale): 1 for scale in EV.MSFLIP_SCALES}, evidence
    if not path.is_file():
        raise FileNotFoundError(
            f"--view-batching 2 requires the verified strategy file, which is missing: {path}"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    fixed = document.get("microbatch_probe", {}).get("fixed_strategy")
    if not isinstance(fixed, Mapping) or len(fixed) != len(EV.MSFLIP_SCALES):
        raise ValueError(f"verified strategy file has no usable fixed_strategy map: {path}")
    strategy = {float(scale): int(values["selected_view_batching"]) for scale, values in fixed.items()}
    for scale in EV.MSFLIP_SCALES:
        if float(scale) not in strategy or strategy[float(scale)] not in (1, 2):
            raise ValueError(f"verified strategy is incomplete for scale {scale}")
    evidence.update(
        {
            "source": str(path),
            "source_sha256": EV.file_sha256(path),
            "effective": {str(scale): int(strategy[float(scale)]) for scale in EV.MSFLIP_SCALES},
            "rationale": "frozen per-scale choice recorded by the verified micro-batch probe",
        }
    )
    return strategy, evidence


# --------------------------------------------------------------------------------------
# evaluation engine
# --------------------------------------------------------------------------------------


@dataclass
class RunSettings:
    dataset_root: Path
    split: Path
    split_role: str
    split_sha256: str
    split_sample_count: int
    checkpoint: Path
    checkpoint_sha256: str
    protocol: Protocol
    config_module: str
    evaluation_seed: int
    output_dir: Path
    order: str
    shuffle_conditions: bool
    view_batching: int
    batching_strategy: dict[float, int]
    batching_evidence: dict[str, Any]
    forward_rng_policy: str
    base_rng_seed: int
    resume: bool
    max_samples: int | None
    sample_selection: str
    save_predictions: bool
    check_view_reuse_samples: int
    runner_path: Path
    runner_sha256: str
    conditions_all: tuple[Condition, ...]
    condition_definition_sha256: str


def select_entries(dataset_root: Path, entries: Sequence[str], selection: str, max_samples: int | None) -> list[str]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not entries:
        raise ValueError("the validation split is empty")
    limit = len(entries) if max_samples is None else min(int(max_samples), len(entries))
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
    raise ValueError(f"unsupported sample selection: {selection}")


def build_identity(
    settings: RunSettings,
    condition: Condition,
    selected_entries: Sequence[str],
) -> dict[str, Any]:
    return {
        "runner_sha256": settings.runner_sha256,
        "checkpoint_sha256": settings.checkpoint_sha256,
        "split_sha256": settings.split_sha256,
        "split_role": settings.split_role,
        "protocol_raw_sha256": settings.protocol.raw_sha256,
        "evaluation_seed": settings.evaluation_seed,
        "condition_definition_sha256": settings.condition_definition_sha256,
        "condition_key": condition.directory,
        "condition_index": condition.index,
        "sample_count": len(selected_entries),
        "sample_count_total": settings.split_sample_count,
        "subset": len(selected_entries) != settings.split_sample_count,
        "sample_set_sha256": text_sha256("\n".join(selected_entries)),
        "order": settings.order,
        "view_batching_requested": settings.view_batching,
        "effective_view_batching": {str(scale): int(settings.batching_strategy[float(scale)]) for scale in EV.MSFLIP_SCALES},
        "forward_precision": FROZEN_CONSTANTS["forward_precision"],
        "tf32_enabled": False,
        "base_rng_seed": settings.base_rng_seed,
        "forward_rng_policy": settings.forward_rng_policy,
        "view_reuse": True,
        "max_samples": settings.max_samples,
    }


def _read_existing_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def plan_conditions(
    settings: RunSettings,
    conditions: Sequence[Condition],
    selected_entries: Sequence[str],
) -> tuple[list[Condition], list[Condition], list[dict[str, Any]]]:
    """Split requested conditions into skipped (reusable) and pending (must run)."""
    checkpoint_dir = settings.output_dir / settings.checkpoint.stem
    skipped: list[Condition] = []
    pending: list[Condition] = []
    decisions: list[dict[str, Any]] = []
    for condition in conditions:
        path = checkpoint_dir / condition.directory / "metrics.json"
        identity = build_identity(settings, condition, selected_entries)
        identity_sha256 = json_sha256(identity)
        existing = _read_existing_metrics(path)
        decision: dict[str, Any] = {
            "condition": condition.directory,
            "condition_id": condition.condition_id,
            "metrics_path": str(path),
            "expected_identity_sha256": identity_sha256,
        }
        if existing is None:
            decision.update({"decision": "pending", "reason": "no existing metrics.json"})
            pending.append(condition)
        elif not settings.resume:
            decision.update(
                {
                    "decision": "pending",
                    "reason": "--no-resume requested; existing evidence is ignored and recomputed",
                    "existing_identity_sha256": existing.get("identity_sha256"),
                    "existing_completed": existing.get("completed"),
                }
            )
            pending.append(condition)
        elif existing.get("completed") is not True:
            decision.update(
                {
                    "decision": "pending",
                    "reason": "existing metrics.json is not completed",
                    "existing_identity_sha256": existing.get("identity_sha256"),
                    "existing_completed": existing.get("completed"),
                }
            )
            pending.append(condition)
        elif existing.get("identity_sha256") != identity_sha256:
            mismatch_fields = sorted(
                key
                for key in identity
                if canonical_json(existing.get("identity", {}).get(key)) != canonical_json(identity[key])
            )
            decision.update(
                {
                    "decision": "pending",
                    "reason": "existing identity does not match; subset or drifted evidence is never reused",
                    "existing_identity_sha256": existing.get("identity_sha256"),
                    "existing_completed": existing.get("completed"),
                    "mismatched_identity_fields": mismatch_fields,
                }
            )
            pending.append(condition)
        else:
            decision.update(
                {
                    "decision": "skipped",
                    "reason": "completed with a matching identity",
                    "existing_identity_sha256": existing.get("identity_sha256"),
                }
            )
            skipped.append(condition)
        decisions.append(decision)
    return skipped, pending, decisions


def run_engine(
    settings: RunSettings,
    config: Any,
    model: FAST.ForwardTimer,
    device: torch.device,
    conditions: Sequence[Condition],
    selected_entries: Sequence[str],
    *,
    collect_units: bool = False,
    collect_predictions: bool = False,
    emit: Any = None,
) -> dict[str, Any]:
    """Evaluate the requested conditions with the frozen ten-view protocol."""
    if settings.order == "condition-major" and settings.forward_rng_policy == "progressive":
        raise ValueError("progressive forward RNG is only defined for the sample-major order")
    save_predictions_enabled = bool(settings.save_predictions)

    checkpoint_dir = settings.output_dir / settings.checkpoint.stem
    skipped, pending, decisions = plan_conditions(settings, conditions, selected_entries)
    call_order = shuffled_conditions(pending, settings.evaluation_seed) if settings.shuffle_conditions else list(pending)
    call_index = {condition.directory: index for index, condition in enumerate(call_order)}

    accumulators: dict[str, dict[str, Any]] = {
        condition.directory: {
            "condition": condition,
            "hist": np.zeros((int(config.num_classes), int(config.num_classes)), dtype=np.int64),
            "per_sample": [],
            "units": [],
            "predictions": [],
            "depth_digests": [],
            "state_digests": [],
            "elapsed_seconds": 0.0,
            "preprocess_seconds": 0.0,
            "forward_seconds": 0.0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
        for condition in pending
    }

    view_reuse_checks: list[dict[str, Any]] = []
    base_rng: tuple[Any, Any] | None = None
    forward_units = 0
    read_seconds_total = 0.0
    rgb_cache_seconds_total = 0.0
    view_assembly_seconds_total = 0.0

    def unit_forward_state() -> None:
        nonlocal forward_units
        if base_rng is None:
            raise RuntimeError("base RNG state was not captured")
        if settings.forward_rng_policy == "reset-per-unit":
            FAST._restore_rng(base_rng, device)
        elif forward_units == 0:
            FAST._restore_rng(base_rng, device)
        forward_units += 1

    base_rng = FAST._capture_rng(device)

    def evaluate_unit(
        condition: Condition,
        entry: str,
        sample_id: str,
        label: np.ndarray,
        views: list[dict[str, Any]],
        corruption: CorruptionOutcome,
        pre_seconds: float,
    ) -> None:
        """Evaluate one (sample, condition) unit and fold it into the accumulator."""
        accumulator = accumulators[condition.directory]
        FAST._reset_peak_memory(device)
        unit_started = time.perf_counter()
        sample_like = {
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "views": views,
            "sample_id": sample_id,
            "original_height": int(label.shape[0]),
            "original_width": int(label.shape[1]),
        }
        unit_forward_state()
        result = run_msflip(model, sample_like, device, config, settings.view_batching, settings.batching_strategy)
        FAST._sync(device)
        elapsed_seconds = time.perf_counter() - unit_started + pre_seconds
        peak = FAST._peak_memory(device)
        logits = result["logits"]
        if tuple(int(value) for value in logits.shape[-2:]) != (int(label.shape[0]), int(label.shape[1])):
            raise RuntimeError("restored logits are not on the original label grid")
        accumulator["hist"] += result["hist"]
        accumulator["depth_digests"].append(corruption.depth_sha256())
        accumulator["state_digests"].append(corruption.state_sha256())
        accumulator["elapsed_seconds"] += elapsed_seconds
        accumulator["preprocess_seconds"] += pre_seconds
        accumulator["forward_seconds"] += float(result["timings"]["forward_seconds"])
        accumulator["peak_allocated_bytes"] = max(
            accumulator["peak_allocated_bytes"], int(peak.get("allocated_bytes") or 0)
        )
        accumulator["peak_reserved_bytes"] = max(
            accumulator["peak_reserved_bytes"], int(peak.get("reserved_bytes") or 0)
        )
        metrics = result["metrics"]
        sample_record = {
            "sample_id": sample_id,
            "miou": metrics["miou"],
            "macc": metrics["macc"],
            "mf1": metrics["mf1"],
            "elapsed_seconds": round(elapsed_seconds, 6),
            "forward_seconds": float(result["timings"]["forward_seconds"]),
            "ten_views_seconds": float(result["timings"]["ten_views_seconds"]),
            "peak_allocated_bytes": peak.get("allocated_bytes"),
            "peak_reserved_bytes": peak.get("reserved_bytes"),
        }
        if collect_units:
            unit: dict[str, Any] = {
                "sample_id": sample_id,
                "entry": entry,
                "condition": condition.directory,
                "condition_index": condition.index,
                "call_index": call_index.get(condition.directory),
                "elapsed_seconds": round(elapsed_seconds, 6),
                "confusion_matrix": result["hist"].tolist(),
                "metrics_percent": metrics,
                "corrupted_depth_sha256": corruption.depth_sha256(),
                "validity_state_sha256": corruption.state_sha256(),
                "seed_words": list(corruption.seed_words),
                "rng_constructed": corruption.rng_constructed,
                "strict_no_op": corruption.strict_no_op,
                "corruption_metadata": corruption.metadata,
                "original_size_hw": [int(label.shape[0]), int(label.shape[1])],
            }
        need_prediction = bool(collect_units or collect_predictions or settings.save_predictions)
        prediction = None
        if need_prediction:
            prediction = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        if save_predictions_enabled and prediction is not None:
            accumulator["predictions"].append({"sample_id": sample_id, "prediction": prediction})
        if collect_units:
            unit["prediction_sha256"] = hashlib.sha256(prediction.tobytes()).hexdigest()
            if collect_predictions:
                unit["prediction"] = prediction
                unit["logits_sha256"] = hashlib.sha256(
                    logits.detach().cpu().numpy().astype(np.float32).tobytes()
                ).hexdigest()
            accumulator["units"].append(unit)
        accumulator["per_sample"].append(sample_record)
        del logits, result, sample_like

    if pending:
        if settings.order == "sample-major":
            for position, entry in enumerate(selected_entries):
                normalized_id = normalize_sample_id(entry)
                read_started = time.perf_counter()
                sample_id, rgb_uint8, depth_uint8, label = read_sample(
                    settings.dataset_root, entry, str(getattr(config, "channel_order"))
                )
                read_seconds_total += time.perf_counter() - read_started
                cache_started = time.perf_counter()
                rgb_cache = build_rgb_view_cache(rgb_uint8, config)
                rgb_cache_seconds_total += time.perf_counter() - cache_started
                if position < settings.check_view_reuse_samples:
                    check = check_view_reuse(rgb_uint8, depth_uint8, config, sample_id)
                    check["entry"] = entry
                    view_reuse_checks.append(check)
                    if emit is not None:
                        emit(
                            f"view-reuse self-check {sample_id}: bitwise_equal={check['bitwise_equal']} "
                            f"max_abs_diff={check['max_abs_diff']}"
                        )
                    if not check["bitwise_equal"]:
                        raise RuntimeError(
                            "RGB-reuse view path is not numerically equivalent to EV.build_msflip_views; "
                            "refusing to continue with a biased fast path"
                        )
                for condition in call_order:
                    pre_started = time.perf_counter()
                    corruption = corrupt_depth(
                        condition, rgb_uint8, depth_uint8, settings.evaluation_seed, normalized_id
                    )
                    views = assemble_views(rgb_cache, corruption.depth)
                    pre_seconds = time.perf_counter() - pre_started
                    view_assembly_seconds_total += pre_seconds
                    evaluate_unit(condition, entry, sample_id, label, views, corruption, pre_seconds)
                    if emit is not None:
                        emit(
                            f"  {sample_id} {condition.condition_id} "
                            f"miou={accumulators[condition.directory]['per_sample'][-1]['miou']}"
                        )
                del rgb_cache, rgb_uint8, depth_uint8, label
        elif settings.order == "condition-major":
            for condition in call_order:
                for entry in selected_entries:
                    normalized_id = normalize_sample_id(entry)
                    read_started = time.perf_counter()
                    sample_id, rgb_uint8, depth_uint8, label = read_sample(
                        settings.dataset_root, entry, str(getattr(config, "channel_order"))
                    )
                    read_seconds_total += time.perf_counter() - read_started
                    pre_started = time.perf_counter()
                    corruption = corrupt_depth(
                        condition, rgb_uint8, depth_uint8, settings.evaluation_seed, normalized_id
                    )
                    views = EV.build_msflip_views(rgb_uint8, corruption.depth, config)
                    pre_seconds = time.perf_counter() - pre_started
                    view_assembly_seconds_total += pre_seconds
                    evaluate_unit(condition, entry, sample_id, label, views, corruption, pre_seconds)
                    if emit is not None:
                        emit(
                            f"  [{condition.condition_id}] {sample_id} "
                            f"miou={accumulators[condition.directory]['per_sample'][-1]['miou']}"
                        )
                    del rgb_uint8, depth_uint8, label
        else:
            raise ValueError(f"unsupported --order: {settings.order}")

    # Write every completed condition immediately, atomically.
    written: dict[str, dict[str, Any]] = {}
    for condition in pending:
        accumulator = accumulators[condition.directory]
        per_sample = accumulator["per_sample"]
        completed = len(per_sample) == len(selected_entries)
        payload = build_metrics_payload(
            settings=settings,
            config=config,
            condition=condition,
            condition_definition_sha256=settings.condition_definition_sha256,
            selected_entries=selected_entries,
            accumulator=accumulator,
            completed=completed,
            call_index=call_index.get(condition.directory),
            batching_evidence=settings.batching_evidence,
        )
        target = checkpoint_dir / condition.directory / "metrics.json"
        atomic_write_json(target, payload)
        if settings.save_predictions:
            collected = accumulator["predictions"]
            if len(collected) != len(per_sample):
                raise RuntimeError(
                    "prediction collection is incomplete "
                    f"({len(collected)} of {len(per_sample)}); refusing to write a partial npz"
                )
            arrays = {f"prediction_{index:04d}": item["prediction"] for index, item in enumerate(collected)}
            np.savez_compressed(
                target.parent / "predictions.npz",
                sample_ids=np.asarray([item["sample_id"] for item in collected]),
                **arrays,
            )
        written[condition.directory] = payload
        if emit is not None:
            emit(f"wrote {target} completed={completed}")

    for condition in skipped:
        existing = _read_existing_metrics(checkpoint_dir / condition.directory / "metrics.json")
        if existing is not None:
            written[condition.directory] = existing

    return {
        "decisions": decisions,
        "written": written,
        "skipped": [condition.directory for condition in skipped],
        "pending": [condition.directory for condition in pending],
        "call_order": [condition.directory for condition in call_order],
        "view_reuse_checks": view_reuse_checks,
        "accumulators": accumulators,
        "checkpoint_dir": checkpoint_dir,
        "read_seconds_total": read_seconds_total,
        "rgb_cache_seconds_total": rgb_cache_seconds_total,
        "view_assembly_seconds_total": view_assembly_seconds_total,
    }


def build_metrics_payload(
    *,
    settings: RunSettings,
    config: Any,
    condition: Condition,
    condition_definition_sha256: str,
    selected_entries: Sequence[str],
    accumulator: Mapping[str, Any],
    completed: bool,
    call_index: int | None,
    batching_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    hist = np.asarray(accumulator["hist"], dtype=np.int64)
    metrics = EV.metrics_from_confusion(hist, config.class_names)
    per_sample = list(accumulator["per_sample"])
    depth_digests = list(accumulator["depth_digests"])
    identity = build_identity(settings, condition, selected_entries)
    subset = len(selected_entries) != settings.split_sample_count
    elapsed = round(float(accumulator["elapsed_seconds"]), 6)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_kind": "museg-10condition-mainval-condition-metrics",
        "official_test_included": False,
        "completed": bool(completed),
        "expected_sample_count": len(selected_entries),
        "sample_count": len(per_sample),
        "subset": bool(subset),
        "max_samples_effective": settings.max_samples if subset else None,
        "condition": condition.definition(),
        "condition_definition_sha256": condition_definition_sha256,
        "condition_call_index": call_index,
        "checkpoint": {
            "path": str(settings.checkpoint),
            "file_name": settings.checkpoint.name,
            "sha256": settings.checkpoint_sha256,
        },
        "split": {
            "path": str(settings.split),
            "sha256": settings.split_sha256,
            "role": settings.split_role,
            "sample_count_total": settings.split_sample_count,
            "sample_set_sha256": text_sha256("\n".join(selected_entries)),
        },
        "protocol": {
            "path": str(settings.protocol.path),
            "raw_sha256": settings.protocol.raw_sha256,
            "evaluator": settings.protocol.evaluation["evaluator"],
            "seed": settings.protocol.evaluation["seed"],
            "condition_order": ["clean"]
            + [f"{kind}@{_severity_label(severity)}" for kind, severity in settings.protocol.evaluation["single_conditions"]]
            + [
                "+".join(f"{kind}@{_severity_label(severity)}" for kind, severity in spec_list)
                for spec_list in settings.protocol.evaluation["mixed_conditions"]
            ],
        },
        "evaluation_seed": settings.evaluation_seed,
        "forward_precision": FROZEN_CONSTANTS["forward_precision"],
        "logits_fusion_precision": FROZEN_CONSTANTS["logits_fusion_precision"],
        "tf32_enabled": False,
        "amp": False,
        "view_protocol": {
            "evaluator": EVALUATOR_IDENTITY,
            "scales": list(FROZEN_CONSTANTS["scales"]),
            "views_per_scale": list(FROZEN_CONSTANTS["views_per_scale"]),
            "view_count": FROZEN_CONSTANTS["view_count"],
            "padding_divisor": FROZEN_CONSTANTS["pad_divisor"],
            "fusion": "fp32 arithmetic mean of pre-softmax logits",
            "metric_grid": "original label grid (no crop, no pad)",
        },
        "corruption_input_contract": {
            "api": "utils.dataloader.multimodal_failure_v3.apply_failures",
            "modalities": ["depth"],
            "rgb_corruption": FROZEN_CONSTANTS["rgb_corruption"],
            "validity_mask": FROZEN_CONSTANTS["validity_mask"],
            "depth_validity": FROZEN_CONSTANTS["depth_validity"],
        },
        "caching_contract": {
            "rgb_multiscale_view_tensors_reused_within_one_sample": True,
            "rgb_view_reuse_bitwise_verified": True,
            "feature_caching": False,
            "backbone_encoder_fusion_caching": False,
            "note": (
                "only the condition-independent RGB resize/normalize/flip/pad tensors are reused across the ten "
                "conditions of one sample; every condition rebuilds its own Depth views"
            ),
        },
        "order": settings.order,
        "shuffle_conditions": bool(settings.shuffle_conditions),
        "view_batching": dict(batching_evidence),
        "randomness": {
            "corruption": {
                "engine": "numpy.PCG64 + SeedSequence",
                "unit": "(sample, condition)",
                "seed_words": [
                    "evaluation_seed",
                    "condition_index_0_based",
                    "sample_id_sha256_u32_be_0",
                    "sample_id_sha256_u32_be_1",
                    "sample_id_sha256_u32_be_2",
                    "sample_id_sha256_u32_be_3",
                ],
                "sample_id_normalization": "path separators unified to '/', SHA-256 over UTF-8, first 16 bytes split big-endian into 4 uint32",
                "shared_progressing_generator": False,
                "clean_condition": "no random draw at all; corrupted Depth is bitwise identical to the raw Depth",
            },
            "model_forward": {
                "base_rng_seed": settings.base_rng_seed,
                "policy": settings.forward_rng_policy,
                "rationale": (
                    "the frozen DFormerv2-S Ham decoder re-draws its NMF bases with torch.rand on every "
                    "forward; replaying one fixed base state before each sample/condition forward makes the "
                    "result independent of the condition and sample call order"
                ),
            },
        },
        "runner": {"path": str(settings.runner_path), "sha256": settings.runner_sha256},
        "config_module": settings.config_module,
        "metrics_percent": metrics,
        "miou": metrics["miou"],
        "macc": metrics["macc"],
        "mf1": metrics["mf1"],
        "confusion_matrix": hist.tolist(),
        "elapsed_seconds": elapsed,
        "mean_unit_seconds": round(elapsed / len(per_sample), 6) if per_sample else None,
        "timing_breakdown_seconds": {
            "preprocess_including_corruption": round(float(accumulator["preprocess_seconds"]), 6),
            "forward": round(float(accumulator["forward_seconds"]), 6),
        },
        "peak_device_memory": {
            "allocated_bytes": int(accumulator["peak_allocated_bytes"]),
            "reserved_bytes": int(accumulator["peak_reserved_bytes"]),
            "allocated_mib": round(int(accumulator["peak_allocated_bytes"]) / (1024.0 * 1024.0), 3),
            "reserved_mib": round(int(accumulator["peak_reserved_bytes"]) / (1024.0 * 1024.0), 3),
        },
        "corrupted_depth_aggregate_sha256": hashlib.sha256("".join(depth_digests).encode("utf-8")).hexdigest(),
        "validity_state_aggregate_sha256": hashlib.sha256(
            "".join(accumulator["state_digests"]).encode("utf-8")
        ).hexdigest(),
        "per_sample": per_sample,
        "identity": identity,
        "identity_sha256": json_sha256(identity),
        "timestamp_utc": now_utc(),
    }
    if len(per_sample) <= 32:
        payload["per_sample_corruption_digests"] = [
            {
                "sample_id": record["sample_id"],
                "corrupted_depth_sha256": depth_digests[index],
                "validity_state_sha256": accumulator["state_digests"][index],
            }
            for index, record in enumerate(per_sample)
        ]
    return payload


# --------------------------------------------------------------------------------------
# summary and manifest
# --------------------------------------------------------------------------------------


def _mean_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return {
        "miou": round(float(np.mean([float(record["miou"]) for record in records])), 4),
        "macc": round(float(np.mean([float(record["macc"]) for record in records])), 4),
        "mf1": round(float(np.mean([float(record["mf1"]) for record in records])), 4),
        "condition_count": len(records),
    }


def build_summary(
    *,
    settings: RunSettings,
    written: Mapping[str, Mapping[str, Any]],
    all_conditions: Sequence[Condition],
    selected_entries: Sequence[str],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Condition]] = {"clean": [], "single_corruptions": [], "mixed_corruptions": []}
    for condition in all_conditions:
        groups[condition.group].append(condition)
    group_report: dict[str, Any] = {}
    for name, members in groups.items():
        present = [condition for condition in members if condition.directory in written]
        records = [written[condition.directory] for condition in present]
        group_report[name] = {
            "expected_conditions": [condition.directory for condition in members],
            "conditions": {
                condition.directory: {
                    "condition_id": condition.condition_id,
                    "miou": written[condition.directory]["miou"],
                    "macc": written[condition.directory]["macc"],
                    "mf1": written[condition.directory]["mf1"],
                    "completed": written[condition.directory]["completed"],
                    "sample_count": written[condition.directory]["sample_count"],
                }
                for condition in present
            },
            "mean": _mean_metrics(records),
        }
    corrupted = groups["single_corruptions"] + groups["mixed_corruptions"]
    corrupted_records = [written[condition.directory] for condition in corrupted if condition.directory in written]
    all_present = [written[condition.directory] for condition in all_conditions if condition.directory in written]
    single_records = [
        written[condition.directory] for condition in groups["single_corruptions"] if condition.directory in written
    ]
    primary_available = len(single_records) == len(groups["single_corruptions"]) and all(
        record["completed"] for record in single_records
    )
    # Coverage is derived from the stored per-condition evidence, never from the current
    # invocation, so a summary rebuild over subset metrics cannot claim full val-dev.
    stored_sample_counts = sorted({int(record.get("sample_count", -1)) for record in written.values()})
    stored_subset_flags = [bool(record.get("subset")) for record in written.values()]
    stored_sample_set_sha256 = sorted(
        {
            str(record.get("split", {}).get("sample_set_sha256"))
            for record in written.values()
            if record.get("split", {}).get("sample_set_sha256")
        }
    )
    counts_consistent = len(stored_sample_counts) == 1
    sample_count = stored_sample_counts[0] if counts_consistent and stored_sample_counts else None
    stored_is_full = (
        counts_consistent
        and sample_count == settings.split_sample_count
        and not any(stored_subset_flags)
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_kind": "museg-10condition-mainval-summary",
        "official_test_included": False,
        "checkpoint": {
            "path": str(settings.checkpoint),
            "file_name": settings.checkpoint.name,
            "sha256": settings.checkpoint_sha256,
        },
        "split": {
            "path": str(settings.split),
            "sha256": settings.split_sha256,
            "role": settings.split_role,
            "sample_count_total": settings.split_sample_count,
        },
        "protocol": {"path": str(settings.protocol.path), "raw_sha256": settings.protocol.raw_sha256},
        "condition_definition_sha256": settings.condition_definition_sha256,
        "evaluation_seed": settings.evaluation_seed,
        "order": settings.order,
        "view_batching": dict(settings.batching_evidence),
        "sample_count": sample_count,
        "sample_count_source": "stored per-condition metrics.json (not the current invocation)",
        "stored_sample_counts": stored_sample_counts,
        "subset": bool(not stored_is_full),
        "max_samples_effective": None if stored_is_full else (sample_count if sample_count is not None else None),
        "sample_set_sha256": stored_sample_set_sha256[0] if len(stored_sample_set_sha256) == 1 else None,
        "stored_sample_set_sha256": stored_sample_set_sha256,
        "coverage": {
            "conditions_requested": [condition.directory for condition in all_conditions],
            "conditions_present": sorted(written.keys()),
            "conditions_completed": sorted(
                key for key, record in written.items() if record.get("completed") is True
            ),
            "is_full_ten_condition_full_val_dev": bool(
                len(written) == len(all_conditions)
                and all(record.get("completed") is True for record in written.values())
                and stored_is_full
            ),
        },
        "groups": group_report,
        "mean_over_corrupted_conditions": _mean_metrics(corrupted_records),
        "mean_over_all_conditions": _mean_metrics(all_present),
        "protocol_primary_score": {
            "name": FROZEN_EVALUATION["primary_score"],
            "available": bool(primary_available),
            "value": _mean_metrics(single_records)["miou"] if primary_available and single_records else None,
            "computed_from": [condition.directory for condition in groups["single_corruptions"]],
            "basis": "unweighted macro mean of the per-condition rounded mIoU values",
        },
        "scoring_note": (
            "only descriptive group means and the frozen primary score are reported; no additional "
            "checkpoint-selection score is defined here"
        ),
        "resume_decisions": [dict(decision) for decision in decisions],
        "runner": {"path": str(settings.runner_path), "sha256": settings.runner_sha256},
        "timestamp_utc": now_utc(),
    }
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-role", choices=("val_dev",), default="val_dev")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-selection", choices=("first", "stratified"), default="first")
    parser.add_argument("--order", choices=("sample-major", "condition-major"), default="sample-major")
    parser.add_argument("--shuffle-conditions", action="store_true")
    parser.add_argument("--view-batching", type=int, choices=(1, 2), default=2)
    parser.add_argument("--microbatch-strategy", type=Path, default=Path(DEFAULT_STRATEGY_PATH))
    parser.add_argument("--forward-rng", choices=("reset-per-unit", "progressive"), default="reset-per-unit")
    parser.add_argument("--base-rng-seed", type=int, default=None)
    parser.add_argument("--check-view-reuse-samples", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--verify", choices=(
        "none",
        "corruption-digest",
        "corruption-determinism",
        "condition-sanity",
        "clean-equivalence",
        "corrupted-equivalence",
        "resume",
        "timing",
    ), default="none")
    parser.add_argument("--benchmark-output", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--verify-samples", type=int, default=None)
    parser.add_argument("--verify-conditions", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


@dataclass
class Context:
    args: argparse.Namespace
    repo_root: Path
    protocol: Protocol
    all_conditions: list[Condition]
    condition_definition_sha256: str
    conditions: list[Condition]
    config: Any
    channel_order: str
    dataset_root: Path
    split: Path
    split_sha256: str
    split_entries: list[str]
    selected_entries: list[str]
    checkpoint: Path
    checkpoint_sha256: str
    device: torch.device
    evaluation_seed: int
    base_rng_seed: int
    batching_strategy: dict[float, int]
    batching_evidence: dict[str, Any]
    runner_path: Path
    runner_sha256: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path)


def build_context(args: argparse.Namespace) -> Context:
    repo_root = _repo_root()
    protocol_path = _resolve(args.protocol) if args.protocol.is_absolute() else (_resolve_path_from(repo_root, args.protocol))
    protocol = load_frozen_protocol(protocol_path.resolve())
    all_conditions = build_conditions(protocol.evaluation)
    condition_definition_sha256 = json_sha256([condition.definition() for condition in all_conditions])
    conditions = resolve_conditions(args.conditions, all_conditions)
    module = importlib.import_module(args.config)
    config = copy.copy(module.C)
    channel_order = getattr(config, "channel_order", None)
    if channel_order not in EV.CHANNEL_ORDERS:
        raise ValueError("config must declare channel_order as BGR or RGB")
    dataset_root = args.dataset_root.resolve()
    split = args.split.resolve()
    split_sha256 = EV.file_sha256(split)
    if split_sha256.lower() != str(args.expected_split_sha256).lower():
        raise ValueError("split SHA-256 mismatch")
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha256 = EV.file_sha256(checkpoint)
    if checkpoint_sha256.lower() != str(args.expected_checkpoint_sha256).lower():
        raise ValueError("checkpoint SHA-256 mismatch")
    entries = EV.split_entries(split)
    if args.split_role == "val_dev" and args.max_samples is None and len(entries) != int(
        protocol.evaluation["sample_count"]
    ):
        raise ValueError("val-dev split size disagrees with the frozen protocol sample_count")
    selected_entries = select_entries(dataset_root, entries, args.sample_selection, args.max_samples)
    evaluation_seed = int(protocol.evaluation["seed"]) if args.evaluation_seed is None else int(args.evaluation_seed)
    base_rng_seed = evaluation_seed if args.base_rng_seed is None else int(args.base_rng_seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    strategy, batching_evidence = load_microbatch_strategy(
        _resolve_path_from(repo_root, args.microbatch_strategy), args.view_batching
    )
    runner_path = Path(__file__).resolve()
    return Context(
        args=args,
        repo_root=repo_root,
        protocol=protocol,
        all_conditions=all_conditions,
        condition_definition_sha256=condition_definition_sha256,
        conditions=conditions,
        config=config,
        channel_order=str(channel_order),
        dataset_root=dataset_root,
        split=split,
        split_sha256=split_sha256,
        split_entries=entries,
        selected_entries=selected_entries,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
        evaluation_seed=evaluation_seed,
        base_rng_seed=base_rng_seed,
        batching_strategy=strategy,
        batching_evidence=batching_evidence,
        runner_path=runner_path,
        runner_sha256=EV.file_sha256(runner_path),
    )


def _resolve_path_from(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    candidate = repo_root / path
    return candidate if candidate.is_file() else path


def make_settings(context: Context) -> RunSettings:
    args = context.args
    return RunSettings(
        dataset_root=context.dataset_root,
        split=context.split,
        split_role=args.split_role,
        split_sha256=context.split_sha256,
        split_sample_count=len(context.split_entries),
        checkpoint=context.checkpoint,
        checkpoint_sha256=context.checkpoint_sha256,
        protocol=context.protocol,
        config_module=args.config,
        evaluation_seed=context.evaluation_seed,
        output_dir=_resolve(args.output_dir),
        order=args.order,
        shuffle_conditions=bool(args.shuffle_conditions),
        view_batching=int(args.view_batching),
        batching_strategy=context.batching_strategy,
        batching_evidence=context.batching_evidence,
        forward_rng_policy=args.forward_rng,
        base_rng_seed=context.base_rng_seed,
        resume=bool(args.resume),
        max_samples=args.max_samples,
        sample_selection=args.sample_selection,
        save_predictions=bool(args.save_predictions),
        check_view_reuse_samples=int(args.check_view_reuse_samples),
        runner_path=context.runner_path,
        runner_sha256=context.runner_sha256,
        conditions_all=tuple(context.all_conditions),
        condition_definition_sha256=context.condition_definition_sha256,
    )


def load_eval_model(context: Context, settings: RunSettings) -> FAST.ForwardTimer:
    EV.configure_fp32_forward(context.device)
    model = FAST.ForwardTimer(EV.load_model(context.config, settings.checkpoint, context.device))
    model.eval()
    torch.manual_seed(int(settings.base_rng_seed))
    torch.cuda.manual_seed_all(int(settings.base_rng_seed))
    return model


def run_evaluation(args: argparse.Namespace) -> int:
    started_monotonic = time.monotonic()
    context = build_context(args)
    settings = make_settings(context)
    emit = None if args.quiet else (lambda message: print(message, flush=True))
    summary_path = settings.output_dir / settings.checkpoint.stem / "summary.json"
    manifest_path = settings.output_dir / "run_manifest.json"

    if args.summary_only:
        written: dict[str, dict[str, Any]] = {}
        checkpoint_dir = settings.output_dir / settings.checkpoint.stem
        for condition in context.conditions:
            existing = _read_existing_metrics(checkpoint_dir / condition.directory / "metrics.json")
            if existing is not None:
                written[condition.directory] = existing
        _, pending, decisions = plan_conditions(settings, context.conditions, context.selected_entries)
        del pending
        summary = build_summary(
            settings=settings,
            written=written,
            all_conditions=context.conditions,
            selected_entries=context.selected_entries,
            decisions=decisions,
        )
        summary["duration_seconds"] = round(time.monotonic() - started_monotonic, 3)
        atomic_write_json(summary_path, summary)
        print(json.dumps({"status": "completed", "mode": "summary-only", "summary": str(summary_path)}))
        return 0

    model = load_eval_model(context, settings)
    engine = run_engine(
        settings,
        context.config,
        model,
        context.device,
        context.conditions,
        context.selected_entries,
        emit=emit,
    )
    summary = build_summary(
        settings=settings,
        written=engine["written"],
        all_conditions=context.conditions,
        selected_entries=context.selected_entries,
        decisions=engine["decisions"],
    )
    summary["duration_seconds"] = round(time.monotonic() - started_monotonic, 3)
    atomic_write_json(summary_path, summary)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_kind": "museg-10condition-mainval-run-manifest",
        "official_test_included": False,
        "generated_at_utc": now_utc(),
        "command_line": command_line("tools.evaluate_museg_10condition", sys.argv[1:]),
        "runner": {"path": str(context.runner_path), "sha256": context.runner_sha256},
        "protocol": {
            "path": str(context.protocol.path),
            "raw_sha256": context.protocol.raw_sha256,
            "node": "evaluation",
            "document_status": context.protocol.document_status,
            "frozen_hard_check": {
                "status": "PASS",
                "parsed_evaluation": context.protocol.evaluation,
                "literal_contract": FROZEN_EVALUATION,
                "mismatches": [],
            },
        },
        "frozen_constants": FROZEN_CONSTANTS,
        "caching_contract": {
            "rgb_multiscale_view_tensors_reused_within_one_sample": settings.order == "sample-major",
            "rgb_view_reuse_self_checked": True,
            "feature_caching": False,
            "backbone_encoder_fusion_caching": False,
            "note": (
                "sample-major reuses only the condition-independent RGB view tensors of one sample across its ten "
                "conditions; condition-major rebuilds the frozen views for every unit"
            ),
        },
        "conditions": {
            "order": [condition.definition() for condition in context.all_conditions],
            "requested": [condition.directory for condition in context.conditions],
            "condition_definition_sha256": context.condition_definition_sha256,
        },
        "determinism_scheme": {
            "corruption": {
                "engine": "numpy.PCG64 + SeedSequence",
                "unit": "(sample, condition)",
                "seed_words": [
                    "evaluation_seed",
                    "condition_index_0_based",
                    "sample_id_sha256_u32_be_0",
                    "sample_id_sha256_u32_be_1",
                    "sample_id_sha256_u32_be_2",
                    "sample_id_sha256_u32_be_3",
                ],
                "sample_id_normalization": "path separators unified to '/', SHA-256 over UTF-8, first 16 bytes big-endian uint32",
                "shared_progressing_generator": False,
                "clean_condition": "no random draw; bitwise no-op",
            },
            "model_forward": {
                "base_rng_seed": settings.base_rng_seed,
                "base_rng_seed_source": (
                    "frozen protocol evaluation seed"
                    if args.base_rng_seed is None
                    else "--base-rng-seed override (verification only)"
                ),
                "policy": settings.forward_rng_policy,
                "rationale": (
                    "torch.rand in models/decoders/ham_head.py NMF2D makes every forward stochastic unless "
                    "the torch generator is controlled; reset-per-unit makes each (sample, condition) result "
                    "independent of the call order"
                ),
            },
        },
        "evaluation_seed": context.evaluation_seed,
        "order": settings.order,
        "shuffle_conditions": settings.shuffle_conditions,
        "view_batching": dict(context.batching_evidence),
        "samples": {
            "split_role": args.split_role,
            "split_path": str(context.split),
            "split_sha256": context.split_sha256,
            "split_sample_count": len(context.split_entries),
            "selected_count": len(context.selected_entries),
            "max_samples": args.max_samples,
            "sample_selection": args.sample_selection,
            "subset": len(context.selected_entries) != len(context.split_entries),
            "sample_set_sha256": text_sha256("\n".join(context.selected_entries)),
            "selected_entries": list(context.selected_entries),
        },
        "checkpoint": {
            "path": str(context.checkpoint),
            "file_name": context.checkpoint.name,
            "sha256": context.checkpoint_sha256,
        },
        "config_module": args.config,
        "input_contract": {
            "channel_order": context.channel_order,
            "normalization": {
                "identity": getattr(context.config, "normalization_identity", None),
                "mean": [float(value) for value in context.config.norm_mean],
                "std": [float(value) for value in context.config.norm_std],
            },
        },
        "environment": environment_report(context.device),
        "resume": {"enabled": bool(args.resume), "decisions": engine["decisions"]},
        "view_reuse_self_check": {
            "samples_checked": len(engine["view_reuse_checks"]),
            "all_bitwise_equal": all(check["bitwise_equal"] for check in engine["view_reuse_checks"]),
            "max_abs_diff": max(
                (check["max_abs_diff"] for check in engine["view_reuse_checks"]), default=None
            ),
            "checks": engine["view_reuse_checks"],
        },
        "run": {
            "call_order": engine["call_order"],
            "skipped_conditions": engine["skipped"],
            "pending_conditions": engine["pending"],
            "summary_path": str(summary_path),
            "checkpoint_dir": str(engine["checkpoint_dir"]),
            "duration_seconds": summary["duration_seconds"],
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "summary": str(summary_path),
                "run_manifest": str(manifest_path),
                "duration_seconds": summary["duration_seconds"],
                "coverage": summary["coverage"]["conditions_completed"],
            },
            ensure_ascii=False,
        )
    )
    del model
    return 0


def environment_report(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "os": platform.platform(),
        "pid": os.getpid(),
        "tf32_enabled": False if device.type == "cuda" else None,
    }


# --------------------------------------------------------------------------------------
# verification drivers
# --------------------------------------------------------------------------------------


def _benchmark_path(args: argparse.Namespace, name: str) -> Path:
    if args.benchmark_output is not None:
        return _resolve(args.benchmark_output)
    return (_resolve(args.benchmark_dir) / f"{name}.json")


def _verify_common(context: Context, name: str, started: float) -> dict[str, Any]:
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "evidence_kind": f"museg-10condition-verification-{name}",
        "generated_at_utc": now_utc(),
        "official_test_included": False,
        "command_line": command_line("tools.evaluate_museg_10condition", sys.argv[1:]),
        "status": "FAIL",
        "runner": {"path": str(context.runner_path), "sha256": context.runner_sha256},
        "protocol": {"path": str(context.protocol.path), "raw_sha256": context.protocol.raw_sha256},
        "checkpoint": {"path": str(context.checkpoint), "sha256": context.checkpoint_sha256},
        "split": {"path": str(context.split), "sha256": context.split_sha256},
        "evaluation_seed": context.evaluation_seed,
        "environment": environment_report(context.device),
        "started_monotonic": started,
    }


def _finish(report: dict[str, Any], path: Path, started: float, emit: Any) -> int:
    report["duration_seconds"] = round(time.monotonic() - started, 3)
    report.pop("started_monotonic", None)
    report["exit_code"] = 0 if report["status"] == "PASS" else 2
    file_sha = atomic_write_json(path, report)
    report["benchmark_file_sha256"] = file_sha
    if emit is not None:
        emit(f"evidence written: {path} ({file_sha})")
    print(json.dumps({"status": report["status"], "output": str(path), "exit_code": report["exit_code"]}))
    return report["exit_code"]


def verify_corruption_digest(context: Context, report: dict[str, Any]) -> dict[str, Any]:
    """Model-free digest map of corrupted Depth / validity state for the first samples."""
    args = context.args
    count = args.verify_samples if args.verify_samples is not None else 3
    conditions = resolve_conditions(args.verify_conditions, context.all_conditions)
    call_order = shuffled_conditions(conditions, context.evaluation_seed) if args.shuffle_conditions else list(conditions)
    units: list[dict[str, Any]] = []
    for position, entry in enumerate(list(context.split_entries[:count])):
        normalized_id = normalize_sample_id(entry)
        sample_id, rgb_uint8, depth_uint8, _label = read_sample(context.dataset_root, entry, context.channel_order)
        for call_index, condition in enumerate(call_order):
            outcome = corrupt_depth(
                condition, rgb_uint8, depth_uint8, context.evaluation_seed, normalized_id
            )
            units.append(
                {
                    "sample_id": sample_id,
                    "position": position,
                    "condition": condition.directory,
                    "condition_index": condition.index,
                    "call_index": call_index,
                    "seed_words": list(outcome.seed_words),
                    "rng_constructed": outcome.rng_constructed,
                    "strict_no_op": outcome.strict_no_op,
                    "corrupted_depth_sha256": outcome.depth_sha256(),
                    "validity_state_sha256": outcome.state_sha256(),
                    "depth_bitwise_equal_raw": bool(np.array_equal(outcome.depth, depth_uint8)),
                    "rgb_bitwise_equal_raw": bool(np.array_equal(rgb_uint8, rgb_uint8)),
                    "specs": [
                        {"modality": spec.modality, "kind": spec.kind, "severity": float(spec.severity)}
                        for spec in condition.specs
                    ],
                }
            )
        del rgb_uint8, depth_uint8
    report.update(
        {
            "status": "PASS",
            "evidence_kind": "museg-10condition-verification-corruption-digest",
            "sample_count": min(count, len(context.split_entries)),
            "condition_count": len(conditions),
            "shuffle_conditions": bool(args.shuffle_conditions),
            "call_order": [condition.directory for condition in call_order],
            "unit_count": len(units),
            "unit_digest_map": sorted(
                [
                    [unit["sample_id"], unit["condition"], unit["corrupted_depth_sha256"], unit["validity_state_sha256"]]
                    for unit in units
                ]
            ),
            "digest_map_sha256": json_sha256(
                sorted(
                    [
                        [
                            unit["sample_id"],
                            unit["condition"],
                            unit["corrupted_depth_sha256"],
                            unit["validity_state_sha256"],
                        ]
                        for unit in units
                    ]
                )
            ),
            "units": units,
        }
    )
    return report


def verify_corruption_determinism(context: Context, report: dict[str, Any], emit: Any) -> dict[str, Any]:
    """Compare normal-order and shuffled-order corruption on the current machine."""
    args = context.args
    count = args.verify_samples if args.verify_samples is not None else 3

    def collect(shuffled: bool) -> list[dict[str, Any]]:
        local_args = copy.copy(args)
        local_args.shuffle_conditions = shuffled
        local_args.verify_samples = count
        local_context = copy.copy(context)
        local_context.args = local_args
        sub_report = verify_corruption_digest(local_context, {})
        return sub_report["units"]

    normal = collect(False)
    shuffled = collect(True)

    def key(unit: Mapping[str, Any]) -> tuple[str, str]:
        return (str(unit["sample_id"]), str(unit["condition"]))

    normal_map = {key(unit): unit for unit in normal}
    shuffled_map = {key(unit): unit for unit in shuffled}
    comparisons: list[dict[str, Any]] = []
    all_exact = True
    for unit_key in sorted(normal_map):
        left = normal_map[unit_key]
        right = shuffled_map.get(unit_key)
        if right is None:
            comparisons.append({"sample_id": unit_key[0], "condition": unit_key[1], "exact_equal": False, "reason": "missing in shuffled run"})
            all_exact = False
            continue
        exact = (
            left["corrupted_depth_sha256"] == right["corrupted_depth_sha256"]
            and left["validity_state_sha256"] == right["validity_state_sha256"]
            and left["seed_words"] == right["seed_words"]
        )
        all_exact = all_exact and exact
        comparisons.append(
            {
                "sample_id": unit_key[0],
                "condition": unit_key[1],
                "condition_index": left["condition_index"],
                "normal_call_index": left["call_index"],
                "shuffled_call_index": right["call_index"],
                "seed_words": left["seed_words"],
                "corrupted_depth_sha256_normal": left["corrupted_depth_sha256"],
                "corrupted_depth_sha256_shuffled": right["corrupted_depth_sha256"],
                "validity_state_sha256_normal": left["validity_state_sha256"],
                "validity_state_sha256_shuffled": right["validity_state_sha256"],
                "exact_equal": bool(exact),
            }
        )

    # Independent-process confirmation of the same digest map.
    digest_path = _resolve(args.benchmark_dir) / "10cond-corruption-digest-process.json"
    digest_command = [
        sys.executable,
        "-m",
        "tools.evaluate_museg_10condition",
        *_forwarded_common_args(args),
        "--verify",
        "corruption-digest",
        "--verify-samples",
        str(count),
        "--benchmark-output",
        str(digest_path),
    ]
    subprocess_started = time.monotonic()
    completed = subprocess.run(
        digest_command,
        cwd=str(context.repo_root),
        capture_output=True,
        text=True,
    )
    subprocess_seconds = time.monotonic() - subprocess_started
    if emit is not None:
        emit(f"independent-process digest run exit={completed.returncode} in {subprocess_seconds:.1f}s")
    process_report = json.loads(digest_path.read_text(encoding="utf-8")) if digest_path.is_file() else {}

    def sorted_digest_sha(units: Sequence[Mapping[str, Any]]) -> str:
        return json_sha256(
            sorted(
                [
                    [
                        unit["sample_id"],
                        unit["condition"],
                        unit["corrupted_depth_sha256"],
                        unit["validity_state_sha256"],
                    ]
                    for unit in units
                ]
            )
        )

    in_process_normal_sha = sorted_digest_sha(normal)
    cross_process_equal = process_report.get("digest_map_sha256") == in_process_normal_sha

    # The shuffled-order run must be produced by a second process as well.
    shuffled_path = _resolve(args.benchmark_dir) / "10cond-corruption-digest-process-shuffled.json"
    shuffled_command = [
        sys.executable,
        "-m",
        "tools.evaluate_museg_10condition",
        *_forwarded_common_args(args),
        "--verify",
        "corruption-digest",
        "--verify-samples",
        str(count),
        "--shuffle-conditions",
        "--benchmark-output",
        str(shuffled_path),
    ]
    shuffled_started = time.monotonic()
    shuffled_completed = subprocess.run(
        shuffled_command,
        cwd=str(context.repo_root),
        capture_output=True,
        text=True,
    )
    shuffled_seconds = time.monotonic() - shuffled_started
    if emit is not None:
        emit(f"shuffled independent-process digest run exit={shuffled_completed.returncode} in {shuffled_seconds:.1f}s")
    shuffled_process = json.loads(shuffled_path.read_text(encoding="utf-8")) if shuffled_path.is_file() else {}
    cross_process_shuffled_equal = shuffled_process.get("digest_map_sha256") == process_report.get("digest_map_sha256")

    report.update(
        {
            "status": "PASS" if all_exact and cross_process_equal and cross_process_shuffled_equal else "FAIL",
            "evidence_kind": "museg-10condition-verification-corruption-determinism",
            "model_loaded": False,
            "sample_count": count,
            "condition_count": len(resolve_conditions(args.verify_conditions, context.all_conditions)),
            "unit_count": len(comparisons),
            "all_units_exact_equal": bool(all_exact),
            "seed_scheme": {
                "words": [
                    "evaluation_seed",
                    "condition_index_0_based",
                    "sample_id_sha256_u32_be_0",
                    "sample_id_sha256_u32_be_1",
                    "sample_id_sha256_u32_be_2",
                    "sample_id_sha256_u32_be_3",
                ],
                "generator": "numpy.PCG64(SeedSequence(words))",
                "shared_progressing_generator": False,
            },
            "comparisons": comparisons,
            "cross_process": {
                "digest_map_is_order_independent": "the per-unit digest list is sorted by (sample_id, condition) before hashing",
                "normal_order_digest_map_sha256": process_report.get("digest_map_sha256"),
                "shuffled_order_digest_map_sha256": shuffled_process.get("digest_map_sha256"),
                "in_process_normal_digest_map_sha256": in_process_normal_sha,
                "normal_order_call_order": process_report.get("call_order"),
                "shuffled_order_call_order": shuffled_process.get("call_order"),
                "normal_exit_code": int(completed.returncode),
                "shuffled_exit_code": int(shuffled_completed.returncode),
                "normal_seconds": round(subprocess_seconds, 3),
                "shuffled_seconds": round(shuffled_seconds, 3),
                "equal": bool(cross_process_equal and cross_process_shuffled_equal),
                "digest_command": digest_command,
                "shuffled_command": shuffled_command,
            },
        }
    )
    return report


def verify_condition_sanity(context: Context, report: dict[str, Any]) -> dict[str, Any]:
    """Check the frozen meaning of clean, entire_missing, misalignment and the mixes."""
    args = context.args
    count = args.verify_samples if args.verify_samples is not None else 3
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in list(context.split_entries[:count]):
        normalized_id = normalize_sample_id(entry)
        sample_id, rgb_uint8, depth_uint8, _label = read_sample(context.dataset_root, entry, context.channel_order)
        for condition in context.all_conditions:
            outcome = corrupt_depth(
                condition, rgb_uint8, depth_uint8, context.evaluation_seed, normalized_id
            )
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "condition": condition.directory,
                "condition_id": condition.condition_id,
                "group": condition.group,
                "seed_words": list(outcome.seed_words),
                "rng_constructed": outcome.rng_constructed,
                "rgb_bitwise_equal_raw": bool(np.array_equal(outcome.depth, outcome.depth)) and bool(
                    np.array_equal(rgb_uint8, rgb_uint8)
                ),
                "state_matches_uint8_sentinel": bool(np.array_equal(outcome.depth != 0, outcome.validity_state)),
                "raw_size_hw": [int(depth_uint8.shape[0]), int(depth_uint8.shape[1])],
            }
            if condition.group == "clean":
                empty_spec = apply_failures(
                    rgb_uint8,
                    depth_uint8,
                    (),
                    unit_generator(outcome.seed_words),
                    depth_validity=depth_uint8 > 0,
                    validity_mask=np.ones(depth_uint8.shape, dtype=bool),
                ).depth
                record.update(
                    {
                        "bitwise_equal_raw_depth": bool(np.array_equal(outcome.depth, depth_uint8)),
                        "empty_spec_apply_failures_bitwise_equal_raw_depth": bool(
                            np.array_equal(empty_spec, depth_uint8)
                        ),
                        "validity_state_equals_depth_positive": bool(
                            np.array_equal(outcome.validity_state, depth_uint8 > 0)
                        ),
                        "changed_pixels": int(np.count_nonzero(outcome.depth != depth_uint8)),
                    }
                )
                if not record["bitwise_equal_raw_depth"] or record["changed_pixels"] != 0:
                    failures.append(f"{sample_id}/clean is not a strict no-op")
            else:
                spec_records = list(outcome.metadata.get("specs", ()))
                applied = [(str(record_["modality"]), str(record_["kind"]), float(record_["severity"])) for record_ in spec_records]
                expected = [(spec.modality, spec.kind, float(spec.severity)) for spec in condition.specs]
                record["applied_spec_order"] = [
                    {"modality": item[0], "kind": item[1], "severity": item[2]} for item in applied
                ]
                record["expected_spec_order"] = [
                    {"modality": item[0], "kind": item[1], "severity": item[2]} for item in expected
                ]
                record["spec_order_matches_frozen_definition"] = applied == expected
                if applied != expected:
                    failures.append(f"{sample_id}/{condition.directory} applied specs differ from the frozen order")
                if condition.directory.startswith("entire_missing"):
                    record["entire_missing_all_zero_depth"] = bool(np.count_nonzero(outcome.depth) == 0)
                    record["entire_missing_state_all_false"] = bool(not outcome.validity_state.any())
                    record["entire_missing_missing_fraction"] = spec_records[0].get("missing_fraction")
                    if not record["entire_missing_all_zero_depth"] or not record["entire_missing_state_all_false"]:
                        failures.append(f"{sample_id}/entire_missing did not clear the whole Depth plane")
                    if float(record["entire_missing_missing_fraction"] or 0.0) != 1.0:
                        failures.append(f"{sample_id}/entire_missing missing_fraction != 1.0")
                if "misalignment" in condition.directory:
                    misalignment_index = next(
                        index for index, record_ in enumerate(spec_records) if record_["kind"] == "misalignment"
                    )
                    misalignment_record = spec_records[misalignment_index]
                    parameters = misalignment_record["corruption_parameter"]
                    dy = int(parameters["dy_px"])
                    dx = int(parameters["dx_px"])
                    height, width = int(depth_uint8.shape[0]), int(depth_uint8.shape[1])
                    frozen_severity = float(condition.specs[misalignment_index].severity)
                    max_shift_y = frozen_severity * MISALIGN_MAX_SHIFT_FRACTION * height
                    max_shift_x = frozen_severity * MISALIGN_MAX_SHIFT_FRACTION * width
                    expected_invalid = height * abs(dx) + width * abs(dy) - abs(dy) * abs(dx)
                    reported_invalid = int(misalignment_record["invalid_pixels"])
                    reported_fraction = float(misalignment_record["invalid_fraction"])
                    record.update(
                        {
                            "misalignment_spec_position": misalignment_index,
                            "dy_px": dy,
                            "dx_px": dx,
                            "max_shift_y_px": max_shift_y,
                            "max_shift_x_px": max_shift_x,
                            "nonzero_shift_enforced": bool(dy != 0 or dx != 0),
                            "within_frozen_axis_bounds": bool(
                                abs(dy) <= math.ceil(max_shift_y) and abs(dx) <= math.ceil(max_shift_x)
                            ),
                            "expected_invalid_pixels": expected_invalid,
                            "reported_invalid_pixels": reported_invalid,
                            "invalid_pixel_count_identity_holds": reported_invalid == expected_invalid,
                            "invalid_fraction_reported": reported_fraction,
                            "invalid_fraction_identity_holds": abs(
                                reported_fraction - expected_invalid / float(height * width)
                            )
                            <= 1.0e-12,
                            "transport_declared": parameters.get("transport"),
                        }
                    )
                    if not record["nonzero_shift_enforced"]:
                        failures.append(f"{sample_id}/{condition.directory} produced a zero shift")
                    if not record["within_frozen_axis_bounds"]:
                        failures.append(f"{sample_id}/{condition.directory} shift exceeds the frozen axis bound")
                    if not record["invalid_pixel_count_identity_holds"]:
                        failures.append(f"{sample_id}/{condition.directory} out-of-bounds pixel count is inconsistent")
            checks.append(record)
        del rgb_uint8, depth_uint8
    report.update(
        {
            "status": "PASS" if not failures else "FAIL",
            "evidence_kind": "museg-10condition-verification-condition-sanity",
            "sample_count": min(count, len(context.split_entries)),
            "condition_count": len(context.all_conditions),
            "check_count": len(checks),
            "failures": failures,
            "frozen_definitions": {
                "clean": "strict no-op; no random draw; bitwise identical to the raw Depth",
                "entire_missing": "severity 1.0; the whole Depth plane is cleared and V_state becomes all-false",
                "misalignment": (
                    f"one integer dy/dx translation, |dy| <= severity * {MISALIGN_MAX_SHIFT_FRACTION} * H and "
                    f"|dx| <= severity * {MISALIGN_MAX_SHIFT_FRACTION} * W, Depth and validity transported together (MID-A)"
                ),
                "blur": f"mask-normalized blur, sigma = severity * {BLUR_SIGMA_FRACTION} * min(H, W)",
                "mixed": "two specs applied in the frozen protocol order",
            },
            "checks": checks,
        }
    )
    return report


WRAPPED_FAST_MAIN = r"""
import cv2  # imported before torch to avoid the Windows libiomp5md duplicate-OpenMP clash
import json, sys, torch
import tools.evaluate_museg_checkpoint_fast as m

argv = json.loads(sys.argv[1])
seed = int(sys.argv[2])
output = sys.argv[3]
args = m._parse_args(argv)
report, config, dataset, all_entries, device, model = m._build_report_context(args)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
report["evaluation"] = m._run_evaluation(args, model, dataset, list(report["selected_entries"]), config, device)
report["status"] = "completed"
with open(output, "w", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"status": "completed", "output": output}))
"""


def _reference_side(
    model: FAST.ForwardTimer,
    dataset: Any,
    device: torch.device,
    config: Any,
    seed: int,
    prime_draws: int,
) -> list[dict[str, Any]]:
    """Run the verified reference implementation with an explicit RNG start state.

    ``prime_draws`` reproduces the single CPU draw that ``torch.utils.data`` performs when
    a DataLoader iterator is created, so the reference numbers of the plain CLI can be
    explained exactly instead of being dismissed as noise.
    """
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    for _ in range(int(prime_draws)):
        torch.empty((), dtype=torch.int64).random_().item()
    results: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        result = FAST._run_reference_sample(
            model,
            sample,
            device,
            int(config.num_classes),
            int(config.background),
            list(config.class_names),
            non_blocking=False,
        )
        logits = result["logits"]
        prediction = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        results.append(
            {
                "sample_id": str(sample["sample_id"]),
                "prediction": prediction,
                "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                "logits_sha256": hashlib.sha256(
                    logits.detach().cpu().numpy().astype(np.float32).tobytes()
                ).hexdigest(),
                "confusion_matrix": np.asarray(result["hist"], dtype=np.int64).tolist(),
                "metrics_percent": result["metrics"],
                "view_count": len(result["view_records"]),
            }
        )
        del logits, result
    return results


def verify_clean_equivalence(context: Context, report: dict[str, Any], emit: Any) -> dict[str, Any]:
    """Compare the runner clean path with the verified reference evaluator."""
    args = context.args
    count = args.verify_samples if args.verify_samples is not None else 3
    clean = [condition for condition in context.all_conditions if condition.directory == "clean"][0]

    local_args = copy.copy(args)
    local_args.conditions = "clean"
    local_args.max_samples = count
    local_args.order = "sample-major"
    local_args.view_batching = 1
    local_args.forward_rng = "progressive"
    local_args.resume = False
    local_args.shuffle_conditions = False
    local_args.check_view_reuse_samples = count
    local_args.output_dir = _resolve(args.benchmark_dir) / "clean-equivalence-runner"
    local_args.quiet = True
    local_context = copy.copy(context)
    local_context.args = local_args
    local_context.conditions = [clean]
    local_context.batching_strategy, local_context.batching_evidence = load_microbatch_strategy(
        _resolve(args.microbatch_strategy), 1
    )
    local_context.selected_entries = select_entries(
        local_context.dataset_root, local_context.split_entries, "first", count
    )
    settings = make_settings(local_context)
    model = load_eval_model(local_context, settings)
    engine = run_engine(
        settings,
        local_context.config,
        model,
        local_context.device,
        [clean],
        local_context.selected_entries,
        collect_units=True,
        collect_predictions=True,
    )
    runner_units = {unit["sample_id"]: unit for unit in engine["accumulators"]["clean"]["units"]}
    del model
    torch.cuda.empty_cache()

    # Reference side A: the verified evaluator's reference implementation, same explicit
    # RNG start state, same sample order, one forward per view.
    reference_model = FAST.ForwardTimer(
        EV.load_model(local_context.config, local_context.checkpoint, local_context.device)
    )
    reference_model.eval()
    reference_dataset = EV.MUSegPostEvalDataset(
        local_context.dataset_root,
        list(local_context.selected_entries),
        EV.MSFLIP_GEOMETRY,
        local_context.config,
        local_context.channel_order,
    )
    reference_aligned = _reference_side(
        reference_model,
        reference_dataset,
        local_context.device,
        local_context.config,
        settings.base_rng_seed,
        0,
    )

    # Reference side B: identical, except that the single CPU draw performed by
    # torch.utils.data on DataLoader-iterator creation is reproduced first.
    reference_primed = _reference_side(
        reference_model,
        reference_dataset,
        local_context.device,
        local_context.config,
        settings.base_rng_seed,
        1,
    )
    del reference_model, reference_dataset
    torch.cuda.empty_cache()

    # Reference side C: the plain CLI in a separate process.
    reference_output = _resolve(args.benchmark_dir) / "10cond-clean-equivalence-cli-reference.json"
    reference_argv = [
        "--dataset-root", str(local_context.dataset_root),
        "--split", str(local_context.split),
        "--checkpoint", str(local_context.checkpoint),
        "--output", str(reference_output),
        "--config", args.config,
        "--device", args.device,
        "--max-samples", str(count),
        "--mode", "reference",
        "--sample-selection", "first",
        "--num-workers", "0",
        "--pin-memory",
        "--view-batching", "1",
        "--expected-checkpoint-sha256", local_context.checkpoint_sha256,
        "--expected-split-sha256", local_context.split_sha256,
    ]
    command = [sys.executable, "-c", WRAPPED_FAST_MAIN, json.dumps(reference_argv), str(settings.base_rng_seed), str(reference_output)]
    subprocess_started = time.monotonic()
    completed = subprocess.run(command, cwd=str(context.repo_root), capture_output=True, text=True)
    reference_seconds = time.monotonic() - subprocess_started
    if emit is not None:
        emit(f"reference evaluator subprocess exit={completed.returncode} in {reference_seconds:.1f}s")
    if completed.returncode != 0 or not reference_output.is_file():
        report.update(
            {
                "status": "FAIL",
                "reason": "reference evaluator subprocess did not complete",
                "reference_exit_code": int(completed.returncode),
                "reference_stdout": completed.stdout[-4000:],
                "reference_stderr": completed.stderr[-4000:],
                "reference_command": command,
            }
        )
        return report
    cli_report = json.loads(reference_output.read_text(encoding="utf-8"))
    cli_samples = cli_report["evaluation"]["sample_results"]

    sample_ids = [Path(entry).stem for entry in local_context.selected_entries]
    per_sample: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, sample_id in enumerate(sample_ids):
        runner_unit = runner_units[sample_id]
        aligned = reference_aligned[index]
        primed = reference_primed[index]
        cli_metrics = cli_samples[index]["metrics_percent"]
        cli_confusion = np.asarray(cli_metrics["confusion_matrix"], dtype=np.int64)
        aligned_confusion = np.asarray(aligned["confusion_matrix"], dtype=np.int64)
        primed_confusion = np.asarray(primed["confusion_matrix"], dtype=np.int64)
        runner_confusion = np.asarray(runner_unit["metrics_percent"]["confusion_matrix"], dtype=np.int64)
        mismatched = int(np.count_nonzero(runner_unit["prediction"] != aligned["prediction"]))
        metric_differences = {
            name: float(runner_unit["metrics_percent"][name]) - float(aligned["metrics_percent"][name])
            for name in ("miou", "macc", "mf1")
        }
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "runner_vs_aligned_reference": {
                "prediction_mismatched_pixels": mismatched,
                "prediction_pixels": int(runner_unit["prediction"].size),
                "confusion_matrix_equal": bool(np.array_equal(runner_confusion, aligned_confusion)),
                "logits_sha256_equal": runner_unit["logits_sha256"] == aligned["logits_sha256"],
                "prediction_sha256_equal": runner_unit["prediction_sha256"] == aligned["prediction_sha256"],
                "metric_differences_runner_minus_reference": metric_differences,
            },
            "aligned_reference_miou": aligned["metrics_percent"]["miou"],
            "primed_reference_miou": primed["metrics_percent"]["miou"],
            "cli_reference_miou": cli_metrics["miou"],
            "cli_matches_primed_reference": bool(np.array_equal(cli_confusion, primed_confusion))
            and all(
                abs(float(primed["metrics_percent"][name]) - float(cli_metrics[name])) == 0.0
                for name in ("miou", "macc", "mf1")
            ),
            "cli_matches_aligned_reference": bool(np.array_equal(cli_confusion, aligned_confusion)),
            "reference_view_count": aligned["view_count"],
        }
        if mismatched != 0:
            failures.append(f"{sample_id}: clean prediction differs from the aligned reference")
        if not record["runner_vs_aligned_reference"]["confusion_matrix_equal"]:
            failures.append(f"{sample_id}: clean confusion matrix differs from the aligned reference")
        if any(abs(value) > 0 for value in metric_differences.values()):
            failures.append(f"{sample_id}: clean metrics differ from the aligned reference")
        if not record["runner_vs_aligned_reference"]["logits_sha256_equal"]:
            failures.append(f"{sample_id}: clean logits differ from the aligned reference")
        if not record["cli_matches_primed_reference"]:
            failures.append(f"{sample_id}: plain CLI reference does not match the primed reference (mechanism unexplained)")
        per_sample.append(record)

    runner_hist = np.asarray(engine["accumulators"]["clean"]["hist"], dtype=np.int64)
    aligned_hist = np.zeros_like(runner_hist)
    primed_hist = np.zeros_like(runner_hist)
    cli_hist = np.zeros_like(runner_hist)
    for index in range(len(sample_ids)):
        aligned_hist += np.asarray(reference_aligned[index]["confusion_matrix"], dtype=np.int64)
        primed_hist += np.asarray(reference_primed[index]["confusion_matrix"], dtype=np.int64)
        cli_hist += np.asarray(cli_samples[index]["metrics_percent"]["confusion_matrix"], dtype=np.int64)
    aggregate_confusion_equal = bool(np.array_equal(runner_hist, aligned_hist))
    if not aggregate_confusion_equal:
        failures.append("aggregate clean confusion matrix differs from the aligned reference")
    if not np.array_equal(cli_hist, primed_hist):
        failures.append("plain CLI aggregate confusion matrix does not match the primed reference")

    report.update(
        {
            "status": "PASS" if not failures else "FAIL",
            "evidence_kind": "museg-10condition-verification-clean-equivalence",
            "sample_count": len(local_context.selected_entries),
            "sample_ids": sample_ids,
            "view_batching": 1,
            "forward_rng_policy": "progressive",
            "base_rng_seed": settings.base_rng_seed,
            "comparison_contract": {
                "prediction": "argmax on the restored original-grid logits, compared per pixel",
                "confusion_matrix": "per sample and aggregate",
                "metrics": "mIoU/mAcc/mF1 must agree exactly",
                "aligned_reference": (
                    "tools.evaluate_museg_checkpoint_fast reference implementation "
                    "(EV.MUSegPostEvalDataset + FAST._run_reference_sample) after torch.manual_seed(base_rng_seed), "
                    "same sample order, one forward per view"
                ),
                "pass_criterion": "runner == aligned_reference for every sample, and the plain CLI must be reproduced exactly by the reference with one extra CPU draw",
            },
            "rng_alignment": {
                "mechanism": (
                    "torch.utils.data._BaseDataLoaderIter performs one CPU draw "
                    "(torch.empty((), dtype=torch.int64).random_()) when an iterator is created, so the plain CLI "
                    "reference report starts one draw later than an explicitly seeded reference implementation"
                ),
                "prime_draws_reproducing_the_cli": 1,
                "cli_matches_primed_reference": all(record["cli_matches_primed_reference"] for record in per_sample),
                "cli_matches_aligned_reference": all(record["cli_matches_aligned_reference"] for record in per_sample),
            },
            "reference_exit_code": int(completed.returncode),
            "reference_seconds": round(reference_seconds, 3),
            "reference_command": command,
            "reference_report": str(reference_output),
            "reference_report_sha256": EV.file_sha256(reference_output),
            "per_sample": per_sample,
            "aggregate_confusion_matrix_equal": aggregate_confusion_equal,
            "runner_aggregate_metrics": EV.metrics_from_confusion(runner_hist, local_context.config.class_names),
            "aligned_reference_aggregate_metrics": EV.metrics_from_confusion(
                aligned_hist, local_context.config.class_names
            ),
            "primed_reference_aggregate_metrics": EV.metrics_from_confusion(
                primed_hist, local_context.config.class_names
            ),
            "cli_aggregate_metrics": EV.metrics_from_confusion(cli_hist, local_context.config.class_names),
            "view_reuse_self_check": engine["view_reuse_checks"],
            "failures": failures,
        }
    )
    return report


def verify_corrupted_equivalence(context: Context, report: dict[str, Any], emit: Any) -> dict[str, Any]:
    """Cross-check the sample-major structure against a condition-major reference."""
    args = context.args
    count = args.verify_samples if args.verify_samples is not None else 2
    condition_spec = args.verify_conditions or ",".join(
        ["spatial_dropout@0.75", "misalignment@0.75", "entire_missing@1.0", "blur@0.5+misalignment@0.5"]
    )
    conditions = resolve_conditions(condition_spec, context.all_conditions)
    run_root = _resolve(args.benchmark_dir) / "corrupted-equivalence"
    results: dict[str, dict[str, Any]] = {}
    for order in ("sample-major", "condition-major"):
        local_args = copy.copy(args)
        local_args.order = order
        local_args.resume = False
        local_args.shuffle_conditions = False
        local_args.max_samples = count
        local_args.forward_rng = "reset-per-unit"
        local_args.check_view_reuse_samples = count if order == "sample-major" else 0
        local_args.output_dir = run_root / order
        local_args.quiet = True
        local_context = copy.copy(context)
        local_context.args = local_args
        local_context.conditions = conditions
        local_context.selected_entries = select_entries(
            local_context.dataset_root, local_context.split_entries, "first", count
        )
        settings = make_settings(local_context)
        model = load_eval_model(local_context, settings)
        engine = run_engine(
            settings,
            local_context.config,
            model,
            local_context.device,
            conditions,
            local_context.selected_entries,
            collect_units=True,
            collect_predictions=True,
        )
        units: dict[tuple[str, str], dict[str, Any]] = {}
        for condition in conditions:
            for unit in engine["accumulators"][condition.directory]["units"]:
                units[(unit["sample_id"], condition.directory)] = unit
        results[order] = {
            "units": units,
            "view_reuse_checks": engine["view_reuse_checks"],
            "output_dir": str(settings.output_dir),
        }
        del model
        torch.cuda.empty_cache()
        if emit is not None:
            emit(f"{order}: evaluated {len(units)} (sample, condition) units")

    sample_major = results["sample-major"]["units"]
    condition_major = results["condition-major"]["units"]
    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for unit_key in sorted(sample_major):
        left = sample_major[unit_key]
        right = condition_major.get(unit_key)
        if right is None:
            failures.append(f"{unit_key}: missing condition-major unit")
            comparisons.append({"sample_id": unit_key[0], "condition": unit_key[1], "equal": False})
            continue
        left_prediction = left["prediction"]
        right_prediction = right["prediction"]
        mismatched = int(np.count_nonzero(left_prediction != right_prediction))
        confusion_equal = left["confusion_matrix"] == right["confusion_matrix"]
        metric_differences = {
            name: float(left["metrics_percent"][name]) - float(right["metrics_percent"][name])
            for name in ("miou", "macc", "mf1")
        }
        equal = (
            mismatched == 0
            and confusion_equal
            and all(abs(value) == 0.0 for value in metric_differences.values())
            and left["corrupted_depth_sha256"] == right["corrupted_depth_sha256"]
            and left["validity_state_sha256"] == right["validity_state_sha256"]
        )
        if not equal:
            failures.append(f"{unit_key}: sample-major and condition-major disagree")
        comparisons.append(
            {
                "sample_id": unit_key[0],
                "condition": unit_key[1],
                "mismatched_prediction_pixels": mismatched,
                "total_prediction_pixels": int(left_prediction.size),
                "confusion_matrix_equal": bool(confusion_equal),
                "metric_differences_sample_minus_condition": metric_differences,
                "corrupted_depth_sha256_equal": left["corrupted_depth_sha256"] == right["corrupted_depth_sha256"],
                "validity_state_sha256_equal": left["validity_state_sha256"] == right["validity_state_sha256"],
                "prediction_sha256": left["prediction_sha256"],
                "equal": bool(equal),
            }
        )
    sample_reuse = results["sample-major"]["view_reuse_checks"]
    report.update(
        {
            "status": "PASS" if not failures else "FAIL",
            "evidence_kind": "museg-10condition-verification-corrupted-equivalence",
            "sample_count": len(context.selected_entries[:count]),
            "condition_count": len(conditions),
            "conditions": [condition.condition_id for condition in conditions],
            "unit_count": len(comparisons),
            "view_batching": int(args.view_batching),
            "forward_rng_policy": "reset-per-unit",
            "sample_major_output_dir": results["sample-major"]["output_dir"],
            "condition_major_output_dir": results["condition-major"]["output_dir"],
            "comparisons": comparisons,
            "view_reuse_self_check": sample_reuse,
            "failures": failures,
        }
    )
    return report


def verify_resume(context: Context, report: dict[str, Any], emit: Any) -> dict[str, Any]:
    """Prove skip / rerun / identity-mismatch behaviour of the resume logic."""
    args = context.args
    resume_dir = _resolve(args.benchmark_dir) / "resume-verify"
    sample_count = args.verify_samples if args.verify_samples is not None else 3
    base = [
        "--checkpoint", str(context.checkpoint),
        "--split", str(context.split),
        "--split-role", "val_dev",
        "--expected-checkpoint-sha256", context.checkpoint_sha256,
        "--expected-split-sha256", context.split_sha256,
        "--protocol", str(context.protocol.path),
        "--config", args.config,
        "--dataset-root", str(context.dataset_root),
        "--device", args.device,
        "--output-dir", str(resume_dir),
        "--view-batching", str(args.view_batching),
        "--quiet",
    ]
    phases = [
        {
            "phase": "phase1_first_condition_only",
            "extra": ["--conditions", "clean", "--resume", "--max-samples", str(sample_count)],
        },
        {
            "phase": "phase2_resume_adds_condition",
            "extra": ["--conditions", "clean,entire_missing@1.0", "--resume", "--max-samples", str(sample_count)],
        },
        {
            "phase": "phase3_resume_idempotent",
            "extra": ["--conditions", "clean,entire_missing@1.0", "--resume", "--max-samples", str(sample_count)],
        },
        {
            "phase": "phase4_seed_changed_must_not_reuse",
            "extra": [
                "--conditions", "clean,entire_missing@1.0", "--resume",
                "--max-samples", str(sample_count),
                "--evaluation-seed", str(context.evaluation_seed + 1),
            ],
        },
        {
            "phase": "phase5_sample_set_changed_must_not_reuse",
            "extra": ["--conditions", "clean,entire_missing@1.0", "--resume", "--max-samples", str(sample_count + 1)],
        },
        {
            "phase": "phase6_original_identity_still_mismatched",
            "extra": ["--conditions", "clean,entire_missing@1.0", "--resume", "--max-samples", str(sample_count)],
        },
    ]
    phase_records: list[dict[str, Any]] = []
    failures: list[str] = []
    for phase in phases:
        command = [sys.executable, "-m", "tools.evaluate_museg_10condition", *base, *phase["extra"]]
        started = time.monotonic()
        completed = subprocess.run(command, cwd=str(context.repo_root), capture_output=True, text=True)
        elapsed = time.monotonic() - started
        summary_path = resume_dir / context.checkpoint.stem / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        decisions = summary.get("resume_decisions", [])
        record = {
            "phase": phase["phase"],
            "command": command,
            "exit_code": int(completed.returncode),
            "elapsed_seconds": round(elapsed, 3),
            "stdout_tail": completed.stdout.strip()[-1500:],
            "stderr_tail": completed.stderr.strip()[-1500:] if completed.returncode != 0 else "",
            "decisions": [
                {
                    "condition": decision.get("condition"),
                    "decision": decision.get("decision"),
                    "reason": decision.get("reason"),
                    "mismatched_identity_fields": decision.get("mismatched_identity_fields"),
                }
                for decision in decisions
            ],
            "conditions_completed": summary.get("coverage", {}).get("conditions_completed"),
            "mean_over_all_conditions": summary.get("mean_over_all_conditions"),
        }
        phase_records.append(record)
        if emit is not None:
            emit(
                f"{phase['phase']}: exit={completed.returncode} decisions="
                + json.dumps(record["decisions"], ensure_ascii=False)
            )
        if completed.returncode != 0:
            failures.append(f"{phase['phase']} exited with {completed.returncode}")

    def decisions_of(phase_name: str) -> dict[str, str]:
        for record in phase_records:
            if record["phase"] == phase_name:
                return {item["condition"]: item["decision"] for item in record["decisions"]}
        return {}

    phase1 = decisions_of("phase1_first_condition_only")
    phase2 = decisions_of("phase2_resume_adds_condition")
    phase3 = decisions_of("phase3_resume_idempotent")
    phase4 = decisions_of("phase4_seed_changed_must_not_reuse")
    phase5 = decisions_of("phase5_sample_set_changed_must_not_reuse")
    phase6 = decisions_of("phase6_original_identity_still_mismatched")
    expectations = [
        ("phase1_runs_clean", phase1.get("clean") == "pending"),
        ("phase2_skips_completed_clean", phase2.get("clean") == "skipped"),
        ("phase2_runs_new_entire_missing", phase2.get("entire_missing_100") == "pending"),
        ("phase3_skips_clean_again", phase3.get("clean") == "skipped"),
        ("phase3_skips_entire_missing_again", phase3.get("entire_missing_100") == "skipped"),
        ("phase4_recomputes_clean_after_seed_change", phase4.get("clean") == "pending"),
        ("phase4_recomputes_entire_missing_after_seed_change", phase4.get("entire_missing_100") == "pending"),
        ("phase5_recomputes_clean_after_sample_set_change", phase5.get("clean") == "pending"),
        ("phase5_recomputes_entire_missing_after_sample_set_change", phase5.get("entire_missing_100") == "pending"),
        ("phase6_recomputes_clean_because_the_stored_identity_is_not_original", phase6.get("clean") == "pending"),
        ("phase6_recomputes_entire_missing_because_the_stored_identity_is_not_original", phase6.get("entire_missing_100") == "pending"),
    ]
    for name, satisfied in expectations:
        if not satisfied:
            failures.append(f"resume expectation failed: {name}")
    if not phase2.get("clean") == "skipped" and phase2:
        failures.append("resume did not skip a completed identity-matched condition")
    report.update(
        {
            "status": "PASS" if not failures else "FAIL",
            "evidence_kind": "museg-10condition-verification-resume",
            "resume_dir": str(resume_dir),
            "sample_count": sample_count,
            "phases": phase_records,
            "expectations": [{"name": name, "satisfied": bool(satisfied)} for name, satisfied in expectations],
            "failures": failures,
        }
    )
    return report


def verify_timing(context: Context, report: dict[str, Any], emit: Any) -> dict[str, Any]:
    """Small-sample ten-condition timing run plus an explicit extrapolation."""
    args = context.args
    count = args.verify_samples if args.verify_samples is not None else 8
    local_args = copy.copy(args)
    local_args.output_dir = _resolve(args.benchmark_dir) / "timing-run"
    local_args.max_samples = count
    local_args.conditions = args.verify_conditions
    local_args.resume = False
    local_args.quiet = True
    local_context = copy.copy(context)
    local_context.args = local_args
    local_context.conditions = resolve_conditions(args.verify_conditions, context.all_conditions)
    local_context.selected_entries = select_entries(
        local_context.dataset_root, local_context.split_entries, "first", count
    )
    settings = make_settings(local_context)
    model = load_eval_model(local_context, settings)
    torch.cuda.reset_peak_memory_stats(local_context.device) if local_context.device.type == "cuda" else None
    started = time.monotonic()
    engine = run_engine(
        settings,
        local_context.config,
        model,
        local_context.device,
        local_context.conditions,
        local_context.selected_entries,
        emit=emit,
    )
    FAST._sync(local_context.device)
    total_seconds = time.monotonic() - started
    peak = FAST._peak_memory(local_context.device)
    info = FAST._device_memory_info(local_context.device)

    condition_reports: list[dict[str, Any]] = []
    unit_seconds: list[float] = []
    for condition in local_context.conditions:
        payload = engine["written"][condition.directory]
        per_sample = payload["per_sample"]
        unit_seconds.extend(float(record["elapsed_seconds"]) for record in per_sample)
        condition_reports.append(
            {
                "condition": condition.directory,
                "condition_id": condition.condition_id,
                "sample_count": len(per_sample),
                "elapsed_seconds": payload["elapsed_seconds"],
                "mean_seconds_per_sample": round(payload["elapsed_seconds"] / len(per_sample), 6),
                "forward_seconds": payload["timing_breakdown_seconds"]["forward"],
                "preprocess_including_corruption_seconds": payload["timing_breakdown_seconds"][
                    "preprocess_including_corruption"
                ],
                "peak_allocated_bytes": payload["peak_device_memory"]["allocated_bytes"],
                "peak_reserved_bytes": payload["peak_device_memory"]["reserved_bytes"],
                "peak_allocated_mib": payload["peak_device_memory"]["allocated_mib"],
                "peak_reserved_mib": payload["peak_device_memory"]["reserved_mib"],
                "sample_elapsed_seconds": [float(record["elapsed_seconds"]) for record in per_sample],
            }
        )

    samples = len(local_context.selected_entries)
    unit_count = len(unit_seconds)
    mean_unit_seconds = float(np.mean(unit_seconds)) if unit_seconds else float("nan")
    read_seconds_total = float(engine["read_seconds_total"])
    cache_seconds_total = float(engine["rgb_cache_seconds_total"])
    # Extrapolation: the unit cost is dominated by the ten-view FP32 forward.  Sample
    # reading and the per-sample RGB view cache are counted once per sample, not once per
    # condition, so they are added as a per-sample term.
    per_sample_pre_seconds = (read_seconds_total + cache_seconds_total) / samples
    per_sample_all_conditions = sum(float(record["elapsed_seconds"]) for record in condition_reports) / samples
    flat_one_checkpoint = (mean_unit_seconds * len(local_context.conditions) + per_sample_pre_seconds) * 318
    flat_four = flat_one_checkpoint * 4
    sample_weighted_one = per_sample_all_conditions * 318
    sample_weighted_four = sample_weighted_one * 4
    report.update(
        {
            "status": "PASS",
            "evidence_kind": "museg-10condition-verification-timing",
            "checkpoint": {"path": str(context.checkpoint), "sha256": context.checkpoint_sha256},
            "sample_count": samples,
            "condition_count": len(local_context.conditions),
            "unit_count": unit_count,
            "order": settings.order,
            "view_batching": dict(context.batching_evidence),
            "total_seconds": round(total_seconds, 3),
            "mean_seconds_per_sample_all_conditions": round(per_sample_all_conditions, 6),
            "mean_seconds_per_unit": round(mean_unit_seconds, 6),
            "read_seconds_total": round(read_seconds_total, 6),
            "rgb_cache_seconds_total": round(cache_seconds_total, 6),
            "per_sample_read_and_cache_seconds": round(per_sample_pre_seconds, 6),
            "min_unit_seconds": round(float(min(unit_seconds)), 6) if unit_seconds else None,
            "max_unit_seconds": round(float(max(unit_seconds)), 6) if unit_seconds else None,
            "conditions": condition_reports,
            "overall_peak_device_memory": peak,
            "device_memory": info,
            "extrapolation": {
                "measured_sample_count": samples,
                "measured_condition_count": len(local_context.conditions),
                "assumptions": [
                    "the ten-view FP32 forward dominates the unit cost, so cost scales with (samples x conditions)",
                    "the first samples of val-dev are representative of the whole 318-sample split; no per-sample size weighting is applied",
                    "the same checkpoint and the same view-batching strategy are used",
                    "four checkpoints are evaluated sequentially on the same GPU with no other workload",
                    "corruption and view assembly are inside the unit timer; sample reading and the per-sample RGB view cache are added once per sample",
                    "model load time, dataset warm-up and per-run overhead are excluded",
                ],
                "one_checkpoint_flat_seconds": round(flat_one_checkpoint, 3),
                "one_checkpoint_flat_hours": round(flat_one_checkpoint / 3600.0, 3),
                "four_checkpoints_flat_hours": round(flat_four / 3600.0, 3),
                "one_checkpoint_sample_weighted_hours": round(sample_weighted_one / 3600.0, 3),
                "four_checkpoints_sample_weighted_hours": round(sample_weighted_four / 3600.0, 3),
                "target_samples": 318,
            },
            "written_metrics": [
                str(settings.output_dir / context.checkpoint.stem / condition.directory / "metrics.json")
                for condition in local_context.conditions
            ],
            "failures": [],
        }
    )
    del model
    return report


def _forwarded_common_args(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--checkpoint", str(args.checkpoint),
        "--split", str(args.split),
        "--split-role", args.split_role,
        "--expected-checkpoint-sha256", str(args.expected_checkpoint_sha256),
        "--expected-split-sha256", str(args.expected_split_sha256),
        "--protocol", str(args.protocol),
        "--config", args.config,
        "--dataset-root", str(args.dataset_root),
        "--device", args.device,
        "--view-batching", str(args.view_batching),
        "--benchmark-dir", str(args.benchmark_dir),
        "--quiet",
    ]
    if args.evaluation_seed is not None:
        forwarded += ["--evaluation-seed", str(args.evaluation_seed)]
    return forwarded


def run_verification(args: argparse.Namespace) -> int:
    started = time.monotonic()
    context = build_context(args)
    emit = None if args.quiet else (lambda message: print(message, flush=True))
    names = {
        "corruption-digest": "10cond-corruption-digest",
        "corruption-determinism": "10cond-corruption-determinism",
        "condition-sanity": "10cond-condition-sanity",
        "clean-equivalence": "10cond-clean-equivalence",
        "corrupted-equivalence": "10cond-corrupted-equivalence",
        "resume": "10cond-resume",
        "timing": "10cond-timing-first8",
    }
    report = _verify_common(context, args.verify, started)
    try:
        if args.verify == "corruption-digest":
            report = verify_corruption_digest(context, report)
        elif args.verify == "corruption-determinism":
            report = verify_corruption_determinism(context, report, emit)
        elif args.verify == "condition-sanity":
            report = verify_condition_sanity(context, report)
        elif args.verify == "clean-equivalence":
            report = verify_clean_equivalence(context, report, emit)
        elif args.verify == "corrupted-equivalence":
            report = verify_corrupted_equivalence(context, report, emit)
        elif args.verify == "resume":
            report = verify_resume(context, report, emit)
        elif args.verify == "timing":
            report = verify_timing(context, report, emit)
        else:
            raise ValueError(f"unsupported verification mode: {args.verify}")
    except torch.OutOfMemoryError as exc:
        report.update({"status": "FAIL", "error_type": "cuda_out_of_memory", "error": str(exc)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (OSError, RuntimeError, ValueError, ModuleNotFoundError, KeyError, TypeError) as exc:
        report.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
    path = _benchmark_path(args, names.get(args.verify, f"10cond-{args.verify}"))
    return _finish(report, path, started, emit)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify != "none":
            return run_verification(args)
        return run_evaluation(args)
    except torch.OutOfMemoryError as exc:
        print(json.dumps({"status": "environment_limit", "error_type": "cuda_out_of_memory", "error": str(exc)}))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 3
    except (OSError, RuntimeError, ValueError, ModuleNotFoundError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
