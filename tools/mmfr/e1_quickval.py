#!/usr/bin/env python3
"""MMFR E1 Batch 1A Quick-Val: single-view ``original-full`` scoring under four frozen Depth conditions.

Why this file exists
--------------------
The frozen Quick-Val text (``e1_batch1_protocol.md`` §11, ``e1_screening_plan.md`` §8,
``r_oe_lite_design.md`` §12) fixes the evaluation as *318 complete ``val-dev`` samples,
``original-full``, scale 1.0, no flip, FP32, TF32 off, original Label grid, fixed final
checkpoint only* for the conditions ``clean``, ``entire_missing@1.0``,
``spatial_dropout@0.75`` and ``misalignment@0.75``.

No existing frozen runner satisfies both halves of that sentence:

* ``tools/evaluate_museg_checkpoint`` provides the ``original-full`` geometry but has no
  condition/corruption scoring entry at all;
* ``tools/evaluate_museg_10condition`` scores conditions but is hard-wired to the
  ten-view ``msflip-whole-original-grid-v1`` protocol (``FROZEN_CONSTANTS`` scales and
  ``views_per_scale`` are not read from ``--protocol``).

This module was therefore written under an explicit user authorization for a *new*
single-view condition-scoring entry. It is intentionally thin and reuses the frozen
stack instead of reimplementing any numeric step:

``tools.evaluate_museg_checkpoint`` (``EV``)
    model construction and strict checkpoint restore (``load_model``), the FP32/TF32-off
    forward contract (``configure_fp32_forward``), the frozen ``original-full`` input
    contract (native-resolution RGB/Depth, Depth replicated to three channels, Depth
    normalization ``(d/255 - 0.48)/0.28``, RGB normalization from the run config, label
    ``-1`` with background mapped to the ignore index), logits restoring to the untouched
    original Label grid (``restore_logits_to_metric_grid``) and the confusion/metric
    helpers (``update_confusion``, ``metrics_from_confusion``, ``split_entries``,
    ``file_sha256``).
``tools.evaluate_museg_10condition`` (``E10``)
    the frozen condition definitions (``build_conditions(FROZEN_EVALUATION)``,
    ``resolve_conditions``), the frozen per-unit corruption seeding and application
    (``corrupt_depth``, whose seed words depend only on the evaluation seed, the condition
    index and the sample id) and the sample reader (``read_sample``).
``tools.evaluate_museg_checkpoint_fast`` (``FAST``)
    the frozen RNG capture/restore and CUDA peak-memory helpers that the ten-view
    evaluator uses to keep every unit's forward identical.

What this file adds
-------------------
Only single-view scheduling and the four-condition bookkeeping. Nothing about the
``original-full`` input contract, the forward, the logits restore, the metric computation,
the condition set or the corruption math is reimplemented here.

Forward RNG policy (``--forward-rng``)
--------------------------------------
The Ham decoder is built with ``RAND_INIT=True``, so ``_build_bases`` consumes the global
RNG on *every* forward and re-rolls its NMF bases each call. The frozen ten-view evaluator
therefore restores one captured RNG state before every unit forward
(``--forward-rng reset-per-unit``, its default). This module keeps that default so every
``(sample, condition)`` unit is scored under an identical decoder draw, which is what makes
two candidate checkpoints comparable. ``progressive`` is provided only so the code path can
be compared against the frozen evaluator, which does not reset.

Status
------
This entry point is **new** and carries no prior frozen qualification. Every artefact it
writes records ``evaluator_identity="e1-quickval-original-full-single-view-v1"`` and a
``deviation`` block so it can never be mistaken for the frozen ten-view evaluator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import evaluate_museg_checkpoint as EV  # noqa: E402
from tools import evaluate_museg_10condition as E10  # noqa: E402
from tools import evaluate_museg_checkpoint_fast as FAST  # noqa: E402
from utils.dataloader.mmfr_training import normalize_sample_id  # noqa: E402

SCHEMA_VERSION = "mmfr-e1-quickval-single-view-v1"
EVALUATOR_IDENTITY = "e1-quickval-original-full-single-view-v1"
SUMMARY_SCHEMA_VERSION = "mmfr-e1-quickval-comparison-v1"

QUICKVAL_CONDITIONS = (
    "clean",
    "spatial_dropout@0.75",
    "misalignment@0.75",
    "entire_missing@1.0",
)
HARD_CONDITIONS = ("entire_missing_100", "spatial_dropout_075", "misalignment_075")
EXPECTED_SAMPLE_COUNT = 318
EXPECTED_EVALUATION_SEED = 2026091401

DEV = {
    "reason": "frozen Quick-Val text requires original-full single view plus four conditions",
    "frozen_original_full_evaluator": "tools/evaluate_museg_checkpoint --geometry original-full (no condition scoring)",
    "frozen_condition_evaluator": "tools/evaluate_museg_10condition (hard-wired ten-view msflip-whole-original-grid-v1)",
    "this_entry": EVALUATOR_IDENTITY,
    "qualification_status": "new-entry-point-no-prior-frozen-qualification",
}


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def canonical_digest(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_original_full_input(
    rgb_uint8: np.ndarray,
    depth_uint8: np.ndarray,
    label: np.ndarray,
    norm_mean: Sequence[float],
    norm_std: Sequence[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-express the frozen ``original-full`` input contract on a batched device tensor.

    The six numeric steps below are exactly ``EV.MUSegPostEvalDataset.__getitem__`` for the
    non-resize, non-msflip branch: Depth replicated to three channels and normalized with
    the frozen ``0.48/0.28`` constants, RGB normalized with the run config, label already
    shifted by ``read_sample``. ``self_check_against_frozen_dataset`` proves the equality.
    """
    depth_three = cv2_merge_three(depth_uint8)
    depth_float = depth_three.astype(np.float32) / 255.0
    depth_float = (depth_float - E10.FROZEN_CONSTANTS["depth_normalization_mean"]) / E10.FROZEN_CONSTANTS[
        "depth_normalization_std"
    ]
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    rgb_float = (rgb_float - np.asarray(norm_mean, dtype=np.float32)) / np.asarray(norm_std, dtype=np.float32)
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_float.transpose(2, 0, 1))).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_float.transpose(2, 0, 1))).unsqueeze(0).to(device)
    label_tensor = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0).to(device)
    return rgb_tensor, depth_tensor, label_tensor


def cv2_merge_three(array: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.merge([array, array, array])


def self_check_against_frozen_dataset(
    dataset_root: Path,
    split_path: Path,
    config: Any,
    channel_order: str,
    *,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Prove the new input assembly equals ``EV.MUSegPostEvalDataset`` ``original-full``.

    Clean Depth is used because the clean condition is a strict no-op copy of raw Depth, so
    this compares exactly the tensors the frozen evaluator would have forwarded.
    """
    entries = EV.split_entries(split_path)
    if sample_limit is not None:
        entries = entries[:sample_limit]
    dataset = EV.MUSegPostEvalDataset(dataset_root, entries, "original-full", config, channel_order)
    inspected = 0
    for index, entry in enumerate(entries):
        sample_id, rgb_uint8, depth_uint8, label = E10.read_sample(dataset_root, entry, channel_order)
        reference = dataset[index]
        rgb_tensor, depth_tensor, label_tensor = build_original_full_input(
            rgb_uint8,
            depth_uint8,
            label,
            config.norm_mean,
            config.norm_std,
            torch.device("cpu"),
        )
        if str(reference["sample_id"]) != sample_id:
            raise RuntimeError(f"sample id mismatch at index {index}: {reference['sample_id']!r} vs {sample_id!r}")
        for name, mine, theirs in (
            ("rgb", rgb_tensor[0], reference["rgb"]),
            ("depth", depth_tensor[0], reference["depth"]),
            ("label", label_tensor[0], reference["label"]),
        ):
            if not torch.equal(mine, theirs):
                raise RuntimeError(f"{name} tensor differs from the frozen original-full dataset at {sample_id}")
        inspected += 1
    return {
        "checked_samples": inspected,
        "compared": ["rgb", "depth", "label"],
        "result": "exact-equal",
        "reference": "tools.evaluate_museg_checkpoint.MUSegPostEvalDataset geometry=original-full",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--candidate", default=None, help="free-form label recorded in the artefacts")
    parser.add_argument("--config", default="local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_C0")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--conditions", default=",".join(QUICKVAL_CONDITIONS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--order", choices=("condition-major", "sample-major"), default="condition-major")
    parser.add_argument("--forward-rng", choices=("reset-per-unit", "progressive"), default="reset-per-unit")
    parser.add_argument(
        "--base-rng-seed",
        type=int,
        default=None,
        help="torch seed applied after model construction; defaults to the frozen evaluation seed",
    )
    parser.add_argument("--self-check", action="store_true", help="only run the frozen-dataset input equivalence check")
    parser.add_argument("--self-check-samples", type=int, default=None)
    parser.add_argument("--gap-candidate", default="C0")
    parser.add_argument("--gap-experimental", default="F-lite")
    parser.add_argument("--compare", nargs="+", default=None, help="summary.json paths, in candidate order")
    parser.add_argument("--compare-output", type=Path, default=None)
    return parser.parse_args(argv)


def load_run_config(module_name: str) -> Any:
    module = importlib.import_module(module_name)
    return module.C


def evaluate(
    args: argparse.Namespace,
    conditions: Sequence[E10.Condition],
    entries: Sequence[str],
    config: Any,
    checkpoint_sha256: str,
    split_sha256: str,
    channel_order: str,
) -> dict[str, Any]:
    device = torch.device(args.device)
    EV.configure_fp32_forward(device)
    model = EV.load_model(config, args.checkpoint, device)
    model.eval()
    num_classes = int(config.num_classes)
    background = int(config.background)
    class_names = list(config.class_names)
    evaluation_seed = EXPECTED_EVALUATION_SEED

    # Frozen convention (E10.load_eval_model): the torch generator is seeded *after* model
    # construction with the protocol evaluation seed, so the stochastic Ham NMF bases are
    # reproducible and identical for both candidates instead of inheriting whatever state
    # model construction happened to consume.
    base_rng_seed = evaluation_seed if args.base_rng_seed is None else int(args.base_rng_seed)
    torch.manual_seed(base_rng_seed)
    torch.cuda.manual_seed_all(base_rng_seed)

    if args.forward_rng == "reset-per-unit":
        base_rng = FAST._capture_rng(device)
    else:
        base_rng = None

    condition_definition_sha256 = E10.json_sha256([condition.definition() for condition in conditions]) if hasattr(
        E10, "json_sha256"
    ) else canonical_digest([json.dumps(condition.definition(), sort_keys=True) for condition in conditions])

    started_at = dt.datetime.now(dt.timezone.utc)
    condition_payloads: dict[str, dict[str, Any]] = {}
    call_order = list(conditions)
    if args.order == "sample-major":
        raise SystemExit("sample-major ordering is not implemented in this thin entry point; use condition-major")

    for completed, condition in enumerate(call_order, start=1):
        hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        per_sample: list[dict[str, Any]] = []
        depth_digests: list[str] = []
        condition_started = time.perf_counter()
        forward_seconds_total = 0.0
        FAST._reset_peak_memory(device)
        for entry in entries:
            sample_id, rgb_uint8, depth_uint8, label = E10.read_sample(args.dataset_root, entry, channel_order)
            normalized_id = normalize_sample_id(entry)
            outcome = E10.corrupt_depth(condition, rgb_uint8, depth_uint8, evaluation_seed, normalized_id)
            rgb_tensor, depth_tensor, label_tensor = build_original_full_input(
                rgb_uint8,
                outcome.depth,
                label,
                config.norm_mean,
                config.norm_std,
                device,
            )
            if base_rng is not None:
                FAST._restore_rng(base_rng, device)
            unit_started = time.perf_counter()
            with torch.inference_mode():
                logits = model(rgb_tensor, depth_tensor)
                logits = EV.restore_logits_to_metric_grid(logits, label_tensor)
            FAST._sync(device)
            forward_seconds_total += time.perf_counter() - unit_started
            if tuple(int(value) for value in logits.shape[-2:]) != tuple(int(value) for value in label_tensor.shape[-2:]):
                raise RuntimeError("restored logits are not on the original Label grid")
            EV.update_confusion(hist, logits, label_tensor, num_classes, background)
            sample_hist = np.zeros((num_classes, num_classes), dtype=np.int64)
            EV.update_confusion(sample_hist, logits, label_tensor, num_classes, background)
            sample_metrics = EV.metrics_from_confusion(sample_hist, class_names)
            depth_digests.append(outcome.depth_sha256())
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "entry": entry,
                    "miou": sample_metrics["miou"],
                    "macc": sample_metrics["macc"],
                    "mf1": sample_metrics["mf1"],
                    "corrupted_depth_sha256": outcome.depth_sha256(),
                    "validity_state_sha256": outcome.state_sha256(),
                    "strict_no_op": bool(outcome.strict_no_op),
                    "rng_constructed": bool(outcome.rng_constructed),
                }
            )
            del rgb_tensor, depth_tensor, label_tensor, logits, outcome
        peak = FAST._peak_memory(device)
        metrics = EV.metrics_from_confusion(hist, class_names)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "evaluator_identity": EVALUATOR_IDENTITY,
            "deviation": DEV,
            "condition": {
                "condition_id": condition.condition_id,
                "directory": condition.directory,
                "group": condition.group,
                "index": condition.index,
                "specs": condition.spec_records() if hasattr(condition, "spec_records") else [],
            },
            "condition_definition_sha256": condition_definition_sha256,
            "call_index": completed,
            "sample_count": len(per_sample),
            "completed": len(per_sample) == len(entries),
            "identity": {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "split": str(args.split),
                "split_sha256": split_sha256,
                "split_role": "val_dev",
                "config_module": args.config,
                "candidate": args.candidate,
                "evaluation_seed": evaluation_seed,
                "official_test_included": False,
            },
            "view_protocol": {
                "geometry": "original-full",
                "scale": 1.0,
                "flip": False,
                "view_count": 1,
                "padding": "none",
                "input_contract": "native-resolution RGB/Depth, Depth replicated to three channels",
                "forward_precision": "fp32",
                "logits_fusion_precision": "not-applicable-single-view",
                "metric_grid": "original label grid (no crop, no pad)",
                "forward_rng_policy": args.forward_rng,
                "base_rng_seed": base_rng_seed,
            },
            "randomness": {
                "corruption": {
                    "engine": "numpy.PCG64 + SeedSequence",
                    "seed_words": ["evaluation_seed", "condition_index_0_based", "sample_id_sha256_u32_be_0..3"],
                    "unit": "(sample, condition)",
                    "shared_progressing_generator": False,
                },
                "model_forward": {
                    "base_rng_seed": base_rng_seed,
                    "base_rng_seed_source": (
                        "frozen protocol evaluation seed"
                        if args.base_rng_seed is None
                        else "explicit --base-rng-seed"
                    ),
                    "policy": args.forward_rng,
                    "rationale": (
                        "the frozen DFormerv2-S Ham decoder re-draws its NMF bases on every forward; replaying one "
                        "fixed base state before each unit forward makes the score independent of call order and "
                        "identical across candidate checkpoints"
                    ),
                },
            },
            "metrics_percent": metrics,
            "confusion_matrix": hist.tolist(),
            "per_sample": per_sample,
            "corrupted_depth_aggregate_sha256": canonical_digest(depth_digests),
            "timings": {
                "condition_seconds": round(time.perf_counter() - condition_started, 6),
                "forward_seconds": round(forward_seconds_total, 6),
            },
            "peak_device_memory": peak,
            "runtime": {
                "device": str(device),
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        target = args.output_dir / condition.directory / "metrics.json"
        atomic_write_json(target, payload)
        condition_payloads[condition.directory] = payload
        print(
            f"[{completed}/{len(call_order)}] {condition.condition_id} "
            f"miou={metrics['miou']} elapsed={payload['timings']['condition_seconds']}s -> {target}",
            flush=True,
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "deviation": DEV,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "split": str(args.split),
            "split_sha256": split_sha256,
            "config_module": args.config,
            "candidate": args.candidate,
            "output_dir": str(args.output_dir),
            "official_test_included": False,
        },
        "conditions": {
            name: {
                "miou": payload["metrics_percent"]["miou"],
                "macc": payload["metrics_percent"]["macc"],
                "mf1": payload["metrics_percent"]["mf1"],
                "sample_count": payload["sample_count"],
                "metrics_path": str(args.output_dir / name / "metrics.json"),
                "corrupted_depth_aggregate_sha256": payload["corrupted_depth_aggregate_sha256"],
            }
            for name, payload in condition_payloads.items()
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def compare(summary_paths: Sequence[Path], gap_candidate: str, gap_experimental: str) -> dict[str, Any]:
    if len(summary_paths) != 2:
        raise SystemExit("--compare requires exactly two summary.json paths (control, experimental)")
    control = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    experimental = json.loads(summary_paths[1].read_text(encoding="utf-8"))
    control_conditions = control["conditions"]
    experimental_conditions = experimental["conditions"]
    shared = [name for name in control_conditions if name in experimental_conditions]
    if not shared:
        raise SystemExit("the two summaries share no condition")
    per_condition = {}
    for name in shared:
        control_miou = float(control_conditions[name]["miou"])
        experimental_miou = float(experimental_conditions[name]["miou"])
        per_condition[name] = {
            "control_miou": control_miou,
            "experimental_miou": experimental_miou,
            "delta_pp": round(experimental_miou - control_miou, 4),
        }
    hard_present = [name for name in HARD_CONDITIONS if name in per_condition]
    if len(hard_present) != len(HARD_CONDITIONS):
        raise SystemExit(f"missing hard conditions in the comparison: {sorted(set(HARD_CONDITIONS) - set(hard_present))}")
    control_hard = sum(per_condition[name]["control_miou"] for name in hard_present) / len(hard_present)
    experimental_hard = sum(per_condition[name]["experimental_miou"] for name in hard_present) / len(hard_present)
    delta_f = experimental_hard - control_hard
    clean_present = "clean" in per_condition
    clean_delta = per_condition["clean"]["delta_pp"] if clean_present else None
    per_hard_ok = all(per_condition[name]["delta_pp"] >= -0.50 for name in hard_present)
    any_hard_stop = any(per_condition[name]["delta_pp"] < -1.00 for name in hard_present)
    if delta_f <= 0.0 or (clean_delta is not None and clean_delta < -0.50) or any_hard_stop:
        verdict = "stop"
    elif delta_f >= 0.50 and (clean_delta is None or clean_delta >= -0.25) and per_hard_ok:
        verdict = "promote"
    else:
        verdict = "inconclusive"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "candidates": {
            "control": {"label": gap_candidate, "summary": str(summary_paths[0]), "config_module": control["identity"].get("config_module"), "checkpoint_sha256": control["identity"].get("checkpoint_sha256")},
            "experimental": {"label": gap_experimental, "summary": str(summary_paths[1]), "config_module": experimental["identity"].get("config_module"), "checkpoint_sha256": experimental["identity"].get("checkpoint_sha256")},
        },
        "per_condition": per_condition,
        "hard_conditions": hard_present,
        "m3_hard": {
            "definition": "(entire_missing@1.0 + spatial_dropout@0.75 + misalignment@0.75) / 3",
            "control": round(control_hard, 4),
            "experimental": round(experimental_hard, 4),
            "delta_pp": round(delta_f, 4),
        },
        "gate": {
            "source": "liu-test-exp/MMFR/MMFR_v4_1_blueprint_and_reference_package_2026-09-20/01_research/e1_batch1_protocol.md §11",
            "promote_rule": "delta_M3_hard >= +0.50 pp; clean >= -0.25 pp; every hard condition >= -0.50 pp",
            "stop_rule": "delta_M3_hard <= 0; or clean < -0.50 pp; or any hard condition < -1.00 pp",
            "otherwise": "inconclusive",
            "observed": {
                "delta_M3_hard_pp": round(delta_f, 4),
                "clean_delta_pp": clean_delta,
                "hard_deltas_pp": {name: per_condition[name]["delta_pp"] for name in hard_present},
            },
            "verdict": verdict,
        },
        "official_test_included": False,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        report = compare([Path(item) for item in args.compare], args.gap_candidate, args.gap_experimental)
        target = args.compare_output or Path("quickval-comparison.json")
        atomic_write_json(target, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return 0

    if args.self_check:
        config = load_run_config(args.config)
        channel_order = str(getattr(config, "channel_order", "RGB"))
        if args.dataset_root is None or args.split is None:
            raise SystemExit("--self-check requires --dataset-root and --split")
        result = self_check_against_frozen_dataset(
            args.dataset_root,
            args.split,
            config,
            channel_order,
            sample_limit=args.self_check_samples,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return 0

    missing = [
        name
        for name in ("checkpoint", "dataset_root", "split", "output_dir", "expected_checkpoint_sha256", "expected_split_sha256")
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join('--' + item.replace('_', '-') for item in missing)}")

    config = load_run_config(args.config)
    channel_order = str(getattr(config, "channel_order", "RGB"))

    checkpoint_sha256 = EV.file_sha256(args.checkpoint)
    if checkpoint_sha256.lower() != str(args.expected_checkpoint_sha256).lower():
        raise SystemExit(f"checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, got {checkpoint_sha256}")
    split_sha256 = EV.file_sha256(args.split)
    if split_sha256.lower() != str(args.expected_split_sha256).lower():
        raise SystemExit(f"split SHA-256 mismatch: expected {args.expected_split_sha256}, got {split_sha256}")
    entries = EV.split_entries(args.split)
    if len(entries) != EXPECTED_SAMPLE_COUNT:
        raise SystemExit(f"val-dev split has {len(entries)} samples, expected {EXPECTED_SAMPLE_COUNT}")

    all_conditions = E10.build_conditions(E10.FROZEN_EVALUATION)
    conditions = E10.resolve_conditions(args.conditions, all_conditions)
    print(
        f"E1 Quick-Val single-view: candidate={args.candidate} conditions={[c.condition_id for c in conditions]} "
        f"samples={len(entries)} checkpoint={checkpoint_sha256[:16]}…",
        flush=True,
    )
    evaluate(args, conditions, entries, config, checkpoint_sha256, split_sha256, channel_order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
