#!/usr/bin/env python3
"""``MMFR-Oracle-A`` Validity-Aware Pairwise Geometry evaluation runner.

This runner is inference-only.  It never trains, never adds parameters and never touches
the frozen corruption basis, the RGB preprocessing, the Depth preprocessing, the
normalization, the decoder, the loss, the checkpoint selector or the evaluator.  It
imports the frozen evaluator (``tools.evaluate_museg_checkpoint``), its verified
accelerated path (``tools.evaluate_museg_checkpoint_fast``) and the frozen 10-condition
runner (``tools.evaluate_museg_10condition``) purely as libraries, and reuses their
corruption pipeline, view construction, logit fusion, grid restoration and metrics.

What it adds is exactly one thing: before a unit's forward passes it injects the
**final** Depth validity state into DFormerv2's geometry path through the additive
``geometry_oracle`` interface, once per variant.

Variants
--------
``original``      the checkpoint's own path, no oracle argument at all.
``strict``        strict validity pair gate ``rho_ij = r_i * r_j``.
``aggregated``    aggregated validity pair gate ``rho_ij = c_i * c_j``.
``geometry_off``  Depth-Geometry-Off control.
``noop_continuous`` / ``noop_strict``  identity controls that must reproduce ``original``
                  bitwise.

Determinism
-----------
The frozen Ham decoder re-draws its NMF bases with ``torch.rand`` on every forward, so the
runner restores one captured base RNG state before every unit, exactly like the frozen
10-condition runner.  Every variant of a unit therefore sees the identical RNG stream.

Usage
-----
    python -m tools.mmfr.run_oracle_a --checkpoint <pth> --output-dir <dir> --resume
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import tools.evaluate_museg_10condition as E10
import tools.evaluate_museg_checkpoint as EV
import tools.evaluate_museg_checkpoint_fast as FAST
from models.encoders.DFormerv2 import build_oracle_stage_reliability
from tools.mmfr.oracle_a_core import (
    REQUIRED_VARIANTS,
    GeometryOracleModel,
    OracleVariant,
    resolve_variants,
)
from utils.dataloader.oracle_a_validity import build_view_invalidity_plan, raw_invalidity, validity_summary

SCHEMA_VERSION = "mmfr-oracle-a-evidence-v1"
STAGE_DIVISORS = (4, 8, 16, 32)
DEFAULT_OUTPUT_DIR = "experiments/MMFR_OracleA/eval"
DEFAULT_CONFIG = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3"
DEFAULT_PROTOCOL = "protocols/mmfr-a2-train-integration-v3.template.json"
DEFAULT_CONDITIONS = "clean,spatial_dropout@0.75,entire_missing@1.0"


# --------------------------------------------------------------------------------------
# helpers
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


def stage_coverage(invalidity_view: np.ndarray, padded_size_hw: Sequence[int]) -> dict[str, Any]:
    """Per-stage reliability summaries for one view, using the model's own aggregation."""
    padded_height, padded_width = int(padded_size_hw[0]), int(padded_size_hw[1])
    tensor = torch.from_numpy(np.ascontiguousarray(invalidity_view))
    stages: dict[str, Any] = {}
    for divisor in STAGE_DIVISORS:
        height, width = padded_height // divisor, padded_width // divisor
        strict = build_oracle_stage_reliability(
            {"mode": "strict", "invalidity": tensor, "depth_geometry_off": False},
            (height, width),
            tensor,
        )
        continuous = build_oracle_stage_reliability(
            {"mode": "continuous", "invalidity": tensor, "depth_geometry_off": False},
            (height, width),
            tensor,
        )
        stages[f"stage_{divisor}"] = {
            "grid_hw": [height, width],
            "strict_invalid_token_fraction": float((strict == 0).float().mean().item()),
            "continuous_reliability_mean": float(continuous.mean().item()),
            "continuous_reliability_min": float(continuous.min().item()),
        }
    return stages


def location_group(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    if len(parts) < 4 or any(not part for part in parts[:4]):
        raise ValueError(f"sample id cannot form a location group: {sample_id!r}")
    return "-".join(parts[:4])


# --------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------


class UnitBuilder:
    """Builds one ``(sample, condition)`` unit once and shares it across variants."""

    def __init__(self, dataset_root: Path, channel_order: str, config: Any, evaluation_seed: int) -> None:
        self.dataset_root = dataset_root
        self.channel_order = channel_order
        self.config = config
        self.evaluation_seed = evaluation_seed

    def build(self, entry: str, condition: E10.Condition) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
        sample_id, rgb, depth, label = E10.read_sample(self.dataset_root, entry, self.channel_order)
        # The frozen Main-Val runner derives the corruption seed words from
        # ``normalize_sample_id(entry)`` -- the split entry (``RGB/<id>.jpg``) -- not from
        # ``Path(entry).stem``.  Reproducing that exact convention is required for the
        # corrupted Depth of ``original`` to be the same array the frozen Main-Val
        # evidence used.
        corruption = E10.corrupt_depth(
            condition, rgb, depth, self.evaluation_seed, E10.normalize_sample_id(entry)
        )
        rgb_cache = E10.build_rgb_view_cache(rgb, self.config)
        views = E10.assemble_views(rgb_cache, corruption.depth)
        del rgb, depth
        sample_like = {
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "views": views,
            "sample_id": sample_id,
            "original_height": int(label.shape[0]),
            "original_width": int(label.shape[1]),
        }
        return sample_like, corruption, views


def run_engine(
    settings: Mapping[str, Any],
    config: Any,
    inner: torch.nn.Module,
    device: torch.device,
    variants: Sequence[OracleVariant],
    conditions: Sequence[E10.Condition],
    entries: Sequence[str],
    builder: UnitBuilder,
) -> dict[str, Any]:
    output_dir = Path(settings["output_dir"])
    base_rng = FAST._capture_rng(device)
    identity = dict(settings["identity"])
    written: dict[str, Any] = {}
    timings: dict[str, float] = {"total_seconds": 0.0}

    wrappers: dict[str, tuple[GeometryOracleModel, FAST.ForwardTimer]] = {}
    for variant in variants:
        wrapped = GeometryOracleModel(inner, variant)
        wrappers[variant.name] = (wrapped, FAST.ForwardTimer(wrapped))

    for condition in conditions:
        for variant in variants:
            key = f"{variant.name}/{condition.directory}"
            target = output_dir / variant.name / condition.directory / "metrics.json"
            if settings["resume"] and target.exists():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if existing.get("identity") == {**identity, "variant": variant.name, "condition": condition.directory}:
                    if existing.get("completed"):
                        written[key] = {"status": "skipped-existing", "path": str(target)}
                        continue
            per_sample: list[dict[str, Any]] = []
            hist = np.zeros((int(config.num_classes), int(config.num_classes)), dtype=np.int64)
            validity_records: list[dict[str, Any]] = []
            elapsed_total = 0.0
            forward_total = 0.0
            peak_allocated = 0
            peak_reserved = 0
            condition_started = time.perf_counter()
            for entry in entries:
                sample_like, corruption, views = builder.build(entry, condition)
                validity_summary_record = validity_summary(corruption.validity_state)
                real_plan = build_view_invalidity_plan(raw_invalidity(corruption.validity_state), views)
                coverage = [stage_coverage(plan, view["padded_size_hw"]) for plan, view in zip(real_plan, views)]
                view_invalid_fractions = [float(plan.mean()) for plan in real_plan]
                wrapped, timed = wrappers[variant.name]
                if variant.requires_mask():
                    plan = (
                        [np.zeros_like(entry_) for entry_ in real_plan]
                        if variant.all_valid_mask
                        else real_plan
                    )
                else:
                    plan = None
                wrapped.push_unit(plan)
                FAST._reset_peak_memory(device)
                unit_started = time.perf_counter()
                FAST._restore_rng(base_rng, device)
                result = E10.run_msflip(timed, sample_like, device, config, settings["view_batching"], settings["batching_strategy"])
                FAST._sync(device)
                elapsed = time.perf_counter() - unit_started
                if wrapped.pending() != 0:
                    raise RuntimeError("oracle queue was not fully consumed; view bookkeeping is broken")
                peak = FAST._peak_memory(device)
                logits = result["logits"]
                label = sample_like["label"]
                if tuple(int(value) for value in logits.shape[-2:]) != (int(label.shape[0]), int(label.shape[1])):
                    raise RuntimeError("restored logits are not on the original label grid")
                metrics = result["metrics"]
                hist += result["hist"]
                elapsed_total += elapsed
                forward_total += float(result["timings"]["forward_seconds"])
                peak_allocated = max(peak_allocated, int(peak.get("allocated_bytes") or 0))
                peak_reserved = max(peak_reserved, int(peak.get("reserved_bytes") or 0))
                per_sample.append(
                    {
                        "sample_id": sample_like["sample_id"],
                        "location_group": location_group(sample_like["sample_id"]),
                        "entry": entry,
                        "miou": float(metrics["miou"]),
                        "macc": float(metrics["macc"]),
                        "mf1": float(metrics["mf1"]),
                        "confusion_matrix": result["hist"].tolist(),
                        "elapsed_seconds": round(elapsed, 6),
                        "forward_seconds": float(result["timings"]["forward_seconds"]),
                        "peak_allocated_bytes": peak.get("allocated_bytes"),
                        "corrupted_depth_sha256": corruption.depth_sha256(),
                        "validity_state_sha256": corruption.state_sha256(),
                    }
                )
                validity_records.append(
                    {
                        "sample_id": sample_like["sample_id"],
                        **validity_summary_record,
                        "view_invalid_fraction_mean": float(np.mean(view_invalid_fractions)),
                        "stage_coverage": coverage[0] if coverage else {},
                    }
                )
                del sample_like, views, result, logits, corruption, real_plan, coverage
            metrics = EV.metrics_from_confusion(hist, list(config.class_names))
            stage_mean = _mean_stage_coverage(validity_records)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "completed": len(per_sample) == len(entries),
                "variant": {
                    "name": variant.name,
                    "mode": variant.mode,
                    "depth_geometry_off": variant.depth_geometry_off,
                    "all_valid_mask": variant.all_valid_mask,
                    "description": variant.description,
                    "geometry_implementation": variant.geometry_implementation,
                },
                "condition": condition.definition(),
                "identity": {**identity, "variant": variant.name, "condition": condition.directory},
                "sample_count": len(per_sample),
                "metrics_percent": {
                    "miou": metrics["miou"],
                    "macc": metrics["macc"],
                    "mf1": metrics["mf1"],
                    "per_class": metrics["per_class"],
                },
                "confusion_matrix": metrics["confusion_matrix"],
                "per_sample": per_sample,
                "validity": {
                    "source": "corruption pipeline final validity state (V_state_final) on the raw aligned grid",
                    "aggregation": {
                        "raw_to_view": "cv2.resize(INTER_LINEAR) -> horizontal flip -> right/bottom zero padding (pad treated as valid/untouched)",
                        "view_to_stage": "F.interpolate(bilinear, align_corners=False), the same operator GeoPriorGen applies to Depth",
                        "strict_rule": "r_i = 1 iff the interpolated invalidity of the token is exactly 0",
                        "continuous_rule": "c_i = 1 - interpolated invalidity (weighted valid fraction)",
                        "pair_rule": "rho_ij = q_i * q_j with q in {r, c}",
                        "intervention_location": "models.encoders.DFormerv2.GeoPriorGen: only self.weight[1] * mask_d",
                    },
                    "per_sample": validity_records,
                    "stage_coverage_mean": stage_mean,
                },
                "timings": {
                    "condition_seconds": round(time.perf_counter() - condition_started, 3),
                    "unit_seconds_total": round(elapsed_total, 6),
                    "forward_seconds_total": round(forward_total, 6),
                },
                "runtime": {
                    "device": str(device),
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "view_batching": settings["view_batching"],
                },
                "official_test_included": False,
                "timestamp_utc": E10.now_utc(),
            }
            E10.atomic_write_json(target, payload)
            written[key] = {"status": "completed", "path": str(target), "miou": metrics["miou"]}
            timings["total_seconds"] += payload["timings"]["condition_seconds"]
    return {"written": written, "timings": timings}


def _mean_stage_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    stages = sorted(records[0]["stage_coverage"].keys())
    result: dict[str, Any] = {}
    for stage in stages:
        strict = [float(record["stage_coverage"][stage]["strict_invalid_token_fraction"]) for record in records]
        continuous = [float(record["stage_coverage"][stage]["continuous_reliability_mean"]) for record in records]
        result[stage] = {
            "grid_hw": list(records[0]["stage_coverage"][stage]["grid_hw"]),
            "strict_invalid_token_fraction_mean": float(np.mean(strict)),
            "strict_invalid_token_fraction_min": float(np.min(strict)),
            "strict_invalid_token_fraction_max": float(np.max(strict)),
            "continuous_reliability_mean": float(np.mean(continuous)),
        }
    raw_invalid = [float(record["invalid_fraction"]) for record in records]
    result["raw_grid"] = {
        "invalid_fraction_mean": float(np.mean(raw_invalid)),
        "invalid_fraction_min": float(np.min(raw_invalid)),
        "invalid_fraction_max": float(np.max(raw_invalid)),
    }
    return result


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


def build_summary(output_dir: Path, variants: Sequence[OracleVariant], conditions: Sequence[E10.Condition]) -> dict[str, Any]:
    table: dict[str, dict[str, Any]] = {}
    for variant in variants:
        for condition in conditions:
            path = output_dir / variant.name / condition.directory / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            table.setdefault(variant.name, {})[condition.condition_id] = {
                "miou": payload["metrics_percent"]["miou"],
                "macc": payload["metrics_percent"]["macc"],
                "mf1": payload["metrics_percent"]["mf1"],
                "path": str(path),
            }
    deltas: dict[str, dict[str, Any]] = {}
    reference = table.get("original", {})
    for variant in variants:
        if variant.name == "original":
            continue
        row: dict[str, Any] = {}
        for condition in conditions:
            key = condition.condition_id
            if key in reference and key in table[variant.name]:
                row[key] = round(
                    float(table[variant.name][key]["miou"]) - float(reference[key]["miou"]), 6
                )
        deltas[variant.name] = row
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": None,
        "table_miou": table,
        "delta_miou_vs_original": deltas,
        "official_test_included": False,
        "timestamp_utc": E10.now_utc(),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--variants", default=",".join(REQUIRED_VARIANTS))
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-selection", choices=("first", "stratified"), default="first")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view-batching", type=int, default=2)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    def resolve(path: Path) -> Path:
        candidate = path if path.is_absolute() else (repo_root / path)
        if not candidate.exists():
            raise FileNotFoundError(f"required path does not exist: {candidate}")
        return candidate.resolve()

    protocol_path = resolve(args.protocol)
    protocol = E10.load_frozen_protocol(protocol_path)
    module = importlib.import_module(args.config)
    config = copy.copy(module.C)
    channel_order = getattr(config, "channel_order", None)
    if channel_order not in EV.CHANNEL_ORDERS:
        raise ValueError("config must declare channel_order as BGR or RGB")

    all_conditions = E10.build_conditions(E10.FROZEN_EVALUATION)
    condition_definition_sha256 = json_sha256([condition.definition() for condition in all_conditions])
    conditions = E10.resolve_conditions(args.conditions, all_conditions)
    variants = resolve_variants(args.variants)

    split = resolve(args.split)
    dataset_root = resolve(args.dataset_root)
    checkpoint = resolve(args.checkpoint)
    split_sha256 = EV.file_sha256(split)
    checkpoint_sha256 = EV.file_sha256(checkpoint)
    if args.expected_split_sha256 and split_sha256.lower() != str(args.expected_split_sha256).lower():
        raise ValueError("split SHA-256 mismatch")
    if args.expected_checkpoint_sha256 and checkpoint_sha256.lower() != str(args.expected_checkpoint_sha256).lower():
        raise ValueError("checkpoint SHA-256 mismatch")
    entries = EV.split_entries(split)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        if args.sample_selection == "first":
            entries = list(entries[: args.max_samples])
        else:
            positions = np.linspace(0, len(entries) - 1, num=args.max_samples, dtype=np.int64)
            seen: list[str] = []
            for position in positions.tolist():
                item = entries[int(position)]
                if item not in seen:
                    seen.append(item)
            entries = seen
    evaluation_seed = int(protocol.evaluation["seed"]) if args.evaluation_seed is None else int(args.evaluation_seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    EV.configure_fp32_forward(device)

    strategy, batching_evidence = E10.load_microbatch_strategy(
        repo_root / "experiments" / "MMFR_A2_v3" / "benchmarks" / "microbatch-strategy-final.json",
        args.view_batching,
    )
    runner_path = Path(__file__).resolve()
    identity = {
        "runner_sha256": EV.file_sha256(runner_path),
        "checkpoint_sha256": checkpoint_sha256,
        "split_sha256": split_sha256,
        "split_role": "val_dev",
        "protocol_raw_sha256": protocol.raw_sha256,
        "condition_definition_sha256": condition_definition_sha256,
        "config_module": str(args.config),
        "evaluation_seed": evaluation_seed,
        "sample_count": len(entries),
        "sample_set_sha256": hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest(),
        "official_test_included": False,
    }
    settings = {
        "output_dir": args.output_dir,
        "resume": bool(args.resume),
        "view_batching": int(args.view_batching),
        "batching_strategy": strategy,
        "identity": identity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inner = EV.load_model(config, checkpoint, device)
    # The frozen Main-Val runner (``tools.evaluate_museg_10condition.load_eval_model``)
    # pins the CPU and CUDA generators to the evaluation seed before it captures the base
    # state, because the Ham decoder re-draws its NMF bases with ``torch.rand`` on every
    # forward.  Reproducing that seeding is what makes the ``original`` variant of this
    # runner bit-comparable with the frozen Main-Val evidence; without it the reference
    # numbers would drift by decoder randomness instead of measuring the oracle.
    torch.manual_seed(evaluation_seed)
    torch.cuda.manual_seed_all(evaluation_seed)
    builder = UnitBuilder(dataset_root, str(channel_order), config, evaluation_seed)
    engine = run_engine(settings, config, inner, device, variants, conditions, entries, builder)
    summary = build_summary(args.output_dir, variants, conditions)
    summary["checkpoint"] = {"path": str(checkpoint), "sha256": checkpoint_sha256}
    summary["identity"] = identity
    summary["batching_evidence"] = batching_evidence
    summary["written"] = engine["written"]
    summary["wall_clock_seconds"] = round(time.perf_counter() - started, 3)
    summary["device"] = str(device)
    E10.atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"),
                      "table_miou": summary["table_miou"],
                      "delta_miou_vs_original": summary["delta_miou_vs_original"],
                      "wall_clock_seconds": summary["wall_clock_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
