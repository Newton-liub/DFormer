#!/usr/bin/env python3
"""Golden bitwise parity harness for the frozen MMFR-A2 v3 corruption helper.

CPU-only.  It fixes a deterministic set of real ``train-dev`` batches, runs the frozen
``build_mmfr_training_batch_v3`` on them and records, for every case and every exported
tensor, the shape, dtype and SHA-256.  ``--verify`` recomputes the same cases and reports
any bitwise difference.

Contract for every future pipeline optimization:

* integer and boolean tensors must be bitwise identical;
* float tensors must be bitwise identical (SHA-256 equality), not merely close;
* a float difference caused by pure execution order must be reported to the senior model
  instead of being absorbed into a tolerance invented here.

The harness never reads the official test split and never writes into the repository tree
except under ``outputs/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.museg_protocol import write_json  # noqa: E402
from utils.dataloader import mmfr_training_v3 as mmfr_v3  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.dataloader.dataloader import get_train_loader  # noqa: E402

TENSOR_KEYS: Sequence[str] = (
    "rgb",
    "depth",
    "raw_rgb",
    "raw_depth",
    "reliability_target",
    "valid_mask",
    "depth_valid_pre",
    "depth_valid_post",
    "telemetry_masks",
)
INPUT_KEYS: Sequence[str] = ("imgs", "modal_xs", "gts")
SCHEMA = "mmfr-a2-v3-corruption-golden-parity-v1"
DEFAULT_CASES = 128


class _SingleProcessEngine:
    """Loader helper only reads ``distributed``; the golden set is single-process."""

    distributed = False
    world_size = 1
    local_rank = 0


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def tensor_record(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "sha256": tensor_sha256(tensor),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(config_module: str):
    config = getattr(importlib.import_module(config_module), "C")
    engine = _SingleProcessEngine()
    # num_workers=0 keeps the geometry augmentation and the batch order fully
    # deterministic, which is what a bitwise parity harness needs.
    config.num_workers = 0
    loader, _ = get_train_loader(engine, RGBXDataset, config)
    return config, loader


def corruption_kwargs(config: Any, *, epoch: int, iteration: int) -> Dict[str, Any]:
    mmfr_a2 = dict(getattr(config, "mmfr_a2", {}) or {})
    corruption_cfg = dict(mmfr_a2.get("corruption") or {})
    return {
        "epoch": epoch,
        "iteration": iteration,
        "niters_per_epoch": int(config.niters_per_epoch),
        "nepochs": int(config.nepochs),
        "rgb_mean": config.norm_mean,
        "rgb_std": config.norm_std,
        "corruption_seed": int(corruption_cfg.get("seed", mmfr_v3.CORRUPTION_SEED)),
        "global_rank": 0,
        "p_clean": float(corruption_cfg.get("p_clean", mmfr_v3.CLEAN_PROBABILITY)),
        "max_specs": int(corruption_cfg.get("max_specs", mmfr_v3.MAX_SPECS)),
        "sample_id_root": getattr(config, "dataset_path", None),
    }


def case_grid(cases: int, nepochs: int, niters_per_epoch: int) -> List[Dict[str, int]]:
    """Deterministic grid that spans the whole curriculum and rotates iterations."""
    grid: List[Dict[str, int]] = []
    for index in range(cases):
        epoch = 1 + int(index * (nepochs - 1) / max(cases - 1, 1))
        iteration = (index * 13) % niters_per_epoch
        grid.append({"epoch": max(1, min(nepochs, epoch)), "iteration": iteration})
    return grid


def run_cases(
    config: Any,
    loader: Any,
    grid: Sequence[Mapping[str, int]],
    *,
    extra_steps: int = 0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    iterator = iter(loader)
    needed = len(grid) + max(extra_steps, 0)
    produced = 0
    while produced < needed:
        try:
            minibatch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            minibatch = next(iterator)
        spec = grid[produced] if produced < len(grid) else None
        epoch = int(spec["epoch"]) if spec else 1 + (produced * 7) % int(config.nepochs)
        iteration = int(spec["iteration"]) if spec else (produced * 29) % int(config.niters_per_epoch)
        imgs = minibatch["data"]
        gts = minibatch["label"]
        modal_xs = minibatch["modal_x"]
        outputs = mmfr_v3.build_mmfr_training_batch_v3(
            imgs,
            modal_xs,
            minibatch.get("fn"),
            **corruption_kwargs(config, epoch=epoch, iteration=iteration),
        )
        record: Dict[str, Any] = {
            "index": produced,
            "epoch": epoch,
            "iteration": iteration,
            "input": {
                "imgs": tensor_record(imgs),
                "modal_xs": tensor_record(modal_xs),
                "gts": tensor_record(gts),
            },
            "outputs": {key: tensor_record(outputs[key]) for key in TENSOR_KEYS},
            "metadata": list(outputs["metadata"]),
            "protocol": outputs["protocol"],
            "supervised_channels": list(outputs["supervised_channels"]),
            "telemetry_categories": list(outputs["telemetry_categories"]),
        }
        records.append(record)
        produced += 1
    return records


def coverage(records: Sequence[Mapping[str, Any]], kinds: Sequence[str]) -> Dict[str, Any]:
    counts = {kind: 0 for kind in kinds}
    severity_buckets = {"light<0.35": 0, "mid0.35-0.70": 0, "heavy>0.70": 0}
    buckets = {
        "clean_samples": 0,
        "two_or_more_specs": 0,
        "sparse_depth_lt_0.25": 0,
        "high_validity_ge_0.90": 0,
        "no_valid_depth_crop": 0,
        "early_curriculum_epoch_le_50": 0,
        "mid_curriculum_150_350": 0,
        "late_curriculum_epoch_ge_450": 0,
    }
    severity_values: List[float] = []
    for record in records:
        epoch = int(record["epoch"])
        if epoch <= 50:
            buckets["early_curriculum_epoch_le_50"] += 1
        if 150 <= epoch <= 350:
            buckets["mid_curriculum_150_350"] += 1
        if epoch >= 450:
            buckets["late_curriculum_epoch_ge_450"] += 1
        for meta in record["metadata"]:
            if meta.get("clean"):
                buckets["clean_samples"] += 1
            if int(meta.get("num_specs", 0)) >= 2:
                buckets["two_or_more_specs"] += 1
            if float(meta.get("depth_valid_pre_fraction", 1.0)) < 0.25:
                buckets["sparse_depth_lt_0.25"] += 1
            if float(meta.get("depth_valid_pre_fraction", 0.0)) >= 0.90:
                buckets["high_validity_ge_0.90"] += 1
            if (
                float(meta.get("valid_fraction", 0.0)) > 0.0
                and float(meta.get("depth_valid_pre_fraction", 1.0)) == 0.0
            ):
                buckets["no_valid_depth_crop"] += 1
            for spec in meta.get("specs") or ():
                kind = spec.get("kind")
                if kind in counts:
                    counts[kind] += 1
                severity = spec.get("severity")
                if severity is not None:
                    severity = float(severity)
                    severity_values.append(severity)
                    if severity < 0.35:
                        severity_buckets["light<0.35"] += 1
                    elif severity <= 0.70:
                        severity_buckets["mid0.35-0.70"] += 1
                    else:
                        severity_buckets["heavy>0.70"] += 1
    return {
        "per_kind_occurrences": counts,
        "severity_buckets": severity_buckets,
        "severity_min": min(severity_values) if severity_values else None,
        "severity_max": max(severity_values) if severity_values else None,
        "buckets": buckets,
    }


def _index_by_key(records: Sequence[Mapping[str, Any]]) -> Dict[tuple, Mapping[str, Any]]:
    return {(int(r["epoch"]), int(r["iteration"])): r for r in records}


def compare(golden: Mapping[str, Any], fresh: Mapping[str, Any]) -> List[Dict[str, Any]]:
    # The golden set round-trips through JSON, which turns tuples into lists; normalise
    # the fresh payload the same way so only real value differences are reported.
    golden = json.loads(json.dumps(golden))
    fresh = json.loads(json.dumps(fresh))
    mismatches: List[Dict[str, Any]] = []
    if golden["schema_version"] != fresh["schema_version"]:
        mismatches.append({"field": "schema_version", "golden": golden["schema_version"], "fresh": fresh["schema_version"]})
    if len(golden["cases"]) != len(fresh["cases"]):
        mismatches.append({"field": "case_count", "golden": len(golden["cases"]), "fresh": len(fresh["cases"])})
    golden_index = _index_by_key(golden["cases"])
    fresh_index = _index_by_key(fresh["cases"])
    if set(golden_index) != set(fresh_index):
        mismatches.append(
            {
                "field": "case_keys",
                "missing_in_fresh": sorted(set(golden_index) - set(fresh_index))[:5],
                "extra_in_fresh": sorted(set(fresh_index) - set(golden_index))[:5],
            }
        )
    for key, golden_case in golden_index.items():
        fresh_case = fresh_index.get(key)
        if fresh_case is None:
            continue
        for group in ("input", "outputs"):
            for name, golden_tensor in golden_case[group].items():
                fresh_tensor = fresh_case[group].get(name)
                if fresh_tensor is None:
                    mismatches.append({"case": list(key), "group": group, "tensor": name, "issue": "missing"})
                    continue
                for field in ("shape", "dtype", "sha256"):
                    if golden_tensor[field] != fresh_tensor[field]:
                        mismatches.append(
                            {
                                "case": list(key),
                                "group": group,
                                "tensor": name,
                                "field": field,
                                "golden": golden_tensor[field],
                                "fresh": fresh_tensor[field],
                            }
                        )
        if golden_case["metadata"] != fresh_case["metadata"]:
            golden_meta = golden_case["metadata"]
            fresh_meta = fresh_case["metadata"]
            for slot, (expected, actual) in enumerate(zip(golden_meta, fresh_meta)):
                if expected != actual:
                    differing = sorted(
                        {
                            key
                            for key in set(expected) | set(actual)
                            if expected.get(key) != actual.get(key)
                        }
                    )
                    mismatches.append(
                        {"case": list(key), "slot": slot, "field": "metadata", "keys": differing[:8]}
                    )
            if len(golden_meta) != len(fresh_meta):
                mismatches.append(
                    {"case": list(key), "field": "metadata_length", "golden": len(golden_meta), "fresh": len(fresh_meta)}
                )
    return mismatches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--config", default="local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3")
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "mmfr-a2-v3-corruption-parity" / "golden-v3-corruption-parity.json"),
    )
    parser.add_argument("--extra-steps", type=int, default=0, help="extra batches appended beyond the grid for coverage")
    parser.add_argument("--seed", type=int, default=2026091402)
    args = parser.parse_args(argv)

    _seed_everything(args.seed)
    config, loader = build_loader(args.config)
    grid = case_grid(args.cases, int(config.nepochs), int(config.niters_per_epoch))
    records = run_cases(config, loader, grid, extra_steps=args.extra_steps)
    kinds = list(mmfr_v3.FAILURE_KINDS)
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "implementation_identity": "MMFR-A2-v3-pipeline-opt1",
        "scientific_protocol": mmfr_v3.PROTOCOL_ID,
        "scientific_semantics_changed": False,
        "official_test_included": False,
        "config_module": args.config,
        "loader_workers": 0,
        "determinism": {
            "python_seed": args.seed,
            "numpy_seed": args.seed,
            "torch_seed": args.seed,
            "note": "num_workers=0 so geometry augmentation and batch order are reproducible",
        },
        "case_count": len(records),
        "tensor_count": len(records) * (len(TENSOR_KEYS) + len(INPUT_KEYS)),
        "coverage": coverage(records, kinds),
        "cases": records,
    }
    output_path = Path(args.output).resolve()

    if args.mode == "build":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, payload)
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        print(
            json.dumps(
                {
                    "mode": "build",
                    "golden": str(output_path),
                    "sha256": digest,
                    "cases": payload["case_count"],
                    "tensors": payload["tensor_count"],
                    "coverage": payload["coverage"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if not output_path.is_file():
        raise SystemExit(f"golden file not found: {output_path}; run --mode build first")
    golden = json.loads(output_path.read_text(encoding="utf-8"))
    mismatches = compare(golden, payload)
    report = {
        "schema_version": "mmfr-a2-v3-corruption-golden-verify-v1",
        "golden": str(output_path),
        "golden_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "cases": payload["case_count"],
        "tensors_compared": payload["tensor_count"],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:40],
        "pass": not mismatches,
        "official_test_included": False,
    }
    report_path = output_path.parent / "golden-parity-report.json"
    write_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2, ensure_ascii=False))
    if mismatches:
        print(json.dumps(mismatches[:10], indent=2, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
