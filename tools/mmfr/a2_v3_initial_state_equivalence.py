#!/usr/bin/env python3
"""MMFR-A2 v3 initial-state equivalence gate.

This CPU-only gate builds the v3 clean-control and Depth-corruption identities in
independent child interpreters.  It compares every common parameter and buffer by
native-dtype and float64 SHA-256, and records the reliability-only state difference.
The Ham decoder is constructed but no forward or backward pass is run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GATE_ID = "MMFR-A2-v3-initial-state-equivalence"
SCHEMA_VERSION = "mmfr-a2-v3-initial-state-equivalence-v1"
PROTOCOL_ID = "MMFR-A2-train-integration-v3"
REPEATS = 3
TRAIN_SOURCE_RELATIVE = "utils/train.py"
RELIABILITY_PREFIX = "reliability_estimator."
EXPECTED_SEED = 772961337
EXPECTED_PRETRAINED_PATH = r"D:\0Project\pretrained\DFormerv2_Small_pretrained.pth"
EXPECTED_PRETRAINED_SHA256 = "19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6"
EXTRA_NORMS_NAMES: Tuple[str, ...] = tuple(
    f"backbone.extra_norms.{layer}.{suffix}"
    for layer in range(3)
    for suffix in ("weight", "bias")
)
IDENTITIES: Tuple[Dict[str, Any], ...] = (
    {
        "label": "clean-control-v3",
        "config_module": "local_configs.MUSeg.DFormerv2_S_MMFR_A2_Clean_v3",
        "config_file": "local_configs/MUSeg/DFormerv2_S_MMFR_A2_Clean_v3.py",
        "expected_run_id": "museg-dformerv2-s-mmfr-a2-clean-control-v3",
        "expected_analysis_identity": "MMFR-A2-clean-control-v3",
        "expected_mode": "clean-control",
        "expects_reliability_head": False,
    },
    {
        "label": "depth-corruption-v3",
        "config_module": "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3",
        "config_file": "local_configs/MUSeg/DFormerv2_S_MMFR_A2_DepthCorrupt_v3.py",
        "expected_run_id": "museg-dformerv2-s-mmfr-a2-depth-corruption-train-v3",
        "expected_analysis_identity": "MMFR-A2-depth-corruption-train-v3",
        "expected_mode": "depth-corruption",
        "expects_reliability_head": True,
    },
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "mmfr-a2-v3-initial-state-equivalence"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "a2-v3-initial-state-equivalence.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def tensor_digests(tensor: torch.Tensor) -> Dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous()
    try:
        native = value.numpy().tobytes()
    except (TypeError, RuntimeError):
        native = value.view(torch.uint8).numpy().tobytes()
    float64 = value.to(torch.float64).contiguous().numpy().tobytes()
    return {
        "shape": [int(size) for size in value.shape],
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "sha256_native_bytes": hashlib.sha256(native).hexdigest(),
        "sha256_float64_bytes": hashlib.sha256(float64).hexdigest(),
    }


def digest_key(entry: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        tuple(entry.get("shape") or ()),
        str(entry.get("dtype")),
        str(entry.get("sha256_native_bytes")),
        str(entry.get("sha256_float64_bytes")),
    )


def extract_set_seed() -> Tuple[str | None, Path]:
    path = REPO_ROOT / TRAIN_SOURCE_RELATIVE
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "set_seed":
            return ast.get_source_segment(source, node) or ast.unparse(node), path
    return None, path


def build_set_seed(source: str) -> Callable[[int], None]:
    namespace: Dict[str, Any] = {"random": random, "np": np, "os": os, "torch": torch}
    exec(compile(source, "<utils/train.py:set_seed>", "exec"), namespace)  # noqa: S102
    setter = namespace.get("set_seed")
    if not callable(setter):
        raise RuntimeError("extracted set_seed did not define a callable")
    return setter


def child_record(args: argparse.Namespace) -> int:
    identity = IDENTITIES[int(args.identity_index)]
    record: Dict[str, Any] = {
        "child": True,
        "status": "ERROR",
        "identity_label": identity["label"],
        "identity_index": int(args.identity_index),
        "repeat": int(args.repeat),
        "config_module": identity["config_module"],
        "config_file": identity["config_file"],
        "error": None,
    }
    started = time.perf_counter()
    try:
        source, source_path = extract_set_seed()
        if source is None:
            raise RuntimeError(f"could not extract set_seed from {TRAIN_SOURCE_RELATIVE}")
        set_seed = build_set_seed(source)
        config = importlib.import_module(identity["config_module"]).C
        set_seed(int(config.seed))
        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
        from models.builder import EncoderDecoder

        model = EncoderDecoder(cfg=config, criterion=criterion, norm_layer=nn.SyncBatchNorm, syncbn=True)
        parameters = {name: tensor_digests(tensor) for name, tensor in model.named_parameters()}
        buffers = {name: tensor_digests(tensor) for name, tensor in model.named_buffers()}
        parameter_lookup = dict(model.named_parameters())
        extra_norms: Dict[str, Any] = {}
        for name in EXTRA_NORMS_NAMES:
            tensor = parameter_lookup.get(name)
            if tensor is not None:
                value = tensor.detach().to("cpu").to(torch.float64)
                extra_norms[name] = {
                    **tensor_digests(tensor),
                    "min": float(value.min().item()),
                    "max": float(value.max().item()),
                    "mean": float(value.mean().item()),
                }
        from utils.init_func import group_weight

        params_list = group_weight([], model, nn.SyncBatchNorm, float(config.lr))
        optimizer = torch.optim.AdamW(
            params_list,
            lr=float(config.lr),
            betas=(0.9, 0.999),
            weight_decay=float(config.weight_decay),
        )
        mmfr = dict(getattr(config, "mmfr_a2", {}) or {})
        reliability_names = sorted(name for name in parameters if name.startswith(RELIABILITY_PREFIX))
        record.update(
            {
                "status": "OK",
                "run_id": str(getattr(config, "run_id", "")),
                "analysis_identity": str(dict(mmfr.get("frozen", {}) or {}).get("analysis_identity", "")),
                "mode": str(mmfr.get("mode", "")),
                "protocol": str(mmfr.get("protocol", "")),
                "seed": int(config.seed),
                "pretrained_model": str(Path(str(config.pretrained_model)).resolve()),
                "pretrained_sha256": sha256_file(Path(str(config.pretrained_model)).resolve()),
                "reliability_head_instantiated": bool(getattr(model, "reliability_estimator", None) is not None),
                "reliability_parameter_names": reliability_names,
                "reliability_parameter_count": len(reliability_names),
                "reliability_parameter_numel": sum(int(parameters[name]["numel"]) for name in reliability_names),
                "parameter_count": len(parameters),
                "buffer_count": len(buffers),
                "parameter_numel": sum(int(item["numel"]) for item in parameters.values()),
                "buffer_numel": sum(int(item["numel"]) for item in buffers.values()),
                "parameters": parameters,
                "buffers": buffers,
                "extra_norms": extra_norms,
                "optimizer": {
                    "name": type(optimizer).__name__,
                    "betas": [float(value) for value in optimizer.defaults["betas"]],
                    "weight_decay": float(optimizer.defaults["weight_decay"]),
                    "lr": float(optimizer.defaults["lr"]),
                    "param_group_count": len(optimizer.param_groups),
                    "param_group_param_counts": [len(group["params"]) for group in optimizer.param_groups],
                },
                "set_seed_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "train_source": relative(source_path),
                "cuda_initialized_after_build": bool(torch.cuda.is_initialized()),
                "forward_run": False,
                "backward_run": False,
                "data_access": False,
                "build_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except BaseException as error:  # noqa: BLE001 - child must leave evidence
        record["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        record["build_seconds"] = round(time.perf_counter() - started, 3)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def run_child(output_dir: Path, identity_index: int, repeat: int) -> Dict[str, Any]:
    label = IDENTITIES[identity_index]["label"]
    child_path = output_dir / f"build-{label}-repeat{repeat}.json"
    stdout_path = output_dir / f"build-{label}-repeat{repeat}-stdout.txt"
    stderr_path = output_dir / f"build-{label}-repeat{repeat}-stderr.txt"
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--identity-index", str(identity_index), "--repeat", str(repeat), "--out", str(child_path)]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    record = None
    if child_path.is_file():
        try:
            record = json.loads(child_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            record = None
    return {
        "identity_label": label,
        "identity_index": identity_index,
        "repeat": repeat,
        "returncode": int(completed.returncode),
        "record_path": relative(child_path),
        "record_sha256": sha256_file(child_path) if child_path.is_file() else None,
        "record": record,
        "stdout_path": relative(stdout_path),
        "stderr_path": relative(stderr_path),
    }


def compare_states(records: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> Dict[str, Any]:
    first_label, second_label = [identity["label"] for identity in IDENTITIES]
    first = records[first_label][0]
    second = records[second_label][0]
    common_parameters = sorted(set(first["parameters"]) & set(second["parameters"]))
    common_buffers = sorted(set(first["buffers"]) & set(second["buffers"]))
    unequal_parameters = [name for name in common_parameters if digest_key(first["parameters"][name]) != digest_key(second["parameters"][name])]
    unequal_buffers = [name for name in common_buffers if digest_key(first["buffers"][name]) != digest_key(second["buffers"][name])]
    clean_only_parameters = sorted(set(first["parameters"]) - set(second["parameters"]))
    corruption_only_parameters = sorted(set(second["parameters"]) - set(first["parameters"]))
    clean_only_buffers = sorted(set(first["buffers"]) - set(second["buffers"]))
    corruption_only_buffers = sorted(set(second["buffers"]) - set(first["buffers"]))
    within: Dict[str, Any] = {}
    for label, identity_records in records.items():
        for kind in ("parameters", "buffers"):
            baseline = identity_records[0][kind]
            unstable = []
            missing = []
            for name, entry in baseline.items():
                for repeat in range(1, REPEATS):
                    other = identity_records[repeat][kind]
                    if name not in other:
                        missing.append(f"{kind}:{name}@repeat{repeat}")
                    elif digest_key(other[name]) != digest_key(entry):
                        unstable.append(f"{kind}:{name}@repeat{repeat}")
            within[f"{label}/{kind}"] = {"key_count": len(baseline), "unstable": unstable, "missing": missing}
    reliability_parameter_difference = [name for name in corruption_only_parameters if name.startswith(RELIABILITY_PREFIX)]
    unexpected_corruption_parameters = [name for name in corruption_only_parameters if not name.startswith(RELIABILITY_PREFIX)]
    unexpected_corruption_buffers = [name for name in corruption_only_buffers if not name.startswith(RELIABILITY_PREFIX)]
    return {
        "common_parameter_count": len(common_parameters),
        "common_buffer_count": len(common_buffers),
        "common_parameters_exact_equal_count": len(common_parameters) - len(unequal_parameters),
        "common_buffers_exact_equal_count": len(common_buffers) - len(unequal_buffers),
        "unequal_parameter_names": unequal_parameters,
        "unequal_buffer_names": unequal_buffers,
        "clean_only_parameter_names": clean_only_parameters,
        "corruption_only_parameter_names": corruption_only_parameters,
        "clean_only_buffer_names": clean_only_buffers,
        "corruption_only_buffer_names": corruption_only_buffers,
        "reliability_parameter_difference_names": reliability_parameter_difference,
        "unexpected_corruption_parameter_names": unexpected_corruption_parameters,
        "unexpected_corruption_buffer_names": unexpected_corruption_buffers,
        "within_identity_repeat_stability": within,
        "max_abs_difference": 0.0 if not unequal_parameters and not unequal_buffers else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--identity-index", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    if args.child:
        if args.out is None:
            parser.error("--child requires --out")
        return child_record(args)

    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report).resolve() if args.report else output_dir / DEFAULT_REPORT.name
    checks: List[Dict[str, Any]] = []
    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")

    source, source_path = extract_set_seed()
    check("train_source.set_seed_extracted", source is not None, f"path={relative(source_path)}")
    for identity in IDENTITIES:
        path = REPO_ROOT / identity["config_file"]
        check(f"config.{identity['label']}.present", path.is_file(), f"path={identity['config_file']}")
    builds: List[Dict[str, Any]] = []
    records: Dict[str, Dict[int, Dict[str, Any]]] = {identity["label"]: {} for identity in IDENTITIES}
    for identity_index, identity in enumerate(IDENTITIES):
        for repeat in range(REPEATS):
            build = run_child(output_dir, identity_index, repeat)
            child = build.get("record") or {}
            build["record_status"] = child.get("status")
            builds.append(build)
            if child.get("status") == "OK":
                records[identity["label"]][repeat] = child
            check(
                f"build.{identity['label']}.repeat{repeat}.succeeded",
                build["returncode"] == 0 and child.get("status") == "OK",
                f"returncode={build['returncode']} record_status={child.get('status')}",
            )

    for identity in IDENTITIES:
        label = identity["label"]
        if len(records[label]) != REPEATS:
            continue
        first = records[label][0]
        check(f"identity.{label}.run_id", first["run_id"] == identity["expected_run_id"], repr(first["run_id"]))
        check(f"identity.{label}.analysis_identity", first["analysis_identity"] == identity["expected_analysis_identity"], repr(first["analysis_identity"]))
        check(f"identity.{label}.mode", first["mode"] == identity["expected_mode"], repr(first["mode"]))
        check(f"identity.{label}.protocol", first["protocol"] == PROTOCOL_ID, repr(first["protocol"]))
        check(f"identity.{label}.seed", first["seed"] == EXPECTED_SEED, str(first["seed"]))
        check(f"identity.{label}.reliability_head", first["reliability_head_instantiated"] == identity["expects_reliability_head"], str(first["reliability_head_instantiated"]))
        check(f"identity.{label}.forward_backward_absent", not first["forward_run"] and not first["backward_run"], "construction only")
        check(f"identity.{label}.pretrained_sha256", first["pretrained_sha256"] == EXPECTED_PRETRAINED_SHA256, first["pretrained_sha256"])

    all_records = [record for identity_records in records.values() for record in identity_records.values()]
    pretrained_paths = sorted({record["pretrained_model"] for record in all_records})
    pretrained_hashes = sorted({record["pretrained_sha256"] for record in all_records})
    comparison: Dict[str, Any] = {}
    if all(len(records[identity["label"]]) == REPEATS for identity in IDENTITIES):
        comparison = compare_states(records)
        check("comparison.common_parameters_bitwise_identical", not comparison["unequal_parameter_names"], f"{comparison['common_parameters_exact_equal_count']}/{comparison['common_parameter_count']}")
        check("comparison.common_buffers_bitwise_identical", not comparison["unequal_buffer_names"], f"{comparison['common_buffers_exact_equal_count']}/{comparison['common_buffer_count']}")
        check("comparison.clean_has_no_extra_parameters", not comparison["clean_only_parameter_names"], repr(comparison["clean_only_parameter_names"]))
        check("comparison.extras_are_reliability_only", not comparison["unexpected_corruption_parameter_names"], repr(comparison["unexpected_corruption_parameter_names"]))
        check("comparison.extras_buffers_are_reliability_only", not comparison["unexpected_corruption_buffer_names"], repr(comparison["unexpected_corruption_buffer_names"]))
        unstable = {name: value for name, value in comparison["within_identity_repeat_stability"].items() if value["unstable"] or value["missing"]}
        check("comparison.repeat_rebuilds_bitwise_stable", not unstable, repr(unstable))
        check("comparison.max_abs_difference_exact_zero", comparison["max_abs_difference"] == 0.0, repr(comparison["max_abs_difference"]))
    else:
        check("comparison.all_builds_available", False, "one or more child builds failed")

    extra_norms: Dict[str, Any] = {}
    for name in EXTRA_NORMS_NAMES:
        entries = []
        for label, identity_records in records.items():
            for repeat, record in sorted(identity_records.items()):
                entry = record.get("extra_norms", {}).get(name)
                entries.append({"identity": label, "repeat": repeat, "present": entry is not None, "sha256_native_bytes": entry.get("sha256_native_bytes") if entry else None, "values": {key: entry.get(key) for key in ("min", "max", "mean")} if entry else None})
        distinct = sorted({entry["sha256_native_bytes"] for entry in entries if entry["present"]})
        extra_norms[name] = {"build_count": len(entries), "present_in_all_builds": bool(entries) and all(entry["present"] for entry in entries), "distinct_sha256_native_bytes": distinct, "reproducible": bool(entries) and all(entry["present"] for entry in entries) and len(distinct) == 1, "builds": entries}
        check(f"extra_norms.{name}.reproducible", extra_norms[name]["reproducible"], f"distinct={len(distinct)}")

    failures = [entry for entry in checks if not entry["ok"]]
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_ID,
        "protocol_id": PROTOCOL_ID,
        "official_test_included": False,
        "checkpoint_read": True,
        "forward_run": False,
        "backward_run": False,
        "dataset_access": False,
        "gpu_used": False,
        "purpose": "prove v3 clean/depth-corruption construction shares bitwise-identical common parameters and buffers while recording reliability-only differences",
        "tool": {"path": relative(Path(__file__)), "sha256": sha256_file(Path(__file__).resolve())},
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__, "cuda_available": bool(torch.cuda.is_available())},
        "train_source": {"path": relative(source_path), "sha256": sha256_file(source_path), "lf_normalized_sha256": sha256_lf(source_path), "set_seed_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None},
        "identities": [identity for identity in IDENTITIES],
        "config_files": {identity["config_file"]: {"sha256": sha256_file(REPO_ROOT / identity["config_file"]), "lf_normalized_sha256": sha256_lf(REPO_ROOT / identity["config_file"]) } for identity in IDENTITIES if (REPO_ROOT / identity["config_file"]).is_file()},
        "pretrained": {"paths": pretrained_paths, "sha256s": pretrained_hashes, "expected_path": EXPECTED_PRETRAINED_PATH, "expected_sha256": EXPECTED_PRETRAINED_SHA256, "same_for_both_identities": len(pretrained_paths) == 1, "matches_frozen": pretrained_hashes == [EXPECTED_PRETRAINED_SHA256]},
        "rebuilds_per_identity": REPEATS,
        "builds": builds,
        "state_comparison": comparison,
        "extra_norms": extra_norms,
        "checks": checks,
        "failures": failures,
        "assertions_total": len(checks),
        "assertions_failed": len(failures),
        "initial_state_equivalence": "PASS" if not failures else "FAIL",
        "status": "PASS" if not failures else "FAIL",
        "exit_code": 0 if not failures else 1,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "not_executed": ["no forward pass", "no backward pass", "no training", "no evaluator", "no official test"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_sha = sha256_file(report_path)
    print(f"shared_parameters={comparison.get('common_parameter_count')} shared_buffers={comparison.get('common_buffer_count')} max_abs_difference={comparison.get('max_abs_difference')}")
    print(f"JSON report: {report_path}")
    print(f"SHA-256: {report_sha}")
    print("OVERALL: " + report["status"])
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
