#!/usr/bin/env python3
"""Independently adjudicate a completed MUSeg Stage-05 seed from immutable v1 evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

EXPECTED_MILESTONES = (1, 10, 20, 50, 100, 200, 300, 400, 500)
EXPECTED_VALIDATION_EPOCHS = tuple(range(10, 501, 10))
CORE_TRAIN_RE = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<epochs>\d+) Iter (?P<iteration>\d+)/(?P<iterations>\d+): "
    r"lr=(?P<lr>[-+0-9.eE]+) loss=(?P<loss>[-+0-9.eE]+) "
    r"total_loss=(?P<total_loss>[-+0-9.eE]+)"
)
VALIDATION_RE = re.compile(
    r"Epoch (?P<epoch>\d+) validation result: mIoU (?P<miou>[-+0-9.eE]+), "
    r"best mIoU (?P<best>[-+0-9.eE]+)"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _same_number(left: object, right: object) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    )


def adjudicate(acceptance_path: Path, run_dir: Path) -> dict[str, Any]:
    acceptance_path = acceptance_path.resolve()
    run_dir = run_dir.resolve()
    original = read_json(acceptance_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: object = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "details": details})

    original_checks = original.get("checks")
    failed_checks = []
    if isinstance(original_checks, list):
        failed_checks = [item.get("name") for item in original_checks if isinstance(item, dict) and item.get("pass") is not True]
    check("original_schema", original.get("schema_version") == "museg-stage05-seed-acceptance-v1")
    check("original_report_failed", original.get("pass") is False)
    check(
        "original_single_correctable_failure",
        original.get("errors") == ["milestones_complete"] and failed_checks == ["milestones_complete"],
        {"errors": original.get("errors"), "failed_checks": failed_checks},
    )
    check(
        "original_other_checks_passed",
        isinstance(original_checks, list)
        and bool(original_checks)
        and all(
            isinstance(item, dict)
            and (item.get("name") == "milestones_complete" or item.get("pass") is True)
            for item in original_checks
        ),
    )

    evidence = original.get("evidence_sha256")
    hash_results: list[dict[str, Any]] = []
    hashes_pass = isinstance(evidence, dict) and bool(evidence)
    if isinstance(evidence, dict):
        for relative, expected in sorted(evidence.items()):
            path = run_dir / relative
            actual = file_sha256(path) if path.is_file() else None
            passed = isinstance(expected, str) and actual == expected.lower()
            hashes_pass = hashes_pass and passed
            hash_results.append({"path": relative, "expected": expected, "actual": actual, "pass": passed})
    check("evidence_sha256_unchanged", hashes_pass, hash_results)

    required_json = {
        name: run_dir / name
        for name in ("run_manifest.json", "training_result.json", "run_config.json")
    }
    required_present = all(path.is_file() for path in required_json.values())
    check("required_final_records_present", required_present, {name: str(path) for name, path in required_json.items()})
    manifest = read_json(required_json["run_manifest.json"]) if required_json["run_manifest.json"].is_file() else {}
    result = read_json(required_json["training_result.json"]) if required_json["training_result.json"].is_file() else {}
    run_config = read_json(required_json["run_config.json"]) if required_json["run_config.json"].is_file() else {}

    train_log = run_dir / "train.log"
    text = train_log.read_text(encoding="utf-8", errors="replace") if train_log.is_file() else ""
    epoch_end: dict[int, dict[str, Any]] = {}
    for match in CORE_TRAIN_RE.finditer(text):
        epoch = int(match["epoch"])
        total_epochs = int(match["epochs"])
        iteration = int(match["iteration"])
        iterations = int(match["iterations"])
        if total_epochs == 500 and iteration == iterations == 128:
            epoch_end[epoch] = {
                "epoch": epoch,
                "iteration": iteration,
                "lr": float(match["lr"]),
                "loss": float(match["loss"]),
                "total_loss": float(match["total_loss"]),
            }
    complete_epoch_ends = tuple(sorted(epoch_end)) == tuple(range(1, 501))
    check("all_500_epoch_end_core_logs", complete_epoch_ends, {"count": len(epoch_end), "missing": sorted(set(range(1, 501)) - set(epoch_end))})
    milestone_records = [epoch_end[epoch] for epoch in EXPECTED_MILESTONES if epoch in epoch_end]
    check(
        "milestones_from_core_logs",
        [row["epoch"] for row in milestone_records] == list(EXPECTED_MILESTONES)
        and all(all(math.isfinite(float(row[key])) for key in ("lr", "loss", "total_loss")) for row in milestone_records),
        milestone_records,
    )

    validation_curve = [
        {"epoch": int(match["epoch"]), "miou": float(match["miou"]), "best_miou": float(match["best"])}
        for match in VALIDATION_RE.finditer(text)
    ]
    check(
        "validation_50_points_complete",
        [row["epoch"] for row in validation_curve] == list(EXPECTED_VALIDATION_EPOCHS)
        and all(math.isfinite(row["miou"]) and math.isfinite(row["best_miou"]) for row in validation_curve),
        {"count": len(validation_curve), "epochs": [row["epoch"] for row in validation_curve]},
    )
    check("validation_curve_matches_original", validation_curve == original.get("validation_curve"))

    sealed_test = run_config.get("data", {}).get("splits", {}).get("test", {})
    identity_fields = ("protocol_id", "protocol_manifest_sha256", "phase", "seed")
    identity_consistent = True
    for field in identity_fields:
        identity_consistent = identity_consistent and manifest.get(field) == result.get(field)
        if field in original:
            identity_consistent = identity_consistent and result.get(field) == original.get(field)
    identity_consistent = identity_consistent and result.get("phase") == "development"
    check(
        "final_identity_consistent",
        identity_consistent,
        {field: {"manifest": manifest.get(field), "result": result.get(field), "acceptance": original.get(field)} for field in identity_fields},
    )
    check(
        "final_result_complete",
        manifest.get("exit_code") == 0
        and manifest.get("process_exit_code") == 0
        and result.get("exit_code") == 0
        and result.get("final_epoch") == 500
        and result.get("checkpoint", {}).get("path")
        and _same_number(result.get("best_val_miou"), original.get("best_val_miou"))
        and result.get("best_val_epoch") == original.get("best_val_epoch"),
        {
            "manifest_exit_code": manifest.get("exit_code"),
            "process_exit_code": manifest.get("process_exit_code"),
            "result_exit_code": result.get("exit_code"),
            "final_epoch": result.get("final_epoch"),
            "best_val_miou": result.get("best_val_miou"),
            "best_val_epoch": result.get("best_val_epoch"),
        },
    )
    check(
        "official_test_remained_sealed",
        original.get("official_test_read") is False
        and result.get("official_test_included") is False
        and sealed_test.get("sealed_unread") is True,
        {
            "acceptance_official_test_read": original.get("official_test_read"),
            "result_official_test_included": result.get("official_test_included"),
            "run_config_sealed_unread": sealed_test.get("sealed_unread"),
        },
    )

    failures = [item["name"] for item in checks if not item["pass"]]
    return {
        "schema_version": "museg-stage05-seed-acceptance-v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass": not failures,
        "errors": failures,
        "adjudication_scope": "correct-v1-milestone-telemetry-requirement-only",
        "original_acceptance": {
            "path": str(acceptance_path),
            "sha256": file_sha256(acceptance_path),
            "schema_version": original.get("schema_version"),
            "pass": original.get("pass"),
            "errors": original.get("errors"),
        },
        "run_dir": str(run_dir),
        "protocol_id": original.get("protocol_id"),
        "protocol_manifest_sha256": original.get("protocol_manifest_sha256"),
        "seed": original.get("seed"),
        "official_test_read": False,
        "best_val_miou": result.get("best_val_miou"),
        "best_val_epoch": result.get("best_val_epoch"),
        "completed_optimizer_steps": result.get("completed_optimizer_steps"),
        "attempted_steps": result.get("attempted_steps"),
        "epoch_end_log_count": len(epoch_end),
        "validation_point_count": len(validation_curve),
        "milestones": milestone_records,
        "checks": checks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", required=True, type=Path, help="immutable v1 acceptance.json")
    parser.add_argument("--run-dir", required=True, type=Path, help="preserved seed run directory")
    parser.add_argument("--output", required=True, type=Path, help="new v2 adjudication report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.resolve() == args.acceptance.resolve():
        raise SystemExit("refusing to overwrite the immutable v1 acceptance report")
    try:
        report = adjudicate(args.acceptance, args.run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"adjudicate_museg_seed_acceptance: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pass": report["pass"], "errors": report["errors"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
