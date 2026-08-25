#!/usr/bin/env python3
"""Convert one bounded MUSeg qualification probe log into structured JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

try:
    from tools.museg_protocol import write_json
except ModuleNotFoundError:
    from museg_protocol import write_json

_TELEMETRY = re.compile(
    r"Iter\s+(?P<step>\d+)/\d+:.*?loss=(?P<loss>[-+0-9.eE]+).*?"
    r"step=(?P<seconds>[-+0-9.eE]+)s\s+throughput=(?P<throughput>[-+0-9.eE]+).*?"
    r"allocated=(?P<allocated>[-+0-9.eE]+)\s+MiB\s+reserved=(?P<reserved>[-+0-9.eE]+)\s+MiB.*?"
    r"free=(?P<free>[-+0-9.eE]+)/(?P<total>[-+0-9.eE]+)\s+MiB.*?"
    r"free_ratio=(?P<ratio>[-+0-9.eE]+)\s+amp_scale=(?P<scale>[-+0-9.eE]+)"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-steps", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--min-free-vram-gib", type=float, required=True)
    parser.add_argument("--min-free-vram-ratio", type=float, required=True)
    args = parser.parse_args(argv)
    if args.required_steps <= 0 or not 0 <= args.warmup_steps < args.required_steps:
        parser.error("required steps must be positive and warmup must be in [0, required steps)")

    run_dir = Path(args.run_dir).resolve()
    def load_json(name: str) -> dict[str, Any]:
        path = run_dir / name
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    evidence_error: str | None = None
    telemetry_path = run_dir / "probe-telemetry.jsonl"
    if not telemetry_path.is_file():
        evidence_error = f"missing telemetry file: {telemetry_path}"
    else:
        for line_number, line in enumerate(telemetry_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                evidence_error = f"telemetry line {line_number} is invalid JSON: {exc}"
                break
            if not isinstance(record, dict) or record.get("attempt") != len(records) + 1:
                evidence_error = "telemetry attempts are missing, duplicated, or non-monotonic"
                break
            fields = ("step_seconds", "images_per_second", "free_mib", "free_ratio", "loss", "amp_scale")
            if any(not isinstance(record.get(field), (int, float)) or not math.isfinite(float(record[field])) for field in fields):
                evidence_error = f"telemetry attempt {record.get('attempt')} has a missing or non-finite metric"
                break
            records.append(record)
        if not records and evidence_error is None:
            evidence_error = "telemetry is empty"

    result = load_json("training_result.json")
    manifest = load_json("run_manifest.json")
    command = load_json("command.json")
    exit_code = int(load_json("train.exit_code").get("exit_code", 127))
    completed = [record for record in records if record.get("optimizer_step_completed") is True]
    stable = [record for record in completed if int(record["completed_optimizer_steps"]) > args.warmup_steps]
    safety_ok = bool(records) and all(
        bool(record.get("safety_passed"))
        and float(record["free_mib"]) >= args.min_free_vram_gib * 1024
        and float(record["free_ratio"]) >= args.min_free_vram_ratio
        for record in records
    )
    if evidence_error is None and (result.get("run_kind") != "probe" or result.get("completed_optimizer_steps") != args.required_steps):
        evidence_error = "training result does not prove the required probe completion"
    if evidence_error is None and len(completed) != args.required_steps:
        evidence_error = f"telemetry contains {len(completed)} completed optimizer steps, expected {args.required_steps}"
    if evidence_error is None and len(stable) != args.required_steps - args.warmup_steps:
        evidence_error = "stable telemetry window has an unexpected count"
    if evidence_error is None and not safety_ok:
        evidence_error = "one or more probe steps violates the configured VRAM safety threshold"

    log_text = (run_dir / "launcher.log").read_text(encoding="utf-8", errors="replace") if (run_dir / "launcher.log").is_file() else ""
    lower_log = log_text.lower()
    if exit_code != 0:
        if "out of memory" in lower_log:
            anomaly: dict[str, str] | None = {"class": "oom", "reason": "launcher log reports CUDA out of memory"}
        elif "xid" in lower_log:
            anomaly = {"class": "cuda_xid", "reason": "launcher log reports CUDA Xid"}
        elif "non-finite" in lower_log:
            anomaly = {"class": "non_finite", "reason": "launcher log reports a non-finite value"}
        elif "gpu safety threshold violated" in lower_log:
            anomaly = {"class": "vram_threshold", "reason": "trainer rejected the configured VRAM safety threshold"}
        elif evidence_error:
            anomaly = {"class": "evidence", "reason": evidence_error}
        else:
            anomaly = {"class": "trainer_failure", "reason": f"trainer exited with {exit_code}"}
    elif evidence_error:
        anomaly = {"class": "evidence", "reason": evidence_error}
    else:
        anomaly = None

    def summarize(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "stdev": None, "cv": None, "p10": None, "p90": None, "min": None, "max": None}
        ordered = sorted(values)
        def percentile(fraction: float) -> float:
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
        return {"count": len(values), "mean": mean, "median": statistics.median(values), "stdev": stdev, "cv": stdev / mean if mean else None, "p10": percentile(0.10), "p90": percentile(0.90), "min": min(values), "max": max(values)}

    argv_values = command.get("argv", []) if isinstance(command, dict) else []
    batch_size = next((argv_values[index + 1] for index, value in enumerate(argv_values[:-1]) if value == "--batch-size"), None)
    payload = {
        "schema_version": "museg-4090-probe-result-v2",
        "identity": {"protocol_id": manifest.get("protocol_id"), "protocol_manifest_sha256": manifest.get("protocol_manifest_sha256"), "git": manifest.get("git"), "split_authority": manifest.get("split_authority"), "splits": manifest.get("splits"), "seed": manifest.get("seed"), "run_id": manifest.get("run_id"), "command": command},
        "batch_size": int(batch_size) if batch_size is not None else None,
        "exit_code": exit_code,
        "completion": {"required_optimizer_steps": args.required_steps, "completed_optimizer_steps": len(completed), "attempted_steps": len(records), "exact_target_met": len(completed) == args.required_steps},
        "windows": {"warmup_steps": args.warmup_steps, "stable_steps": len(stable)},
        "thresholds": {"min_free_vram_gib": args.min_free_vram_gib, "min_free_vram_ratio": args.min_free_vram_ratio, "all_steps_passed": safety_ok},
        "stable_throughput_images_per_second": summarize([float(record["images_per_second"]) for record in stable]),
        "step_time_seconds": summarize([float(record["step_seconds"]) for record in stable]),
        "memory": {"minimum_free_mib": min((float(record["free_mib"]) for record in records), default=None), "minimum_free_ratio": min((float(record["free_ratio"]) for record in records), default=None), "maximum_allocated_mib": max((float(record["allocated_mib"]) for record in records), default=None), "maximum_reserved_mib": max((float(record["reserved_mib"]) for record in records), default=None)},
        "anomaly": anomaly,
        "eligible": exit_code == 0 and anomaly is None,
        "rejection_reasons": [] if exit_code == 0 and anomaly is None else [anomaly["reason"] if anomaly else "non-zero exit code"],
        "evidence_error": evidence_error,
    }
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())