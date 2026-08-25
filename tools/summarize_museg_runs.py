#!/usr/bin/env python3
"""Summarize completed MUSeg training/validation runs without reading official test."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.museg_protocol import ProtocolError, file_sha256, load_protocol, read_json, write_json
except ModuleNotFoundError:
    from museg_protocol import ProtocolError, file_sha256, load_protocol, read_json, write_json  # type: ignore[no-redef]


def _number(value: Any, field: str, seed: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProtocolError(f"seed {seed} has invalid {field}")
    return float(value)


def summarize(protocol, output: str | Path) -> dict[str, Any]:
    expected_identity = {
        "protocol_id": protocol.protocol_id,
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "phase": protocol.phase,
        "split_authority": protocol.authority_identity(),
    }
    expected_result_identity = {
        "protocol_id": protocol.protocol_id,
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "phase": protocol.phase,
    }
    discovered: list[int] = []
    for manifest_path in protocol.run_root.glob("seed-*/run_manifest.json"):
        discovered_manifest = read_json(manifest_path)
        if not isinstance(discovered_manifest, dict) or not isinstance(discovered_manifest.get("seed"), int):
            raise ProtocolError(f"invalid discovered run manifest: {manifest_path}")
        for field, expected in expected_identity.items():
            if discovered_manifest.get(field) != expected:
                raise ProtocolError(
                    f"discovered run manifest {manifest_path} has mismatched {field}"
                )
        discovered.append(discovered_manifest["seed"])
    duplicates = sorted({seed for seed in discovered if discovered.count(seed) > 1})
    if duplicates:
        raise ProtocolError(f"duplicate seed metadata: {duplicates}")
    extras = sorted(set(discovered) - set(protocol.seeds))
    if extras:
        raise ProtocolError(f"undeclared seed runs are present: {extras}")

    runs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for seed in protocol.seeds:
        run_dir = protocol.seed_output_dir(seed)
        manifest_path = run_dir / "run_manifest.json"
        result_path = run_dir / "training_result.json"
        if not manifest_path.is_file() or not result_path.is_file():
            raise ProtocolError(f"seed {seed} is missing run_manifest.json or training_result.json")
        manifest = read_json(manifest_path)
        result = read_json(result_path)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != "museg-run-manifest-v2":
            raise ProtocolError(f"seed {seed} has invalid run manifest schema")
        if not isinstance(result, dict) or result.get("schema_version") != "museg-training-result-v1":
            raise ProtocolError(f"seed {seed} has invalid training result schema")
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                raise ProtocolError(f"seed {seed} run manifest has mismatched {field}")
        for field, expected in expected_result_identity.items():
            if result.get(field) != expected:
                raise ProtocolError(f"seed {seed} training result has mismatched {field}")
        if result.get("official_test_included") is not False:
            raise ProtocolError(f"seed {seed} training result does not preserve official test sealing")
        declared_seed = manifest.get("seed")
        if declared_seed != seed or result.get("seed") != seed:
            raise ProtocolError(f"seed directory {seed} contains mismatched seed metadata")
        if seed in seen:
            raise ProtocolError(f"duplicate seed metadata: {seed}")
        seen.add(seed)
        exit_code = manifest.get("exit_code")
        if exit_code != 0 or result.get("exit_code") != 0:
            raise ProtocolError(f"seed {seed} has non-zero exit status")

        checkpoint = result.get("checkpoint")
        checkpoint_record = None
        if checkpoint is not None:
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("path"), str):
                raise ProtocolError(f"seed {seed} checkpoint record is invalid")
            expected_sha = checkpoint.get("sha256")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ProtocolError(f"seed {seed} checkpoint SHA-256 is invalid")
            checkpoint_path = Path(checkpoint["path"])
            if not checkpoint_path.is_file():
                raise ProtocolError(f"seed {seed} checkpoint is missing: {checkpoint_path}")
            actual_sha = file_sha256(checkpoint_path)
            if expected_sha.lower() != actual_sha:
                raise ProtocolError(f"seed {seed} checkpoint SHA-256 mismatch")
            checkpoint_record = {"path": str(checkpoint_path.resolve()), "sha256": actual_sha}
        elif protocol.phase in {"development", "official"}:
            raise ProtocolError(f"seed {seed} {protocol.phase} run lacks a checkpoint record")

        raw_best = result.get("best_val_miou")
        raw_best_epoch = result.get("best_val_epoch")
        if raw_best is None and protocol.phase == "official":
            best_val_miou = None
            best_val_epoch = None
        else:
            best_val_miou = _number(raw_best, "best_val_miou", seed)
            if raw_best_epoch is None:
                raise ProtocolError(f"seed {seed} has no best_val_epoch")
            best_val_epoch = int(raw_best_epoch)
        runs.append(
            {
                "seed": seed,
                "run_id": manifest.get("run_id"),
                "best_val_miou": best_val_miou,
                "best_val_epoch": best_val_epoch,
                "final_epoch": int(result["final_epoch"]),
                "duration_seconds": _number(result.get("duration_seconds"), "duration_seconds", seed),
                "exit_code": 0,
                "checkpoint": checkpoint_record,
            }
        )
    values = [item["best_val_miou"] for item in runs if item["best_val_miou"] is not None]
    summary = {
        "schema_version": "museg-training-summary-v1",
        "protocol_id": protocol.protocol_id,
        "protocol_manifest": str(protocol.path),
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "split_authority": protocol.authority_identity(),
        "phase": protocol.phase,
        "seeds": list(protocol.seeds),
        "runs": runs,
        "best_val_miou_mean": statistics.fmean(values) if values else None,
        "best_val_miou_std_population": statistics.pstdev(values) if values else None,
        "total_duration_seconds": sum(item["duration_seconds"] for item in runs),
        "official_test_included": False,
    }
    write_json(output, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol_manifest)
        output = Path(args.output).resolve() if args.output else protocol.run_root / "summary.json"
        summarize(protocol, output)
        return 0
    except (OSError, ProtocolError, KeyError, TypeError, ValueError) as exc:
        print(f"summarize_museg_runs: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())