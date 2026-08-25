#!/usr/bin/env python3
"""Inspect or compare versioned MUSeg training checkpoints without loading them on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.museg_protocol import write_json
from utils.training_checkpoint import compare_checkpoint_inspections, inspect_training_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--checkpoint", required=True)
    inspect_parser.add_argument("--output", required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--expected-checkpoint", required=True)
    compare_parser.add_argument("--actual-checkpoint", required=True)
    compare_parser.add_argument("--expected-trace")
    compare_parser.add_argument("--actual-trace")
    compare_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "inspect":
        write_json(args.output, inspect_training_checkpoint(args.checkpoint))
        return 0
    if bool(args.expected_trace) != bool(args.actual_trace):
        parser.error("trace comparison requires both --expected-trace and --actual-trace")

    expected = inspect_training_checkpoint(args.expected_checkpoint)
    actual = inspect_training_checkpoint(args.actual_checkpoint)
    mismatches = compare_checkpoint_inspections(expected, actual)
    trace = None
    if args.expected_trace:
        expected_records = [json.loads(line) for line in Path(args.expected_trace).read_text(encoding="utf-8").splitlines() if line.strip()]
        actual_records = [json.loads(line) for line in Path(args.actual_trace).read_text(encoding="utf-8").splitlines() if line.strip()]
        trace = {
            "expected_path": str(Path(args.expected_trace).resolve()),
            "actual_path": str(Path(args.actual_trace).resolve()),
            "expected_records": len(expected_records),
            "actual_records": len(actual_records),
            "sha256_equal": expected_records == actual_records,
        }
        if expected_records != actual_records:
            mismatches.append("audit_trace")
    write_json(
        args.output,
        {
            "schema_version": "museg-stage04-checkpoint-comparison-v1",
            "pass": not mismatches,
            "expected": expected,
            "actual": actual,
            "trace": trace,
            "mismatches": mismatches,
        },
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())