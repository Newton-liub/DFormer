#!/usr/bin/env python3
"""Generate isolated, non-metric evidence for the MUSeg lifecycle-test gate."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Sequence

try:
    from tools.museg_protocol import file_sha256, write_json
except ModuleNotFoundError:  # direct execution from tools/
    from museg_protocol import file_sha256, write_json  # type: ignore[no-redef]


RUN_KIND = "lifecycle-test"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--simulated-exit-code",
        type=int,
        default=0,
        help="test-only terminal workload code; no model or metric is produced",
    )
    return parser.parse_args(argv)


def generate_lifecycle_evidence(
    output_dir: Path,
    *,
    run_id: str,
    simulated_exit_code: int = 0,
) -> dict[str, object]:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if simulated_exit_code < 0:
        raise ValueError("simulated_exit_code cannot be negative")
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"lifecycle-test output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    terminal = {
        "schema_version": "museg-lifecycle-terminal-result-v1",
        "run_kind": RUN_KIND,
        "simulation": True,
        "run_id": run_id,
        "exit_code": int(simulated_exit_code),
        "status": "completed" if simulated_exit_code == 0 else "failed",
        "generated_at_utc": generated_at,
        "official_test_included": False,
    }
    summary = {
        "schema_version": "museg-lifecycle-summary-v1",
        "run_kind": RUN_KIND,
        "simulation": True,
        "run_id": run_id,
        "status": terminal["status"],
        "terminal_exit_code": int(simulated_exit_code),
        "produced_metrics": False,
        "official_test_included": False,
        "generated_at_utc": generated_at,
    }
    terminal_path = output / "terminal-result.json"
    summary_path = output / "summary.json"
    write_json(terminal_path, terminal)
    write_json(summary_path, summary)
    evidence = []
    for path in (terminal_path, summary_path):
        evidence.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema_version": "museg-lifecycle-evidence-manifest-v1",
        "run_kind": RUN_KIND,
        "simulation": True,
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "files": evidence,
    }
    manifest_path = output / "evidence-manifest.json"
    write_json(manifest_path, manifest)
    return {
        "terminal": terminal,
        "summary": summary,
        "manifest": manifest,
        "manifest_sha256": file_sha256(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generate_lifecycle_evidence(
            args.output_dir,
            run_id=args.run_id,
            simulated_exit_code=args.simulated_exit_code,
        )
        return int(args.simulated_exit_code)
    except (OSError, ValueError) as exc:
        print(f"run_museg_lifecycle_test: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
