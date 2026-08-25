#!/usr/bin/env python3
"""Run manifest-declared MUSeg seeds strictly sequentially and stop on failure."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

try:
    from tools.museg_protocol import ProtocolError, load_protocol, write_json
    from tools.summarize_museg_runs import summarize
except ModuleNotFoundError:
    from museg_protocol import ProtocolError, load_protocol, write_json  # type: ignore[no-redef]
    from summarize_museg_runs import summarize  # type: ignore[no-redef]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed-launcher", default=str(Path(__file__).with_name("run_museg_seed.py")))
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--train-program")
    parser.add_argument("--resume-seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--resume-parent-run-id")
    parser.add_argument("--resume-checkpoint-sha256")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol_manifest)
        seeds = tuple(args.seeds or protocol.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ProtocolError("orchestrator seeds must be non-empty and unique")
        undeclared = [seed for seed in seeds if seed not in protocol.seeds]
        if undeclared:
            raise ProtocolError(f"orchestrator seeds are not declared by the protocol: {undeclared}")
        resume_fields = (
            args.resume_seed,
            args.resume,
            args.resume_parent_run_id,
            args.resume_checkpoint_sha256,
        )
        if any(value is not None for value in resume_fields) and not all(value is not None for value in resume_fields):
            raise ProtocolError(
                "orchestrator resume requires --resume-seed, --resume, --resume-parent-run-id, "
                "and --resume-checkpoint-sha256 together"
            )
        if args.resume_seed is not None and args.resume_seed not in seeds:
            raise ProtocolError("resume seed must be one of the requested orchestrator seeds")

        report_path = protocol.run_root / "orchestrator.json"
        protocol.run_root.mkdir(parents=True, exist_ok=True)
        started = dt.datetime.now(dt.timezone.utc)
        start_clock = time.monotonic()
        records: list[dict[str, object]] = []
        completed_seeds: list[int] = []
        failed_seed: int | None = None
        final_exit_code = 0
        summary_error: str | None = None

        for seed in seeds:
            command = [
                args.python, args.seed_launcher,
                "--protocol-manifest", str(protocol.path),
                "--seed", str(seed),
                "--python", args.python,
            ]
            if args.direct:
                command.append("--direct")
            if args.train_program:
                command += ["--train-program", args.train_program]
            if seed == args.resume_seed:
                command += [
                    "--resume", str(Path(args.resume).resolve()),
                    "--resume-parent-run-id", args.resume_parent_run_id,
                    "--resume-checkpoint-sha256", args.resume_checkpoint_sha256,
                ]
            if args.dry_run:
                command.append("--dry-run")
            seed_started = dt.datetime.now(dt.timezone.utc)
            launch_error = None
            try:
                completed = subprocess.run(command, check=False)
                exit_code = int(completed.returncode)
            except (OSError, subprocess.SubprocessError) as exc:
                launch_error = f"{type(exc).__name__}: {exc}"
                exit_code = 127
            records.append(
                {
                    "seed": seed,
                    "command": command,
                    "started_at_utc": seed_started.isoformat(),
                    "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "exit_code": exit_code,
                    "launch_error": launch_error,
                }
            )
            if exit_code != 0:
                failed_seed = seed
                final_exit_code = exit_code
                break
            completed_seeds.append(seed)

        if final_exit_code == 0 and not args.dry_run:
            try:
                summarize(protocol, protocol.run_root / "summary.json")
            except (OSError, ProtocolError, TypeError, ValueError) as exc:
                summary_error = f"{type(exc).__name__}: {exc}"
                final_exit_code = 3

        write_json(
            report_path,
            {
                "schema_version": "museg-three-seed-orchestrator-v1",
                "protocol_id": protocol.protocol_id,
                "protocol_manifest": str(protocol.path),
                "protocol_manifest_sha256": protocol.manifest_sha256,
                "split_authority": protocol.authority_identity(),
                "phase": protocol.phase,
                "requested_seeds": list(seeds),
                "completed_seeds": completed_seeds,
                "failed_seed": failed_seed,
                "exit_code": final_exit_code,
                "summary_error": summary_error,
                "started_at_utc": started.isoformat(),
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - start_clock,
                "runs": records,
            },
        )
        return final_exit_code
    except (OSError, ProtocolError, subprocess.SubprocessError) as exc:
        print(f"run_museg_3seed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())