#!/usr/bin/env python3
"""Run a lifecycle-test workload, verify its evidence, then stop in finally."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

try:
    from tools.museg_protocol import file_sha256
except ModuleNotFoundError:  # direct execution from tools/
    from museg_protocol import file_sha256  # type: ignore[no-redef]


STOPPED_STATE = "Stopped"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--simulated-exit-code", type=int, default=0)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--stop-timeout", type=int, default=600)
    parser.add_argument("--schedule-at", required=True)
    parser.add_argument("--compshare", default="compshare")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_lifecycle_evidence(output_dir: Path, *, run_id: str) -> dict[str, object]:
    output = output_dir.resolve()
    terminal = _load_json(output / "terminal-result.json")
    summary = _load_json(output / "summary.json")
    manifest = _load_json(output / "evidence-manifest.json")
    for record in (terminal, summary, manifest):
        if record.get("run_kind") != "lifecycle-test" or record.get("simulation") is not True:
            raise ValueError("lifecycle evidence is not isolated as simulation=true")
        if record.get("run_id") != run_id:
            raise ValueError("lifecycle evidence run_id mismatch")
    if terminal.get("official_test_included") is not False or summary.get("official_test_included") is not False:
        raise ValueError("lifecycle evidence must exclude official test")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("lifecycle evidence manifest has no files")
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("lifecycle evidence manifest contains an invalid file record")
        path = output / str(record.get("name", ""))
        if not path.is_file() or path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"lifecycle evidence file size mismatch: {path}")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"lifecycle evidence SHA-256 mismatch: {path}")
    return manifest


def _payload_contains_state(value: object, expected: str) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"status", "state", "instance_state"} and item == expected:
                return True
            if _payload_contains_state(item, expected):
                return True
    elif isinstance(value, list):
        return any(_payload_contains_state(item, expected) for item in value)
    return False


def run_lifecycle_controller(
    output_dir: Path,
    *,
    run_id: str,
    instance_id: str,
    simulated_exit_code: int,
    schedule_at: str,
    stop_timeout: int = 600,
    compshare: str = "compshare",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    if stop_timeout <= 0:
        raise ValueError("stop_timeout must be positive")
    if not schedule_at.strip():
        raise ValueError("schedule_at must be non-empty")
    workload_exit = 2
    verified = False
    workload_error: Exception | None = None
    timestamps: dict[str, str] = {
        "controller_started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()
    }
    stop_result: subprocess.CompletedProcess[str] | None = None
    status_result: subprocess.CompletedProcess[str] | None = None
    try:
        schedule_set = runner(
            [compshare, "--json", "instance", "schedule", "set", instance_id, "--at", schedule_at],
            check=False,
            text=True,
            capture_output=True,
        )
        schedule_show = runner(
            [compshare, "--json", "instance", "schedule", "show", instance_id],
            check=False,
            text=True,
            capture_output=True,
        )
        if schedule_set.returncode != 0 or schedule_show.returncode != 0:
            raise RuntimeError("control-plane shutdown schedule could not be set and verified")
        timestamps["schedule_verified_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        schedule_payloads: list[dict[str, object]] = []
        for response in (schedule_set, schedule_show):
            payload = json.loads(response.stdout or "{}")
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                raise RuntimeError("control-plane shutdown schedule did not return ok=true")
            schedule_payloads.append(payload)
        if not schedule_payloads[1].get("data"):
            raise RuntimeError("control-plane shutdown schedule show returned no schedule data")
        workload = runner(
            [
                sys.executable,
                "-m",
                "tools.run_museg_lifecycle_test",
                "--output-dir",
                str(output_dir),
                "--run-id",
                run_id,
                "--simulated-exit-code",
                str(simulated_exit_code),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        workload_exit = int(workload.returncode)
        timestamps["workload_finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if workload_exit == 0:
            try:
                verify_lifecycle_evidence(output_dir, run_id=run_id)
                verified = True
            except (OSError, ValueError) as exc:
                workload_error = exc
    except (OSError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        workload_error = exc
    finally:
        stop_result = runner(
            [
                compshare,
                "--json",
                "instance",
                "stop",
                instance_id,
                "--yes",
                "--wait",
                "--timeout",
                str(stop_timeout),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        status_result = runner(
            [compshare, "--json", "instance", "show", instance_id, "--status"],
            check=False,
            text=True,
            capture_output=True,
        )
        timestamps["stop_sequence_finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    if stop_result.returncode != 0:
        raise RuntimeError(f"control-plane stop failed with exit code {stop_result.returncode}")
    try:
        stop_payload = json.loads(stop_result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("control-plane stop did not return JSON") from exc
    if not isinstance(stop_payload, dict) or stop_payload.get("ok") is not True:
        raise RuntimeError("control-plane stop did not return ok=true")
    if status_result.returncode != 0:
        raise RuntimeError(f"instance status check failed with exit code {status_result.returncode}")
    try:
        status_payload = json.loads(status_result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("instance status check did not return JSON") from exc
    if not isinstance(status_payload, dict) or status_payload.get("ok") is not True:
        raise RuntimeError("instance status check did not return ok=true")
    if not _payload_contains_state(status_payload, STOPPED_STATE):
        raise RuntimeError("instance did not reach Stopped after control-plane stop")
    passed = workload_exit == 0 and verified and workload_error is None
    result = {
        "schema_version": "museg-lifecycle-controller-result-v1",
        "run_kind": "lifecycle-test",
        "simulation": True,
        "run_id": run_id,
        "schedule_at": schedule_at,
        "workload_exit_code": workload_exit,
        "evidence_verified": verified,
        "stop_requested": True,
        "stop_exit_code": int(stop_result.returncode),
        "instance_id": instance_id,
        "instance_state": STOPPED_STATE,
        "status": "passed" if passed else "failed",
        "error": str(workload_error) if workload_error is not None else None,
        "timestamps": timestamps,
        "official_test_included": False,
    }
    result_path = output_dir.resolve() / "controller-result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        detail = f": {workload_error}" if workload_error is not None else ""
        raise RuntimeError(f"lifecycle workload failed or its evidence was not verified{detail}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_lifecycle_controller(
            args.output_dir,
            run_id=args.run_id,
            instance_id=args.instance_id,
            simulated_exit_code=args.simulated_exit_code,
            schedule_at=args.schedule_at,
            stop_timeout=args.stop_timeout,
            compshare=args.compshare,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"control_museg_lifecycle_test: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
