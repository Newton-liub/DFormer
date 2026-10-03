"""Durable, Natural-only numerical diagnosis from the frozen C0 (never formal)."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tools.mmfr.natural_missing_formal import bounded_call, save_json, terminate
from utils.experiment_tracker import ExperimentTracker

REPO = Path(__file__).resolve().parents[2]
C0_SHA = "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--successful-updates", type=int, default=1664)
    parser.add_argument("--diagnostic-start-attempt", type=int, default=1536)
    args = parser.parse_args()
    if not 1 <= args.diagnostic_start_attempt <= args.successful_updates <= 1664:
        parser.error("diagnosis requires 1 <= observation start <= update cap <= 1664")
    root = args.output_dir.resolve()
    path = root / "numerical-run-receipt.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite an earlier numerical receipt")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True)
    if os.environ.get("SOURCE_COMMIT") != sha or dirty.strip():
        raise RuntimeError("numerical run requires matching SOURCE_COMMIT and tracked-clean Git")
    c0 = args.c0_checkpoint.resolve(strict=True)
    if c0.stat().st_size != 321150608:
        raise RuntimeError("frozen original C0 size changed")
    root.mkdir(parents=True, exist_ok=True)
    tracking = {"status": "INITIALIZING", "attempt_limit": 1,
                "credential_environment_present": bool(os.environ.get("SWANLAB_API_KEY")),
                "secret_values_recorded": False}
    receipt = {"schema_version": 1, "status": "RUNNING", "mode": "diagnosis", "strategy": "Natural",
               "source_commit": sha, "c0_checkpoint": str(c0), "c0_sha256_verified": C0_SHA,
               "started_at_unix": time.time(), "update_cap": args.successful_updates,
               "observation_start_attempt": args.diagnostic_start_attempt,
               "formal_checkpoint_eligible": False, "tracking": tracking,
               "last_attempt": None, "last_success": None, "nonfinite_events": []}
    save_json(path, receipt)
    tracker = ExperimentTracker()
    process = None
    previous = {}
    def interrupted(signum, _frame):
        raise InterruptedError(signal.Signals(signum).name)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupted)
    code = 1
    try:
        # SDK console output can contain authentication prompts; discard it. Persist
        # only exception class and SDK stack locations, never credential contents.
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                bounded_call(45, lambda: tracker.start(
                    mode="online", project=os.environ.get("SWANLAB_PROJ_NAME", "DFormer-liu"),
                    workspace=os.environ.get("SWANLAB_WORKSPACE", "Newton_liub"),
                    name="NaturalMissing-Natural-numerical-diagnosis", log_dir=str(root / "swanlab"),
                    config={"source_commit": sha, "mode": "diagnosis", "update_cap": args.successful_updates}))
            tracking["status"] = "ONLINE"
        except Exception as exc:
            tracking.update(status="LOG_ONLY", error_type=type(exc).__name__,
                            error_frames=[{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                                          for frame in traceback.extract_tb(exc.__traceback__)])
        save_json(path, receipt)
        print("SWANLAB_STATUS", tracking["status"], flush=True)
        command = [sys.executable, "-u", "-m", "tools.mmfr.natural_missing_train", "--mode", "diagnosis",
                   "--strategy", "Natural", "--c0-checkpoint", str(c0), "--verified-c0-sha256", C0_SHA,
                   "--successful-updates", str(args.successful_updates),
                   "--diagnostic-start-attempt", str(args.diagnostic_start_attempt)]
        with (root / "Natural.stdout.log").open("x", encoding="utf-8") as stdout, \
                (root / "Natural.stderr.log").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=stderr, text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in process.stdout:
                stdout.write(line)
                stdout.flush()
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("event") == "attempt_started":
                    receipt["last_attempt"] = payload
                if payload.get("optimizer_step_applied") is True:
                    receipt["last_success"] = {key: value for key, value in payload.items() if key != "telemetry"}
                    successful = payload["successful_updates"]
                    if successful % 128 == 0:
                        print(f"DIAGNOSIS successful={successful}/{args.successful_updates} loss={payload['loss']:.6f}", flush=True)
                        save_json(path, receipt)
                        if tracking["status"] == "ONLINE":
                            try:
                                bounded_call(10, lambda: tracker.log(
                                    {key: payload[key] for key in ("loss", "lr", "amp_scale", "successful_updates")},
                                    step=successful))
                            except Exception as exc:
                                tracking.update(status="LOG_ONLY", log_error_type=type(exc).__name__)
                if payload.get("event") == "training_attempt_failed":
                    receipt["failed_attempt"] = payload
                    save_json(path, receipt)
                elif "nonfinite" in str(payload.get("event", "")).lower():
                    receipt["nonfinite_events"].append(payload)
                    save_json(path, receipt)
                if payload.get("mode") == "diagnosis" and "complete" in payload:
                    receipt["result"] = payload
            code = process.wait()
        receipt["child_exit_code"] = code
        receipt["status"] = "CHILD_FAILED" if code else "CAP_REACHED_FINITE"
        if not code and not receipt.get("result", {}).get("complete"):
            raise RuntimeError("diagnostic child exited without a complete capped result")
    except BaseException as exc:
        receipt.update(status="INTERRUPTED" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED",
                       failure_type=type(exc).__name__)
        code = 1
    finally:
        if process is not None:
            terminate(process)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                bounded_call(15, tracker.finish)
            tracking["finish_returned"] = True
        except Exception as exc:
            tracking["finish_error_type"] = type(exc).__name__
        receipt["finished_at_unix"] = time.time()
        save_json(path, receipt)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        print(f"NUMERICAL_TERMINAL status={receipt['status']} exit={code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
