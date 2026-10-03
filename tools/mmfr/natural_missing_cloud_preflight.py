"""Sequential cloud wrapper for the bounded NaturalMissing preflight."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_tracker import ExperimentTracker

STRATEGIES = ("Natural", "Grid", "Replay")
C0_SHA256 = "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a"
SUCCESSFUL_UPDATES = 3
PREFLIGHT_WORKERS = 8
RUN_NAME = "NaturalMissing-4090-preflight"
COUNTER_FIELDS = ("attempted_steps", "successful_updates", "skipped_steps")
SCALAR_FIELDS = (
    "attempted_steps",
    "successful_updates",
    "skipped_steps",
    "loss",
    "lr",
    "amp_scale",
    "epoch",
    "last_batch_index",
    "elapsed_seconds",
    "peak_memory_mb",
    "peak_reserved_memory_mb",
)
RESULT_FIELDS = (
    "mode",
    "strategy",
    "successful_updates",
    "attempted_steps",
    "skipped_steps",
    "attempt_cap",
    "complete",
    "preflight_worker_override",
    "preflight_checkpoint_formal_eligible",
    "elapsed_seconds",
    "gpu_name",
    "checkpoint_dir",
    "telemetry",
)


class _PreflightInterrupted(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tracking-mode", choices=("online", "disabled"), default="disabled")
    return parser


def _source_commit() -> tuple[str, str | None]:
    for name in ("SOURCE_COMMIT", "GIT_COMMIT", "CI_COMMIT_SHA", "GITHUB_SHA", "BUILD_SOURCEVERSION"):
        value = os.environ.get(name, "").strip()
        if value:
            return value, name
    return "unknown", None


def _finite_scalar(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _step_record(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    if payload.get("event") != "attempt_started" and "loss" not in payload and "complete" not in payload:
        return None
    record: dict[str, Any] = {}
    if payload.get("event") is not None:
        record["event"] = str(payload["event"])
    for key in SCALAR_FIELDS:
        value = payload.get(key)
        if _finite_scalar(value):
            record[key] = value
    for key in ("complete", "loss_finite", "optimizer_step_applied"):
        if isinstance(payload.get(key), bool):
            record[key] = payload[key]
    return record


def _compact_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in RESULT_FIELDS if key in payload}


def _entry_counter(entry: Mapping[str, Any], key: str) -> int:
    result = entry.get("result")
    if isinstance(result, Mapping) and isinstance(result.get(key), int):
        return int(result[key])
    values = [
        int(record[key])
        for record in entry.get("step_records", [])
        if isinstance(record, Mapping) and isinstance(record.get(key), int)
    ]
    return max(values, default=0)


def _save_receipt(path: Path, receipt: dict[str, Any]) -> None:
    strategies = receipt["strategies"]
    aggregate = {
        "status_by_strategy": {name: item["status"] for name, item in strategies.items()},
        "succeeded": sum(item["status"] == "SUCCEEDED" for item in strategies.values()),
        "failed": sum(item["status"] == "FAILED" for item in strategies.values()),
        "not_run": sum(item["status"] == "NOT_RUN" for item in strategies.values()),
        "interrupted": sum(item["status"] == "INTERRUPTED" for item in strategies.values()),
    }
    for key in COUNTER_FIELDS:
        aggregate[key] = sum(_entry_counter(item, key) for item in strategies.values())
    receipt["aggregate"] = aggregate
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _terminate_child(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _consume_child(
    strategy: str,
    checkpoint: Path,
    log_path: Path,
    payload_handler: Callable[[Mapping[str, Any]], None],
) -> tuple[int, dict[str, Any] | None]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.mmfr.natural_missing_train",
        "--mode",
        "preflight",
        "--strategy",
        strategy,
        "--c0-checkpoint",
        str(checkpoint),
        "--verified-c0-sha256",
        C0_SHA256,
        "--successful-updates",
        str(SUCCESSFUL_UPDATES),
        "--preflight-workers",
        str(PREFLIGHT_WORKERS),
    ]
    process: subprocess.Popen[str] | None = None
    final_result: dict[str, Any] | None = None
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{strategy}] {line.rstrip()}" if line.rstrip() else f"[{strategy}]", flush=True)
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict) or payload.get("strategy") != strategy:
                    continue
                if payload.get("mode") == "preflight" and "complete" in payload:
                    final_result = _compact_result(payload)
                payload_handler(payload)
            exit_code = process.wait()
            return exit_code, final_result
    finally:
        if process is not None:
            _terminate_child(process)
            if process.stdout is not None:
                process.stdout.close()


def _finish_tracker(tracker: ExperimentTracker, exit_code: int) -> bool:
    swanlab = getattr(tracker, "_swanlab", None)
    finish = getattr(swanlab, "finish", None)
    if callable(finish):
        try:
            supports_exit_code = "exit_code" in inspect.signature(finish).parameters
        except (TypeError, ValueError):
            supports_exit_code = False
        if supports_exit_code:
            try:
                finish(exit_code=exit_code)
            finally:
                tracker._finished = True
            return True
    tracker.finish()
    return False


def _mark_unstarted_not_run(receipt: dict[str, Any]) -> None:
    for item in receipt["strategies"].values():
        if item["status"] == "PENDING":
            item["status"] = "NOT_RUN"


def run(args: argparse.Namespace) -> int:
    checkpoint = Path(args.c0_checkpoint).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("C0 checkpoint path is not a file")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "preflight-result.json"
    source_commit, source_commit_env = _source_commit()
    project = os.environ.get("SWANLAB_PROJ_NAME") or os.environ.get("SWANLAB_PROJECT") or "DFormer-liu"
    workspace = os.environ.get("SWANLAB_WORKSPACE") or "Newton_liub"
    log_paths = {strategy: output_dir / f"{strategy}.log" for strategy in STRATEGIES}
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "status": "RUNNING",
        "source_commit": source_commit,
        "source_commit_env": source_commit_env,
        "c0_checkpoint": str(checkpoint),
        "c0_sha256_verified": C0_SHA256,
        "tracking": {
            "mode_requested": args.tracking_mode,
            "status": "DISABLED" if args.tracking_mode == "disabled" else "INITIALIZING",
            "project": project,
            "workspace": workspace,
            "initialization_exception_type": None,
            "finish_exit_code_supported": None,
        },
        "strategies": {
            strategy: {
                "status": "PENDING",
                "log_file": str(log_paths[strategy]),
                "exit_code": None,
                "failure_reason": None,
                "result": None,
                "step_records": [],
            }
            for strategy in STRATEGIES
        },
    }
    _save_receipt(receipt_path, receipt)

    tracker = ExperimentTracker()
    terminal_reason: str | None = None
    current_strategy: str | None = None
    monitor_step = 0
    old_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        raise _PreflightInterrupted(signal.Signals(signum).name)

    for signal_name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            try:
                old_handlers[signum] = signal.signal(signum, handle_signal)
            except (OSError, ValueError):
                pass

    try:
        try:
            tracker.start(
                mode=args.tracking_mode,
                is_primary=True,
                project=project,
                workspace=workspace,
                name=RUN_NAME,
                log_dir=str(output_dir / "swanlab"),
                config={
                    "run_type": "bounded_cloud_preflight",
                    "strategies": list(STRATEGIES),
                    "successful_updates_per_strategy": SUCCESSFUL_UPDATES,
                    "preflight_workers": PREFLIGHT_WORKERS,
                    "c0_sha256_verified": C0_SHA256,
                    "source_commit": source_commit,
                },
            )
            if args.tracking_mode == "online":
                receipt["tracking"]["status"] = "ONLINE"
        except Exception as exc:
            if args.tracking_mode != "online":
                raise
            receipt["tracking"]["status"] = "LOG_ONLY"
            receipt["tracking"]["initialization_exception_type"] = type(exc).__name__
            print(f"[preflight] optional SwanLab initialization unavailable: {type(exc).__name__}", flush=True)
            _save_receipt(receipt_path, receipt)

        for index, strategy in enumerate(STRATEGIES):
            current_strategy = strategy
            entry = receipt["strategies"][strategy]
            entry["status"] = "RUNNING"
            _save_receipt(receipt_path, receipt)

            def handle_payload(payload: Mapping[str, Any], *, name: str = strategy) -> None:
                nonlocal monitor_step
                record = _step_record(payload)
                if record is not None:
                    receipt["strategies"][name]["step_records"].append(record)
                    _save_receipt(receipt_path, receipt)
                metrics = {
                    f"{name}/{key}": value
                    for key in SCALAR_FIELDS
                    if _finite_scalar(value := payload.get(key))
                }
                if metrics:
                    monitor_step += 1
                    tracker.log(metrics, step=monitor_step)

            exit_code, child_result = _consume_child(
                strategy,
                checkpoint,
                log_paths[strategy],
                handle_payload,
            )
            entry["exit_code"] = exit_code
            entry["result"] = child_result
            if exit_code != 0:
                failure_reason = "child_exit_nonzero"
            elif child_result is None or child_result.get("complete") is not True:
                failure_reason = "missing_or_incomplete_result"
            elif any(child_result.get(key) != expected for key, expected in (
                ("successful_updates", SUCCESSFUL_UPDATES),
                ("attempted_steps", SUCCESSFUL_UPDATES),
                ("skipped_steps", 0),
            )):
                failure_reason = "counter_mismatch"
            else:
                failure_reason = None
            if failure_reason is None:
                entry["status"] = "SUCCEEDED"
            else:
                entry["status"] = "FAILED"
                entry["failure_reason"] = failure_reason
                receipt["status"] = "FAILED"
                terminal_reason = failure_reason
                for remaining in STRATEGIES[index + 1 :]:
                    receipt["strategies"][remaining]["status"] = "NOT_RUN"
                _save_receipt(receipt_path, receipt)
                break
            _save_receipt(receipt_path, receipt)
            current_strategy = None
        if receipt["status"] == "RUNNING":
            receipt["status"] = "SUCCEEDED"
    except _PreflightInterrupted as exc:
        receipt["status"] = "INTERRUPTED"
        terminal_reason = exc.reason
        if current_strategy and receipt["strategies"][current_strategy]["status"] == "RUNNING":
            receipt["strategies"][current_strategy]["status"] = "INTERRUPTED"
        _mark_unstarted_not_run(receipt)
    except BaseException as exc:
        receipt["status"] = "INTERRUPTED"
        terminal_reason = type(exc).__name__
        receipt["exception_type"] = type(exc).__name__
        if current_strategy and receipt["strategies"][current_strategy]["status"] == "RUNNING":
            receipt["strategies"][current_strategy]["status"] = "INTERRUPTED"
        _mark_unstarted_not_run(receipt)
        print(f"[preflight] interrupted: {type(exc).__name__}", flush=True)
    finally:
        _mark_unstarted_not_run(receipt)
        if terminal_reason is not None:
            receipt["terminal_reason"] = terminal_reason
        status = receipt["status"]
        exit_code = 0 if status == "SUCCEEDED" else 1
        succeeded = status == "SUCCEEDED"
        terminal_metrics = {
            "terminal/succeeded": int(succeeded),
            "terminal/failed": int(status == "FAILED"),
            "terminal/interrupted": int(status == "INTERRUPTED"),
            "terminal/exit_code": exit_code,
        }
        receipt["terminal_metrics"] = terminal_metrics
        try:
            monitor_step += 1
            tracker.log(terminal_metrics, step=monitor_step)
        except Exception as exc:
            receipt["tracking"]["terminal_log_exception_type"] = type(exc).__name__
        _save_receipt(receipt_path, receipt)
        try:
            receipt["tracking"]["finish_exit_code_supported"] = _finish_tracker(tracker, exit_code)
        except Exception as exc:
            receipt["tracking"]["finish_exception_type"] = type(exc).__name__
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)
        _save_receipt(receipt_path, receipt)

    return 0 if receipt["status"] == "SUCCEEDED" else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"[preflight] interrupted: {type(exc).__name__}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
