"""Bounded formal Round-1 coordinator; the scientific runners stay unchanged."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from utils.experiment_tracker import ExperimentTracker

REPO = Path(__file__).resolve().parents[2]
STRATEGIES = ("Natural", "Grid", "Replay")
C0_SHA = "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a"
CONDITIONS = {"natural_original", "rectangle_add50_current_valid", "entire_missing"}
METRICS = ("attempted_steps", "successful_updates", "skipped_steps", "loss", "lr", "amp_scale",
           "elapsed_seconds", "peak_memory_mb", "peak_reserved_memory_mb")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def bounded_call(seconds: int, function):
    """Online monitoring has a small independent timeout on the Linux runner."""
    def timeout(_signum, _frame):
        raise TimeoutError("optional monitoring time budget exhausted")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(seconds)
    try:
        return function()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def tracking_failure(tracking: dict, stage: str, exc: Exception) -> None:
    tracking["status"] = "LOG_ONLY"
    # Exception strings can contain credentials; classify without persisting them.
    text = str(exc).lower()
    reason = "authentication" if any(word in text for word in ("login", "token", "api_key", "credential")) else "initialization_or_transport"
    tracking[f"{stage}_error"] = {"type": type(exc).__name__, "category": reason, "detail": "redacted"}
    print(f"SWANLAB=LOG_ONLY stage={stage} type={type(exc).__name__} category={reason}", flush=True)


class InputSummary:
    """Aggregate the runner's actual slots schema once per attempted batch."""
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.counts = {"clean": 0, "matched": 0, "paired_skip": 0}
        self.numbers = {}
        self.slots = 0
        self.validated_slots = 0
        self.group_checks = 0
        self.plan = hashlib.sha256()
        self.targets = hashlib.sha256()

    def number(self, name: str, value):
        if value is None:
            return
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                f"non-finite telemetry {name}")
        entry = self.numbers.setdefault(name, {"count": 0, "sum": 0, "min": value, "max": value})
        entry["count"] += 1
        entry["sum"] += value
        entry["min"] = min(entry["min"], value)
        entry["max"] = max(entry["max"], value)

    def consume(self, payload: dict):
        telemetry = payload["telemetry"]
        slots = telemetry["slots"]
        require(len(slots) == 10, "input batch slot count drift")
        for slot in slots:
            self.slots += 1
            status = slot["status"]
            category = "clean" if status == "clean_control" else status
            require(category in self.counts, "unknown input status")
            self.counts[category] += 1
            applied = slot["applied_deletion_count"]
            self.number("actual_deletion_pixels", applied)
            self.number("actual_deletion_rate", slot.get("applied_rate"))
            if category != "matched":
                require(applied == 0 and slot.get("applied_rate") in (None, 0), "clean/skip input deletion drift")
            if self.strategy == "Natural":
                require(status == "clean_control" and applied == 0, "Natural input changed")
            else:
                target_group = slot["target_group"]
                source = slot.get("source_group")
                if source is not None:
                    self.group_checks += 1
                    require(source != target_group, "source and target collection group overlap")
                for candidate in slot["candidate_sources"]:
                    self.group_checks += 1
                    require(candidate["group"] != target_group, "candidate collection group overlap")
                require(0 <= slot["attempts"] <= 8, "source candidate budget drift")
                if category == "matched":
                    error = slot["matching_error"]
                    replay = slot["replay_rate"]
                    require(source is not None and 0 <= error <= .02 and .10 <= replay <= .50,
                            "matched input budget or group contract drift")
                    expected_count = slot["grid_deletion_count" if self.strategy == "Grid" else "replay_deletion_count"]
                    require(applied == expected_count, "actual deletion differs from selected plan")
                    self.number("matched_error", error)
                    self.number("matched_replay_rate", replay)
                    self.number("matched_grid_rate", slot["grid_rate"])
                # Selected execution fields differ by strategy; the paired plan does not.
                common = {key: value for key, value in slot.items()
                          if key not in {"strategy", "applied_rate", "applied_deletion_count"}}
                self.plan.update(json.dumps([payload["attempted_steps"], common], sort_keys=True).encode())
            self.targets.update(json.dumps([payload["attempted_steps"], telemetry["epoch"],
                                            slot["batch_slot"], slot["target_sample"], slot["target_group"]]).encode())
            self.validated_slots += 1

    def result(self) -> dict:
        numbers = {key: {**value, "mean": value["sum"] / value["count"]}
                   for key, value in self.numbers.items()}
        return {"slots": self.slots, "slot_counts": self.counts, "statistics": numbers,
                "source_target_group_checks": self.group_checks,
                "contract_violations": self.slots - self.validated_slots,
                "validated_slots": self.validated_slots,
                "target_order_fingerprint": self.targets.hexdigest(),
                "paired_plan_fingerprint": self.plan.hexdigest() if self.strategy != "Natural" else None,
                "statistics_scope": "once per attempt; matched rates/errors only; failed candidates not treated as applied"}


def terminate(process: subprocess.Popen):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def train(strategy: str, args, root: Path, entry: dict, tracker, tracking: dict, receipt: dict, receipt_path: Path):
    summary = InputSummary(strategy)
    started = time.monotonic()
    command = [sys.executable, "-u", "-m", "tools.mmfr.natural_missing_train", "--mode", "formal",
               "--strategy", strategy, "--c0-checkpoint", str(args.c0_checkpoint),
               "--verified-c0-sha256", C0_SHA, "--successful-updates", "2560", "--authorize-formal-training"]
    final = None
    last = {}
    peak = {"allocated_mb": 0., "reserved_mb": 0.}
    successful = 0
    attempt = 0
    with (root / f"{strategy}.stdout.log").open("x", encoding="utf-8") as stdout, \
            (root / f"{strategy}.stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                stdout.write(line)
                stdout.flush()
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    if line.strip():
                        print(f"[{strategy}] {line.rstrip()}", flush=True)
                    continue
                if not isinstance(payload, dict) or payload.get("mode") != "formal" or payload.get("strategy") != strategy:
                    continue
                if payload.get("event") == "attempt_started":
                    attempt = payload["attempted_steps"]
                    require(attempt == successful + 1, "attempt counter order mismatch")
                    summary.consume(payload)
                elif payload.get("optimizer_step_applied") is True:
                    successful = payload["successful_updates"]
                    require(payload["loss_finite"] is True and payload["skipped_steps"] == 0
                            and successful == attempt, "optimizer counter/finite contract mismatch")
                    last = {key: payload[key] for key in METRICS if key in payload}
                    peak["allocated_mb"] = max(peak["allocated_mb"], payload["peak_memory_mb"])
                    peak["reserved_mb"] = max(peak["reserved_mb"], payload["peak_reserved_memory_mb"])
                    print(f"TRAIN strategy={strategy} attempted={attempt} successful={successful}/2560 skipped=0 "
                          f"loss={payload['loss']:.6f} lr={payload['lr']:.8g} elapsed={time.monotonic()-started:.1f}s "
                          f"peak_VRAM={peak['allocated_mb']:.1f}MiB", flush=True)
                    if tracking["status"] == "ONLINE":
                        try:
                            bounded_call(15, lambda: tracker.log({f"{strategy}/{key}": value for key, value in last.items()},
                                                                  step=STRATEGIES.index(strategy) * 2560 + successful))
                        except Exception as exc:
                            tracking_failure(tracking, "log", exc)
                    if successful % 128 == 0:
                        entry["progress"] = last
                        save_json(receipt_path, receipt)
                if payload.get("complete") is True and "checkpoint_dir" in payload:
                    final = payload
            entry["exit_code"] = process.wait()
            require(entry["exit_code"] == 0 and final is not None, "training child failed or final result missing")
            for name, expected in {"successful_updates": 2560, "attempted_steps": 2560,
                                   "skipped_steps": 0, "complete": True, "formal_config_num_workers": 8}.items():
                require(final[name] == expected, f"final training contract mismatch: {name}")
            directory = Path(os.environ["DFORMER_OUTPUT_ROOT"]) / f"NaturalMissing-{strategy}-formal" / "development" / "seed-772961337" / "checkpoint"
            require(Path(final["checkpoint_dir"]).resolve() == directory.resolve(), "formal checkpoint directory mismatch")
            entry["checkpoints"] = {}
            for update in (640, 1280, 1920, 2560):
                checkpoint = directory / f"update-{update}.pth"
                require(checkpoint.is_file() and checkpoint.stat().st_size > 300_000_000, "recovery/final checkpoint missing or too small")
                entry["checkpoints"][str(update)] = str(checkpoint)
                print(f"CHECKPOINT strategy={strategy} update={update} path={checkpoint}", flush=True)
            from tools.mmfr.natural_missing_formal_artifacts import read_final
            entry["final_readback"] = read_final(directory / "update-2560.pth", strategy,
                                                  receipt["source_commit"])
            save_json(root / f"{strategy}-final-readback.json", entry["final_readback"])
            print(f"FINAL_READBACK strategy={strategy} passed=true "
                  f"sha256={entry['final_readback']['sha256']} CPU_ONLY_NO_MODEL_FORWARD", flush=True)
            entry["result"] = {key: value for key, value in final.items() if key != "telemetry"}
            entry["status"] = "SUCCEEDED"
            return directory / "update-2560.pth"
        finally:
            terminate(process)
            entry["wall_seconds"] = time.monotonic() - started
            entry["last_step"] = last
            entry["peak_vram"] = peak
            entry["input_telemetry"] = summary.result()
            entry["attempted_observed"] = attempt
            entry["successful_observed"] = successful
            save_json(root / f"{strategy}-training-summary.json", entry)


def evaluate(args, root: Path, finals: list[Path], entry: dict, final_hashes: dict[Path, str]):
    from tools.mmfr.natural_missing_eval import run_evaluation
    print("S1_START checkpoints=4 samples=318 conditions=3 expected_views=3816 FP32 TF32_OFF", flush=True)
    started = time.monotonic()
    with (root / "S1.stdout.log").open("x", encoding="utf-8") as stdout, \
            (root / "S1.stderr.log").open("x", encoding="utf-8") as stderr, \
            contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        paths = run_evaluation(dataset_root=args.dataset_root, checkpoints=[args.c0_checkpoint, *finals],
                               output_dir=root / "s1", sample_limit=318, formal_dev_authorization=True,
                               device_name="cuda", verified_checkpoint_sha256={args.c0_checkpoint: C0_SHA, **final_hashes})
    require(len(paths) == 4 and len(set(paths)) == 4, "S1 four-report completeness mismatch")
    reports = []
    for path, checkpoint in zip(paths, [args.c0_checkpoint, *finals]):
        report = json.loads(path.read_text(encoding="utf-8"))
        require(report["status"] == "completed" and report["official_test_included"] is False
                and Path(report["identity"]["checkpoint"]).resolve() == checkpoint.resolve()
                and report["identity"]["sample_count"] == 318, "S1 identity/status mismatch")
        require(set(report["s1_conditions"]) == CONDITIONS
                and all(value["sample_count"] == 318 and len(value["sample_observations"]) == 318
                        for value in report["s1_conditions"].values()), "S1 condition completeness mismatch")
        require(report["forward_rng_policy"]["counts"]["forward_calls"] == 954, "S1 forward completeness mismatch")
        expected_sha = C0_SHA if checkpoint == args.c0_checkpoint else final_hashes[checkpoint]
        require(report["identity"]["checkpoint_sha256"] == expected_sha, "S1 verified final identity mismatch")
        reports.append({"path": str(path), "checkpoint": str(checkpoint),
                        "sha256": report["identity"]["checkpoint_sha256"], "views": 954})
    require(reports[0]["sha256"] == C0_SHA, "S1 original C0 identity mismatch")
    entry.update(status="SUCCEEDED", reports=reports, total_views=3816, wall_seconds=time.monotonic()-started)
    save_json(root / "S1-summary.json", entry)
    print(f"S1_COMPLETE views=3816 elapsed={entry['wall_seconds']:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tracking-mode", choices=("online", "disabled"), default="online")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    receipt_path = root / "formal-run-receipt.json"
    require(not receipt_path.exists(), "refusing duplicate formal receipt")
    args.c0_checkpoint = args.c0_checkpoint.resolve(strict=True)
    args.dataset_root = args.dataset_root.resolve(strict=True)
    require(args.c0_checkpoint.stat().st_size == 321150608, "original C0 file size changed")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    require(os.environ.get("SOURCE_COMMIT") == sha, "formal-run source SHA mismatch")
    os.environ.setdefault("DFORMER_OUTPUT_ROOT", str(root.parent / "checkpoints"))
    os.environ["DFORMER_DATA_ROOT"] = str(args.dataset_root.parent)
    os.environ["MUSEG_DEPTH16_ROOT"] = str(args.dataset_root / "Depth16")
    tracking = {"status": "INITIALIZING", "mode_requested": args.tracking_mode,
                "project": os.environ.get("SWANLAB_PROJ_NAME", "DFormer-liu"),
                "workspace": os.environ.get("SWANLAB_WORKSPACE", "Newton_liub")}
    receipt = {"schema_version": 1, "run_name": "NaturalMissing-Round1-formal", "status": "RUNNING",
               "source_commit": sha, "c0_checkpoint": str(args.c0_checkpoint), "c0_sha256_verified": C0_SHA,
               "tracking": tracking, "started_at_unix": time.time(), "stage": "INITIALIZING",
               "strategies": {strategy: {"status": "PENDING", "strategy": strategy} for strategy in STRATEGIES},
               "evaluation": {"status": "PENDING"}}
    save_json(receipt_path, receipt)
    previous = {}
    def interrupted(signum, _frame):
        raise InterruptedError(signal.Signals(signum).name)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupted)
    tracker = ExperimentTracker()
    current = None
    exit_code = 1
    try:
        if args.tracking_mode == "online":
            try:
                bounded_call(45, lambda: tracker.start(mode="online", is_primary=True, project=tracking["project"],
                             workspace=tracking["workspace"], name=receipt["run_name"], log_dir=str(root / "swanlab"),
                             config={"source_commit": sha, "c0_sha256": C0_SHA, "updates_per_group": 2560, "s1_views": 3816}))
                tracking["status"] = "ONLINE"
                swan = getattr(tracker, "_swanlab", None)
                getter = getattr(swan, "get_run", None)
                if callable(getter):
                    run = getter()
                    for name in ("id", "url"):
                        value = getattr(run, name, None)
                        if isinstance(value, str):
                            tracking[f"run_{name}"] = value
            except Exception as exc:
                tracking_failure(tracking, "init", exc)
        else:
            tracker.start(mode="disabled")
            tracking["status"] = "DISABLED"
        finals = []
        for strategy in STRATEGIES:
            current = receipt["strategies"][strategy]
            current["status"] = "RUNNING"
            receipt["stage"] = f"TRAINING_{strategy.upper()}"
            save_json(receipt_path, receipt)
            print(f"TRAIN_START strategy={strategy} C0={args.c0_checkpoint} budget=2560", flush=True)
            finals.append(train(strategy, args, root, current, tracker, tracking, receipt, receipt_path))
            save_json(receipt_path, receipt)
        telemetry = {name: receipt["strategies"][name]["input_telemetry"] for name in STRATEGIES}
        require(len({value["target_order_fingerprint"] for value in telemetry.values()}) == 1,
                "three-strategy target order pairing drift")
        require(telemetry["Grid"]["paired_plan_fingerprint"] == telemetry["Replay"]["paired_plan_fingerprint"],
                "Grid/Replay input plan pairing drift")
        save_json(root / "input-telemetry-summary.json", telemetry)
        current = receipt["evaluation"]
        current["status"] = "RUNNING"
        receipt["stage"] = "S1_EVALUATION"
        save_json(receipt_path, receipt)
        final_hashes = {path: receipt["strategies"][strategy]["final_readback"]["sha256"]
                        for strategy, path in zip(STRATEGIES, finals)}
        evaluate(args, root, finals, current, final_hashes)
        receipt["status"] = "SUCCEEDED"
        receipt["stage"] = "COMPLETE"
        exit_code = 0
    except BaseException as exc:
        receipt["status"] = "INTERRUPTED" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED"
        receipt["failure"] = {"type": type(exc).__name__, "reason": str(exc)[:300]}
        if current is not None and current["status"] == "RUNNING":
            current["status"] = receipt["status"]
            if "strategy" in current:
                save_json(root / f"{current['strategy']}-training-summary.json", current)
        for entry in receipt["strategies"].values():
            if entry["status"] == "PENDING":
                entry["status"] = "NOT_RUN"
        print(f"FORMAL_{receipt['status']} {receipt['failure']}", flush=True)
    finally:
        receipt["finished_at_unix"] = time.time()
        save_json(receipt_path, receipt)
        try:
            if tracking["status"] == "ONLINE":
                bounded_call(15, lambda: tracker.log({"terminal/exit_code": exit_code,
                                                      "terminal/succeeded": int(exit_code == 0)}, step=7681))
            bounded_call(30, tracker.finish)
            tracking["finish_returned"] = True
        except Exception as exc:
            tracking_failure(tracking, "finish", exc)
        save_json(receipt_path, receipt)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        print(f"FORMAL_TERMINAL status={receipt['status']} exit={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
