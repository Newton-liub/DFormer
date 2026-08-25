#!/usr/bin/env python3
"""Launch exactly one manifest-defined MUSeg seed and record structured evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.museg_protocol import (
        COMMAND_SCHEMA_VERSION,
        ENVIRONMENT_SCHEMA_VERSION,
        RUN_MANIFEST_SCHEMA_VERSION,
        ProtocolError,
        file_sha256,
        load_protocol,
        read_json,
        write_json,
    )
except ModuleNotFoundError:  # direct execution from tools/
    from museg_protocol import (  # type: ignore[no-redef]
        COMMAND_SCHEMA_VERSION,
        ENVIRONMENT_SCHEMA_VERSION,
        RUN_MANIFEST_SCHEMA_VERSION,
        ProtocolError,
        file_sha256,
        load_protocol,
        read_json,
        write_json,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-program", default="utils/train.py")
    parser.add_argument("--direct", action="store_true", help="run the trainer directly instead of torch.distributed.run")
    parser.add_argument("--resume")
    parser.add_argument("--resume-parent-run-id")
    parser.add_argument("--resume-checkpoint-sha256")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument("--output-dir", help="qualification-only isolated output override")
    parser.add_argument("--batch-size", type=int, help="qualification-only batch override")
    parser.add_argument("--max-train-iters", type=int, help="qualification-only bounded probe")
    parser.add_argument("--min-free-vram-gib", type=float, default=0.0)
    parser.add_argument("--min-free-vram-ratio", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="write launch records without starting the trainer")
    return parser.parse_args(argv)


def _git(repo_root: Path, generated_roots: Sequence[Path] = ()) -> dict[str, Any]:
    def is_generated(path: Path) -> bool:
        for root in generated_roots:
            try:
                path.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        return False

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, encoding="utf-8", errors="surrogateescape", stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo_root, encoding="utf-8", errors="surrogateescape", stderr=subprocess.DEVNULL
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
            cwd=repo_root,
            encoding="utf-8",
            errors="surrogateescape",
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}
    dirty_paths = []
    for record in porcelain.split("\0"):
        if not record:
            continue
        path_text = record[3:] if len(record) > 3 else record
        resolved = (repo_root / path_text).resolve()
        if not is_generated(resolved):
            dirty_paths.append(path_text)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def collect_environment(
    repo_root: Path,
    *,
    generated_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "numpy", "swanlab", "timm", "mmengine"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    driver = None
    try:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "--id=0"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if query:
            gpu, driver = [part.strip() for part in query.split(",", 1)]
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git": _git(repo_root, generated_roots),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
        "driver": driver,
        "cuda": None,
        "cudnn": None,
    }


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def build_training_argv(args: argparse.Namespace, protocol, run_dir: Path, run_id: str) -> list[str]:
    training = protocol.training
    train_role, val_role, test_role = protocol.phase_roles()
    if args.direct:
        command = [args.python, args.train_program]
    else:
        command = [
            args.python, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
            args.train_program,
        ]
    command += [
        "--config", protocol.config_module,
        "--gpus", "1",
        "--experiment-phase", protocol.phase,
        "--train-source", str(protocol.split_path(train_role)),
        "--test-source", str(protocol.split_path(test_role)),
        "--pretrained-model", str(protocol.resolve_declared_path(str(protocol.pretrained["path"]))),
        "--expected-train-split-sha256", str(protocol.splits[train_role]["sha256"]),
        "--expected-train-samples", str(protocol.splits[train_role]["samples"]),
        "--expected-test-split-sha256", str(protocol.splits[test_role]["sha256"]),
        "--expected-test-samples", str(protocol.splits[test_role]["samples"]),
        "--epochs", str(training["epochs"]),
        "--batch-size", str(args.batch_size if args.batch_size is not None else training["batch_size"]),
        "--val-batch-size", str(training["val_batch_size"]),
        "--workers", str(training["workers"]),
        "--eval-start-epoch", str(training["eval_start_epoch"]),
        "--eval-interval", str(training["eval_interval"]),
        "--save-interval", str(training["save_interval"]),
        "--seed", str(args.seed),
        "--run-id", run_id,
        "--protocol-id", protocol.protocol_id,
        "--schedule-version", protocol.schedule_version,
        "--protocol-manifest-sha256", protocol.manifest_sha256,
        "--required-git-commit", str(protocol.git["required_commit"]),
        "--output-dir", str(run_dir),
        "--checkpoint-dir", str(run_dir / "checkpoint"),
        _bool_flag("amp", bool(training["amp"])),
        _bool_flag("compile", bool(training["compile"])),
        _bool_flag("syncbn", bool(training["syncbn"])),
        _bool_flag("sliding", bool(training["sliding"])),
        _bool_flag("mst", bool(training["mst"])),
        "--use_seed",
        "--swanlab-mode", str(protocol.swanlab["mode"]),
        "--swanlab-project", str(protocol.swanlab.get("project", "DFormer-liu")),
        "--swanlab-workspace", str(protocol.swanlab.get("workspace", "Newton_liub")),
    ]
    if val_role:
        command += [
            "--val-source", str(protocol.split_path(val_role)),
            "--expected-val-split-sha256", str(protocol.splits[val_role]["sha256"]),
            "--expected-val-samples", str(protocol.splits[val_role]["samples"]),
        ]
    if args.max_train_iters is not None:
        command += ["--max-train-iters", str(args.max_train_iters)]
    if args.min_free_vram_gib > 0:
        command += ["--min-free-vram-gib", str(args.min_free_vram_gib)]
    if args.min_free_vram_ratio > 0:
        command += ["--min-free-vram-ratio", str(args.min_free_vram_ratio)]
    explicit_name = args.swanlab_run_name or protocol.swanlab.get("run_name") or run_id
    command += ["--swanlab-run-name", str(explicit_name)]
    if args.resume:
        command += [
            "--resume", str(Path(args.resume).resolve()),
            "--resume-parent-run-id", args.resume_parent_run_id,
            "--resume-checkpoint-sha256", args.resume_checkpoint_sha256.lower(),
        ]
    return command


def _validate_resume(args: argparse.Namespace) -> dict[str, Any] | None:
    supplied = [args.resume, args.resume_parent_run_id, args.resume_checkpoint_sha256]
    if not any(supplied):
        return None
    if not all(supplied):
        raise ProtocolError(
            "resume requires --resume, --resume-parent-run-id, and --resume-checkpoint-sha256 together"
        )
    checkpoint = Path(args.resume).resolve()
    if not checkpoint.is_file():
        raise ProtocolError(f"resume checkpoint does not exist: {checkpoint}")
    actual = file_sha256(checkpoint)
    if actual.lower() != args.resume_checkpoint_sha256.lower():
        raise ProtocolError(
            f"resume checkpoint SHA-256 mismatch: expected {args.resume_checkpoint_sha256}, got {actual}"
        )
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": actual, "parent_run_id": args.resume_parent_run_id}


def _validate_training_result(protocol, run_dir: Path, seed: int, run_id: str) -> None:
    result_path = run_dir / "training_result.json"
    if not result_path.is_file():
        raise ProtocolError(f"successful trainer did not write {result_path}")
    result = read_json(result_path)
    if not isinstance(result, dict) or result.get("schema_version") != "museg-training-result-v1":
        raise ProtocolError("training_result.json has an invalid schema")
    expected_identity = {
        "protocol_id": protocol.protocol_id,
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "phase": protocol.phase,
        "seed": seed,
        "run_id": run_id,
        "exit_code": 0,
        "official_test_included": False,
    }
    for field, expected in expected_identity.items():
        if result.get(field) != expected:
            raise ProtocolError(
                f"training_result.json {field} mismatch: expected {expected!r}, got {result.get(field)!r}"
            )
    if protocol.phase in {"development", "official"}:
        checkpoint = result.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ProtocolError(f"{protocol.phase} training_result.json must identify a checkpoint")
        path_value = checkpoint.get("path")
        sha_value = checkpoint.get("sha256")
        if not isinstance(path_value, str) or not isinstance(sha_value, str):
            raise ProtocolError("training_result.json checkpoint path/SHA-256 is invalid")
        checkpoint_path = Path(path_value).resolve()
        if not checkpoint_path.is_file():
            raise ProtocolError(f"training_result.json checkpoint is missing: {checkpoint_path}")
        actual_sha = file_sha256(checkpoint_path)
        if actual_sha.lower() != sha_value.lower():
            raise ProtocolError(
                f"training_result.json checkpoint SHA-256 mismatch: expected {sha_value}, got {actual_sha}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol_manifest)
        protocol.validate_consumed_splits()
        if args.seed not in protocol.seeds:
            raise ProtocolError(f"seed {args.seed} is not declared by the protocol manifest")
        if (args.batch_size is not None or args.max_train_iters is not None) and protocol.phase != "qualification":
            raise ProtocolError("batch/step overrides are restricted to qualification protocols")
        if args.batch_size is not None and args.batch_size <= 0:
            raise ProtocolError("--batch-size must be positive")
        if args.max_train_iters is not None and args.max_train_iters <= 0:
            raise ProtocolError("--max-train-iters must be positive")
        resume = _validate_resume(args)
        run_dir = Path(args.output_dir).resolve() if args.output_dir else protocol.seed_output_dir(args.seed)
        if args.output_dir and protocol.phase != "qualification":
            raise ProtocolError("--output-dir override is restricted to qualification protocols")
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ProtocolError(f"seed output directory is non-empty: {run_dir}")
        if run_dir.exists():
            run_dir.rmdir()
        run_id = (
            f"{protocol.protocol_id}-{protocol.phase}-{protocol.model}-"
            f"{protocol.schedule_version}-seed-{args.seed}"
        )
        command = build_training_argv(args, protocol, run_dir, run_id)
        protocol.run_root.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[1]
        command_record = {
            "schema_version": COMMAND_SCHEMA_VERSION,
            "cwd": str(repo_root),
            "argv": command,
        }
        environment = collect_environment(
            repo_root,
            generated_roots=(protocol.output_root, run_dir.parent),
        )
        started = dt.datetime.now(dt.timezone.utc)
        start_clock = time.monotonic()
        descriptor, temporary_log = tempfile.mkstemp(
            prefix=f".seed-{args.seed}-", suffix=".launcher.log", dir=protocol.run_root
        )
        os.close(descriptor)
        exit_code = 0
        process_exit_code = 0
        launch_error = None
        evidence_error = None
        try:
            if not args.dry_run:
                with Path(temporary_log).open("w", encoding="utf-8", newline="\n") as log:
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=repo_root,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                        process_exit_code = int(completed.returncode)
                        exit_code = process_exit_code
                    except (OSError, subprocess.SubprocessError) as exc:
                        launch_error = f"{type(exc).__name__}: {exc}"
                        log.write(f"launcher error: {launch_error}\n")
                        process_exit_code = 127
                        exit_code = process_exit_code
            run_dir.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_log, run_dir / "launcher.log")
        finally:
            Path(temporary_log).unlink(missing_ok=True)
        if exit_code == 0 and not args.dry_run:
            try:
                _validate_training_result(protocol, run_dir, args.seed, run_id)
            except (OSError, ProtocolError) as exc:
                evidence_error = str(exc)
                exit_code = 3
        finished = dt.datetime.now(dt.timezone.utc)
        write_json(run_dir / "command.json", command_record)
        write_json(run_dir / "environment.json", environment)
        write_json(
            run_dir / "train.exit_code",
            {"schema_version": "museg-exit-code-v1", "exit_code": exit_code},
        )
        write_json(
            run_dir / "run_manifest.json",
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "protocol_schema_version": protocol.schema_version,
                "protocol_id": protocol.protocol_id,
                "protocol_manifest": str(protocol.path),
                "protocol_manifest_sha256": protocol.manifest_sha256,
                "schedule_version": protocol.schedule_version,
                "phase": protocol.phase,
                "model": protocol.model,
                "git": environment["git"],
                "split_authority": protocol.authority_identity(),
                "splits": protocol.splits,
                "pretrained": protocol.pretrained,
                "seed": args.seed,
                "run_id": run_id,
                "output_dir": str(run_dir),
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
                "duration_seconds": time.monotonic() - start_clock,
                "exit_code": exit_code,
                "process_exit_code": process_exit_code,
                "launch_error": launch_error,
                "evidence_error": evidence_error,
                "resume": resume,
            },
        )
        return exit_code
    except (OSError, ProtocolError, subprocess.SubprocessError) as exc:
        print(f"run_museg_seed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())