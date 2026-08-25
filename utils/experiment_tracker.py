"""Small, optional SwanLab adapter and training-operation helpers."""

from __future__ import annotations

import datetime
import importlib
import subprocess
from pathlib import Path
from typing import Any, Mapping


def build_run_name(
    dataset: str,
    backbone: str,
    *,
    explicit_name: str | None = None,
    now: datetime.datetime | None = None,
    git_commit: str | None = None,
    repo_root: str | Path | None = None,
) -> str:
    """Return an explicit run name or a reproducible automatic name."""
    if explicit_name:
        return explicit_name
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    timestamp = now.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if git_commit is None:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = "nogit"
    return f"{dataset}-{backbone}-{timestamp}-{git_commit or 'nogit'}"


def build_museg_run_config(
    *,
    config: Any,
    args: Any,
    protocol_id: str,
    schedule_version: str,
    phase: str,
    run_id: str,
    seed: int,
    git_commit: str,
    split_metadata: Mapping[str, Any],
    output_dir: str,
    resume_parent: str | None,
    resume_checkpoint_sha256: str | None,
    pretrained_sha256: str | None,
    environment: Mapping[str, Any],
    val_batch_size: int,
) -> dict[str, Any]:
    """Build the complete, versioned MUSeg local/SwanLab metadata contract."""
    iterations = int(config.niters_per_epoch)
    epochs = int(config.nepochs)
    warmup_epochs = int(getattr(config, "warm_up_epoch", 0))
    return {
        "schema_version": "museg-run-config-v1",
        "protocol": {"id": protocol_id, "schedule_version": schedule_version},
        "identity": {
            "phase": phase,
            "run_id": run_id,
            "seed": int(seed),
            "git_commit": git_commit,
            "dirty": False,
        },
        "data": {
            "dataset": config.dataset_name,
            "model": config.backbone,
            "backbone": config.backbone,
            "input_modalities": ["RGB", str(getattr(config, "x", "Depth"))],
            "depth_version": "single-channel" if bool(getattr(config, "x_is_single_channel", False)) else "configured",
            "splits": dict(split_metadata),
        },
        "schedule": {
            "epochs": epochs,
            "iterations_per_epoch": iterations,
            "total_iterations": epochs * iterations,
            "batch_size": int(config.batch_size),
            "val_batch_size": int(val_batch_size),
            "workers": int(config.num_workers),
            "amp": bool(args.amp),
            "validation_amp": bool(args.val_amp),
            "compile": bool(args.compile),
            "syncbn": bool(args.syncbn),
        },
        "optimization": {
            "optimizer": config.optimizer,
            "base_lr": float(config.lr),
            "poly_power": float(config.lr_power),
            "warmup_epochs": warmup_epochs,
            "warmup_iterations": warmup_epochs * iterations,
            "weight_decay": float(config.weight_decay),
        },
        "augmentation": {
            "train_scale": list(getattr(config, "train_scale_array", [])),
            "eval_scale": list(getattr(config, "eval_scale_array", [1.0])),
            "flip": bool(getattr(config, "eval_flip", False)),
            "sliding": bool(args.sliding),
            "mst": bool(args.mst),
        },
        "evaluation": {
            "start_epoch": int(config.eval_start_epoch),
            "interval": int(config.eval_interval),
            "save_interval": int(config.save_interval),
            "best_metric": "val_miou",
            "tie_break": "strict-greater-keeps-earliest",
        },
        "output": {
            "directory": str(output_dir),
            "checkpoint_schema": "dformer-training-checkpoint-v2",
        },
        "resume": {
            "parent_run_id": resume_parent,
            "checkpoint_sha256": resume_checkpoint_sha256,
        },
        "environment": dict(environment),
        "pretrained": {
            "path": str(getattr(config, "pretrained_model", "")),
            "sha256": pretrained_sha256,
        },
    }


def gpu_safety_violation(
    free_bytes: int,
    total_bytes: int,
    *,
    min_free_gib: float = 0.0,
    min_free_ratio: float = 0.0,
) -> str | None:
    """Return a failure reason when configured free-VRAM limits are violated."""
    free_gib = free_bytes / 1024**3
    free_ratio = free_bytes / total_bytes if total_bytes > 0 else 0.0
    if min_free_gib > 0 and free_gib < min_free_gib:
        return f"free VRAM {free_gib:.2f} GiB is below required {min_free_gib:.2f} GiB"
    if min_free_ratio > 0 and free_ratio < min_free_ratio:
        return f"free VRAM ratio {free_ratio:.3f} is below required {min_free_ratio:.3f}"
    return None


class ExperimentTracker:
    """Track one rank-zero experiment without making SwanLab mandatory."""

    VALID_MODES = ("disabled", "offline", "online")

    def __init__(self) -> None:
        self._swanlab = None
        self._started = False
        self._finished = False
        self.mode = "disabled"

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finish()

    def start(
        self,
        *,
        mode: str = "disabled",
        is_primary: bool = True,
        project: str = "DFormer-liu",
        workspace: str = "Newton_liub",
        name: str | None = None,
        log_dir: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"unsupported SwanLab mode: {mode}")
        if self._started:
            raise RuntimeError("experiment tracker has already been started")
        self._started = True
        self.mode = mode
        if not is_primary or mode == "disabled":
            return

        try:
            swanlab = importlib.import_module("swanlab")
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab tracking was requested but swanlab is not installed; "
                "install requirements-monitoring.txt"
            ) from exc

        init_kwargs: dict[str, Any] = {
            "project": project,
            "workspace": workspace,
            "mode": mode,
            "config": dict(config or {}),
            # Batch jobs must never offer SwanLab's interactive online-to-offline
            # fallback. Missing credentials therefore fail before training starts.
            "settings": swanlab.Settings(interactive=False),
        }
        if name:
            init_kwargs["experiment_name"] = name
        if log_dir:
            init_kwargs["logdir"] = log_dir

        # Deliberately allow all initialization failures to propagate. In online
        # mode, continuing without the requested remote record is unsafe.
        swanlab.init(**init_kwargs)
        self._swanlab = swanlab

    @property
    def enabled(self) -> bool:
        return self._swanlab is not None

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        if self._swanlab is None:
            return
        self._swanlab.log(dict(metrics), step=int(step))

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._swanlab is not None:
            self._swanlab.finish()
