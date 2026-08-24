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
