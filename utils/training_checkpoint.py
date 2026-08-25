"""Deterministic training schedules and versioned epoch-boundary checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = "dformer-training-checkpoint-v2"
BEST_TIE_BREAK_RULE = "strict-greater-keeps-earliest"
_ALLOWED_PHASES = frozenset({"development", "official", "qualification"})
_PROTOCOL_COMPATIBILITY_FIELDS = (
    "phase",
    "run_id",
    "git_commit",
    "seed",
    "model_name",
    "optimizer_name",
    "total_epochs",
    "iterations_per_epoch",
    "warmup_steps",
    "poly_power",
    "base_lr",
    "split_metadata",
    "config_sha256",
)
_REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "model",
        "optimizer",
        "amp_scaler",
        "completed_epoch",
        "next_epoch",
        "global_optimizer_step",
        "best_val_miou",
        "best_val_epoch",
        "best_tie_break_rule",
        "rng_state",
        "protocol",
    }
)
_EPOCH_CHECKPOINT_RE = re.compile(r"^epoch-(\d+)\.pth$")


class CheckpointError(RuntimeError):
    """Base class for checkpoint validation failures."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint cannot be decoded or is structurally invalid."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint does not match the frozen training protocol."""


@dataclass(frozen=True)
class TrainingSources:
    phase: str
    train_source: str
    val_source: str | None
    test_source: str | None


@dataclass(frozen=True)
class CheckpointProtocol:
    phase: str
    run_id: str
    git_commit: str
    seed: int
    model_name: str
    optimizer_name: str
    total_epochs: int
    iterations_per_epoch: int
    warmup_steps: int
    poly_power: float
    base_lr: float
    split_metadata: Mapping[str, Mapping[str, Any] | None]
    config_summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["split_metadata"] = _to_jsonable(data["split_metadata"])
        data["config_summary"] = _to_jsonable(data["config_summary"])
        data["config_sha256"] = stable_sha256(data["config_summary"])
        return data


@dataclass(frozen=True)
class ResumeState:
    next_epoch: int
    global_optimizer_step: int
    best_val_miou: float | None
    best_val_epoch: int | None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def get_git_commit(repo_root: str | os.PathLike[str]) -> str:
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot determine the full Git commit for checkpoint metadata") from exc
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise RuntimeError(f"invalid Git commit returned by git: {commit!r}")
    return commit.lower()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_source_samples(path: str | os.PathLike[str]) -> int:
    with open(path, encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def build_split_metadata(
    sources: TrainingSources,
    expected_sha256: Mapping[str, str] | None = None,
    expected_samples: Mapping[str, int] | None = None,
    *,
    read_test_source: bool = False,
) -> dict[str, dict[str, Any] | None]:
    expected_hashes = dict(expected_sha256 or {})
    expected_counts = dict(expected_samples or {})
    metadata: dict[str, dict[str, Any] | None] = {}
    for role, source in (
        ("train", sources.train_source),
        ("val", sources.val_source),
        ("test", sources.test_source),
    ):
        if source is None:
            metadata[role] = None
            continue
        resolved = str(Path(source).resolve())
        configured_sha256 = expected_hashes.get(role)
        if role == "test" and not read_test_source:
            if configured_sha256 is None or role not in expected_counts:
                raise ValueError(
                    "sealed test metadata requires configured SHA-256 and sample count; the training process will not read it"
                )
            metadata[role] = {
                "path": resolved,
                "samples": int(expected_counts[role]),
                "sha256": configured_sha256.lower(),
                "sealed_unread": True,
            }
            continue
        actual_sha256 = file_sha256(resolved)
        if configured_sha256 is not None and actual_sha256.lower() != configured_sha256.lower():
            raise CheckpointCompatibilityError(
                f"{role} split SHA-256 mismatch: expected {configured_sha256}, got {actual_sha256}"
            )
        actual_samples = count_source_samples(resolved)
        configured_samples = expected_counts.get(role)
        if configured_samples is not None and actual_samples != int(configured_samples):
            raise CheckpointCompatibilityError(
                f"{role} split sample-count mismatch: expected {configured_samples}, got {actual_samples}"
            )
        metadata[role] = {
            "path": resolved,
            "samples": actual_samples,
            "sha256": actual_sha256,
        }
    return metadata


def resolve_training_sources(config: Any) -> TrainingSources:
    phase = str(getattr(config, "experiment_phase", "qualification"))
    if phase not in _ALLOWED_PHASES:
        raise ValueError(f"experiment_phase must be one of {sorted(_ALLOWED_PHASES)}, got {phase!r}")

    train_source = getattr(config, "train_source", None)
    if not train_source:
        raise ValueError("train_source is required")
    explicit_val = hasattr(config, "val_source")
    val_source = getattr(config, "val_source", None) if explicit_val else getattr(config, "eval_source", None)
    test_source = getattr(config, "test_source", None)

    if phase in {"development", "qualification"} and not val_source:
        raise ValueError(f"{phase} phase requires an explicit val_source")
    if phase == "official":
        val_source = None
    if test_source and Path(test_source).resolve() == Path(train_source).resolve():
        raise ValueError("training source must not be the sealed test source")
    if test_source and val_source and Path(test_source).resolve() == Path(val_source).resolve():
        raise ValueError("validation source must not be the sealed test source")

    return TrainingSources(
        phase=phase,
        train_source=str(train_source),
        val_source=str(val_source) if val_source else None,
        test_source=str(test_source) if test_source else None,
    )


def phase_uses_validation(phase: str) -> bool:
    """Return whether a training phase owns a validation split and best checkpoint."""
    if phase not in _ALLOWED_PHASES:
        raise ValueError(f"unsupported training phase: {phase!r}")
    return phase in {"qualification", "development"}


def should_evaluate(
    epoch: int,
    total_epochs: int,
    start_epoch: int,
    interval: int,
) -> bool:
    _validate_schedule_inputs(epoch, total_epochs, start_epoch, interval)
    if epoch < start_epoch:
        return False
    return epoch == total_epochs or (epoch - start_epoch) % interval == 0


def should_save_epoch(epoch: int, total_epochs: int, interval: int) -> bool:
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be within [1, total_epochs]")
    if interval <= 0:
        raise ValueError("save interval must be positive")
    return epoch == total_epochs or epoch % interval == 0


def optimizer_step_was_applied(scale_before_step: float, scale_after_update: float) -> bool:
    """Return whether GradScaler executed the optimizer step instead of skipping it."""
    return float(scale_after_update) >= float(scale_before_step)


def _validate_schedule_inputs(epoch: int, total_epochs: int, start_epoch: int, interval: int) -> None:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be within [1, total_epochs]")
    if not 1 <= start_epoch <= total_epochs:
        raise ValueError("start_epoch must be within [1, total_epochs]")
    if interval <= 0:
        raise ValueError("evaluation interval must be positive")


def select_best_metric(
    best_value: float | None,
    best_epoch: int | None,
    candidate: float,
    candidate_epoch: int,
) -> tuple[bool, float, int]:
    if not math.isfinite(candidate):
        raise FloatingPointError(f"non-finite validation mIoU at epoch {candidate_epoch}: {candidate}")
    if best_value is None or candidate > best_value:
        return True, float(candidate), candidate_epoch
    if best_epoch is None:
        raise ValueError("best_epoch is required when best_value is set")
    return False, float(best_value), best_epoch


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise CheckpointCorruptionError(f"RNG state missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise CheckpointCompatibilityError(
                "checkpoint CUDA RNG device count does not match the current process"
            )
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _model_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        normalized = key[7:] if key.startswith("module.") else key
        state[normalized] = value
    return state


def create_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    completed_epoch: int,
    global_optimizer_step: int,
    best_val_miou: float | None,
    best_val_epoch: int | None,
    protocol: CheckpointProtocol,
) -> dict[str, Any]:
    if completed_epoch < 1:
        raise ValueError("completed_epoch must be positive; only epoch-boundary checkpoints are supported")
    if global_optimizer_step < 0:
        raise ValueError("global_optimizer_step cannot be negative")
    if (best_val_miou is None) != (best_val_epoch is None):
        raise ValueError("best_val_miou and best_val_epoch must either both be set or both be None")
    if best_val_miou is not None and not math.isfinite(best_val_miou):
        raise FloatingPointError("best validation mIoU must be finite")

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": _model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "amp_scaler": scaler.state_dict() if scaler is not None else None,
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "global_optimizer_step": global_optimizer_step,
        "best_val_miou": best_val_miou,
        "best_val_epoch": best_val_epoch,
        "best_tie_break_rule": BEST_TIE_BREAK_RULE,
        "rng_state": capture_rng_state(),
        "protocol": protocol.to_dict(),
    }


def atomic_save_checkpoint(checkpoint: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(checkpoint), temporary_path)
        with open(temporary_path, "r+b") as saved:
            saved.flush()
            os.fsync(saved.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def canonical_state_sha256(value: Any) -> str:
    """Hash nested checkpoint state without relying on torch.save byte layout."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(repr(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                update(child)
        elif isinstance(item, bytes):
            digest.update(b"bytes\0")
            digest.update(item)
        else:
            digest.update(b"scalar\0")
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def inspect_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_protocol: CheckpointProtocol | None = None,
    expected_checkpoint_run_id: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint on CPU and return a compact, auditable state summary."""
    checkpoint_path = Path(path)
    try:
        raw = _torch_load_cpu(checkpoint_path)
    except Exception as exc:
        raise CheckpointCorruptionError(f"cannot load checkpoint {checkpoint_path}: {exc}") from exc
    checkpoint = _validate_checkpoint_structure(raw, checkpoint_path)
    if expected_protocol is not None:
        _validate_protocol(
            checkpoint["protocol"],
            expected_protocol,
            expected_checkpoint_run_id=expected_checkpoint_run_id,
        )
    optimizer_lrs = [float(group["lr"]) for group in checkpoint["optimizer"].get("param_groups", [])]
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": file_sha256(checkpoint_path),
        "schema_version": checkpoint["schema_version"],
        "completed_epoch": checkpoint["completed_epoch"],
        "next_epoch": checkpoint["next_epoch"],
        "global_optimizer_step": checkpoint["global_optimizer_step"],
        "best_val_miou": checkpoint["best_val_miou"],
        "best_val_epoch": checkpoint["best_val_epoch"],
        "optimizer_lrs": optimizer_lrs,
        "protocol": checkpoint["protocol"],
        "component_sha256": {
            "model": canonical_state_sha256(checkpoint["model"]),
            "optimizer": canonical_state_sha256(checkpoint["optimizer"]),
            "amp_scaler": canonical_state_sha256(checkpoint["amp_scaler"]),
            "rng_state": canonical_state_sha256(checkpoint["rng_state"]),
        },
    }


def compare_checkpoint_inspections(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> list[str]:
    """Return deterministic field paths that differ between two audit summaries."""
    fields = (
        "schema_version", "completed_epoch", "next_epoch", "global_optimizer_step",
        "best_val_miou", "best_val_epoch", "optimizer_lrs", "protocol", "component_sha256",
    )
    return [field for field in fields if _to_jsonable(expected.get(field)) != _to_jsonable(actual.get(field))]


def _validate_checkpoint_structure(checkpoint: Any, path: Path) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise CheckpointCorruptionError(f"checkpoint {path} is not a mapping")
    missing = _REQUIRED_CHECKPOINT_KEYS.difference(checkpoint)
    if missing:
        raise CheckpointCorruptionError(f"checkpoint {path} missing required keys: {sorted(missing)}")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint schema {checkpoint['schema_version']!r}; expected {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    completed_epoch = checkpoint["completed_epoch"]
    if not isinstance(completed_epoch, int) or completed_epoch < 1:
        raise CheckpointCorruptionError("completed_epoch must be a positive integer")
    if checkpoint["next_epoch"] != completed_epoch + 1:
        raise CheckpointCorruptionError("next_epoch is inconsistent with completed_epoch")
    if checkpoint["best_tie_break_rule"] != BEST_TIE_BREAK_RULE:
        raise CheckpointCompatibilityError("checkpoint best-metric tie-break rule is incompatible")
    best_value = checkpoint["best_val_miou"]
    best_epoch = checkpoint["best_val_epoch"]
    if (best_value is None) != (best_epoch is None):
        raise CheckpointCorruptionError(
            "best validation mIoU and epoch must either both be set or both be absent"
        )
    if best_value is not None:
        if not isinstance(best_value, (int, float)) or not math.isfinite(best_value):
            raise CheckpointCorruptionError("best validation mIoU must be finite")
        if not isinstance(best_epoch, int) or not 1 <= best_epoch <= completed_epoch:
            raise CheckpointCorruptionError("best validation epoch is outside completed epochs")
    if not isinstance(checkpoint["protocol"], dict):
        raise CheckpointCorruptionError("checkpoint protocol must be a mapping")
    protocol = checkpoint["protocol"]
    if not isinstance(protocol.get("config_summary"), Mapping):
        raise CheckpointCorruptionError("checkpoint protocol config_summary must be a mapping")
    if protocol.get("config_sha256") != stable_sha256(protocol["config_summary"]):
        raise CheckpointCorruptionError("checkpoint protocol config SHA-256 is invalid")
    return checkpoint


def _validate_protocol(
    actual: Mapping[str, Any],
    expected_protocol: CheckpointProtocol,
    *,
    expected_checkpoint_run_id: str | None = None,
) -> None:
    expected = expected_protocol.to_dict()
    for field in _PROTOCOL_COMPATIBILITY_FIELDS:
        if field not in actual:
            raise CheckpointCorruptionError(f"checkpoint protocol missing required field {field!r}")
        expected_value = expected_checkpoint_run_id if field == "run_id" and expected_checkpoint_run_id else expected[field]
        if _to_jsonable(actual[field]) != _to_jsonable(expected_value):
            raise CheckpointCompatibilityError(
                f"checkpoint protocol mismatch for {field}: expected {expected_value!r}, got {actual[field]!r}"
            )


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_protocol: CheckpointProtocol,
    expected_checkpoint_run_id: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        raw = _torch_load_cpu(checkpoint_path)
    except Exception as exc:
        raise CheckpointCorruptionError(f"cannot load checkpoint {checkpoint_path}: {exc}") from exc
    checkpoint = _validate_checkpoint_structure(raw, checkpoint_path)
    _validate_protocol(
        checkpoint["protocol"],
        expected_protocol,
        expected_checkpoint_run_id=expected_checkpoint_run_id,
    )
    completed_epoch = checkpoint["completed_epoch"]
    if completed_epoch > expected_protocol.total_epochs:
        raise CheckpointCompatibilityError("checkpoint completed epoch exceeds target total epochs")
    global_optimizer_step = checkpoint["global_optimizer_step"]
    max_possible_steps = completed_epoch * expected_protocol.iterations_per_epoch
    if not isinstance(global_optimizer_step, int) or not 0 <= global_optimizer_step <= max_possible_steps:
        raise CheckpointCorruptionError("global optimizer step is outside the completed-epoch boundary")
    expected_amp = bool(expected_protocol.config_summary.get("amp", False))
    if expected_amp != (checkpoint["amp_scaler"] is not None):
        raise CheckpointCompatibilityError("checkpoint AMP scaler presence disagrees with the training protocol")
    return checkpoint


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    restore_rng: bool = True,
) -> ResumeState:
    model_target = model.module if hasattr(model, "module") else model
    model_target.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler_state = checkpoint["amp_scaler"]
    if scaler is None and scaler_state is not None:
        raise CheckpointCompatibilityError("checkpoint contains AMP scaler state but AMP is disabled")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return ResumeState(
        next_epoch=int(checkpoint["next_epoch"]),
        global_optimizer_step=int(checkpoint["global_optimizer_step"]),
        best_val_miou=checkpoint["best_val_miou"],
        best_val_epoch=checkpoint["best_val_epoch"],
    )


def inspect_checkpoint_directory(
    checkpoint_dir: str | os.PathLike[str],
    *,
    expected_protocol: CheckpointProtocol,
    expected_checkpoint_run_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    directory = Path(checkpoint_dir)
    latest_path = directory / "latest.pth"
    latest = load_training_checkpoint(
        latest_path,
        expected_protocol=expected_protocol,
        expected_checkpoint_run_id=expected_checkpoint_run_id,
    )
    periodic: list[tuple[int, Path]] = []
    for candidate in directory.glob("epoch-*.pth"):
        match = _EPOCH_CHECKPOINT_RE.match(candidate.name)
        if match:
            periodic.append((int(match.group(1)), candidate))
    if not periodic:
        return latest, None
    filename_epoch, highest_path = max(periodic, key=lambda item: item[0])
    highest = load_training_checkpoint(
        highest_path,
        expected_protocol=expected_protocol,
        expected_checkpoint_run_id=expected_checkpoint_run_id,
    )
    if highest["completed_epoch"] != filename_epoch:
        raise CheckpointCorruptionError(
            f"periodic checkpoint filename {highest_path.name} disagrees with completed_epoch"
        )
    if latest["completed_epoch"] < highest["completed_epoch"]:
        raise CheckpointCompatibilityError(
            "latest checkpoint is older than the highest periodic checkpoint; refusing to guess resume state"
        )
    if latest["completed_epoch"] == highest["completed_epoch"]:
        consistency_fields = (
            "next_epoch",
            "global_optimizer_step",
            "best_val_miou",
            "best_val_epoch",
        )
        if any(latest[field] != highest[field] for field in consistency_fields):
            raise CheckpointCompatibilityError(
                "latest and same-epoch periodic checkpoints disagree; refusing to guess resume state"
            )
    return latest, highest


def prepare_output_directory(
    output_dir: str | os.PathLike[str],
    *,
    resume_path: str | os.PathLike[str] | None,
) -> Path:
    path = Path(output_dir).resolve()
    if resume_path is not None and not Path(resume_path).is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


IMMUTABLE_PROTOCOL_FIELDS = MappingProxyType(
    {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "best_tie_break_rule": BEST_TIE_BREAK_RULE,
        "resume_boundary": "completed-epoch-only",
    }
)