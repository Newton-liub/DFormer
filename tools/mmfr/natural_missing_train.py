"""Safety-gated Natural/Grid/Replay continuation runner.

Default execution is CPU-only readiness reporting. Preflight is limited to at most
three successful updates; any non-finite or AMP-skipped update aborts immediately.
The 2,560-update run requires explicit authorization and is never started implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from local_configs.MUSeg.DFormerv2_S_NaturalMissing import make_config
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import get_train_loader
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.natural_missing_runtime import (
    _load_segmentation_state,
    build_model,
    load_segmentation_checkpoint,
)
from utils.training_checkpoint import (
    BEST_TIE_BREAK_RULE,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointProtocol,
    atomic_save_checkpoint,
    build_split_metadata,
    capture_rng_state,
    get_git_commit,
    optimizer_step_was_applied,
    resolve_training_sources,
    restore_rng_state,
    restore_training_state,
)


STRATEGIES = ("Natural", "Grid", "Replay")
C0_SHA256 = "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a"
FORMAL_SUCCESSFUL_UPDATES = 2560
PREFLIGHT_MAX_SUCCESSFUL_UPDATES = 3
INPUT_BUILDER_ID = "utils.dataloader.natural_missing.build_natural_missing_batch-v1"


class EpochDeterministicDataset(torch.utils.data.Dataset):
    """Give each sample/epoch stable Python, NumPy and CPU-Torch randomness."""

    def __init__(self, dataset, *, base_seed: int, epoch: int = 1):
        self.dataset = dataset
        self.base_seed = int(base_seed)
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        digest = hashlib.sha256(f"{self.base_seed}:{self.epoch}:{int(index)}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        try:
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("readiness", "preflight", "formal"),
        default="readiness",
        help="readiness (default), bounded GPU preflight, or explicitly authorized formal run",
    )
    parser.add_argument("--strategy", choices=STRATEGIES, default="Natural")
    parser.add_argument("--c0-checkpoint", default=None)
    parser.add_argument(
        "--verified-c0-sha256",
        default=None,
        help="SHA-256 identity already verified by the checkpoint recovery owner; this runner does not re-hash C0",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--successful-updates", type=int, default=None)
    parser.add_argument(
        "--authorize-formal-training",
        action="store_true",
        help="required in addition to --mode formal; does not bypass C0 identity checks",
    )
    parser.add_argument(
        "--preflight-workers",
        type=int,
        default=None,
        help="preflight-only DataLoader worker override; defaults to 0 on this Windows workflow",
    )
    parser.add_argument(
        "--cpu-model-check",
        action="store_true",
        help="readiness-only model construction and strict in-memory key-path check",
    )
    return parser


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "readiness":
        if args.authorize_formal_training or args.resume or args.successful_updates is not None:
            parser.error("readiness mode does not train, resume, or accept an update budget")
        if args.preflight_workers is not None:
            parser.error("--preflight-workers is valid only in preflight mode")
        if args.verified_c0_sha256 is not None and args.c0_checkpoint is None:
            parser.error("--verified-c0-sha256 requires --c0-checkpoint")
    elif args.mode == "preflight":
        requested = 3 if args.successful_updates is None else args.successful_updates
        if requested < 1 or requested > PREFLIGHT_MAX_SUCCESSFUL_UPDATES:
            parser.error("preflight is strictly limited to 1-3 successful updates per group")
        args.successful_updates = requested
        if args.authorize_formal_training:
            parser.error("preflight must not be marked as formal training authorization")
        if args.preflight_workers is not None and args.preflight_workers < 0:
            parser.error("--preflight-workers cannot be negative")
    else:
        if not args.authorize_formal_training:
            parser.error("formal mode requires the explicit --authorize-formal-training flag")
        if args.successful_updates != FORMAL_SUCCESSFUL_UPDATES:
            parser.error("formal mode requires exactly --successful-updates 2560")
        if args.preflight_workers is not None:
            parser.error("preflight-only worker overrides cannot be used for formal training")

    if args.mode in {"preflight", "formal"}:
        if not args.verified_c0_sha256:
            parser.error("preflight/formal mode requires the recovery owner's verified C0 SHA-256")
        if args.verified_c0_sha256.lower() != C0_SHA256:
            parser.error("verified C0 identity does not match the frozen NaturalMissing C0 identity")
        if args.resume is None and args.c0_checkpoint is None:
            parser.error("a C0 checkpoint path is required unless resuming a NaturalMissing checkpoint")
        if args.resume and not Path(args.resume).is_file():
            parser.error(f"resume checkpoint does not exist: {args.resume}")
        if args.c0_checkpoint and not Path(args.c0_checkpoint).is_file():
            parser.error(f"C0 checkpoint does not exist: {args.c0_checkpoint}")
    elif args.resume:
        parser.error("--resume is valid only for preflight/formal mode")

    if args.cpu_model_check and args.mode != "readiness":
        parser.error("--cpu-model-check is readiness-only")


def _validate_config(config) -> None:
    if config.natural_missing["strategy"] not in STRATEGIES:
        raise ValueError("invalid NaturalMissing strategy")
    if not config.natural_missing.get("enabled") or not config.natural_missing.get("metadata_enabled"):
        raise ValueError("NaturalMissing loader metadata must be enabled")
    if config.train_source.endswith("official-test.txt") or config.test_source is not None:
        raise ValueError("NaturalMissing training config must not resolve the official-test source")
    if not config.train_source.endswith("train-dev.txt"):
        raise ValueError("NaturalMissing training source must be train-dev.txt")
    if float(config.lr) != 1e-6:
        raise ValueError("NaturalMissing continuation LR must be 1e-6")
    if int(config.successful_update_budget) != FORMAL_SUCCESSFUL_UPDATES:
        raise ValueError("NaturalMissing formal budget must be 2560 successful updates")
    if int(config.max_attempts) != int(config.successful_update_budget):
        raise ValueError("NaturalMissing attempt cap must equal its successful-update budget")
    if int(config.recovery_interval_successful_updates) != 640:
        raise ValueError("NaturalMissing recovery interval must be 640 successful updates")
    if float(config.aux_rate) != 0.0:
        raise ValueError("segmentation auxiliary CE head must be disabled")
    for name in (
        "mmfr_a2",
        "mmfr_a2_execution_profile",
        "e1_batch1",
        "e1_batch1b",
        "e1_roe",
        "mmfr_av1",
        "mmfr_f_lite",
        "mmfr_roe",
    ):
        if getattr(config, name, None):
            raise ValueError(f"legacy branch unexpectedly enabled: {name}")


def _config_comparison(configs: list[Any]) -> bool:
    """Confirm factories produce separate objects and differ only by strategy identity."""
    if len({id(config) for config in configs}) != len(configs):
        return False
    for left_idx, left in enumerate(configs):
        for right in configs[left_idx + 1 :]:
            if left is right or left.class_names is right.class_names:
                return False
            left_fields = {
                key: value
                for key, value in left.items()
                if key not in {"run_id", "log_dir", "tb_dir", "log_dir_link", "checkpoint_dir", "log_file", "link_log_file", "val_log_file", "link_val_log_file", "natural_missing"}
            }
            right_fields = {
                key: value
                for key, value in right.items()
                if key not in {"run_id", "log_dir", "tb_dir", "log_dir_link", "checkpoint_dir", "log_file", "link_log_file", "val_log_file", "link_val_log_file", "natural_missing"}
            }
            if _jsonable(left_fields) != _jsonable(right_fields):
                return False
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, default=str))


def _run_readiness(args: argparse.Namespace) -> dict[str, Any]:
    config = make_config(args.strategy)
    _validate_config(config)
    all_configs = [make_config(strategy) for strategy in STRATEGIES]
    isolation_ok = _config_comparison(all_configs)
    depth16_mask_root = Path(config.depth16_root)
    if depth16_mask_root.name.lower() != "depth16":
        depth16_mask_root = depth16_mask_root / "Depth16"
    report: dict[str, Any] = {
        "mode": "readiness",
        "strategy": args.strategy,
        "config_isolation": "PASS" if isolation_ok else "FAIL",
        "legacy_training_branches": "OFF",
        "train_source": config.train_source,
        "train_role": "train-dev",
        "official_test_source": None,
        "depth16_root": config.depth16_root,
        "depth16_root_exists": Path(config.depth16_root).is_dir(),
        "depth16_mask_root": str(depth16_mask_root),
        "depth16_mask_root_exists": depth16_mask_root.is_dir(),
        "grid_replay_source_pool_status": (
            "DEPTH16_DIRECTORY_PRESENT_FILE_CHECK_PENDING"
            if depth16_mask_root.is_dir()
            else "BLOCKED_DEPTH16_DIRECTORY_MISSING"
        ),
        "dataset_root_exists": Path(config.dataset_path).is_dir(),
        "formal_successful_updates_per_group": int(config.successful_update_budget),
        "preflight_successful_update_limit_per_group": PREFLIGHT_MAX_SUCCESSFUL_UPDATES,
        "c0_identity_required": C0_SHA256,
        "c0_status": "NOT_SUPPLIED",
        "gpu_training_started": False,
    }

    if args.cpu_model_check or args.c0_checkpoint:
        model = build_model(config, torch.device("cpu"))
        if args.c0_checkpoint:
            if args.verified_c0_sha256 is None or args.verified_c0_sha256.lower() != C0_SHA256:
                raise ValueError("a supplied C0 must carry the already verified canonical SHA-256 identity")
            load_report = load_segmentation_checkpoint(model, args.c0_checkpoint)
            report["c0_status"] = "LOADED_STRICTLY"
            report["c0_loaded_key_count"] = len(load_report["loaded"])
            report["c0_intentionally_dropped_key_count"] = len(load_report["intentionally_dropped"])
            if report["c0_intentionally_dropped_key_count"] != 10:
                raise ValueError("verified C0 must contain the exact 10 confirmed auxiliary tensors")
        else:
            load_report = _load_segmentation_state(model, model.state_dict())
            report["cpu_model_state_roundtrip"] = "PASS" if not load_report["missing"] and not load_report["unexpected"] else "FAIL"
        report["model_branches"] = {
            "reliability_estimator": getattr(model, "reliability_estimator", None) is None,
            "roe_substitute": getattr(model, "roe_substitute", None) is None,
            "feature_adapter": getattr(model, "feature_adapter", None) is None,
            "av1": getattr(model, "av1", None) is None,
            "aux_head": getattr(model, "aux_head", None) is None,
            "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        }
        if not all(report["model_branches"].values()):
            raise ValueError(f"NaturalMissing model branch check failed: {report['model_branches']}")

    if not isolation_ok:
        raise ValueError("Natural/Grid/Replay configuration isolation check failed")
    return report


def _seed_everything(seed: int, *, cuda: bool, config) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = bool(config.tf32_cudnn)
    torch.backends.cuda.matmul.allow_tf32 = bool(config.tf32_matmul)
    torch.set_float32_matmul_precision(str(config.tf32_matmul_precision))
    torch.use_deterministic_algorithms(True, warn_only=True)



def _build_optimizer(model: nn.Module, config):
    norm_layer = nn.SyncBatchNorm if bool(config.syncbn) else nn.BatchNorm2d
    param_groups = group_weight([], model, norm_layer, float(config.lr))
    if config.optimizer != "AdamW":
        raise ValueError(f"unsupported NaturalMissing optimizer {config.optimizer!r}")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(config.lr),
        betas=(0.9, 0.999),
        weight_decay=float(config.weight_decay),
    )
    return optimizer


def _build_scaler(config):
    if not bool(config.amp):
        return None
    values = dict(config.grad_scaler)
    return torch.cuda.amp.GradScaler(
        init_scale=float(values["initial_scale"]),
        growth_factor=float(values["growth_factor"]),
        backoff_factor=float(values["backoff_factor"]),
        growth_interval=int(values["growth_interval"]),
    )


def _split_identity(config) -> dict[str, Any]:
    sources = resolve_training_sources(config)
    expected_hashes = dict(config.expected_split_sha256)
    expected_counts = dict(config.expected_split_samples)
    return build_split_metadata(
        sources,
        expected_sha256=expected_hashes,
        expected_samples=expected_counts,
        read_test_source=False,
    )


def _resume_identity(config, split_metadata: Mapping[str, Any], mode: str, workers: int, c0_sha256: str) -> dict[str, Any]:
    train_metadata = split_metadata.get("train")
    if not train_metadata:
        raise ValueError("train-dev split identity is required")
    return {
        "schema": "natural-missing-resume-v2",
        "mode": mode,
        "strategy": config.natural_missing["strategy"],
        "c0_sha256": c0_sha256.lower(),
        "train_source": str(Path(config.train_source).resolve()),
        "train_split_sha256": str(train_metadata["sha256"]).lower(),
        "train_sample_count": int(train_metadata["samples"]),
        "seed": int(config.seed),
        "batch_size": int(config.batch_size),
        "num_workers": int(workers),
        "iterations_per_epoch": int(config.niters_per_epoch),
        "image_size": [int(config.image_height), int(config.image_width)],
        "train_scale_array": list(config.train_scale_array),
        "mirror": True,
        "sampler": "torch.RandomSampler-epoch-seeded-independent-generator-v1",
        "loader_generator": "epoch-seeded-independent-worker-generator-v1",
        "augmentation_rng": "sha256-base-seed-epoch-index-python-numpy-cpu-torch-v1",
        "resume_replay": "rebuild-epoch-sampler-and-replay-cursor-batches-v2",
        "input_builder": INPUT_BUILDER_ID,
        "paired_input_position": "successful_update+epoch+sample_identity+batch_slot",
        "natural_missing_config": _jsonable(config.natural_missing),
    }


def _protocol(config, split_metadata, mode: str, run_id: str, total_epochs: int) -> CheckpointProtocol:
    summary = {
        "config_module": "local_configs.MUSeg.DFormerv2_S_NaturalMissing",
        "dataset": config.dataset_name,
        "seed": int(config.seed),
        "backbone": config.backbone,
        "strategy": config.natural_missing["strategy"],
        "execution_mode": mode,
        "c0_sha256": C0_SHA256,
        "optimizer": config.optimizer,
        "base_lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "amp": bool(config.amp),
        "amp_dtype": config.amp_dtype,
        "syncbn": bool(config.syncbn),
        "tf32_matmul_precision": config.tf32_matmul_precision,
        "tf32_matmul": bool(config.tf32_matmul),
        "tf32_cudnn": bool(config.tf32_cudnn),
        "nmf_training_precision": config.nmf_training_precision,
        "successful_update_budget": FORMAL_SUCCESSFUL_UPDATES,
        "recovery_interval_successful_updates": int(config.recovery_interval_successful_updates),
        "input_builder": INPUT_BUILDER_ID,
        "legacy_auxiliary_training": False,
    }
    project_root = Path(__file__).resolve().parents[2]
    return CheckpointProtocol(
        phase=str(config.experiment_phase),
        run_id=run_id,
        git_commit=get_git_commit(str(project_root)),
        seed=int(config.seed),
        model_name=str(config.backbone),
        optimizer_name=str(config.optimizer),
        total_epochs=int(total_epochs),
        iterations_per_epoch=int(config.niters_per_epoch),
        warmup_steps=int(config.warmup_successful_updates),
        poly_power=float(config.lr_power),
        base_lr=float(config.lr),
        split_metadata=dict(split_metadata),
        config_summary=summary,
    )


def _load_resume(path: str, *, protocol: CheckpointProtocol, model, optimizer, scaler, expected_identity):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"cannot load NaturalMissing resume checkpoint: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError("NaturalMissing resume checkpoint must be a mapping")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("NaturalMissing resume checkpoint schema mismatch")
    if checkpoint.get("best_tie_break_rule") != BEST_TIE_BREAK_RULE:
        raise ValueError("NaturalMissing resume checkpoint tie-break identity mismatch")
    metadata = checkpoint.get("natural_missing")
    if not isinstance(metadata, Mapping):
        raise ValueError("resume checkpoint lacks NaturalMissing cursor metadata")
    if metadata.get("execution_mode") != expected_identity["mode"]:
        raise ValueError("preflight checkpoints cannot be resumed for formal training")
    if bool(metadata.get("preflight_only")) != (expected_identity["mode"] == "preflight"):
        raise ValueError("resume checkpoint preflight/formal marker is inconsistent")
    if metadata.get("resume_identity") != expected_identity:
        raise ValueError("NaturalMissing resume data/config/augmentation identity mismatch")
    if checkpoint.get("protocol") != protocol.to_dict():
        raise ValueError("NaturalMissing resume checkpoint protocol identity mismatch")
    required_checkpoint = {
        "model", "optimizer", "amp_scaler", "completed_epoch", "next_epoch",
        "global_optimizer_step", "best_val_miou", "best_val_epoch", "rng_state",
    }
    if not required_checkpoint.issubset(checkpoint):
        raise ValueError("NaturalMissing resume checkpoint is missing training-state fields")
    if (checkpoint["amp_scaler"] is not None) != (scaler is not None):
        raise ValueError("resume checkpoint AMP scaler presence disagrees with the current config")

    cursor = metadata.get("cursor")
    if not isinstance(cursor, Mapping):
        raise ValueError("resume checkpoint lacks a data cursor")
    required_cursor = {"epoch", "batch_position", "successful_updates", "attempted_steps", "skipped_steps"}
    if not required_cursor.issubset(cursor):
        raise ValueError("resume cursor is incomplete")
    cursor_values = {key: int(cursor[key]) for key in required_cursor}
    if cursor_values["skipped_steps"] != 0:
        raise ValueError("NaturalMissing resume checkpoints with skipped optimizer updates are ineligible")
    if cursor_values["epoch"] < 1 or not 0 <= cursor_values["batch_position"] < int(expected_identity["iterations_per_epoch"]):
        raise ValueError("resume data cursor is outside the frozen epoch geometry")
    if cursor_values["attempted_steps"] != cursor_values["successful_updates"] + cursor_values["skipped_steps"]:
        raise ValueError("resume optimizer counters violate attempted=successful+skipped")
    if int(checkpoint["global_optimizer_step"]) != cursor_values["successful_updates"]:
        raise ValueError("resume checkpoint update count disagrees with its data cursor")
    if not isinstance(metadata.get("epoch_rng_state"), Mapping):
        raise ValueError("resume checkpoint lacks the epoch-start RNG state")
    if not isinstance(checkpoint.get("rng_state"), Mapping):
        raise ValueError("resume checkpoint lacks the training RNG state")
    restore_training_state(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        restore_rng=False,
    )
    return checkpoint, metadata


def _save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer,
    scaler,
    protocol: CheckpointProtocol,
    cursor: Mapping[str, int],
    epoch_rng_state: Mapping[str, Any],
    current_rng_state: Mapping[str, Any],
    resume_identity: Mapping[str, Any],
    mode: str,
    telemetry: Mapping[str, Any] | None,
) -> None:
    # The shared high-level checkpoint builder only represents epoch boundaries.
    # NaturalMissing recovery points can be mid-epoch after skipped attempts, so the
    # epoch/batch cursor and epoch-start RNG are added to the shared schema explicitly.
    completed_epoch = max(0, int(cursor["epoch"]) - 1)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp_scaler": scaler.state_dict() if scaler is not None else None,
        "completed_epoch": completed_epoch,
        "next_epoch": int(cursor["epoch"]),
        "global_optimizer_step": int(cursor["successful_updates"]),
        "best_val_miou": None,
        "best_val_epoch": None,
        "best_tie_break_rule": BEST_TIE_BREAK_RULE,
        "rng_state": dict(current_rng_state),
        "protocol": protocol.to_dict(),
        "natural_missing": {
            "execution_mode": mode,
            "preflight_only": mode == "preflight",
            "resume_identity": _jsonable(resume_identity),
            "cursor": {key: int(value) for key, value in cursor.items()},
            "epoch_rng_state": dict(epoch_rng_state),
            "last_telemetry": _jsonable(telemetry or {}),
        },
    }
    atomic_save_checkpoint(checkpoint, path)


def _make_iterator(
    loader,
    *,
    base_seed: int,
    batch_position: int,
    checkpoint_rng_state=None,
    epoch: int,
):
    """Recreate one epoch's sample order without touching model-side RNG streams."""
    if hasattr(loader.dataset, "epoch"):
        loader.dataset.epoch = int(epoch)
    sampler_digest = hashlib.sha256(f"{int(base_seed)}:{int(epoch)}:sampler".encode("utf-8")).digest()
    loader_digest = hashlib.sha256(f"{int(base_seed)}:{int(epoch)}:loader".encode("utf-8")).digest()
    sampler_generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(sampler_digest[:8], "big"))
    loader_generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(loader_digest[:8], "big"))
    if not hasattr(loader.sampler, "generator"):
        raise TypeError("NaturalMissing training loader must use a generator-backed sampler")
    loader.sampler.generator = sampler_generator
    loader.generator = loader_generator
    iterator = iter(loader)
    for _ in range(batch_position):
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError("saved batch cursor exceeds the current DataLoader length") from exc
    if checkpoint_rng_state is not None:
        restore_rng_state(checkpoint_rng_state)
    return iterator


def _is_finite_gradients(model: nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )


def _run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = make_config(args.strategy)
    _validate_config(config)
    if args.strategy in {"Grid", "Replay"}:
        depth16_root = Path(config.natural_missing["depth16_root"])
        if depth16_root.name.lower() != "depth16":
            depth16_root = depth16_root / "Depth16"
        if not depth16_root.is_dir():
            raise RuntimeError(f"GRID_REPLAY_PREFLIGHT=BLOCKED: Depth16 mask directory not found: {depth16_root}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_PREFLIGHT=BLOCKED: no local CUDA device is available")
    device = torch.device("cuda")
    workers = config.num_workers
    if args.mode == "preflight":
        # Windows-safe worker setting applies only to this bounded preflight copy.
        workers = 0 if args.preflight_workers is None else int(args.preflight_workers)
        config.num_workers = workers
    elif workers != 8:
        raise ValueError("formal NaturalMissing worker contract must remain at the E1 C0 value of 8")

    config.run_id = f"NaturalMissing-{args.strategy}-{args.mode}"
    output_root = Path(config.log_dir).parents[2]
    config.log_dir = str(output_root / config.run_id / "development" / f"seed-{config.seed}")
    config.checkpoint_dir = str(Path(config.log_dir) / "checkpoint")
    if args.mode == "preflight":
        config.checkpoint_dir = str(Path(config.checkpoint_dir) / "preflight-only")

    if args.mode == "preflight":
        target_successful_updates = int(args.successful_updates)
        attempt_cap = target_successful_updates
    else:
        target_successful_updates = FORMAL_SUCCESSFUL_UPDATES
        attempt_cap = int(config.max_attempts)
        if attempt_cap != target_successful_updates:
            raise ValueError("formal NaturalMissing attempt cap must equal its successful-update budget")
    epoch_horizon = int(math.ceil(attempt_cap / float(config.niters_per_epoch)))

    _seed_everything(int(config.seed), cuda=True, config=config)

    split_metadata = _split_identity(config)
    if split_metadata.get("test") is not None:
        raise ValueError("training split resolution exposed an official-test source; refusing to continue")
    if not split_metadata.get("train"):
        raise ValueError("train-dev split metadata is missing")
    if int(split_metadata["train"]["samples"]) != 1277:
        raise ValueError("train-dev split sample count differs from the frozen 1,277 sample identity")
    config.num_train_imgs = int(split_metadata["train"]["samples"])
    config.niters_per_epoch = config.num_train_imgs // int(config.batch_size) + 1
    if config.niters_per_epoch != 128:
        raise ValueError("NaturalMissing requires the E1 C0 128-batch epoch geometry")

    c0_sha256 = args.verified_c0_sha256.lower()
    run_id = config.run_id
    protocol = _protocol(config, split_metadata, args.mode, run_id, epoch_horizon)
    resume_identity = _resume_identity(config, split_metadata, args.mode, workers, c0_sha256)

    model = build_model(config, device)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("NaturalMissing requires full-parameter continuation")

    optimizer = _build_optimizer(model, config)
    scaler = _build_scaler(config)
    successful_updates = 0
    attempted_steps = 0
    skipped_steps = 0
    epoch = 1
    batch_position = 0
    epoch_rng_state = capture_rng_state()
    checkpoint_rng_state = None
    resume_metadata = None

    if args.resume:
        checkpoint, resume_metadata = _load_resume(
            args.resume,
            protocol=protocol,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_identity=resume_identity,
        )
        cursor_data = resume_metadata["cursor"]
        epoch = int(cursor_data["epoch"])
        batch_position = int(cursor_data["batch_position"])
        successful_updates = int(cursor_data["successful_updates"])
        attempted_steps = int(cursor_data["attempted_steps"])
        skipped_steps = int(cursor_data["skipped_steps"])
        epoch_rng_state = resume_metadata["epoch_rng_state"]
        checkpoint_rng_state = checkpoint["rng_state"]
        if successful_updates > target_successful_updates:
            raise ValueError("resume checkpoint is already beyond the requested successful-update budget")
        if attempted_steps > attempt_cap:
            raise ValueError("resume checkpoint already exceeds the configured attempt cap")
    else:
        if not args.c0_checkpoint:
            raise ValueError("a C0 checkpoint is required for a fresh NaturalMissing run")
        load_report = load_segmentation_checkpoint(model, args.c0_checkpoint)
        if len(load_report["intentionally_dropped"]) != 10:
            raise ValueError("verified C0 checkpoint must contain the exact 10 confirmed auxiliary tensors")
        if load_report["missing"] or load_report["unexpected"]:
            raise ValueError("strict C0 segmentation load reported missing/unexpected keys")
        if not load_report["loaded"]:
            raise ValueError("strict C0 segmentation load matched no model keys")

    loader_engine = SimpleNamespace(distributed=False, world_size=1)
    base_loader, train_sampler = get_train_loader(loader_engine, RGBXDataset, config)
    if train_sampler is not None:
        raise ValueError("NaturalMissing single-device runner unexpectedly received a distributed sampler")
    deterministic_dataset = EpochDeterministicDataset(
        base_loader.dataset,
        base_seed=int(config.seed),
        epoch=1,
    )
    train_loader = torch.utils.data.DataLoader(
        deterministic_dataset,
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    if len(train_loader) != int(config.niters_per_epoch):
        raise ValueError(
            f"train loader length {len(train_loader)} disagrees with frozen {config.niters_per_epoch} iterations"
        )

    source_pool = None
    if args.strategy in {"Grid", "Replay"}:
        from utils.dataloader.natural_missing import build_source_pool

        source_pool = build_source_pool(config.train_source, config.depth16_root)
        config.natural_missing["source_pool"] = source_pool

    iterator = None
    if checkpoint_rng_state is not None:
        iterator = _make_iterator(
            train_loader,
            base_seed=int(config.seed),
            batch_position=batch_position,
            checkpoint_rng_state=checkpoint_rng_state,
            epoch=epoch,
        )
    else:
        iterator = _make_iterator(
            train_loader,
            base_seed=int(config.seed),
            batch_position=0,
            epoch=epoch,
        )

    scheduler = WarmUpPolyLR(
        float(config.lr),
        float(config.lr_power),
        int(config.successful_update_budget),
        int(config.warmup_successful_updates),
    )
    model.train()
    last_telemetry: Mapping[str, Any] = {}
    started = time.perf_counter()
    last_saved_update = -1

    while successful_updates < target_successful_updates and attempted_steps < attempt_cap:
        if iterator is None:
            epoch_rng_state = capture_rng_state()
            batch_position = 0
            iterator = _make_iterator(
                train_loader,
                base_seed=int(config.seed),
                batch_position=0,
                epoch=epoch,
            )

        try:
            original_batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("NaturalMissing DataLoader ended before its frozen epoch length") from exc

        from utils.dataloader.natural_missing import build_natural_missing_batch

        batch, last_telemetry = build_natural_missing_batch(
            original_batch,
            config,
            successful_update=successful_updates,
            epoch=epoch,
        )
        for required in ("data", "label", "modal_x", "fn"):
            if required not in batch:
                raise ValueError(f"NaturalMissing batch is missing required base key {required!r}")
        images = batch["data"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        depth = batch["modal_x"].to(device, non_blocking=True)

        lr = scheduler.get_lr(successful_updates)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        optimizer.zero_grad(set_to_none=True)
        attempted_steps += 1
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        scale_before = scaler.get_scale() if scaler is not None else 1.0
        _emit({
            "event": "attempt_started", "mode": args.mode, "strategy": args.strategy,
            "attempted_steps": attempted_steps, "successful_updates": successful_updates,
            "skipped_steps": skipped_steps, "amp_scale": float(scale_before),
            "batch_shape": list(images.shape), "telemetry": last_telemetry,
        })

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(config.amp),
        ):
            loss = model(images, depth, labels)
        loss_finite = bool(torch.isfinite(loss.detach()).all().item())
        if not loss_finite:
            raise FloatingPointError("NaturalMissing loss is non-finite; refusing to skip the optimizer update")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if not _is_finite_gradients(model):
            raise FloatingPointError("NaturalMissing gradients are non-finite; refusing to skip the optimizer update")
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            applied = optimizer_step_was_applied(scale_before, scaler.get_scale())
            if not applied:
                raise RuntimeError("AMP GradScaler skipped a NaturalMissing optimizer update")
        else:
            optimizer.step()
            applied = True

        optimizer.zero_grad(set_to_none=True)
        successful_updates += 1

        batch_position += 1
        if batch_position >= len(train_loader):
            epoch += 1
            batch_position = 0
            iterator = None
            epoch_rng_state = capture_rng_state()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        _emit(
            {
                "mode": args.mode,
                "strategy": args.strategy,
                "attempted_steps": attempted_steps,
                "successful_updates": successful_updates,
                "skipped_steps": skipped_steps,
                "optimizer_step_applied": applied,
                "loss_finite": loss_finite,
                "loss": float(loss.detach().float().cpu().item()),
                "lr": float(lr),
                "amp_scale": float(scaler.get_scale()) if scaler is not None else 1.0,
                "epoch": epoch - (1 if batch_position == 0 and iterator is None else 0),
                "last_batch_index": batch_position,
                "elapsed_seconds": elapsed,
                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "peak_reserved_memory_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
                "telemetry": last_telemetry,
            }
        )

        is_preflight_done = args.mode == "preflight" and successful_updates >= target_successful_updates
        is_recovery_point = (
            args.mode == "formal"
            and successful_updates > 0
            and successful_updates % int(config.recovery_interval_successful_updates) == 0
        )
        is_final = args.mode == "formal" and successful_updates == FORMAL_SUCCESSFUL_UPDATES
        if is_preflight_done or is_recovery_point or is_final:
            cursor = {
                "epoch": int(epoch),
                "batch_position": int(batch_position),
                "successful_updates": int(successful_updates),
                "attempted_steps": int(attempted_steps),
                "skipped_steps": int(skipped_steps),
            }
            current_rng_state = capture_rng_state()
            filename = (
                f"preflight-update-{successful_updates:06d}.pth"
                if args.mode == "preflight"
                else f"update-{successful_updates}.pth"
            )
            checkpoint_path = Path(config.checkpoint_dir) / filename
            _save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                protocol=protocol,
                cursor=cursor,
                epoch_rng_state=epoch_rng_state,
                current_rng_state=current_rng_state,
                resume_identity=resume_identity,
                mode=args.mode,
                telemetry=last_telemetry,
            )
            last_saved_update = successful_updates
            if args.mode == "formal":
                _save_checkpoint(
                    path=Path(config.checkpoint_dir) / "latest.pth",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    protocol=protocol,
                    cursor=cursor,
                    epoch_rng_state=epoch_rng_state,
                    current_rng_state=current_rng_state,
                    resume_identity=resume_identity,
                    mode=args.mode,
                    telemetry=last_telemetry,
                )

    elapsed_total = time.perf_counter() - started
    complete = successful_updates == target_successful_updates
    if args.mode == "preflight" and not complete and last_saved_update != successful_updates:
        cursor = {
            "epoch": int(epoch),
            "batch_position": int(batch_position),
            "successful_updates": int(successful_updates),
            "attempted_steps": int(attempted_steps),
            "skipped_steps": int(skipped_steps),
        }
        _save_checkpoint(
            path=Path(config.checkpoint_dir) / f"preflight-update-{successful_updates:06d}-incomplete.pth",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            protocol=protocol,
            cursor=cursor,
            epoch_rng_state=epoch_rng_state,
            current_rng_state=capture_rng_state(),
            resume_identity=resume_identity,
            mode=args.mode,
            telemetry=last_telemetry,
        )

    result = {
        "mode": args.mode,
        "strategy": args.strategy,
        "successful_updates": successful_updates,
        "attempted_steps": attempted_steps,
        "skipped_steps": skipped_steps,
        "attempt_cap": attempt_cap,
        "complete": complete,
        "formal_config_num_workers": 8,
        "preflight_worker_override": workers if args.mode == "preflight" else None,
        "preflight_checkpoint_formal_eligible": False if args.mode == "preflight" else None,
        "checkpoint_dir": config.checkpoint_dir,
        "elapsed_seconds": elapsed_total,
        "telemetry": last_telemetry,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    _emit(result)
    if attempted_steps != successful_updates + skipped_steps:
        raise RuntimeError("optimizer telemetry invariant failed")
    if not complete:
        raise RuntimeError(
            f"attempt cap reached before target: successful={successful_updates}, "
            f"attempted={attempted_steps}, skipped={skipped_steps}, cap={attempt_cap}"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    if args.mode == "readiness":
        _emit(_run_readiness(args))
        return 0
    _run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
