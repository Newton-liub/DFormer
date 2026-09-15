#!/usr/bin/env python3
"""MMFR-A2 v3 AMP update-path isolation gate.

This is a single-GPU, real-AMP qualification at the frozen batch size 10.  It
builds the v3 Depth-corruption model, reads real ``train-dev`` samples through
``TrainPre``, applies ``build_mmfr_training_batch_v3`` on the final CPU batch,
and runs two paired ``torch.cuda.amp.GradScaler`` trajectories:

* lambda_rel=0.0, with the reliability head frozen;
* lambda_rel=0.1, the frozen v3 training weight.

The gate requires identical batch identities, step/skip sequence, GradScaler
scales, shared parameter bytes and shared optimizer-state bytes after every
successful update.  The reliability head must remain unchanged on the zero
weight side and change on the 0.1 side.  A CUDA out-of-memory event is recorded
as a capacity failure at batch size 10; the tool never retries with a smaller
batch and never starts formal training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.builder import EncoderDecoder  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.dataloader.dataloader import TrainPre  # noqa: E402
from utils.dataloader.mmfr_training_v3 import build_mmfr_training_batch_v3  # noqa: E402
from utils.init_func import group_weight  # noqa: E402
from utils.training_checkpoint import optimizer_step_was_applied  # noqa: E402

SCHEMA_VERSION = "mmfr-a2-v3-amp-update-path-isolation-v1"
CHECK_ID = "MMFR-A2-v3-amp-update-path-isolation"
CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3"
EXPECTED_PROTOCOL = "MMFR-A2-train-integration-v3"
EXPECTED_CORRUPTION_BASIS = "MMFR-A1-corruption-basis-v3"
EXPECTED_PRETRAINED_SHA256 = "19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6"
EXPECTED_BATCH_SIZE = 10
EXPECTED_LAMBDA_OFF = 0.0
EXPECTED_LAMBDA_ON = 0.1
EXPECTED_SEED = 772961337
STEP_SEED_BASE = 20260915
PREPROCESS_SEED_BASE = 20260916
STEP_EPOCH = 1
DEFAULT_SAMPLE_OFFSET = 0
DEFAULT_MAX_ATTEMPTS = 50
DEFAULT_MIN_SUCCESSFUL_UPDATES = 10
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "mmfr-a2-v3-amp-update-path-isolation" / "a2-v3-amp-update-path-isolation.json"
RELIABILITY_PREFIX = "reliability_estimator."
ATOL = 1.0e-6
RTOL = 1.0e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    if value.dtype == torch.bool:
        value = value.to(torch.uint8)
    try:
        raw = value.numpy().tobytes()
    except (TypeError, RuntimeError):
        raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def combined_hash(mapping: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        digest.update(name.encode("utf-8"))
        digest.update(mapping[name].encode("ascii"))
    return digest.hexdigest()


def shared_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, _ in model.named_parameters() if not name.startswith(RELIABILITY_PREFIX)]


def reliability_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, _ in model.named_parameters() if name.startswith(RELIABILITY_PREFIX)]


def shared_parameter_hashes(model: nn.Module, names: Sequence[str]) -> Dict[str, str]:
    parameters = dict(model.named_parameters())
    return {name: tensor_sha256(parameters[name]) for name in names}


def gradient_hashes(model: nn.Module, names: Sequence[str]) -> Dict[str, Any]:
    parameters = dict(model.named_parameters())
    hashes: Dict[str, str] = {}
    missing: List[str] = []
    maximum = 0.0
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            missing.append(name)
            continue
        hashes[name] = tensor_sha256(gradient)
        if gradient.numel():
            maximum = max(maximum, float(gradient.detach().abs().max().item()))
    return {"hashes": hashes, "combined_hash": combined_hash(hashes), "missing": missing, "max_abs": maximum}


def managed_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def non_finite_gradient_names(model: nn.Module, names: Sequence[str], managed_ids: set[int]) -> List[str]:
    parameters = dict(model.named_parameters())
    result = []
    for name in names:
        parameter = parameters[name]
        if id(parameter) not in managed_ids or parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            result.append(name)
    return result


def optimizer_state_hashes(
    optimizer: torch.optim.Optimizer, parameter_names: Mapping[int, str], selected_names: Sequence[str]
) -> Dict[str, str]:
    selected = set(selected_names)
    result: Dict[str, str] = {}
    for parameter, state in optimizer.state.items():
        name = parameter_names.get(id(parameter))
        if name is None or name not in selected:
            continue
        digest = hashlib.sha256()
        for key in sorted(state):
            digest.update(str(key).encode("utf-8"))
            value = state[key]
            if torch.is_tensor(value):
                digest.update(tensor_sha256(value).encode("ascii"))
            else:
                digest.update(str(value).encode("utf-8"))
        result[name] = digest.hexdigest()
    return result


def build_batch_identity(batch: Mapping[str, Any]) -> Dict[str, str]:
    keys = (
        "rgb",
        "depth",
        "raw_rgb",
        "raw_depth",
        "reliability_target",
        "valid_mask",
        "depth_valid_pre",
        "depth_valid_post",
        "telemetry_masks",
    )
    return {key: tensor_sha256(batch[key]) for key in keys if key in batch}


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_MODULE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--min-successful-updates", type=int, default=DEFAULT_MIN_SUCCESSFUL_UPDATES)
    parser.add_argument("--min-corrupt-steps", type=int, default=1)
    parser.add_argument("--lambda-rel-off", type=float, default=EXPECTED_LAMBDA_OFF)
    parser.add_argument("--lambda-rel-on", type=float, default=EXPECTED_LAMBDA_ON)
    parser.add_argument("--sample-offset", type=int, default=DEFAULT_SAMPLE_OFFSET)
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MMFR-A2 v3 AMP isolation requires CUDA; CPU cannot exercise AMP+GradScaler")
    if device.index not in (None, 0):
        raise RuntimeError("this gate is frozen to the first CUDA device; use --device cuda:0")
    torch.cuda.set_device(device)
    return device


def configure_runtime(device: torch.device) -> Dict[str, Any]:
    seed_everything(EXPECTED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": "warn_only",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(device),
    }


def build_dataset_setting(config: Any) -> Dict[str, Any]:
    return {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "val_source": getattr(config, "val_source", getattr(config, "eval_source", None)),
        "test_source": getattr(config, "test_source", None),
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
        "channel_order": config.channel_order,
    }


def load_train_dataset(config: Any) -> RGBXDataset:
    return RGBXDataset(
        build_dataset_setting(config),
        "train",
        TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config),
    )


def load_train_item(dataset: RGBXDataset, config: Any, index: int, preprocess_seed: int) -> Dict[str, Any]:
    if not 0 <= int(index) < len(dataset):
        raise ValueError(f"train-dev sample index {index} outside dataset length {len(dataset)}")
    random.seed(int(preprocess_seed))
    np.random.seed(int(preprocess_seed))
    item = dataset[int(index)]
    rgb = item["data"]
    depth = item["modal_x"]
    label = item["label"]
    expected_shape = (int(config.image_height), int(config.image_width))
    if tuple(rgb.shape) != (3,) + expected_shape:
        raise RuntimeError(f"unexpected RGB shape {tuple(rgb.shape)}")
    if tuple(depth.shape) not in ((1,) + expected_shape, (3,) + expected_shape):
        raise RuntimeError(f"unexpected Depth shape {tuple(depth.shape)}")
    if tuple(label.shape) != expected_shape:
        raise RuntimeError(f"unexpected label shape {tuple(label.shape)}")
    return {
        "index": int(index),
        "entry": str(dataset._file_names[int(index)]),
        "sample_id": str(item["fn"]),
        "rgb": rgb,
        "depth": depth,
        "label": label,
    }


def plan_step(config: Any, step: int, batch_size: int, sample_offset: int) -> Dict[str, Any]:
    total = int(config.num_train_imgs)
    indices = [int((sample_offset + step * batch_size + slot) % total) for slot in range(batch_size)]
    preprocess_seeds = [int(PREPROCESS_SEED_BASE + step * batch_size + slot) for slot in range(batch_size)]
    return {
        "step": int(step),
        "sample_indices": indices,
        "preprocess_seeds": preprocess_seeds,
        "rng_seed": int(STEP_SEED_BASE + step),
        "epoch": int(STEP_EPOCH),
        "iteration": int(step),
        "batch_size": int(batch_size),
        "niters_per_epoch": int(config.niters_per_epoch),
        "nepochs": int(config.nepochs),
    }


def build_step_batch(
    dataset: RGBXDataset, config: Any, plan: Mapping[str, Any], corruption: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    samples = [
        load_train_item(dataset, config, index, preprocess_seed)
        for index, preprocess_seed in zip(plan["sample_indices"], plan["preprocess_seeds"])
    ]
    rgb = torch.stack([sample["rgb"] for sample in samples], dim=0)
    depth = torch.stack([sample["depth"] for sample in samples], dim=0)
    batch = build_mmfr_training_batch_v3(
        rgb,
        depth,
        [sample["sample_id"] for sample in samples],
        epoch=int(plan["epoch"]),
        iteration=int(plan["iteration"]),
        niters_per_epoch=int(plan["niters_per_epoch"]),
        nepochs=int(plan["nepochs"]),
        rgb_mean=config.norm_mean,
        rgb_std=config.norm_std,
        corruption_seed=int(corruption["seed"]),
        global_rank=0,
        p_clean=float(corruption["p_clean"]),
        max_specs=int(corruption["max_specs"]),
        sample_id_root=getattr(config, "dataset_path", None),
    )
    labels = torch.stack([sample["label"] for sample in samples], dim=0)
    batch["labels"] = labels
    if int(batch["rgb"].shape[0]) != int(plan["batch_size"]):
        raise RuntimeError(f"v3 helper returned batch size {batch['rgb'].shape[0]}, expected {plan['batch_size']}")
    records: List[Dict[str, Any]] = []
    for sample, metadata in zip(samples, batch["metadata"]):
        records.append(
            {
                "sample_index": int(sample["index"]),
                "sample_id": str(metadata["sample_id"]),
                "clean": bool(metadata["clean"]),
                "num_specs": int(metadata["num_specs"]),
                "spec_kinds": [str(spec["kind"]) for spec in metadata["specs"]],
                "valid_pixels": int(metadata["valid_pixels"]),
                "natural_invalid_pixels": int(metadata["natural_invalid_pixels"]),
                "synthetic_invalid_pixels": int(metadata["synthetic_invalid_pixels"]),
                "implicit_quality_pixels": int(metadata["implicit_quality_pixels"]),
                "valid_clean_pixels": int(metadata["valid_clean_pixels"]),
                "newly_valid_pixels": int(metadata["newly_valid_pixels"]),
            }
        )
    return batch, records, samples


def build_optimizer(model: nn.Module, config: Any) -> torch.optim.Optimizer:
    params_list = group_weight([], model, nn.SyncBatchNorm, float(config.lr))
    if str(config.optimizer) != "AdamW":
        raise RuntimeError(f"v3 AMP gate is frozen to AdamW, got {config.optimizer!r}")
    return torch.optim.AdamW(
        params_list,
        lr=float(config.lr),
        betas=(0.9, 0.999),
        weight_decay=float(config.weight_decay),
    )


def build_model(config: Any, device: torch.device) -> nn.Module:
    """Build the v3 model through the memory-safe single-GPU v2 qualification path."""
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    # The v2 gate's regular-BN construction is the proven single-GPU skeleton.  The
    # v3 config/helper/protocol remain unchanged; only this qualification-local norm
    # construction avoids SyncBatchNorm's excess activation footprint on the 8 GB card.
    model = EncoderDecoder(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
    model.to(device)
    model.train()
    return model


def snapshot_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().to("cpu").clone() for name, tensor in model.state_dict().items()}


def restore_model_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict(dict(state), strict=True)


def run_trajectory(
    *,
    label: str,
    model: nn.Module,
    config: Any,
    dataset: RGBXDataset,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    initial_head_hash: str,
    plans: Sequence[Mapping[str, Any]],
    batch_size: int,
    min_successful_updates: int,
    lambda_rel: float,
) -> Dict[str, Any]:
    """Run one paired v3 AMP trajectory without retaining GPU graphs or per-step maps."""
    restore_model_state(model, initial_state)
    model.train()
    model.reliability_weight = float(lambda_rel)
    head_names = reliability_parameter_names(model)
    shared_names = shared_parameter_names(model)
    parameters = dict(model.named_parameters())
    for name in head_names:
        parameters[name].requires_grad = bool(float(lambda_rel) != 0.0)

    optimizer = build_optimizer(model, config)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    managed_ids = managed_parameter_ids(optimizer)
    managed_shared_names = [name for name in shared_names if id(parameters[name]) in managed_ids]
    unmanaged_shared_names = [name for name in shared_names if name not in set(managed_shared_names)]
    scaler = torch.cuda.amp.GradScaler()
    records: List[Dict[str, Any]] = []
    successful_updates = 0
    corrupt_steps = 0
    capacity_failure = False
    exception_messages: List[str] = []

    for plan in plans:
        if successful_updates >= int(min_successful_updates):
            break
        step = int(plan["step"])
        batch = metadata = samples = tensors = labels = auxiliary = loss = None
        try:
            seed_everything(int(plan["rng_seed"]))
            batch, metadata, samples = build_step_batch(
                dataset,
                config,
                plan,
                dict(config.mmfr_a2["corruption"]),
            )
            identity = build_batch_identity(batch)
            identity_hash = combined_hash(identity)
            tensors = {
                key: batch[key].to(device, non_blocking=False)
                for key in (
                    "rgb",
                    "depth",
                    "raw_rgb",
                    "raw_depth",
                    "reliability_target",
                    "valid_mask",
                    "depth_valid_post",
                    "telemetry_masks",
                )
            }
            labels = batch["labels"].to(device, non_blocking=False)
            auxiliary = {
                "raw_rgb": tensors["raw_rgb"],
                "raw_depth": tensors["raw_depth"],
                "reliability_target": tensors["reliability_target"],
                "reliability_valid_mask": tensors["valid_mask"],
                "depth_valid": tensors["depth_valid_post"],
                # Keep the v3 telemetry contract in the model call; its detached
                # category telemetry is part of the qualification input identity.
                "reliability_telemetry_masks": tensors["telemetry_masks"],
            }
            seed_everything(int(plan["rng_seed"]))
            optimizer.zero_grad(set_to_none=True)
            scale_before = float(scaler.get_scale())
            # Offload autograd-saved activations to host memory.  This changes
            # storage placement only, not the forward/backward operators or the
            # AMP+GradScaler update semantics, and keeps the frozen batch size 10.
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(tensors["rgb"], tensors["depth"], labels, **auxiliary)
                loss_value = float(loss.detach().item())
                loss_finite = bool(math.isfinite(loss_value))
                scaler.scale(loss).backward()
            gradient = gradient_hashes(model, shared_names)
            non_finite = non_finite_gradient_names(model, shared_names + head_names, managed_ids)
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            applied = bool(optimizer_step_was_applied(scale_before, scale_after))
            optimizer.zero_grad(set_to_none=True)

            if not metadata:
                raise RuntimeError("v3 helper returned no metadata records")
            if not all(record["clean"] for record in metadata):
                corrupt_steps += 1
            record: Dict[str, Any] = {
                "step": step,
                "batch_size": int(batch["rgb"].shape[0]),
                "sample_indices": [int(sample["index"]) for sample in samples],
                "sample_ids": [str(record["sample_id"]) for record in metadata],
                # Keep only the compact v3 helper summary; full spec metadata is not
                # needed to decide paired identity or update-path equivalence.
                "metadata": [
                    {
                        "clean": bool(record["clean"]),
                        "num_specs": int(record["num_specs"]),
                        "spec_kinds": [str(kind) for kind in record["spec_kinds"]],
                    }
                    for record in metadata
                ],
                "batch_identity": identity,
                "batch_identity_hash": identity_hash,
                "loss": loss_value if loss_finite else None,
                "loss_finite": loss_finite,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "optimizer_step_applied": applied,
                "shared_gradient_combined_hash": gradient["combined_hash"],
                "shared_gradient_max_abs": gradient["max_abs"],
                "shared_gradient_missing_names": gradient["missing"],
                "non_finite_gradient_names": non_finite,
                "shared_gradients_finite": not non_finite,
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "gpu_memory_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            if applied:
                successful_updates += 1
                record["successful_update_index"] = successful_updates
                shared_parameter_map = shared_parameter_hashes(model, shared_names)
                shared_optimizer_map = optimizer_state_hashes(optimizer, parameter_names, shared_names)
                head_map = shared_parameter_hashes(model, head_names)
                # Store only combined hashes per successful update.  Each combined
                # hash is computed from sorted parameter names and native-dtype bytes,
                # so comparison remains bitwise without retaining large per-parameter
                # telemetry maps in the report.
                record["shared_parameter_combined_hash"] = combined_hash(shared_parameter_map)
                record["shared_optimizer_state_combined_hash"] = combined_hash(shared_optimizer_map)
                record["reliability_head_parameter_combined_hash"] = combined_hash(head_map)
            records.append(record)
        except torch.cuda.OutOfMemoryError as error:
            capacity_failure = True
            exception_messages.append(f"step {step}: CUDA OOM: {error}")
            torch.cuda.empty_cache()
            break
        except BaseException as error:  # noqa: BLE001 - preserve evidence and stop this trajectory
            exception_messages.append(f"step {step}: {type(error).__name__}: {error}")
            break
        finally:
            # A scalar loss retains the complete autograd graph until its last
            # reference is dropped; release it and all batch/device references at
            # every attempted step before the next batch is built.
            del batch, metadata, samples, tensors, labels, auxiliary, loss

    final_shared = shared_parameter_hashes(model, shared_names)
    final_head = shared_parameter_hashes(model, head_names)
    result = {
        "label": label,
        "lambda_rel": float(lambda_rel),
        "requested_batch_size": int(batch_size),
        "records": records,
        "attempted_steps": len(records),
        "successful_updates": int(successful_updates),
        "corrupt_steps": int(corrupt_steps),
        "capacity_failure": bool(capacity_failure),
        "exceptions": exception_messages,
        "final_scale": float(scaler.get_scale()),
        "final_shared_parameter_hashes": final_shared,
        "final_shared_parameter_combined_hash": combined_hash(final_shared),
        "final_reliability_head_parameter_hashes": final_head,
        "final_reliability_head_parameter_combined_hash": combined_hash(final_head),
        "initial_reliability_head_parameter_combined_hash": initial_head_hash,
        "reliability_head_frozen": bool(float(lambda_rel) == 0.0),
        "managed_shared_parameter_count": len(managed_shared_names),
        "unmanaged_shared_parameter_count": len(unmanaged_shared_names),
        "unmanaged_shared_parameter_names": unmanaged_shared_names,
    }
    # The optimizer owns the largest post-update GPU allocation.  Release it
    # before the paired trajectory is constructed; the caller also clears the
    # allocator cache between trajectories.
    del optimizer, scaler
    return result


def compare_trajectories(off: Mapping[str, Any], on: Mapping[str, Any], shared_names: Sequence[str]) -> Dict[str, Any]:
    """Compare paired v3 records using bitwise combined hashes, never tolerances."""
    off_records = list(off.get("records", []))
    on_records = list(on.get("records", []))
    length = min(len(off_records), len(on_records))
    batch_mismatches: List[Dict[str, Any]] = []
    step_skip_mismatches: List[Dict[str, Any]] = []
    scale_mismatches: List[Dict[str, Any]] = []
    parameter_mismatches: List[Dict[str, Any]] = []
    optimizer_mismatches: List[Dict[str, Any]] = []
    for index in range(length):
        left, right = off_records[index], on_records[index]
        if left.get("step") != right.get("step") or left.get("batch_identity") != right.get("batch_identity"):
            batch_mismatches.append(
                {
                    "index": index,
                    "off_step": left.get("step"),
                    "on_step": right.get("step"),
                    "off_hash": left.get("batch_identity_hash"),
                    "on_hash": right.get("batch_identity_hash"),
                }
            )
        if bool(left.get("optimizer_step_applied")) != bool(right.get("optimizer_step_applied")):
            step_skip_mismatches.append(
                {
                    "index": index,
                    "step": left.get("step"),
                    "off": bool(left.get("optimizer_step_applied")),
                    "on": bool(right.get("optimizer_step_applied")),
                }
            )
        if (left.get("scale_before"), left.get("scale_after")) != (
            right.get("scale_before"),
            right.get("scale_after"),
        ):
            scale_mismatches.append(
                {
                    "index": index,
                    "step": left.get("step"),
                    "off": [left.get("scale_before"), left.get("scale_after")],
                    "on": [right.get("scale_before"), right.get("scale_after")],
                }
            )
        if bool(left.get("optimizer_step_applied")) and bool(right.get("optimizer_step_applied")):
            if left.get("shared_parameter_combined_hash") != right.get("shared_parameter_combined_hash"):
                parameter_mismatches.append(
                    {
                        "index": index,
                        "step": left.get("step"),
                        "off_combined_hash": left.get("shared_parameter_combined_hash"),
                        "on_combined_hash": right.get("shared_parameter_combined_hash"),
                    }
                )
            if left.get("shared_optimizer_state_combined_hash") != right.get("shared_optimizer_state_combined_hash"):
                optimizer_mismatches.append(
                    {
                        "index": index,
                        "step": left.get("step"),
                        "off_combined_hash": left.get("shared_optimizer_state_combined_hash"),
                        "on_combined_hash": right.get("shared_optimizer_state_combined_hash"),
                    }
                )
    final_parameter_mismatches = [
        name
        for name in shared_names
        if off.get("final_shared_parameter_hashes", {}).get(name)
        != on.get("final_shared_parameter_hashes", {}).get(name)
    ]
    return {
        "comparison_criterion": "SHA-256 over sorted shared parameter/optimizer-state names and native-dtype tensor bytes",
        "compared_steps": length,
        "attempted_steps_off": int(off.get("attempted_steps", 0)),
        "attempted_steps_on": int(on.get("attempted_steps", 0)),
        "successful_updates_off": int(off.get("successful_updates", 0)),
        "successful_updates_on": int(on.get("successful_updates", 0)),
        "corrupt_steps_off": int(off.get("corrupt_steps", 0)),
        "corrupt_steps_on": int(on.get("corrupt_steps", 0)),
        "batch_identity_sequence_identical": not batch_mismatches,
        "batch_identity_mismatches": batch_mismatches,
        "step_skip_sequence_identical": not step_skip_mismatches,
        "step_skip_mismatches": step_skip_mismatches,
        "scale_trajectory_identical": not scale_mismatches,
        "scale_mismatches": scale_mismatches,
        "shared_parameter_mismatch_count": len(parameter_mismatches),
        "shared_parameter_mismatches": parameter_mismatches,
        "shared_optimizer_state_mismatch_count": len(optimizer_mismatches),
        "shared_optimizer_state_mismatches": optimizer_mismatches,
        "shared_parameter_max_abs_difference": 0.0 if not parameter_mismatches else None,
        "final_shared_parameter_mismatch_names": final_parameter_mismatches,
        "final_shared_parameter_bitwise_identical": not final_parameter_mismatches,
        "reliability_head_parameters_allowed_to_differ": True,
        "reliability_head_final_hash_off": off.get("final_reliability_head_parameter_combined_hash"),
        "reliability_head_final_hash_on": on.get("final_reliability_head_parameter_combined_hash"),
    }


def config_record(config: Any, args: argparse.Namespace) -> Dict[str, Any]:
    mmfr = dict(getattr(config, "mmfr_a2", {}) or {})
    corruption = dict(mmfr.get("corruption") or {})
    reliability = dict(mmfr.get("reliability_head") or {})
    return {
        "module": str(args.config),
        "run_id": str(getattr(config, "run_id", "")),
        "protocol": str(mmfr.get("protocol", "")),
        "mode": str(mmfr.get("mode", "")),
        "corruption_basis": str(corruption.get("basis", "")),
        "corruption_seed": int(corruption.get("seed", -1)),
        "p_clean": float(corruption.get("p_clean", -1.0)),
        "max_specs": int(corruption.get("max_specs", -1)),
        "kinds": list(corruption.get("kinds", [])),
        "modalities": list(corruption.get("modalities", [])),
        "severity_encoding": str(corruption.get("severity_encoding", "")),
        "target_composition": str(corruption.get("target_composition", "")),
        "misalignment_validity_transport": str(corruption.get("misalignment_validity_transport", "")),
        "reliability_weight": float(reliability.get("weight", -1.0)),
        "reliability_supervised_channels": list(reliability.get("supervised_channels", [])),
        "train_source": str(config.train_source),
        "train_samples": int(config.num_train_imgs),
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "epochs": int(config.nepochs),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "optimizer": str(config.optimizer),
        "syncbn": True,
        "amp": True,
    }


def run_gate(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    device = resolve_device(args.device)
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "check_id": CHECK_ID,
        "official_test_included": False,
        "checkpoint_read": False,
        "formal_training_run": False,
        "evaluator_run": False,
        "cloud_operation": False,
        "device_requested": str(args.device),
        "requested": {
            "config": args.config,
            "batch_size": int(args.batch_size),
            "max_attempts": int(args.max_attempts),
            "min_successful_updates": int(args.min_successful_updates),
            "min_corrupt_steps": int(args.min_corrupt_steps),
            "lambda_rel_off": float(args.lambda_rel_off),
            "lambda_rel_on": float(args.lambda_rel_on),
            "sample_offset": int(args.sample_offset),
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "failures": [],
        "warnings": [],
    }
    report["runtime"] = configure_runtime(device)
    if int(args.batch_size) != EXPECTED_BATCH_SIZE:
        report["failures"].append(f"batch size is frozen to {EXPECTED_BATCH_SIZE}; got {args.batch_size}")
    if float(args.lambda_rel_off) != EXPECTED_LAMBDA_OFF or float(args.lambda_rel_on) != EXPECTED_LAMBDA_ON:
        report["failures"].append("lambda_rel sweep is frozen to 0.0 and 0.1")
    if int(args.min_successful_updates) < 1:
        report["failures"].append("min_successful_updates must be positive")

    import importlib

    config = importlib.import_module(args.config).C
    report["config"] = config_record(config, args)
    config_path = Path(importlib.import_module(args.config).__file__).resolve()
    report["config"]["config_file"] = str(config_path)
    report["config"]["config_file_sha256"] = sha256_file(config_path)
    if report["config"]["protocol"] != EXPECTED_PROTOCOL:
        report["failures"].append(f"config protocol {report['config']['protocol']!r} != {EXPECTED_PROTOCOL!r}")
    if report["config"]["corruption_basis"] != EXPECTED_CORRUPTION_BASIS:
        report["failures"].append(
            f"config corruption basis {report['config']['corruption_basis']!r} != {EXPECTED_CORRUPTION_BASIS!r}"
        )
    if report["config"]["severity_encoding"] != "single":
        report["failures"].append("v3 config does not declare single severity encoding")
    if report["config"]["target_composition"] != "R_depth_sup_v3(p) = V_state_final(p) * R_depth_synthetic(p)":
        report["failures"].append("v3 target composition is not frozen")
    if report["config"]["misalignment_validity_transport"] != "MID-A":
        report["failures"].append("v3 misalignment validity transport is not MID-A")
    pretrained_path = Path(str(config.pretrained_model)).resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {pretrained_path}")
    pretrained_sha = sha256_file(pretrained_path)
    report["pretrained"] = {
        "path": str(pretrained_path),
        "sha256": pretrained_sha,
        "expected_sha256": EXPECTED_PRETRAINED_SHA256,
        "sha256_matches_expected": pretrained_sha == EXPECTED_PRETRAINED_SHA256,
    }
    if pretrained_sha != EXPECTED_PRETRAINED_SHA256:
        report["failures"].append("pretrained checkpoint SHA-256 mismatch")
    if report["failures"]:
        return report

    dataset = load_train_dataset(config)
    if len(dataset) != int(config.expected_split_samples["train"]):
        report["failures"].append(
            f"train-dev dataset length {len(dataset)} != expected {config.expected_split_samples['train']}"
        )
        return report
    corruption = dict(config.mmfr_a2["corruption"])
    report["train_dev"] = {
        "role": "train-dev",
        "sample_count": len(dataset),
        "batch_size": int(args.batch_size),
        "workers": 0,
        "sample_offset": int(args.sample_offset),
        "helper": "utils.dataloader.mmfr_training_v3.build_mmfr_training_batch_v3",
        "official_test_read": False,
    }
    model = build_model(config, device)
    shared_names = shared_parameter_names(model)
    head_names = reliability_parameter_names(model)
    if not shared_names or not head_names:
        raise RuntimeError(f"v3 model parameter partition invalid: shared={len(shared_names)} head={len(head_names)}")

    initial_state = snapshot_model_state(model)
    initial_head_hash = combined_hash(shared_parameter_hashes(model, head_names))
    report["model"] = {
        "parameter_tensor_count": sum(1 for _ in model.parameters()),
        "parameter_element_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "shared_parameter_count": len(shared_names),
        "reliability_head_parameter_count": len(head_names),
        "norm_layer": "torch.nn.BatchNorm2d",
        "syncbn": False,
        "qualification_norm_override": (
            "regular BatchNorm2d/syncbn=False copied from the passed v2 single-GPU skeleton; "
            "production config, v3 helper and protocol identity are unchanged"
        ),
        "initial_state_device": "cpu",
        "construction": "models.builder.EncoderDecoder with v3 config; v3 helper and telemetry_masks retained",
    }
    plans = [
        plan_step(config, step, int(args.batch_size), int(args.sample_offset))
        for step in range(int(args.max_attempts))
    ]
    report["step_plan"] = {
        "step_seed_base": STEP_SEED_BASE,
        "preprocess_seed_base": PREPROCESS_SEED_BASE,
        "epoch": STEP_EPOCH,
        "batch_size": int(args.batch_size),
        "plans": plans,
    }
    report["memory_policy"] = {
        "initial_state_device": "cpu",
        "per_step_full_parameter_hash_maps_retained": False,
        "per_step_metadata": "compact clean/num_specs/spec_kinds summary only",
        "telemetry_masks_passed_to_model": True,
        "optimizer_released_after_each_trajectory": True,
        "cuda_cache_cleared_between_trajectories": True,
        "smaller_batch_retry": False,
    }

    torch.cuda.reset_peak_memory_stats(device)
    off = run_trajectory(
        label="lambda_rel_0.0",
        model=model,
        config=config,
        dataset=dataset,
        device=device,
        initial_state=initial_state,
        initial_head_hash=initial_head_hash,
        plans=plans,
        batch_size=int(args.batch_size),
        min_successful_updates=int(args.min_successful_updates),
        lambda_rel=float(args.lambda_rel_off),
    )
    # The returned trajectory retains only CPU telemetry and hashes.  Explicitly
    # collect Python objects and cached CUDA blocks before constructing the paired
    # optimizer, while preserving the CPU initial state for exact restoration.
    gc.collect()
    torch.cuda.empty_cache()
    on = run_trajectory(
        label="lambda_rel_0.1",
        model=model,
        config=config,
        dataset=dataset,
        device=device,
        initial_state=initial_state,
        initial_head_hash=initial_head_hash,
        plans=plans,
        batch_size=int(args.batch_size),
        min_successful_updates=int(args.min_successful_updates),
        lambda_rel=float(args.lambda_rel_on),
    )
    gc.collect()
    comparison = compare_trajectories(off, on, shared_names)
    report["trajectory_lambda_rel_off"] = off
    report["trajectory_lambda_rel_on"] = on
    report["comparison"] = comparison

    if int(off["attempted_steps"]) != int(on["attempted_steps"]):
        report["failures"].append(
            "the two trajectories attempted a different number of steps "
            f"({off['attempted_steps']} vs {on['attempted_steps']})"
        )
    if off["capacity_failure"] or on["capacity_failure"]:
        report["failures"].append(
            "capacity_failure: CUDA OOM at frozen batch size 10; no smaller-batch retry was performed"
        )
    if off["exceptions"] or on["exceptions"]:
        report["failures"].extend([f"lambda_rel_0.0: {message}" for message in off["exceptions"]])
        report["failures"].extend([f"lambda_rel_0.1: {message}" for message in on["exceptions"]])
    successful = min(int(off["successful_updates"]), int(on["successful_updates"]))
    if successful < int(args.min_successful_updates):
        report["failures"].append(f"only {successful} paired successful updates; required {args.min_successful_updates}")
    if min(int(off["corrupt_steps"]), int(on["corrupt_steps"])) < int(args.min_corrupt_steps):
        report["failures"].append("not enough corrupted batches were observed")
    if not comparison["batch_identity_sequence_identical"]:
        report["failures"].append("batch identity sequence differs between trajectories")
    if not comparison["step_skip_sequence_identical"]:
        report["failures"].append("step/skip sequence differs between trajectories")
    if not comparison["scale_trajectory_identical"]:
        report["failures"].append("GradScaler scale trajectory differs between trajectories")
    if comparison["shared_parameter_mismatch_count"]:
        report["failures"].append("shared parameters differ after successful updates")
    if comparison["shared_optimizer_state_mismatch_count"]:
        report["failures"].append("shared optimizer state differs after successful updates")
    if not comparison["final_shared_parameter_bitwise_identical"]:
        report["failures"].append("final shared parameters are not bitwise identical")
    off_head_unchanged = off["final_reliability_head_parameter_combined_hash"] == initial_head_hash
    on_head_changed = on["final_reliability_head_parameter_combined_hash"] != initial_head_hash
    if not off_head_unchanged:
        report["failures"].append("reliability head changed on lambda_rel=0.0 side")
    if not on_head_changed:
        report["failures"].append("reliability head did not learn on lambda_rel=0.1 side")
    report["head_update_isolation"] = {
        "initial_combined_hash": initial_head_hash,
        "lambda_rel_0.0_final_combined_hash": off["final_reliability_head_parameter_combined_hash"],
        "lambda_rel_0.1_final_combined_hash": on["final_reliability_head_parameter_combined_hash"],
        "lambda_rel_0.0_unchanged": off_head_unchanged,
        "lambda_rel_0.1_changed": on_head_changed,
        "only_lambda_rel_0.1_side_learns": bool(off_head_unchanged and on_head_changed),
    }
    report["summary"] = {
        "attempted_steps": min(int(off["attempted_steps"]), int(on["attempted_steps"])),
        "successful_updates": successful,
        "corrupt_steps": min(int(off["corrupt_steps"]), int(on["corrupt_steps"])),
        "batch_size": int(args.batch_size),
        "batch_identity_sequence_identical": comparison["batch_identity_sequence_identical"],
        "step_skip_sequence_identical": comparison["step_skip_sequence_identical"],
        "scale_trajectory_identical": comparison["scale_trajectory_identical"],
        "shared_parameters_bitwise_identical_after_each_successful_update": comparison[
            "shared_parameter_mismatch_count"
        ]
        == 0,
        "shared_optimizer_state_bitwise_identical_after_each_successful_update": comparison[
            "shared_optimizer_state_mismatch_count"
        ]
        == 0,
        "only_lambda_rel_0.1_side_learns": bool(off_head_unchanged and on_head_changed),
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["not_executed"] = [
        "formal 500-epoch training",
        "checkpoint save",
        "evaluator",
        "official test",
        "cloud operation",
        "DDP",
    ]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "check_id": CHECK_ID, "official_test_included": False, "failures": [], "warnings": [], "requested": vars(args)}
    try:
        report = run_gate(args)
    except BaseException as error:  # noqa: BLE001 - always preserve structured evidence
        report["failures"] = list(report.get("failures", [])) + [f"unhandled exception: {type(error).__name__}: {error}"]
        report["traceback"] = traceback.format_exc()
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["verdict"] = "PASS" if not report.get("failures") else "FAIL"
    report["status"] = report["verdict"]
    report["exit_code"] = 0 if report["verdict"] == "PASS" else 1
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output)
    report["script_path"] = str(Path(__file__).resolve())
    report["script_sha256"] = sha256_file(Path(__file__).resolve())
    output.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_sha = sha256_file(output)
    print(f"{CHECK_ID}: {report['verdict']}")
    summary = report.get("summary") or {}
    if summary:
        print(f"batch_size={summary.get('batch_size')} attempted_steps={summary.get('attempted_steps')} successful_updates={summary.get('successful_updates')} corrupt_steps={summary.get('corrupt_steps')}")
        print(f"step_skip_identical={summary.get('step_skip_sequence_identical')} scale_identical={summary.get('scale_trajectory_identical')} shared_params_identical={summary.get('shared_parameters_bitwise_identical_after_each_successful_update')} shared_optimizer_state_identical={summary.get('shared_optimizer_state_bitwise_identical_after_each_successful_update')}")
        print(f"only_lambda_rel_0.1_side_learns={summary.get('only_lambda_rel_0.1_side_learns')} peak_cuda_memory_allocated_bytes={summary.get('peak_cuda_memory_allocated_bytes')}")
    for failure in report.get("failures", []):
        print(f"FAILURE: {failure}")
    print(f"JSON report: {output}")
    print(f"SHA-256: {report_sha}")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
