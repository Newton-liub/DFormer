#!/usr/bin/env python3
"""Qualify MMFR E1 Batch 1A C0 and F-lite without starting formal training.

The gate verifies frozen source identities, weights-only restart, four-group
optimizer membership, C0/F initial equivalence, F-lite two-step gradient startup,
Geo.weight updates, and batch-1 CUDA memory/latency. It reads one train-dev item,
performs at most one C0 update and two F-lite updates, and never evaluates val-dev
or official test.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.builder import EncoderDecoder  # noqa: E402
from models.losses.safe_masked_loss import safe_masked_mean  # noqa: E402
from tools.mmfr.a2_v3_amp_update_path_isolation import (  # noqa: E402
    load_train_dataset,
    load_train_item,
)
from utils.dataloader.mmfr_training_v3 import build_mmfr_training_batch_v3  # noqa: E402
from utils.init_func import build_e1_optimizer_param_groups  # noqa: E402
from utils.training_checkpoint import (  # noqa: E402
    file_sha256,
    load_weights_only_model_state,
    optimizer_step_was_applied,
)

SCHEMA_VERSION = "mmfr-e1-batch1a-gateb-v1"
CHECK_ID = "MMFR-E1-Batch1A-C0-F-GateB"
SOURCE_CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3"
C0_CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_C0"
F_CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_FLite"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "mmfr-e1-batch1a-gateb" / "e1-batch1a-gateb.json"
SOURCE_CHECKPOINT = REPO_ROOT / "experiments" / "MMFR_A2_v3" / "checkpoints" / "selector-epoch-420.pth"
SOURCE_CHECKPOINT_SHA256 = "2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597"
SOURCE_MODEL_KEY_COUNT = 812
SOURCE_HASHES = {
    "local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common_v3.py": "3f309effc1a8e005844d8885bfd31cdb05c816e686632aa0d34f0f23bcffa595",
    "local_configs/MUSeg/DFormerv2_S_MMFR_A2_DepthCorrupt_v3.py": "b8ba2dd0c5ac4a3c445779bc44ce1d4e9a283fe309a8e190c8c84b5da2c02bef",
    "models/encoders/DFormerv2.py": "029ce7c5659c9e537165cdbfa5da94a2b003ced07b51e6ee3ee63b9ab10f2695",
    "utils/dataloader/mmfr_training_v3.py": "fff423f7d5208db315b7e521da462775ba6f52dc6d68e17cad07b97e6c54d717",
    "utils/dataloader/multimodal_failure_v3.py": "0a729bcb1b60a71120445894d71459895d2fd61a6bc30dfbccb3ac94108b110f",
    "data/splits/MUSeg/dev-v1/train-dev.txt": "a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470",
    "data/splits/MUSeg/dev-v1/val-dev.txt": "1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83",
}
AUTHORIZED_IMPLEMENTATION_BASELINES = {
    "models/builder.py": "7a843f581f3e87e74baad315be18269eee67ffb4a7a706b17729f7776183ec7e",
    "utils/train.py": "aba1b407d6b53af623145aebeae28cfab0803cbe8d1c2e96d5a3e1a7ea036ad7",
    "utils/init_func.py": "453bf17a613a31898f17d1bba768fbc592bcf8f476c3e2809acd2b9553229fff",
}
GEO_SUFFIX = ".Geo.weight"
FEATURE_PREFIX = "feature_adapter."
EXPECTED_GROUP_ORDER = ("base_decay", "base_no_decay", "new_decay", "new_no_decay")
EXPECTED_GEO_NAMES = tuple(
    sorted(
        f"backbone.layers.{stage}.blocks.{block}.Geo.weight"
        for stage, block_count in enumerate((3, 4, 18, 4))
        for block in range(block_count)
    )
)
EXPECTED_ADAPTER_NAMES = tuple(
    sorted(
        f"feature_adapter.adapters.{stage}.{projection}.{parameter}"
        for stage in (1, 2, 3)
        for projection in ("down", "up")
        for parameter in ("weight", "bias")
    )
)
SYNCBN_NAMES = (
    "backbone.patch_embed.proj.1.weight",
    "backbone.patch_embed.proj.1.bias",
    "backbone.patch_embed.proj.4.weight",
    "backbone.patch_embed.proj.4.bias",
    "backbone.patch_embed.proj.7.weight",
    "backbone.patch_embed.proj.7.bias",
    "backbone.patch_embed.proj.10.weight",
    "backbone.patch_embed.proj.10.bias",
    "backbone.layers.0.downsample.norm.weight",
    "backbone.layers.0.downsample.norm.bias",
    "backbone.layers.1.downsample.norm.weight",
    "backbone.layers.1.downsample.norm.bias",
    "backbone.layers.2.downsample.norm.weight",
    "backbone.layers.2.downsample.norm.bias",
)
SEED = 772961337
PREPROCESS_SEED = 20260921
ATOL = 0.0
RTOL = 0.0


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    try:
        raw = value.numpy().tobytes()
    except (TypeError, RuntimeError):
        raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def names_sha256(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def comparison(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    left_value = left.detach().to(torch.float32).cpu()
    right_value = right.detach().to(torch.float32).cpu()
    delta = (left_value - right_value).abs()
    denominator = torch.maximum(left_value.abs(), right_value.abs()).clamp_min(1.0e-12)
    return {
        "exact_equal": bool(torch.equal(left_value, right_value)),
        "allclose": bool(torch.allclose(left_value, right_value, atol=ATOL, rtol=RTOL)),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "max_rel": float((delta / denominator).max().item()) if delta.numel() else 0.0,
        "left_sha256": tensor_sha256(left),
        "right_sha256": tensor_sha256(right),
    }


def import_config(module_name: str) -> Any:
    return copy.deepcopy(importlib.import_module(module_name).C)


def build_model(config: Any, *, candidate: str) -> tuple[nn.Module, Dict[str, Any]]:
    config.pretrained_model = None
    seed_everything(SEED)
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    model = EncoderDecoder(cfg=config, criterion=criterion, norm_layer=nn.SyncBatchNorm, syncbn=True)
    missing_prefixes = (FEATURE_PREFIX,) if candidate == "F-lite" else ()
    load = load_weights_only_model_state(
        SOURCE_CHECKPOINT,
        model=model,
        expected_sha256=SOURCE_CHECKPOINT_SHA256,
        expected_model_key_count=SOURCE_MODEL_KEY_COUNT,
        allowed_missing_prefixes=missing_prefixes,
    )
    return model, load


def model_counts(model: nn.Module) -> Dict[str, int]:
    parameters = list(model.parameters())
    return {
        "parameter_tensors": len(parameters),
        "trainable_parameter_tensors": sum(int(parameter.requires_grad) for parameter in parameters),
        "total_parameters": sum(int(parameter.numel()) for parameter in parameters),
        "trainable_parameters": sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad),
        "state_keys": len(model.state_dict()),
    }


def shared_state_identity(left: nn.Module, right: nn.Module, *, exclude_prefix: str = "") -> Dict[str, Any]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    names = sorted(name for name in left_state if not exclude_prefix or not name.startswith(exclude_prefix))
    missing = sorted(name for name in names if name not in right_state)
    unequal = sorted(
        name for name in names if name in right_state and not torch.equal(left_state[name], right_state[name])
    )
    extra = sorted(
        name for name in right_state if name not in left_state and not (exclude_prefix and name.startswith(exclude_prefix))
    )
    return {
        "compared_keys": len(names),
        "missing_keys": missing,
        "extra_keys": extra,
        "unequal_keys": unequal,
        "exact_equal": not missing and not extra and not unequal,
    }


def build_real_batch(config: Any) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    dataset = load_train_dataset(config)
    sample = load_train_item(dataset, config, index=0, preprocess_seed=PREPROCESS_SEED)
    rgb = sample["rgb"].unsqueeze(0)
    depth = sample["depth"].unsqueeze(0)
    corruption = dict(config.mmfr_a2["corruption"])
    batch = build_mmfr_training_batch_v3(
        rgb,
        depth,
        [sample["sample_id"]],
        epoch=421,
        iteration=0,
        niters_per_epoch=128,
        nepochs=500,
        rgb_mean=config.norm_mean,
        rgb_std=config.norm_std,
        corruption_seed=int(corruption["seed"]),
        global_rank=0,
        p_clean=float(corruption["p_clean"]),
        max_specs=int(corruption["max_specs"]),
        sample_id_root=getattr(config, "dataset_path", None),
    )
    tensors = {
        "rgb": batch["rgb"],
        "depth": batch["depth"],
        "label": sample["label"].unsqueeze(0),
        "raw_rgb": batch["raw_rgb"],
        "raw_depth": batch["raw_depth"],
        "reliability_target": batch["reliability_target"],
        "reliability_valid_mask": batch["valid_mask"],
        "depth_valid": batch["depth_valid_post"],
        "reliability_telemetry_masks": batch["telemetry_masks"],
    }
    metadata = dict(batch["metadata"][0])
    return tensors, {
        "sample_index": 0,
        "sample_id": str(metadata["sample_id"]),
        "clean": bool(metadata["clean"]),
        "spec_kinds": [str(spec["kind"]) for spec in metadata["specs"]],
        "curriculum_progress": float(metadata["curriculum_progress"]),
        "official_test_included": False,
    }


def to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=False) for name, value in batch.items()}


def auxiliary_kwargs(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "raw_rgb": batch["raw_rgb"],
        "raw_depth": batch["raw_depth"],
        "reliability_target": batch["reliability_target"],
        "reliability_valid_mask": batch["reliability_valid_mask"],
        "depth_valid": batch["depth_valid"],
        "reliability_telemetry_masks": batch["reliability_telemetry_masks"],
    }


def loss_components(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    seed_everything(SEED)
    logits = model(batch["rgb"], batch["depth"])
    valid = batch["label"].long() != int(model.cfg.background)
    segmentation = safe_masked_mean(model.criterion(logits, batch["label"].long()), valid)
    reliability = model.reliability_auxiliary_loss(**auxiliary_kwargs(batch))
    total = segmentation + model.reliability_weight * reliability
    return {
        "segmentation": float(segmentation.detach().item()),
        "reliability": float(reliability.detach().item()),
        "total": float(total.detach().item()),
        "finite": all(math.isfinite(float(value.detach().item())) for value in (segmentation, reliability, total)),
    }


def capture_forward(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    decoder_inputs: list[tuple[torch.Tensor, ...]] = []
    original_forward = model.decode_head.forward

    def capture_decoder_forward(values, *args, **kwargs):
        decoder_inputs.append(tuple(value.detach().clone() for value in values))
        return original_forward(values, *args, **kwargs)

    model.decode_head.forward = capture_decoder_forward
    try:
        model.eval()
        seed_everything(SEED)
        with torch.no_grad():
            logits = model(batch["rgb"], batch["depth"])
            prediction = logits.argmax(dim=1)
        seed_everything(SEED)
        with torch.no_grad():
            total_loss = model(batch["rgb"], batch["depth"], batch["label"], **auxiliary_kwargs(batch))
    finally:
        model.decode_head.forward = original_forward
    if not decoder_inputs:
        raise RuntimeError("decoder input capture did not observe decode_head.forward")
    return {
        "logits": logits.detach(),
        "prediction": prediction.detach(),
        "loss": total_loss.detach(),
        "decoder_inputs": decoder_inputs[0],
    }


def optimizer_and_summary(model: nn.Module, config: Any) -> tuple[torch.optim.Optimizer, Dict[str, Any]]:
    e1 = dict(config.e1_batch1)
    groups = build_e1_optimizer_param_groups(
        model,
        nn.SyncBatchNorm,
        base_lr=float(e1["base_lr"]),
        new_lr=float(e1["new_module_lr"]),
        weight_decay=float(e1["weight_decay"]),
        new_parameter_prefixes=(FEATURE_PREFIX,),
        expected_geo_weight_count=29,
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    membership = {name: 0 for name, parameter in model.named_parameters() if parameter.requires_grad}
    summaries = []
    group_by_name = {}
    for group in optimizer.param_groups:
        names = [names_by_id[id(parameter)] for parameter in group["params"]]
        for name in names:
            membership[name] += 1
            group_by_name[name] = str(group["group_name"])
        summaries.append(
            {
                "name": str(group["group_name"]),
                "parameter_tensors": len(names),
                "parameter_elements": sum(int(parameter.numel()) for parameter in group["params"]),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "name_digest": names_sha256(names),
                "parameter_names": sorted(names),
            }
        )
    invalid = {name: count for name, count in membership.items() if count != 1}
    geo_names = sorted(name for name in membership if name.endswith(GEO_SUFFIX))
    reliability_names = sorted(name for name in membership if name.startswith("reliability_estimator."))
    feature_names = sorted(name for name in membership if name.startswith(FEATURE_PREFIX))
    return optimizer, {
        "groups": summaries,
        "group_order": [summary["name"] for summary in summaries],
        "trainable_parameter_tensors": len(membership),
        "trainable_parameter_elements": sum(
            int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
        ),
        "membership_all_one": not invalid,
        "invalid_membership": invalid,
        "geo_weight_count": len(geo_names),
        "geo_names": geo_names,
        "geo_names_exact": tuple(geo_names) == EXPECTED_GEO_NAMES,
        "geo_all_base_decay": all(group_by_name.get(name) == "base_decay" for name in geo_names),
        "syncbn_all_base_no_decay": all(group_by_name.get(name) == "base_no_decay" for name in SYNCBN_NAMES),
        "reliability_groups": {name: group_by_name.get(name) for name in reliability_names},
        "feature_names": feature_names,
        "feature_names_exact": tuple(feature_names) in ((), EXPECTED_ADAPTER_NAMES),
        "feature_groups": {name: group_by_name.get(name) for name in feature_names},
    }


def clone_parameters(model: nn.Module, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    lookup = dict(model.named_parameters())
    return {name: lookup[name].detach().cpu().clone() for name in names}


def gradient_status(model: nn.Module, names: Sequence[str]) -> Dict[str, Any]:
    lookup = dict(model.named_parameters())
    missing = []
    non_finite = []
    nonzero = []
    norms = {}
    for name in names:
        gradient = lookup[name].grad
        if gradient is None:
            missing.append(name)
            continue
        finite = bool(torch.isfinite(gradient).all().item())
        if not finite:
            non_finite.append(name)
        norm = float(gradient.detach().to(torch.float32).norm().item())
        norms[name] = norm
        if norm > 0.0:
            nonzero.append(name)
    return {
        "count": len(names),
        "missing": missing,
        "non_finite": non_finite,
        "finite_all_present": not missing and not non_finite,
        "nonzero_count": len(nonzero),
        "nonzero_names": nonzero,
        "norms": norms,
    }


def change_status(model: nn.Module, before: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    lookup = dict(model.named_parameters())
    changed = []
    max_abs = {}
    for name, old in before.items():
        current = lookup[name].detach().cpu()
        delta = (current - old).abs()
        value = float(delta.max().item()) if delta.numel() else 0.0
        max_abs[name] = value
        if value > 0.0:
            changed.append(name)
    return {
        "count": len(before),
        "changed_count": len(changed),
        "changed_names": changed,
        "all_changed": len(changed) == len(before),
        "max_abs_change": max_abs,
    }


def make_scaler(config: Any) -> torch.cuda.amp.GradScaler:
    scaler = dict(config.e1_batch1["grad_scaler"])
    return torch.cuda.amp.GradScaler(
        init_scale=float(scaler["initial_scale"]),
        growth_factor=float(scaler["growth_factor"]),
        backoff_factor=float(scaler["backoff_factor"]),
        growth_interval=int(scaler["growth_interval"]),
    )


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    batch: Mapping[str, torch.Tensor],
    *,
    seed: int,
) -> Dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seed_everything(seed)
    scale_before = float(scaler.get_scale())
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(batch["rgb"], batch["depth"], batch["label"], **auxiliary_kwargs(batch))
    scaler.scale(loss).backward()
    scale_after_backward = float(scaler.get_scale())
    result = {
        "loss": float(loss.detach().item()),
        "loss_finite": bool(torch.isfinite(loss.detach()).all().item()),
        "scale_before": scale_before,
        "scale_after_backward": scale_after_backward,
    }
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    result.update(
        {
            "scale_after": scale_after,
            "optimizer_step_applied": bool(optimizer_step_was_applied(scale_before, scale_after)),
        }
    )
    return result


def timed_cuda(callable_fn, *, warmup: int, repeats: int) -> Dict[str, Any]:
    for _ in range(warmup):
        callable_fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = callable_fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
        del result
    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.mean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "samples_ms": values,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=20)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "check_id": CHECK_ID,
        "status": "ERROR",
        "official_test_included": False,
        "checks": checks,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Gate-B cost and update qualification requires one local CUDA GPU")
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

        source_identity = {}
        for relative, expected in SOURCE_HASHES.items():
            path = REPO_ROOT / relative
            actual = file_sha256(path)
            source_identity[relative] = {"expected": expected, "actual": actual, "match": actual == expected}
            check(f"source_hash.{relative}", actual == expected, actual)
        source_identity["authorized_implementation_baselines"] = {
            relative: {
                "preimplementation_sha256": expected,
                "gateb_implementation_sha256": file_sha256(REPO_ROOT / relative),
                "preimplementation_identity_verified_before_edit": True,
            }
            for relative, expected in AUTHORIZED_IMPLEMENTATION_BASELINES.items()
        }
        checkpoint_hash = file_sha256(SOURCE_CHECKPOINT)
        check("source_checkpoint.sha256", checkpoint_hash == SOURCE_CHECKPOINT_SHA256, checkpoint_hash)
        source_identity["checkpoint"] = {
            "path": str(SOURCE_CHECKPOINT),
            "size_bytes": SOURCE_CHECKPOINT.stat().st_size,
            "expected_sha256": SOURCE_CHECKPOINT_SHA256,
            "actual_sha256": checkpoint_hash,
        }
        source_identity["qualification_git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()

        source_config = import_config(SOURCE_CONFIG_MODULE)
        c0_config = import_config(C0_CONFIG_MODULE)
        f_config = import_config(F_CONFIG_MODULE)
        implementation_hashes = {
            relative: file_sha256(REPO_ROOT / relative)
            for relative in (
                "local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_Common.py",
                "local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_C0.py",
                "local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_FLite.py",
                "models/feature_adapter.py",
                "models/builder.py",
                "utils/init_func.py",
                "utils/train.py",
                "utils/training_checkpoint.py",
                "tools/mmfr/e1_batch1a_gateb.py",
            )
        }
        source_identity["gateb_implementation_hashes"] = implementation_hashes
        for label, config, candidate in (
            ("c0", c0_config, "C0"),
            ("f", f_config, "F-lite"),
        ):
            e1 = dict(config.e1_batch1)
            source_frozen = dict(config.mmfr_a2.get("frozen") or {})
            active_schedule = {
                "candidate": e1.get("candidate"),
                "epochs": int(config.nepochs),
                "iterations_per_epoch": int(config.niters_per_epoch),
                "base_lr": float(e1.get("base_lr", -1.0)),
                "new_lr": float(e1.get("new_module_lr", -1.0)),
                "weight_decay": float(e1.get("weight_decay", -1.0)),
                "warmup_successful_updates": int(e1.get("warmup_successful_updates", -1)),
                "required_successful_updates": int(e1.get("required_successful_updates", -1)),
                "optimizer_groups": list(e1.get("optimizer_groups") or ()),
                "metadata_scope": source_frozen.get("metadata_scope"),
                "active_schedule_authority": source_frozen.get("active_schedule_authority"),
            }
            expected_schedule = {
                "candidate": candidate,
                "epochs": 20,
                "iterations_per_epoch": 128,
                "base_lr": 1e-5,
                "new_lr": 3e-5,
                "weight_decay": 0.01,
                "warmup_successful_updates": 128,
                "required_successful_updates": 2560,
                "optimizer_groups": list(EXPECTED_GROUP_ORDER),
                "metadata_scope": "source-a2-v3-history-only",
                "active_schedule_authority": "e1_batch1",
            }
            check(f"config.{label}.active_schedule", active_schedule == expected_schedule, active_schedule)
        source_model, source_load = build_model(source_config, candidate="source-A2")
        c0_model, c0_load = build_model(c0_config, candidate="C0")
        f_model, f_load = build_model(f_config, candidate="F-lite")
        check("checkpoint.model_key_count", source_load["model_key_count"] == 812, source_load["model_key_count"])
        check("checkpoint.source_strict", not source_load["missing_keys"] and not source_load["unexpected_keys"], source_load)
        check("checkpoint.c0_strict", not c0_load["missing_keys"] and not c0_load["unexpected_keys"], c0_load)
        expected_f_missing = list(EXPECTED_ADAPTER_NAMES)
        check("checkpoint.f_only_adapter_missing", f_load["missing_keys"] == expected_f_missing, f_load["missing_keys"])
        source_identity["checkpoint_metadata"] = source_load

        counts = {
            "source": model_counts(source_model),
            "C0": model_counts(c0_model),
            "F-lite": model_counts(f_model),
        }
        added_parameters = counts["F-lite"]["trainable_parameters"] - counts["C0"]["trainable_parameters"]
        configured_added_parameters = int(f_config.e1_batch1["feature_adapter"]["expected_trainable_parameters"])
        check("f.parameter_count", added_parameters == 173152, added_parameters)
        check(
            "f.parameter_count_matches_config",
            added_parameters == configured_added_parameters,
            {"actual": added_parameters, "configured": configured_added_parameters},
        )
        source_c0_state = shared_state_identity(source_model, c0_model)
        source_f_state = shared_state_identity(source_model, f_model, exclude_prefix=FEATURE_PREFIX)
        check("c0.weights_equal_source", source_c0_state["exact_equal"], source_c0_state)
        check("f.shared_weights_equal_source", source_f_state["exact_equal"], source_f_state)

        c0_optimizer, c0_optimizer_summary = optimizer_and_summary(c0_model, c0_config)
        f_optimizer, f_optimizer_summary = optimizer_and_summary(f_model, f_config)
        expected_group_specs = {
            "base_decay": {"lr": 1e-5, "weight_decay": 0.01},
            "base_no_decay": {"lr": 1e-5, "weight_decay": 0.0},
            "new_decay": {"lr": 3e-5, "weight_decay": 0.01},
            "new_no_decay": {"lr": 3e-5, "weight_decay": 0.0},
        }
        expected_reliability_groups = {
            name: ("base_no_decay" if name.endswith(".bias") else "base_decay")
            for name, _ in c0_model.named_parameters()
            if name.startswith("reliability_estimator.")
        }
        for label, summary in (("c0", c0_optimizer_summary), ("f", f_optimizer_summary)):
            check(f"optimizer.{label}.membership_all_one", summary["membership_all_one"], summary["invalid_membership"])
            check(
                f"optimizer.{label}.group_order",
                tuple(summary["group_order"]) == EXPECTED_GROUP_ORDER,
                summary["group_order"],
            )
            actual_group_specs = {
                group["name"]: {"lr": group["lr"], "weight_decay": group["weight_decay"]}
                for group in summary["groups"]
            }
            check(
                f"optimizer.{label}.group_lr_weight_decay",
                actual_group_specs == expected_group_specs,
                actual_group_specs,
            )
            check(
                f"optimizer.{label}.geo_exact_29_base_decay",
                summary["geo_names_exact"] and summary["geo_all_base_decay"],
                summary["geo_names"],
            )
            check(f"optimizer.{label}.syncbn_14_base_no_decay", summary["syncbn_all_base_no_decay"], "14 audited SyncBN tensors")
            check(
                f"optimizer.{label}.reliability_exact_groups",
                summary["reliability_groups"] == expected_reliability_groups,
                summary["reliability_groups"],
            )
        c0_new_groups = {
            group["name"]: group["parameter_tensors"]
            for group in c0_optimizer_summary["groups"]
            if group["name"].startswith("new_")
        }
        check("optimizer.c0.new_groups_empty", c0_new_groups == {"new_decay": 0, "new_no_decay": 0}, c0_new_groups)
        check(
            "optimizer.f.adapter_names_exact",
            tuple(f_optimizer_summary["feature_names"]) == EXPECTED_ADAPTER_NAMES,
            f_optimizer_summary["feature_names"],
        )
        expected_feature_groups = {
            name: ("new_no_decay" if name.endswith(".bias") else "new_decay")
            for name in EXPECTED_ADAPTER_NAMES
        }
        check(
            "optimizer.f.adapter_exact_groups",
            f_optimizer_summary["feature_groups"] == expected_feature_groups,
            f_optimizer_summary["feature_groups"],
        )

        batch_cpu, batch_identity = build_real_batch(c0_config)
        curriculum_bounds = (
            float(c0_config.e1_batch1["curriculum_progress_start"]),
            float(c0_config.e1_batch1["curriculum_progress_end"]),
        )
        check(
            "config.curriculum_progress_in_frozen_range",
            curriculum_bounds[0] <= batch_identity["curriculum_progress"] <= curriculum_bounds[1],
            {"progress": batch_identity["curriculum_progress"], "bounds": curriculum_bounds},
        )
        batch = to_device(batch_cpu, device)
        source_model.to(device)
        c0_model.to(device)
        f_model.to(device)

        source_forward = capture_forward(source_model, batch)
        c0_forward = capture_forward(c0_model, batch)
        f_forward = capture_forward(f_model, batch)
        c0_identity = {
            "logits": comparison(source_forward["logits"], c0_forward["logits"]),
            "prediction": comparison(source_forward["prediction"], c0_forward["prediction"]),
            "loss": comparison(source_forward["loss"], c0_forward["loss"]),
            "decoder_inputs": [
                comparison(left, right)
                for left, right in zip(source_forward["decoder_inputs"], c0_forward["decoder_inputs"])
            ],
        }
        check("c0.logits_identity", c0_identity["logits"]["exact_equal"], c0_identity["logits"])
        check("c0.prediction_identity", c0_identity["prediction"]["exact_equal"], c0_identity["prediction"])
        check("c0.loss_identity", c0_identity["loss"]["exact_equal"], c0_identity["loss"])

        f_noop = {
            "logits": comparison(source_forward["logits"], f_forward["logits"]),
            "prediction": comparison(source_forward["prediction"], f_forward["prediction"]),
            "loss": comparison(source_forward["loss"], f_forward["loss"]),
            "decoder_inputs": [
                comparison(left, right)
                for left, right in zip(source_forward["decoder_inputs"], f_forward["decoder_inputs"])
            ],
            "stages": {},
        }
        f_model.eval()
        seed_everything(SEED)
        with torch.no_grad():
            backbone_features = f_model.backbone(batch["rgb"], batch["depth"])
            if len(backbone_features) == 2:
                backbone_features = backbone_features[0]
            for index in (1, 2, 3):
                adapter = f_model.feature_adapter.adapters[str(index)]
                residual = adapter.residual(backbone_features[index])
                adapted = adapter(backbone_features[index])
                f_noop["stages"][str(index)] = {
                    "residual_exact_zero": bool(torch.count_nonzero(residual).item() == 0),
                    "feature_identity": comparison(backbone_features[index], adapted),
                    "residual_max_abs": float(residual.abs().max().item()),
                }
        check("f.noop_stages", all(item["residual_exact_zero"] and item["feature_identity"]["exact_equal"] for item in f_noop["stages"].values()), f_noop["stages"])
        check("f.decoder_input_identity", all(item["exact_equal"] for item in f_noop["decoder_inputs"]), f_noop["decoder_inputs"])
        check("f.logits_identity", f_noop["logits"]["exact_equal"], f_noop["logits"])
        check("f.prediction_identity", f_noop["prediction"]["exact_equal"], f_noop["prediction"])
        check("f.loss_identity", f_noop["loss"]["exact_equal"], f_noop["loss"])

        loss_values = {
            "C0": loss_components(c0_model, batch),
            "F-lite": loss_components(f_model, batch),
        }
        check("c0.loss_components_finite", loss_values["C0"]["finite"], loss_values["C0"])
        check("f.loss_components_finite", loss_values["F-lite"]["finite"], loss_values["F-lite"])

        # Isolated C0 one-step memory/gradient/update qualification.
        source_model.cpu()
        f_model.cpu()
        torch.cuda.empty_cache()
        c0_model.to(device)
        batch = to_device(batch_cpu, device)
        c0_geo_names = sorted(name for name, _ in c0_model.named_parameters() if name.endswith(GEO_SUFFIX))
        c0_shared_name = "backbone.patch_embed.proj.0.weight"
        c0_reliability_names = sorted(name for name, _ in c0_model.named_parameters() if name.startswith("reliability_estimator."))
        c0_geo_before = clone_parameters(c0_model, c0_geo_names)
        c0_scaler = make_scaler(c0_config)
        torch.cuda.reset_peak_memory_stats(device)
        c0_step = training_step(c0_model, c0_optimizer, c0_scaler, batch, seed=SEED + 1)
        c0_geo_grad = gradient_status(c0_model, c0_geo_names)
        c0_shared_grad = gradient_status(c0_model, [c0_shared_name])
        c0_reliability_grad = gradient_status(c0_model, c0_reliability_names)
        c0_geo_change = change_status(c0_model, c0_geo_before)
        torch.cuda.synchronize()
        c0_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        check("c0.step_applied", c0_step["optimizer_step_applied"], c0_step)
        check("c0.geo_gradient_finite", c0_geo_grad["finite_all_present"], c0_geo_grad)
        check("c0.geo_updated", c0_geo_change["changed_count"] > 0, c0_geo_change["changed_count"])
        check("c0.shared_gradient_finite", c0_shared_grad["finite_all_present"], c0_shared_grad)
        check("c0.reliability_gradient_finite", c0_reliability_grad["finite_all_present"], c0_reliability_grad)

        # Isolated F two-step memory/gradient/update qualification.
        c0_model.cpu()
        del c0_optimizer
        torch.cuda.empty_cache()
        f_model.to(device)
        batch = to_device(batch_cpu, device)
        f_optimizer, _ = optimizer_and_summary(f_model, f_config)
        f_scaler = make_scaler(f_config)
        up_names = sorted(
            name for name, _ in f_model.named_parameters() if name.startswith(FEATURE_PREFIX) and ".up." in name
        )
        down_names = sorted(
            name for name, _ in f_model.named_parameters() if name.startswith(FEATURE_PREFIX) and ".down." in name
        )
        f_geo_names = sorted(name for name, _ in f_model.named_parameters() if name.endswith(GEO_SUFFIX))
        f_reliability_names = sorted(name for name, _ in f_model.named_parameters() if name.startswith("reliability_estimator."))
        up_before = clone_parameters(f_model, up_names)
        down_before = clone_parameters(f_model, down_names)
        geo_before = clone_parameters(f_model, f_geo_names)
        torch.cuda.reset_peak_memory_stats(device)
        f_step1 = training_step(f_model, f_optimizer, f_scaler, batch, seed=SEED + 1)
        f_step1_up_grad = gradient_status(f_model, up_names)
        f_step1_down_grad = gradient_status(f_model, down_names)
        f_step1_up_change = change_status(f_model, up_before)
        f_step1_geo_grad = gradient_status(f_model, f_geo_names)
        f_step1_geo_change = change_status(f_model, geo_before)
        f_step1_reliability_grad = gradient_status(f_model, f_reliability_names)
        f_memory_after_step1 = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        down_after_step1 = clone_parameters(f_model, down_names)
        f_step2 = training_step(f_model, f_optimizer, f_scaler, batch, seed=SEED + 2)
        f_step2_down_grad = gradient_status(f_model, down_names)
        f_step2_down_change = change_status(f_model, down_after_step1)
        torch.cuda.synchronize()
        f_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        check("f.step1_applied", f_step1["optimizer_step_applied"], f_step1)
        check("f.step1_up_gradient", f_step1_up_grad["finite_all_present"] and f_step1_up_grad["nonzero_count"] == len(up_names), f_step1_up_grad)
        check("f.step1_up_updated", f_step1_up_change["all_changed"], f_step1_up_change)
        check("f.step1_down_zero_allowed", f_step1_down_grad["finite_all_present"] and f_step1_down_grad["nonzero_count"] == 0, f_step1_down_grad)
        check("f.step2_applied", f_step2["optimizer_step_applied"], f_step2)
        check("f.step2_down_gradient", f_step2_down_grad["finite_all_present"] and f_step2_down_grad["nonzero_count"] == len(down_names), f_step2_down_grad)
        check("f.step2_down_updated", f_step2_down_change["all_changed"], f_step2_down_change)
        check("f.geo_gradient_finite", f_step1_geo_grad["finite_all_present"], f_step1_geo_grad)
        check("f.geo_updated", f_step1_geo_change["changed_count"] > 0, f_step1_geo_change["changed_count"])
        check("f.reliability_gradient_finite", f_step1_reliability_grad["finite_all_present"], f_step1_reliability_grad)

        memory_delta = {
            "allocated_bytes": f_memory["peak_allocated_bytes"] - c0_memory["peak_allocated_bytes"],
            "reserved_bytes": f_memory["peak_reserved_bytes"] - c0_memory["peak_reserved_bytes"],
            "allocated_mib": (f_memory["peak_allocated_bytes"] - c0_memory["peak_allocated_bytes"]) / (1024 ** 2),
            "reserved_mib": (f_memory["peak_reserved_bytes"] - c0_memory["peak_reserved_bytes"]) / (1024 ** 2),
        }
        check("f.memory_gate", memory_delta["allocated_mib"] <= 512.0, memory_delta)

        # Interleaved batch-1 latency after update qualification; no evaluation dataset is read.
        c0_model.to(device)
        f_model.to(device)
        batch = to_device(batch_cpu, device)
        c0_model.eval()
        f_model.eval()

        def c0_inference():
            with torch.no_grad():
                return c0_model(batch["rgb"], batch["depth"])

        def f_inference():
            with torch.no_grad():
                return f_model(batch["rgb"], batch["depth"])

        def c0_forward_loss():
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                return c0_model(batch["rgb"], batch["depth"], batch["label"], **auxiliary_kwargs(batch))

        def f_forward_loss():
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                return f_model(batch["rgb"], batch["depth"], batch["label"], **auxiliary_kwargs(batch))

        latency = {
            "C0": {
                "forward": timed_cuda(c0_forward_loss, warmup=args.latency_warmup, repeats=args.latency_repeats),
                "inference": timed_cuda(c0_inference, warmup=args.latency_warmup, repeats=args.latency_repeats),
            },
            "F-lite": {
                "forward": timed_cuda(f_forward_loss, warmup=args.latency_warmup, repeats=args.latency_repeats),
                "inference": timed_cuda(f_inference, warmup=args.latency_warmup, repeats=args.latency_repeats),
            },
        }
        latency["delta"] = {
            "forward_ms": latency["F-lite"]["forward"]["median_ms"] - latency["C0"]["forward"]["median_ms"],
            "forward_percent": 100.0 * (
                latency["F-lite"]["forward"]["median_ms"] / latency["C0"]["forward"]["median_ms"] - 1.0
            ),
            "inference_ms": latency["F-lite"]["inference"]["median_ms"] - latency["C0"]["inference"]["median_ms"],
            "inference_percent": 100.0 * (
                latency["F-lite"]["inference"]["median_ms"] / latency["C0"]["inference"]["median_ms"] - 1.0
            ),
        }
        check("f.inference_latency_gate", latency["delta"]["inference_percent"] <= 5.0, latency["delta"])

        c0_pass = all(
            item["ok"]
            for item in checks
            if item["name"].startswith(("source_", "checkpoint.", "config.", "c0.", "optimizer.c0"))
        )
        f_pass = c0_pass and all(
            item["ok"]
            for item in checks
            if item["name"].startswith(("f.", "optimizer.f"))
        )
        report.update(
            {
                "status": "PASS" if c0_pass and f_pass else "FAIL",
                "gate_b": {"C0": "PASS" if c0_pass else "FAIL", "F-lite": "PASS" if f_pass else "FAIL"},
                "source_identity": source_identity,
                "batch_identity": batch_identity,
                "model_counts": counts,
                "added_f_trainable_parameters": added_parameters,
                "optimizer": {"C0": c0_optimizer_summary, "F-lite": f_optimizer_summary},
                "c0_identity": c0_identity,
                "f_initial_noop": f_noop,
                "loss_components": loss_values,
                "gradients_and_updates": {
                    "C0": {
                        "step": c0_step,
                        "geo_gradient": c0_geo_grad,
                        "geo_change": c0_geo_change,
                        "shared_gradient": c0_shared_grad,
                        "reliability_gradient": c0_reliability_grad,
                    },
                    "F-lite": {
                        "step1": f_step1,
                        "step1_up_gradient": f_step1_up_grad,
                        "step1_up_change": f_step1_up_change,
                        "step1_down_gradient": f_step1_down_grad,
                        "step2": f_step2,
                        "step2_down_gradient": f_step2_down_grad,
                        "step2_down_change": f_step2_down_change,
                        "geo_gradient": f_step1_geo_grad,
                        "geo_change": f_step1_geo_change,
                        "reliability_gradient": f_step1_reliability_grad,
                    },
                },
                "cost_probe": {
                    "C0": c0_memory,
                    "F-lite_step1": f_memory_after_step1,
                    "F-lite": f_memory,
                    "memory_delta": memory_delta,
                    "latency": latency,
                    "device": torch.cuda.get_device_name(device),
                    "batch_size": 1,
                },
                "source_state_identity": {"C0": source_c0_state, "F-lite_shared": source_f_state},
            }
        )
    except BaseException as error:  # noqa: BLE001 - qualification must preserve failure evidence
        report["status"] = "ERROR"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        print(report["error"]["traceback"], end="")
    report["duration_seconds"] = float(time.perf_counter() - started)
    report["failed_checks"] = [item for item in checks if not item["ok"]]
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    report["report_sha256"] = file_sha256(output)
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": report["report_sha256"]}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
