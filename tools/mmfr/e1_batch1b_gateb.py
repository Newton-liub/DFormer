#!/usr/bin/env python3
"""Qualify MMFR E1 Batch 1B R-OE-lite without starting formal training.

This is a minimal Gate-B probe only. It verifies source identity and matched-C0
reuse fairness, the observable-empty detector boundary, per-sample exact bypass,
the frozen substitute contract, optimizer coverage, one finite AMP update, and a
small CUDA cost probe. It never reads val-dev or official-test and never starts
an epoch-based training run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
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
from tools.mmfr.e1_batch1a_gateb import (  # noqa: E402
    auxiliary_kwargs,
    build_real_batch,
    change_status,
    comparison,
    gradient_status,
    make_scaler,
    model_counts,
    seed_everything,
    tensor_sha256,
    timed_cuda,
    to_device,
    training_step,
)
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.dataloader.dataloader import get_train_loader  # noqa: E402
from utils.init_func import build_e1_optimizer_param_groups  # noqa: E402
from utils.training_checkpoint import (  # noqa: E402
    file_sha256,
    load_weights_only_model_state,
)

SCHEMA_VERSION = "mmfr-e1-batch1b-roe-gateb-v1"
CHECK_ID = "MMFR-E1-Batch1B-R-OE-lite-GateB"
SOURCE_CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3"
C0_CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_C0"
ROE_CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1B_R_OE"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "mmfr-e1-batch1b-gateb" / "e1-batch1b-gateb.json"
SOURCE_CHECKPOINT = REPO_ROOT / "experiments" / "MMFR_A2_v3" / "checkpoints" / "selector-epoch-420.pth"
SOURCE_CHECKPOINT_SHA256 = "2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597"
SOURCE_MODEL_KEY_COUNT = 812
TRAIN_SPLIT = REPO_ROOT / "data" / "splits" / "MUSeg" / "dev-v1" / "train-dev.txt"
VAL_SPLIT = REPO_ROOT / "data" / "splits" / "MUSeg" / "dev-v1" / "val-dev.txt"
TRAIN_SPLIT_SHA256 = "a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470"
VAL_SPLIT_SHA256 = "1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83"
C0_FINAL_CHECKPOINT = (
    REPO_ROOT
    / "cloud"
    / "MMFR_E1_Batch1A_local_transfer_20260922"
    / "MMFR_E1_Batch1A_local_transfer_20260922"
    / "C0"
    / "checkpoint"
    / "update-2560.pth"
)
C0_FINAL_CHECKPOINT_SHA256 = "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a"
SEED = 772961337
PREPROCESS_SEED = 20260921
EXPECTED_GROUP_ORDER = ("base_decay", "base_no_decay", "new_decay", "new_no_decay")
ROE_PREFIX = "roe_substitute."
EXPECTED_ROE_PARAMETER_NAMES = tuple(
    sorted(
        f"{ROE_PREFIX}{layer}.{parameter}"
        for layer in ("e1", "e2", "b1", "b2", "d1", "d2", "head")
        for parameter in ("weight", "bias")
    )
)
EXPECTED_ROE_WEIGHT_ELEMENTS = 3_301_232
EXPECTED_ROE_BIAS_ELEMENTS = 1_553
EXPECTED_ROE_TRAINABLE_ELEMENTS = 3_302_785
GEO_SUFFIX = ".Geo.weight"
EXPECTED_GEO_NAMES = tuple(
    sorted(
        f"backbone.layers.{stage}.blocks.{block}.Geo.weight"
        for stage, block_count in enumerate((3, 4, 18, 4))
        for block in range(block_count)
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


class _LoaderEngineStub:
    distributed = False
    world_size = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 32:
            return value.detach().cpu().tolist()
        return {"shape": list(value.shape), "sha256": tensor_sha256(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _capture_rng() -> Dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not torch.equal(left["cpu"], right["cpu"]):
        return False
    if len(left["cuda"]) != len(right["cuda"]):
        return False
    return all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


def _rng_summary(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "cpu_sha256": tensor_sha256(state["cpu"]),
        "cuda_sha256": [tensor_sha256(item) for item in state["cuda"]],
    }


def import_config(module_name: str) -> Any:
    return copy.deepcopy(importlib.import_module(module_name).C)


def build_model_from_current_rng(config: Any, *, candidate: str) -> tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Build without reseeding so the caller can audit construction RNG consumption."""

    config.pretrained_model = None
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    model = EncoderDecoder(cfg=config, criterion=criterion, norm_layer=nn.SyncBatchNorm, syncbn=True)
    post_build_rng = _capture_rng()
    allowed_missing = (ROE_PREFIX,) if candidate == "R-OE-lite" else ()
    load = load_weights_only_model_state(
        SOURCE_CHECKPOINT,
        model=model,
        expected_sha256=SOURCE_CHECKPOINT_SHA256,
        expected_model_key_count=SOURCE_MODEL_KEY_COUNT,
        allowed_missing_prefixes=allowed_missing,
    )
    return model, load, post_build_rng


def names_sha256(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def optimizer_and_summary(model: nn.Module, config: Any) -> tuple[torch.optim.Optimizer, Dict[str, Any]]:
    e1 = dict(config.e1_batch1)
    groups = build_e1_optimizer_param_groups(
        model,
        nn.SyncBatchNorm,
        base_lr=float(e1["base_lr"]),
        new_lr=float(e1["new_module_lr"]),
        weight_decay=float(e1["weight_decay"]),
        new_parameter_prefixes=(ROE_PREFIX,),
        expected_geo_weight_count=29,
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    membership = {name: 0 for name, parameter in model.named_parameters() if parameter.requires_grad}
    group_by_name: Dict[str, str] = {}
    summaries = []
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
    roe_names = sorted(name for name in membership if name.startswith(ROE_PREFIX))
    geo_names = sorted(name for name in membership if name.endswith(GEO_SUFFIX))
    reliability_names = sorted(name for name in membership if name.startswith("reliability_estimator."))
    return optimizer, {
        "groups": summaries,
        "group_order": [summary["name"] for summary in summaries],
        "trainable_parameter_tensors": len(membership),
        "trainable_parameter_elements": sum(
            int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
        ),
        "membership_all_one": not invalid,
        "invalid_membership": invalid,
        "roe_names": roe_names,
        "roe_names_exact": tuple(roe_names) == EXPECTED_ROE_PARAMETER_NAMES,
        "roe_weight_elements": sum(
            int(parameter.numel()) for name, parameter in model.named_parameters() if name.startswith(ROE_PREFIX) and name.endswith(".weight")
        ),
        "roe_bias_elements": sum(
            int(parameter.numel()) for name, parameter in model.named_parameters() if name.startswith(ROE_PREFIX) and name.endswith(".bias")
        ),
        "roe_groups": {name: group_by_name.get(name) for name in roe_names},
        "geo_names": geo_names,
        "geo_names_exact": tuple(geo_names) == EXPECTED_GEO_NAMES,
        "geo_all_base_decay": all(group_by_name.get(name) == "base_decay" for name in geo_names),
        "syncbn_all_base_no_decay": all(group_by_name.get(name) == "base_no_decay" for name in SYNCBN_NAMES),
        "reliability_groups": {name: group_by_name.get(name) for name in reliability_names},
    }


def _plain_config_block(config: Any) -> Dict[str, Any]:
    return _jsonable(dict(config.e1_batch1))


def _common_config_identity(c0: Any, roe: Any) -> Dict[str, Any]:
    fields = (
        "dataset_name",
        "rgb_root_folder",
        "gt_root_folder",
        "x_root_folder",
        "train_source",
        "eval_source",
        "test_source",
        "num_train_imgs",
        "num_eval_imgs",
        "backbone",
        "decoder",
        "decoder_embed_dim",
        "aux_rate",
        "batch_size",
        "num_workers",
        "nepochs",
        "niters_per_epoch",
        "lr",
        "lr_power",
        "warm_up_epoch",
        "weight_decay",
        "optimizer",
        "train_scale_array",
        "channel_order",
        "normalization_identity",
        "norm_mean",
        "norm_std",
        "training_validation_enabled",
        "save_interval",
        "save_epoch_checkpoints",
        "save_latest_checkpoint",
        "official_test",
    )
    mismatches = {}
    for field in fields:
        left = _jsonable(getattr(c0, field, None))
        right = _jsonable(getattr(roe, field, None))
        if left != right:
            mismatches[field] = {"C0": left, "R-OE-lite": right}
    c0_e1 = dict(c0.e1_batch1)
    roe_e1 = dict(roe.e1_batch1)
    excluded = {"candidate", "protocol", "matched_control", "roe_substitute", "feature_adapter", "tf32_training_behavior"}
    for key in sorted(set(c0_e1) | set(roe_e1)):
        if key in excluded:
            continue
        left = _jsonable(c0_e1.get(key))
        right = _jsonable(roe_e1.get(key))
        if left != right:
            mismatches[f"e1_batch1.{key}"] = {"C0": left, "R-OE-lite": right}
    return {"fields": list(fields), "mismatches": mismatches, "exact_equal": not mismatches}


def _first_epoch_permutation(config: Any, *, candidate: str) -> tuple[nn.Module, Dict[str, Any], Dict[str, Any], list[int]]:
    """Mirror train.py: create the shuffled loader, then build the model, then start epoch 1."""

    seed_everything(SEED)
    train_loader, _ = get_train_loader(_LoaderEngineStub(), RGBXDataset, config)
    model, load, post_build_rng = build_model_from_current_rng(config, candidate=candidate)
    permutation = list(iter(train_loader.sampler))
    del train_loader
    return model, load, post_build_rng, permutation


def _permutation_sha256(permutation: Sequence[int]) -> str:
    value = torch.tensor(list(permutation), dtype=torch.int64)
    return tensor_sha256(value)


def _make_observable_probe(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Create one legal current-observable empty batch without using cause metadata."""

    geometry = batch["reliability_valid_mask"].clone().to(dtype=torch.bool)
    raw_depth = torch.zeros_like(batch["raw_depth"])
    depth = torch.zeros_like(batch["depth"])
    empty_value = float(-0.48 / 0.28)
    depth_value = torch.full_like(depth, empty_value)
    depth = torch.where(geometry.expand_as(depth), depth_value, depth)
    target = batch["reliability_target"].clone()
    if target.shape[1] >= 2:
        target[:, 1:2] = torch.where(
            geometry,
            torch.zeros_like(target[:, 1:2]),
            target[:, 1:2],
        )
    return {
        "rgb": batch["rgb"].clone(),
        "depth": depth,
        "label": batch["label"].clone(),
        "raw_rgb": batch["raw_rgb"].clone(),
        "raw_depth": raw_depth,
        "reliability_target": target,
        "reliability_valid_mask": geometry,
        "depth_valid": torch.zeros_like(batch["depth_valid"]),
        "reliability_telemetry_masks": batch["reliability_telemetry_masks"].clone(),
    }


def _make_detector_probe(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Make a four-sample mixed route probe from one train-dev sample."""

    rgb = batch["rgb"][:1].repeat(4, 1, 1, 1)
    modal_x = batch["depth"][:1].repeat(4, 1, 1, 1)
    geometry = batch["reliability_valid_mask"][:1].repeat(4, 1, 1, 1).to(dtype=torch.bool)
    if int(geometry[0].sum().item()) <= 0:
        raise RuntimeError("train probe has no geometry-valid pixel")
    first_valid = geometry[0, 0].nonzero(as_tuple=False)[0]
    y, x = int(first_valid[0].item()), int(first_valid[1].item())
    raw_depth = torch.zeros((4, 1, geometry.shape[-2], geometry.shape[-1]), dtype=torch.float32)
    raw_depth[0, 0, y, x] = 1.0 / 255.0
    raw_depth[3, 0, y, x] = 2.0 / 255.0
    geometry[2].zero_()
    return {
        "rgb": rgb,
        "modal_x": modal_x,
        "raw_depth": raw_depth,
        "geometry": geometry,
        "expected_trigger": torch.tensor([False, True, False, False], dtype=torch.bool),
        "first_valid_pixel": torch.tensor([y, x], dtype=torch.int64),
    }


def _route_with_hook(model: nn.Module, probe: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, list[int], Dict[str, Any]]:
    calls: list[int] = []
    handle = model.roe_substitute.register_forward_pre_hook(
        lambda _module, args: calls.append(int(args[0].shape[0]))
    )
    try:
        with torch.no_grad():
            routed = model._route_roe_modal_x(
                probe["rgb"],
                probe["modal_x"],
                raw_depth=probe["raw_depth"],
                geometry_mask=probe["geometry"],
            )
        telemetry = dict(model.last_roe_route or {})
    finally:
        handle.remove()
    return routed, calls, telemetry


def _check_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=10)
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checks: list[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        entry = {"name": name, "ok": bool(ok), "detail": _jsonable(detail)}
        checks.append(entry)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {entry['detail']}")

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "check_id": CHECK_ID,
        "status": "BLOCKED",
        "official_test_included": False,
        "formal_training_started": False,
        "checks": checks,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("R-OE-lite Gate-B cost and AMP update probe requires one local CUDA GPU")
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

        source_config = import_config(SOURCE_CONFIG_MODULE)
        c0_config = import_config(C0_CONFIG_MODULE)
        roe_config = import_config(ROE_CONFIG_MODULE)

        source_checkpoint_hash = file_sha256(SOURCE_CHECKPOINT)
        train_split_hash = file_sha256(TRAIN_SPLIT)
        val_split_hash = file_sha256(VAL_SPLIT)
        check("source.checkpoint_path", SOURCE_CHECKPOINT == Path(roe_config.e1_batch1["source_checkpoint"]).resolve(), str(SOURCE_CHECKPOINT))
        check("source.checkpoint_sha256", source_checkpoint_hash == SOURCE_CHECKPOINT_SHA256, source_checkpoint_hash)
        check("source.train_split_sha256", train_split_hash == TRAIN_SPLIT_SHA256, train_split_hash)
        check("source.val_split_sha256", val_split_hash == VAL_SPLIT_SHA256, val_split_hash)
        check("source.official_test_sealed", roe_config.e1_batch1["official_test"] == "sealed_unread", roe_config.e1_batch1["official_test"])
        check("source.config_module", SOURCE_CONFIG_MODULE.endswith("DFormerv2_S_MMFR_A2_DepthCorrupt_v3"), SOURCE_CONFIG_MODULE)
        check("source.roe_config_module", ROE_CONFIG_MODULE.endswith("DFormerv2_S_MMFR_E1_Batch1B_R_OE"), ROE_CONFIG_MODULE)

        expected_roe_block = {
            "architecture": "observable-empty-geometry-substitute",
            "channels": [122, 398, 256, 256, 398, 122, 1],
            "pool": "avgpool2d-kernel2-stride2-ceil-mode-true",
            "activation": "GELU",
            "resize": "bilinear-align-corners-false",
            "normalization": "none",
            "skip_connections": False,
            "output": "straight-through-clamp-0-255",
            "head_bias": 127.5,
            "uses_depth": False,
            "uses_reliability": False,
            "uses_condition": False,
            "uses_severity": False,
            "uses_oracle": False,
            "expected_trainable_parameters": EXPECTED_ROE_TRAINABLE_ELEMENTS,
        }
        check("config.roe_substitute_contract", dict(roe_config.e1_batch1["roe_substitute"]) == expected_roe_block, roe_config.e1_batch1["roe_substitute"])
        check("config.protocol_identity", roe_config.e1_batch1["protocol"] == "MMFR-E1-Batch1B-R-OE-lite-v1", roe_config.e1_batch1["protocol"])
        check("config.candidate_identity", roe_config.e1_batch1["candidate"] == "R-OE-lite", roe_config.e1_batch1["candidate"])
        check("config.matched_control_identity", roe_config.e1_batch1["matched_control"] == "MMFR-E1-Batch1A-C0", roe_config.e1_batch1["matched_control"])
        check("config.common_contract_exact", _common_config_identity(c0_config, roe_config)["exact_equal"], _common_config_identity(c0_config, roe_config))

        schedule = {
            "epochs": int(roe_config.nepochs),
            "attempts_per_epoch": int(roe_config.niters_per_epoch),
            "successful_updates": int(roe_config.e1_batch1["required_successful_updates"]),
            "batch_size": int(roe_config.batch_size),
            "workers": int(roe_config.num_workers),
            "base_lr": float(roe_config.e1_batch1["base_lr"]),
            "new_lr": float(roe_config.e1_batch1["new_module_lr"]),
            "warmup": int(roe_config.e1_batch1["warmup_successful_updates"]),
            "poly_power": float(roe_config.e1_batch1["poly_power"]),
            "p_clean": float(roe_config.mmfr_a2["corruption"]["p_clean"]),
            "max_specs": int(roe_config.mmfr_a2["corruption"]["max_specs"]),
            "corruption_kinds": list(roe_config.mmfr_a2["corruption"]["kinds"]),
            "tf32_training_behavior": str(roe_config.e1_batch1["tf32_training_behavior"]),
        }
        expected_schedule = {
            "epochs": 20,
            "attempts_per_epoch": 128,
            "successful_updates": 2560,
            "batch_size": 10,
            "workers": 8,
            "base_lr": 1e-5,
            "new_lr": 3e-5,
            "warmup": 128,
            "poly_power": 0.9,
            "p_clean": 0.25,
            "max_specs": 2,
            "corruption_kinds": ["entire_missing", "spatial_dropout", "gaussian_noise", "blur", "quantization", "misalignment"],
            "tf32_training_behavior": "batch-1a-actual-preserved",
        }
        check("config.batch1b_common_schedule", schedule == expected_schedule, schedule)
        train_source = (REPO_ROOT / "utils" / "train.py").read_text(encoding="utf-8")
        check("config.tf32_actual_training_behavior_preserved", 'torch.set_float32_matmul_precision("high")' in train_source, "set_float32_matmul_precision high present")

        source_model = None
        seed_everything(SEED)
        source_model, source_load, source_post_rng = build_model_from_current_rng(source_config, candidate="source-A2")
        check("checkpoint.schema", source_load["schema_version"] == "dformer-training-checkpoint-v2", source_load)
        check("checkpoint.model_key_count", source_load["model_key_count"] == SOURCE_MODEL_KEY_COUNT, source_load["model_key_count"])
        check("checkpoint.source_strict", not source_load["missing_keys"] and not source_load["unexpected_keys"], source_load)
        check("checkpoint.epoch_identity", (source_load["completed_epoch"], source_load["next_epoch"], source_load["global_optimizer_step"]) == (420, 421, 53735), source_load)

        c0_model, c0_load, c0_post_rng, c0_permutation = _first_epoch_permutation(c0_config, candidate="C0")
        roe_model, roe_load, roe_post_rng, roe_permutation = _first_epoch_permutation(roe_config, candidate="R-OE-lite")
        check("checkpoint.c0_strict", not c0_load["missing_keys"] and not c0_load["unexpected_keys"], c0_load)
        check("checkpoint.roe_only_substitute_missing", roe_load["missing_keys"] == list(EXPECTED_ROE_PARAMETER_NAMES), roe_load["missing_keys"])
        check("rng.post_build_identity", _rng_equal(c0_post_rng, roe_post_rng), {"C0": _rng_summary(c0_post_rng), "R-OE-lite": _rng_summary(roe_post_rng)})
        check("sampler.first_epoch_permutation_identity", c0_permutation == roe_permutation, {"count": len(c0_permutation), "C0_sha256": _permutation_sha256(c0_permutation), "R-OE-lite_sha256": _permutation_sha256(roe_permutation), "first_16": c0_permutation[:16]})

        source_state = source_model.state_dict()
        c0_state = c0_model.state_dict()
        roe_shared_state = {name: value for name, value in roe_model.state_dict().items() if not name.startswith(ROE_PREFIX)}
        source_c0_equal = not set(source_state) ^ set(c0_state) and all(torch.equal(source_state[name], c0_state[name]) for name in source_state)
        source_roe_equal = set(source_state) == set(roe_shared_state) and all(torch.equal(source_state[name], roe_shared_state[name]) for name in source_state)
        check("weights.c0_equal_source", source_c0_equal, {"source_keys": len(source_state), "c0_keys": len(c0_state)})
        check("weights.roe_shared_equal_source", source_roe_equal, {"source_keys": len(source_state), "roe_shared_keys": len(roe_shared_state)})

        counts = {"source": model_counts(source_model), "C0": model_counts(c0_model), "R-OE-lite": model_counts(roe_model)}
        roe_added = counts["R-OE-lite"]["trainable_parameters"] - counts["C0"]["trainable_parameters"]
        check("roe.parameter_count", roe_added == EXPECTED_ROE_TRAINABLE_ELEMENTS, {"C0": counts["C0"], "R-OE-lite": counts["R-OE-lite"], "added": roe_added})
        check("roe.parameter_count_matches_config", roe_added == int(roe_config.e1_batch1["roe_substitute"]["expected_trainable_parameters"]), roe_added)

        c0_optimizer, c0_optimizer_summary = optimizer_and_summary(c0_model, c0_config)
        roe_optimizer, roe_optimizer_summary = optimizer_and_summary(roe_model, roe_config)
        expected_group_specs = {
            "base_decay": {"lr": 1e-5, "weight_decay": 0.01},
            "base_no_decay": {"lr": 1e-5, "weight_decay": 0.0},
            "new_decay": {"lr": 3e-5, "weight_decay": 0.01},
            "new_no_decay": {"lr": 3e-5, "weight_decay": 0.0},
        }
        expected_reliability_groups = {
            name: ("base_no_decay" if name.endswith(".bias") else "base_decay")
            for name, _ in roe_model.named_parameters()
            if name.startswith("reliability_estimator.")
        }
        for label, summary in (("c0", c0_optimizer_summary), ("roe", roe_optimizer_summary)):
            actual_specs = {group["name"]: {"lr": group["lr"], "weight_decay": group["weight_decay"]} for group in summary["groups"]}
            check(f"optimizer.{label}.membership_all_one", summary["membership_all_one"], summary["invalid_membership"])
            check(f"optimizer.{label}.group_order", tuple(summary["group_order"]) == EXPECTED_GROUP_ORDER, summary["group_order"])
            check(f"optimizer.{label}.group_specs", actual_specs == expected_group_specs, actual_specs)
            check(f"optimizer.{label}.geo_29_base_decay", summary["geo_names_exact"] and summary["geo_all_base_decay"], summary["geo_names"])
            check(f"optimizer.{label}.syncbn_14_base_no_decay", summary["syncbn_all_base_no_decay"], "14 audited SyncBN tensors")
            check(f"optimizer.{label}.reliability_groups", summary["reliability_groups"] == expected_reliability_groups, summary["reliability_groups"])
        check("optimizer.roe_names_exact", roe_optimizer_summary["roe_names_exact"], roe_optimizer_summary["roe_names"])
        check("optimizer.roe_weight_bias_elements", (roe_optimizer_summary["roe_weight_elements"], roe_optimizer_summary["roe_bias_elements"]) == (EXPECTED_ROE_WEIGHT_ELEMENTS, EXPECTED_ROE_BIAS_ELEMENTS), roe_optimizer_summary)
        expected_roe_groups = {name: ("new_no_decay" if name.endswith(".bias") else "new_decay") for name in EXPECTED_ROE_PARAMETER_NAMES}
        check("optimizer.roe_exact_groups", roe_optimizer_summary["roe_groups"] == expected_roe_groups, roe_optimizer_summary["roe_groups"])

        batch_cpu, batch_identity = build_real_batch(c0_config)
        check("batch.legal_train_dev_probe", not bool(batch_identity["official_test_included"]), batch_identity)
        if int(batch_cpu["reliability_valid_mask"].sum().item()) <= 0:
            raise RuntimeError("legal train-dev probe has no geometry-valid pixels")
        detector_probe_cpu = _make_detector_probe(batch_cpu)
        trigger_probe_cpu = _make_observable_probe(batch_cpu)
        source_model.cpu()
        del source_model
        torch.cuda.empty_cache()
        roe_model.to(device)
        c0_model.to(device)
        detector_probe = to_device(detector_probe_cpu, device)
        trigger_probe = to_device(trigger_probe_cpu, device)

        mixed_routed, mixed_calls, mixed_telemetry = _route_with_hook(roe_model, detector_probe)
        expected_trigger = detector_probe_cpu["expected_trigger"].tolist()
        actual_trigger = mixed_telemetry["trigger"].cpu().tolist()
        check("detector.observable_empty_trigger", actual_trigger == expected_trigger, {"expected": expected_trigger, "actual": actual_trigger, "telemetry": mixed_telemetry})
        check("detector.mixed_batch_per_sample_route", mixed_calls == [1], mixed_calls)
        nontrigger_indices = [0, 2, 3]
        bypass_exact = all(torch.equal(mixed_routed[index], detector_probe["modal_x"][index]) for index in nontrigger_indices)
        check("bypass.mixed_nontrigger_bitwise_exact", bypass_exact, {"indices": nontrigger_indices})

        no_geometry_probe = {
            "rgb": detector_probe["rgb"][:1],
            "modal_x": detector_probe["modal_x"][:1],
            "raw_depth": torch.zeros_like(detector_probe["raw_depth"][:1]),
            "geometry": torch.zeros_like(detector_probe["geometry"][:1]),
        }
        no_geometry_routed, no_geometry_calls, no_geometry_telemetry = _route_with_hook(roe_model, no_geometry_probe)
        check("detector.no_geometry_bypass", no_geometry_telemetry["trigger"].cpu().tolist() == [False] and no_geometry_calls == [] and torch.equal(no_geometry_routed, no_geometry_probe["modal_x"]), {"telemetry": no_geometry_telemetry, "hook_batch_sizes": no_geometry_calls})

        nonempty_probe = {
            "rgb": detector_probe["rgb"][:1],
            "modal_x": detector_probe["modal_x"][:1],
            "raw_depth": detector_probe["raw_depth"][:1],
            "geometry": detector_probe["geometry"][:1],
        }
        nonempty_routed, nonempty_calls, nonempty_telemetry = _route_with_hook(roe_model, nonempty_probe)
        check("detector.nonempty_bypass", nonempty_telemetry["trigger"].cpu().tolist() == [False] and nonempty_calls == [], {"telemetry": nonempty_telemetry, "hook_batch_sizes": nonempty_calls})
        check("bypass.nonempty_bitwise_exact", torch.equal(nonempty_routed, nonempty_probe["modal_x"]), {"exact_equal": torch.equal(nonempty_routed, nonempty_probe["modal_x"])})

        with torch.no_grad():
            substitute_raw = roe_model.roe_substitute(trigger_probe["rgb"], trigger_probe["reliability_valid_mask"])
            routed_trigger = roe_model._route_roe_modal_x(
                trigger_probe["rgb"],
                trigger_probe["depth"],
                raw_depth=trigger_probe["raw_depth"],
                geometry_mask=trigger_probe["reliability_valid_mask"],
            )
        substitute_pad = ~trigger_probe["reliability_valid_mask"].expand_as(substitute_raw)
        substitute_pad_exact_zero = not bool(torch.count_nonzero(substitute_raw.masked_select(substitute_pad)).item())
        substitute_range = (float(substitute_raw.min().item()), float(substitute_raw.max().item()))
        expected_normalized = (substitute_raw.repeat(1, 3, 1, 1) / 255.0 - 0.48) / 0.28
        repeat_and_normalize_exact = torch.equal(routed_trigger, expected_normalized)
        check("roe.output.finite", _check_finite(substitute_raw), {"finite": _check_finite(substitute_raw)})
        check("roe.output.range", substitute_range[0] >= 0.0 and substitute_range[1] <= 255.0, substitute_range)
        check("roe.output.padding_exact_zero", substitute_pad_exact_zero, {"nonzero_padding_elements": int(torch.count_nonzero(substitute_raw.masked_select(substitute_pad)).item())})
        check("roe.output.repeat_depth_normalization", repeat_and_normalize_exact, {"exact_equal": repeat_and_normalize_exact, "normalized_shape": list(routed_trigger.shape)})
        check("roe.trigger_substitute_changes_route", not torch.equal(routed_trigger, trigger_probe["depth"]), {"exact_equal": torch.equal(routed_trigger, trigger_probe["depth"])})

        # Compare C0 and R-OE on an explicitly non-triggering batch with paired decoder RNG.
        seed_everything(SEED)
        with torch.no_grad():
            c0_nontrigger_logits = c0_model(
                nonempty_probe["rgb"],
                nonempty_probe["modal_x"],
                raw_depth=nonempty_probe["raw_depth"],
                reliability_valid_mask=nonempty_probe["geometry"],
            )
        seed_everything(SEED)
        with torch.no_grad():
            roe_nontrigger_logits = roe_model(
                nonempty_probe["rgb"],
                nonempty_probe["modal_x"],
                raw_depth=nonempty_probe["raw_depth"],
                reliability_valid_mask=nonempty_probe["geometry"],
            )
        check("bypass.segmentation_logits_exact_vs_c0", torch.equal(c0_nontrigger_logits, roe_nontrigger_logits), comparison(c0_nontrigger_logits, roe_nontrigger_logits))

        # Reliability receives the original raw/target tensors even when segmentation routes.
        observed_reliability: Dict[str, torch.Tensor] = {}
        original_reliability_loss = roe_model.reliability_auxiliary_loss

        def capture_reliability(raw_rgb, raw_depth, reliability_target, reliability_valid_mask=None, depth_valid=None, reliability_telemetry_masks=None):
            observed_reliability["raw_rgb"] = raw_rgb
            observed_reliability["raw_depth"] = raw_depth
            observed_reliability["reliability_target"] = reliability_target
            return original_reliability_loss(raw_rgb, raw_depth, reliability_target, reliability_valid_mask, depth_valid, reliability_telemetry_masks)

        roe_model.reliability_auxiliary_loss = capture_reliability
        try:
            seed_everything(SEED)
            with torch.no_grad():
                trigger_total_loss = roe_model(trigger_probe["rgb"], trigger_probe["depth"], trigger_probe["label"], **auxiliary_kwargs(trigger_probe))
        finally:
            roe_model.reliability_auxiliary_loss = original_reliability_loss
        reliability_inputs_exact = (
            torch.equal(observed_reliability["raw_rgb"], trigger_probe["raw_rgb"])
            and torch.equal(observed_reliability["raw_depth"], trigger_probe["raw_depth"])
            and torch.equal(observed_reliability["reliability_target"], trigger_probe["reliability_target"])
        )
        check("reliability.original_raw_target_unchanged", reliability_inputs_exact, {"observed": {key: tensor_sha256(value) for key, value in observed_reliability.items()}, "expected": {key: tensor_sha256(trigger_probe[key]) for key in ("raw_rgb", "raw_depth", "reliability_target")}})
        seed_everything(SEED)
        with torch.no_grad():
            routed_for_loss = roe_model._route_roe_modal_x(trigger_probe["rgb"], trigger_probe["depth"], raw_depth=trigger_probe["raw_depth"], geometry_mask=trigger_probe["reliability_valid_mask"])
            segmentation_logits = roe_model.encode_decode(trigger_probe["rgb"], routed_for_loss)
            segmentation_loss = safe_masked_mean(roe_model.criterion(segmentation_logits, trigger_probe["label"].long()), trigger_probe["label"].long() != int(roe_model.cfg.background))
            expected_reliability = original_reliability_loss(**auxiliary_kwargs(trigger_probe))
            expected_total = segmentation_loss + roe_model.reliability_weight * expected_reliability
        check("reliability.substitute_not_in_auxiliary_loss", torch.equal(trigger_total_loss, expected_total), comparison(trigger_total_loss, expected_total))
        check("probe.loss_finite", _check_finite(trigger_total_loss), float(trigger_total_loss.item()))

        # One C0 and one R-OE AMP update only, used for gradient/update and memory qualification.
        roe_model.cpu()
        c0_model.cpu()
        del c0_optimizer
        torch.cuda.empty_cache()
        c0_model.to(device)
        trigger_probe_for_step = trigger_probe
        c0_optimizer, _ = optimizer_and_summary(c0_model, c0_config)
        c0_scaler = make_scaler(c0_config)
        torch.cuda.reset_peak_memory_stats(device)
        c0_step = training_step(c0_model, c0_optimizer, c0_scaler, trigger_probe_for_step, seed=SEED + 1)
        c0_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

        c0_model.cpu()
        del c0_optimizer, c0_scaler
        torch.cuda.empty_cache()
        roe_model.cpu()
        torch.cuda.empty_cache()
        roe_model.to(device)
        roe_optimizer, _ = optimizer_and_summary(roe_model, roe_config)
        roe_scaler = make_scaler(roe_config)
        roe_new_names = sorted(name for name, _ in roe_model.named_parameters() if name.startswith(ROE_PREFIX))
        representative_names = [
            "roe_substitute.e1.weight",
            "roe_substitute.e1.bias",
            "roe_substitute.head.weight",
            "roe_substitute.head.bias",
        ]
        roe_before = {name: parameter.detach().cpu().clone() for name, parameter in roe_model.named_parameters() if name in representative_names}
        torch.cuda.reset_peak_memory_stats(device)
        roe_step = training_step(roe_model, roe_optimizer, roe_scaler, trigger_probe_for_step, seed=SEED + 1)
        roe_gradients = gradient_status(roe_model, representative_names)
        roe_changes = change_status(roe_model, roe_before)
        roe_all_new_gradients = gradient_status(roe_model, roe_new_names)
        roe_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        check("gradient.loss_finite", roe_step["loss_finite"], roe_step)
        check("gradient.optimizer_step_applied", roe_step["optimizer_step_applied"], roe_step)
        check("gradient.roe_representative_finite_nonzero", roe_gradients["finite_all_present"] and roe_gradients["nonzero_count"] == len(representative_names), roe_gradients)
        check("gradient.roe_representative_updated", roe_changes["all_changed"], roe_changes)
        check("gradient.roe_all_new_gradients_finite", roe_all_new_gradients["finite_all_present"], {"missing": roe_all_new_gradients["missing"], "non_finite": roe_all_new_gradients["non_finite"], "nonzero_count": roe_all_new_gradients["nonzero_count"], "count": roe_all_new_gradients["count"]})

        memory_delta = {
            "allocated_bytes": roe_memory["peak_allocated_bytes"] - c0_memory["peak_allocated_bytes"],
            "reserved_bytes": roe_memory["peak_reserved_bytes"] - c0_memory["peak_reserved_bytes"],
            "allocated_mib": (roe_memory["peak_allocated_bytes"] - c0_memory["peak_allocated_bytes"]) / (1024 ** 2),
            "reserved_mib": (roe_memory["peak_reserved_bytes"] - c0_memory["peak_reserved_bytes"]) / (1024 ** 2),
        }
        check("cost.training_peak_memory_recorded", roe_memory["peak_allocated_bytes"] > 0 and c0_memory["peak_allocated_bytes"] > 0, {"C0": c0_memory, "R-OE-lite": roe_memory})

        roe_model.eval()
        nontrigger_batch1 = {key: value[:1] for key, value in nonempty_probe.items()}
        trigger_batch1 = {key: value[:1] for key, value in trigger_probe.items()}

        def nontrigger_inference():
            with torch.no_grad():
                return roe_model(
                    nontrigger_batch1["rgb"],
                    nontrigger_batch1["modal_x"],
                    raw_depth=nontrigger_batch1["raw_depth"],
                    reliability_valid_mask=nontrigger_batch1["geometry"],
                )

        def trigger_inference():
            with torch.no_grad():
                return roe_model(
                    trigger_batch1["rgb"],
                    trigger_batch1["depth"],
                    raw_depth=trigger_batch1["raw_depth"],
                    reliability_valid_mask=trigger_batch1["reliability_valid_mask"],
                )

        latency = {
            "non_trigger_batch1": timed_cuda(nontrigger_inference, warmup=args.latency_warmup, repeats=args.latency_repeats),
            "trigger_batch1": timed_cuda(trigger_inference, warmup=args.latency_warmup, repeats=args.latency_repeats),
        }
        check("cost.latency_recorded", latency["non_trigger_batch1"]["median_ms"] > 0.0 and latency["trigger_batch1"]["median_ms"] > 0.0, latency)

        common_identity = _common_config_identity(c0_config, roe_config)
        existing_c0_ok = C0_FINAL_CHECKPOINT.is_file() and file_sha256(C0_FINAL_CHECKPOINT) == C0_FINAL_CHECKPOINT_SHA256
        fairness_ok = common_identity["exact_equal"] and _rng_equal(c0_post_rng, roe_post_rng) and c0_permutation == roe_permutation and existing_c0_ok
        check("c0.reuse_fairness", fairness_ok, {"common_contract": common_identity, "rng_equal": _rng_equal(c0_post_rng, roe_post_rng), "sampler_equal": c0_permutation == roe_permutation, "existing_c0_checkpoint": existing_c0_ok, "checkpoint": str(C0_FINAL_CHECKPOINT)})

        report.update(
            {
                "status": "PASS" if all(item["ok"] for item in checks) else "BLOCKED",
                "gate_b": "PASS" if all(item["ok"] for item in checks) else "BLOCKED",
                "source_identity": {
                    "source_config_module": SOURCE_CONFIG_MODULE,
                    "c0_config_module": C0_CONFIG_MODULE,
                    "roe_config_module": ROE_CONFIG_MODULE,
                    "checkpoint": {"path": str(SOURCE_CHECKPOINT), "size_bytes": SOURCE_CHECKPOINT.stat().st_size, "sha256": source_checkpoint_hash, "load": source_load},
                    "train_split": {"path": str(TRAIN_SPLIT), "samples": 1277, "sha256": train_split_hash},
                    "val_split": {"path": str(VAL_SPLIT), "samples": 318, "sha256": val_split_hash},
                    "official_test": "sealed_unread",
                    "implementation_hashes": {relative: file_sha256(REPO_ROOT / relative) for relative in ("models/builder.py", "models/roe_substitute.py", "utils/init_func.py", "utils/train.py", "local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1B_R_OE.py", "tools/mmfr/e1_batch1b_gateb.py")},
                },
                "protocol": {"config": _plain_config_block(roe_config), "schedule": schedule, "common_contract": common_identity},
                "c0_reuse": {"existing_checkpoint": str(C0_FINAL_CHECKPOINT), "existing_checkpoint_sha256": file_sha256(C0_FINAL_CHECKPOINT) if C0_FINAL_CHECKPOINT.is_file() else None, "existing_checkpoint_expected_sha256": C0_FINAL_CHECKPOINT_SHA256, "existing_c0_reusable": fairness_ok},
                "rng_fairness": {"C0_post_build": _rng_summary(c0_post_rng), "R-OE-lite_post_build": _rng_summary(roe_post_rng), "equal": _rng_equal(c0_post_rng, roe_post_rng)},
                "sampler_fairness": {"count": len(c0_permutation), "C0_sha256": _permutation_sha256(c0_permutation), "R-OE-lite_sha256": _permutation_sha256(roe_permutation), "equal": c0_permutation == roe_permutation, "first_16": c0_permutation[:16]},
                "model_counts": counts,
                "new_trainable_parameters": roe_added,
                "optimizer": {"C0": c0_optimizer_summary, "R-OE-lite": roe_optimizer_summary},
                "detector_and_bypass": {"mixed_telemetry": mixed_telemetry, "mixed_hook_batch_sizes": mixed_calls, "no_geometry_telemetry": no_geometry_telemetry, "nonempty_telemetry": nonempty_telemetry, "probe_pixel": detector_probe_cpu["first_valid_pixel"]},
                "output_contract": {"substitute_range": substitute_range, "padding_exact_zero": substitute_pad_exact_zero, "repeat_and_normalize_exact": repeat_and_normalize_exact},
                "reliability_path": {"input_identity": reliability_inputs_exact, "total_loss_identity": True},
                "gradients_and_updates": {"C0": {"step": c0_step, "peak_memory": c0_memory}, "R-OE-lite": {"step": roe_step, "representative_gradients": roe_gradients, "representative_updates": roe_changes, "all_new_gradients": roe_all_new_gradients, "peak_memory": roe_memory}},
                "cost_probe": {"device": torch.cuda.get_device_name(device), "batch_size": 1, "training_peak_memory": {"C0": c0_memory, "R-OE-lite": roe_memory, "delta": memory_delta}, "latency": latency},
                "formal_training_started": False,
            }
        )
    except BaseException as error:  # noqa: BLE001 - preserve a failure artifact
        report["status"] = "BLOCKED"
        report["gate_b"] = "BLOCKED"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], end="")

    report["duration_seconds"] = float(time.perf_counter() - started)
    report["failed_checks"] = [item for item in checks if not item["ok"]]
    report["report_path"] = str(output)
    output.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_sha256 = file_sha256(output)
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": report_sha256}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
