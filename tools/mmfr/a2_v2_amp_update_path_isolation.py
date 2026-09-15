#!/usr/bin/env python3
"""Gate ``MMFR-A2-v2-amp-update-path-isolation``: paired AMP optimizer-update trajectories.

为什么需要这个门禁（大白话）
----------------------------
`tools/mmfr/gradient_path_isolation.py` 已经在 FP32、batch size 1、单进程、单样本条件下证明
“辅助可靠性损失不改变共享参数的**梯度**”。但正式训练使用的是 AMP + GradScaler，而历史 v1
preflight 的前 6 次尝试中有 5 次被 GradScaler 跳过 optimizer update。也就是说：梯度层面的隔离
不等于**实际 AMP 更新轨迹**的隔离——scale 的变化会改变哪些 step 被应用，一旦 skip 模式不同，
共享参数就会走出不同轨迹。

本工具因此构造两条严格配对的轨迹：

* **A**：``lambda_rel = 0.0``（不加入可靠性辅助项）
* **B**：``lambda_rel = 0.1``（正式训练权重）

两条轨迹在同一初始状态、同一 batch 序列、同一 corruption、同一 RNG 下逐 step 运行真实
v2 Depth-corruption 模型（官方 pretrained + Ham decoder + reliability head）的 AMP 前向/反向与
``GradScaler`` 更新。每个 attempted step 记录：``GradScaler.get_scale()`` 前后值、是否真正执行了
optimizer step、共享梯度是否 finite、以及（在成功更新后）共享参数的字节级 SHA-256 与共享
optimizer state 的 SHA-256。允许 reliability head 自身参数不同。

判定标准（全部必须成立，否则退出码非零）：

1. step/skip 序列完全相同；
2. scale 轨迹完全相同；
3. 每次成功更新之后，所有非 ``reliability_estimator.*`` 参数在容差内一致（默认按字节哈希逐位比较，
   并在保留快照时给出最大绝对差）；
4. reliability head 的参数允许不同（不作为判定依据）；
5. 至少获得 ``--min-successful-updates``（默认 10）个真实成功更新；若 ``--max-attempts``
   （默认 50）内达不到，门禁失败；
6. batch 身份必须逐 step 匹配（否则门禁失败，避免用不同输入比较轨迹）；
7. 抽样中至少出现若干个 corrupted step（默认至少 1 个），否则判定为 vacuous 失败。

运行
----
::

    python tools/mmfr/a2_v2_amp_update_path_isolation.py

常用参数：``--batch-size``（默认 10，与冻结训练一致）、``--max-attempts``、``--min-successful-updates``、
``--lambda-rel-on``、``--output``。若 batch size 10 在本机显存下 OOM，工具会显式记录
``capacity_failure`` 并以非零退出码结束；此时可另用较小的 ``--batch-size`` 做补充证据，但**不得**
把补充证据写成冻结 batch size 10 已通过。

证据
----
``outputs/mmfr-a2-v2-amp-update-path-isolation/a2-v2-amp-update-path-isolation.json``：记录请求与实际的
batch size、每一步两条轨迹的完整 telemetry、逐 step 比较结果、最终结论与退出码。工具不写 checkpoint、
不读 official test、不启动训练循环之外的任何行为。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from tools.mmfr.gradient_path_isolation import (
    REPO_ROOT as ISOLATION_REPO_ROOT,
    build_corrupted_batch,
    build_model,
    load_config,
    load_training_sample,
    move_batch_to_device,
    resolve_device,
    resolve_pretrained_path,
    restore_model_state,
    snapshot_model_state,
    verify_pretrained_file,
)
from tools.museg_protocol import file_sha256, write_json
from utils.init_func import group_weight
from utils.training_checkpoint import optimizer_step_was_applied

SCHEMA_VERSION = "museg-mmfr-a2-amp-update-path-isolation-v1"
CHECK_ID = "MMFR-A2-v2-amp-update-path-isolation"
RELIABILITY_HEAD_PARAMETER_PREFIX = "reliability_estimator."
DEFAULT_OUTPUT_PATH = (
    ISOLATION_REPO_ROOT / "outputs" / "mmfr-a2-v2-amp-update-path-isolation" / "a2-v2-amp-update-path-isolation.json"
)
#: Frozen identity of the pair under test.
EXPECTED_PROTOCOL = "MMFR-A2-train-integration-v2"
EXPECTED_CORRUPTION_BASIS = "MMFR-A1-corruption-basis-v2"
EXPECTED_PRETRAINED_SHA256 = "19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6"
#: Deterministic per-step seed policy: step ``k`` uses ``STEP_SEED_BASE + k`` for python/numpy/torch RNG
#: right before the batch is built and again right before the forward, in **both** trajectories.
STEP_SEED_BASE = 20260915
PREPROCESS_SEED_BASE = 20260916
#: Frozen training position used for the corruption curriculum words.
STEP_EPOCH = 1
#: Snapshot guard: keep per-update shared-parameter clones only while they stay under this budget.
SNAPSHOT_MEMORY_LIMIT_BYTES = 1_600_000_000
#: Byte-wise comparison tolerance used only when snapshots allow a numeric difference report.
ATOL = 1e-6
RTOL = 1e-6


# ---------------------------------------------------------------------------
# Hashing and small helpers
# ---------------------------------------------------------------------------
def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 over the native dtype bytes of a tensor (bitwise identity evidence)."""
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bool:
        value = value.to(torch.uint8)
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.to(torch.float32)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def shared_parameter_hashes(model: nn.Module, names: Sequence[str]) -> Dict[str, str]:
    return {name: tensor_sha256(dict(model.named_parameters())[name]) for name in names}


def combined_hash(mapping: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        digest.update(name.encode("utf-8"))
        digest.update(mapping[name].encode("utf-8"))
    return digest.hexdigest()


def finite_or_none(value: float) -> float | None:
    """JSON-safe float: non-finite values are recorded as ``null`` next to an explicit finite flag.

    AMP skips commonly come with non-finite gradients, so ``inf``/``nan`` are expected evidence rather
    than a serialization bug; the report keeps them distinguishable via ``*_finite`` flags.
    """
    return float(value) if math.isfinite(float(value)) else None


def shared_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, _ in model.named_parameters() if not name.startswith(RELIABILITY_HEAD_PARAMETER_PREFIX)]


def reliability_head_names(model: nn.Module) -> List[str]:
    return [name for name, _ in model.named_parameters() if name.startswith(RELIABILITY_HEAD_PARAMETER_PREFIX)]


def optimizer_state_hashes(
    optimizer: torch.optim.Optimizer,
    param_names: Mapping[int, str],
    shared_names: Sequence[str],
) -> Dict[str, str]:
    """Hash the AdamW state tensors of the **shared** parameters, keyed by parameter name.

    The reliability head carries its own optimizer state; that state is explicitly allowed to differ
    between the two trajectories, so it is excluded here by name.
    """
    shared = set(shared_names)
    records: Dict[str, str] = {}
    for parameter, state in optimizer.state.items():
        name = param_names.get(id(parameter))
        if name is None or name not in shared:
            continue
        digest = hashlib.sha256()
        for key in sorted(state):
            value = state[key]
            digest.update(key.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                digest.update(tensor_sha256(value).encode("utf-8"))
            else:
                digest.update(str(value).encode("utf-8"))
        records[name] = digest.hexdigest()
    return records


def managed_parameter_ids(optimizer: torch.optim.Optimizer) -> set:
    """Ids of the parameters the optimizer actually owns.

    ``utils/init_func.py:group_weight`` does not collect every tensor in this model (bare
    ``nn.Parameter`` attributes such as ``Geo.weight``, the ``patch_embed.proj.*`` projections and the
    ``downsample.norm.*`` tensors are skipped). Those tensors never enter ``GradScaler``'s unscale/inf
    check and are never updated, so their gradient finiteness must not be reported as if it belonged to
    the optimizer trajectory.
    """
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def non_finite_parameter_names(
    model: nn.Module,
    names: Sequence[str],
    managed_ids: set | None = None,
) -> List[str]:
    parameters = dict(model.named_parameters())
    selected = []
    for name in names:
        if managed_ids is not None and id(parameters[name]) not in managed_ids:
            continue
        gradient = parameters[name].grad
        if gradient is not None and not torch.isfinite(gradient).all().item():
            selected.append(name)
    return selected


def shared_gradient_hashes(model: nn.Module, names: Sequence[str]) -> Dict[str, Any]:
    parameters = dict(model.named_parameters())
    hashes: Dict[str, str] = {}
    missing: List[str] = []
    maximum_abs = 0.0
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            missing.append(name)
            continue
        hashes[name] = tensor_sha256(gradient)
        maximum_abs = max(maximum_abs, float(gradient.abs().max().item()))
    return {
        "gradient_hashes": hashes,
        "combined_hash": combined_hash(hashes),
        "missing_gradient_names": missing,
        "max_abs_gradient": maximum_abs,
    }


def build_batch_identity(batch: Mapping[str, torch.Tensor]) -> Dict[str, str]:
    keys = ("rgb", "depth", "raw_rgb", "raw_depth", "reliability_target", "valid_mask", "depth_valid_pre", "depth_valid_post")
    return {key: tensor_sha256(batch[key]) for key in keys if key in batch}


def combined_batch_hash(identity: Mapping[str, str]) -> str:
    return combined_hash(identity)


def seed_everything(seed: int) -> None:
    """The per-step RNG policy: python, numpy and torch (CPU+CUDA) are all reset identically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v2")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--min-successful-updates", type=int, default=10)
    parser.add_argument("--min-corrupt-steps", type=int, default=1)
    parser.add_argument("--lambda-rel-off", type=float, default=0.0)
    parser.add_argument("--lambda-rel-on", type=float, default=0.1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--keep-snapshots", default="auto", choices=("auto", "yes", "no"))
    return parser.parse_args(argv)


def _base_report(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "check_id": CHECK_ID,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "requested": {
            "config": args.config,
            "batch_size": int(args.batch_size),
            "max_attempts": int(args.max_attempts),
            "min_successful_updates": int(args.min_successful_updates),
            "min_corrupt_steps": int(args.min_corrupt_steps),
            "lambda_rel_off": float(args.lambda_rel_off),
            "lambda_rel_on": float(args.lambda_rel_on),
            "sample_offset": int(args.sample_offset),
            "device": str(device),
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "failures": [],
        "warnings": [],
        "official_test_included": False,
    }


def configure_runtime(device: torch.device) -> Dict[str, Any]:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.init()
    return {
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_enabled": True,
        "use_deterministic_algorithms": "warn_only",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
    }


def build_optimizer(model: nn.Module, config: Any) -> torch.optim.Optimizer:
    """Mirror ``utils/train.py``: ``group_weight`` then ``torch.optim.AdamW``."""
    params_list = group_weight([], model, nn.BatchNorm2d, float(config.lr))
    if str(config.optimizer) != "AdamW":
        raise RuntimeError(f"this gate is frozen to AdamW, got {config.optimizer!r}")
    return torch.optim.AdamW(
        params_list,
        lr=float(config.lr),
        betas=(0.9, 0.999),
        weight_decay=float(config.weight_decay),
    )


def plan_step(config: Any, step: int, sample_offset: int) -> Dict[str, Any]:
    total = int(config.num_train_imgs)
    return {
        "step": int(step),
        "sample_index": int((sample_offset + step) % total),
        "preprocess_seed": int(PREPROCESS_SEED_BASE + step),
        "rng_seed": int(STEP_SEED_BASE + step),
        "epoch": int(STEP_EPOCH),
        "iteration": int(step),
        "niters_per_epoch": int(config.niters_per_epoch),
        "nepochs": int(config.nepochs),
    }


def build_step_batch(config: Any, plan: Mapping[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    """Deterministically build one real training batch for a planned step."""
    seed_everything(int(plan["rng_seed"]))
    sample = load_training_sample(config, int(plan["sample_index"]), int(plan["preprocess_seed"]))
    corruption = build_corrupted_batch(
        config,
        sample,
        int(plan["epoch"]),
        int(plan["iteration"]),
        0,
        1,
        False,
    )
    batch = corruption["batch"]
    metadata = corruption["metadata"]
    return batch, metadata, sample


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def run_trajectory(
    *,
    label: str,
    model: nn.Module,
    config: Any,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    batch_size: int,
    steps: Sequence[Mapping[str, Any]],
    min_successful_updates: int,
    lambda_rel: float,
    batch_identity_reference: Mapping[int, Mapping[str, str]] | None,
    shared_names: Sequence[str],
    head_names: Sequence[str],
    keep_snapshots: bool,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one AMP trajectory and return its telemetry records."""
    restore_model_state(model, initial_state)
    model.train()
    model.reliability_weight = float(lambda_rel)
    optimizer = build_optimizer(model, config)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    managed_ids = managed_parameter_ids(optimizer)
    managed_shared_names = [name for name in shared_names if id(dict(model.named_parameters())[name]) in managed_ids]
    unmanaged_shared_names = [name for name in shared_names if name not in set(managed_shared_names)]
    report.setdefault("optimizer_parameter_ownership", {})[label] = {
        "managed_parameter_tensors": len(managed_ids),
        "managed_shared_parameter_tensors": len(managed_shared_names),
        "unmanaged_shared_parameter_tensors": len(unmanaged_shared_names),
        "unmanaged_shared_parameter_names": unmanaged_shared_names,
        "note": (
            "parameters outside every optimizer param group are never updated and never enter GradScaler's "
            "unscale/inf check; they cannot contribute to a divergence between the two trajectories"
        ),
    }
    scaler = torch.cuda.amp.GradScaler()
    records: List[Dict[str, Any]] = []
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    successful_updates = 0
    corrupt_steps = 0
    capacity_failure = False

    for plan in steps:
        if successful_updates >= int(min_successful_updates):
            break
        step = int(plan["step"])
        try:
            batch, metadata, sample = build_step_batch(config, plan)
        except torch.cuda.OutOfMemoryError:
            capacity_failure = True
            report["failures"].append(f"{label}: CUDA OOM while building the batch at step {step}")
            break
        identity = build_batch_identity(batch)
        identity_hash = combined_batch_hash(identity)
        tensors = move_batch_to_device(batch, device)
        label_tensor = sample["label"].unsqueeze(0).to(device)
        auxiliary = {
            "raw_rgb": tensors["raw_rgb"],
            "raw_depth": tensors["raw_depth"],
            "reliability_target": tensors["reliability_target"],
            "reliability_valid_mask": tensors["valid_mask"],
            "depth_valid": tensors["depth_valid_post"],
        }
        seed_everything(int(plan["rng_seed"]))
        optimizer.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale())
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(tensors["rgb"], tensors["depth"], label_tensor, **auxiliary)
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError:
            capacity_failure = True
            torch.cuda.empty_cache()
            report["failures"].append(f"{label}: CUDA OOM during forward/backward at step {step}")
            break
        gradient_record = shared_gradient_hashes(model, shared_names)
        non_finite = non_finite_parameter_names(model, shared_names + head_names, managed_ids)
        unmanaged_non_finite = non_finite_parameter_names(model, unmanaged_shared_names, None)
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        applied = bool(optimizer_step_was_applied(scale_before, scale_after))
        optimizer.zero_grad(set_to_none=True)

        record: Dict[str, Any] = {
            "step": step,
            "sample_index": int(plan["sample_index"]),
            "sample_id": str(metadata["sample_id"]),
            "corruption_clean": bool(metadata["clean"]),
            "num_specs": int(metadata["num_specs"]),
            "spec_kinds": [str(spec["kind"]) for spec in metadata["specs"]],
            "batch_identity": identity,
            "batch_identity_hash": identity_hash,
            "loss": finite_or_none(float(loss.detach().item())),
            "loss_finite": bool(math.isfinite(float(loss.detach().item()))),
            "scale_before": scale_before,
            "scale_after": scale_after,
            "optimizer_step_applied": applied,
            "shared_gradient_combined_hash": gradient_record["combined_hash"],
            "shared_gradient_max_abs": finite_or_none(gradient_record["max_abs_gradient"]),
            "shared_gradient_missing_names": gradient_record["missing_gradient_names"],
            "non_finite_gradient_names": non_finite,
            "unmanaged_non_finite_gradient_names": unmanaged_non_finite,
            "shared_gradients_finite": not non_finite,
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None,
            "gpu_memory_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        }
        if not bool(metadata["clean"]):
            corrupt_steps += 1
        if batch_identity_reference is not None:
            reference = batch_identity_reference.get(step)
            record["batch_identity_matches_reference"] = bool(reference == identity)
            if reference != identity:
                report["failures"].append(f"{label}: batch identity differs from the reference trajectory at step {step}")
        if applied:
            successful_updates += 1
            parameter_hash = combined_hash(shared_parameter_hashes(model, shared_names))
            state_hash = combined_hash(optimizer_state_hashes(optimizer, parameter_names, shared_names))
            record["successful_update_index"] = successful_updates
            record["shared_parameter_combined_hash"] = parameter_hash
            record["shared_optimizer_state_combined_hash"] = state_hash
            record["reliability_head_parameter_combined_hash"] = combined_hash(
                shared_parameter_hashes(model, head_names)
            )
            if keep_snapshots:
                snapshots[step] = {
                    name: dict(model.named_parameters())[name].detach().cpu().clone() for name in shared_names
                }
        records.append(record)

    return {
        "label": label,
        "lambda_rel": float(lambda_rel),
        "records": records,
        "attempted_steps": len(records),
        "successful_updates": successful_updates,
        "corrupt_steps": corrupt_steps,
        "capacity_failure": capacity_failure,
        "final_scale": float(scaler.get_scale()),
        "final_shared_parameter_hashes": shared_parameter_hashes(model, shared_names),
        "final_reliability_head_parameter_hashes": shared_parameter_hashes(model, head_names),
        "snapshots": snapshots,
        "final_model": model,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_trajectories(
    off: Mapping[str, Any],
    on: Mapping[str, Any],
    shared_names: Sequence[str],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    off_records = list(off["records"])
    on_records = list(on["records"])
    length = min(len(off_records), len(on_records))
    step_skip_mismatches: List[Dict[str, Any]] = []
    scale_mismatches: List[Dict[str, Any]] = []
    parameter_mismatches: List[Dict[str, Any]] = []
    optimizer_mismatches: List[Dict[str, Any]] = []
    max_abs_difference = 0.0
    for index in range(length):
        left, right = off_records[index], on_records[index]
        if bool(left["optimizer_step_applied"]) != bool(right["optimizer_step_applied"]):
            step_skip_mismatches.append(
                {
                    "index": index,
                    "step": left["step"],
                    "lambda_rel_off_applied": bool(left["optimizer_step_applied"]),
                    "lambda_rel_on_applied": bool(right["optimizer_step_applied"]),
                }
            )
        if (left["scale_before"], left["scale_after"]) != (right["scale_before"], right["scale_after"]):
            scale_mismatches.append(
                {
                    "index": index,
                    "step": left["step"],
                    "lambda_rel_off_scale": [left["scale_before"], left["scale_after"]],
                    "lambda_rel_on_scale": [right["scale_before"], right["scale_after"]],
                }
            )
        if bool(left.get("optimizer_step_applied")):
            if left.get("shared_parameter_combined_hash") != right.get("shared_parameter_combined_hash"):
                detail: Dict[str, Any] = {
                    "index": index,
                    "step": left["step"],
                    "lambda_rel_off_hash": left.get("shared_parameter_combined_hash"),
                    "lambda_rel_on_hash": right.get("shared_parameter_combined_hash"),
                }
                reference_snapshot = off["snapshots"].get(int(left["step"]))
                candidate_snapshot = on["snapshots"].get(int(left["step"]))
                if reference_snapshot is not None and candidate_snapshot is not None:
                    differences = []
                    for name in shared_names:
                        current = candidate_snapshot[name]
                        reference = reference_snapshot[name]
                        difference = (current - reference).abs()
                        differences.append(float(difference.max().item()) if difference.numel() else 0.0)
                    if differences:
                        detail["max_abs_difference"] = finite_or_none(max(differences))
                        if detail["max_abs_difference"] is not None:
                            max_abs_difference = max(max_abs_difference, detail["max_abs_difference"])
                parameter_mismatches.append(detail)
            if left.get("shared_optimizer_state_combined_hash") != right.get("shared_optimizer_state_combined_hash"):
                optimizer_mismatches.append(
                    {
                        "index": index,
                        "step": left["step"],
                        "lambda_rel_off_hash": left.get("shared_optimizer_state_combined_hash"),
                        "lambda_rel_on_hash": right.get("shared_optimizer_state_combined_hash"),
                    }
                )
    final_parameter_mismatch_names = [
        name
        for name in shared_names
        if off["final_shared_parameter_hashes"].get(name) != on["final_shared_parameter_hashes"].get(name)
    ]
    return {
        "compared_steps": length,
        "attempted_steps_off": int(off["attempted_steps"]),
        "attempted_steps_on": int(on["attempted_steps"]),
        "successful_updates_off": int(off["successful_updates"]),
        "successful_updates_on": int(on["successful_updates"]),
        "corrupt_steps_off": int(off["corrupt_steps"]),
        "corrupt_steps_on": int(on["corrupt_steps"]),
        "step_skip_sequence_identical": not step_skip_mismatches,
        "step_skip_mismatches": step_skip_mismatches,
        "scale_trajectory_identical": not scale_mismatches,
        "scale_mismatches": scale_mismatches,
        "shared_parameter_mismatch_count": len(parameter_mismatches),
        "shared_parameter_mismatches": parameter_mismatches,
        "shared_optimizer_state_mismatch_count": len(optimizer_mismatches),
        "shared_optimizer_state_mismatches": optimizer_mismatches,
        "shared_parameter_max_abs_difference": max_abs_difference,
        "final_shared_parameter_mismatch_names": final_parameter_mismatch_names,
        "final_shared_parameter_bitwise_identical": not final_parameter_mismatch_names,
        "reliability_head_parameters_allowed_to_differ": True,
        "reliability_head_final_hash_off": combined_hash(off["final_reliability_head_parameter_hashes"]),
        "reliability_head_final_hash_on": combined_hash(on["final_reliability_head_parameter_hashes"]),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    report = _base_report(args, device)
    try:
        qualify(args, device, report)
    except Exception:  # noqa: BLE001 - evidence must be written even on abort
        report["failures"].append("unhandled exception: " + traceback.format_exc().splitlines()[-1])
        report["traceback"] = traceback.format_exc()
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    verdict = "PASS" if not report["failures"] else "FAIL"
    report["verdict"] = f"{CHECK_ID}: {verdict}"
    report["exit_code"] = 0 if verdict == "PASS" else 1
    write_json(Path(args.output), report)
    print_summary(report, Path(args.output))
    return int(report["exit_code"])


def qualify(args: argparse.Namespace, device: torch.device, report: Dict[str, Any]) -> None:
    started = time.time()
    report["determinism"] = configure_runtime(device)
    config = load_config(args.config)
    report["config"] = {
        "module": args.config,
        "run_id": str(getattr(config, "run_id", "")),
        "protocol": str(dict(getattr(config, "mmfr_a2", {})).get("protocol", "")),
        "mode": str(dict(getattr(config, "mmfr_a2", {})).get("mode", "")),
        "corruption_basis": str(dict(dict(getattr(config, "mmfr_a2", {})).get("corruption", {})).get("basis", "")),
        "reliability_weight": float(dict(dict(getattr(config, "mmfr_a2", {})).get("reliability_head", {})).get("weight", 0.0)),
        "reliability_supervised_channels": list(
            dict(dict(getattr(config, "mmfr_a2", {})).get("reliability_head", {})).get("supervised_channels", [])
        ),
        "train_source": str(config.train_source),
        "num_train_imgs": int(config.num_train_imgs),
        "niters_per_epoch": int(config.niters_per_epoch),
        "nepochs": int(config.nepochs),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "optimizer": str(config.optimizer),
        "config_file": str(Path(importlib.import_module(args.config).__file__).resolve()),
        "config_file_sha256": file_sha256(Path(importlib.import_module(args.config).__file__).resolve()),
    }
    if report["config"]["protocol"] != EXPECTED_PROTOCOL:
        report["failures"].append(f"protocol identity {report['config']['protocol']!r} is not {EXPECTED_PROTOCOL!r}")
    if report["config"]["corruption_basis"] != EXPECTED_CORRUPTION_BASIS:
        report["failures"].append(
            f"corruption basis {report['config']['corruption_basis']!r} is not {EXPECTED_CORRUPTION_BASIS!r}"
        )
    if abs(float(args.lambda_rel_on) - float(report["config"]["reliability_weight"])) > 0:
        report["warnings"].append(
            "the requested lambda_rel_on differs from config.mmfr_a2.reliability_head.weight; the sweep no longer "
            "matches the frozen training weight"
        )
    pretrained_path, source = resolve_pretrained_path(config)
    report["pretrained"] = verify_pretrained_file(pretrained_path, EXPECTED_PRETRAINED_SHA256)
    report["pretrained"]["path_source"] = source

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    model = build_model(config, criterion, device)
    report["model"] = {
        "parameter_tensor_count": sum(1 for _ in model.parameters()),
        "parameter_element_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "batch_size": int(args.batch_size),
        "construction_note": "built through tools/mmfr/gradient_path_isolation.build_model (same path as the FP32 gate)",
    }
    shared_names = shared_parameter_names(model)
    head_names = reliability_head_names(model)
    report["parameter_groups"] = {
        "shared_parameter_count": len(shared_names),
        "reliability_head_parameter_count": len(head_names),
        "shared_parameter_names_sample": shared_names[:5],
        "reliability_head_parameter_names": head_names,
        "optimizer_managed_expected": "677 of 720 tensors (43 unmanaged: 29 Geo.weight, 8 patch_embed.proj.*, 6 downsample.norm.*) per the verified group_weight behaviour",
    }
    if not shared_names:
        raise RuntimeError("no shared parameter was found; the grouping rule cannot be applied")
    if not head_names:
        raise RuntimeError("no reliability-head parameter was found; the gate would be vacuous")

    steps = [plan_step(config, step, int(args.sample_offset)) for step in range(int(args.max_attempts))]
    report["step_plan"] = {
        "step_epoch": STEP_EPOCH,
        "step_seed_base": STEP_SEED_BASE,
        "preprocess_seed_base": PREPROCESS_SEED_BASE,
        "rng_policy": (
            "python/numpy/torch CPU+CUDA RNG are reset to STEP_SEED_BASE + step right before the batch is built and "
            "again right before the forward, identically in both trajectories"
        ),
        "steps": steps,
    }

    initial_state = snapshot_model_state(model)
    shared_bytes = sum(dict(model.named_parameters())[name].numel() for name in shared_names) * 4
    keep_snapshots = args.keep_snapshots == "yes" or (
        args.keep_snapshots == "auto" and shared_bytes * max(1, int(args.min_successful_updates)) <= SNAPSHOT_MEMORY_LIMIT_BYTES
    )
    report["snapshot_policy"] = {
        "keep_snapshots": bool(keep_snapshots),
        "shared_parameter_bytes": int(shared_bytes),
        "memory_limit_bytes": int(SNAPSHOT_MEMORY_LIMIT_BYTES),
        "note": (
            "snapshots allow a numeric max-abs-difference report; without them the comparison is byte-hash based "
            "(bitwise identity), which is the stronger criterion"
        ),
    }
    offset_records: Dict[int, Dict[str, str]] = {}
    offset_off = run_trajectory(
        label=f"lambda_rel_{float(args.lambda_rel_off)}",
        model=model,
        config=config,
        device=device,
        initial_state=initial_state,
        batch_size=int(args.batch_size),
        steps=steps,
        min_successful_updates=int(args.min_successful_updates),
        lambda_rel=float(args.lambda_rel_off),
        batch_identity_reference=None,
        shared_names=shared_names,
        head_names=head_names,
        keep_snapshots=keep_snapshots,
        report=report,
    )
    for record in offset_off["records"]:
        offset_records[int(record["step"])] = record["batch_identity"]
    offset_on = run_trajectory(
        label=f"lambda_rel_{float(args.lambda_rel_on)}",
        model=model,
        config=config,
        device=device,
        initial_state=initial_state,
        batch_size=int(args.batch_size),
        steps=steps,
        min_successful_updates=int(args.min_successful_updates),
        lambda_rel=float(args.lambda_rel_on),
        batch_identity_reference=offset_records,
        shared_names=shared_names,
        head_names=head_names,
        keep_snapshots=keep_snapshots,
        report=report,
    )
    report["trajectory_lambda_rel_off"] = {key: value for key, value in offset_off.items() if key not in ("snapshots", "final_model")}
    report["trajectory_lambda_rel_on"] = {key: value for key, value in offset_on.items() if key not in ("snapshots", "final_model")}
    comparison = compare_trajectories(offset_off, offset_on, shared_names, report)
    report["comparison"] = comparison

    if int(offset_on["attempted_steps"]) != int(offset_off["attempted_steps"]):
        report["failures"].append(
            "the two trajectories attempted a different number of steps "
            f"({offset_off['attempted_steps']} vs {offset_on['attempted_steps']})"
        )
    successful = min(int(offset_off["successful_updates"]), int(offset_on["successful_updates"]))
    if successful < int(args.min_successful_updates):
        report["failures"].append(
            f"only {successful} successful optimizer updates were obtained within {int(args.max_attempts)} attempted "
            f"steps; the gate requires {int(args.min_successful_updates)}"
        )
    if min(int(offset_off["corrupt_steps"]), int(offset_on["corrupt_steps"])) < int(args.min_corrupt_steps):
        report["failures"].append(
            "not enough corrupted steps were drawn; the sweep does not exercise the reliability branch"
        )
    if not comparison["step_skip_sequence_identical"]:
        report["failures"].append("the two trajectories do not share the same step/skip sequence")
    if not comparison["scale_trajectory_identical"]:
        report["failures"].append("the two trajectories do not share the same GradScaler scale trajectory")
    if comparison["shared_parameter_mismatch_count"]:
        report["failures"].append(
            f"{comparison['shared_parameter_mismatch_count']} successful update(s) produced different shared parameters"
        )
    if comparison["shared_optimizer_state_mismatch_count"]:
        report["failures"].append(
            f"{comparison['shared_optimizer_state_mismatch_count']} successful update(s) produced different shared optimizer state"
        )
    if not comparison["final_shared_parameter_bitwise_identical"]:
        report["failures"].append(
            "the final shared parameters are not bitwise identical: "
            + ", ".join(comparison["final_shared_parameter_mismatch_names"][:8])
        )
    if not offset_off["records"] or all(record["shared_gradient_missing_names"] for record in offset_off["records"]):
        report["failures"].append("no shared gradient was observed; the comparison would be vacuous")
    report["summary"] = {
        "attempted_steps": int(offset_off["attempted_steps"]),
        "successful_updates": successful,
        "corrupt_steps": int(offset_off["corrupt_steps"]),
        "step_skip_sequence_identical": comparison["step_skip_sequence_identical"],
        "scale_trajectory_identical": comparison["scale_trajectory_identical"],
        "shared_parameter_bitwise_identical_at_every_successful_update": comparison["shared_parameter_mismatch_count"] == 0,
        "shared_optimizer_state_identical_at_every_successful_update": comparison["shared_optimizer_state_mismatch_count"] == 0,
        "reliability_head_parameters_differ_allowed": True,
        "reliability_head_final_hash_changed": comparison["reliability_head_final_hash_off"]
        != comparison["reliability_head_final_hash_on"],
        "elapsed_seconds": round(time.time() - started, 3),
    }


def print_summary(report: Mapping[str, Any], output_path: Path) -> None:
    print(report.get("verdict", "MMFR-A2-v2-amp-update-path-isolation: UNKNOWN"))
    summary = report.get("summary", {})
    if summary:
        print(
            f"  attempted steps: {summary['attempted_steps']}  successful updates: {summary['successful_updates']}  "
            f"corrupt steps: {summary['corrupt_steps']}"
        )
        print(
            f"  step/skip identical: {summary['step_skip_sequence_identical']}  "
            f"scale trajectory identical: {summary['scale_trajectory_identical']}"
        )
        print(
            "  shared params bitwise identical at every successful update: "
            f"{summary['shared_parameter_bitwise_identical_at_every_successful_update']}"
        )
        print(
            "  shared optimizer state identical at every successful update: "
            f"{summary['shared_optimizer_state_identical_at_every_successful_update']}"
        )
        print(f"  reliability head final hash changed: {summary['reliability_head_final_hash_changed']}")
        print(f"  elapsed seconds: {summary['elapsed_seconds']}")
    for failure in report.get("failures", []):
        print(f"  FAILURE: {failure}")
    for warning in report.get("warnings", []):
        print(f"  WARNING: {warning}")
    print(f"  evidence: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
