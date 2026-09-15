#!/usr/bin/env python3
"""Pointed GPU qualification: the MMFR-A2 v2 reliability head is gradient-path isolated.

本工具尚未被执行过；其正确性未经验证；正式 A2 v2 训练前必须运行并通过本检查（单次运行
预计需要本地 GPU、batch 1、真实 pretrained 与真实 train-dev 样本）。

Recommended run command::

    python tools/mmfr/gradient_path_isolation.py

What one run does
-----------------
It builds the frozen ``MMFR-A2-depth-corruption-train-v2`` model (DFormerv2-S backbone plus
Ham decoder, official pretrained, ``reliability_estimator`` auxiliary head), applies the
frozen v2 Depth corruption to one real ``train-dev`` sample in the main process, and then
runs the reliability-weight sweep on that single batch:

1. two backwards over the *same* model state, input batch, corruption draw, RNG state and
   segmentation loss, once with ``lambda_rel = 0.0`` and once with ``lambda_rel = 0.1``;
2. every **shared** parameter (any parameter whose name does not start with
   ``reliability_estimator.``, i.e. the DFormerv2 backbone, the Ham decoder, the geometry
   prior and any other non-reliability head) must receive the same gradient in both runs
   inside the ``ATOL`` / ``RTOL`` tolerance declared below; the report lists the max
   absolute deviation, the max relative deviation, how many parameters exceed the tolerance
   and their names;
3. the ``reliability_estimator`` parameters must be present-and-exactly-zero at
   ``lambda_rel = 0.0`` and must carry a non-zero gradient norm at ``lambda_rel = 0.1``;
4. the segmentation logits must be identical with the reliability branch off and on
   (``torch.equal`` first, otherwise the largest deviation inside the tolerance) and the
   segmentation loss must be the same scalar in both runs;
5. any deviation makes the process exit non-zero and the report carries the literal verdict
   ``gradient-path-isolation: PASS`` or ``gradient-path-isolation: FAIL``.

Evidence
--------
``outputs/mmfr-a2-gradient-path-isolation/gradient-path-isolation.json`` (schema
``museg-mmfr-a2-gradient-path-isolation-v1``) records the actual pretrained SHA-256 plus the
checkpoint-versus-backbone key overlap and any mismatching key, the config/run identity, the
resolved sample id with its input file hashes, the fixed corruption
seed/epoch/iteration/global-rank/slot words, the RNG replay, the per-parameter gradient
deviations and the tolerances. The tool writes no checkpoint and no metric, never reads the
sealed ``official-test`` split and never starts training or evaluation. The JSON is written
even when the run aborts early, so a failed qualification still leaves evidence behind.

Requirements, deviations and unverified points
----------------------------------------------
* A CUDA device is mandatory, not optional: the Ham decoder builds its NMF bases on the GPU
  (``torch.rand(...).cuda()``), so the forward cannot run on CPU.
* The official pretrained checkpoint is loaded through the repository's own backbone
  ``init_weights`` path, which calls ``load_state_dict(..., strict=False)``. The tool does **not**
  assume whether a strict load would succeed or fail: it compares the checkpoint key set with the
  built backbone key set, *computes* ``strict_load_possible`` from the two sets and records the
  missing and unexpected keys, so the first real run settles that point from evidence. The load is
  accepted only when every overlapping tensor matches the checkpoint exactly.
* The training profile uses AMP plus SyncBN. This qualification deliberately runs a plain FP32
  single-process forward (no autocast, TF32 disabled). The DFormerv2 backbone always builds
  ``nn.SyncBatchNorm`` inside ``PatchEmbed``/``PatchMerging`` regardless of ``norm_cfg``; because
  no process group is initialized, PyTorch falls back to per-process batch normalization for those
  layers, which is exactly the plain-BN semantics a single-process isolation check wants. Both
  sweeps travel the same numeric path, and the deviation from the distributed training profile is
  recorded in the report.
* Not verified by any execution yet: whether the captured CPU/CUDA RNG replay makes the two
  forwards bitwise reproducible, that the chosen sample and corruption draw really exercise the
  reliability branch, and that every shared parameter really receives a gradient at all. The
  ``encode_decode`` / ``reliability_auxiliary_loss`` signatures and keyword names used below were
  checked statically against this repository, not by running the tool.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import math
import os
import platform
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# The training entry point sets this before the CUDA context exists; keep parity so the
# cuBLAS workspace does not silently change the comparison.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from models.builder import EncoderDecoder
from models.losses.safe_masked_loss import safe_masked_mean
from tools.evaluate_museg_checkpoint import configure_fp32_forward
from tools.museg_protocol import file_sha256, write_json
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import TrainPre
from utils.dataloader.mmfr_training_v2 import build_mmfr_training_batch_v2

# ---------------------------------------------------------------------------
# Frozen identity, tolerance and geometry constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "museg-mmfr-a2-gradient-path-isolation-v1"
CHECK_ID = "gradient-path-isolation"
#: The v2 child config; importing it binds ``MMFR-A2-depth-corruption-train-v2``.
CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v2"
#: ``DFORMER_PRETRAINED`` wins through the config; this is the repository default fallback.
PRETRAINED_ENVIRONMENT_VARIABLE = "DFORMER_PRETRAINED"
DEFAULT_PRETRAINED_PATH = r"D:\0Project\pretrained\DFormerv2_Small_pretrained.pth"
EXPECTED_PRETRAINED_SHA256 = "19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "mmfr-a2-gradient-path-isolation" / "gradient-path-isolation.json"

#: Gradient/logit tolerance. The comparison first reports bitwise equality (atol = rtol = 0)
#: and then decides PASS/FAIL on ``|reference - candidate| <= ATOL + RTOL * |reference|``.
ATOL = 1e-6
RTOL = 1e-6
#: Floor for the reported relative deviation, so an all-zero gradient cannot divide by zero.
RELATIVE_DEVIATION_FLOOR = 1e-12

#: Frozen parameter grouping rule: the reliability head is exactly this module prefix, and
#: every other parameter is a shared model parameter that must not see the auxiliary loss.
RELIABILITY_HEAD_PARAMETER_PREFIX = "reliability_estimator."
#: Name marker of the DFormerv2 geometry prior inside the backbone GSA blocks.
GEOMETRY_PRIOR_NAME_MARKER = ".Geo."
#: The frozen A2 v2 weights swept here (``L = L_seg + lambda_rel * L_rel``).
LAMBDA_REL_OFF = 0.0
LAMBDA_REL_ON = 0.1

#: Fixed corruption/geometry position of the qualification. Batch size is 1, so the single
#: sample slot is 0 and the global rank is 0. ``iteration`` is advanced deterministically (see
#: ``CORRUPTION_DRAW_SEARCH_LIMIT``) until the frozen draw is a corrupted, not a clean, sample.
CORRUPTION_EPOCH = 1
CORRUPTION_ITERATION = 0
CORRUPTION_GLOBAL_RANK = 0
CORRUPTION_SAMPLE_SLOT = 0
CORRUPTION_DRAW_SEARCH_LIMIT = 32
#: Python ``random`` seed of the frozen ``TrainPre`` mirror/scale/crop, so the 480x640 crop is
#: reproducible. ``TrainPre`` draws from the process-wide ``random`` module, nothing else.
PREPROCESS_SEED = 2026091403
DEFAULT_SAMPLE_INDEX = 0
#: Seeds the model build and the initial global RNG state for reproducibility.
RUNTIME_SEED = 772961337


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars, arrays and tensors into JSON-safe values.

    The corruption metadata carries numpy scalars, and the evidence writer refuses NaN/Inf,
    so non-finite floats are recorded as their string form instead of breaking the write.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    return str(value)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 of the raw float32 bytes, so two runs can be compared by identity."""
    array = np.ascontiguousarray(tensor.detach().to(torch.float32).cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def _l2_norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().to(torch.float64)).item())


def _rank_parameter_name(name: str) -> str:
    """Report-only grouping of a shared parameter name."""
    if name.startswith(RELIABILITY_HEAD_PARAMETER_PREFIX):
        return "reliability_head"
    if GEOMETRY_PRIOR_NAME_MARKER in name:
        return "geometry_prior"
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith(("decode_head.", "aux_head.")):
        return "decoder"
    return "other"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MMFR-A2 v2 gradient-path isolation qualification on GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=CONFIG_MODULE, help="v2 train config module")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="JSON evidence path")
    parser.add_argument("--device", default="cuda:0", help="CUDA device; CPU cannot run this check")
    parser.add_argument("--sample-index", type=int, default=DEFAULT_SAMPLE_INDEX, help="index into train-dev")
    parser.add_argument("--epoch", type=int, default=CORRUPTION_EPOCH)
    parser.add_argument("--iteration", type=int, default=CORRUPTION_ITERATION)
    parser.add_argument("--global-rank", type=int, default=CORRUPTION_GLOBAL_RANK)
    parser.add_argument("--preprocess-seed", type=int, default=PREPROCESS_SEED)
    parser.add_argument("--lambda-rel-on", type=float, default=LAMBDA_REL_ON)
    parser.add_argument(
        "--require-corrupt-draw",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="advance the iteration until the frozen corruption draws a corrupted sample",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Environment, config and pretrained
# ---------------------------------------------------------------------------
def configure_runtime(device: torch.device, seed: int) -> Dict[str, Any]:
    """Pin the single-process determinism flags and the FP32 forward path."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    configure_fp32_forward(device)
    return {
        "seed": int(seed),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": True,
        "deterministic_algorithms_mode": "warn_only",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "autocast": "disabled (fp32)",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def load_config(module_name: str) -> Any:
    """Import the frozen v2 config and return a copy of ``C`` without mutating the module."""
    module = importlib.import_module(module_name)
    if not hasattr(module, "C"):
        raise RuntimeError(f"config module {module_name!r} does not expose C")
    config = copy.copy(getattr(module, "C"))
    mmfr_a2 = dict(getattr(config, "mmfr_a2", None) or {})
    if not mmfr_a2.get("corruption") or not mmfr_a2.get("reliability_head"):
        raise RuntimeError(f"{module_name!r} does not bind MMFR-A2 Depth corruption with a reliability head")
    return config


def resolve_pretrained_path(config: Any) -> Tuple[Path, str]:
    """Return the official pretrained path and where it came from.

    ``C.pretrained_model`` already prefers ``DFORMER_PRETRAINED`` and otherwise resolves the
    repository default, so the config stays the single source of truth here.
    """
    environment_value = os.environ.get(PRETRAINED_ENVIRONMENT_VARIABLE)
    configured = getattr(config, "pretrained_model", None)
    if configured not in (None, ""):
        source = "config.pretrained_model"
        if environment_value:
            source += f" ({PRETRAINED_ENVIRONMENT_VARIABLE} is set)"
        return Path(str(configured)).resolve(), source
    return Path(DEFAULT_PRETRAINED_PATH).resolve(), "script default constant"


def verify_pretrained_file(path: Path, expected_sha256: str) -> Dict[str, Any]:
    """Fail closed on a missing or substituted official pretrained checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"official pretrained checkpoint not found: {path}")
    actual_sha256 = file_sha256(path)
    record = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "sha256_matches_expected": actual_sha256.lower() == expected_sha256.lower(),
    }
    if not record["sha256_matches_expected"]:
        raise RuntimeError(
            f"official pretrained SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return record


def checkpoint_backbone_state(path: Path) -> Dict[str, torch.Tensor]:
    """Read the official pretrained state and strip the wrapper/``backbone.`` prefixes."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("the official pretrained checkpoint does not decode to a mapping")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise RuntimeError("the official pretrained checkpoint carries no state mapping")
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = str(key).removeprefix("module.").removeprefix("backbone.")
        normalized[name] = value
    return normalized


def verify_pretrained_load(model: EncoderDecoder, backbone_state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Verify what the repository's non-strict pretrained load actually applied.

    The repository path is a ``strict=False`` load, so the tool does not trust it: it compares
    the checkpoint key set with the built backbone key set, computes whether a strict load could
    have succeeded, and accepts the load only when every overlapping tensor matches the
    checkpoint exactly. An overlapping mismatch means the pretrained load silently failed.
    """
    current = model.backbone.state_dict()
    overlapping = sorted(set(current) & set(backbone_state))
    missing_in_checkpoint = sorted(set(current) - set(backbone_state))
    unexpected_in_checkpoint = sorted(set(backbone_state) - set(current))
    mismatched: List[str] = []
    for name in overlapping:
        source = backbone_state[name]
        target = current[name]
        if tuple(source.shape) != tuple(target.shape) or not torch.equal(
            source.to(target.dtype), target.detach().cpu()
        ):
            mismatched.append(name)
    strict_load_possible = not missing_in_checkpoint and not unexpected_in_checkpoint
    record = {
        "backbone_parameter_and_buffer_keys": len(current),
        "checkpoint_keys": len(backbone_state),
        "overlapping_keys": len(overlapping),
        "overlapping_tensors_exact_match": not mismatched,
        "mismatched_overlapping_keys": mismatched,
        "missing_in_checkpoint": missing_in_checkpoint,
        "unexpected_in_checkpoint": unexpected_in_checkpoint,
        "strict_load_used": False,
        "strict_load_possible": strict_load_possible,
        "strict_load_decision_source": (
            "computed from the checkpoint key set versus the built backbone key set; not assumed"
        ),
        "strict_load_blockers": sorted(missing_in_checkpoint + unexpected_in_checkpoint),
        "load_path_note": (
            "the repository backbone init_weights path uses load_state_dict(..., strict=False), and "
            "load_state_dict ignores a missing num_batches_tracked; the tool reproduces the load and "
            "then requires every overlapping tensor to match exactly"
        ),
    }
    if mismatched:
        raise RuntimeError(
            "the official pretrained load did not apply exactly: "
            + ", ".join(mismatched[:8])
        )
    return record


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(config: Any, criterion: nn.Module, device: torch.device) -> EncoderDecoder:
    """Build DFormerv2-S plus Ham decoder exactly the way the training entry point does.

    Passing the criterion is what makes ``EncoderDecoder.init_weights`` load the official
    pretrained backbone; the reliability head is created afterwards and keeps its module
    defaults. ``syncbn=False`` only changes ``norm_cfg`` for the decoder and for the backbone's
    configurable norms; the backbone's ``PatchEmbed``/``PatchMerging`` build ``nn.SyncBatchNorm``
    unconditionally, and because this single-process tool initializes no process group those
    layers fall back to per-process batch normalization.
    """
    model = EncoderDecoder(
        cfg=config,
        criterion=criterion,
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
    )
    model.to(device)
    model.train()
    return model


def model_record(model: EncoderDecoder, config: Any, device: torch.device) -> Dict[str, Any]:
    if model.reliability_estimator is None:
        raise RuntimeError("the v2 config did not build the reliability head")
    hamburger = getattr(model.decode_head, "hamburger", None)
    ham = getattr(hamburger, "ham", None)
    return {
        "run_id": str(getattr(config, "run_id", "")),
        "analysis_identity": str(dict(config.mmfr_a2.get("frozen", {})).get("analysis_identity", "")),
        "protocol": str(config.mmfr_a2.get("protocol", "")),
        "mode": str(config.mmfr_a2.get("mode", "")),
        "backbone": str(config.backbone),
        "decoder": str(config.decoder),
        "num_classes": int(config.num_classes),
        "background": int(config.background),
        "image_height": int(config.image_height),
        "image_width": int(config.image_width),
        "epochs": int(config.nepochs),
        "iterations_per_epoch": int(config.niters_per_epoch),
        "syncbn_flag_passed_to_builder": False,
        "decoder_norm_layer": "torch.nn.BatchNorm2d",
        "backbone_hardcoded_sync_batchnorm_modules": True,
        "backbone_sync_batchnorm_fallback": (
            "PatchEmbed/PatchMerging always build nn.SyncBatchNorm; with no process group "
            "initialized PyTorch falls back to per-process batch normalization"
        ),
        "distributed_initialized": bool(torch.distributed.is_initialized()),
        "model_mode": "train",
        "backbone_norm_eval": bool(getattr(model.backbone, "norm_eval", False)),
        "aux_head_present": model.aux_head is not None,
        "reliability_weight_in_model": float(model.reliability_weight),
        "reliability_supervised_channels": list(model.reliability_supervised_channels),
        "device": str(device),
        "decoder_rand_init": bool(getattr(ham, "rand_init", False)) if ham is not None else None,
    }


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------
def build_dataset_setting(config: Any) -> Dict[str, Any]:
    """Mirror ``get_train_loader``'s dataset setting exactly."""
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


def load_training_sample(config: Any, sample_index: int, preprocess_seed: int) -> Dict[str, Any]:
    """Load one real ``train-dev`` sample through the frozen ``TrainPre`` geometry.

    ``TrainPre`` mirrors/scales/crops with the process-wide ``random`` module, so the seed is
    fixed immediately before the sample is read; that keeps the 480x640 crop reproducible.
    """
    dataset = RGBXDataset(
        build_dataset_setting(config),
        "train",
        TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config),
    )
    if not 0 <= sample_index < len(dataset):
        raise ValueError(f"sample index {sample_index} is outside train-dev ({len(dataset)} samples)")
    entry = dataset._file_names[sample_index]
    random.seed(preprocess_seed)
    np.random.seed(preprocess_seed)
    item = dataset[sample_index]
    rgb = item["data"]
    depth = item["modal_x"]
    label = item["label"]
    expected_shape = (int(config.image_height), int(config.image_width))
    if tuple(rgb.shape) != (3,) + expected_shape or tuple(depth.shape) != (3,) + expected_shape:
        raise RuntimeError(f"unexpected preprocessed sample shape: {tuple(rgb.shape)} / {tuple(depth.shape)}")
    if tuple(label.shape) != expected_shape:
        raise RuntimeError(f"unexpected label shape {tuple(label.shape)}")
    stem = Path(entry).stem
    inputs = {
        "rgb": Path(str(config.rgb_root_folder)) / f"{stem}{config.rgb_format}",
        "depth": Path(str(config.x_root_folder)) / f"{stem}{config.x_format}",
        "label": Path(str(config.gt_root_folder)) / f"{stem}{config.gt_format}",
    }
    return {
        "index": int(sample_index),
        "entry": str(entry),
        "stem": stem,
        "rgb": rgb,
        "depth": depth,
        "label": label,
        "function_path": str(item["fn"]),
        "preprocess_sign": bool(config.x_is_single_channel),
        "preprocess_seed": int(preprocess_seed),
        "tensor_sha256": {
            "rgb": _tensor_sha256(rgb),
            "depth": _tensor_sha256(depth),
            "label": _tensor_sha256(label.float()),
        },
        "input_files": {
            role: {"path": str(path), "sha256": file_sha256(path) if path.is_file() else None}
            for role, path in inputs.items()
        },
    }


def build_corrupted_batch(
    config: Any,
    sample: Mapping[str, Any],
    epoch: int,
    iteration: int,
    global_rank: int,
    search_limit: int,
    require_corrupt_draw: bool,
) -> Dict[str, Any]:
    """Run the frozen v2 corruption helper for one batch-1 sample.

    Everything except the iteration is fixed. The iteration is advanced deterministically
    until the frozen draw is a corrupted sample, because a clean draw would leave the
    reliability target at its neutral value and weaken check 3; every scanned iteration is
    recorded so the choice stays auditable.
    """
    corruption = dict(config.mmfr_a2["corruption"])
    rgb = sample["rgb"].unsqueeze(0)
    depth = sample["depth"].unsqueeze(0)
    scanned: List[Dict[str, Any]] = []
    chosen: Dict[str, Any] | None = None
    warnings: List[str] = []
    for offset in range(max(1, int(search_limit))):
        candidate_iteration = int(iteration) + offset
        batch = build_mmfr_training_batch_v2(
            rgb,
            depth,
            [sample["function_path"]],
            epoch=int(epoch),
            iteration=candidate_iteration,
            niters_per_epoch=int(config.niters_per_epoch),
            nepochs=int(config.nepochs),
            rgb_mean=config.norm_mean,
            rgb_std=config.norm_std,
            corruption_seed=int(corruption["seed"]),
            global_rank=int(global_rank),
            p_clean=float(corruption["p_clean"]),
            max_specs=int(corruption["max_specs"]),
            sample_id_root=getattr(config, "dataset_path", None),
        )
        metadata = batch["metadata"][0]
        clean = bool(metadata["clean"])
        scanned.append({"iteration": candidate_iteration, "clean": clean, "num_specs": int(metadata["num_specs"])})
        if chosen is None:
            chosen = batch
            chosen_iteration = candidate_iteration
        if not clean:
            chosen = batch
            chosen_iteration = candidate_iteration
            break
    else:
        warnings.append(
            f"no corrupted draw within {search_limit} iteration(s) from {iteration}; the fallback "
            "sample is clean, so the reliability target is neutral and check 3 is weaker"
        )
    assert chosen is not None
    metadata = chosen["metadata"][0]
    if require_corrupt_draw and bool(metadata["clean"]):
        raise RuntimeError(
            "the frozen corruption draw is a clean sample; pass --no-require-corrupt-draw to accept a "
            "weaker check or choose another --sample-index"
        )
    return {
        "batch": chosen,
        "metadata": metadata,
        "chosen_iteration": int(chosen_iteration),
        "requested_iteration": int(iteration),
        "corruption": {
            "protocol": str(config.mmfr_a2["protocol"]),
            "basis": str(corruption["basis"]),
            "seed": int(corruption["seed"]),
            "p_clean": float(corruption["p_clean"]),
            "max_specs": int(corruption["max_specs"]),
            "modalities": list(corruption["modalities"]),
            "kinds": list(corruption["kinds"]),
        },
        "position": {
            "epoch": int(epoch),
            "iteration": int(chosen_iteration),
            "global_rank": int(global_rank),
            "sample_slot": CORRUPTION_SAMPLE_SLOT,
            "batch_size": 1,
            "num_workers": 0,
        },
        "search": {
            "requested_iteration": int(iteration),
            "limit": int(search_limit),
            "scanned": scanned,
            "selected_iteration": int(chosen_iteration),
            "selected_clean": bool(metadata["clean"]),
        },
        "warnings": warnings,
    }


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move only the tensors the model consumes; the audit metadata stays on the CPU."""
    keys = (
        "rgb",
        "depth",
        "raw_rgb",
        "raw_depth",
        "reliability_target",
        "valid_mask",
        "depth_valid_post",
    )
    return {key: batch[key].to(device, non_blocking=False) for key in keys}


# ---------------------------------------------------------------------------
# Forward, loss and backward
# ---------------------------------------------------------------------------
def capture_rng_states(device: torch.device) -> Dict[str, Any]:
    states: Dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    return states


def replay_rng_states(states: Mapping[str, Any], device: torch.device) -> None:
    """Replay the captured CPU/CUDA RNG state before a forward.

    The Ham decoder rebuilds its NMF bases with ``torch.rand`` on every forward and the
    backbone applies stochastic drop path in training mode, so two forwards over identical
    inputs are only comparable when both consume the same RNG state.
    """
    torch.set_rng_state(states["cpu"])
    if device.type == "cuda" and states.get("cuda") is not None:
        torch.cuda.set_rng_state(states["cuda"], device)


def snapshot_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def restore_model_state(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore parameters *and* buffers so every sweep starts from the same model state."""
    model.load_state_dict(dict(snapshot), strict=True)


def segmentation_logits(model: EncoderDecoder, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    out = model.encode_decode(tensors["rgb"], tensors["depth"])
    if isinstance(out, tuple):
        out = out[0]
    return out


def segmentation_loss(model: EncoderDecoder, logits: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Reproduce the builder's primary segmentation term exactly."""
    target = label.long()
    valid_mask = target != model.cfg.background
    loss = safe_masked_mean(model.criterion(logits, target), valid_mask)
    return loss, int(valid_mask.sum().item())


def run_forward_only(
    model: EncoderDecoder,
    tensors: Mapping[str, torch.Tensor],
    label: torch.Tensor,
    rng_states: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Segmentation-only reference: the reliability auxiliary branch is never computed."""
    replay_rng_states(rng_states, device)
    logits = segmentation_logits(model, tensors)
    loss, valid_pixels = segmentation_loss(model, logits, label)
    record = {
        "role": "reliability head off: the auxiliary reliability branch is not computed",
        "lambda_rel": None,
        "logits": logits.detach().clone(),
        "logits_sha256": _tensor_sha256(logits),
        "segmentation_loss": loss.detach().clone(),
        "segmentation_loss_value": float(loss.detach().item()),
        "segmentation_valid_pixels": valid_pixels,
    }
    del logits, loss
    torch.cuda.empty_cache()
    return record


def run_forward_backward(
    model: EncoderDecoder,
    tensors: Mapping[str, torch.Tensor],
    label: torch.Tensor,
    lambda_rel: float,
    rng_states: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """One full sweep: zero grads, replay RNG, forward, ``L_seg + lambda_rel * L_rel``, backward.

    The auxiliary branch is computed at ``lambda_rel = 0`` as well, so the reliability head
    really receives a gradient tensor that must be exactly zero (check 3) instead of ``None``.
    """
    model.zero_grad(set_to_none=True)
    replay_rng_states(rng_states, device)
    logits = segmentation_logits(model, tensors)
    seg_loss, valid_pixels = segmentation_loss(model, logits, label)
    reliability_loss = model.reliability_auxiliary_loss(
        tensors["raw_rgb"],
        tensors["raw_depth"],
        tensors["reliability_target"],
        tensors["valid_mask"],
        tensors["depth_valid_post"],
    )
    total_loss = seg_loss + float(lambda_rel) * reliability_loss
    if not bool(torch.isfinite(total_loss.detach()).item()):
        raise FloatingPointError(f"non-finite total loss at lambda_rel={lambda_rel}")
    if not bool(torch.isfinite(seg_loss.detach()).item()):
        raise FloatingPointError(f"non-finite segmentation loss at lambda_rel={lambda_rel}")
    total_loss.backward()
    record = {
        "lambda_rel": float(lambda_rel),
        "logits": logits.detach().clone(),
        "logits_sha256": _tensor_sha256(logits),
        "segmentation_loss": seg_loss.detach().clone(),
        "segmentation_loss_value": float(seg_loss.detach().item()),
        "segmentation_valid_pixels": valid_pixels,
        "reliability_loss_value": float(reliability_loss.detach().item()),
        "total_loss_value": float(total_loss.detach().item()),
        "gradients": {
            name: (None if parameter.grad is None else parameter.grad.detach().clone())
            for name, parameter in model.named_parameters()
        },
    }
    del logits, seg_loss, reliability_loss, total_loss
    torch.cuda.empty_cache()
    return record


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------
def partition_parameters(model: nn.Module) -> Dict[str, Any]:
    """Apply the frozen grouping rule: ``reliability_estimator.`` versus everything else."""
    shared: List[str] = []
    head: List[str] = []
    group_counts: Dict[str, int] = {}
    for name, _ in model.named_parameters():
        if name.startswith(RELIABILITY_HEAD_PARAMETER_PREFIX):
            head.append(name)
            continue
        shared.append(name)
        group = _rank_parameter_name(name)
        group_counts[group] = group_counts.get(group, 0) + 1
    if not shared:
        raise RuntimeError("no shared model parameter was found; the grouping rule cannot be applied")
    if not head:
        raise RuntimeError(
            f"no parameter starts with {RELIABILITY_HEAD_PARAMETER_PREFIX!r}; the reliability head is missing"
        )
    return {
        "rule": (
            f"shared = every parameter whose name does not start with "
            f"'{RELIABILITY_HEAD_PARAMETER_PREFIX}'; "
            f"reliability head = every parameter whose name does start with it"
        ),
        "shared_parameter_count": len(shared),
        "shared_group_counts": group_counts,
        "reliability_head_parameter_names": head,
    }


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float) -> Dict[str, Any]:
    """Exact-first tensor comparison with an explicit tolerance decision."""
    if reference.shape != candidate.shape:
        return {
            "bitwise_equal": False,
            "shape_mismatch": True,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "within_tolerance": False,
        }
    finite = bool(torch.isfinite(reference).all().item()) and bool(torch.isfinite(candidate).all().item())
    if not finite:
        return {"bitwise_equal": False, "non_finite": True, "within_tolerance": False}
    difference = (reference - candidate).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    scale = reference.abs().clamp_min(RELATIVE_DEVIATION_FLOOR)
    max_rel = float((difference / scale).max().item()) if difference.numel() else 0.0
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_abs_difference": max_abs,
        "max_relative_difference": max_rel,
        "elements_over_tolerance": int((difference > (atol + rtol * reference.abs())).sum().item()),
        "within_tolerance": bool((difference <= (atol + rtol * reference.abs())).all().item()),
    }


def compare_gradient_sets(
    reference: Mapping[str, torch.Tensor | None],
    candidate: Mapping[str, torch.Tensor | None],
    names: Sequence[str],
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    """Per-parameter comparison of all shared gradients between the two sweeps."""
    records: List[Dict[str, Any]] = []
    violations: List[str] = []
    bitwise_differences: List[str] = []
    absent_in_both: List[str] = []
    observed_max_abs = 0.0
    observed_max_rel = 0.0
    for name in names:
        reference_gradient = reference.get(name)
        candidate_gradient = candidate.get(name)
        if reference_gradient is None or candidate_gradient is None:
            # ``None`` on both sides means the forward never touched the parameter, so the two
            # sweeps cannot disagree about it; that is recorded, not counted as a deviation.
            # ``None`` on exactly one side is a real disconnect and fails the check.
            both_absent = reference_gradient is None and candidate_gradient is None
            if both_absent:
                absent_in_both.append(name)
            else:
                violations.append(name)
            records.append(
                {
                    "name": name,
                    "group": _rank_parameter_name(name),
                    "status": "absent_in_both_runs" if both_absent else "present_in_only_one_run",
                    "present_at_lambda_rel_off": reference_gradient is not None,
                    "present_at_lambda_rel_on": candidate_gradient is not None,
                    "within_tolerance": both_absent,
                }
            )
            continue
        comparison = compare_tensors(reference_gradient, candidate_gradient, atol, rtol)
        record = {"name": name, "group": _rank_parameter_name(name), **comparison}
        record["reference_gradient_max_abs"] = float(reference_gradient.abs().max().item())
        record["reference_gradient_l2_norm"] = _l2_norm(reference_gradient)
        if comparison.get("bitwise_equal"):
            record["status"] = "bitwise_identical"
        elif comparison.get("within_tolerance"):
            record["status"] = "within_tolerance"
        else:
            record["status"] = "over_tolerance"
        observed_max_abs = max(observed_max_abs, float(record.get("max_abs_difference", 0.0)))
        observed_max_rel = max(observed_max_rel, float(record.get("max_relative_difference", 0.0)))
        if not comparison.get("bitwise_equal"):
            bitwise_differences.append(name)
        if not comparison.get("within_tolerance"):
            violations.append(name)
        records.append(record)
    return {
        "parameters_compared": len(names),
        "atol": float(atol),
        "rtol": float(rtol),
        "observed_max_abs_difference": observed_max_abs,
        "observed_max_relative_difference": observed_max_rel,
        "bitwise_identical": not bitwise_differences,
        "bitwise_difference_count": len(bitwise_differences),
        "bitwise_difference_names": bitwise_differences,
        "absent_in_both_runs_count": len(absent_in_both),
        "absent_in_both_runs_names": absent_in_both,
        "present_in_only_one_run_names": [
            name for name in violations if name not in absent_in_both
        ],
        "over_tolerance_count": len(violations),
        "over_tolerance_names": violations,
        "within_tolerance": not violations,
        "parameter_records": records,
    }


def reliability_head_gradient_report(
    gradients: Mapping[str, torch.Tensor | None],
    names: Sequence[str],
) -> Dict[str, Any]:
    """Report whether the reliability head has a gradient and whether it is identically zero."""
    present = [name for name in names if gradients.get(name) is not None]
    missing = [name for name in names if gradients.get(name) is None]
    per_parameter: List[Dict[str, Any]] = []
    total_square = 0.0
    max_abs = 0.0
    all_zero = True
    for name in names:
        gradient = gradients.get(name)
        if gradient is None:
            per_parameter.append({"name": name, "gradient_present": False})
            all_zero = False
            continue
        finite = bool(torch.isfinite(gradient).all().item())
        parameter_max_abs = float(gradient.abs().max().item())
        parameter_l2 = _l2_norm(gradient)
        total_square += parameter_l2 * parameter_l2
        max_abs = max(max_abs, parameter_max_abs)
        zero = parameter_max_abs == 0.0 and finite
        all_zero = all_zero and zero
        per_parameter.append(
            {
                "name": name,
                "gradient_present": True,
                "finite": finite,
                "max_abs": parameter_max_abs,
                "l2_norm": parameter_l2,
                "exactly_zero": zero,
            }
        )
    return {
        "parameter_count": len(names),
        "gradient_present_count": len(present),
        "gradient_missing_names": missing,
        "gradient_all_parameters_present": not missing,
        "gradient_max_abs": max_abs,
        "gradient_norm": math.sqrt(total_square),
        "gradient_exactly_zero": all_zero,
        "per_parameter": per_parameter,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    """Resolve and validate the CUDA device the qualification must run on."""
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "the gradient-path isolation qualification needs CUDA: the Ham decoder builds its NMF "
            "bases on the GPU, so a CPU run is impossible"
        )
    if device.index not in (None, 0):
        raise RuntimeError("the Ham decoder builds its NMF bases on device 0; use --device cuda:0")
    return device


def _base_report() -> Dict[str, Any]:
    """The evidence skeleton; it exists before anything can fail, so evidence survives aborts."""
    return {
        "schema_version": SCHEMA_VERSION,
        "check_id": CHECK_ID,
        "created_at_utc": _utc_now(),
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
            "executed_before": False,
            "execution_note": (
                "this tool had not been executed when it was written; its first real run is the "
                "qualification and must pass before any formal A2 v2 training"
            ),
        },
        "tolerances": {
            "atol": ATOL,
            "rtol": RTOL,
            "bitwise_equality_reported_first": True,
            "relative_deviation_floor": RELATIVE_DEVIATION_FLOOR,
        },
        "official_test_included": False,
        "warnings": [],
        "failures": [],
    }


def _environment_record(device: torch.device) -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device),
        "device": str(device),
    }


def print_summary(report: Mapping[str, Any]) -> None:
    """Print the headline numbers; absent on an early abort, in which case only failures print."""
    summary = report.get("summary") or {}
    if not summary:
        return
    print(
        f"  shared parameters compared: {summary['shared_parameters_compared']} "
        f"(over tolerance: {summary['shared_parameters_over_tolerance']})"
    )
    print(
        f"  segmentation loss off/on: {summary['segmentation_loss_off']!r} / "
        f"{summary['segmentation_loss_on']!r} "
        f"(bitwise equal: {summary['segmentation_loss_bitwise_equal']})"
    )
    print(
        "  reliability head gradient norm off/on: "
        f"{summary['reliability_head_gradient_norm_off']!r} / "
        f"{summary['reliability_head_gradient_norm_on']!r}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the qualification, always write evidence and always return a PASS/FAIL exit code."""
    args = parse_args(argv)
    report = _base_report()
    try:
        device = resolve_device(args.device)
        report["environment"] = _environment_record(device)
        qualify(args, device, report)
    except Exception as exc:  # evidence must survive any abort, including a failed precondition
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    failures = report["failures"]
    verdict = f"{CHECK_ID}: {'PASS' if not failures else 'FAIL'}"
    report["verdict"] = verdict
    report["passed"] = not failures
    report = _json_safe(report)
    report["verdict"] = verdict

    output_path = Path(args.output).resolve()
    write_json(output_path, report)
    print(verdict)
    print_summary(report)
    for failure in failures:
        print(f"  FAIL: {failure}")
    print(f"  evidence: {output_path}")
    return 0 if not failures else 1


def qualify(args: argparse.Namespace, device: torch.device, report: Dict[str, Any]) -> None:
    """Fill ``report`` with the two-sweep comparison; verdict and exit code stay in ``main``."""
    config = load_config(args.config)
    report["determinism"] = configure_runtime(device, RUNTIME_SEED)
    report["config"] = {
        "module": args.config,
        "config_file": str(Path(importlib.import_module(args.config).__file__).resolve()),
        "train_source": str(config.train_source),
        "dataset_path": str(config.dataset_path),
        "channel_order": str(config.channel_order),
        "normalization_identity": str(config.normalization_identity),
        "normalization_mean": [float(value) for value in config.norm_mean],
        "normalization_std": [float(value) for value in config.norm_std],
        "lambda_rel_off": LAMBDA_REL_OFF,
        "lambda_rel_on": float(args.lambda_rel_on),
        "lambda_rel_on_config_weight": float(config.mmfr_a2["reliability_head"].get("weight", 0.0)),
    }
    if float(args.lambda_rel_on) != float(config.mmfr_a2["reliability_head"].get("weight", 0.0)):
        report["warnings"].append(
            "the requested lambda_rel differs from config.mmfr_a2.reliability_head.weight; the "
            "qualification no longer sweeps the frozen training weight"
        )

    pretrained_path, pretrained_source = resolve_pretrained_path(config)
    report["pretrained"] = verify_pretrained_file(pretrained_path, EXPECTED_PRETRAINED_SHA256)
    report["pretrained"]["path_source"] = pretrained_source
    report["pretrained"]["environment_variable"] = {
        "name": PRETRAINED_ENVIRONMENT_VARIABLE,
        "set": bool(os.environ.get(PRETRAINED_ENVIRONMENT_VARIABLE)),
    }

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    model = build_model(config, criterion, device)
    report["model"] = model_record(model, config, device)
    report["pretrained"]["load_verification"] = verify_pretrained_load(
        model, checkpoint_backbone_state(pretrained_path)
    )
    report["parameters"] = partition_parameters(model)

    sample = load_training_sample(config, args.sample_index, args.preprocess_seed)
    corruption = build_corrupted_batch(
        config,
        sample,
        args.epoch,
        args.iteration,
        args.global_rank,
        CORRUPTION_DRAW_SEARCH_LIMIT,
        bool(args.require_corrupt_draw),
    )
    report["warnings"].extend(corruption["warnings"])
    tensors = move_batch_to_device(corruption["batch"], device)
    label = sample["label"].unsqueeze(0).to(device)
    metadata = corruption["metadata"]
    report["batch"] = {
        "source": {
            "role": "train-dev",
            "split_path": str(config.train_source),
            "split_samples": int(config.num_train_imgs),
            "sample_index": sample["index"],
            "sample_entry": sample["entry"],
            "sample_stem": sample["stem"],
            "sample_id_normalized": str(metadata["sample_id"]),
            "function_path": sample["function_path"],
            "input_files": sample["input_files"],
            "tensor_sha256": sample["tensor_sha256"],
            "preprocess": f"TrainPre(sign={sample['preprocess_sign']}) mirror/scale/crop_pad",
            "preprocess_seed": sample["preprocess_seed"],
        },
        "geometry": {
            "batch_size": 1,
            "num_workers": 0,
            "rgb_shape": list(tensors["rgb"].shape),
            "depth_shape": list(tensors["depth"].shape),
            "label_shape": list(label.shape),
        },
        "position": corruption["position"],
        "corruption": corruption["corruption"],
        "draw_selection": corruption["search"],
        "draw": {
            "clean": bool(metadata["clean"]),
            "num_specs": int(metadata["num_specs"]),
            "seed_words": [int(word) for word in metadata["seed_words"]],
            "curriculum_progress": float(metadata["curriculum_progress"]),
            "specs": _json_safe(metadata["specs"]),
            "target_composition": str(metadata["target_composition"]),
            "supervised_channels": list(metadata["supervised_channels"]),
            "depth_target_min": float(metadata["depth_target_min"]),
            "depth_target_mean": float(metadata["depth_target_mean"]),
            "depth_synthetic_reliability_mean": float(metadata["depth_synthetic_reliability_mean"]),
            "valid_fraction": float(metadata["valid_fraction"]),
            "depth_valid_pre_fraction": float(metadata["depth_valid_pre_fraction"]),
            "depth_valid_post_fraction": float(metadata["depth_valid_post_fraction"]),
            "natural_invalid_pixels": int(metadata["natural_invalid_pixels"]),
            "synthetic_missing_pixels": int(metadata["synthetic_missing_pixels"]),
            "post_corruption_invalid_pixels": int(metadata["post_corruption_invalid_pixels"]),
            "implicit_quality_pixels": int(metadata["implicit_quality_pixels"]),
        },
        "reliability_inputs": {
            "raw_rgb_shape": list(tensors["raw_rgb"].shape),
            "raw_depth_shape": list(tensors["raw_depth"].shape),
            "reliability_target_shape": list(tensors["reliability_target"].shape),
            "valid_mask_shape": list(tensors["valid_mask"].shape),
            "depth_valid_post_shape": list(tensors["depth_valid_post"].shape),
            "valid_mask_true_pixels": int(tensors["valid_mask"].sum().item()),
            "depth_valid_post_true_pixels": int(tensors["depth_valid_post"].sum().item()),
        },
    }

    state_snapshot = snapshot_model_state(model)
    rng_states = capture_rng_states(device)
    report["rng_control"] = {
        "reason": (
            "the Ham decoder rebuilds NMF bases with torch.rand on every forward and the backbone "
            "applies stochastic drop path in training mode, so both sweeps must consume the same "
            "CPU/CUDA RNG state"
        ),
        "cpu_rng_state_captured": True,
        "cuda_rng_state_captured": device.type == "cuda",
        "cpu_rng_state_replayed_before_each_forward": True,
        "cuda_rng_state_replayed_before_each_forward": device.type == "cuda",
        "decoder_rand_init": bool(report["model"]["decoder_rand_init"]),
        "captured_after_model_construction": True,
        "model_state_restored_between_runs": True,
        "model_state_restore_scope": "parameters and buffers (state_dict)",
    }

    head_off = run_forward_only(model, tensors, label, rng_states, device)
    restore_model_state(model, state_snapshot)
    lambda_off = run_forward_backward(model, tensors, label, LAMBDA_REL_OFF, rng_states, device)
    restore_model_state(model, state_snapshot)
    lambda_on = run_forward_backward(model, tensors, label, float(args.lambda_rel_on), rng_states, device)
    restore_model_state(model, state_snapshot)

    shared_names = [
        name
        for name, _ in model.named_parameters()
        if not name.startswith(RELIABILITY_HEAD_PARAMETER_PREFIX)
    ]
    head_names = list(report["parameters"]["reliability_head_parameter_names"])

    gradient_comparison = compare_gradient_sets(
        lambda_off["gradients"], lambda_on["gradients"], shared_names, ATOL, RTOL
    )
    head_off_report = reliability_head_gradient_report(lambda_off["gradients"], head_names)
    head_on_report = reliability_head_gradient_report(lambda_on["gradients"], head_names)
    logits_head_off_vs_off = compare_tensors(head_off["logits"], lambda_off["logits"], ATOL, RTOL)
    logits_off_vs_on = compare_tensors(lambda_off["logits"], lambda_on["logits"], ATOL, RTOL)
    segmentation_loss_comparison = compare_tensors(
        lambda_off["segmentation_loss"].reshape(-1), lambda_on["segmentation_loss"].reshape(-1), ATOL, RTOL
    )
    segmentation_loss_comparison["equal"] = bool(
        torch.equal(lambda_off["segmentation_loss"], lambda_on["segmentation_loss"])
    )

    report["runs"] = {
        "reliability_head_off": {
            "role": head_off["role"],
            "segmentation_loss": head_off["segmentation_loss_value"],
            "segmentation_valid_pixels": head_off["segmentation_valid_pixels"],
            "logits_sha256": head_off["logits_sha256"],
            "logits_shape": list(head_off["logits"].shape),
        },
        f"lambda_rel_{LAMBDA_REL_OFF}": {
            "lambda_rel": lambda_off["lambda_rel"],
            "segmentation_loss": lambda_off["segmentation_loss_value"],
            "reliability_loss": lambda_off["reliability_loss_value"],
            "total_loss": lambda_off["total_loss_value"],
            "segmentation_valid_pixels": lambda_off["segmentation_valid_pixels"],
            "logits_sha256": lambda_off["logits_sha256"],
        },
        f"lambda_rel_{float(args.lambda_rel_on)}": {
            "lambda_rel": lambda_on["lambda_rel"],
            "segmentation_loss": lambda_on["segmentation_loss_value"],
            "reliability_loss": lambda_on["reliability_loss_value"],
            "total_loss": lambda_on["total_loss_value"],
            "segmentation_valid_pixels": lambda_on["segmentation_valid_pixels"],
            "logits_sha256": lambda_on["logits_sha256"],
        },
    }
    report["shared_gradient_comparison"] = gradient_comparison
    report["reliability_head_gradients"] = {
        f"lambda_rel_{LAMBDA_REL_OFF}": head_off_report,
        f"lambda_rel_{float(args.lambda_rel_on)}": head_on_report,
    }
    report["segmentation_consistency"] = {
        "description": (
            "segmentation logits and loss must not depend on whether the reliability auxiliary "
            "branch is active"
        ),
        "logits_head_off_vs_lambda_rel_off": logits_head_off_vs_off,
        "logits_lambda_rel_off_vs_on": logits_off_vs_on,
        "segmentation_loss": segmentation_loss_comparison,
        "segmentation_loss_off": lambda_off["segmentation_loss_value"],
        "segmentation_loss_on": lambda_on["segmentation_loss_value"],
    }

    failures: List[str] = report["failures"]
    if gradient_comparison["over_tolerance_count"]:
        failures.append(
            f"{gradient_comparison['over_tolerance_count']} shared parameter(s) changed gradient beyond "
            f"atol={ATOL} rtol={RTOL}: {', '.join(gradient_comparison['over_tolerance_names'][:8])}"
        )
    if gradient_comparison["absent_in_both_runs_count"]:
        report["warnings"].append(
            f"{gradient_comparison['absent_in_both_runs_count']} shared parameter(s) received no gradient "
            "in either sweep, so they could not be compared: "
            + ", ".join(gradient_comparison["absent_in_both_runs_names"][:8])
        )
    if not gradient_comparison["bitwise_identical"]:
        report["warnings"].append(
            f"{gradient_comparison['bitwise_difference_count']} shared parameter(s) differ bitwise but stay "
            "inside the declared tolerance"
        )
    if not head_off_report["gradient_all_parameters_present"]:
        report["warnings"].append("the reliability head has no gradient tensor at lambda_rel=0")
    if not head_off_report["gradient_exactly_zero"]:
        failures.append("the reliability head carries a non-zero gradient at lambda_rel=0")
    if not head_on_report["gradient_all_parameters_present"]:
        failures.append("the reliability head has no gradient at lambda_rel>0; the auxiliary loss is disconnected")
    elif head_on_report["gradient_exactly_zero"]:
        failures.append("the reliability head gradient is identically zero at lambda_rel>0")
    if not logits_head_off_vs_off["within_tolerance"]:
        failures.append("segmentation logits differ between the reliability-head-off forward and lambda_rel=0")
    if not logits_off_vs_on["within_tolerance"]:
        failures.append("segmentation logits differ between lambda_rel=0 and lambda_rel>0")
    if not segmentation_loss_comparison["within_tolerance"]:
        failures.append("the segmentation loss differs between the two sweeps")

    if int(lambda_off["segmentation_valid_pixels"]) < 1:
        failures.append(
            "the selected sample has no supervised segmentation pixel; the comparison would be vacuous, "
            "choose another --sample-index"
        )
    if gradient_comparison["observed_max_abs_difference"] == 0.0 and segment_shared_is_trivial(
        lambda_off["gradients"], shared_names
    ):
        failures.append(
            "every shared gradient is exactly zero; the selected sample does not exercise the shared path"
        )
    if not torch.isfinite(lambda_off["logits"]).all() or not torch.isfinite(lambda_on["logits"]).all():
        failures.append("non-finite segmentation logits")

    report["summary"] = {
        "shared_parameters_compared": gradient_comparison["parameters_compared"],
        "shared_parameters_over_tolerance": gradient_comparison["over_tolerance_count"],
        "shared_gradients_bitwise_identical": gradient_comparison["bitwise_identical"],
        "shared_gradients_absent_in_both_runs": gradient_comparison["absent_in_both_runs_count"],
        "shared_max_abs_difference": gradient_comparison["observed_max_abs_difference"],
        "shared_max_relative_difference": gradient_comparison["observed_max_relative_difference"],
        "segmentation_loss_off": lambda_off["segmentation_loss_value"],
        "segmentation_loss_on": lambda_on["segmentation_loss_value"],
        "segmentation_loss_bitwise_equal": segmentation_loss_comparison["equal"],
        "logits_head_off_bitwise_equal_lambda_rel_off": logits_head_off_vs_off.get("bitwise_equal", False),
        "logits_lambda_rel_off_bitwise_equal_on": logits_off_vs_on.get("bitwise_equal", False),
        "reliability_head_gradient_norm_off": head_off_report["gradient_norm"],
        "reliability_head_gradient_norm_on": head_on_report["gradient_norm"],
        "reliability_head_gradients_exactly_zero_off": head_off_report["gradient_exactly_zero"],
        "reliability_head_gradients_present_on": head_on_report["gradient_all_parameters_present"],
    }


def segment_shared_is_trivial(
    gradients: Mapping[str, torch.Tensor | None],
    names: Sequence[str],
) -> bool:
    """True when no shared parameter received a usable gradient at all."""
    for name in names:
        gradient = gradients.get(name)
        if gradient is not None and bool(torch.isfinite(gradient).all().item()) and float(
            gradient.abs().max().item()
        ) > 0.0:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
