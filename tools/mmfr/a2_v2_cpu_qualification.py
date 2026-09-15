#!/usr/bin/env python3
"""Re-runnable CPU qualification of the frozen MMFR-A2 v2 Depth-corruption batch helper.

Purpose
-------
The ``MMFR-A2-train-integration-v2`` batch helper
(:func:`utils.dataloader.mmfr_training_v2.build_mmfr_training_batch_v2`) was qualified by a
one-shot inline CPU check that reported ``47`` assertions and ``0`` failures and was then
deleted, so an external auditor could not re-run it. This script freezes that check into a
re-runnable tool: it rebuilds the same frozen fixture from a real ``train-dev`` sample,
calls the helper on the clean, the corrupted and the default branch, re-asserts every
historical assertion family, and writes a machine-readable JSON report containing the
script hash, the hashed code identity, the sample identity and every assertion result.

History
-------
本脚本是对历史 47 项一次性检查的可复跑固化版本，历史结果保留为 47/0。历史脚本没有留在
工作区，本脚本不覆盖也不替换该历史结论，只提供一份可复跑的等价证据。

Run
---
    python -m py_compile tools/mmfr/a2_v2_cpu_qualification.py
    python tools/mmfr/a2_v2_cpu_qualification.py

CPU only
--------
The fixture tensors are built on the CPU, the helper is called on the CPU and no CUDA API
is used. The report records that CUDA stayed uninitialized for the whole run.

Fixture (frozen rule, identical to the historical one-shot check)
----------------------------------------------------------------
* sample ``sample_index=0`` of ``data/splits/MUSeg/dev-v1/train-dev.txt``;
* RGB from ``<data root>/MUSeg_DFormer/RGB/<stem>.jpg``, Depth from
  ``<data root>/MUSeg_DFormer/Depth/<stem>.*`` read with ``cv2.IMREAD_UNCHANGED`` (a
  three-channel Depth keeps channel 0);
* both resized to ``480 x 640`` (RGB ``INTER_LINEAR``, Depth ``INTER_NEAREST`` then
  uint8), no random augmentation;
* a native-invalid Depth block is injected at rows ``10:20``, columns ``30:80``;
* the right ``32`` columns and the bottom ``32`` rows are padding;
* RGB is normalized with the config ``norm_mean``/``norm_std``, Depth with the fixed
  ``0.48``/``0.28``, then both are set to the exact normalized value ``0.0`` on the pad;
* the resulting tensors are RGB ``[1, 3, 480, 640]`` and Depth ``[1, 1, 480, 640]``
  float32.

Scope and failure policy
------------------------
The script only reads the repository and the MUSeg dataset and never modifies the code
under test. If a measured value disagrees with the frozen expectation, the assertion
fails, the measured value is written into the report and the process exits non-zero.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Sequence

# The console on Windows may default to a legacy code page; every line this script prints
# is plain ASCII, and the reconfigure keeps an unexpected non-ASCII traceback printable.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - best effort only
        pass

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_PATH)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen fixture rule and frozen expectations
# ---------------------------------------------------------------------------
CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v2"
BATCH_HELPER_MODULE = "utils.dataloader.mmfr_training_v2"
CORRUPTION_MODULE = "utils.dataloader.multimodal_failure_v2"
SAMPLE_INDEX = 0
CROP_HEIGHT = 480
CROP_WIDTH = 640
PAD_BOTTOM = 32
PAD_RIGHT = 32
INJECT_ROW_START = 10
INJECT_ROW_STOP = 20
INJECT_COL_START = 30
INJECT_COL_STOP = 80
DEPTH_MEAN = 0.48
DEPTH_STD = 0.28
EPOCH = 500
ITERATION = 0
GLOBAL_RANK = 0
SAMPLE_SLOT = 0
BRANCH_ITERATIONS = 40
HISTORICAL_BRANCH_SPLIT = (13, 27)

EXPECTED_RUN_ID = "museg-dformerv2-s-mmfr-a2-depth-corruption-v2"
EXPECTED_ANALYSIS_IDENTITY = "MMFR-A2-depth-corruption-train-v2"
EXPECTED_PROTOCOL = "MMFR-A2-train-integration-v2"
EXPECTED_CORRUPTION_BASIS = "MMFR-A1-corruption-basis-v2"
EXPECTED_SUPERVISED_CHANNELS = ["depth"]
EXPECTED_TARGET_COMPOSITION = "R_depth_sup(p) = depth_valid_pre(p) * R_depth_synthetic(p)"
EXPECTED_COUNTS = {
    "valid_pixels": 272384,
    "natural_invalid_pixels": 267802,
    "synthetic_missing_pixels": 871,
    "newly_valid_pixels": 132827,
    "post_corruption_invalid_pixels": 135846,
}

REQUIRED_OUTPUT_KEYS = (
    "rgb",
    "depth",
    "raw_rgb",
    "raw_depth",
    "reliability_target",
    "valid_mask",
    "depth_valid_pre",
    "depth_valid_post",
    "protocol",
    "supervised_channels",
    "metadata",
)
V1_ONLY_KEYS = ("depth_valid",)
METADATA_COUNT_KEYS = (
    "valid_pixels",
    "natural_invalid_pixels",
    "synthetic_missing_pixels",
    "post_corruption_invalid_pixels",
    "newly_valid_pixels",
    "implicit_quality_pixels",
)

CODE_IDENTITY_FILES = (
    ("utils/dataloader/mmfr_training_v2.py", True),
    ("utils/dataloader/multimodal_failure_v2.py", True),
    ("local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common_v2.py", True),
    ("local_configs/MUSeg/DFormerv2_S_MMFR_A2_DepthCorrupt_v2.py", True),
    ("utils/dataloader/mmfr_training.py", False),
    ("utils/dataloader/multimodal_failure.py", False),
)
TENSOR_KEYS = (
    "rgb",
    "depth",
    "raw_rgb",
    "raw_depth",
    "reliability_target",
    "valid_mask",
    "depth_valid_pre",
    "depth_valid_post",
)

FAMILY_TITLES = {
    "0": "fixture and dataset identity",
    "1": "v2 config import, binding and frozen identity fields",
    "2": "output contract (keys, shapes, dtypes, protocol fields)",
    "3": "clean path is an exact no-op on the model input",
    "4": "clean Depth target equals native Depth validity",
    "5": "RGB reliability channel is exactly 1",
    "6": "pad semantics (mask, normalized, raw, target)",
    "7": "corrupted path changes Depth only",
    "8": "corrupted Depth target semantics",
    "9": "range and finiteness",
    "10": "validity masks are subsets of the geometry mask",
    "11": "invalidity accounting closes and is recomputable",
    "12": "determinism under the frozen seed key",
    "13": "clean and corrupt branches are both reachable at p_clean=0.25",
    "14": "frozen measured pixel counts on this fixture",
    "15": "CPU-only execution",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_raw(path: str) -> str:
    """SHA-256 of the raw bytes of ``path``."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def sha256_lf(path: str) -> str:
    """SHA-256 of ``path`` after normalizing CRLF to LF (the project's text hash)."""
    with open(path, "rb") as handle:
        data = handle.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/torch containers into plain JSON values."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def tensor_np(tensor: torch.Tensor) -> np.ndarray:
    """Return a CPU numpy view of a produced tensor without touching CUDA."""
    return tensor.detach().cpu().numpy()


def rect_mask(height: int, width: int) -> np.ndarray:
    """Boolean geometry mask of the frozen crop/pad rule (pad is right and bottom)."""
    mask = np.zeros((height, width), dtype=bool)
    mask[: height - PAD_BOTTOM, : width - PAD_RIGHT] = True
    return mask


class CheckRegistry:
    """Ordered PASS/FAIL registry; every entry is written into the JSON report."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self.entries: List[Dict[str, Any]] = []

    def check(self, family: str, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.entries.append({"family": family, "name": name, "ok": ok, "detail": str(detail)})
        self._emit("{:4s}  {:>2s}  {:<56s}  {}".format("PASS" if ok else "FAIL", family, name, detail))
        return ok

    def failures(self) -> List[Dict[str, Any]]:
        return [entry for entry in self.entries if not entry["ok"]]

    def count_for(self, family: str) -> int:
        return sum(1 for entry in self.entries if entry["family"] == family)


# ---------------------------------------------------------------------------
# Frozen fixture
# ---------------------------------------------------------------------------
def load_and_resize_depth(depth_root: str, stem: str) -> Dict[str, Any]:
    """Resolve the single Depth file of ``stem`` and return its decoded original."""
    matches = sorted(glob.glob(os.path.join(depth_root, stem + ".*")))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Depth file for {stem!r}, got {matches!r}")
    path = matches[0]
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"cv2 could not read {path!r}")
    channels = 1 if raw.ndim == 2 else int(raw.shape[2])
    plane = raw if raw.ndim == 2 else raw[:, :, 0]
    if raw.dtype != np.uint8:
        raise RuntimeError(f"Depth {path!r} must be uint8 (Depth8), got {raw.dtype}")
    if channels == 3 and not (
        np.array_equal(raw[:, :, 0], raw[:, :, 1]) and np.array_equal(raw[:, :, 0], raw[:, :, 2])
    ):
        raise RuntimeError(f"Depth {path!r} has three non-identical channels")
    return {
        "path": path,
        "sha256": sha256_raw(path),
        "decoded_shape": list(raw.shape),
        "decoded_dtype": str(raw.dtype),
        "decoded_channels": channels,
        "plane": plane,
    }


def build_fixture(config: Any) -> Dict[str, Any]:
    """Build the frozen normalized fixture tensors exactly as the historical check did."""
    split_path = str(config.train_source)
    with open(split_path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if SAMPLE_INDEX >= len(lines):
        raise RuntimeError(f"sample index {SAMPLE_INDEX} is outside {split_path!r}")

    sample_id = lines[SAMPLE_INDEX]
    rgb_root = str(config.rgb_root_folder)
    depth_root = str(config.x_root_folder)
    rgb_relative = sample_id.split("/", 1)[1] if sample_id.split("/", 1)[0].upper() == "RGB" else sample_id
    rgb_path = os.path.join(rgb_root, rgb_relative.replace("/", os.sep))
    stem = os.path.splitext(os.path.basename(sample_id))[0]

    rgb_raw = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if rgb_raw is None:
        raise RuntimeError(f"cv2 could not read {rgb_path!r}")
    depth_info = load_and_resize_depth(depth_root, stem)

    rgb_resized = cv2.resize(rgb_raw, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(
        depth_info["plane"], (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)
    zero_before_injection = int(np.count_nonzero(depth_resized == 0))
    block = (slice(INJECT_ROW_START, INJECT_ROW_STOP), slice(INJECT_COL_START, INJECT_COL_STOP))
    injection_block_pixels = int(depth_resized[block].size)
    injection_already_zero_pixels = int(np.count_nonzero(depth_resized[block] == 0))

    depth_injected = depth_resized.copy()
    depth_injected[block] = np.uint8(0)
    injection_newly_zeroed_pixels = int(np.count_nonzero(depth_injected == 0)) - zero_before_injection

    rgb_mean = np.asarray(config.norm_mean, dtype=np.float64).reshape(3)
    rgb_std = np.asarray(config.norm_std, dtype=np.float64).reshape(3)
    rgb_normalized = (rgb_resized.astype(np.float64) / 255.0 - rgb_mean.reshape(1, 1, 3)) / rgb_std.reshape(1, 1, 3)
    depth_normalized = (depth_injected.astype(np.float64) / 255.0 - DEPTH_MEAN) / DEPTH_STD

    rgb_tensor = np.ascontiguousarray(rgb_normalized.transpose(2, 0, 1)[None].astype(np.float32))
    depth_tensor = np.ascontiguousarray(depth_normalized[None, None].astype(np.float32))
    # The pad is an exact zero in normalized space for both modalities; the helper detects
    # the geometry mask from that value, so it must be written after normalization.
    rgb_tensor[:, :, CROP_HEIGHT - PAD_BOTTOM :, :] = 0.0
    rgb_tensor[:, :, :, CROP_WIDTH - PAD_RIGHT :] = 0.0
    depth_tensor[:, :, CROP_HEIGHT - PAD_BOTTOM :, :] = 0.0
    depth_tensor[:, :, :, CROP_WIDTH - PAD_RIGHT :] = 0.0

    valid = rect_mask(CROP_HEIGHT, CROP_WIDTH)
    depth_plane = depth_tensor[0, 0]
    return {
        "sample_id_in_split": sample_id,
        "sample_index": SAMPLE_INDEX,
        "split_path": split_path,
        "split_sha256": sha256_raw(split_path),
        "split_line_count": len(lines),
        "rgb_path": rgb_path,
        "rgb_sha256": sha256_raw(rgb_path),
        "rgb_decoded_shape": list(rgb_raw.shape),
        "rgb_decoded_dtype": str(rgb_raw.dtype),
        "depth_path": depth_info["path"],
        "depth_sha256": depth_info["sha256"],
        "depth_decoded_shape": depth_info["decoded_shape"],
        "depth_decoded_dtype": depth_info["decoded_dtype"],
        "depth_decoded_channels": depth_info["decoded_channels"],
        "rgb_tensor": rgb_tensor,
        "depth_tensor": depth_tensor,
        "valid_expected": valid,
        "native_valid_expected": (depth_plane > 0.0) & valid,
        "depth_zero_pixels_resized": zero_before_injection,
        "depth_zero_pixels_valid_region": int(np.count_nonzero(depth_plane[valid] == 0.0)),
        "injection_block_rows": [INJECT_ROW_START, INJECT_ROW_STOP],
        "injection_block_cols": [INJECT_COL_START, INJECT_COL_STOP],
        "injection_block_pixels": injection_block_pixels,
        "injection_block_already_zero_pixels": injection_already_zero_pixels,
        "injection_block_newly_zeroed_pixels": injection_newly_zeroed_pixels,
    }


def call_helper(
    helper: Callable[..., Dict[str, Any]],
    config: Any,
    fixture: Mapping[str, Any],
    p_clean: float,
    iteration: int,
) -> Dict[str, Any]:
    """Call the frozen helper with the frozen key; CPU tensors only."""
    return helper(
        torch.from_numpy(fixture["rgb_tensor"]),
        torch.from_numpy(fixture["depth_tensor"]),
        [fixture["sample_id_in_split"]],
        epoch=EPOCH,
        iteration=iteration,
        niters_per_epoch=int(config.niters_per_epoch),
        nepochs=int(config.nepochs),
        rgb_mean=config.norm_mean,
        rgb_std=config.norm_std,
        corruption_seed=int(config.mmfr_a2["corruption"]["seed"]),
        global_rank=GLOBAL_RANK,
        p_clean=p_clean,
        max_specs=int(config.mmfr_a2["corruption"]["max_specs"]),
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def run_checks(registry: CheckRegistry, config: Any, modules: Mapping[str, Any]) -> Dict[str, Any]:
    """Run every assertion family and return the collected evidence."""
    helper = modules["helper"]
    helper_module = modules["helper_module"]
    corruption = modules["corruption"]
    fixture = build_fixture(config)

    unit = fixture["valid_expected"]
    clean = call_helper(helper, config, fixture, p_clean=1.0, iteration=ITERATION)
    corrupt = call_helper(helper, config, fixture, p_clean=0.0, iteration=ITERATION)
    default_first = call_helper(helper, config, fixture, p_clean=float(config.mmfr_a2["corruption"]["p_clean"]), iteration=ITERATION)
    default_second = call_helper(helper, config, fixture, p_clean=float(config.mmfr_a2["corruption"]["p_clean"]), iteration=ITERATION)

    # -- family 0: fixture and dataset identity -----------------------------
    expected_split_sha = str(config.expected_split_sha256["train"])
    registry.check(
        "0",
        "fixture.split_file_sha256_matches_config",
        fixture["split_sha256"] == expected_split_sha,
        f"sha256={fixture['split_sha256']} expected={expected_split_sha}",
    )
    expected_train_samples = int(config.expected_split_samples["train"])
    registry.check(
        "0",
        "fixture.split_line_count_matches_config",
        fixture["split_line_count"] == expected_train_samples,
        f"lines={fixture['split_line_count']} expected={expected_train_samples}",
    )
    registry.check(
        "0",
        "fixture.depth_file_resolved_unique_uint8",
        fixture["depth_decoded_dtype"] == "uint8" and fixture["depth_decoded_channels"] in (1, 3),
        f"path={fixture['depth_path']} shape={fixture['depth_decoded_shape']} "
        f"dtype={fixture['depth_decoded_dtype']} channels={fixture['depth_decoded_channels']}",
    )
    registry.check(
        "0",
        "fixture.rgb_decoded_uint8_three_channel",
        fixture["rgb_decoded_dtype"] == "uint8" and fixture["rgb_decoded_shape"][-1] == 3,
        f"path={fixture['rgb_path']} shape={fixture['rgb_decoded_shape']} dtype={fixture['rgb_decoded_dtype']}",
    )

    # -- family 1: config identity -----------------------------------------
    mmfr = config.mmfr_a2
    frozen = mmfr.get("frozen") or {}
    registry.check(
        "1",
        "config.module_import_and_frozen_guard_accepted_identity",
        config.run_id == EXPECTED_RUN_ID and mmfr.get("corruption") and mmfr.get("reliability_head"),
        f"module={CONFIG_MODULE} imported, run_id={config.run_id!r}, guard accepted the binding",
    )
    registry.check(
        "1",
        "config.run_id_matches_frozen",
        str(config.run_id) == EXPECTED_RUN_ID,
        f"run_id={config.run_id!r} expected={EXPECTED_RUN_ID!r}",
    )
    registry.check(
        "1",
        "config.protocol_matches_frozen",
        str(mmfr.get("protocol")) == EXPECTED_PROTOCOL,
        f"protocol={mmfr.get('protocol')!r} expected={EXPECTED_PROTOCOL!r}",
    )
    registry.check(
        "1",
        "config.corruption_basis_matches_frozen",
        str((mmfr.get("corruption") or {}).get("basis")) == EXPECTED_CORRUPTION_BASIS,
        f"basis={(mmfr.get('corruption') or {}).get('basis')!r} expected={EXPECTED_CORRUPTION_BASIS!r}",
    )
    supervised = list((mmfr.get("reliability_head") or {}).get("supervised_channels") or ())
    registry.check(
        "1",
        "config.supervised_channels_matches_frozen",
        supervised == EXPECTED_SUPERVISED_CHANNELS,
        f"supervised_channels={supervised!r} expected={EXPECTED_SUPERVISED_CHANNELS!r}",
    )
    registry.check(
        "1",
        "config.mode_is_depth_corruption",
        str(mmfr.get("mode")) == "depth-corruption",
        f"mode={mmfr.get('mode')!r}",
    )
    registry.check(
        "1",
        "config.frozen_block_binding_matches",
        str(frozen.get("run_id")) == EXPECTED_RUN_ID
        and str(frozen.get("analysis_identity")) == EXPECTED_ANALYSIS_IDENTITY
        and str(frozen.get("mode")) == "depth-corruption",
        f"frozen.run_id={frozen.get('run_id')!r} frozen.analysis_identity={frozen.get('analysis_identity')!r}",
    )
    registry.check(
        "1",
        "helper.module_constants_match_config",
        str(helper_module.PROTOCOL_ID) == EXPECTED_PROTOCOL
        and str(helper_module.CORRUPTION_BASIS) == EXPECTED_CORRUPTION_BASIS
        and list(helper_module.SUPERVISED_CHANNELS) == EXPECTED_SUPERVISED_CHANNELS
        and str(helper_module.TARGET_COMPOSITION) == EXPECTED_TARGET_COMPOSITION,
        f"{BATCH_HELPER_MODULE}: protocol={helper_module.PROTOCOL_ID!r} "
        f"basis={helper_module.CORRUPTION_BASIS!r} supervised={list(helper_module.SUPERVISED_CHANNELS)!r}",
    )
    registry.check(
        "1",
        "corruption.module_protocol_matches_basis",
        str(corruption.PROTOCOL_ID) == EXPECTED_CORRUPTION_BASIS,
        f"{CORRUPTION_MODULE}: protocol={corruption.PROTOCOL_ID!r} expected={EXPECTED_CORRUPTION_BASIS!r}",
    )
    corruption_record = mmfr.get("corruption") or {}
    registry.check(
        "1",
        "config.corruption_seed_and_curriculum_match_frozen_values",
        int(corruption_record.get("seed")) == int(helper_module.CORRUPTION_SEED)
        and float(corruption_record.get("p_clean")) == float(helper_module.CLEAN_PROBABILITY)
        and int(corruption_record.get("max_specs")) == int(helper_module.MAX_SPECS)
        and str(corruption_record.get("target_composition")) == EXPECTED_TARGET_COMPOSITION,
        f"seed={corruption_record.get('seed')} p_clean={corruption_record.get('p_clean')} "
        f"max_specs={corruption_record.get('max_specs')} helper.CORRUPTION_SEED={helper_module.CORRUPTION_SEED}",
    )

    # -- family 2: output contract -----------------------------------------
    keys = set(corrupt.keys())
    missing = sorted(set(REQUIRED_OUTPUT_KEYS) - keys)
    registry.check(
        "2",
        "contract.required_keys_present",
        not missing,
        f"keys={sorted(keys)} missing={missing}",
    )
    leaked = sorted(set(V1_ONLY_KEYS) & keys)
    registry.check("2", "contract.v1_only_key_absent", not leaked, f"leaked={leaked}")
    target = corrupt["reliability_target"]
    valid_mask = corrupt["valid_mask"]
    pre_mask = corrupt["depth_valid_pre"]
    post_mask = corrupt["depth_valid_post"]
    registry.check(
        "2",
        "contract.rgb_shape_and_dtype",
        tuple(corrupt["rgb"].shape) == (1, 3, CROP_HEIGHT, CROP_WIDTH) and corrupt["rgb"].dtype == torch.float32,
        f"shape={tuple(corrupt['rgb'].shape)} dtype={corrupt['rgb'].dtype}",
    )
    registry.check(
        "2",
        "contract.depth_shape_and_dtype",
        tuple(corrupt["depth"].shape) == (1, 1, CROP_HEIGHT, CROP_WIDTH) and corrupt["depth"].dtype == torch.float32,
        f"shape={tuple(corrupt['depth'].shape)} dtype={corrupt['depth'].dtype}",
    )
    registry.check(
        "2",
        "contract.raw_rgb_shape_and_dtype",
        tuple(corrupt["raw_rgb"].shape) == (1, 3, CROP_HEIGHT, CROP_WIDTH) and corrupt["raw_rgb"].dtype == torch.float32,
        f"shape={tuple(corrupt['raw_rgb'].shape)} dtype={corrupt['raw_rgb'].dtype}",
    )
    registry.check(
        "2",
        "contract.raw_depth_shape_and_dtype",
        tuple(corrupt["raw_depth"].shape) == (1, 1, CROP_HEIGHT, CROP_WIDTH) and corrupt["raw_depth"].dtype == torch.float32,
        f"shape={tuple(corrupt['raw_depth'].shape)} dtype={corrupt['raw_depth'].dtype}",
    )
    registry.check(
        "2",
        "contract.reliability_target_shape_and_dtype",
        tuple(target.shape) == (1, 2, CROP_HEIGHT, CROP_WIDTH) and target.dtype == torch.float32,
        f"shape={tuple(target.shape)} dtype={target.dtype}",
    )
    registry.check(
        "2",
        "contract.valid_mask_shape_and_dtype",
        tuple(valid_mask.shape) == (1, 1, CROP_HEIGHT, CROP_WIDTH) and valid_mask.dtype == torch.bool,
        f"shape={tuple(valid_mask.shape)} dtype={valid_mask.dtype}",
    )
    registry.check(
        "2",
        "contract.depth_valid_pre_shape_and_dtype",
        tuple(pre_mask.shape) == (1, 1, CROP_HEIGHT, CROP_WIDTH) and pre_mask.dtype == torch.bool,
        f"shape={tuple(pre_mask.shape)} dtype={pre_mask.dtype}",
    )
    registry.check(
        "2",
        "contract.depth_valid_post_shape_and_dtype",
        tuple(post_mask.shape) == (1, 1, CROP_HEIGHT, CROP_WIDTH) and post_mask.dtype == torch.bool,
        f"shape={tuple(post_mask.shape)} dtype={post_mask.dtype}",
    )
    registry.check(
        "2",
        "contract.protocol_value",
        str(corrupt["protocol"]) == EXPECTED_PROTOCOL,
        f"protocol={corrupt['protocol']!r}",
    )
    registry.check(
        "2",
        "contract.supervised_channels_value",
        list(corrupt["supervised_channels"]) == EXPECTED_SUPERVISED_CHANNELS,
        f"supervised_channels={corrupt['supervised_channels']!r}",
    )
    record = corrupt["metadata"][0]
    required_record_keys = {"sample_slot", "sample_id", "seed_words", "curriculum_progress", "clean", "specs"} | set(
        METADATA_COUNT_KEYS
    )
    registry.check(
        "2",
        "contract.metadata_record_keys",
        isinstance(corrupt["metadata"], list)
        and len(corrupt["metadata"]) == 1
        and not (required_record_keys - set(record.keys())),
        f"missing={sorted(required_record_keys - set(record.keys()))}",
    )
    registry.check(
        "2",
        "contract.geometry_valid_mask_identical_across_branches",
        torch.equal(valid_mask, clean["valid_mask"]),
        "clean and corrupt calls produced the same geometry valid_mask",
    )
    registry.check(
        "2",
        "contract.depth_valid_pre_identical_across_branches",
        torch.equal(pre_mask, clean["depth_valid_pre"]),
        "pre-corruption validity does not depend on the branch",
    )

    # -- family 3: clean exact no-op ---------------------------------------
    registry.check(
        "3",
        "clean.rgb_bitwise_noop",
        torch.equal(clean["rgb"], torch.from_numpy(fixture["rgb_tensor"])),
        "torch.equal(clean['rgb'], rgb_input) is True",
    )
    registry.check(
        "3",
        "clean.depth_bitwise_noop",
        torch.equal(clean["depth"], torch.from_numpy(fixture["depth_tensor"])),
        "torch.equal(clean['depth'], depth_input) is True",
    )

    # -- family 4: clean Depth target --------------------------------------
    clean_target = tensor_np(clean["reliability_target"])[0, 1]
    native_valid = tensor_np(clean["depth_valid_pre"])[0, 0]
    registry.check(
        "4",
        "clean.target_equals_native_validity_on_valid_region",
        bool(np.array_equal(clean_target[unit], native_valid[unit].astype(np.float32))),
        f"valid_pixels={int(unit.sum())} native_valid={int(native_valid[unit].sum())}",
    )
    natural = unit & ~native_valid
    registry.check(
        "4",
        "clean.target_exactly_zero_on_native_invalid",
        bool(np.all(clean_target[natural] == 0.0)),
        f"natural_invalid_pixels={int(natural.sum())} target_max_there={float(clean_target[natural].max()):.6g}",
    )
    block = (slice(INJECT_ROW_START, INJECT_ROW_STOP), slice(INJECT_COL_START, INJECT_COL_STOP))
    clean_raw_depth = tensor_np(clean["raw_depth"])[0, 0]
    block_zero = bool(np.all(clean_raw_depth[block] == 0.0))
    block_inside = bool(np.all(unit[block]))
    block_counted = bool(np.all(natural[block]))
    registry.check(
        "4",
        "injection.invalid_block_present_and_natively_invalid",
        block_zero and block_inside and block_counted,
        f"block_pixels={fixture['injection_block_pixels']} zero_in_raw_depth={block_zero} "
        f"inside_valid_mask={block_inside} counted_as_natural_invalid={block_counted} "
        f"already_zero_before_injection={fixture['injection_block_already_zero_pixels']} "
        f"newly_zeroed_by_injection={fixture['injection_block_newly_zeroed_pixels']}",
    )
    registry.check(
        "4",
        "clean.target_is_binary_on_valid_region",
        bool(np.isin(clean_target[unit], (0.0, 1.0)).all()),
        f"distinct_values={sorted(set(clean_target[unit].tolist()))[:4]}",
    )

    # -- family 5: RGB reliability channel ---------------------------------
    clean_rgb_target = tensor_np(clean["reliability_target"])[0, 0]
    corrupt_target = tensor_np(target)[0]
    registry.check(
        "5",
        "rgb_channel.exactly_one_on_clean_path",
        bool(clean_rgb_target.min() == 1.0 and clean_rgb_target.max() == 1.0),
        f"min={float(clean_rgb_target.min())} max={float(clean_rgb_target.max())}",
    )
    registry.check(
        "5",
        "rgb_channel.exactly_one_on_corrupt_path",
        bool(corrupt_target[0].min() == 1.0 and corrupt_target[0].max() == 1.0),
        f"min={float(corrupt_target[0].min())} max={float(corrupt_target[0].max())}",
    )

    # -- family 6: pad semantics -------------------------------------------
    valid_np = tensor_np(valid_mask)[0, 0]
    pad = ~valid_np
    pad_pixels = int(pad.sum())
    expected_pad = PAD_BOTTOM * CROP_WIDTH + PAD_RIGHT * (CROP_HEIGHT - PAD_BOTTOM)
    registry.check(
        "6",
        "pad.valid_mask_is_expected_rectangle",
        bool(np.array_equal(valid_np, unit)),
        f"valid_pixels={int(valid_np.sum())} expected={int(unit.sum())}",
    )
    registry.check(
        "6",
        "pad.pad_pixel_count_matches_geometry",
        pad_pixels == expected_pad and not bool(valid_np[CROP_HEIGHT - PAD_BOTTOM :, :].any())
        and not bool(valid_np[:, CROP_WIDTH - PAD_RIGHT :].any()),
        f"pad_pixels={pad_pixels} expected={expected_pad}",
    )
    normalized_pad_max = max(
        float(np.abs(tensor_np(clean["rgb"])[0][:, pad]).max()),
        float(np.abs(tensor_np(clean["depth"])[0][:, pad]).max()),
        float(np.abs(tensor_np(corrupt["rgb"])[0][:, pad]).max()),
        float(np.abs(tensor_np(corrupt["depth"])[0][:, pad]).max()),
    )
    registry.check(
        "6",
        "pad.normalized_values_exactly_zero",
        normalized_pad_max == 0.0,
        f"max_abs_normalized_on_pad={normalized_pad_max:.6g}",
    )
    raw_pad_max = max(
        float(np.abs(tensor_np(clean["raw_rgb"])[0][:, pad]).max()),
        float(np.abs(tensor_np(clean["raw_depth"])[0][0][pad]).max()),
        float(np.abs(tensor_np(corrupt["raw_rgb"])[0][:, pad]).max()),
        float(np.abs(tensor_np(corrupt["raw_depth"])[0][0][pad]).max()),
    )
    registry.check(
        "6",
        "pad.raw_signals_exactly_zero",
        raw_pad_max == 0.0,
        f"max_abs_raw_on_pad={raw_pad_max:.6g}",
    )
    target_pad_clean = tensor_np(clean["reliability_target"])[0][:, pad]
    target_pad_corrupt = corrupt_target[:, pad]
    registry.check(
        "6",
        "pad.target_is_neutral_one",
        bool(np.all(target_pad_clean == 1.0) and np.all(target_pad_corrupt == 1.0)),
        f"clean_min={float(target_pad_clean.min())} corrupt_min={float(target_pad_corrupt.min())}",
    )

    # -- family 7: corrupted path ------------------------------------------
    registry.check(
        "7",
        "corrupt.depth_input_changed",
        not torch.equal(corrupt["depth"], torch.from_numpy(fixture["depth_tensor"])),
        f"max_abs_delta={float((tensor_np(corrupt['depth']) - fixture['depth_tensor']).__abs__().max()):.6g}",
    )
    registry.check(
        "7",
        "corrupt.rgb_input_bitwise_unchanged",
        torch.equal(corrupt["rgb"], torch.from_numpy(fixture["rgb_tensor"])),
        "torch.equal(corrupt['rgb'], rgb_input) is True",
    )

    # -- family 8: corrupted Depth target ----------------------------------
    corrupt_depth_target = corrupt_target[1]
    post = tensor_np(post_mask)[0, 0]
    registry.check(
        "8",
        "corrupt.target_exactly_zero_on_natural_invalid",
        bool(np.all(corrupt_depth_target[natural] == 0.0)),
        f"natural_invalid_pixels={int(natural.sum())} max_there={float(corrupt_depth_target[natural].max()):.6g}",
    )
    graded_region = unit & native_valid & post
    registry.check(
        "8",
        "corrupt.target_positive_on_pre_and_post_valid",
        bool(np.all(corrupt_depth_target[graded_region] > 0.0) and np.all(corrupt_depth_target[graded_region] <= 1.0)),
        f"pixels={int(graded_region.sum())} min={float(corrupt_depth_target[graded_region].min()):.6g} "
        f"max={float(corrupt_depth_target[graded_region].max()):.6g}",
    )
    registry.check(
        "8",
        "corrupt.target_within_unit_interval",
        bool(corrupt_target.min() >= 0.0 and corrupt_target.max() <= 1.0),
        f"min={float(corrupt_target.min())} max={float(corrupt_target.max())}",
    )
    registry.check(
        "8",
        "corrupt.target_is_graded_not_constant",
        float(corrupt_depth_target[unit].std()) > 0.0,
        f"std_on_valid={float(corrupt_depth_target[unit].std()):.6g} "
        f"mean_on_valid={float(corrupt_depth_target[unit].mean()):.6g}",
    )

    # -- family 9: ranges and finiteness -----------------------------------
    raw_rgb = corrupt["raw_rgb"]
    raw_depth = corrupt["raw_depth"]
    registry.check(
        "9",
        "ranges.raw_rgb_within_unit_interval",
        bool(float(raw_rgb.min()) >= 0.0 and float(raw_rgb.max()) <= 1.0),
        f"min={float(raw_rgb.min())} max={float(raw_rgb.max())}",
    )
    registry.check(
        "9",
        "ranges.raw_depth_within_unit_interval",
        bool(float(raw_depth.min()) >= 0.0 and float(raw_depth.max()) <= 1.0),
        f"min={float(raw_depth.min())} max={float(raw_depth.max())}",
    )
    registry.check(
        "9",
        "finite.rgb",
        bool(torch.isfinite(corrupt["rgb"]).all()),
        "all rgb values are finite",
    )
    registry.check(
        "9",
        "finite.depth",
        bool(torch.isfinite(corrupt["depth"]).all()),
        "all depth values are finite",
    )
    registry.check(
        "9",
        "finite.raw_signals",
        bool(torch.isfinite(raw_rgb).all() and torch.isfinite(raw_depth).all()),
        "raw_rgb and raw_depth are finite on both branches",
    )
    registry.check(
        "9",
        "finite.reliability_target",
        bool(torch.isfinite(clean["reliability_target"]).all() and torch.isfinite(target).all()),
        "reliability_target is finite on both branches",
    )

    # -- family 10: mask subset relations ----------------------------------
    registry.check(
        "10",
        "subsets.depth_valid_pre_within_valid_mask",
        not bool((pre_mask & ~valid_mask).any()),
        f"violations={int((pre_mask & ~valid_mask).sum())}",
    )
    registry.check(
        "10",
        "subsets.depth_valid_post_within_valid_mask",
        not bool((post_mask & ~valid_mask).any()),
        f"violations={int((post_mask & ~valid_mask).sum())}",
    )

    # -- family 11: invalidity accounting ----------------------------------
    counted = {key: int(record.get(key, -1)) for key in METADATA_COUNT_KEYS}
    registry.check(
        "11",
        "accounting.metadata_count_fields_present",
        all(key in record for key in METADATA_COUNT_KEYS),
        f"missing={sorted(set(METADATA_COUNT_KEYS) - set(record.keys()))}",
    )
    balance_left = counted["post_corruption_invalid_pixels"] + counted["newly_valid_pixels"]
    balance_right = counted["natural_invalid_pixels"] + counted["synthetic_missing_pixels"]
    registry.check(
        "11",
        "accounting.balance_closes",
        balance_left == balance_right,
        f"post+newly={balance_left} natural+synthetic={balance_right}",
    )
    recomputed_natural = int(np.count_nonzero(natural))
    recomputed_post = int(np.count_nonzero(unit & ~post))
    recomputed_newly = int(np.count_nonzero(unit & ~native_valid & post))
    registry.check(
        "11",
        "accounting.natural_recomputed_matches",
        counted["natural_invalid_pixels"] == recomputed_natural,
        f"metadata={counted['natural_invalid_pixels']} recomputed={recomputed_natural}",
    )
    registry.check(
        "11",
        "accounting.post_recomputed_matches",
        counted["post_corruption_invalid_pixels"] == recomputed_post,
        f"metadata={counted['post_corruption_invalid_pixels']} recomputed={recomputed_post}",
    )
    registry.check(
        "11",
        "accounting.newly_valid_recomputed_matches",
        counted["newly_valid_pixels"] == recomputed_newly,
        f"metadata={counted['newly_valid_pixels']} recomputed={recomputed_newly}",
    )
    corrupt_raw_depth = tensor_np(corrupt["raw_depth"])[0, 0]
    implicit_region = unit & native_valid & post
    recomputed_implicit = int(np.count_nonzero(implicit_region & (clean_raw_depth != corrupt_raw_depth)))
    registry.check(
        "11",
        "accounting.implicit_quality_recomputed_matches",
        counted["implicit_quality_pixels"] == recomputed_implicit,
        f"metadata={counted['implicit_quality_pixels']} recomputed={recomputed_implicit}",
    )

    # -- family 12: determinism --------------------------------------------
    repeated_tensors = all(torch.equal(default_first[key], default_second[key]) for key in TENSOR_KEYS)
    registry.check(
        "12",
        "determinism.repeat_tensors_bitwise_identical",
        repeated_tensors,
        f"compared={list(TENSOR_KEYS)}",
    )
    registry.check(
        "12",
        "determinism.repeat_metadata_identical",
        to_jsonable(default_first["metadata"]) == to_jsonable(default_second["metadata"]),
        "identical metadata records for the frozen seed key",
    )
    registry.check(
        "12",
        "determinism.repeat_protocol_fields_identical",
        str(default_first["protocol"]) == str(default_second["protocol"])
        and list(default_first["supervised_channels"]) == list(default_second["supervised_channels"]),
        f"protocol={default_first['protocol']!r} supervised_channels={default_first['supervised_channels']!r}",
    )

    # -- family 13: branch reachability ------------------------------------
    p_clean = float(config.mmfr_a2["corruption"]["p_clean"])
    branch_flags: List[bool] = []
    for iteration in range(BRANCH_ITERATIONS):
        result = default_first if iteration == ITERATION else call_helper(helper, config, fixture, p_clean, iteration)
        branch_flags.append(bool(result["metadata"][0]["clean"]))
    clean_count = sum(1 for flag in branch_flags if flag)
    corrupt_count = BRANCH_ITERATIONS - clean_count
    registry.check(
        "13",
        "branches.clean_reachable",
        clean_count > 0,
        f"clean={clean_count}/{BRANCH_ITERATIONS} corrupt={corrupt_count}/{BRANCH_ITERATIONS} p_clean={p_clean}",
    )
    registry.check(
        "13",
        "branches.corrupt_reachable",
        corrupt_count > 0,
        f"clean={clean_count}/{BRANCH_ITERATIONS} corrupt={corrupt_count}/{BRANCH_ITERATIONS} p_clean={p_clean}",
    )
    registry.check(
        "13",
        "branches.counts_sum_to_iterations",
        clean_count + corrupt_count == BRANCH_ITERATIONS,
        f"clean+corrupt={clean_count + corrupt_count} expected={BRANCH_ITERATIONS}",
    )

    # -- family 14: frozen measured counts ---------------------------------
    for key in (
        "valid_pixels",
        "natural_invalid_pixels",
        "synthetic_missing_pixels",
        "newly_valid_pixels",
        "post_corruption_invalid_pixels",
    ):
        expected = EXPECTED_COUNTS[key]
        measured = int(record.get(key, -1))
        registry.check(
            "14",
            f"frozen_counts.{key}",
            measured == expected,
            f"measured={measured} expected={expected}",
        )

    # -- family 15: CPU-only ------------------------------------------------
    registry.check(
        "15",
        "cpu.cuda_never_initialized",
        not torch.cuda.is_initialized(),
        f"torch.cuda.is_initialized()={torch.cuda.is_initialized()} device_used=cpu",
    )

    corrupt_specs = [(str(spec.get("kind")), round(float(spec.get("severity")), 6)) for spec in record.get("specs", ())]
    observations = [
        f"default branch at iteration {ITERATION} for p_clean={p_clean}: "
        f"clean={bool(default_first['metadata'][0]['clean'])}",
        "forced p_clean=1.0 call is bitwise equal to the forced p_clean=0.0 call on the "
        "RGB input and on the geometry masks (Depth and target differ by design)",
        f"branch split over {BRANCH_ITERATIONS} iterations: clean={clean_count} corrupt={corrupt_count} "
        f"(historical expectation {HISTORICAL_BRANCH_SPLIT[0]}/{HISTORICAL_BRANCH_SPLIT[1]})",
        f"the frozen forced-corrupt draw is {corrupt_specs}",
        f"family 8 asserts target > 0 on the 'natively valid AND still valid after corruption' subset: "
        f"of the {int((unit & native_valid).sum())} natively valid pixels, "
        f"{counted['synthetic_missing_pixels']} were structurally zeroed by the frozen corruption, and "
        f"{int(graded_region.sum())} remain positive. Asserting 'positive on every natively valid pixel' "
        f"would contradict the frozen structural corruption semantics of the drawn misalignment.",
        f"injected block is {INJECT_ROW_STOP - INJECT_ROW_START} rows x {INJECT_COL_STOP - INJECT_COL_START} cols "
        f"= {fixture['injection_block_pixels']} pixels; {fixture['injection_block_already_zero_pixels']} of them were "
        f"already zero in the resized sample, so the injection adds "
        f"{fixture['injection_block_newly_zeroed_pixels']} newly invalid pixels and the block is present as a "
        f"natively invalid region inside the valid crop",
        "the historical one-shot check reported 47 assertions and 0 failures; this frozen "
        "re-runnable version keeps that result as history and does not overwrite it",
    ]

    evidence = {
        "fixture": {
            "sample_index": fixture["sample_index"],
            "sample_id_in_split": fixture["sample_id_in_split"],
            "split_path": fixture["split_path"],
            "split_sha256": fixture["split_sha256"],
            "split_line_count": fixture["split_line_count"],
            "rgb_path": fixture["rgb_path"],
            "rgb_sha256": fixture["rgb_sha256"],
            "rgb_decoded_shape": fixture["rgb_decoded_shape"],
            "rgb_decoded_dtype": fixture["rgb_decoded_dtype"],
            "depth_path": fixture["depth_path"],
            "depth_sha256": fixture["depth_sha256"],
            "depth_decoded_shape": fixture["depth_decoded_shape"],
            "depth_decoded_dtype": fixture["depth_decoded_dtype"],
            "depth_decoded_channels": fixture["depth_decoded_channels"],
            "depth_zero_pixels_resized_480x640": fixture["depth_zero_pixels_resized"],
            "depth_zero_pixels_in_valid_region": fixture["depth_zero_pixels_valid_region"],
            "injection_block_rows": fixture["injection_block_rows"],
            "injection_block_cols": fixture["injection_block_cols"],
            "injection_block_pixels": fixture["injection_block_pixels"],
            "injection_block_already_zero_pixels": fixture["injection_block_already_zero_pixels"],
            "injection_block_newly_zeroed_pixels": fixture["injection_block_newly_zeroed_pixels"],
            "fixture_rule": {
                "crop_height": CROP_HEIGHT,
                "crop_width": CROP_WIDTH,
                "pad_right_columns": PAD_RIGHT,
                "pad_bottom_rows": PAD_BOTTOM,
                "rgb_interpolation": "INTER_LINEAR",
                "depth_interpolation": "INTER_NEAREST_then_uint8",
                "random_augmentation": False,
                "rgb_normalization": {"mean": [float(v) for v in np.asarray(config.norm_mean).reshape(-1)],
                                      "std": [float(v) for v in np.asarray(config.norm_std).reshape(-1)]},
                "depth_normalization": {"mean": DEPTH_MEAN, "std": DEPTH_STD},
                "padding_normalized_value": 0.0,
                "tensor_shapes": {"rgb": [1, 3, CROP_HEIGHT, CROP_WIDTH],
                                  "depth": [1, 1, CROP_HEIGHT, CROP_WIDTH]},
            },
        },
        "seed_key": {
            "sample_id": fixture["sample_id_in_split"],
            "epoch": EPOCH,
            "iteration": ITERATION,
            "global_rank": GLOBAL_RANK,
            "sample_slot": SAMPLE_SLOT,
            "corruption_seed": int(config.mmfr_a2["corruption"]["seed"]),
            "iters_per_epoch": int(config.niters_per_epoch),
            "nepochs": int(config.nepochs),
        },
        "clean_call": {"metadata": to_jsonable(clean["metadata"])},
        "corrupt_call": {"metadata": to_jsonable(corrupt["metadata"])},
        "default_call": {"metadata": to_jsonable(default_first["metadata"])},
        "branch_split": {
            "iterations": BRANCH_ITERATIONS,
            "clean": clean_count,
            "corrupt": corrupt_count,
            "clean_iterations": [i for i, flag in enumerate(branch_flags) if flag],
            "corrupt_iterations": [i for i, flag in enumerate(branch_flags) if not flag],
        },
        "observations": observations,
    }
    return evidence


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-runnable CPU qualification of the MMFR-A2 v2 batch helper.")
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "outputs", "mmfr-a2-v2-cpu-qualification", "a2-v2-cpu-qualification.json"),
        help="path of the JSON report",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    stdout_lines: List[str] = []

    def emit(text: str = "") -> None:
        stdout_lines.append(text)
        print(text)

    registry = CheckRegistry(emit)
    emit("MMFR-A2 v2 CPU qualification (frozen re-runnable version of the historical 47-item check)")
    emit(f"repository root: {PROJECT_ROOT}")
    emit("")

    report: Dict[str, Any] = {
        "tool": os.path.relpath(SCRIPT_PATH, PROJECT_ROOT).replace(os.sep, "/"),
        "tool_sha256_raw_bytes": sha256_raw(SCRIPT_PATH),
        "tool_sha256_lf_normalized": sha256_lf(SCRIPT_PATH),
        "purpose": (
            "re-runnable CPU qualification of utils/dataloader/mmfr_training_v2.build_mmfr_training_batch_v2 "
            "for protocol MMFR-A2-train-integration-v2 on one frozen real train-dev fixture"
        ),
        "historical_result": {
            "assertions": 47,
            "failures": 0,
            "note": "one-shot inline CPU check, deleted after the run; kept as history, not overwritten",
        },
        "cpu_only": True,
        "cuda_available_on_host": bool(torch.cuda.is_available()),
        "cuda_used": False,
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "cv2": str(cv2.__version__),
        },
        "code_identity": {},
        "config_identity": {},
        "family_titles": FAMILY_TITLES,
        "checks": [],
        "assertions_total": 0,
        "assertions_failed": 0,
        "status": "ERROR",
        "exit_code": 2,
        "runtime_seconds": 0.0,
        "report_path": os.path.abspath(args.output),
    }

    exit_code = 2
    try:
        from utils.dataloader import mmfr_training_v2 as helper_module
        from utils.dataloader import multimodal_failure_v2 as corruption_module

        config = __import__("importlib").import_module(CONFIG_MODULE).C
        modules = {
            "helper": helper_module.build_mmfr_training_batch_v2,
            "helper_module": helper_module,
            "corruption": corruption_module,
        }

        code_identity = {}
        for relative_path, required in CODE_IDENTITY_FILES:
            absolute = os.path.join(PROJECT_ROOT, relative_path.replace("/", os.sep))
            if not os.path.isfile(absolute):
                code_identity[relative_path] = {"required": required, "present": False}
                continue
            code_identity[relative_path] = {
                "required": required,
                "present": True,
                "raw_sha256": sha256_raw(absolute),
                "lf_normalized_sha256": sha256_lf(absolute),
                "bytes": os.path.getsize(absolute),
            }
        report["code_identity"] = code_identity
        report["config_identity"] = {
            "module": CONFIG_MODULE,
            "run_id": str(config.run_id),
            "protocol": str(config.mmfr_a2.get("protocol")),
            "mode": str(config.mmfr_a2.get("mode")),
            "corruption_basis": str((config.mmfr_a2.get("corruption") or {}).get("basis")),
            "corruption_seed": int(config.mmfr_a2["corruption"]["seed"]),
            "p_clean": float(config.mmfr_a2["corruption"]["p_clean"]),
            "max_specs": int(config.mmfr_a2["corruption"]["max_specs"]),
            "supervised_channels": list((config.mmfr_a2.get("reliability_head") or {}).get("supervised_channels") or ()),
            "analysis_identity": str((config.mmfr_a2.get("frozen") or {}).get("analysis_identity")),
            "schedule_version": str((config.mmfr_a2.get("frozen") or {}).get("schedule_version")),
        }
        if code_identity:
            missing_required = [path for path, info in code_identity.items() if info.get("required") and not info.get("present")]
            if missing_required:
                raise RuntimeError(f"required code identity files are missing: {missing_required}")

        evidence = run_checks(registry, config, modules)
        report.update(evidence)
        emit("")
    except BaseException as error:  # noqa: BLE001 - the report must still be written
        last_line = traceback.format_exc().strip().splitlines()[-1]
        registry.check("harness", "harness.unexpected_exception", False, last_line)
        emit("")
        emit("HARNESS EXCEPTION")
        emit(traceback.format_exc())
        report["harness_exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}

    failures = registry.failures()
    report["checks"] = registry.entries
    report["assertion_families"] = sorted({entry["family"] for entry in registry.entries})
    report["family_coverage"] = {
        family: registry.count_for(family) for family in sorted({entry["family"] for entry in registry.entries})
    }
    report["assertions_total"] = len(registry.entries)
    report["assertions_failed"] = len(failures)
    report["failed_assertions"] = [entry["name"] for entry in failures]
    report["status"] = "PASS" if registry.entries and not failures and "harness_exception" not in report else "FAIL"
    exit_code = 0 if report["status"] == "PASS" else 1
    report["exit_code"] = exit_code
    report["runtime_seconds"] = round(time.perf_counter() - started, 3)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(report), handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    report_sha256 = sha256_raw(output_path)

    emit("")
    emit(f"report_path: {output_path}")
    emit(f"report_sha256: {report_sha256}")
    emit(f"tool_sha256_raw_bytes: {report['tool_sha256_raw_bytes']}")
    emit(
        "SUMMARY assertions={} failed={} status={} exit_code={} runtime={:.3f}s".format(
            report["assertions_total"], report["assertions_failed"], report["status"], exit_code, report["runtime_seconds"]
        )
    )
    if failures:
        emit("FAILED CHECKS:")
        for entry in failures:
            emit(f"  {entry['family']}.{entry['name']}: {entry['detail']}")

    stdout_path = os.path.join(os.path.dirname(output_path), "run-stdout.txt")
    with open(stdout_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(stdout_lines) + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
