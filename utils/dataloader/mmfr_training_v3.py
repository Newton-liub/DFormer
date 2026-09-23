#!/usr/bin/env python3
"""Post-crop / pre-GPU MMFR-A2 v3 Depth-corruption batch helper.

The v3 helper is a new identity derived from the audited v2 helper.  It keeps the v2
RNG, uint8 round-trip, Depth-only curriculum and burden semantics, but transports an
explicit sequential validity state.  The state starts at ``depth_valid_pre`` and is
updated by the six v3 corruption operations.  The supervised target is
``V_state_final * R_depth_synthetic``; padding remains neutral and excluded.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from utils.dataloader.mmfr_training import (
    CLEAN_PROBABILITY,
    CORRUPTION_SEED,
    CURRICULUM_LIGHT_MAX_SEVERITY as A1_CURRICULUM_LIGHT_MAX_SEVERITY,
    CURRICULUM_MODERATE_MAX_SEVERITY as A1_CURRICULUM_MODERATE_MAX_SEVERITY,
    CURRICULUM_SEVERITY_FLOOR as A1_CURRICULUM_SEVERITY_FLOOR,
    DEPTH_PLANE_MEAN,
    DEPTH_PLANE_STD,
    ENTIRE_MISSING_HEAVY_SEVERITY as A1_ENTIRE_MISSING_HEAVY_SEVERITY,
    FAILURE_KINDS as A1_FAILURE_KINDS,
    MAX_SPECS,
    NORMALIZED_PAD_VALUE,
    build_seed_words,
    curriculum_progress,
    make_sample_generator,
    normalize_sample_id,
    normalized_to_uint8,
    sample_id_words,
    uint8_to_normalized,
)
from utils.dataloader.multimodal_failure_v3 import (
    CURRICULUM_LIGHT_MAX_SEVERITY,
    CURRICULUM_MODERATE_MAX_SEVERITY,
    CURRICULUM_SEVERITY_FLOOR,
    ENTIRE_MISSING_HEAVY_SEVERITY,
    FAILURE_KINDS,
    MODALITY_INDEX,
    RELIABILITY_CHANNELS,
    SEVERITY_ENCODING,
    FailureSpec,
    apply_failures,
)

PROTOCOL_ID = "MMFR-A2-train-integration-v3"
SUPERSEDES = "MMFR-A2-train-integration-v2"
CORRUPTION_BASIS = "MMFR-A1-corruption-basis-v3"
CURRICULUM_KINDS: Tuple[str, ...] = FAILURE_KINDS
DEPTH_MODALITY = "depth"
RGB_MODALITY = "rgb"
RGB_RELIABILITY_INDEX = MODALITY_INDEX[RGB_MODALITY]
DEPTH_RELIABILITY_INDEX = MODALITY_INDEX[DEPTH_MODALITY]
SUPERVISED_CHANNELS: Tuple[str, ...] = ("depth",)
TARGET_COMPOSITION = "R_depth_sup_v3(p) = V_state_final(p) * R_depth_synthetic(p)"
V1_ONLY_KEYS: Tuple[str, ...] = ("depth_valid",)

if tuple(CURRICULUM_KINDS) != tuple(A1_FAILURE_KINDS):
    raise RuntimeError("A1 v3 failure kinds drifted from the frozen six-kind tuple")
if (
    CURRICULUM_SEVERITY_FLOOR != A1_CURRICULUM_SEVERITY_FLOOR
    or CURRICULUM_LIGHT_MAX_SEVERITY != A1_CURRICULUM_LIGHT_MAX_SEVERITY
    or CURRICULUM_MODERATE_MAX_SEVERITY != A1_CURRICULUM_MODERATE_MAX_SEVERITY
    or ENTIRE_MISSING_HEAVY_SEVERITY != A1_ENTIRE_MISSING_HEAVY_SEVERITY
):
    raise RuntimeError("A1 v3 curriculum severities drifted from the frozen values")
if SEVERITY_ENCODING != "single":
    raise RuntimeError("A1 v3 must encode severity exactly once")

__all__ = [
    "PROTOCOL_ID",
    "SUPERSEDES",
    "CORRUPTION_BASIS",
    "CORRUPTION_SEED",
    "CLEAN_PROBABILITY",
    "MAX_SPECS",
    "CURRICULUM_KINDS",
    "SUPERVISED_CHANNELS",
    "TARGET_COMPOSITION",
    "sample_depth_failure_specs_v3",
    "build_mmfr_training_batch_v3",
]


def sample_depth_failure_specs_v3(
    progress: float,
    rng: np.random.Generator,
    max_specs: int = MAX_SPECS,
) -> Tuple[Tuple[FailureSpec, ...], int]:
    """Sample the unchanged three-stage curriculum after restricting to Depth."""
    value = float(progress)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"progress must be a finite value in [0, 1], got {progress!r}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator supplied by the caller")
    if not isinstance(max_specs, int) or max_specs < 1:
        raise ValueError(f"max_specs must be a positive int, got {max_specs!r}")
    if value < 1.0 / 3.0:
        max_severity, max_count, include_entire_missing = CURRICULUM_LIGHT_MAX_SEVERITY, 1, False
    elif value < 2.0 / 3.0:
        max_severity, max_count, include_entire_missing = (
            CURRICULUM_MODERATE_MAX_SEVERITY,
            min(2, max_specs),
            False,
        )
    else:
        max_severity, max_count, include_entire_missing = 1.0, max_specs, True
    kinds = tuple(kind for kind in CURRICULUM_KINDS if include_entire_missing or kind != "entire_missing")
    count = min(max_count, len(kinds))
    if count > 1:
        count = int(rng.integers(1, count + 1))
    chosen = np.asarray(rng.choice(len(kinds), size=count, replace=False), dtype=np.int64).reshape(-1)
    specs = []
    for index in chosen:
        kind = kinds[int(index)]
        severity = float(rng.uniform(CURRICULUM_SEVERITY_FLOOR, max_severity))
        if kind == "entire_missing":
            severity = ENTIRE_MISSING_HEAVY_SEVERITY
        specs.append(FailureSpec(modality=DEPTH_MODALITY, kind=kind, severity=severity))
    return tuple(specs), 1


def _require_normalized_image(tensor: Any, name: str, channels: Sequence[int]) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if tensor.dim() != 4:
        raise ValueError(f"{name} must be [B, C, H, W], got shape {tuple(tensor.shape)}")
    if tensor.shape[1] not in tuple(channels):
        raise ValueError(f"{name} must have channels in {tuple(channels)}, got {tensor.shape[1]}")
    if tensor.shape[0] == 0 or tensor.shape[2] == 0 or tensor.shape[3] == 0:
        raise ValueError(f"{name} must have non-empty dims, got shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _resolve_sample_ids(sample_ids: Any, batch: int, root: str | None) -> Tuple[str, ...]:
    if isinstance(sample_ids, str):
        raw = [sample_ids]
    else:
        if not isinstance(sample_ids, Sequence):
            raise TypeError("sample_ids must be a string or a sequence of strings")
        raw = [str(value) for value in sample_ids]
    if len(raw) != batch:
        raise ValueError(f"expected {batch} sample ids, got {len(raw)}")
    return tuple(normalize_sample_id(value, root) for value in raw)


def _assert_tensor_is_zero_at_pad(tensor: torch.Tensor, pad: torch.Tensor, name: str) -> None:
    selected = torch.masked_select(tensor, pad)
    if selected.numel() and float(selected.abs().max().item()) != 0.0:
        raise RuntimeError(f"{name} is not exactly zero on the padding region")


def _pixel_statistics(
    valid: np.ndarray,
    depth_pre_uint8: np.ndarray,
    depth_post_uint8: np.ndarray,
    depth_valid_post: np.ndarray,
    depth_changed: np.ndarray,
) -> Dict[str, int]:
    """Return the four telemetry populations and closed validity census."""
    pre_valid = depth_pre_uint8 > 0
    post_valid = depth_post_uint8 > 0
    state = np.asarray(depth_valid_post, dtype=bool)
    if not np.array_equal(state, post_valid & valid):
        raise RuntimeError("depth_valid_post must equal (raw_depth_post > 0) & valid_mask")
    natural_invalid = int(np.count_nonzero(valid & ~pre_valid))
    synthetic_invalid = int(np.count_nonzero(valid & pre_valid & ~state))
    newly_valid = int(np.count_nonzero(valid & ~pre_valid & state))
    post_corruption_invalid = int(np.count_nonzero(valid & ~state))
    if post_corruption_invalid + newly_valid != natural_invalid + synthetic_invalid:
        raise RuntimeError(
            "v3 validity census is inconsistent: post_corruption_invalid_pixels + newly_valid_pixels "
            "must equal natural_invalid_pixels + synthetic_missing_pixels"
        )
    # Opt-A: the caller already computed (depth_pre != depth_post); reuse it instead of
    # rebuilding the same comparison twice, and derive valid-clean from the same partition.
    both_valid = valid & pre_valid & state
    both_valid_pixels = int(np.count_nonzero(both_valid))
    changed_valid = int(np.count_nonzero(both_valid & depth_changed))
    return {
        "valid_pixels": int(np.count_nonzero(valid)),
        "natural_invalid_pixels": natural_invalid,
        "synthetic_invalid_pixels": synthetic_invalid,
        "synthetic_missing_pixels": synthetic_invalid,
        "post_corruption_invalid_pixels": post_corruption_invalid,
        "newly_valid_pixels": newly_valid,
        "implicit_quality_pixels": changed_valid,
        "valid_clean_pixels": both_valid_pixels - changed_valid,
    }


def build_mmfr_training_batch_v3(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    sample_ids: Any,
    *,
    epoch: int,
    iteration: int,
    niters_per_epoch: int,
    nepochs: int,
    rgb_mean: Any,
    rgb_std: Any,
    corruption_seed: int = CORRUPTION_SEED,
    global_rank: int = 0,
    p_clean: float = CLEAN_PROBABILITY,
    max_specs: int = MAX_SPECS,
    sample_id_root: str | None = None,
) -> Dict[str, Any]:
    """Apply v3 Depth corruption to one final CPU training batch."""
    rgb_tensor = _require_normalized_image(rgb, "rgb", (3,))
    depth_tensor = _require_normalized_image(depth, "depth", (1, 3))
    if rgb_tensor.shape[0] != depth_tensor.shape[0]:
        raise ValueError("rgb and depth must share the batch size")
    if rgb_tensor.shape[-2:] != depth_tensor.shape[-2:]:
        raise ValueError("rgb and depth must share the spatial shape")
    if not isinstance(max_specs, int) or max_specs < 1:
        raise ValueError(f"max_specs must be a positive int, got {max_specs!r}")
    if not 0.0 <= float(p_clean) <= 1.0:
        raise ValueError(f"p_clean must lie in [0, 1], got {p_clean!r}")
    if tuple(CURRICULUM_KINDS) != tuple(FAILURE_KINDS):
        raise RuntimeError("the A2 curriculum must keep the six frozen failure kinds")

    batch = int(rgb_tensor.shape[0])
    normalized_ids = _resolve_sample_ids(sample_ids, batch, sample_id_root)
    progress = curriculum_progress(epoch, iteration, niters_per_epoch, nepochs)
    train_seed = int(corruption_seed)
    rank = int(global_rank)
    if not 0 <= train_seed < (1 << 32):
        raise ValueError(f"corruption_seed must be a uint32 seed word, got {corruption_seed!r}")
    if not 0 <= rank < (1 << 32):
        raise ValueError(f"global_rank must be a uint32 seed word, got {global_rank!r}")
    rgb_mean64 = np.asarray(rgb_mean, dtype=np.float64).reshape(-1)
    rgb_std64 = np.asarray(rgb_std, dtype=np.float64).reshape(-1)
    if rgb_mean64.shape != (3,) or rgb_std64.shape != (3,):
        raise ValueError("rgb_mean and rgb_std must hold three per-channel values")
    if not np.isfinite(rgb_mean64).all() or not np.isfinite(rgb_std64).all():
        raise ValueError("rgb_mean and rgb_std must be finite")
    if (rgb_std64 <= 0.0).any():
        raise ValueError("rgb_std must be strictly positive")
    depth_mean64 = np.asarray((DEPTH_PLANE_MEAN,), dtype=np.float64)
    depth_std64 = np.asarray((DEPTH_PLANE_STD,), dtype=np.float64)

    rgb_np = rgb_tensor.detach().to(torch.float32).cpu().numpy()
    depth_np = depth_tensor.detach().to(torch.float32).cpu().numpy()
    depth_channels = int(depth_np.shape[1])
    if depth_channels == 3 and (
        not np.array_equal(depth_np[:, 0], depth_np[:, 1])
        or not np.array_equal(depth_np[:, 0], depth_np[:, 2])
    ):
        raise ValueError("depth must be single-channel or three identical channels")

    rgb_outputs: List[np.ndarray] = []
    depth_outputs: List[np.ndarray] = []
    raw_rgb_outputs: List[np.ndarray] = []
    raw_depth_outputs: List[np.ndarray] = []
    target_outputs: List[np.ndarray] = []
    valid_outputs: List[np.ndarray] = []
    depth_valid_pre_outputs: List[np.ndarray] = []
    depth_valid_post_outputs: List[np.ndarray] = []
    telemetry_mask_outputs: List[np.ndarray] = []
    metadata: List[Mapping[str, Any]] = []

    for slot in range(batch):
        rgb_sample = np.ascontiguousarray(rgb_np[slot], dtype=np.float32)
        depth_sample_channels = np.ascontiguousarray(depth_np[slot], dtype=np.float32)
        depth_sample = np.ascontiguousarray(depth_sample_channels[:1], dtype=np.float32)
        height, width = rgb_sample.shape[-2:]
        rgb_pad = np.all(rgb_sample == np.float32(NORMALIZED_PAD_VALUE), axis=0)
        depth_pad = np.all(depth_sample == np.float32(NORMALIZED_PAD_VALUE), axis=0)
        valid = ~(rgb_pad & depth_pad)
        if not valid.any():
            raise RuntimeError("sample has no geometrically valid pixel after crop/pad")
        invalid = ~valid

        rgb_uint8 = normalized_to_uint8(rgb_sample, rgb_mean64, rgb_std64)
        depth_uint8 = normalized_to_uint8(depth_sample, depth_mean64, depth_std64)[0]
        rgb_uint8[:, invalid] = np.uint8(0)
        depth_uint8[invalid] = np.uint8(0)
        depth_valid_pre = (depth_uint8 > 0) & valid

        sample_words = sample_id_words(normalized_ids[slot])
        words = build_seed_words(train_seed, epoch, iteration, rank, slot, sample_words)
        rng = make_sample_generator(words)
        is_clean = bool(float(rng.random()) < float(p_clean))

        specs: Tuple[FailureSpec, ...] = ()
        spec_draws = 0
        spec_records: Tuple[Mapping[str, Any], ...] = ()
        corrupted_depth_uint8 = depth_uint8
        synthetic_target = np.ones((height, width), dtype=np.float32)
        depth_valid_state = depth_valid_pre.copy()
        if is_clean:
            rgb_normalized = rgb_sample
            depth_normalized = depth_sample_channels
        else:
            specs, spec_draws = sample_depth_failure_specs_v3(progress, rng, max_specs=max_specs)
            result = apply_failures(
                np.ascontiguousarray(rgb_uint8.transpose(1, 2, 0)),
                depth_uint8,
                specs,
                rng,
                depth_validity=depth_valid_pre,
                validity_mask=valid,
            )
            spec_records = tuple(result.metadata.get("specs", ()))
            corrupted_depth_uint8 = np.ascontiguousarray(result.depth)
            depth_valid_state = np.ascontiguousarray(result.validity_state, dtype=bool)
            synthetic_target = np.ascontiguousarray(
                result.reliability[DEPTH_RELIABILITY_INDEX], dtype=np.float32
            )
            depth_plane = corrupted_depth_uint8 if corrupted_depth_uint8.ndim == 2 else corrupted_depth_uint8[:, :, 0]
            if not np.array_equal(depth_plane != 0, depth_valid_state):
                raise RuntimeError("v3 corrupted Depth does not match its explicit validity state")
            corrupted_normalized = uint8_to_normalized(
                corrupted_depth_uint8[:, :, None], depth_mean64, depth_std64
            )[:, :, 0]
            corrupted_plane = (
                corrupted_normalized[None]
                if depth_channels == 1
                else np.repeat(corrupted_normalized[None], depth_channels, axis=0)
            )
            depth_normalized = np.where(valid[None], corrupted_plane, depth_sample_channels)
            rgb_normalized = rgb_sample

        supervised_depth = (depth_valid_state.astype(np.float32) * synthetic_target).astype(np.float32)
        target = np.ones((RELIABILITY_CHANNELS, height, width), dtype=np.float32)
        target[DEPTH_RELIABILITY_INDEX] = supervised_depth
        target[:, ~valid] = np.float32(1.0)
        if float(target[RGB_RELIABILITY_INDEX].min()) != 1.0 or float(target[RGB_RELIABILITY_INDEX].max()) != 1.0:
            raise RuntimeError("the RGB reliability scaffold channel must stay exactly 1 in A2 v3")

        raw_rgb = (rgb_uint8.astype(np.float64) / 255.0).astype(np.float32)
        raw_depth = (corrupted_depth_uint8.astype(np.float64) / 255.0).astype(np.float32)
        raw_rgb = np.where(valid[None], raw_rgb, np.float32(0.0))
        raw_depth = np.where(valid[None], raw_depth[None], np.float32(0.0))
        depth_valid_post = (corrupted_depth_uint8 > 0) & valid
        if not np.array_equal(depth_valid_post, depth_valid_state):
            raise RuntimeError("depth_valid_post must equal the final explicit v3 validity state")
        if not np.array_equal(depth_valid_post, (raw_depth[0] > 0.0) & valid):
            raise RuntimeError("depth_valid_post must equal (raw_depth_uint8 > 0) & valid_mask")

        depth_changed = depth_uint8 != corrupted_depth_uint8
        statistics = _pixel_statistics(
            valid, depth_uint8, corrupted_depth_uint8, depth_valid_post, depth_changed
        )
        telemetry_masks = np.stack(
            (
                valid & ~depth_valid_pre,
                valid & depth_valid_pre & ~depth_valid_post,
                valid & depth_valid_pre & depth_valid_post & depth_changed,
                valid & depth_valid_pre & depth_valid_post & ~depth_changed,
            ),
            axis=0,
        ).astype(bool)
        if np.any(telemetry_masks.sum(axis=0) > 1):
            raise RuntimeError("v3 telemetry masks must be mutually exclusive")
        if not np.array_equal(telemetry_masks.any(axis=0), valid):
            raise RuntimeError("v3 telemetry masks must partition the geometry-valid region")
        rgb_outputs.append(rgb_normalized.astype(np.float32))
        depth_outputs.append(depth_normalized.astype(np.float32))
        raw_rgb_outputs.append(raw_rgb)
        raw_depth_outputs.append(raw_depth)
        target_outputs.append(target)
        valid_outputs.append(valid[None])
        depth_valid_pre_outputs.append(depth_valid_pre[None])
        depth_valid_post_outputs.append(depth_valid_post[None])
        telemetry_mask_outputs.append(telemetry_masks)
        depth_record = {
            "sample_slot": slot,
            "sample_id": normalized_ids[slot],
            "seed_words": words,
            "curriculum_progress": progress,
            "clean": is_clean,
            "spec_draws": spec_draws,
            "num_specs": len(specs),
            "specs": spec_records,
            "target_composition": TARGET_COMPOSITION,
            "supervised_channels": list(SUPERVISED_CHANNELS),
            "validity_state_semantics": "explicit sequential state with MID-A integer transport",
            "depth_target_min": float(target[DEPTH_RELIABILITY_INDEX][valid].min()),
            "depth_target_mean": float(target[DEPTH_RELIABILITY_INDEX][valid].mean()),
            "depth_synthetic_reliability_mean": float(synthetic_target[valid].mean()),
            "valid_fraction": float(valid.mean()),
            "depth_valid_pre_fraction": float(depth_valid_pre.mean()),
            "depth_valid_post_fraction": float(depth_valid_post.mean()),
            "validity_state_final_pixels": int(np.count_nonzero(depth_valid_state)),
        }
        depth_record.update(statistics)
        metadata.append(depth_record)

    def _stack(arrays: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(np.stack(arrays, axis=0)))

    batch_outputs: Dict[str, Any] = {
        "rgb": _stack(rgb_outputs),
        "depth": _stack(depth_outputs),
        "raw_rgb": _stack(raw_rgb_outputs),
        "raw_depth": _stack(raw_depth_outputs),
        "reliability_target": _stack(target_outputs),
        "valid_mask": _stack(valid_outputs),
        "depth_valid_pre": _stack(depth_valid_pre_outputs),
        "depth_valid_post": _stack(depth_valid_post_outputs),
        "telemetry_masks": _stack(telemetry_mask_outputs),
        "telemetry_categories": (
            "natural-invalid",
            "synthetic-invalid",
            "implicit-quality",
            "valid-clean",
        ),
        "protocol": PROTOCOL_ID,
        "supervised_channels": list(SUPERVISED_CHANNELS),
        "metadata": metadata,
    }
    if int(batch_outputs["depth"].shape[1]) != depth_channels:
        raise RuntimeError(
            f"depth channel count changed from {depth_channels} to {batch_outputs['depth'].shape[1]}"
        )
    _validate_batch_outputs_v3(batch_outputs)
    return batch_outputs


def _validate_batch_outputs_v3(outputs: Mapping[str, Any]) -> None:
    """Fail-closed structural, state, range, finite and padding checks."""
    for key in V1_ONLY_KEYS:
        if key in outputs:
            raise RuntimeError(f"A2 v3 must not re-export the ambiguous v1 key {key!r}")
    rgb = outputs["rgb"]
    depth = outputs["depth"]
    raw_rgb = outputs["raw_rgb"]
    raw_depth = outputs["raw_depth"]
    target = outputs["reliability_target"]
    valid_mask = outputs["valid_mask"]
    depth_valid_pre = outputs["depth_valid_pre"]
    depth_valid_post = outputs["depth_valid_post"]
    telemetry_masks = outputs["telemetry_masks"]
    batch, channels, height, width = tuple(depth.shape)
    if tuple(rgb.shape) != (batch, 3, height, width):
        raise RuntimeError(f"unexpected rgb shape {tuple(rgb.shape)}")
    if tuple(raw_rgb.shape) != (batch, 3, height, width):
        raise RuntimeError(f"unexpected raw_rgb shape {tuple(raw_rgb.shape)}")
    if tuple(raw_depth.shape) != (batch, 1, height, width):
        raise RuntimeError(f"unexpected raw_depth shape {tuple(raw_depth.shape)}")
    if tuple(target.shape) != (batch, RELIABILITY_CHANNELS, height, width):
        raise RuntimeError(f"unexpected reliability_target shape {tuple(target.shape)}")
    for name, mask in (
        ("valid_mask", valid_mask),
        ("depth_valid_pre", depth_valid_pre),
        ("depth_valid_post", depth_valid_post),
    ):
        if tuple(mask.shape) != (batch, 1, height, width) or mask.dtype != torch.bool:
            raise RuntimeError(f"unexpected {name} {tuple(mask.shape)} / {mask.dtype}")
    if channels not in (1, 3):
        raise RuntimeError(f"unexpected depth channel count {channels}")
    if tuple(telemetry_masks.shape) != (batch, 4, height, width) or telemetry_masks.dtype != torch.bool:
        raise RuntimeError(
            f"unexpected telemetry_masks {tuple(telemetry_masks.shape)} / {telemetry_masks.dtype}"
        )
    if tuple(outputs.get("telemetry_categories", ())) != (
        "natural-invalid",
        "synthetic-invalid",
        "implicit-quality",
        "valid-clean",
    ):
        raise RuntimeError("unexpected v3 telemetry category order")
    # Opt-A: reuse one union of the four masks for both the exclusivity census and the
    # partition check instead of an int64 channel sum plus a second full traversal.
    telemetry_union = (
        telemetry_masks[:, 0]
        | telemetry_masks[:, 1]
        | telemetry_masks[:, 2]
        | telemetry_masks[:, 3]
    )
    overlap_total = (
        int(torch.count_nonzero(telemetry_masks[:, 0]))
        + int(torch.count_nonzero(telemetry_masks[:, 1]))
        + int(torch.count_nonzero(telemetry_masks[:, 2]))
        + int(torch.count_nonzero(telemetry_masks[:, 3]))
        - int(torch.count_nonzero(telemetry_union))
    )
    if overlap_total != 0:
        raise RuntimeError("v3 telemetry masks must be mutually exclusive")
    if not torch.equal(telemetry_union.unsqueeze(1), valid_mask):
        raise RuntimeError("v3 telemetry masks must partition the geometry-valid region")
    # Opt-A: one fused extreme reduction per tensor replaces torch.isfinite(t).all()
    # (which allocated a full-size boolean tensor) plus separate min/max reductions.
    # For a tensor, "all values finite" is exactly "min and max are finite", because
    # torch min/max propagate NaN and +/-inf.
    extremes = {
        name: torch.aminmax(tensor)
        for name, tensor in (
            ("rgb", rgb),
            ("depth", depth),
            ("raw_rgb", raw_rgb),
            ("raw_depth", raw_depth),
            ("target", target),
        )
    }
    for name, (minimum, maximum) in extremes.items():
        if not math.isfinite(float(minimum)) or not math.isfinite(float(maximum)):
            raise RuntimeError(f"{name} contains non-finite values")
    if float(extremes["target"][0]) < 0.0 or float(extremes["target"][1]) > 1.0:
        raise RuntimeError("reliability_target left [0, 1]")
    if (
        float(extremes["raw_rgb"][0]) < 0.0
        or float(extremes["raw_rgb"][1]) > 1.0
        or float(extremes["raw_depth"][0]) < 0.0
        or float(extremes["raw_depth"][1]) > 1.0
    ):
        raise RuntimeError("raw signals left [0, 1]")
    if torch.any(depth_valid_pre & ~valid_mask) or torch.any(depth_valid_post & ~valid_mask):
        raise RuntimeError("Depth validity masks must be subsets of the geometry valid mask")
    scaffold = target[:, RGB_RELIABILITY_INDEX]
    scaffold_min, scaffold_max = torch.aminmax(scaffold)
    if float(scaffold_min) != 1.0 or float(scaffold_max) != 1.0:
        raise RuntimeError("the RGB reliability scaffold channel must be exactly 1 in A2 v3")
    expected_post = (raw_depth > 0.0) & valid_mask
    if not torch.equal(depth_valid_post, expected_post):
        raise RuntimeError("depth_valid_post must equal (raw_depth_uint8 > 0) & valid_mask")
    # Opt-A: scan the padding region once and reuse the same index set for every
    # pad-value assertion instead of running one masked_select per tensor.
    pad_index = (~valid_mask)[:, 0].nonzero(as_tuple=False)
    if pad_index.numel():
        row_index = pad_index[:, 0]
        height_index = pad_index[:, 1]
        width_index = pad_index[:, 2]
        padded_target = target[row_index, :, height_index, width_index]
        if padded_target.numel() and float((padded_target - 1.0).abs().max()) != 0.0:
            raise RuntimeError("reliability_target must be neutral (1) on padding")
        for name, tensor in (("rgb", rgb), ("depth", depth), ("raw_rgb", raw_rgb), ("raw_depth", raw_depth)):
            padded_values = tensor[row_index, :, height_index, width_index]
            if padded_values.numel() and float(padded_values.abs().max()) != 0.0:
                raise RuntimeError(f"{name} is not exactly zero on the padding region")
