#!/usr/bin/env python3
"""Post-crop / pre-GPU Depth corruption batch helper for ``MMFR-A2-train-integration-v2``.

This is the v2 revision of ``utils/dataloader/mmfr_training.py``. The v1 helper is frozen
and keeps its own module, protocol identity and semantics. v2 differs in exactly three
places; everything else is either imported from the v1 helper (so the RNG identity, the
uint8 round trip and the curriculum position are provably shared) or asserted equal at
import time.

Changes relative to v1
----------------------
1. **Depth validity is split into pre- and post-corruption fields.**
   ``depth_valid_pre`` is the *real input* validity before corruption,
   ``raw_depth_pre > 0 AND valid_mask``; ``depth_valid_post`` is the validity of the
   corrupted Depth actually handed to the model, ``raw_depth_post > 0 AND valid_mask``.
   The old single ``depth_valid`` (= post) is deliberately not re-exported, so a caller
   cannot silently read the wrong field.
2. **The supervised Depth target honours native MUSeg Depth invalidity.** The A1 v2
   synthetic target is ``R_D^syn``; the target actually used for supervision is

   $$
   R_D^{sup}(p) = V_D^{pre}(p) \\cdot R_D^{syn}(p),
   $$

   so on a clean sample a natively valid Depth pixel has target ``1`` and a natively
   invalid (``raw Depth == 0``) pixel has target ``0``. Crop/pad pixels stay at the
   neutral target ``1`` and are excluded by ``valid_mask`` as before. The RGB channel of
   the ``[B, 2, H, W]`` target remains an all-ones scaffold channel: A2 v2 supervises
   Depth only and the RGB channel is not a trained reliability estimate.
3. **The invalidity populations are recorded separately**, never merged:
   ``natural_invalid_pixels`` (natively invalid before corruption) and
   ``synthetic_missing_pixels`` (valid before corruption, zeroed by corruption) are the
   two sources of invalid pixels, but they are *not* a partition of
   ``post_corruption_invalid_pixels``: ``gaussian_noise`` and ``misalignment`` can turn a
   natively invalid (``raw Depth == 0``) pixel into a nonzero one. That fourth population
   is therefore recorded explicitly as ``newly_valid_pixels`` and the census keeps the
   exact balance
   ``post_corruption_invalid_pixels + newly_valid_pixels
   == natural_invalid_pixels + synthetic_missing_pixels``.
   ``implicit_quality_pixels`` additionally counts pixels that stay nonzero but are
   changed by corruption, which the ``depth_valid`` mask cannot see.

Frozen semantics carried over unchanged
---------------------------------------
* Corruption runs in the main training process, after the frozen ``DataLoader`` geometry
  (mirror/scale/crop/pad) and before the batch is moved to the GPU. Neither ``TrainPre``
  nor ``RGBXDataset`` is touched.
* Each sample gets its own stateless ``numpy.random.Generator(PCG64(SeedSequence(
  words)))``; process-wide Python/NumPy/PyTorch random state is never read or written.
* ``words`` order is ``train_seed, epoch(1-based), iteration(0-based), global_rank,
  sample_slot, sample_id_sha256_u32_be_0..3`` with path separators unified to ``/``.
* Only Depth specs are applied. ``p_clean = 0.25``; corrupt samples call the frozen
  three-stage curriculum with ``max_specs = 2`` over the six A1 failure kinds.
* ``progress`` is ``((epoch - 1) * N_iter + iteration) / (N_epoch * N_iter - 1)``.
* Clean samples reuse the original normalized segmentation tensors bit for bit; corrupted
  pads are restored to the original exact normalized ``0``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

#: Frozen, protocol-neutral infrastructure is imported from the v1 helper on purpose:
#: the two protocols must agree byte for byte on the sample-id hash, the seed-word order,
#: the uint8 round trip and the curriculum position.
from utils.dataloader.mmfr_training import (
    CLEAN_PROBABILITY,
    CORRUPTION_SEED,
    DEPTH_PLANE_MEAN,
    DEPTH_PLANE_STD,
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
from utils.dataloader.multimodal_failure import (
    CURRICULUM_LIGHT_MAX_SEVERITY as V1_CURRICULUM_LIGHT_MAX_SEVERITY,
    CURRICULUM_MODERATE_MAX_SEVERITY as V1_CURRICULUM_MODERATE_MAX_SEVERITY,
    CURRICULUM_SEVERITY_FLOOR as V1_CURRICULUM_SEVERITY_FLOOR,
    ENTIRE_MISSING_HEAVY_SEVERITY as V1_ENTIRE_MISSING_HEAVY_SEVERITY,
    FAILURE_KINDS as V1_FAILURE_KINDS,
)
from utils.dataloader.multimodal_failure_v2 import (
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

PROTOCOL_ID = "MMFR-A2-train-integration-v2"
SUPERSEDES = "MMFR-A2-train-integration-v1"
CORRUPTION_BASIS = "MMFR-A1-corruption-basis-v2"
CURRICULUM_KINDS: Tuple[str, ...] = FAILURE_KINDS
DEPTH_MODALITY = "depth"
RGB_MODALITY = "rgb"
RGB_RELIABILITY_INDEX = MODALITY_INDEX[RGB_MODALITY]
DEPTH_RELIABILITY_INDEX = MODALITY_INDEX[DEPTH_MODALITY]
#: The only channel A2 v2 supervises; the RGB channel stays an unscored scaffold.
SUPERVISED_CHANNELS: Tuple[str, ...] = ("depth",)
#: Frozen composition of the supervised Depth surrogate.
TARGET_COMPOSITION = "R_depth_sup(p) = depth_valid_pre(p) * R_depth_synthetic(p)"
#: v1 kwargs this helper deliberately does not provide, so a stale caller fails loudly.
V1_ONLY_KEYS: Tuple[str, ...] = ("depth_valid",)

if tuple(CURRICULUM_KINDS) != tuple(V1_FAILURE_KINDS):
    raise RuntimeError("A1 v2 failure kinds drifted from the frozen v1 six-kind tuple")
if (
    CURRICULUM_SEVERITY_FLOOR != V1_CURRICULUM_SEVERITY_FLOOR
    or CURRICULUM_LIGHT_MAX_SEVERITY != V1_CURRICULUM_LIGHT_MAX_SEVERITY
    or CURRICULUM_MODERATE_MAX_SEVERITY != V1_CURRICULUM_MODERATE_MAX_SEVERITY
    or ENTIRE_MISSING_HEAVY_SEVERITY != V1_ENTIRE_MISSING_HEAVY_SEVERITY
):
    raise RuntimeError("A1 v2 curriculum severities drifted from the frozen v1 values")
if SEVERITY_ENCODING != "single":
    raise RuntimeError("A1 v2 must encode severity exactly once")

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
    "sample_depth_failure_specs_v2",
    "build_mmfr_training_batch_v2",
]


def sample_depth_failure_specs_v2(
    progress: float,
    rng: np.random.Generator,
    max_specs: int = MAX_SPECS,
) -> Tuple[Tuple[FailureSpec, ...], int]:
    """Sample the frozen three-stage curriculum after restricting candidates to Depth.

    Candidate restriction happens *before* drawing the spec count and kinds, so the
    count distribution is not distorted by drawing RGB+Depth specs and filtering RGB
    afterwards. The returned second value is always ``1`` and is kept for audit-schema
    compatibility with the v1 helper.
    """
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

    kinds = tuple(
        kind for kind in CURRICULUM_KINDS if include_entire_missing or kind != "entire_missing"
    )
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
    """Enforce the frozen pad rule: invalid crop/pad pixels stay exact normalized zero."""
    selected = torch.masked_select(tensor, pad)
    if selected.numel() and float(selected.abs().max().item()) != 0.0:
        raise RuntimeError(f"{name} is not exactly zero on the padding region")


def _pixel_statistics(
    valid: np.ndarray,
    depth_pre_uint8: np.ndarray,
    depth_post_uint8: np.ndarray,
) -> Dict[str, int]:
    """Split the valid region into strictly separated invalidity populations.

    ``natural_invalid_pixels`` and ``synthetic_missing_pixels`` are the two sources of
    invalid pixels, but they do not partition ``post_corruption_invalid_pixels``:
    ``gaussian_noise`` and ``misalignment`` can make a natively invalid (``raw Depth ==
    0``) pixel nonzero. Those pixels are counted separately as ``newly_valid_pixels`` and
    the census is closed by

    ``post_corruption_invalid_pixels + newly_valid_pixels
    == natural_invalid_pixels + synthetic_missing_pixels``.

    The identity is asserted fail-closed: a real bookkeeping mistake still stops the run.
    """
    pre_valid = depth_pre_uint8 > 0
    post_valid = depth_post_uint8 > 0
    natural_invalid = int(np.count_nonzero(valid & ~pre_valid))
    synthetic_missing = int(np.count_nonzero(valid & pre_valid & ~post_valid))
    newly_valid = int(np.count_nonzero(valid & ~pre_valid & post_valid))
    post_corruption_invalid = int(np.count_nonzero(valid & ~post_valid))
    if post_corruption_invalid + newly_valid != natural_invalid + synthetic_missing:
        raise RuntimeError(
            "invalidity populations are inconsistent: post_corruption_invalid_pixels + newly_valid_pixels "
            "must equal natural_invalid_pixels + synthetic_missing_pixels "
            f"(got {post_corruption_invalid} + {newly_valid} != {natural_invalid} + {synthetic_missing})"
        )
    return {
        "valid_pixels": int(np.count_nonzero(valid)),
        "natural_invalid_pixels": natural_invalid,
        "synthetic_missing_pixels": synthetic_missing,
        "post_corruption_invalid_pixels": post_corruption_invalid,
        "newly_valid_pixels": newly_valid,
        "implicit_quality_pixels": int(np.count_nonzero(valid & pre_valid & post_valid & (depth_pre_uint8 != depth_post_uint8))),
    }


def build_mmfr_training_batch_v2(
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
    """Apply the frozen A2 v2 Depth corruption to one final CPU training batch.

    ``rgb`` is ``[B, 3, H, W]`` and ``depth`` is ``[B, 1, H, W]`` or ``[B, 3, H, W]`` with
    three identical channels, both normalized float32. ``rgb_mean``/``rgb_std`` are the
    configured RGB normalization statistics; Depth always uses the fixed ``0.48``/``0.28``
    statistics. See the module docstring for the v2 target and validity semantics.
    """
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
        raise RuntimeError("the A2 curriculum must keep the six frozen A1 failure kinds")

    batch = int(rgb_tensor.shape[0])
    normalized_ids = _resolve_sample_ids(sample_ids, batch, sample_id_root)
    progress = curriculum_progress(epoch, iteration, niters_per_epoch, nepochs)
    train_seed = int(corruption_seed)
    if train_seed < 0 or train_seed >= (1 << 32):
        raise ValueError(f"corruption_seed must be a uint32 seed word, got {corruption_seed!r}")
    rank = int(global_rank)
    if rank < 0 or rank >= (1 << 32):
        raise ValueError(f"global_rank must be a uint32 seed word, got {global_rank!r}")
    rgb_mean64 = np.asarray(rgb_mean, dtype=np.float64).reshape(-1)
    rgb_std64 = np.asarray(rgb_std, dtype=np.float64).reshape(-1)
    if rgb_mean64.shape[0] != 3 or rgb_std64.shape[0] != 3:
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
    if depth_channels == 3:
        if not np.array_equal(depth_np[:, 0], depth_np[:, 1]) or not np.array_equal(depth_np[:, 0], depth_np[:, 2]):
            raise ValueError("depth must be single-channel or three identical channels")

    rgb_outputs: List[np.ndarray] = []
    depth_outputs: List[np.ndarray] = []
    raw_rgb_outputs: List[np.ndarray] = []
    raw_depth_outputs: List[np.ndarray] = []
    target_outputs: List[np.ndarray] = []
    valid_outputs: List[np.ndarray] = []
    depth_valid_pre_outputs: List[np.ndarray] = []
    depth_valid_post_outputs: List[np.ndarray] = []
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

        rgb_uint8 = normalized_to_uint8(rgb_sample, rgb_mean64, rgb_std64)
        depth_uint8 = normalized_to_uint8(depth_sample, depth_mean64, depth_std64)[0]
        # Normalized crop padding is exact zero, which inverse-normalizes to the channel
        # means (Depth is about 122). It is not an observation: zero it before any spatial
        # corruption so misalignment/blur cannot move a fabricated mean value into the
        # geometrically valid image region.
        rgb_uint8[:, ~valid] = np.uint8(0)
        depth_uint8[~valid] = np.uint8(0)
        # Pre-corruption *real* validity: the native MUSeg Depth observation, not the pad.
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
        if is_clean:
            rgb_normalized = rgb_sample
            # Exact reuse of the original normalized tensor: no uint8 round trip, no
            # channel broadcast, no copy of the values. The clean path therefore cannot
            # perturb the segmentation input.
            depth_normalized = depth_sample_channels
        else:
            specs, spec_draws = sample_depth_failure_specs_v2(progress, rng, max_specs=max_specs)
            result = apply_failures(
                np.ascontiguousarray(rgb_uint8.transpose(1, 2, 0)),
                depth_uint8,
                specs,
                rng,
            )
            spec_records = tuple(result.metadata.get("specs", ()))
            corrupted_depth_uint8 = np.ascontiguousarray(result.depth)
            # A1 v2 returns the synthetic Depth reliability surrogate ``R_D^syn``.
            synthetic_target = np.ascontiguousarray(result.reliability[DEPTH_RELIABILITY_INDEX], dtype=np.float32)
            corrupted_normalized = uint8_to_normalized(
                corrupted_depth_uint8[:, :, None],
                depth_mean64,
                depth_std64,
            )[:, :, 0]
            depth_normalized = np.where(
                valid[None],
                np.repeat(corrupted_normalized[None], depth_channels, axis=0),
                depth_sample_channels,
            ).astype(np.float32)
            rgb_normalized = rgb_sample

        # The supervised Depth surrogate is ``R_D^sup = V_D^pre * R_D^syn``: native MUSeg
        # Depth invalidity is part of the supervision, while crop/pad stays neutral and is
        # excluded by ``valid_mask``. The RGB channel remains an all-ones scaffold channel
        # because A2 v2 supervises Depth only.
        supervised_depth = np.where(
            depth_valid_pre,
            synthetic_target,
            np.float32(0.0),
        ).astype(np.float32)
        target = np.ones((RELIABILITY_CHANNELS, height, width), dtype=np.float32)
        target[DEPTH_RELIABILITY_INDEX] = supervised_depth
        target[:, ~valid] = np.float32(1.0)
        if float(target[RGB_RELIABILITY_INDEX].min()) != 1.0:
            raise RuntimeError("the RGB reliability scaffold channel must stay exactly 1 in A2 v2")

        raw_rgb = (rgb_uint8.astype(np.float64) / 255.0).astype(np.float32)
        raw_depth = (corrupted_depth_uint8.astype(np.float64) / 255.0).astype(np.float32)
        raw_rgb = np.where(valid[None], raw_rgb, np.float32(0.0)).astype(np.float32)
        raw_depth = np.where(valid[None], raw_depth[None], np.float32(0.0)).astype(np.float32)
        depth_valid_post = (corrupted_depth_uint8 > 0) & valid

        statistics = _pixel_statistics(valid, depth_uint8, corrupted_depth_uint8)
        rgb_outputs.append(rgb_normalized.astype(np.float32))
        depth_outputs.append(depth_normalized.astype(np.float32))
        raw_rgb_outputs.append(raw_rgb)
        raw_depth_outputs.append(raw_depth)
        target_outputs.append(target)
        valid_outputs.append(valid[None])
        depth_valid_pre_outputs.append(depth_valid_pre[None])
        depth_valid_post_outputs.append(depth_valid_post[None])
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
            "depth_target_min": float(target[DEPTH_RELIABILITY_INDEX][valid].min()),
            "depth_target_mean": float(target[DEPTH_RELIABILITY_INDEX][valid].mean()),
            "depth_synthetic_reliability_mean": float(synthetic_target[valid].mean()),
            "valid_fraction": float(valid.mean()),
            "depth_valid_pre_fraction": float(depth_valid_pre.mean()),
            "depth_valid_post_fraction": float(depth_valid_post.mean()),
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
        "protocol": PROTOCOL_ID,
        "supervised_channels": list(SUPERVISED_CHANNELS),
        "metadata": metadata,
    }
    # The segmentation Depth input keeps the caller's channel count on both the clean and
    # the corrupted path, so A2 never changes the batch shape the backbone sees.
    if int(batch_outputs["depth"].shape[1]) != depth_channels:
        raise RuntimeError(
            f"depth channel count changed from {depth_channels} to {batch_outputs['depth'].shape[1]}"
        )
    _validate_batch_outputs_v2(batch_outputs)
    return batch_outputs


def _validate_batch_outputs_v2(outputs: Mapping[str, Any]) -> None:
    """Cheap structural, range and pad invariants of the produced v2 batch."""
    for key in V1_ONLY_KEYS:
        if key in outputs:
            raise RuntimeError(f"A2 v2 must not re-export the ambiguous v1 key {key!r}")
    rgb = outputs["rgb"]
    depth = outputs["depth"]
    raw_rgb = outputs["raw_rgb"]
    raw_depth = outputs["raw_depth"]
    target = outputs["reliability_target"]
    valid_mask = outputs["valid_mask"]
    depth_valid_pre = outputs["depth_valid_pre"]
    depth_valid_post = outputs["depth_valid_post"]
    batch, channels, height, width = tuple(depth.shape)
    if tuple(rgb.shape) != (batch, 3, height, width):
        raise RuntimeError(f"unexpected rgb shape {tuple(rgb.shape)}")
    if tuple(raw_rgb.shape) != (batch, 3, height, width):
        raise RuntimeError(f"unexpected raw_rgb shape {tuple(raw_rgb.shape)}")
    if tuple(raw_depth.shape) != (batch, 1, height, width):
        raise RuntimeError(f"unexpected raw_depth shape {tuple(raw_depth.shape)}")
    if tuple(target.shape) != (batch, RELIABILITY_CHANNELS, height, width):
        raise RuntimeError(f"unexpected reliability_target shape {tuple(target.shape)}")
    for name, mask in (("valid_mask", valid_mask), ("depth_valid_pre", depth_valid_pre), ("depth_valid_post", depth_valid_post)):
        if tuple(mask.shape) != (batch, 1, height, width) or mask.dtype != torch.bool:
            raise RuntimeError(f"unexpected {name} {tuple(mask.shape)} / {mask.dtype}")
    if channels not in (1, 3):
        raise RuntimeError(f"unexpected depth channel count {channels}")

    for name, tensor in (("rgb", rgb), ("depth", depth), ("raw_rgb", raw_rgb), ("raw_depth", raw_depth)):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains non-finite values")
    if target.min() < 0.0 or target.max() > 1.0:
        raise RuntimeError("reliability_target left [0, 1]")
    target_pad = ~valid_mask.expand_as(target)
    if torch.masked_select(target, target_pad).numel() and not torch.all(
        torch.masked_select(target, target_pad) == 1.0
    ):
        raise RuntimeError("reliability_target must be neutral (1) on padding")
    if raw_rgb.min() < 0.0 or raw_rgb.max() > 1.0 or raw_depth.min() < 0.0 or raw_depth.max() > 1.0:
        raise RuntimeError("raw signals left [0, 1]")
    if torch.any(depth_valid_pre & ~valid_mask) or torch.any(depth_valid_post & ~valid_mask):
        raise RuntimeError("Depth validity masks must be subsets of the geometry valid mask")
    if float(target[:, RGB_RELIABILITY_INDEX].min().item()) != 1.0 or float(
        target[:, RGB_RELIABILITY_INDEX].max().item()
    ) != 1.0:
        raise RuntimeError("the RGB reliability scaffold channel must be exactly 1 in A2 v2")
    # The supervised Depth channel must be exactly the pre-corruption validity indicator
    # inside the geometry valid region (blur or noise may legitimately turn a natively
    # invalid pixel into a nonzero one, so post is not required to be a subset of pre).
    supervised = target[:, DEPTH_RELIABILITY_INDEX]
    valid_plane = valid_mask[:, 0]
    pre_plane = depth_valid_pre[:, 0]
    outside_native_validity = valid_plane & ~pre_plane
    if bool(outside_native_validity.any()) and float(supervised[outside_native_validity].abs().max().item()) != 0.0:
        raise RuntimeError("the supervised Depth target must be exactly 0 where native Depth is invalid")
    if supervised.min() < 0.0 or supervised.max() > 1.0:
        raise RuntimeError("the supervised Depth target left [0, 1]")

    pad = ~valid_mask.expand(batch, channels, height, width)
    for name, tensor in (("rgb", rgb), ("depth", depth)):
        _assert_tensor_is_zero_at_pad(tensor, pad, name)
    for name, tensor in (("raw_rgb", raw_rgb), ("raw_depth", raw_depth)):
        _assert_tensor_is_zero_at_pad(tensor, pad[:, :1].expand_as(tensor), name)
