#!/usr/bin/env python3
"""Post-crop / pre-GPU Depth corruption batch helper for ``MMFR-A2-train-integration-v1``.

The helper turns the final CPU training batch that the ``DataLoader`` already
mirrored, scaled, cropped and padded into the A2 batch contract:

* ``rgb``, ``depth``: normalized float32 tensors for the segmentation backbone, with
  the Depth channel count equal to the input channel count (the frozen loader supplies
  three identical Depth channels). A clean sample is bit-identical to the pre-A2 input;
* ``raw_rgb``, ``raw_depth``: ``[0, 1]`` float32 signals for the reliability head;
* ``reliability_target``: ``[B, 2, H, W]`` continuous target, channel order RGB then
  Depth;
* ``valid_mask``: ``[B, 1, H, W]`` bool geometry mask that excludes only crop/pad;
* ``depth_valid``: ``[B, 1, H, W]`` bool mask from corrupted raw Depth ``> 0`` and
  ``valid_mask``;
* ``metadata``: per-sample CPU audit records (sample id, seed words, spec order,
  severities, reliability summary).

Frozen semantics
----------------
* Corruption runs in the main training process, after the frozen ``DataLoader``
  geometry (mirror/scale/crop/pad) and before the batch is moved to the GPU. Neither
  ``TrainPre`` nor ``RGBXDataset`` is touched.
* Each sample gets its own stateless ``numpy.random.Generator(PCG64(SeedSequence(
  words)))``; process-wide Python/NumPy/PyTorch random state is never read or written.
* ``words`` order is ``train_seed, epoch(1-based), iteration(0-based), global_rank,
  sample_slot, sample_id_sha256_u32_be_0..3``. The 16 leading SHA-256 bytes of the
  sample id (path separators unified to ``/``) are split big-endian into four uint32.
* Only Depth specs are applied. ``p_clean = 0.25``; corrupt samples call the frozen
  A1 curriculum with ``max_specs = 2`` and the six A1 failure kinds.
* ``progress`` is ``((epoch - 1) * N_iter + iteration) / (N_epoch * N_iter - 1)``,
  clipped to ``[0, 1]``.
* Values are recovered from the normalized crop by an exhaustive-qualified uint8
  round trip: RGB uses the config mean/std, Depth the fixed ``0.48``/``0.28`` stats of
  ``TrainPre(sign=True)``. A crop/pad pixel is geometrically invalid exactly when RGB
  and Depth are both all-channel exact zeros; corrupted pads are restored to the
  original exact normalized ``0`` and clean samples reuse the original tensors bit for
  bit, so the clean path applies no reconstruction at all.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from utils.dataloader.multimodal_failure import (
    CURRICULUM_LIGHT_MAX_SEVERITY,
    CURRICULUM_MODERATE_MAX_SEVERITY,
    CURRICULUM_SEVERITY_FLOOR,
    ENTIRE_MISSING_HEAVY_SEVERITY,
    FAILURE_KINDS,
    MODALITY_INDEX,
    RELIABILITY_CHANNELS,
    FailureSpec,
    apply_failures,
)

PROTOCOL_ID = "MMFR-A2-train-integration-v1"
#: Training corruption seed, deliberately independent of the model seed.
CORRUPTION_SEED = 2026091402
CLEAN_PROBABILITY = 0.25
MAX_SPECS = 2
CURRICULUM_KINDS: Tuple[str, ...] = FAILURE_KINDS
DEPTH_MODALITY = "depth"
RGB_MODALITY = "rgb"
RGB_RELIABILITY_INDEX = MODALITY_INDEX[RGB_MODALITY]
DEPTH_RELIABILITY_INDEX = MODALITY_INDEX[DEPTH_MODALITY]
#: Fixed Depth normalization of ``TrainPre(sign=True)``; identical on all channels.
DEPTH_NORMALIZATION_MEAN: Tuple[float, ...] = (0.48, 0.48, 0.48)
DEPTH_NORMALIZATION_STD: Tuple[float, ...] = (0.28, 0.28, 0.28)
DEPTH_PLANE_MEAN = DEPTH_NORMALIZATION_MEAN[0]
DEPTH_PLANE_STD = DEPTH_NORMALIZATION_STD[0]
SAMPLE_ID_HASH_BYTES = 16
UINT32_LIMIT = 1 << 32
#: Seed words are ``train_seed``, four position words and four id words.
SEED_WORD_COUNT = 5 + SAMPLE_ID_HASH_BYTES // 4
#: Frozen normalized value of a training crop/pad pixel; the pad test uses exact equality.
NORMALIZED_PAD_VALUE = 0.0

if len(set(DEPTH_NORMALIZATION_MEAN)) != 1 or len(set(DEPTH_NORMALIZATION_STD)) != 1:
    raise RuntimeError("Depth normalization must be uniform across channels")

__all__ = [
    "PROTOCOL_ID",
    "CORRUPTION_SEED",
    "CLEAN_PROBABILITY",
    "MAX_SPECS",
    "CURRICULUM_KINDS",
    "DEPTH_NORMALIZATION_MEAN",
    "DEPTH_NORMALIZATION_STD",
    "curriculum_progress",
    "normalize_sample_id",
    "sample_id_words",
    "build_seed_words",
    "make_sample_generator",
    "normalized_to_uint8",
    "uint8_to_normalized",
    "qualify_uint8_round_trip",
    "sample_depth_failure_specs",
    "build_mmfr_training_batch",
]


def _as_uint32(value: Any, name: str) -> int:
    number = int(value)
    if number < 0 or number >= UINT32_LIMIT:
        raise ValueError(f"{name} must be a uint32 seed word, got {value!r}")
    return number


def _as_per_channel_stats(values: Any, channels: int, name: str) -> np.ndarray:
    stats = np.asarray(values, dtype=np.float64).reshape(-1)
    if stats.shape[0] != channels:
        raise ValueError(f"{name} must hold {channels} per-channel values, got {stats.shape[0]}")
    if not np.isfinite(stats).all():
        raise ValueError(f"{name} contains non-finite values")
    return stats


def curriculum_progress(epoch: int, iteration: int, niters_per_epoch: int, nepochs: int) -> float:
    """Frozen A2 curriculum position of the current attempt, clipped to ``[0, 1]``."""
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError(f"epoch must be a 1-based int, got {epoch!r}")
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"iteration must be a 0-based int, got {iteration!r}")
    if not isinstance(niters_per_epoch, int) or niters_per_epoch < 1:
        raise ValueError(f"niters_per_epoch must be a positive int, got {niters_per_epoch!r}")
    if not isinstance(nepochs, int) or nepochs < 1:
        raise ValueError(f"nepochs must be a positive int, got {nepochs!r}")
    total = nepochs * niters_per_epoch - 1
    if total <= 0:
        return 0.0
    value = ((epoch - 1) * niters_per_epoch + iteration) / float(total)
    return float(min(1.0, max(0.0, value)))


def normalize_sample_id(sample_id: str, root: str | None = None) -> str:
    """Return the auditable sample id used for the SHA-256 seed words.

    Path separators are unified to ``/``. When ``root`` is given and the id is inside
    it, the root prefix is stripped so the hashed identity stays stable across
    machines; otherwise the normalized path is used unchanged.
    """
    normalized = str(sample_id).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if root:
        prefix = str(root).strip().replace("\\", "/").rstrip("/")
        if prefix and normalized.startswith(prefix + "/"):
            normalized = normalized[len(prefix) + 1 :]
    if not normalized:
        raise ValueError("sample id must be a non-empty path")
    return normalized


def sample_id_words(normalized_sample_id: str) -> Tuple[int, int, int, int]:
    """Split the 16 leading SHA-256 bytes of the sample id into four big-endian uint32."""
    digest = hashlib.sha256(str(normalized_sample_id).encode("utf-8")).digest()
    head = digest[:SAMPLE_ID_HASH_BYTES]
    return tuple(int.from_bytes(head[index * 4 : (index + 1) * 4], "big") for index in range(4))  # type: ignore[return-value]


def build_seed_words(
    train_seed: int,
    epoch: int,
    iteration: int,
    global_rank: int,
    sample_slot: int,
    sample_words: Sequence[int],
) -> Tuple[int, ...]:
    """Assemble the frozen nine-word seed sequence in its frozen order."""
    words = (
        _as_uint32(train_seed, "train_seed"),
        _as_uint32(epoch, "epoch"),
        _as_uint32(iteration, "iteration"),
        _as_uint32(global_rank, "global_rank"),
        _as_uint32(sample_slot, "sample_slot"),
    ) + tuple(_as_uint32(word, "sample_id_word") for word in sample_words)
    if len(words) != SEED_WORD_COUNT:
        raise ValueError(f"expected {SEED_WORD_COUNT} seed words, got {len(words)}")
    return words


def make_sample_generator(words: Sequence[int]) -> np.random.Generator:
    """Build the stateless per-sample generator without touching global RNG state."""
    entropy = [_as_uint32(word, "seed word") for word in words]
    if len(entropy) != SEED_WORD_COUNT:
        raise ValueError(f"expected {SEED_WORD_COUNT} seed words, got {len(entropy)}")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def normalized_to_uint8(normalized: np.ndarray, mean: Any, std: Any) -> np.ndarray:
    """Recover uint8 from a ``[C, H, W]`` normalized float array.

    The inverse runs in float64 from the float32 tensor values, so for every legal
    byte the round trip ``uint8 -> normalize -> float32 -> this function`` is exact;
    :func:`qualify_uint8_round_trip` verifies that exhaustively for all 256 values.
    """
    values = np.asarray(normalized, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"normalized must be [C, H, W], got shape {values.shape}")
    mean64 = _as_per_channel_stats(mean, values.shape[0], "mean")
    std64 = _as_per_channel_stats(std, values.shape[0], "std")
    if (std64 <= 0.0).any():
        raise ValueError("std must be strictly positive")
    restored = np.rint((values * std64[:, None, None] + mean64[:, None, None]) * 255.0)
    return np.clip(restored, 0.0, 255.0).astype(np.uint8)


def uint8_to_normalized(images: np.ndarray, mean: Any, std: Any) -> np.ndarray:
    """Forward normalization matching ``utils.transforms.normalize`` plus float32 cast.

    ``images`` is uint8 with channels on the last axis (``H x W x C``); the result is
    float32 with the same shape, which is exactly what the frozen training transform
    would have produced for those bytes.
    """
    values = np.asarray(images)
    if values.dtype != np.uint8:
        raise ValueError(f"images must be uint8, got {values.dtype}")
    if values.ndim not in (2, 3):
        raise ValueError(f"images must be HxW or HxWxC, got shape {values.shape}")
    channels = 1 if values.ndim == 2 else values.shape[-1]
    mean64 = _as_per_channel_stats(mean, channels, "mean")
    std64 = _as_per_channel_stats(std, channels, "std")
    normalized = values.astype(np.float64) / 255.0
    shape = (1, 1, channels) if values.ndim == 3 else (1, 1, 1)
    normalized = (normalized - mean64.reshape(shape)) / std64.reshape(shape)
    return normalized.astype(np.float32)


def qualify_uint8_round_trip(mean: Any, std: Any) -> None:
    """Exhaustive ``0..255`` qualification of the normalization round trip.

    Raises ``RuntimeError`` with the failing bytes when any value is not recovered,
    which is the frozen ``qualification-blocked`` condition of the A2 protocol.
    """
    values = np.arange(256, dtype=np.uint8)
    mean64 = np.asarray(mean, dtype=np.float64).reshape(-1)
    std64 = np.asarray(std, dtype=np.float64).reshape(-1)
    if mean64.shape != std64.shape:
        raise ValueError("mean and std must share their per-channel shape")
    for channel in range(mean64.shape[0]):
        normalized = (values.astype(np.float64) / 255.0 - mean64[channel]) / std64[channel]
        normalized32 = normalized.astype(np.float32).reshape(1, 256, 1)
        restored = normalized_to_uint8(
            normalized32,
            mean64[channel : channel + 1],
            std64[channel : channel + 1],
        ).reshape(-1)
        if not np.array_equal(restored, values):
            mismatches = np.flatnonzero(restored != values)[:8].tolist()
            raise RuntimeError(
                f"uint8 round trip failed for channel {channel} (mean={mean64[channel]}, "
                f"std={std64[channel]}) at bytes {mismatches}"
            )


def sample_depth_failure_specs(
    progress: float,
    rng: np.random.Generator,
    max_specs: int = MAX_SPECS,
) -> Tuple[Tuple[FailureSpec, ...], int]:
    """Sample the frozen A1 curriculum after restricting candidates to Depth.

    Candidate restriction happens before drawing the spec count and kinds. This keeps
    the A1 three-stage severity/count rules while avoiding the count distortion caused
    by drawing RGB+Depth specs and filtering RGB afterwards. The second return value is
    retained for audit-schema compatibility and is always one because no re-draw occurs.
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


def _assert_tensor_is_zero_at_pad(
    tensor: torch.Tensor,
    pad: torch.Tensor,
    name: str,
) -> None:
    """Enforce the frozen pad rule: invalid crop/pad pixels stay exact normalized zero."""
    selected = torch.masked_select(tensor, pad)
    if selected.numel() and float(selected.abs().max().item()) != 0.0:
        raise RuntimeError(f"{name} is not exactly zero on the padding region")


def build_mmfr_training_batch(
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
    """Apply the frozen A2 Depth corruption to one final CPU training batch.

    ``rgb`` is ``[B, 3, H, W]`` and ``depth`` is ``[B, 1, H, W]`` or ``[B, 3, H, W]``
    with three identical channels, both normalized float32. ``rgb_mean``/``rgb_std``
    are the configured RGB normalization statistics; Depth always uses the fixed
    ``0.48``/``0.28`` statistics.
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
    train_seed = _as_uint32(corruption_seed, "corruption_seed")
    rank = _as_uint32(global_rank, "global_rank")
    rgb_mean64 = _as_per_channel_stats(rgb_mean, 3, "rgb_mean")
    rgb_std64 = _as_per_channel_stats(rgb_std, 3, "rgb_std")
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
    depth_valid_outputs: List[np.ndarray] = []
    metadata: List[Mapping[str, Any]] = []

    for slot in range(batch):
        rgb_sample = np.ascontiguousarray(rgb_np[slot], dtype=np.float32)
        #: Full-channel Depth plane of this sample; re-used bit for bit on the clean path
        #: so the helper never changes the segmentation input on a clean sample.
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
        # means (Depth is about 122). It is not an observation: zero it before any
        # spatial corruption so misalignment/blur cannot move a fabricated mean value
        # into the geometrically valid image region.
        rgb_uint8[:, ~valid] = np.uint8(0)
        depth_uint8[~valid] = np.uint8(0)

        sample_words = sample_id_words(normalized_ids[slot])
        words = build_seed_words(train_seed, epoch, iteration, rank, slot, sample_words)
        rng = make_sample_generator(words)
        is_clean = bool(float(rng.random()) < float(p_clean))

        specs: Tuple[FailureSpec, ...] = ()
        spec_draws = 0
        spec_records: Tuple[Mapping[str, Any], ...] = ()
        corrupted_depth_uint8 = depth_uint8
        if is_clean:
            rgb_normalized = rgb_sample
            # Exact reuse of the original normalized tensor: no uint8 round trip, no
            # channel broadcast, no copy of the values (``astype`` below only copies
            # bytes). The clean path therefore cannot perturb the segmentation input.
            depth_normalized = depth_sample_channels
            target = np.ones((RELIABILITY_CHANNELS, height, width), dtype=np.float32)
        else:
            specs, spec_draws = sample_depth_failure_specs(progress, rng, max_specs=max_specs)
            result = apply_failures(
                np.ascontiguousarray(rgb_uint8.transpose(1, 2, 0)),
                depth_uint8,
                specs,
                rng,
            )
            spec_records = tuple(result.metadata.get("specs", ()))
            corrupted_depth_uint8 = np.ascontiguousarray(result.depth)
            target = np.ascontiguousarray(result.reliability, dtype=np.float32)
            # Padding is outside the image geometry, not a failed sensor observation.
            # Keep it neutral in the target as well as excluding it with valid_mask.
            target[:, ~valid] = np.float32(1.0)
            if float(target[RGB_RELIABILITY_INDEX].min()) != 1.0:
                raise RuntimeError("Depth-only A2 corruption must keep the RGB reliability target at 1")
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

        raw_rgb = (rgb_uint8.astype(np.float64) / 255.0).astype(np.float32)
        raw_depth = (corrupted_depth_uint8.astype(np.float64) / 255.0).astype(np.float32)
        raw_rgb = np.where(valid[None], raw_rgb, np.float32(0.0)).astype(np.float32)
        raw_depth = np.where(valid[None], raw_depth[None], np.float32(0.0)).astype(np.float32)
        depth_valid = (raw_depth > 0.0) & valid[None]

        rgb_outputs.append(rgb_normalized.astype(np.float32))
        depth_outputs.append(depth_normalized.astype(np.float32))
        raw_rgb_outputs.append(raw_rgb)
        raw_depth_outputs.append(raw_depth)
        target_outputs.append(target)
        valid_outputs.append(valid[None])
        depth_valid_outputs.append(depth_valid)
        metadata.append(
            {
                "sample_slot": slot,
                "sample_id": normalized_ids[slot],
                "seed_words": words,
                "curriculum_progress": progress,
                "clean": is_clean,
                "spec_draws": spec_draws,
                "num_specs": len(specs),
                "specs": spec_records,
                "depth_reliability_min": float(target[DEPTH_RELIABILITY_INDEX][valid].min()),
                "depth_reliability_mean": float(target[DEPTH_RELIABILITY_INDEX][valid].mean()),
                "valid_fraction": float(valid.mean()),
                "depth_valid_fraction": float(depth_valid.mean()),
            }
        )

    def _stack(arrays: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(np.stack(arrays, axis=0)))

    batch_outputs: Dict[str, Any] = {
        "rgb": _stack(rgb_outputs),
        "depth": _stack(depth_outputs),
        "raw_rgb": _stack(raw_rgb_outputs),
        "raw_depth": _stack(raw_depth_outputs),
        "reliability_target": _stack(target_outputs),
        "valid_mask": _stack(valid_outputs),
        "depth_valid": _stack(depth_valid_outputs),
        "metadata": metadata,
    }
    # The segmentation Depth input keeps the caller's channel count on both the clean and
    # the corrupted path, so A2 never changes the batch shape the backbone sees.
    if int(batch_outputs["depth"].shape[1]) != depth_channels:
        raise RuntimeError(
            f"depth channel count changed from {depth_channels} to {batch_outputs['depth'].shape[1]}"
        )
    _validate_batch_outputs(batch_outputs)
    return batch_outputs


def _validate_batch_outputs(outputs: Mapping[str, Any]) -> None:
    """Cheap structural, range and pad invariants of the produced batch."""
    rgb = outputs["rgb"]
    depth = outputs["depth"]
    raw_rgb = outputs["raw_rgb"]
    raw_depth = outputs["raw_depth"]
    target = outputs["reliability_target"]
    valid_mask = outputs["valid_mask"]
    depth_valid = outputs["depth_valid"]
    shapes = {
        "rgb": tuple(rgb.shape),
        "depth": tuple(depth.shape),
        "raw_rgb": tuple(raw_rgb.shape),
        "raw_depth": tuple(raw_depth.shape),
        "reliability_target": tuple(target.shape),
    }
    batch, channels, height, width = shapes["depth"]
    if shapes["rgb"] != (batch, 3, height, width):
        raise RuntimeError(f"unexpected rgb shape {shapes['rgb']}")
    if shapes["raw_rgb"] != (batch, 3, height, width):
        raise RuntimeError(f"unexpected raw_rgb shape {shapes['raw_rgb']}")
    if shapes["raw_depth"] != (batch, 1, height, width):
        raise RuntimeError(f"unexpected raw_depth shape {shapes['raw_depth']}")
    if shapes["reliability_target"] != (batch, RELIABILITY_CHANNELS, height, width):
        raise RuntimeError(f"unexpected reliability_target shape {shapes['reliability_target']}")
    if tuple(valid_mask.shape) != (batch, 1, height, width) or valid_mask.dtype != torch.bool:
        raise RuntimeError(f"unexpected valid_mask {tuple(valid_mask.shape)} / {valid_mask.dtype}")
    if tuple(depth_valid.shape) != (batch, 1, height, width) or depth_valid.dtype != torch.bool:
        raise RuntimeError(f"unexpected depth_valid {tuple(depth_valid.shape)} / {depth_valid.dtype}")
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
    if torch.any(depth_valid & ~valid_mask):
        raise RuntimeError("depth_valid must be a subset of the geometry valid mask")

    pad = ~valid_mask.expand(batch, channels, height, width)
    for name, tensor in (("rgb", rgb), ("depth", depth)):
        _assert_tensor_is_zero_at_pad(tensor, pad, name)
    for name, tensor in (("raw_rgb", raw_rgb), ("raw_depth", raw_depth)):
        _assert_tensor_is_zero_at_pad(tensor, pad[:, :1].expand_as(tensor), name)
