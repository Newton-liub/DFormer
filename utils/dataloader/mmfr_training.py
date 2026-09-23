#!/usr/bin/env python3
"""Shared Depth-corruption primitives and frozen A1 constants for the MMFR A2 line.

``MMFR-A2-train-integration-v3`` and the frozen MUSeg evaluators import these helpers:
the deterministic per-sample seed words, the uint8 round trip against the fixed Depth
normalization, the sample-id normalization, and the frozen constants (corruption seed,
``p_clean``, ``max_specs``, six-kind curriculum tuple, normalized pad value).

Frozen semantics
----------------
* Each sample gets its own stateless ``numpy.random.Generator(PCG64(SeedSequence(
  words)))``; process-wide Python/NumPy/PyTorch random state is never read or written.
* ``words`` order is ``train_seed, epoch(1-based), iteration(0-based), global_rank,
  sample_slot, sample_id_sha256_u32_be_0..3``. The 16 leading SHA-256 bytes of the
  sample id (path separators unified to ``/``) are split big-endian into four uint32.
* Only Depth specs are applied. ``p_clean = 0.25``; corrupt samples call the frozen
  A1 curriculum with ``max_specs = 2`` and the six A1 failure kinds.
* Values are recovered from the normalized crop by a uint8 round trip: RGB uses the
  config mean/std, Depth the fixed ``0.48``/``0.28`` stats of ``TrainPre(sign=True)``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence, Tuple

import numpy as np

#: Frozen A1 six-kind tuple. It lived in the retired ``multimodal_failure.py`` (A1 v1) and is
#: kept here because it is the only place the A2 v3 helper and this shared module both consume.
FAILURE_KINDS: Tuple[str, ...] = (
    "entire_missing",
    "spatial_dropout",
    "gaussian_noise",
    "blur",
    "quantization",
    "misalignment",
)
#: Frozen A1 curriculum severities: severity floor, light/moderate caps and the
#: ``entire_missing`` heavy severity. Values are unchanged from A1 v1.
CURRICULUM_SEVERITY_FLOOR = 0.05
CURRICULUM_LIGHT_MAX_SEVERITY = 0.3
CURRICULUM_MODERATE_MAX_SEVERITY = 0.6
ENTIRE_MISSING_HEAVY_SEVERITY = 1.0

#: Training corruption seed, deliberately independent of the model seed.
CORRUPTION_SEED = 2026091402
CLEAN_PROBABILITY = 0.25
MAX_SPECS = 2
CURRICULUM_KINDS: Tuple[str, ...] = FAILURE_KINDS
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
    "FAILURE_KINDS",
    "CURRICULUM_SEVERITY_FLOOR",
    "CURRICULUM_LIGHT_MAX_SEVERITY",
    "CURRICULUM_MODERATE_MAX_SEVERITY",
    "ENTIRE_MISSING_HEAVY_SEVERITY",
    "CORRUPTION_SEED",
    "CLEAN_PROBABILITY",
    "MAX_SPECS",
    "CURRICULUM_KINDS",
    "DEPTH_NORMALIZATION_MEAN",
    "DEPTH_NORMALIZATION_STD",
    "DEPTH_PLANE_MEAN",
    "DEPTH_PLANE_STD",
    "NORMALIZED_PAD_VALUE",
    "curriculum_progress",
    "normalize_sample_id",
    "sample_id_words",
    "build_seed_words",
    "make_sample_generator",
    "normalized_to_uint8",
    "uint8_to_normalized",
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
    byte the round trip ``uint8 -> normalize -> float32 -> this function`` is exact.
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
