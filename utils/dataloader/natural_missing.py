"""CPU-only paired NaturalMissing input augmentation for MUSeg depth data.

This module handles only training-input masks.  It does not import the model,
optimizer, loss, checkpoint, or GPU runtime.  Replay source identities are built
exclusively from ``config.train_source`` and raw invalid pixels are defined by
``Depth16 == 0``.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch


STRATEGIES = ("Natural", "Grid", "Replay")
CLEAN_SLOT_FRACTION = 0.25
MIN_ADDED_VALID_RATE = 0.10
MAX_ADDED_VALID_RATE = 0.50
GRID_MATCH_TOLERANCE = 0.02
MAX_CANDIDATES = 8
DEPTH16_MAX_RAW = 13932
_CANONICAL_TRAIN_SOURCE = os.path.normcase(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "splits",
            "MUSeg",
            "dev-v1",
            "train-dev.txt",
        )
    )
)
_CANONICAL_TRAIN_SOURCE_SHA256 = "a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470"


@dataclass(frozen=True)
class SourceEntry:
    """One train-source RGB sample and its raw Depth16 source mask path."""

    sample_id: str
    group_id: str
    depth16_path: str


@dataclass(frozen=True)
class NaturalMissingSourcePool:
    """Lightweight train-only source index; image masks are read on demand."""

    entries: Tuple[SourceEntry, ...]
    depth16_root: str
    _by_group: Mapping[str, Tuple[SourceEntry, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        by_group: Dict[str, list] = defaultdict(list)
        for entry in self.entries:
            by_group[entry.group_id].append(entry)
        object.__setattr__(self, "_by_group", {key: tuple(value) for key, value in by_group.items()})

    @property
    def groups(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_group))

    def candidates(
        self, target_group: str, rng: random.Random, limit: int = MAX_CANDIDATES
    ) -> Tuple[SourceEntry, ...]:
        """Sample distinct groups uniformly, then one frame from each group."""
        groups = [group for group in self._by_group if group != target_group]
        rng.shuffle(groups)
        selected = []
        for group in groups[: max(0, int(limit))]:
            selected.append(rng.choice(self._by_group[group]))
        return tuple(selected)

    def read_invalid_mask(self, entry: SourceEntry) -> np.ndarray:
        return _read_depth16_invalid_mask(entry.depth16_path, entry.sample_id)


@lru_cache(maxsize=64)
def _read_depth16_invalid_mask(path: str, sample_id: str) -> np.ndarray:
    depth16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth16 is None:
        raise FileNotFoundError(
            "cannot read train-only NaturalMissing source Depth16: " + path
        )
    if depth16.dtype != np.uint16 or depth16.ndim != 2:
        raise ValueError(
            f"{sample_id}: expected 2-D uint16 Depth16, "
            f"got shape={depth16.shape}, dtype={depth16.dtype}"
        )
    if depth16.size == 0 or int(depth16.max()) > DEPTH16_MAX_RAW:
        raise ValueError(f"{sample_id}: Depth16 values outside the audited MUSeg range")
    return depth16 == 0


def _depth16_file(root: str, sample_id: str) -> str:
    root_path = Path(root)
    if root_path.name.lower() == "depth16":
        path = root_path / (sample_id + ".png")
    else:
        path = root_path / "Depth16" / (sample_id + ".png")
    return str(path)


@lru_cache(maxsize=4)
def _read_source_entries(train_source: str, depth16_root: str) -> NaturalMissingSourcePool:
    # This function receives only the frozen canonical train-dev path. Validate
    # that identity in build_source_pool before any file access; the digest also
    # prevents a different split revision at the same path from becoming a pool.
    with open(train_source, "rb") as source_file:
        source_bytes = source_file.read()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != _CANONICAL_TRAIN_SOURCE_SHA256:
        raise ValueError(
            "canonical NaturalMissing train-dev list has unexpected SHA256: "
            f"{actual_sha256}"
        )
    source_text = source_bytes.decode("utf-8-sig")

    entries = []
    seen = set()
    for line_number, line in enumerate(source_text.splitlines(), start=1):
        item = line.strip().replace("\\", "/")
        if not item:
            continue
        rel = PurePosixPath(item)
        if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != "RGB" or rel.suffix.lower() != ".jpg":
            raise ValueError(
                f"{train_source}:{line_number}: expected a train-only RGB/*.jpg entry, got {item!r}"
            )
        sample_id = Path(rel.name).stem
        if sample_id in seen:
            raise ValueError(f"{train_source}:{line_number}: duplicate sample {sample_id!r}")
        seen.add(sample_id)
        parts = sample_id.split("-")
        if len(parts) != 7 or not all(re.fullmatch(r"[0-9]+", part) for part in parts):
            raise ValueError(f"{train_source}:{line_number}: unexpected MUSeg sample identity {sample_id!r}")
        entries.append(
            SourceEntry(
                sample_id=sample_id,
                group_id="-".join(parts[:4]),
                depth16_path=_depth16_file(depth16_root, sample_id),
            )
        )
    if not entries:
        raise ValueError(f"train-only NaturalMissing source list is empty: {train_source}")
    return NaturalMissingSourcePool(tuple(entries), depth16_root)


def build_source_pool(train_source: str, depth16_root: str) -> NaturalMissingSourcePool:
    """Build/reuse a pool only from the frozen canonical train-dev list."""
    if not train_source:
        raise ValueError("config.train_source is required for Replay/Grid paired planning")
    if not depth16_root:
        raise ValueError("natural_missing.depth16_root is required for Replay/Grid paired planning")
    source_identity = os.path.normcase(os.path.abspath(os.fspath(train_source)))
    if source_identity != _CANONICAL_TRAIN_SOURCE:
        raise ValueError(
            "NaturalMissing source pool accepts only the frozen canonical "
            "data/splits/MUSeg/dev-v1/train-dev.txt path"
        )
    return _read_source_entries(source_identity, os.path.abspath(depth16_root))


def parse_museg_identity(sample: str) -> Tuple[str, str]:
    """Return ``(sample_id, first-four-segment group_id)`` from a path or stem."""
    stem = Path(str(sample).replace("\\", "/")).stem
    parts = stem.split("-")
    if len(parts) != 7 or not all(re.fullmatch(r"[0-9]+", part) for part in parts):
        raise ValueError(f"unexpected MUSeg sample identity: {sample!r}")
    return stem, "-".join(parts[:4])


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(int(seed))] + [str(part) for part in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _zero_plan(shape: Tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.bool_)


def _transform_source_mask(
    source_invalid: np.ndarray,
    transform: Mapping[str, Any],
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """Resize source mask to target raw size, then mirror/scale/crop/pad with NN."""
    raw_h = int(transform["raw_height"])
    raw_w = int(transform["raw_width"])
    mask = np.asarray(source_invalid, dtype=np.uint8)
    if mask.ndim != 2 or not mask.size:
        raise ValueError(f"source invalid mask must be a non-empty 2-D array, got {mask.shape}")
    if mask.shape != (raw_h, raw_w):
        mask = cv2.resize(mask, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
    if bool(transform["mirror"]):
        mask = cv2.flip(mask, 1)

    scaled_h = int(transform["scaled_height"])
    scaled_w = int(transform["scaled_width"])
    if mask.shape != (scaled_h, scaled_w):
        mask = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

    crop_y = int(transform["crop_y"])
    crop_x = int(transform["crop_x"])
    crop_h = int(transform["crop_height"])
    crop_w = int(transform["crop_width"])
    if crop_y < 0 or crop_x < 0 or crop_h < 1 or crop_w < 1:
        raise ValueError(f"invalid recorded crop metadata: {transform}")
    mask = mask[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    if mask.shape != (crop_h, crop_w):
        raise ValueError("recorded crop extends beyond the scaled target geometry")

    pad_top = int(transform["pad_top"])
    pad_bottom = int(transform["pad_bottom"])
    pad_left = int(transform["pad_left"])
    pad_right = int(transform["pad_right"])
    if any(value < 0 for value in (pad_top, pad_bottom, pad_left, pad_right)):
        raise ValueError("recorded padding must be non-negative")
    if any((pad_top, pad_bottom, pad_left, pad_right)):
        mask = cv2.copyMakeBorder(
            mask, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0,
        )
    if mask.shape != output_shape:
        raise ValueError(f"transformed source mask {mask.shape} does not match target grid {output_shape}")
    return mask.astype(np.bool_, copy=False)


def _grid_cell_size(height: int, width: int, attempt: int) -> int:
    base = max(8, min(height, width) // 16)
    scales = (1.0, 0.75, 1.25, 0.5, 1.5, 0.625, 0.875, 0.375)
    return max(8, int(round(base * scales[attempt % len(scales)])))


def _grid_candidate(
    target_valid: np.ndarray,
    target_delete_count: int,
    attempt: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one ordinary random grid-block deletion candidate."""
    height, width = target_valid.shape
    cell = _grid_cell_size(height, width, attempt)
    grid_h = int(math.ceil(height / float(cell)))
    grid_w = int(math.ceil(width / float(cell)))
    total = grid_h * grid_w
    valid_count = int(np.count_nonzero(target_valid))
    if valid_count <= 0:
        return _zero_plan(target_valid.shape)

    rate = float(target_delete_count) / float(valid_count)
    offsets = (0, -1, 1, -2, 2, -3, 3, -4)
    block_count = int(round(rate * total)) + offsets[attempt % len(offsets)]
    block_count = min(total, max(1, block_count))
    chosen = rng.choice(total, size=block_count, replace=False)
    cells = np.zeros(total, dtype=np.bool_)
    cells[np.asarray(chosen, dtype=np.int64)] = True
    mask = np.repeat(np.repeat(cells.reshape(grid_h, grid_w), cell, axis=0), cell, axis=1)
    return mask[:height, :width]


@dataclass(frozen=True)
class PairedPlan:
    """One deterministic Replay/Grid slot plan; skip means neither arm applies."""

    target_sample: str
    target_group: str
    source_sample: Optional[str]
    source_group: Optional[str]
    replay_source_mask: np.ndarray
    replay_delete_mask: np.ndarray
    grid_delete_mask: np.ndarray
    target_valid_count: int
    replay_deletion_count: int
    grid_deletion_count: int
    replay_rate: Optional[float]
    grid_rate: Optional[float]
    matching_error: Optional[float]
    attempts: int
    candidate_sources: Tuple[Tuple[str, str], ...]
    clean_slot: bool
    reason: str

    @property
    def matched(self) -> bool:
        return self.reason == "matched"


def build_paired_plan(
    target_raw_depth: np.ndarray,
    target_support: np.ndarray,
    transform: Mapping[str, Any],
    source_pool: NaturalMissingSourcePool,
    *,
    target_sample: str,
    target_group: str,
    seed: int,
    successful_update: int,
    epoch: int = 0,
    batch_slot: int = 0,
    clean_slot_fraction: float = CLEAN_SLOT_FRACTION,
    min_added_valid_rate: float = MIN_ADDED_VALID_RATE,
    max_added_valid_rate: float = MAX_ADDED_VALID_RATE,
    matching_tolerance: float = GRID_MATCH_TOLERANCE,
    max_candidates: int = MAX_CANDIDATES,
) -> PairedPlan:
    """Create a shared deterministic source mask and deletion-matched grid plan.

    ``target_raw_depth`` is the current uint8 Depth plane after the target's linear
    resize and crop/pad.  Existing missing pixels are never counted as new deletion.
    """
    raw = np.asarray(target_raw_depth)
    support = np.asarray(target_support, dtype=np.bool_)
    if raw.ndim != 2 or raw.shape != support.shape:
        raise ValueError(f"target raw depth/support must share HxW shape, got {raw.shape}/{support.shape}")
    if raw.dtype != np.uint8:
        raise ValueError(f"target raw depth must be uint8, got {raw.dtype}")
    if not 0.0 <= clean_slot_fraction <= 1.0:
        raise ValueError("clean_slot_fraction must lie in [0, 1]")
    if not 0.0 <= min_added_valid_rate <= max_added_valid_rate <= 1.0:
        raise ValueError("added-valid deletion rate bounds must lie in [0, 1]")
    if matching_tolerance < 0.0 or max_candidates < 1 or max_candidates > MAX_CANDIDATES:
        raise ValueError(f"max_candidates must lie in [1, {MAX_CANDIDATES}] and matching_tolerance be non-negative")

    shape = raw.shape
    no_mask = _zero_plan(shape)
    target_valid = (raw > 0) & support
    valid_count = int(np.count_nonzero(target_valid))
    if valid_count == 0:
        return PairedPlan(
            target_sample, target_group, None, None, no_mask, no_mask.copy(), no_mask.copy(),
            0, 0, 0, None, None, None, 0, (), False, "no_valid_target_depth",
        )

    clean_rng = random.Random(
        _stable_seed(seed, "clean-slot", epoch, successful_update, target_sample, target_group, batch_slot)
    )
    if clean_rng.random() < clean_slot_fraction:
        return PairedPlan(
            target_sample, target_group, None, None, no_mask, no_mask.copy(), no_mask.copy(),
            valid_count, 0, 0, None, None, None, 0, (), True, "clean_slot",
        )

    plan_rng = random.Random(
        _stable_seed(seed, "paired-source", epoch, successful_update, target_sample, target_group, batch_slot)
    )
    entries = source_pool.candidates(target_group, plan_rng, max_candidates)
    if not entries:
        return PairedPlan(
            target_sample, target_group, None, None, no_mask, no_mask.copy(), no_mask.copy(),
            valid_count, 0, 0, None, None, None, 0, (), False, "no_other_source_group",
        )

    candidate_sources = []
    last_source_mask = no_mask
    last_replay_mask = no_mask.copy()
    last_grid_mask = no_mask.copy()
    last_replay_count = 0
    last_grid_count = 0
    last_replay_rate = None
    last_grid_rate = None
    last_error = None
    eligible_replay_seen = False

    for attempt, entry in enumerate(entries):
        candidate_sources.append((entry.sample_id, entry.group_id))
        source_invalid = source_pool.read_invalid_mask(entry)
        replay_source_mask = _transform_source_mask(source_invalid, transform, shape)
        replay_delete_mask = replay_source_mask & target_valid
        replay_count = int(np.count_nonzero(replay_delete_mask))
        replay_rate = replay_count / float(valid_count)
        last_source_mask = replay_source_mask
        last_replay_mask = replay_delete_mask
        last_replay_count = replay_count
        last_replay_rate = replay_rate

        if replay_count == 0 or replay_rate < min_added_valid_rate or replay_rate > max_added_valid_rate:
            continue

        eligible_replay_seen = True
        grid_rng = np.random.default_rng(
            _stable_seed(
                seed, "paired-grid", epoch, successful_update, target_sample, target_group,
                batch_slot, entry.sample_id, attempt,
            )
        )
        grid_delete_mask = _grid_candidate(target_valid, replay_count, attempt, grid_rng) & target_valid
        grid_count = int(np.count_nonzero(grid_delete_mask))
        grid_rate = grid_count / float(valid_count)
        error = abs(grid_rate - replay_rate)
        last_grid_mask = grid_delete_mask
        last_grid_count = grid_count
        last_grid_rate = grid_rate
        last_error = error
        if error <= matching_tolerance + 1e-12:
            return PairedPlan(
                target_sample, target_group, entry.sample_id, entry.group_id,
                replay_source_mask, replay_delete_mask, grid_delete_mask,
                valid_count, replay_count, grid_count, replay_rate, grid_rate, error,
                attempt + 1, tuple(candidate_sources), False, "matched",
            )

    reason = "grid_match_not_found" if eligible_replay_seen else "no_replay_candidate_in_valid_range"
    return PairedPlan(
        target_sample, target_group, None, None, last_source_mask, last_replay_mask, last_grid_mask,
        valid_count, last_replay_count, last_grid_count, last_replay_rate, last_grid_rate, last_error,
        len(candidate_sources), tuple(candidate_sources), False, reason,
    )


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_cpu_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "device") and getattr(value.device, "type", "cpu") != "cpu":
        raise ValueError(f"{name} must remain on CPU until NaturalMissing input processing")
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "numpy"):
            value = value.numpy()
    return np.asarray(value)


def _clone_cpu(value: Any, name: str) -> Any:
    if hasattr(value, "device") and getattr(value.device, "type", "cpu") != "cpu":
        raise ValueError(f"{name} must remain on CPU until NaturalMissing input processing")
    if hasattr(value, "clone"):
        return value.clone()
    return np.array(value, copy=True)


def _batch_plane(value: Any, name: str, batch_size: Optional[int] = None) -> Tuple[np.ndarray, bool]:
    array = _as_cpu_numpy(value, name)
    had_batch = array.ndim >= 3
    if array.ndim == 2:
        array = array[None, ...]
        had_batch = False
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"{name} must be HW or BxHxW (or Bx1xHxW), got {array.shape}")
    if batch_size is not None and array.shape[0] != batch_size:
        raise ValueError(f"{name} batch size {array.shape[0]} differs from {batch_size}")
    return array, had_batch


def _batch_item(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return value[index]
        return value
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index]
    if hasattr(value, "device") or hasattr(value, "detach"):
        array = _as_cpu_numpy(value, "metadata field")
        if array.ndim > 0 and array.shape[0] == batch_size:
            return array[index]
        if array.ndim == 0:
            return array.item()
    return value


def _metadata_for_slot(metadata: Any, index: int, batch_size: int) -> Dict[str, Any]:
    if isinstance(metadata, (list, tuple)) and len(metadata) == batch_size and isinstance(metadata[index], Mapping):
        return dict(metadata[index])
    if not isinstance(metadata, Mapping):
        raise ValueError("NaturalMissing metadata must be a mapping or a batch of mappings")
    return {key: _batch_item(value, index, batch_size) for key, value in metadata.items()}


def _scalar(value: Any, name: str) -> Any:
    if hasattr(value, "detach") or isinstance(value, np.ndarray):
        value = _as_cpu_numpy(value, name)
        if value.size != 1:
            raise ValueError(f"metadata field {name!r} must be scalar per item, got {value.shape}")
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _sample_identity(batch: Mapping[str, Any], meta: Mapping[str, Any], slot: int, batch_size: int) -> Tuple[str, str]:
    sample_value = meta.get("sample_id")
    if sample_value is None:
        fn = _batch_item(batch.get("fn", ""), slot, batch_size)
        sample, group = parse_museg_identity(str(fn))
        return sample, group
    sample = str(sample_value)
    group_value = meta.get("group_id")
    if group_value is None:
        return parse_museg_identity(sample)
    return sample, str(group_value)


def _transform_from_metadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "raw_height", "raw_width", "mirror", "scaled_height", "scaled_width",
        "crop_y", "crop_x", "crop_height", "crop_width",
        "pad_top", "pad_bottom", "pad_left", "pad_right",
    )
    missing = [name for name in required if name not in meta]
    if missing:
        raise ValueError("incomplete NaturalMissing transform metadata: " + ", ".join(missing))
    result = {name: _scalar(meta[name], name) for name in required}
    result["mirror"] = bool(result["mirror"])
    for name in required:
        if name != "mirror":
            result[name] = int(result[name])
    return result


def _depth_zero_normalized(config: Any, channels: int) -> np.ndarray:
    if bool(_get_value(config, "x_is_single_channel", False)):
        mean = np.full(3, 0.48, dtype=np.float32)
        std = np.full(3, 0.28, dtype=np.float32)
    else:
        mean = np.asarray(_get_value(config, "norm_mean"), dtype=np.float32).reshape(-1)
        std = np.asarray(_get_value(config, "norm_std"), dtype=np.float32).reshape(-1)
    if mean.size == 1:
        mean = np.repeat(mean, channels)
    if std.size == 1:
        std = np.repeat(std, channels)
    if mean.size != channels or std.size != channels or np.any(std <= 0):
        raise ValueError("normalization mean/std do not match modal_x channels")
    return -mean / std


def _resolve_depth16_root(config: Any, natural_config: Mapping[str, Any]) -> Optional[str]:
    explicit = natural_config.get("depth16_root")
    if explicit:
        return os.path.abspath(os.fspath(explicit))
    for field_name in ("x_root_folder", "rgb_root_folder", "dataset_root", "data_root"):
        value = _get_value(config, field_name)
        if not value:
            continue
        root = Path(os.fspath(value))
        if root.name.lower() in {"depth", "rgb"}:
            return str(root.parent / "Depth16")
        if root.name.lower() == "depth16":
            return str(root)
        return str(root / "Depth16")
    return None


def _slot_telemetry(plan: PairedPlan, strategy: str, applied_count: int, applied_rate: Optional[float]) -> Dict[str, Any]:
    status = "matched" if plan.matched else ("clean" if plan.clean_slot else "paired_skip")
    return {
        "batch_slot": None,
        "target_sample": plan.target_sample,
        "target_group": plan.target_group,
        "source_sample": plan.source_sample,
        "source_group": plan.source_group,
        "candidate_sources": [
            {"sample": sample, "group": group} for sample, group in plan.candidate_sources
        ],
        "strategy": strategy,
        "status": status,
        "reason": plan.reason,
        "attempts": plan.attempts,
        "clean_slot": plan.clean_slot,
        "target_valid_count": plan.target_valid_count,
        "replay_deletion_count": plan.replay_deletion_count,
        "grid_deletion_count": plan.grid_deletion_count,
        "applied_deletion_count": applied_count,
        "replay_rate": plan.replay_rate,
        "grid_rate": plan.grid_rate,
        "applied_rate": applied_rate,
        "matching_error": plan.matching_error,
    }


def build_natural_missing_batch(
    batch: Mapping[str, Any],
    config: Any,
    *,
    successful_update: int,
    epoch: int = 0,
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Apply one deterministic Natural/Grid/Replay CPU input plan to a batch.

    Required opt-in loader fields are ``metadata``, ``raw_depth`` (current uint8
    Depth after linear resize and geometry transforms), and boolean ``support``
    (true only for non-padding pixels).  The original batch is not mutated.
    """
    natural_config = _get_value(config, "natural_missing")
    if not isinstance(natural_config, Mapping):
        raise ValueError("config.natural_missing must be a mapping with a strategy")
    strategy = natural_config.get("strategy")
    if strategy not in STRATEGIES:
        raise ValueError(f"natural_missing.strategy must be one of {STRATEGIES}, got {strategy!r}")
    seed = int(_get_value(config, "seed", 0))
    result = dict(batch)

    if strategy == "Natural":
        telemetry = {
            "strategy": strategy,
            "successful_update": int(successful_update),
            "epoch": int(epoch),
            "slots": [],
        }
        fn_value = batch.get("fn", "")
        if "raw_depth" in batch:
            raw_batch, _ = _batch_plane(batch["raw_depth"], "raw_depth")
            batch_size = raw_batch.shape[0]
        else:
            fn_array = fn_value if isinstance(fn_value, (list, tuple)) else [fn_value]
            batch_size = len(fn_array)
        for slot in range(batch_size):
            fn = _batch_item(fn_value, slot, batch_size)
            try:
                sample, group = parse_museg_identity(str(fn))
            except ValueError:
                sample, group = str(fn), ""
            telemetry["slots"].append({
                "batch_slot": slot,
                "target_sample": sample,
                "target_group": group,
                "source_sample": None,
                "source_group": None,
                "status": "clean_control",
                "reason": "natural_input_unchanged",
                "attempts": 0,
                "applied_deletion_count": 0,
                "applied_rate": None,
            })
        return result, telemetry

    for required in ("metadata", "raw_depth", "support", "modal_x"):
        if required not in batch:
            raise ValueError(f"NaturalMissing opt-in loader batch is missing {required!r}")
    raw_batch, _ = _batch_plane(batch["raw_depth"], "raw_depth")
    if raw_batch.dtype != np.uint8:
        raise ValueError(f"raw_depth must be uint8, got {raw_batch.dtype}")
    batch_size = raw_batch.shape[0]
    support_batch, _ = _batch_plane(batch["support"], "support", batch_size)
    support_batch = support_batch.astype(np.bool_, copy=False)
    if raw_batch.shape != support_batch.shape:
        raise ValueError(f"raw_depth/support batch shapes differ: {raw_batch.shape}/{support_batch.shape}")

    modal_copy = _clone_cpu(batch["modal_x"], "modal_x")
    modal_batch = _as_cpu_numpy(modal_copy, "modal_x")
    if modal_batch.ndim == 3:
        modal_batch = modal_batch[None, ...]
    if modal_batch.ndim != 4 or modal_batch.shape[0] != batch_size:
        raise ValueError(f"modal_x must be BxCxHxW, got {modal_batch.shape}")
    if modal_batch.shape[-2:] != raw_batch.shape[-2:]:
        raise ValueError("modal_x and raw_depth spatial shapes differ")

    raw_copy = _clone_cpu(batch["raw_depth"], "raw_depth")
    raw_copy_batch, _ = _batch_plane(raw_copy, "raw_depth", batch_size)
    zero_norm = _depth_zero_normalized(config, modal_batch.shape[1])

    train_source = _get_value(config, "train_source")
    depth16_root = _resolve_depth16_root(config, natural_config)
    if not depth16_root:
        raise ValueError("cannot resolve Depth16 root; set config.natural_missing['depth16_root']")
    source_pool = natural_config.get("source_pool")
    if source_pool is None:
        source_pool = build_source_pool(os.fspath(train_source), depth16_root)
    if not isinstance(source_pool, NaturalMissingSourcePool):
        raise TypeError("natural_missing.source_pool must be a NaturalMissingSourcePool")

    metadata = batch["metadata"]
    telemetry = {
        "strategy": strategy,
        "successful_update": int(successful_update),
        "epoch": int(epoch),
        "slots": [],
    }
    for slot in range(batch_size):
        meta = _metadata_for_slot(metadata, slot, batch_size)
        sample, group = _sample_identity(batch, meta, slot, batch_size)
        transform = _transform_from_metadata(meta)
        raw_plane = raw_batch[slot]
        support_plane = support_batch[slot]
        if raw_plane.shape != modal_batch.shape[-2:]:
            raise ValueError("sample raw_depth/support size does not match modal_x")
        plan = build_paired_plan(
            raw_plane,
            support_plane,
            transform,
            source_pool,
            target_sample=sample,
            target_group=group,
            seed=seed,
            successful_update=int(successful_update),
            epoch=int(epoch),
            batch_slot=slot,
        )

        applied_count = 0
        applied_rate = None
        if plan.matched:
            delete_mask = plan.grid_delete_mask if strategy == "Grid" else plan.replay_delete_mask
            applied_count = int(np.count_nonzero(delete_mask))
            applied_rate = applied_count / float(plan.target_valid_count)
            if applied_count:
                raw_copy_batch[slot][delete_mask] = np.uint8(0)
                modal_batch[slot][:, delete_mask] = zero_norm[:, None]
        slot_info = _slot_telemetry(plan, strategy, applied_count, applied_rate)
        slot_info["batch_slot"] = slot
        telemetry["slots"].append(slot_info)

    result["modal_x"] = modal_copy
    result["raw_depth"] = raw_copy
    current_validity = (raw_copy_batch > 0) & support_batch
    result["current_validity"] = torch.from_numpy(np.ascontiguousarray(current_validity, dtype=np.bool_))
    return result, telemetry
