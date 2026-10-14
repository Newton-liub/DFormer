"""OC-RGA data protocol: the three support markers and the shared degradation.

Design section 9.  This module owns the only place where the three markers are
created and where the simulated depth degradation is drawn, so both arms of the
comparison see the *same* RGB augmentation, the *same* ``S``/``O`` and the *same*
fault realisation.

Markers
-------
``S``    image geometry support -- the un-padded finite image domain.  Artificial
         deletions never change ``S``.
``O``    currently available observation -- ``O <= S``; known deletions (and the
         whole-modality dropout) set it to 0, non-zero corruption does not.
``Vref`` training-only encoded reference filter: the original value is finite,
         un-padded and strictly inside ``(0, 255)`` (the two endpoints are
         excluded conservatively).  It never reaches the model.

Inference inputs are exactly ``RGB``, the degraded Depth, ``S`` and ``O``.  The
uncorrupted reference depth, ``Vref``, the fault type/position/strength and the
semantic label are returned in the batch for the loss/metric functions only.

Randomness
----------
Every stochastic step in ``__getitem__`` draws from a *stable local stream*
seeded by ``(seed, epoch, stable_sample_id, repeat_index, stream_name)``.  The
global torch/python RNG is deliberately untouched, so the main-branch dropout /
drop-path and the HAM random basis keep consuming the training RNG exactly as
before and the auxiliary hold-out sampling cannot perturb them.

The epoch permutation is produced once per epoch by
:class:`EpochPermutationSampler`, which yields ``(line_index, repeat_index,
epoch)`` tuples so the epoch travels with the index and a worker can never hold a
stale epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.deliver import (  # noqa: E402
    DeLiVERDataset,
    _read_depth,
    _read_label_ids,
    _read_rgb,
    deliver_relative_paths,
    map_deliver_label,
)
from research.oc_rga import Z_MAX, Z_MIN, ZERO_FILL  # noqa: E402
from utils.transforms import normalize, random_crop_pad_to_shape  # noqa: E402

__all__ = [
    "VREF_RAW_MIN",
    "VREF_RAW_MAX",
    "stream_seed",
    "degrade_depth",
    "_rectangle_holes",
    "_trim_rectangle_boundary",
    "OCRGADataset",
    "OCRGAEvalDataset",
    "EpochPermutationSampler",
    "build_train_loader",
    "build_eval_loader",
    "dataset_setting",
    "scale_long_side",
    "pad_to_multiple",
]

# Conservative reference candidate: the original raw value must be strictly
# inside (0, 255).  The endpoints are excluded, not declared invalid.
VREF_RAW_MIN = 1
VREF_RAW_MAX = 254

DEPTH_NORM_MEAN = 0.48
DEPTH_NORM_STD = 0.28


# --------------------------------------------------------------------------- #
# Stable local random streams
# --------------------------------------------------------------------------- #
def stream_seed(seed: int, epoch: int, sample_id: int, repeat: int, name: str) -> int:
    """Deterministic 63-bit seed for one (seed, epoch, sample, repeat, stream)."""
    key = "%d|%d|%d|%d|%s" % (int(seed), int(epoch), int(sample_id), int(repeat), str(name))
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _rng(seed: int, epoch: int, sample_id: int, repeat: int, name: str) -> np.random.Generator:
    return np.random.default_rng(stream_seed(seed, epoch, sample_id, repeat, name))


# --------------------------------------------------------------------------- #
# Simulated degradation (design 9.2)
# --------------------------------------------------------------------------- #
def _pick(range_low: float, range_high: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(float(range_low), float(range_high)))


def _delete_exactly(candidate: np.ndarray, support: np.ndarray, target: int,
                    rng: np.random.Generator, extend: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean deletion mask over ``support`` with exactly ``target`` pixels.

    ``candidate`` (a union of local windows, or all-zeros for independent pixel
    deletion) is preferred; a shortfall is grown through ``extend`` (the dilated
    window neighbourhood) and finally through arbitrary support pixels.  The final
    boundary adjustment is what makes the requested ratio an exact integer count.
    """
    support_flat = support.reshape(-1) > 0
    support_idx = np.flatnonzero(support_flat)
    out = np.zeros(support.size, dtype=bool)
    if target <= 0 or support_idx.size == 0:
        return out.reshape(support.shape)
    target = int(min(int(target), int(support_idx.size)))

    chosen = np.flatnonzero(candidate.reshape(-1) & support_flat)
    if chosen.size > target:
        chosen = rng.choice(chosen, size=target, replace=False)
        out[chosen] = True
        return out.reshape(support.shape)
    if chosen.size == target:
        out[chosen] = True
        return out.reshape(support.shape)

    need = target - chosen.size
    pool = np.array([], dtype=np.int64)
    if extend is not None:
        pool = np.flatnonzero(extend.reshape(-1) & support_flat)
        pool = np.setdiff1d(pool, chosen, assume_unique=False)
    if pool.size < need:
        rest = np.setdiff1d(support_idx, chosen, assume_unique=False)
        pool = np.concatenate([pool, np.setdiff1d(rest, pool, assume_unique=False)])
    take = rng.choice(pool, size=min(need, pool.size), replace=False)
    chosen = np.concatenate([chosen, take])
    if chosen.size < target:
        rest = np.setdiff1d(support_idx, chosen, assume_unique=False)
        chosen = np.concatenate([chosen, rng.choice(rest, size=target - chosen.size, replace=False)])
    out[chosen] = True
    return out.reshape(support.shape)


def _trim_rectangle_boundary(mask: np.ndarray, target: int) -> np.ndarray:
    """Peel a rectangle union down to exactly ``target`` pixels from its boundary."""
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    current = mask.copy()
    while int(current.sum()) > target:
        excess = int(current.sum()) - target
        eroded = cv2.erode(current.astype(np.uint8), kernel) > 0
        boundary = current & (~eroded)
        count = int(boundary.sum())
        if count == 0 or count > excess:
            flat = current.reshape(-1)
            candidates = np.flatnonzero(boundary.reshape(-1))
            if candidates.size == 0:
                candidates = np.flatnonzero(flat)
            flat[candidates[:excess]] = False
            current = flat.reshape(current.shape)
        else:
            current = eroded
    return current


def _rectangle_holes(support: np.ndarray, target: int, short_side: float,
                     low: float, high: float, rng: np.random.Generator,
                     max_windows: int = 64, stats: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Rectangle-dominated local depletion with exactly ``target`` support pixels.

    Rectangles of side ``0.1-0.4`` of the crop short side are added, and existing
    rectangles are locally expanded, until the requested count is reached.  Each
    new rectangle is capped to the remaining budget, so the union lands close to
    the target and the final adjustment only peels a small part of its boundary.
    There is deliberately no fallback that scatters independent pixels over the
    whole support domain: that family is the separate ``pixel_missing`` condition
    (design 9.2).
    """
    height, width = support.shape
    support_mask = support > 0
    deleted = np.zeros((height, width), dtype=bool)
    total_support = int(support_mask.sum())
    target = int(min(int(target), total_support))
    windows = 0
    grown = 0
    current = 0
    while current < target and windows < max_windows:
        remaining = target - current
        if deleted.any() and float(rng.random()) < 0.5:
            ys, xs = np.nonzero(deleted)
            top, bottom = int(ys.min()), int(ys.max()) + 1
            left, right = int(xs.min()), int(xs.max()) + 1
            grow = max(1, int(round(0.15 * max(bottom - top, right - left))))
            ntop, nbottom = max(0, top - grow), min(height, bottom + grow)
            nleft, nright = max(0, left - grow), min(width, right + grow)
            rectangle = np.zeros_like(deleted)
            rectangle[ntop:nbottom, nleft:nright] = True
            added = int((rectangle & support_mask & (~deleted)).sum())
            if 0 < added <= remaining + max(1, grow * 4):
                deleted |= rectangle
                grown += 1
                current = int((deleted & support_mask).sum())
                continue
        windows += 1
        side = max(2.0, float(rng.uniform(low, high)) * float(short_side))
        h_side = min(height, max(1, int(round(side * float(rng.uniform(0.6, 1.4))))))
        w_side = min(width, max(1, int(round(side * float(rng.uniform(0.6, 1.4))))))
        # Cap the rectangle so it does not overshoot the remaining budget by much.
        max_w = max(1, int(min(width, np.ceil(remaining / float(h_side)))))
        w_side = min(w_side, max_w)
        top = int(rng.integers(0, height - h_side + 1))
        left = int(rng.integers(0, width - w_side + 1))
        deleted[top: top + h_side, left: left + w_side] = True
        current = int((deleted & support_mask).sum())

    # Coverage guarantee: keep expanding the local rectangle hull (still one
    # rectangle) until the target is at least reachable, never by random scatter.
    guard = 0
    while current < target and guard <= max(height, width):
        guard += 1
        ys, xs = np.nonzero(deleted)
        if ys.size == 0:
            deleted[:] = True
            break
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        deleted[max(0, top - 1): min(height, bottom + 1),
                max(0, left - 1): min(width, right + 1)] = True
        current = int((deleted & support_mask).sum())

    deleted &= support_mask
    union_pixels = int(deleted.sum())
    target = min(target, union_pixels)
    trimmed = 0
    if union_pixels > target:
        deleted = _trim_rectangle_boundary(deleted, target)
        trimmed = union_pixels - int(deleted.sum())
    if stats is not None:
        stats["windows"] = float(windows)
        stats["grown"] = float(grown)
        stats["union_pixels"] = float(union_pixels)
        stats["trimmed_pixels"] = float(trimmed)
        stats["trim_fraction"] = float(trimmed) / float(max(union_pixels, 1))
    return deleted


def _local_rectangle(height: int, width: int, area_fraction: float,
                     rng: np.random.Generator) -> np.ndarray:
    target = max(1, int(round(area_fraction * height * width)))
    ratio = float(rng.uniform(0.5, 2.0))
    h_side = int(max(1, min(height, round(math.sqrt(target / ratio)))))
    w_side = int(max(1, min(width, round(target / max(h_side, 1)))))
    while h_side * w_side < target and w_side < width:
        w_side += 1
    top = int(rng.integers(0, height - h_side + 1))
    left = int(rng.integers(0, width - w_side + 1))
    mask = np.zeros((height, width), dtype=bool)
    mask[top: top + h_side, left: left + w_side] = True
    return mask


def degrade_depth(reference: np.ndarray, support: np.ndarray, config,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Apply exactly one degradation family to one training crop.

    ``reference`` is the un-corrupted encoded crop ``[H,W]`` FP32 and ``support``
    the image geometry support ``S``.  Returns ``(observed, O, info)`` where
    ``observed`` already carries the canonical raw-zero fill outside ``O``.
    """
    reference = np.asarray(reference, dtype=np.float32)
    support = (np.asarray(support) > 0).astype(np.float32)
    height, width = reference.shape
    observed = reference.copy()
    observation = support.copy()
    info: Dict[str, object] = {"mode": "clean", "requested": 0.0, "realised": 0.0,
                               "support_pixels": int(support.sum())}

    weights = np.array([
        float(getattr(config, "degrade_clean", 0.40)),
        float(getattr(config, "degrade_local_holes", 0.25)),
        float(getattr(config, "degrade_pixel_missing", 0.15)),
        float(getattr(config, "degrade_nonzero", 0.15)),
        float(getattr(config, "degrade_full_missing", 0.05)),
    ], dtype=np.float64)
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError("the degradation distribution must sum to 1, got %.6f" % weights.sum())
    family = int(rng.choice(5, p=weights / weights.sum()))

    if family in (1, 2):
        low = float(getattr(config, "degrade_hole_ratio_low", 0.25))
        high = float(getattr(config, "degrade_hole_ratio_high", 0.75))
        ratio = _pick(low, high, rng)
        target = int(round(ratio * float(support.sum())))
        if family == 1:
            # Rectangle-dominated local depletion (design 9.2), never independent
            # pixel scattering: that is the separate ``pixel_missing`` family.
            deleted = _rectangle_holes(
                support,
                target,
                float(min(height, width)),
                float(getattr(config, "degrade_local_window_low", 0.10)),
                float(getattr(config, "degrade_local_window_high", 0.40)),
                rng,
            )
            info["mode"] = "local_holes"
        else:
            deleted = _delete_exactly(np.zeros_like(support, dtype=bool), support, target, rng)
            info["mode"] = "pixel_missing"
        info["requested"] = float(ratio)
        info["realised"] = float(deleted.sum()) / float(max(support.sum(), 1.0))
        observed = np.where(deleted, np.float32(ZERO_FILL), observed)
        observation = np.where(deleted, np.float32(0.0), observation)

    elif family == 3:
        info["mode"] = "nonzero"
        if rng.random() < 0.5:
            std = _pick(getattr(config, "degrade_noise_std_low", 0.10),
                        getattr(config, "degrade_noise_std_high", 0.40), rng)
            noise = rng.normal(0.0, std, size=reference.shape).astype(np.float32)
            observed = observed + noise * support
            info["detail"] = "gaussian std=%.3f" % std
        else:
            area = _pick(getattr(config, "degrade_shift_area_low", 0.10),
                         getattr(config, "degrade_shift_area_high", 0.35), rng)
            region = _local_rectangle(height, width, area, rng) & (support > 0)
            magnitude = _pick(getattr(config, "degrade_shift_offset_low", 0.20),
                              getattr(config, "degrade_shift_offset_high", 0.80), rng)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            observed = observed + np.float32(sign * magnitude) * region.astype(np.float32)
            outlier_rate = float(getattr(config, "degrade_shift_outlier_rate", 0.10))
            region_idx = np.flatnonzero(region.reshape(-1))
            if region_idx.size:
                count = max(1, int(round(outlier_rate * region_idx.size)))
                picked = rng.choice(region_idx, size=count, replace=False)
                random_values = rng.uniform(Z_MIN, Z_MAX, size=picked.size).astype(np.float32)
                flat = observed.reshape(-1)
                flat[picked] = random_values
                observed = flat.reshape(reference.shape)
            info["detail"] = "local value offset %+0.3f over %.3f of the image" % (sign * magnitude, area)
        observed = np.clip(observed, np.float32(Z_MIN), np.float32(Z_MAX))

    elif family == 4:
        info["mode"] = "entire_missing"
        info["requested"] = 1.0
        info["realised"] = 1.0
        observation = np.zeros_like(support)
        observed = np.full_like(reference, np.float32(ZERO_FILL))

    observed = np.where(observation > 0, observed, np.float32(ZERO_FILL)).astype(np.float32)
    info["observed_ratio"] = float(observation.sum()) / float(max(support.sum(), 1.0))
    return observed, observation, info


# --------------------------------------------------------------------------- #
# Geometric helpers for the shared 640 evaluation protocol
# --------------------------------------------------------------------------- #
def scale_long_side(height: int, width: int, long_side: int) -> Tuple[int, int, float]:
    """Scale factor so the long side becomes ``long_side`` (aspect preserved)."""
    longest = max(int(height), int(width))
    if longest <= 0:
        raise ValueError("empty image")
    factor = float(long_side) / float(longest)
    return max(1, int(round(height * factor))), max(1, int(round(width * factor))), factor


def pad_to_multiple(array: np.ndarray, multiple: int, value) -> np.ndarray:
    """Right/bottom pad a HxW(xC) array to a multiple of ``multiple``."""
    height, width = array.shape[:2]
    pad_h = (-height) % int(multiple)
    pad_w = (-width) % int(multiple)
    if pad_h == 0 and pad_w == 0:
        return array
    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, pad_width, mode="constant", constant_values=value)


# --------------------------------------------------------------------------- #
# Training dataset
# --------------------------------------------------------------------------- #
class OCRGADataset(DeLiVERDataset):
    """DeLiVER training crops with the three support markers.

    ``__getitem__`` accepts either a plain integer index (epoch 0, repeat 0) or
    the ``(line_index, repeat_index, epoch)`` tuple produced by
    :class:`EpochPermutationSampler`.
    """

    def __init__(self, setting, split_name, config, epoch: int = 0, file_length: Optional[int] = None):
        super().__init__(setting, split_name, None, file_length)
        self.config = config
        self.epoch = int(epoch)
        self.seed = int(getattr(config, "seed", 12345))
        self.norm_mean = list(config.norm_mean)
        self.norm_std = list(config.norm_std)
        self.crop = (int(config.image_height), int(config.image_width))
        scales = getattr(config, "train_scale_array", None)
        self.scales = tuple(float(s) for s in scales) if scales else None

    # -- helpers --------------------------------------------------------- #
    def _spatial_transform(self, rgb, depth, label, support, vref, rng):
        if float(rng.random()) < 0.5:
            rgb = cv2.flip(rgb, 1)
            depth = cv2.flip(depth, 1)
            label = cv2.flip(label, 1)
            support = cv2.flip(support, 1)
            vref = cv2.flip(vref, 1)
        if self.scales:
            scale = float(self.scales[int(rng.integers(0, len(self.scales)))])
            sh = int(rgb.shape[0] * scale)
            sw = int(rgb.shape[1] * scale)
            if sh > 0 and sw > 0:
                rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (sw, sh), interpolation=cv2.INTER_NEAREST)
                support = cv2.resize(support, (sw, sh), interpolation=cv2.INTER_NEAREST)
                # Same bilinear weights as the depth resize: only a reference that
                # never mixed in an unknown endpoint stays legal.
                vref = cv2.resize(vref, (sw, sh), interpolation=cv2.INTER_LINEAR)
        return rgb, depth, label, support, vref

    def _crop_position(self, height, width, rng):
        crop_h, crop_w = self.crop
        pos_h = int(rng.integers(0, height - crop_h + 1)) if height > crop_h else 0
        pos_w = int(rng.integers(0, width - crop_w + 1)) if width > crop_w else 0
        return pos_h, pos_w

    # -- item ------------------------------------------------------------ #
    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            index, repeat, epoch = int(key[0]), int(key[1]), int(key[2])
        else:
            index, repeat, epoch = int(key), 0, int(self.epoch)

        item_name = self._file_names[index]
        rgb_rel, depth_rel, label_rel = deliver_relative_paths(item_name)
        rgb = _read_rgb(self._rgb_root / str(rgb_rel))
        depth_raw = _read_depth(self._depth_root / str(depth_rel))
        label = map_deliver_label(_read_label_ids(self._gt_root / str(label_rel)))
        if depth_raw.shape != label.shape:
            raise ValueError("depth %s and label %s shapes differ" % (depth_raw.shape, label.shape))

        support = np.isfinite(depth_raw).astype(np.float32)
        vref = ((depth_raw >= VREF_RAW_MIN) & (depth_raw <= VREF_RAW_MAX)).astype(np.float32)
        depth = depth_raw.astype(np.float32)

        rgb, depth, label, support, vref = self._spatial_transform(
            rgb, depth, label, support, vref, _rng(self.seed, epoch, index, repeat, "spatial")
        )

        height, width = rgb.shape[:2]
        pos = self._crop_position(height, width, _rng(self.seed, epoch, index, repeat, "crop"))
        crop_h, crop_w = self.crop

        rgb_crop, _ = random_crop_pad_to_shape(normalize(rgb, self.norm_mean, self.norm_std), pos, self.crop, 0)
        depth_crop, _ = random_crop_pad_to_shape(depth, pos, self.crop, 0.0)
        label_crop, _ = random_crop_pad_to_shape(label, pos, self.crop, 255)
        support_crop, _ = random_crop_pad_to_shape(support, pos, self.crop, 0.0)
        vref_crop, _ = random_crop_pad_to_shape(vref, pos, self.crop, 0.0)

        support_crop = support_crop * np.isfinite(depth_crop).astype(np.float32)
        reference = ((depth_crop.astype(np.float64) / 255.0) - DEPTH_NORM_MEAN) / DEPTH_NORM_STD
        reference = reference.astype(np.float32)
        vref_crop = ((vref_crop >= 1.0 - 1e-6) & (support_crop > 0)).astype(np.float32)

        observed, observation, info = degrade_depth(
            reference, support_crop, self.config,
            _rng(self.seed, epoch, index, repeat, "degrade"),
        )

        return dict(
            data=torch.from_numpy(np.ascontiguousarray(rgb_crop.transpose(2, 0, 1))).float(),
            modal_x=torch.from_numpy(np.ascontiguousarray(np.repeat(observed[None], 3, axis=0))).float(),
            depth_support=torch.from_numpy(np.ascontiguousarray(observation[None])).float(),
            image_support=torch.from_numpy(np.ascontiguousarray(support_crop[None])).float(),
            reference_depth=torch.from_numpy(np.ascontiguousarray(reference[None])).float(),
            reference_valid=torch.from_numpy(np.ascontiguousarray(vref_crop[None])).float(),
            label=torch.from_numpy(np.ascontiguousarray(label_crop)).long(),
            fn=str(self._rgb_root / str(rgb_rel)),
            sample_id=int(index),
            epoch=int(epoch),
            repeat_index=int(repeat),
            degradation=str(info["mode"]),
        )


# --------------------------------------------------------------------------- #
# Evaluation dataset (shared 640 protocol, no extra degradation)
# --------------------------------------------------------------------------- #
class OCRGAEvalDataset(DeLiVERDataset):
    """Evaluation items on the 640 long-side padded grid with native-size labels.

    ``__getitem__`` returns the un-degraded observation set (``O = S``) plus the
    native label and sizes.  The fault is generated by the evaluation entry *after*
    this resize/pad step, so a fault can never be smeared by a later bilinear
    interpolation of an already hidden value.
    """

    def __init__(self, setting, split_name, config, file_length: Optional[int] = None):
        super().__init__(setting, split_name, None, file_length)
        self.config = config
        self.norm_mean = list(config.norm_mean)
        self.norm_std = list(config.norm_std)
        self.long_side = int(getattr(config, "eval_long_side", 640))

    def __getitem__(self, index):
        item_name = self._file_names[int(index)]
        rgb_rel, depth_rel, label_rel = deliver_relative_paths(item_name)
        rgb = _read_rgb(self._rgb_root / str(rgb_rel))
        depth_raw = _read_depth(self._depth_root / str(depth_rel))
        label = map_deliver_label(_read_label_ids(self._gt_root / str(label_rel)))
        native_h, native_w = label.shape

        height, width, _factor = scale_long_side(native_h, native_w, self.long_side)
        rgb_s = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        depth_s = cv2.resize(depth_raw.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

        # The real image support ``S`` is established *before* any padding, so the
        # added border can never be mistaken for a valid observation just because
        # its fill value is finite (audit R7).  RGB is normalised first and only
        # then padded with the standardised zero.
        support = np.ones((height, width), dtype=np.float32)
        rgb_n = normalize(rgb_s, self.norm_mean, self.norm_std)
        rgb_n = pad_to_multiple(rgb_n, 32, 0.0)
        depth_s = pad_to_multiple(depth_s, 32, 0.0)
        support = pad_to_multiple(support, 32, 0.0)
        encoded = ((depth_s.astype(np.float64) / 255.0) - DEPTH_NORM_MEAN) / DEPTH_NORM_STD

        return dict(
            data=torch.from_numpy(np.ascontiguousarray(rgb_n.transpose(2, 0, 1))).float(),
            modal_x=torch.from_numpy(np.ascontiguousarray(np.repeat(encoded[None].astype(np.float32), 3, axis=0))).float(),
            depth_support=torch.from_numpy(np.ascontiguousarray(support[None])).float(),
            image_support=torch.from_numpy(np.ascontiguousarray(support[None])).float(),
            label=torch.from_numpy(np.ascontiguousarray(label)).long(),
            fn=str(self._rgb_root / str(rgb_rel)),
            # ``original_size`` is the native label grid of the confusion matrix,
            # ``scaled_size`` is the 640 long-side grid (what the pad is removed
            # back to), ``padded_size`` is the padded model input grid.
            original_size=torch.tensor([native_h, native_w], dtype=torch.long),
            scaled_size=torch.tensor([height, width], dtype=torch.long),
            padded_size=torch.tensor([int(encoded.shape[0]), int(encoded.shape[1])], dtype=torch.long),
            sample_id=int(index),
        )


# --------------------------------------------------------------------------- #
# Sampler and loaders
# --------------------------------------------------------------------------- #
def _permutation_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha1(("perm|%d|%d" % (int(seed), int(epoch))).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class EpochPermutationSampler(torch.utils.data.Sampler):
    """One fixed permutation per epoch plus the trailing repeats.

    Yields ``(line_index, repeat_index, epoch)``.  The permutation is drawn once
    from an independent ``torch.Generator``, so it never consumes the training
    RNG and cannot be perturbed by the geometry branch consuming initialisation
    randomness.
    """

    def __init__(self, num_samples: int, effective_batch: int, seed: int, epoch: int = 0):
        self.num_samples = int(num_samples)
        self.effective_batch = int(effective_batch)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.updates_per_epoch = -(-self.num_samples // self.effective_batch)
        self.samples_per_epoch = self.updates_per_epoch * self.effective_batch
        self.seed = int(seed)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(_permutation_seed(self.seed, self.epoch))
        permutation = torch.randperm(self.num_samples, generator=generator).tolist()
        extra = self.samples_per_epoch - self.num_samples
        order: List[int] = list(permutation)
        repeats: List[int] = [0] * len(permutation)
        for i in range(extra):
            order.append(permutation[i % self.num_samples])
            repeats.append(1 + i // self.num_samples)
        for index, repeat in zip(order, repeats):
            yield (int(index), int(repeat), int(self.epoch))


def dataset_setting(config, train_source, eval_source) -> Dict[str, object]:
    return {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": getattr(config, "rgb_format", ".png"),
        "gt_root": config.gt_root_folder,
        "gt_format": getattr(config, "gt_format", ".png"),
        "transform_gt": False,
        "x_root": config.x_root_folder,
        "x_format": getattr(config, "x_format", ".png"),
        "x_single_channel": True,
        "class_names": config.class_names,
        "train_source": str(train_source),
        "eval_source": str(eval_source),
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }


def loader_generator(seed: int, name: str, epoch: int = 0) -> torch.Generator:
    """Independent base-seed stream for a DataLoader.

    Seeding loader workers from the global RNG would make the CPU HAM basis and
    the training DropPath depend on how many loader iterators were built, which
    changes on resume.  This generator comes from the stable stream key instead,
    so the workers never perturb the training RNG.
    """
    generator = torch.Generator()
    generator.manual_seed(stream_seed(int(seed), int(epoch), 0, 0, "loader-%s" % name))
    return generator


def build_train_loader(config, train_source, eval_source, micro_batch, accum_steps,
                       effective_batch, num_workers, seed, epoch=0):
    from research.odg_schedule import manifest_lines, plan_epoch

    setting = dataset_setting(config, train_source, eval_source)
    num_samples = manifest_lines(train_source, fallback=getattr(config, "num_train_imgs", None))
    plan = plan_epoch(num_samples, micro_batch, accum_steps, effective_batch)
    dataset = OCRGADataset(setting, "train", config, epoch=epoch, file_length=None)
    sampler = EpochPermutationSampler(num_samples, effective_batch, seed, epoch=epoch)
    if len(sampler) != plan["samples_per_epoch"]:
        raise RuntimeError("sampler length %d != plan samples_per_epoch %d"
                           % (len(sampler), plan["samples_per_epoch"]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(micro_batch),
        sampler=sampler,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
        generator=loader_generator(seed, "train", epoch),
    )
    return loader, plan, num_samples


def build_eval_loader(config, eval_source, train_source, num_workers=0):
    setting = dataset_setting(config, eval_source, train_source)
    dataset = OCRGAEvalDataset(setting, "val", config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
        generator=loader_generator(int(getattr(config, "seed", 12345)), "eval"),
    )
    return loader, len(dataset)


# --------------------------------------------------------------------------- #
# CLI: read a few real batches without touching the model
# --------------------------------------------------------------------------- #
def cmd_inspect(args) -> int:
    from copy import deepcopy
    from importlib import import_module

    config = deepcopy(getattr(import_module(args.config), "C"))
    config.seed = int(args.seed)
    setting = dataset_setting(config, config.train_dev_source, config.dev_eval_source)
    dataset = OCRGADataset(setting, "train", config, epoch=0)
    report = {"config": args.config, "dataset": str(config.dataset_name), "num_items": len(dataset),
              "samples": []}
    modes: Dict[str, int] = {}
    for index in range(int(args.limit)):
        item = dataset[(index, 0, 0)]
        modes[item["degradation"]] = modes.get(item["degradation"], 0) + 1
        support = item["image_support"]
        observation = item["depth_support"]
        reference = item["reference_depth"]
        vref = item["reference_valid"]
        row = {
            "sample_id": item["sample_id"],
            "degradation": item["degradation"],
            "shapes": {k: list(item[k].shape) for k in
                       ("data", "modal_x", "depth_support", "image_support",
                        "reference_depth", "reference_valid", "label")},
            "support_fraction": round(float(support.mean()), 6),
            "observation_fraction": round(float(observation.mean()), 6),
            "reference_valid_fraction": round(float(vref.mean()), 6),
            "o_le_s": bool(torch.all(observation <= support + 1e-6)),
            "vref_le_s": bool(torch.all(vref <= support + 1e-6)),
            "reference_finite": bool(torch.isfinite(reference).all()),
            "observed_in_encoding_range": bool(
                torch.all(item["modal_x"][0][observation[0] > 0] >= Z_MIN - 1e-5)
                and torch.all(item["modal_x"][0][observation[0] > 0] <= Z_MAX + 1e-5)
            ),
            "missing_fill_is_raw_zero": bool(
                torch.allclose(item["modal_x"][0][observation[0] <= 0],
                               torch.full_like(item["modal_x"][0][observation[0] <= 0], ZERO_FILL),
                               atol=1e-6)
            ) if float(observation.mean()) < 1.0 else True,
        }
        report["samples"].append(row)
        print("[inspect] id=%-5d %-15s S=%.4f O=%.4f Vref=%.4f" %
              (row["sample_id"], row["degradation"], row["support_fraction"],
               row["observation_fraction"], row["reference_valid_fraction"]))
    report["mode_histogram"] = modes
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("[inspect] wrote %s" % args.report)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OC-RGA data protocol (three support markers)")
    parser.add_argument("--inspect", action="store_true", help="read N real training crops and report the markers")
    parser.add_argument("--config", default="local_configs.research.OC_RGA_DeLiVER")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    if not args.inspect:
        parser.print_help()
        return 0
    return cmd_inspect(args)


if __name__ == "__main__":
    sys.exit(main())
