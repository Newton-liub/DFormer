"""Direction-A frozen diagnostic entry (plan section 13).

Phases::

    self-check    CPU tensors only; no checkpoint, no dataset
    pilot32       fixed 32-image screening: clean / holes75 / cyclic16
    noncyclic32   the same 32 images on the artifact-control offsets
    confirm496    the strict complement of the 32 screening images

Everything is frozen inference: ``model.eval()``, no optimiser, no weight or
buffer update, no checkpoint written.  Original (O), depth-off (OFF), the KL
projection A and the bounded clip/gate strong controls (C/G) share one ``p0``
definition, one signed ``Bd`` construction, one FP32 aggregation protocol and
one per-image paired random state, so their differences are not reporting
artifacts (plan sections 4.4, 13.3-13.6).

Region bookkeeping (plan sections 6.2, 6.5 and 13.5):

* ``C_delta`` is the set of pixels whose depth really changed, computed on a
  reconstructed 8-bit depth so FP32 tail noise cannot masquerade as an
  intervention;
* ``J`` is the seam / assignment boundary neighbourhood (a diagnostic label, it
  is never fed to the model);
* all methods on one image and one condition share the same ``C_delta`` and the
  same common denominators, and every region reports repair *and* harm plus the
  net correct-pixel change.

Importing this module never trains, never touches the official test split and
never launches anything in the cloud.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from importlib import import_module

import cv2
import numpy as np
import torch
import torch.nn as nn

from research import odg_schedule as sched
from research.influence_projection import (
    BOUND_TOL,
    DEFAULT_ITERATIONS,
    MODE_CLIP,
    MODE_DEPTH_OFF,
    MODE_GATE,
    MODE_KL,
    MODE_ORIGINAL,
    SUPPORTED_MODES,
    TINY_P0,
    controlled_attention,
    run_self_check as run_operator_self_check,
    validate_tau,
)

DEV_SIZE = 528
SCREEN_COUNT = 32
HOLES_RATIO = 0.75
HOLES_SEED = 12345
OFFSET_PIXELS = 16
OPERATION_MAX_SIDE = 192

PHASE_CHOICES = ("self-check", "pilot32", "noncyclic32", "confirm496")
CONDITION_CHOICES = ("clean", "holes75", "cyclic16", "noncyclic16")
TAU_VALUES = {"log2": math.log(2.0), "log4": math.log(4.0)}
PHASE_CONDITIONS = {
    "pilot32": ("clean", "holes75", "cyclic16"),
    "noncyclic32": ("clean", "holes75", "noncyclic16"),
    "confirm496": ("clean", "holes75", "noncyclic16"),
}
PHASE_DEFAULT_METHODS = {
    "pilot32": ["original", "depth_off", "kl", "clip", "gate"],
    "noncyclic32": ["original", "kl", "clip", "gate"],
    "confirm496": ["original", "depth_off", "kl", "clip", "gate"],
}
# Author depth normalisation: normalised = (raw8 / 255 - 0.48) / 0.28.
DEPTH_RAW8_SCALE = 0.28
DEPTH_RAW8_OFFSET = 0.48

COUNTER_KEYS = (
    "pixels",
    "o_clean_correct",
    "o_corrupt_correct",
    "method_correct",
    "new_err_original",
    "new_err_method",
    "repairable",
    "repair",
    "harm_den",
    "harm",
    "gained_from_clean_wrong",
    "net_correct_delta",
    "net_correct_vs_clean",
)
REGION_COLUMNS = ["phase", "condition", "method", "image_index", "region"] + list(COUNTER_KEYS)
TASK_COLUMNS = ["phase", "condition", "method", "image_index", "gt_valid_pixels", "correct_pixels",
                "net_correct_delta_vs_corrupt", "net_correct_delta_vs_clean"]

_BAND4_KERNEL = np.ones((9, 9), np.uint8)


# --------------------------------------------------------------------------- #
# Control objects installed on the built model (never part of state_dict)
# --------------------------------------------------------------------------- #
class _InfluenceControl:
    """Plain per-block/per-attention control (not an ``nn.Module``).

    It is stored in the module ``__dict__``, holds no parameter and no buffer, so
    installing or removing it cannot change ``state_dict`` and cannot alter the
    legacy path: when it is ``None`` the author code runs untouched.
    """

    __slots__ = ("mode", "tau", "query_chunk", "observer", "stage", "block", "context", "iterations")

    def __init__(self, mode, tau, query_chunk, observer, stage, block, context, iterations):
        self.mode = mode
        self.tau = float(tau)
        self.query_chunk = int(query_chunk) if query_chunk else 0
        self.observer = observer
        self.stage = int(stage)
        self.block = int(block)
        self.context = context
        self.iterations = int(iterations)

    def attend(self, qk, bs, bd, axis):
        meta = {"stage": self.stage, "block": self.block, "axis": axis,
                "mode": self.mode, "tau": self.tau}
        meta.update(self.context)
        total = int(qk.shape[-2])
        chunk = self.query_chunk
        if chunk <= 0 or chunk >= total:
            return controlled_attention(qk, bs, bd, self.mode, self.tau, observer=self.observer,
                                        meta=meta, iterations=self.iterations)
        parts = []
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            sub_meta = dict(meta)
            sub_meta["query_slice"] = [start, stop]
            parts.append(controlled_attention(
                qk[..., start:stop, :],
                None if bs is None else bs[..., start:stop, :],
                None if bd is None else bd[..., start:stop, :],
                self.mode, self.tau, observer=self.observer, meta=sub_meta, iterations=self.iterations))
        return torch.cat(parts, dim=-2)


def configure_influence(model, mode="legacy", tau=0.0, query_chunk=32, observer=None,
                        *, context=None, iterations=DEFAULT_ITERATIONS):
    """Install (or clear) the direction-A control on all 29 DFormerv2 blocks.

    ``mode='legacy'`` restores the author path exactly by clearing the control
    attributes.  Returns the number of blocks that received a control.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise ValueError("configure_influence needs an EncoderDecoder with a DFormerv2 backbone")
    if mode != "legacy" and mode not in SUPPORTED_MODES:
        raise ValueError("unknown influence mode %r" % (mode,))
    chunk = int(query_chunk) if query_chunk else 0
    ctx = {} if context is None else context
    iters = int(iterations)
    tau_value = validate_tau(0.0 if tau is None else tau)

    installed = 0
    for stage_idx, layer in enumerate(backbone.layers):
        for block_idx, block in enumerate(layer.blocks):
            attn = block.Attention
            if mode == "legacy":
                block.influence_control = None
                attn.influence_control = None
                continue
            control = _InfluenceControl(mode, tau_value, chunk, observer, stage_idx, block_idx, ctx, iters)
            block.influence_control = control
            attn.influence_control = control
            installed += 1
    return installed


# --------------------------------------------------------------------------- #
# Online attention summary
# --------------------------------------------------------------------------- #
def _new_group():
    return {
        "calls": 0, "rows": 0, "keys": 0,
        "norm_err_max": 0.0, "abs_t_max": 0.0, "bound_violations": 0,
        "sat_upper": 0, "sat_lower": 0, "tv_sum": 0.0, "kl_sum": 0.0,
        "r_abs_max": 0.0, "r_over_rows_ln2": 0, "r_over_rows_ln4": 0,
        "tiny_p0_keys": 0, "zero_p_with_p0_keys": 0,
    }


class AttentionObserver:
    """Online per-(stage, block, axis, mode, tau) attention summary.

    Nothing is stored per image or per key: only running counters and maxima, so
    enabling it on a few images cannot blow up memory.
    """

    def __init__(self):
        self.groups = {}
        self.calls = 0
        self.log_tiny_p0 = math.log(TINY_P0)

    def __call__(self, *, meta, l0, r, t, p, tau):
        self.calls += 1
        key = (int(meta["stage"]), int(meta["block"]), str(meta["axis"]),
               str(meta["mode"]), round(float(tau), 6))
        acc = self.groups.get(key)
        if acc is None:
            acc = _new_group()
            self.groups[key] = acc
        acc["calls"] += 1
        acc["rows"] += int(np.prod(l0.shape[:-1]))
        acc["keys"] += int(l0.numel())

        with torch.no_grad():
            tau = float(tau)
            acc["norm_err_max"] = max(acc["norm_err_max"], float((p.sum(dim=-1) - 1.0).abs().max().item()))
            acc["abs_t_max"] = max(acc["abs_t_max"], float(t.abs().max().item()))
            acc["bound_violations"] += int(((t > tau + BOUND_TOL) | (t < -tau - BOUND_TOL)).sum().item())
            acc["sat_upper"] += int((t >= tau).sum().item())
            acc["sat_lower"] += int((t <= -tau).sum().item())
            p0 = torch.exp(l0)
            acc["tv_sum"] += float((0.5 * (p - p0).abs().sum(dim=-1)).sum().item())
            if r is not None:
                acc["kl_sum"] += float((p * (t - r)).sum(dim=-1).sum().item())
                acc["r_abs_max"] = max(acc["r_abs_max"], float(r.abs().max().item()))
                abs_r = r.abs()
                acc["r_over_rows_ln2"] += int((abs_r > TAU_VALUES["log2"] + BOUND_TOL).any(dim=-1).sum().item())
                acc["r_over_rows_ln4"] += int((abs_r > TAU_VALUES["log4"] + BOUND_TOL).any(dim=-1).sum().item())
            acc["tiny_p0_keys"] += int((l0 < self.log_tiny_p0).sum().item())
            acc["zero_p_with_p0_keys"] += int(((p == 0) & (p0 > 0)).sum().item())

    def summary(self):
        rows = []
        for key, acc in sorted(self.groups.items()):
            stage, block, axis, mode, tau = key
            n_rows = max(1, acc["rows"])
            n_keys = max(1, acc["keys"])
            rows.append({
                "stage": stage, "block": block, "axis": axis, "mode": mode, "tau": tau,
                "calls": acc["calls"], "rows": acc["rows"], "keys": acc["keys"],
                "norm_err_max": acc["norm_err_max"],
                "abs_t_max": acc["abs_t_max"],
                "bound_violations": acc["bound_violations"],
                "bound_violation_fraction": acc["bound_violations"] / n_keys,
                "sat_upper_fraction": acc["sat_upper"] / n_keys,
                "sat_lower_fraction": acc["sat_lower"] / n_keys,
                "tv_mean": acc["tv_sum"] / n_rows,
                "kl_mean": acc["kl_sum"] / n_rows,
                "r_abs_max": acc["r_abs_max"],
                "r_over_row_fraction_ln2": acc["r_over_rows_ln2"] / n_rows,
                "r_over_row_fraction_ln4": acc["r_over_rows_ln4"] / n_rows,
                "tiny_p0_key_fraction": acc["tiny_p0_keys"] / n_keys,
                "zero_prob_with_positive_p0_fraction": acc["zero_p_with_p0_keys"] / n_keys,
            })
        return rows


def geo_weight_summary(model):
    """The 29 signed ``(ws, wd)`` geometry kernels of the loaded checkpoint."""
    rows = []
    for stage_idx, layer in enumerate(model.backbone.layers):
        for block_idx, block in enumerate(layer.blocks):
            values = block.Geo.weight.detach().float().cpu().reshape(-1).numpy()
            rows.append({"stage": stage_idx, "block": block_idx,
                         "ws": float(values[0]), "wd": float(values[1])})
    return rows


# --------------------------------------------------------------------------- #
# Conditions, regions and counting
# --------------------------------------------------------------------------- #
class ConditionInput:
    __slots__ = ("depth", "support", "region_info", "skipped_reason")

    def __init__(self, depth, support, region_info, skipped_reason=None):
        self.depth = depth
        self.support = support
        self.region_info = region_info
        self.skipped_reason = skipped_reason


def depth_to_raw8(depth_bchw):
    """Recover the author's 8-bit depth of channel 0 from the normalised tensor."""
    channel = depth_bchw[0, 0]
    raw = 255.0 * (DEPTH_RAW8_SCALE * channel + DEPTH_RAW8_OFFSET)
    return torch.round(raw).clamp(0.0, 255.0)


def support_rect(support_2d):
    """``(top, left, height, width)`` of a rectangular support, else ``None``."""
    observed = support_2d > 0
    rows = observed.any(dim=1).nonzero().flatten()
    cols = observed.any(dim=0).nonzero().flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    if (bottom - top) * (right - left) != int(observed.sum()):
        return None
    return top, left, bottom - top, right - left


def band4(mask):
    """``dilate_square(mask, 4) & ~erode_square(mask, 4)`` (9x9, outside = False)."""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    dilated = cv2.dilate(m, _BAND4_KERNEL, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    eroded = cv2.erode(m, _BAND4_KERNEL, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (dilated > 0) & ~(eroded > 0)


def distance_region_masks(changed):
    """Euclidean distance buckets to ``C_delta`` (0-16, 17-64, >64 pixels)."""
    if not changed.any():
        return {}
    inverse = (~changed).astype(np.uint8)
    dist = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
    d = np.ceil(dist)
    return {
        "dist_0_16": d <= 16.0,
        "dist_17_64": (d >= 17.0) & (d <= 64.0),
        "dist_gt64": d > 64.0,
    }


def _freeze_regions(region):
    frozen = dict(region)
    frozen["distance_masks"] = distance_region_masks(region["changed_mask"])
    return frozen


def make_conditions(depth_bchw, support_b1hw, original_index, names, hole_seed):
    """Build every requested condition for one image (batch size 1).

    The clean RGB/GT/padding are untouched; only depth (and, for holes, the
    explicit support) changes.  ``region_info`` masks are plain numpy arrays and
    are diagnostic labels only.
    """
    height, width = int(depth_bchw.shape[-2]), int(depth_bchw.shape[-1])
    clean_raw8 = depth_to_raw8(depth_bchw).detach().cpu().numpy()
    clean_support = support_b1hw[0, 0] > 0
    clean_support_np = clean_support.detach().cpu().numpy()
    out = {}

    if "clean" in names:
        out["clean"] = ConditionInput(depth_bchw, support_b1hw, _freeze_regions({
            "changed_mask": np.zeros((height, width), bool),
            "seam_mask": np.zeros((height, width), bool),
            "deleted_mask": None, "operation_mask": None, "assignment_mask": None,
        }))

    rect = support_rect(clean_support)

    if "holes75" in names:
        try:
            sup, dep, _info = sched.apply_artificial_holes(
                support_b1hw, depth_bchw, HOLES_RATIO, hole_seed, start_index=original_index)
        except ValueError as exc:  # irregular support: refuse, never silently change severity
            out["holes75"] = ConditionInput(None, None, None, "artificial holes refused: %s" % exc)
        else:
            deleted = clean_support_np & (sup[0, 0].detach().cpu().numpy() <= 0)
            changed = deleted & (clean_raw8 != 0)
            out["holes75"] = ConditionInput(dep, sup, _freeze_regions({
                "changed_mask": changed, "seam_mask": band4(deleted),
                "deleted_mask": deleted, "operation_mask": deleted, "assignment_mask": None,
            }))

    offset_reason = None
    if rect is None:
        offset_reason = "no rectangular support; offset condition skipped"
    elif rect[3] <= 2 * OFFSET_PIXELS:
        offset_reason = "support width %d <= %d; offset condition skipped" % (rect[3], 2 * OFFSET_PIXELS)

    if offset_reason is None:
        top, left, height_s, width_s = rect
        ch = min(height_s, OPERATION_MAX_SIDE)
        cw = min(width_s, OPERATION_MAX_SIDE)
        oy = top + (height_s - ch) // 2
        ox = left + (width_s - cw) // 2

    if "cyclic16" in names:
        if offset_reason is not None:
            out["cyclic16"] = ConditionInput(None, None, None, offset_reason)
        else:
            dep = depth_bchw.clone()
            window = dep[..., oy:oy + ch, ox:ox + cw]
            dep[..., oy:oy + ch, ox:ox + cw] = torch.roll(window, shifts=OFFSET_PIXELS, dims=-1)
            operation = np.zeros((height, width), bool)
            operation[oy:oy + ch, ox:ox + cw] = True
            strip = np.zeros((height, width), bool)
            strip[oy:oy + ch, ox:ox + OFFSET_PIXELS] = True
            corrupt_raw8 = depth_to_raw8(dep).detach().cpu().numpy()
            changed = clean_support_np & (corrupt_raw8 != clean_raw8)
            out["cyclic16"] = ConditionInput(dep, support_b1hw, _freeze_regions({
                "changed_mask": changed,
                "seam_mask": strip | band4(strip) | band4(operation),
                "deleted_mask": None, "operation_mask": operation, "assignment_mask": None,
            }))

    if "noncyclic16" in names:
        if offset_reason is not None:
            out["noncyclic16"] = ConditionInput(None, None, None, offset_reason)
        else:
            operation = np.zeros((height, width), bool)
            operation[oy:oy + ch, ox:ox + cw] = True
            local_support = clean_support_np[oy:oy + ch, ox:ox + cw]
            local_raw8 = clean_raw8[oy:oy + ch, ox:ox + cw]
            source_support = np.zeros_like(local_support)
            source_support[:, OFFSET_PIXELS:] = local_support[:, :-OFFSET_PIXELS]
            source_raw8 = np.zeros_like(local_raw8)
            source_raw8[:, OFFSET_PIXELS:] = local_raw8[:, :-OFFSET_PIXELS]
            assignment_local = local_support & source_support & (local_raw8 > 0) & (source_raw8 > 0)
            assignment = np.zeros((height, width), bool)
            assignment[oy:oy + ch, ox:ox + cw] = assignment_local

            clean_depth_np = depth_bchw[0].detach().cpu().numpy()
            shifted = np.zeros_like(clean_depth_np)
            shifted[:, :, OFFSET_PIXELS:] = clean_depth_np[:, :, :-OFFSET_PIXELS]
            target = depth_bchw.clone()
            target_np = target[0].detach().cpu().numpy()
            target_np[:, assignment] = shifted[:, assignment]
            target[0] = torch.from_numpy(np.ascontiguousarray(target_np)).to(
                device=depth_bchw.device, dtype=depth_bchw.dtype)
            corrupt_raw8 = depth_to_raw8(target).detach().cpu().numpy()
            changed = clean_support_np & (corrupt_raw8 != clean_raw8)
            out["noncyclic16"] = ConditionInput(target, support_b1hw, _freeze_regions({
                "changed_mask": changed, "seam_mask": band4(assignment),
                "deleted_mask": None, "operation_mask": operation, "assignment_mask": assignment,
            }))

    return out


def build_regions(region_info, gt_valid):
    """Region masks for one condition, all already restricted to valid GT."""
    changed = region_info["changed_mask"]
    seam = region_info["seam_mask"]
    regions = {
        "inside": gt_valid & changed,
        "outside": gt_valid & ~changed,
        "inside_J": gt_valid & changed & seam,
        "outside_J": gt_valid & ~changed & seam,
        "inside_notJ": gt_valid & changed & ~seam,
        "outside_notJ": gt_valid & ~changed & ~seam,
    }
    deleted = region_info.get("deleted_mask")
    if deleted is not None and deleted.any():
        regions["deleted_all"] = gt_valid & deleted
    operation = region_info.get("operation_mask")
    if operation is not None and operation.any():
        regions["operation"] = gt_valid & operation
    for name, mask in (region_info.get("distance_masks") or {}).items():
        regions[name] = gt_valid & mask
    return regions


class RegionPack:
    """Region slices for one image/condition, reused by every method."""

    def __init__(self, gt, o_clean, o_corrupt, regions):
        gt_flat = gt.reshape(-1)
        clean_flat = o_clean.reshape(-1)
        corrupt_flat = o_corrupt.reshape(-1)
        self.entries = {}
        for name, mask in regions.items():
            idx = np.flatnonzero(mask.reshape(-1))
            if idx.size == 0:
                continue
            g = gt_flat[idx]
            a = clean_flat[idx]
            b = corrupt_flat[idx]
            self.entries[name] = {"idx": idx, "g": g, "a": a, "b": b,
                                  "a_ok": a == g, "b_ok": b == g}

    def names(self):
        return sorted(self.entries.keys())

    def counts(self, name, method_pred):
        entry = self.entries.get(name)
        if entry is None:
            return None
        c = method_pred.reshape(-1)[entry["idx"]]
        c_ok = c == entry["g"]
        a_ok = entry["a_ok"]
        b_ok = entry["b_ok"]
        return {
            "pixels": int(entry["idx"].size),
            "o_clean_correct": int(a_ok.sum()),
            "o_corrupt_correct": int(b_ok.sum()),
            "method_correct": int(c_ok.sum()),
            "new_err_original": int((a_ok & ~b_ok).sum()),
            "new_err_method": int((a_ok & ~c_ok).sum()),
            "repairable": int((a_ok & ~b_ok).sum()),
            "repair": int((a_ok & ~b_ok & c_ok).sum()),
            "harm_den": int(b_ok.sum()),
            "harm": int((b_ok & ~c_ok).sum()),
            "gained_from_clean_wrong": int((~a_ok & c_ok).sum()),
            "net_correct_delta": int(c_ok.sum()) - int(b_ok.sum()),
            "net_correct_vs_clean": int(c_ok.sum()) - int(a_ok.sum()),
        }


def hist_metrics(hist):
    diag = np.diag(hist).astype(np.float64)
    row = hist.sum(1).astype(np.float64)
    col = hist.sum(0).astype(np.float64)
    denom = col + row - diag
    iou = np.where(denom > 0, diag / np.maximum(denom, 1.0), 0.0)
    acc = np.where(row > 0, diag / np.maximum(row, 1.0), 0.0)
    f1 = np.where(col + row > 0, 2 * diag / np.maximum(col + row, 1.0), 0.0)
    total = float(hist.sum())
    return {
        "miou": round(float(iou.mean()) * 100, 4),
        "macc": round(float(acc.mean()) * 100, 4),
        "mf1": round(float(f1.mean()) * 100, 4),
        "pixel_acc": round(float(diag.sum() / total) * 100, 4) if total > 0 else None,
        "per_class_iou": [round(float(v) * 100, 4) for v in iou],
    }


def _rate(numerator, denominator):
    return (float(numerator) / float(denominator)) if denominator else None


def summarise_regions(rows):
    agg = {}
    for row in rows:
        key = (row["phase"], row["condition"], row["method"], row["region"])
        acc = agg.setdefault(key, {k: 0 for k in COUNTER_KEYS})
        for k in COUNTER_KEYS:
            value = row.get(k)
            if value is not None:
                acc[k] += int(value)
    out = []
    for (phase, condition, method, region), acc in sorted(agg.items()):
        out.append({
            "phase": phase, "condition": condition, "method": method, "region": region,
            "pixels": acc["pixels"],
            "new_err_original_rate": _rate(acc["new_err_original"], acc["pixels"]),
            "new_err_method_rate": _rate(acc["new_err_method"], acc["pixels"]),
            "repair_rate": _rate(acc["repair"], acc["repairable"]),
            "harm_rate": _rate(acc["harm"], acc["harm_den"]),
            "net_correct_delta_rate": _rate(acc["net_correct_delta"], acc["pixels"]),
            "gained_from_clean_wrong_rate": _rate(acc["gained_from_clean_wrong"], acc["pixels"]),
            "counts": dict(acc),
        })
    return out


# --------------------------------------------------------------------------- #
# Method / phase planning
# --------------------------------------------------------------------------- #
class MethodSpec:
    __slots__ = ("label", "mode", "tau")

    def __init__(self, label, mode, tau):
        self.label = label
        self.mode = mode
        self.tau = tau


def expand_methods(names, tau_arg, phase):
    if not names:
        names = PHASE_DEFAULT_METHODS[phase]
    if phase in ("noncyclic32", "confirm496") and tau_arg == "both":
        raise SystemExit("phase %s requires a single --tau; 'both' is only valid for pilot32" % phase)
    tau_labels = ["log2", "log4"] if tau_arg == "both" else [tau_arg]
    specs = []
    for name in names:
        if name in ("original", "depth_off"):
            specs.append(MethodSpec(name, name, TAU_VALUES[tau_labels[0]]))
        elif name in ("kl", "clip", "gate"):
            for label in tau_labels:
                specs.append(MethodSpec("%s-%s" % (name, label), name, TAU_VALUES[label]))
        else:
            raise SystemExit("unknown method %r; use original, depth_off, kl, clip or gate" % (name,))
    return specs


def phase_indices(phase, total):
    screening = [k * total // SCREEN_COUNT for k in range(SCREEN_COUNT)]
    if phase in ("pilot32", "noncyclic32"):
        return screening
    taken = set(screening)
    return [i for i in range(total) if i not in taken]


# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #
def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(column) for column in header])


def _read_csv_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coerce_region_row(row):
    out = {"phase": row["phase"], "condition": row["condition"], "method": row["method"],
           "region": row["region"], "image_index": int(row["image_index"])}
    for key in COUNTER_KEYS:
        value = row.get(key)
        out[key] = None if value in (None, "", "None") else int(float(value))
    return out


def _git_state():
    def run(*args):
        try:
            return subprocess.check_output(["git"] + list(args), cwd=os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), stderr=subprocess.DEVNULL).decode("utf-8", "replace").strip()
        except Exception:
            return None
    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--porcelain")}


# --------------------------------------------------------------------------- #
# Reuse of an earlier phase's statistics (no inference is repeated)
# --------------------------------------------------------------------------- #
def _locate_phase_dir(reuse_path, prefer):
    if not reuse_path:
        return None
    for name in prefer:
        candidate = os.path.join(str(reuse_path), name)
        if os.path.isfile(os.path.join(candidate, "regions.csv")):
            return candidate
    if os.path.isfile(os.path.join(str(reuse_path), "regions.csv")):
        return str(reuse_path)
    return None


def load_reuse_for_conditions(reuse_path, phase, conditions, methods, region_rows, confusions):
    """Copy clean/holes75 statistics from an earlier 32-image phase."""
    source = _locate_phase_dir(reuse_path, ["pilot32"])
    if source is None:
        raise SystemExit("--reuse %s holds no reusable 32-image phase" % reuse_path)
    labels = {m.label for m in methods}
    wanted = [c for c in conditions if c != "noncyclic16"]
    copied_rows = 0
    for row in _read_csv_rows(os.path.join(source, "regions.csv")):
        if row["condition"] in wanted and row["method"] in labels:
            region_rows.append(_coerce_region_row(row))
            copied_rows += 1
    with open(os.path.join(source, "confusion.json"), "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    copied_matrices = 0
    for condition in wanted:
        for method in methods:
            matrix = saved.get(condition, {}).get(method.label)
            if matrix is None:
                continue
            confusions[(condition, method.label)] += np.array(matrix, dtype=np.int64)
            copied_matrices += 1
    return {"source": source, "region_rows": copied_rows, "confusion_matrices": copied_matrices,
            "conditions": wanted}


def merge_528(reuse_path, phase_rows, phase_confusions, conditions, methods):
    """Legally combine the 32-image and 496-image halves into a 528 summary."""
    source = _locate_phase_dir(reuse_path, ["noncyclic32", "pilot32"])
    if source is None:
        return None
    with open(os.path.join(source, "confusion.json"), "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    src_rows = [_coerce_region_row(r) for r in _read_csv_rows(os.path.join(source, "regions.csv"))]
    combined_rows = src_rows + phase_rows
    tasks = {}
    missing_in_32 = []
    for condition in conditions:
        tasks[condition] = {}
        for method in methods:
            matrix = saved.get(condition, {}).get(method.label)
            if matrix is None:
                missing_in_32.append({"condition": condition, "method": method.label})
                base = np.zeros_like(phase_confusions[(condition, method.label)])
            else:
                base = np.array(matrix, dtype=np.int64)
            tasks[condition][method.label] = hist_metrics(base + phase_confusions[(condition, method.label)])
    return {
        "source_32": source,
        "source_496_phase": "confirm496",
        "confirm_conditions": list(conditions),
        "note": ("528 = 32-image screening/verification half + 496-image confirmation half; "
                 "each confusion matrix is summed before the metric is recomputed, the per-image "
                 "mIoU values are never averaged.  Only the conditions present in both halves are "
                 "genuine 528-image results; anything else is a 32-image result and is labelled so"),
        "missing_in_32": missing_in_32,
        "tasks": tasks,
        "regions": summarise_regions(combined_rows),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(description="Direction-A frozen influence diagnostic (SUN RGB-D).")
    parser.add_argument("--phase", required=True, choices=PHASE_CHOICES)
    parser.add_argument("--config", default="local_configs.research.INFLUENCE_SUNRGBD")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="original depth_off kl clip gate (defaults per phase)")
    parser.add_argument("--conditions", nargs="+", default=None, choices=CONDITION_CHOICES,
                        help="override the phase conditions (only when a condition is settled)")
    parser.add_argument("--tau", choices=("log2", "log4", "both"), default=None,
                        help="per-axis budget (default: both for pilot32, log2 otherwise)")
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERATIONS,
                        help="bisection depth of the KL projection (default 24)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ham-seed", type=int, default=12345)
    parser.add_argument("--hole-seed", type=int, default=HOLES_SEED)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    parser.add_argument("--reuse", default=None, help="earlier phase directory (statistics only)")
    parser.add_argument("--observe-images", type=int, default=None,
                        help="images with the online attention summary (default 4, 0 for confirm496)")
    parser.add_argument("--perf-images", type=int, default=8)
    parser.add_argument("--legacy-compare", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="debug cap on images (0 = all)")
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.phase == "self-check":
        run_operator_self_check(verbose=True)
        return 0

    if not args.checkpoint or not args.out:
        raise SystemExit("--checkpoint and --out are required for phase %s" % args.phase)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("REFUSED: no CUDA device available; the diagnostic never falls back to a long CPU run.",
              file=sys.stderr)
        return 3

    if args.observe_images is None:
        args.observe_images = 0 if args.phase == "confirm496" else 4
    if args.tau is None:
        args.tau = "both" if args.phase == "pilot32" else "log2"

    config = deepcopy(getattr(import_module(args.config), "C"))
    config.geometry_mode = "original"
    use_support, model, device = build_model(args, config)

    eval_source = str(config.dev_eval_source)
    train_source = str(config.train_dev_source)
    num_workers = int(args.num_workers) if args.num_workers is not None else int(config.num_workers)
    loader, total = sched.build_eval_loader(config, eval_source, train_source, 1, num_workers)
    dataset = loader.dataset
    if total != DEV_SIZE:
        print("WARNING: dev split holds %d images, the plan assumes %d" % (total, DEV_SIZE), file=sys.stderr)

    indices = phase_indices(args.phase, total)
    if args.phase == "confirm496":
        screening = set(phase_indices("pilot32", total))
        assert screening.isdisjoint(indices), "confirmation set must be the strict complement"
        assert len(indices) == total - SCREEN_COUNT, "confirmation set size mismatch"

    return run_phase(args, config, model, device, use_support, dataset, indices, num_workers)


def build_model(args, config):
    state_dict, payload = _load_checkpoint(args.checkpoint)
    contract = payload.get("contract", {}) if payload else {}
    for field in ("dataset_name", "backbone", "num_classes"):
        saved = contract.get(field)
        expected = getattr(config, field)
        if saved is not None and str(saved) != str(expected):
            raise SystemExit("checkpoint contract %s=%r conflicts with config %r; refusing to run"
                             % (field, saved, expected))
    saved_geometry = payload.get("geometry_mode") if payload else None
    if saved_geometry is not None and str(saved_geometry) != "original":
        raise SystemExit("checkpoint geometry_mode=%r is not the frozen original path" % (saved_geometry,))

    device = torch.device(args.device)
    sched.seed_everything(int(args.seed))
    from models.builder import EncoderDecoder as segmodel
    model = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    restored = sched.restore_decoder_norm_config(model, config, norm_layer=nn.BatchNorm2d)
    if not restored:
        raise SystemExit("no decoder norm layer found; refusing to evaluate with unknown norm config")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise SystemExit("strict load reported missing=%s unexpected=%s" % (missing, unexpected))
    use_support = sched.supports_depth_support(model)
    model.to(device)
    model.eval()
    print("[influence] checkpoint=%s geometry=original device=%s strict=True" % (args.checkpoint, device))
    return use_support, model, device


def _load_checkpoint(path):
    from research import evaluate_odg
    return evaluate_odg.load_state_dict(path, allow_raw=False)


# --------------------------------------------------------------------------- #
# Phase execution
# --------------------------------------------------------------------------- #
def run_phase(args, config, model, device, use_support, dataset, indices, num_workers):
    phase = args.phase
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    conditions = list(args.conditions) if args.conditions else list(PHASE_CONDITIONS[phase])
    methods = expand_methods(args.methods, args.tau, phase)
    num_classes = int(config.num_classes)
    ignore_index = int(config.background)
    chunk = int(args.query_chunk) if args.query_chunk else 0
    amp = bool(args.amp) and device.type == "cuda"
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp else contextlib.nullcontext())

    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=int(num_workers), drop_last=False,
        pin_memory=device.type == "cuda", persistent_workers=int(num_workers) > 0)

    observer = AttentionObserver() if int(args.observe_images) > 0 else None
    context = {}
    confusions = {(c, m.label): np.zeros((num_classes, num_classes), np.int64)
                  for c in conditions for m in methods}
    region_rows = []
    task_rows = []
    skipped = []
    perf = {}
    legacy_records = []
    images_run = 0

    reuse_note = None
    if phase == "noncyclic32":
        if not args.reuse:
            raise SystemExit("noncyclic32 reuses clean/holes75 from pilot32; pass --reuse <pilot32 dir>")
        reuse_note = load_reuse_for_conditions(args.reuse, phase, conditions, methods, region_rows, confusions)

    # Conditions whose statistics came from the earlier phase are not re-evaluated
    # (no inference is repeated).  The clean condition is still run for Original
    # only, because the shared O-clean reference the region counts need is a
    # prediction, not a summary statistic.
    reused_conditions = set(reuse_note["conditions"]) if reuse_note else set()
    active_conditions = [c for c in conditions if c not in reused_conditions]
    inference_conditions = (["clean"] + active_conditions) if "clean" in reused_conditions else list(active_conditions)

    def methods_for(condition):
        if condition == "clean" and "clean" in reused_conditions:
            return [m for m in methods if m.label == "original"]
        return methods

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.time()
    for subset_pos, sample in enumerate(loader):
        if args.max_images and subset_pos >= int(args.max_images):
            break
        original_index = int(indices[subset_pos])
        data = sample["data"].to(device)
        label = sample["label"].to(device)
        modal_x = sample["modal_x"].to(device)
        support = sample["depth_support"].to(device)
        gt_np = label[0].detach().cpu().numpy().astype(np.int64)

        cond_inputs = make_conditions(modal_x, support, original_index, conditions, int(args.hole_seed))
        for name, cond in cond_inputs.items():
            if cond.skipped_reason:
                skipped.append({"image_index": original_index, "condition": name,
                                "reason": cond.skipped_reason})

        sched.seed_everything((int(args.ham_seed) + original_index) % (2 ** 31))
        paired_rng = sched.capture_rng()

        predictions = {}
        for condition in inference_conditions:
            cond = cond_inputs[condition]
            if cond.skipped_reason:
                continue
            context["condition"] = condition
            context["original_index"] = original_index
            for method in methods_for(condition):
                sched.restore_rng(paired_rng)
                active_observer = observer if subset_pos < int(args.observe_images) else None
                configure_influence(model, method.mode, method.tau, chunk, active_observer,
                                    context=context, iterations=int(args.iters))
                # The first image of the phase warms the kernels up; timing starts after it.
                timed = 0 < subset_pos <= int(args.perf_images)
                if timed and device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    with autocast_ctx:
                        logits = sched.forward_logits(model, data, cond.depth, cond.support, use_support)
                        if tuple(logits.shape[-2:]) != tuple(label.shape[-2:]):
                            logits = torch.nn.functional.interpolate(
                                logits, size=tuple(label.shape[-2:]), mode="bilinear", align_corners=False)
                        probs = logits.float().softmax(dim=1)
                if timed and device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                predictions[(condition, method.label)] = (
                    probs.argmax(dim=1)[0].to(torch.uint8).detach().cpu().numpy())
                if timed:
                    acc = perf.setdefault((condition, method.label), {"images": 0, "seconds": 0.0})
                    acc["images"] += 1
                    acc["seconds"] += elapsed
                del logits, probs

        if (phase == "pilot32" and subset_pos < int(args.legacy_compare)
                and not cond_inputs["clean"].skipped_reason):
            legacy_records.append(_legacy_compare(
                model, data, label, cond_inputs["clean"], paired_rng, chunk, autocast_ctx,
                device, use_support, int(args.iters), context, original_index, methods))

        o_clean = predictions.get(("clean", "original"))
        if o_clean is None:
            skipped.append({"image_index": original_index, "condition": "clean",
                            "reason": "no clean original prediction; image excluded from statistics"})
            continue
        o_clean_i = o_clean.astype(np.int64)
        gt_valid = gt_np != ignore_index
        for condition in active_conditions:
            cond = cond_inputs[condition]
            if cond.skipped_reason:
                continue
            o_corrupt = predictions.get((condition, "original"))
            if o_corrupt is None:
                continue
            o_corrupt_i = o_corrupt.astype(np.int64)
            regions = build_regions(cond.region_info, gt_valid)
            pack = RegionPack(gt_np, o_clean_i, o_corrupt_i, regions)
            for method in methods:
                pred = predictions.get((condition, method.label))
                if pred is None:
                    continue
                pred_i = pred.astype(np.int64)
                for name in pack.names():
                    counts = pack.counts(name, pred_i)
                    if counts is None:
                        continue
                    row = {"phase": phase, "condition": condition, "method": method.label,
                           "image_index": original_index, "region": name}
                    row.update(counts)
                    region_rows.append(row)
                valid_pred = pred_i[gt_valid]
                valid_gt = gt_np[gt_valid]
                corrupt_ok = int((o_corrupt_i[gt_valid] == valid_gt).sum())
                clean_ok = int((o_clean_i[gt_valid] == valid_gt).sum())
                method_ok = int((valid_pred == valid_gt).sum())
                task_rows.append({
                    "phase": phase, "condition": condition, "method": method.label,
                    "image_index": original_index, "gt_valid_pixels": int(gt_valid.sum()),
                    "correct_pixels": method_ok,
                    "net_correct_delta_vs_corrupt": method_ok - corrupt_ok,
                    "net_correct_delta_vs_clean": method_ok - clean_ok,
                })
                hist = confusions[(condition, method.label)]
                idx = valid_gt * num_classes + valid_pred
                hist += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        images_run += 1
        del predictions

    configure_influence(model, "legacy")
    elapsed_total = time.time() - started
    peak_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None

    _write_csv(os.path.join(out_dir, "regions.csv"), REGION_COLUMNS, region_rows)
    _write_csv(os.path.join(out_dir, "tasks.csv"), TASK_COLUMNS, task_rows)
    sched.write_json(os.path.join(out_dir, "confusion.json"), {
        condition: {method.label: confusions[(condition, method.label)].tolist() for method in methods}
        for condition in conditions})
    _write_csv(os.path.join(out_dir, "geo_weights.csv"), ["stage", "block", "ws", "wd"],
               geo_weight_summary(model))
    if observe_records := (observer.summary() if observer is not None else None):
        _write_csv(os.path.join(out_dir, "attention_summary.csv"),
                   ["stage", "block", "axis", "mode", "tau", "calls", "rows", "keys", "norm_err_max",
                    "abs_t_max", "bound_violations", "bound_violation_fraction", "sat_upper_fraction",
                    "sat_lower_fraction", "tv_mean", "kl_mean", "r_abs_max",
                    "r_over_row_fraction_ln2", "r_over_row_fraction_ln4", "tiny_p0_key_fraction",
                    "zero_prob_with_positive_p0_fraction"], observe_records)

    summary = {
        "phase": phase,
        "config": args.config,
        "checkpoint": os.path.abspath(args.checkpoint),
        "conditions": conditions,
        "methods": [{"label": m.label, "mode": m.mode, "tau": m.tau} for m in methods],
        "tau_argument": args.tau,
        "projection_iterations": int(args.iters),
        "query_chunk": chunk,
        "ham_seed": int(args.ham_seed),
        "hole_seed": int(args.hole_seed),
        "device": str(device),
        "amp": amp,
        "images_run": images_run,
        "indices": indices,
        "reuse": reuse_note,
        "tasks": {condition: {method.label: hist_metrics(confusions[(condition, method.label)])
                              for method in methods} for condition in conditions},
        "regions": summarise_regions(region_rows),
        "skipped": skipped,
        "perf": [{"condition": c, "method": m, "images": v["images"], "seconds": v["seconds"],
                  "seconds_per_image": v["seconds"] / max(1, v["images"])}
                 for (c, m), v in sorted(perf.items())],
        "observer": observer.summary() if observer is not None else None,
        "legacy_compare": legacy_records,
        "elapsed_seconds": round(elapsed_total, 2),
        "peak_gpu_memory_mb": peak_mb,
        "git": _git_state(),
    }
    sched.write_json(os.path.join(out_dir, "summary.json"), summary)

    if phase == "confirm496" and args.reuse:
        merged = merge_528(args.reuse, region_rows, confusions, conditions, methods)
        if merged is not None:
            sched.write_json(os.path.join(out_dir, "merged528.json"), merged)

    print("[influence] phase=%s images=%d elapsed=%.1fs" % (phase, images_run, elapsed_total))
    for condition in conditions:
        for method in methods:
            metrics = summary["tasks"][condition][method.label]
            print("[influence] %-9s %-12s mIoU=%.4f pixel_acc=%s"
                  % (condition, method.label, metrics["miou"], metrics["pixel_acc"]))
    return 0


def _legacy_compare(model, data, label, clean_input, paired_rng, chunk, autocast_ctx, device,
                    use_support, iters, context, original_index, methods):
    """Compare the author legacy path against the shared-FP32 original baseline.

    Everything is reduced to numpy inside the inference block so nothing has to
    travel out of inference mode, and the legacy path is run twice on the same
    paired RNG state to show that enabling/clearing the control is reversible.
    """

    def _forward():
        with torch.inference_mode():
            with autocast_ctx:
                logits = sched.forward_logits(model, data, clean_input.depth, clean_input.support, use_support)
                if tuple(logits.shape[-2:]) != tuple(label.shape[-2:]):
                    logits = torch.nn.functional.interpolate(
                        logits, size=tuple(label.shape[-2:]), mode="bilinear", align_corners=False)
                probs = logits.float().softmax(dim=1)
        return logits.detach().float().cpu().numpy(), probs.detach().cpu().numpy()

    sched.restore_rng(paired_rng)
    configure_influence(model, "legacy")
    legacy_logits, legacy_probs = _forward()

    sched.restore_rng(paired_rng)
    configure_influence(model, "legacy")
    _repeat_logits, repeat_probs = _forward()

    shared_spec = next(m for m in methods if m.label == "original")
    sched.restore_rng(paired_rng)
    configure_influence(model, MODE_ORIGINAL, shared_spec.tau, chunk, None,
                        context=context, iterations=iters)
    shared_logits, shared_probs = _forward()

    legacy_pred = legacy_probs.argmax(axis=1)[0]
    repeat_pred = repeat_probs.argmax(axis=1)[0]
    shared_pred = shared_probs.argmax(axis=1)[0]
    record = {
        "image_index": int(original_index),
        "legacy_repeat_argmax_identical": bool((legacy_pred == repeat_pred).all()),
        "legacy_vs_shared_argmax_agreement": float((legacy_pred == shared_pred).mean()),
        "legacy_vs_shared_max_logit_diff": float(np.abs(legacy_logits - shared_logits).max()),
        "legacy_vs_shared_max_prob_diff": float(np.abs(legacy_probs - shared_probs).max()),
    }
    print("[influence] legacy vs shared-FP32 original (image %d): agreement=%.4f max|dlogit|=%.3e"
          % (original_index, record["legacy_vs_shared_argmax_agreement"],
             record["legacy_vs_shared_max_logit_diff"]))
    return record


if __name__ == "__main__":
    sys.exit(main())
