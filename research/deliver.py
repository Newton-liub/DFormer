"""DeLiVER RGB-D dataset layer for the DFormer research entries.

Approved scope (``doc/plans/2026-10-10-deliver-integration.md``, 2026-10-10):
**RGB + the native single-channel ``depth/`` uint8 image**.  ``hha/``, ``lidar/``
and ``event/`` are deliberately unused, so this protocol is *not* the official
DeLiVER HHA RGB-D protocol and its scores are not comparable with the official
numbers.  ``depth/`` values are treated as a fixed image representation
(normalised with the author rule ``(D8/255 - 0.48)/0.28``); their physical unit
and invalid-value convention are unknown and are not invented here.

Layout (``<root>/<modality>/<weather>/<split>/<scene>/<name>_<modality>_<view>.png``)::

    img/cloud/train/MAP_1_point102/110100_rgb_front.png
    depth/cloud/train/MAP_1_point102/110100_depth_front.png
    semantic/cloud/train/MAP_1_point102/110100_semantic_front.png

The manifests hold the path relative to ``img/`` with forward slashes.  The depth
and label paths are the same relative path with ``_rgb`` replaced by ``_depth`` /
``_semantic`` and the modality root swapped.  This mirrors the official loader
(``semseg/datasets/deliver.py``: ``rgb.replace('/img', '/hha').replace('_rgb', '_depth')``)
except that the official "depth" modality is ``hha/`` while this protocol uses the
raw ``depth/`` directory.

Labels are single-frame ID images.  The official reader takes the *red* channel
(``io.read_image(...)[0]`` is RGB order), zeroes 255 and subtracts one with uint8
wraparound, so ``1..25 -> 0..24`` while both ``0`` and ``255 -> ignore (255)``.
:func:`map_deliver_label` reproduces that mapping explicitly and refuses unknown
ids instead of silently dropping them.

CLI (run from a repository root with the project ``dformer`` environment)::

    python -m research.deliver prepare --root D:/0Project/dataset/DELIVER \
        --split-dir research/splits/deliver_official
    python -m research.deliver preview --config local_configs.research.DFormerv2_S_DeLiVER \
        --limit 6 --out outputs/deliver-preparation
    python -m research.deliver smoke --config local_configs.research.DFormerv2_S_DeLiVER \
        --device cpu --size 128 --out outputs/deliver-preparation
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch

# Make the repository root importable when the module is executed directly
# (``python research/deliver.py``), while keeping the author-relative imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from research.data import (  # noqa: E402
    ObservationTrainPre,
    ObservationValPre,
)

__all__ = [
    "DELIVER_CLASS_NAMES",
    "DELIVER_PALETTE",
    "DELIVER_NUM_CLASSES",
    "DELIVER_IGNORE",
    "DELIVER_WEATHERS",
    "DELIVER_CAMERA_FAULTS",
    "DELIVER_UNUSED_FAULTS",
    "DELIVER_EXPECTED_COUNTS",
    "DeLiVERDataset",
    "ObservationDataset",
    "ObservationTrainPre",
    "ObservationValPre",
    "map_deliver_label",
    "deliver_relative_paths",
    "scene_parts",
    "split_weather_fault",
    "complete_optimizer_groups",
    "deliver_setting",
    "enumerate_split_lines",
]

# --------------------------------------------------------------------------- #
# Label protocol (25 classes; order and palette copied from the official loader
# ``semseg/datasets/deliver.py`` so the research numbers can be compared with
# the paper once the input protocol is aligned)
# --------------------------------------------------------------------------- #
DELIVER_CLASS_NAMES = [
    "Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road",
    "SideWalk", "Vegetation", "Cars", "Wall", "TrafficSign", "Sky", "Ground",
    "Bridge", "RailTrack", "GroundRail", "TrafficLight", "Static", "Dynamic",
    "Water", "Terrain", "TwoWheeler", "Bus", "Truck",
]
DELIVER_PALETTE = [
    [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60], [153, 153, 153],
    [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35],
    [0, 0, 142], [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81],
    [150, 100, 100], [230, 150, 140], [180, 165, 180], [250, 170, 30],
    [110, 190, 160], [170, 120, 50], [45, 60, 150], [145, 170, 100],
    [0, 0, 230], [0, 60, 100], [0, 0, 70],
]
DELIVER_NUM_CLASSES = len(DELIVER_CLASS_NAMES)
DELIVER_IGNORE = 255

DELIVER_WEATHERS = ("cloud", "fog", "night", "rain", "sun")
# Native camera faults really present in the shipped scenes.
DELIVER_CAMERA_FAULTS = ("underexposure", "overexposure", "motionblur")
# Faults that act on modalities this protocol does not read.
DELIVER_UNUSED_FAULTS = ("lidarjitter", "eventlowres")
DELIVER_SPLITS = ("train", "val", "test")
# Reference totals from the dataset handoff record; a deviation is reported, not fixed.
DELIVER_EXPECTED_COUNTS = {"train": 3983, "val": 2005, "test": 1897}

RGB_TOKEN = "_rgb"
DEPTH_TOKEN = "_depth"
SEMANTIC_TOKEN = "_semantic"


# --------------------------------------------------------------------------- #
# Path / name helpers
# --------------------------------------------------------------------------- #
def _posix(value):
    return PurePosixPath(str(value).strip().replace("\\", "/"))


def deliver_relative_paths(item_name):
    """Return ``(rgb, depth, label)`` relative paths for one manifest line.

    Raises if the rgb name does not carry the ``_rgb`` token, so a malformed
    manifest fails loudly on the offending sample instead of silently reading the
    rgb file as its own depth/label.
    """
    item = _posix(item_name)
    name = item.name
    if RGB_TOKEN not in name:
        raise ValueError("rgb file name lacks %r: %s" % (RGB_TOKEN, item_name))
    return (
        item,
        item.parent / name.replace(RGB_TOKEN, DEPTH_TOKEN),
        item.parent / name.replace(RGB_TOKEN, SEMANTIC_TOKEN),
    )


def scene_parts(scene_dir_name):
    """``MAP_1_point102`` -> ``('MAP_1_point102', None)``; ``MAP_1_point12_motionblur`` -> ``(...,'motionblur')``."""
    tokens = str(scene_dir_name).split("_")
    base = "_".join(tokens[:3])
    fault = "_".join(tokens[3:]) if len(tokens) > 3 else None
    return base, fault


def split_weather_fault(item_name):
    """``cloud/train/MAP_1_point12_motionblur/x_rgb_front.png`` -> ``('cloud','train','motionblur')``."""
    parts = _posix(item_name).parts
    if len(parts) < 4:
        raise ValueError("manifest line is not <weather>/<split>/<scene>/<file>: %s" % item_name)
    weather, split, scene = parts[0], parts[1], parts[2]
    _base, fault = scene_parts(scene)
    return weather, split, fault


def map_deliver_label(raw_ids):
    """Official DeLiVER id mapping: ``1..25 -> 0..24``, ``0``/``255 -> 255`` (ignore).

    ``raw_ids`` must be the original uint8 image ids (single channel or the red
    channel already extracted).  Unknown ids raise instead of being swallowed.
    """
    raw = np.asarray(raw_ids)
    if raw.dtype != np.uint8:
        raise ValueError("DeLiVER label ids must be uint8, got %s" % raw.dtype)
    valid = (raw >= 1) & (raw <= DELIVER_NUM_CLASSES)
    known = valid | (raw == 0) | (raw == DELIVER_IGNORE)
    if not bool(known.all()):
        unknown = np.unique(raw[~known]).tolist()
        raise ValueError(
            "unexpected DeLiVER label ids %s (expected 1..%d, 0, %d)"
            % (unknown, DELIVER_NUM_CLASSES, DELIVER_IGNORE)
        )
    out = np.full(raw.shape, DELIVER_IGNORE, dtype=np.int64)
    out[valid] = raw[valid].astype(np.int64) - 1
    return out


# --------------------------------------------------------------------------- #
# Raw readers (one decode per array; no silent channel or bit-depth guessing)
# --------------------------------------------------------------------------- #
def _read_rgb(path):
    """Read a three-channel RGB image (native 8-bit; no palette handling)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError("cannot read rgb: %s" % path)
    if raw.ndim != 3 or raw.shape[2] not in (3, 4):
        raise ValueError("expected a 3/4-channel RGB image, got shape %s: %s" % (raw.shape, path))
    if raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def _read_depth(path):
    """Read the native single-channel uint8 depth image (no scaling, no channel pick)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError("cannot read depth: %s" % path)
    if raw.ndim == 3:
        if raw.shape[2] != 1:
            raise ValueError(
                "expected single-channel depth, got %s channels in %s; "
                "report the format difference instead of picking a channel" % (raw.shape[2], path)
            )
        raw = raw[..., 0]
    if raw.dtype != np.uint8:
        raise ValueError("expected uint8 depth, got %s in %s" % (raw.dtype, path))
    return raw


def _read_label_ids(path):
    """Read the original label ids: red channel for 3/4-channel images, else the plane."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError("cannot read label: %s" % path)
    if raw.ndim == 3:
        # OpenCV returns BGR/BGRA, so index 2 is the red channel the official
        # RGB reader would have used.  Never IMREAD_GRAYSCALE here: that would
        # colour-convert the palette and destroy the ids.
        raw = raw[..., 2]
    return raw


def read_modalities(rgb_root, depth_root, label_root, item_name):
    """Read the three protocol modalities for a manifest line.

    Manifest lines are relative to ``img/``, so the caller passes the RGB, depth
    and label roots explicitly (``config.rgb_root_folder`` / ``x_root_folder`` /
    ``gt_root_folder``).  Returns ``{"rgb": (H,W,3) uint8 RGB,
    "depth_raw": (H,W) uint8, "label_raw": (H,W) uint8 original ids,
    "paths": {...}}``.
    """
    rgb_rel, depth_rel, label_rel = deliver_relative_paths(item_name)
    rgb_path = Path(rgb_root) / Path(str(rgb_rel))
    depth_path = Path(depth_root) / Path(str(depth_rel))
    label_path = Path(label_root) / Path(str(label_rel))
    return {
        "rgb": _read_rgb(rgb_path),
        "depth_raw": _read_depth(depth_path),
        "label_raw": _read_label_ids(label_path),
        "paths": {"rgb": str(rgb_path), "depth": str(depth_path), "label": str(label_path)},
    }


# --------------------------------------------------------------------------- #
# Dataset (author-compatible interface for the research loader factories)
# --------------------------------------------------------------------------- #
class DeLiVERDataset(RGBXDataset):
    """``RGBXDataset`` that reads DeLiVER RGB + ``depth/`` and 25-class labels.

    Interface matches ``research.data.ObservationDataset`` so the existing
    ``research/odg_schedule.py`` loader factories, ``ObservationTrainPre`` /
    ``ObservationValPre`` preprocessors and the evaluation entry can be reused
    unchanged::

        DeLiVERDataset(setting, split_name, preprocess=None, file_length=None)

    ``__getitem__`` returns ``{data, label, modal_x, depth_support, fn, n}`` with
    ``data``/``modal_x`` float32 ``(3,H,W)``, ``label`` int64 ``(H,W)``
    (``0..24`` plus ``255`` ignore) and ``depth_support`` float32 ``(1,H,W)``.
    """

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super().__init__(setting, split_name, preprocess, file_length)
        if list(self.x_modal) != ["d"]:
            raise NotImplementedError(
                "DeLiVERDataset only supports x_modal=['d'], got %r" % (self.x_modal,)
            )
        if setting.get("support_mask_root"):
            raise NotImplementedError(
                "the DeLiVER protocol does not use support_mask_root; use "
                "research/evaluate_odg.py --hole-ratio for explicit deletions"
            )
        self._rgb_root = Path(setting["rgb_root"])
        self._depth_root = Path(setting["x_root"])
        self._gt_root = Path(setting["gt_root"])

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        rgb_rel, depth_rel, label_rel = deliver_relative_paths(item_name)
        rgb_path = self._rgb_root / Path(str(rgb_rel))
        depth_path = self._depth_root / Path(str(depth_rel))
        label_path = self._gt_root / Path(str(label_rel))

        rgb = _read_rgb(rgb_path)
        depth_raw = _read_depth(depth_path)
        # The label id mapping is explicit; the author ``_gt_transform`` (``gt-1``)
        # must not run on top of it, hence ``transform_gt=False`` in the setting.
        gt = map_deliver_label(_read_label_ids(label_path))

        # Author depth representation: replicate the single channel to three so
        # the existing (3,H,W) tensor interface is unchanged.
        depth = cv2.merge([depth_raw, depth_raw, depth_raw])
        # The support domain is the finite image domain; padding added by the
        # preprocessor becomes 0.  Raw depth zeros are kept as observations.
        support = np.isfinite(depth_raw).astype(np.float32)

        if self.preprocess is not None:
            rgb, gt, depth, support = self.preprocess(rgb, gt, depth, support)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        depth = torch.from_numpy(np.ascontiguousarray(depth)).float()
        support = torch.from_numpy(np.ascontiguousarray(support)).float()

        return dict(
            data=rgb,
            label=gt,
            modal_x=depth,
            depth_support=support,
            fn=str(rgb_path),
            n=len(self._file_names),
        )


# The loader factories call ``data.ObservationDataset``; keep that name working
# without a dataset registry.
ObservationDataset = DeLiVERDataset


def deliver_setting(config, train_source, eval_source):
    """Build the ``setting`` dict for :class:`DeLiVERDataset` from a resolved config."""
    return {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": getattr(config, "rgb_format", ".png"),
        "gt_root": config.gt_root_folder,
        "gt_format": getattr(config, "gt_format", ".png"),
        "transform_gt": False,  # explicit id mapping above; never ``gt - 1``
        "x_root": config.x_root_folder,
        "x_format": getattr(config, "x_format", ".png"),
        "x_single_channel": True,
        "class_names": config.class_names,
        "train_source": str(train_source),
        "eval_source": str(eval_source),
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }


# --------------------------------------------------------------------------- #
# Optimizer group completion (the author ``group_weight`` misses nn.Parameters
# held directly by a module, e.g. every ``Geo.weight``)
# --------------------------------------------------------------------------- #
def complete_optimizer_groups(model, params_list, lr=0.0):
    """Add every trainable parameter the author grouping missed as a no-decay group.

    ``utils.init_func.group_weight`` walks ``module.modules()`` and tests
    ``isinstance(m, nn.Parameter)``; a module never yields its own ``nn.Parameter``
    children through ``modules()``, so parameters such as ``Geo.weight`` are never
    placed in a group and silently stay frozen.  This collects the leftovers by
    parameter object identity, deduplicates and appends them as a single no-decay
    group, then asserts that every ``requires_grad`` parameter is grouped exactly
    once.  Returns ``(params_list, added_names)``.
    """
    grouped_ids = set()
    for group in params_list:
        for param in group["params"]:
            grouped_ids.add(id(param))

    added_names, added_params, added_ids = [], [], set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in grouped_ids or id(param) in added_ids:
            continue
        added_names.append(name)
        added_params.append(param)
        added_ids.add(id(param))

    params_list = list(params_list)
    if added_params:
        params_list.append(dict(params=added_params, weight_decay=0.0, lr=lr))

    counts = collections.Counter()
    for group in params_list:
        for param in group["params"]:
            counts[id(param)] += 1
    duplicated, uncovered = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        seen = counts[id(param)]
        if seen == 0:
            uncovered.append(name)
        elif seen > 1:
            duplicated.append(name)
    if duplicated or uncovered:
        raise RuntimeError(
            "optimizer group coverage must be exactly once per trainable tensor: "
            "duplicated=%s uncovered=%s" % (duplicated[:20], uncovered[:20])
        )
    return params_list, added_names


# --------------------------------------------------------------------------- #
# Manifest preparation
# --------------------------------------------------------------------------- #
def enumerate_split_lines(root, split):
    """Enumerate ``<root>/img/*/<split>/*/*.png`` once and return sorted relative paths."""
    base = Path(root) / "img"
    if not base.is_dir():
        raise FileNotFoundError("missing DeLiVER img root: %s" % base)
    lines = [p.relative_to(base).as_posix() for p in base.glob("*/%s/*/*.png" % split)]
    return sorted(lines)


def _count_report(lines):
    by_weather = collections.Counter()
    by_fault = collections.Counter()
    for item in lines:
        weather, _split, fault = split_weather_fault(item)
        by_weather[weather] += 1
        by_fault[fault or "clean"] += 1
    return {
        "total": len(lines),
        "by_weather": {k: by_weather[k] for k in sorted(by_weather)},
        "by_case": {k: by_fault[k] for k in sorted(by_fault)},
    }


def _write_manifest(path, lines):
    """Write a manifest once.  Never silently overwrite a differing existing file."""
    path = Path(path)
    payload = "\n".join(lines) + ("\n" if lines else "")
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return "unchanged"
        raise SystemExit(
            "refusing to overwrite a differing manifest: %s\n"
            "  existing: %d lines, new: %d lines.  Inspect both, then remove the old file "
            "explicitly if the new content is wanted." % (path, len(path.read_text(encoding='utf-8').splitlines()), len(lines))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return "written"


def cmd_prepare(args):
    root = Path(args.root)
    split_dir = Path(args.split_dir)
    summary = {}
    for split in DELIVER_SPLITS:
        lines = enumerate_split_lines(root, split)
        report = _count_report(lines)
        status = _write_manifest(split_dir / ("%s.txt" % split), lines)
        if args.case:
            case_lines = [line for line in lines if args.case in line]
            if not case_lines:
                raise SystemExit("case %r matches no %s sample in %s" % (args.case, split, root))
            case_status = _write_manifest(split_dir / ("%s-%s.txt" % (split, args.case)), case_lines)
            report["case_filter"] = {"name": args.case, "file": "%s-%s.txt" % (split, args.case),
                                     "count": len(case_lines), "write": case_status}
            print("[prepare] %-5s case=%s -> %s (%d lines, %s)"
                  % (split, args.case, report["case_filter"]["file"], len(case_lines), case_status))
        report["manifest"] = str(split_dir / ("%s.txt" % split))
        report["write"] = status
        expected = DELIVER_EXPECTED_COUNTS.get(split)
        if expected is not None:
            report["expected_total"] = expected
            report["matches_reference"] = (expected == report["total"])
        summary[split] = report
        print("[prepare] %-5s total=%-5d by_weather=%s by_case=%s -> %s (%s)"
              % (split, report["total"], report["by_weather"], report["by_case"], report["manifest"], status))
        unexpected = sorted(set(report["by_weather"]) - set(DELIVER_WEATHERS))
        if unexpected:
            print("[prepare] WARNING: unexpected weather directories %s (expected %s)"
                  % (unexpected, list(DELIVER_WEATHERS)))
        unused_modality = sorted(set(report["by_case"]) & set(DELIVER_UNUSED_FAULTS))
        if unused_modality:
            print("[prepare] note: %s act on modalities this protocol does not read, so they are "
                  "not depth-degradation evidence" % unused_modality)
        if expected is not None and expected != report["total"]:
            print("[prepare] WARNING: %s has %d samples, handoff record says %d (report only, not fixed)"
                  % (split, report["total"], expected))
    print("[prepare] split dir: %s" % split_dir)
    print("[prepare] counts are printed and returned; no data is downloaded, moved or re-split")
    return summary


# --------------------------------------------------------------------------- #
# Preview / sample checks
# --------------------------------------------------------------------------- #
# One sample per (split, weather, case) preference slot; the first available line
# wins.  Covers train/val/test, several weathers and one native camera fault.
_PREVIEW_PREFERENCE = [
    ("train", "cloud", "clean"),
    ("train", "fog", "clean"),
    ("train", "night", "motionblur"),
    ("train", "rain", "underexposure"),
    ("val", "sun", "clean"),
    ("test", "cloud", "clean"),
]


def _middle_line(lines):
    """A cheap deterministic pick from a sorted manifest: the middle line."""
    if not lines:
        raise ValueError("cannot pick a sample from an empty manifest")
    return lines[len(lines) // 2]


def _pick_samples(sources, limit):
    """Pick up to ``limit`` manifest lines across splits/weathers/cases."""
    pools = {}
    for split, source in sources.items():
        if not Path(source).exists():
            print("[preview] WARNING: %s manifest missing: %s" % (split, source))
            continue
        pools[split] = Path(source).read_text(encoding="utf-8").splitlines()
    picks, used = [], set()
    for split, weather, case in _PREVIEW_PREFERENCE:
        if len(picks) >= limit:
            break
        for item in pools.get(split, []):
            if item in used:
                continue
            got_weather, got_split, fault = split_weather_fault(item)
            if got_weather != weather or (fault or "clean") != case:
                continue
            picks.append((split, item))
            used.add(item)
            break
        else:
            print("[preview] WARNING: no %s/%s/%s sample available" % (split, weather, case))
    return picks, pools


def _assert_mapping():
    """Small-array assertion of the official id mapping."""
    raw = np.array([1, 25, 0, 255], dtype=np.uint8)
    got = map_deliver_label(raw)
    expected = np.array([0, 24, 255, 255], dtype=np.int64)
    if not np.array_equal(got, expected):
        raise AssertionError("label mapping %s != %s" % (got.tolist(), expected.tolist()))
    try:
        map_deliver_label(np.array([26], dtype=np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown label id 26 was not rejected")
    return {"raw": raw.tolist(), "mapped": got.tolist(), "unknown_id_rejected": True}


def _render_preview(item_name, sample, gt, out_png, checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = sample["rgb"]
    depth_raw = sample["depth_raw"]
    palette = np.array(DELIVER_PALETTE, dtype=np.uint8)
    gt_show = np.where(gt == DELIVER_IGNORE, 0, gt)
    gt_color = palette[gt_show]
    ignore_mask = gt == DELIVER_IGNORE

    depth_norm = (depth_raw.astype(np.float64) / 255.0 - 0.48) / 0.28
    depth_show = np.where(depth_raw > 0, depth_norm, np.nan)

    overlay = (0.55 * rgb.astype(np.float64) + 0.45 * gt_color.astype(np.float64)).astype(np.uint8)
    overlay[ignore_mask] = rgb[ignore_mask]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (%s)" % Path(sample["paths"]["rgb"]).name)
    im = axes[1].imshow(depth_show, cmap="viridis")
    axes[1].set_title("depth/ raw -> (D8/255-0.48)/0.28 (0 shown empty)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(gt_color)
    axes[2].set_title("GT (official palette, ignore=255 black)")
    axes[3].imshow(overlay)
    axes[3].set_title("RGB + GT overlay")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("%s | %s | %s" % (item_name, Path(sample["paths"]["rgb"]).name,
                                   "depth %s..%s" % (checks["depth_min"], checks["depth_max"])), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return str(out_png)


def cmd_preview(args):
    config = _load_config(args.config)
    if bool(getattr(config, "pad", False)):
        raise SystemExit("DeLiVER preview expects pad=False; got pad=True in %s" % args.config)
    roots = (config.rgb_root_folder, config.x_root_folder, config.gt_root_folder)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "train": str(config.train_dev_source),
        "val": str(config.dev_eval_source),
        "test": str(config.full_test_source),
    }
    picks, pools = _pick_samples(sources, int(args.limit))
    if not picks:
        raise SystemExit("no preview sample could be selected from %s" % sources)

    report = {
        "config": args.config,
        "dataset_path": str(config.dataset_path),
        "rgb_root": str(config.rgb_root_folder),
        "rgb_order": "RGB",
        "depth_source": "depth/ native single-channel uint8, normalised (D8/255-0.48)/0.28",
        "label_source": "semantic/ red channel ids, official mapping, ignore=255",
        "protocol_note": "raw-depth protocol, NOT the official HHA RGB-D protocol",
        "mapping_assertion": _assert_mapping(),
        "sources": sources,
        "manifest_totals": {k: len(v) for k, v in pools.items()},
        "samples": [],
        "figures": [],
    }
    print("[preview] mapping assertion ok: %s" % report["mapping_assertion"])

    for idx, (split, item_name) in enumerate(picks):
        sample = read_modalities(*roots, item_name)
        rgb, depth_raw, label_raw = sample["rgb"], sample["depth_raw"], sample["label_raw"]
        weather, _split, fault = split_weather_fault(item_name)

        # Format invariants of the approved protocol.
        assert rgb.ndim == 3 and rgb.shape[2] == 3, rgb.shape
        assert depth_raw.ndim == 2 and depth_raw.dtype == np.uint8, (depth_raw.shape, depth_raw.dtype)
        assert label_raw.ndim == 2, label_raw.shape
        assert label_raw.shape == depth_raw.shape == rgb.shape[:2], (label_raw.shape, depth_raw.shape, rgb.shape)
        mapped = map_deliver_label(label_raw)
        raw_unique = np.unique(label_raw).tolist()
        assert set(raw_unique) <= set(range(1, DELIVER_NUM_CLASSES + 1)) | {0, DELIVER_IGNORE}, raw_unique
        assert set(np.unique(mapped).tolist()) <= set(range(DELIVER_NUM_CLASSES)) | {DELIVER_IGNORE}

        checks = {
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": str(rgb.dtype),
            "depth_shape": list(depth_raw.shape),
            "depth_dtype": str(depth_raw.dtype),
            "depth_min": int(depth_raw.min()),
            "depth_max": int(depth_raw.max()),
            "label_shape": list(label_raw.shape),
            "label_raw_unique": raw_unique,
            "label_mapped_unique": np.unique(mapped).tolist(),
            "ignore_fraction": round(float((mapped == DELIVER_IGNORE).mean()), 6),
            "weather": weather,
            "case": fault or "clean",
            "split": split,
            "paths": sample["paths"],
        }
        if idx < int(args.figures):
            out_png = out_dir / ("preview-%s-%s-%s.png" % (split, weather, Path(item_name).stem))
            checks["figure"] = _render_preview(item_name, sample, mapped, out_png, checks)
            report["figures"].append(checks["figure"])
        report["samples"].append({"item": item_name, **checks})
        print("[preview] split=%-5s weather=%-5s case=%-12s rgb=%s depth=%s(%s %d..%d) label_raw=%s ignore=%.3f"
              % (split, weather, fault or "clean", checks["rgb_shape"], checks["depth_shape"],
                 checks["depth_dtype"], checks["depth_min"], checks["depth_max"], raw_unique,
                 checks["ignore_fraction"]))

    # Preprocessing invariants through the real preprocessors.
    mean, std = config.norm_mean, config.norm_std
    train_setting = deliver_setting(config, sources["train"], sources["val"])
    train_ds = DeLiVERDataset(train_setting, "train",
                              ObservationTrainPre(mean, std, True, config), None)
    train_item = train_ds[0]
    assert tuple(train_item["data"].shape) == (3, int(config.image_height), int(config.image_width))
    assert tuple(train_item["modal_x"].shape) == (3, int(config.image_height), int(config.image_width))
    assert tuple(train_item["depth_support"].shape) == (1, int(config.image_height), int(config.image_width))
    assert tuple(train_item["label"].shape) == (int(config.image_height), int(config.image_width))
    assert train_item["label"].dtype == torch.int64 and train_item["data"].dtype == torch.float32
    support_unique = sorted(np.unique(train_item["depth_support"].numpy()).tolist())
    assert set(support_unique) <= {0.0, 1.0}, support_unique
    assert bool((((train_item["label"] >= 0) & (train_item["label"] < DELIVER_NUM_CLASSES))
                 | (train_item["label"] == DELIVER_IGNORE)).all())

    val_setting = deliver_setting(config, sources["train"], sources["val"])
    val_ds = DeLiVERDataset(val_setting, "val", ObservationValPre(mean, std, True, config), None)
    val_item = val_ds[0]
    native_h, native_w = int(val_item["label"].shape[0]), int(val_item["label"].shape[1])
    assert tuple(val_item["modal_x"].shape) == (3, native_h, native_w)
    assert tuple(val_item["depth_support"].shape) == (1, native_h, native_w)
    assert float(val_item["depth_support"].min()) == 1.0 and float(val_item["depth_support"].max()) == 1.0, \
        "pad=False val support must be all-observed"

    report["preprocess_checks"] = {
        "train_crop": list(train_item["data"].shape),
        "train_label": list(train_item["label"].shape),
        "train_support_values": support_unique,
        "train_support_mean": round(float(train_item["depth_support"].mean()), 4),
        "val_native_size": [native_h, native_w],
        "val_support_all_observed": True,
        "pad": bool(getattr(config, "pad", False)),
    }
    print("[preview] train crop=%s label=%s support=%s | val native=%dx%d support all-observed (pad=False)"
          % (list(train_item["data"].shape), list(train_item["label"].shape), support_unique,
             native_h, native_w))

    report_path = out_dir / "preview-summary.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print("[preview] wrote %s (%d samples, %d figures)" % (report_path, len(report["samples"]), len(report["figures"])))
    return report


# --------------------------------------------------------------------------- #
# Model smoke check (CPU)
# --------------------------------------------------------------------------- #
def cmd_smoke(args):
    import torch.nn as nn
    from utils.init_func import group_weight
    from research import odg_schedule as sched

    config = _load_config(args.config)
    device = torch.device(args.device)
    size = int(args.size)
    torch.manual_seed(int(getattr(config, "seed", 12345)))

    roots = (config.rgb_root_folder, config.x_root_folder, config.gt_root_folder)
    train_lines = Path(config.train_dev_source).read_text(encoding="utf-8").splitlines()
    item_name = _middle_line(train_lines)
    sample = read_modalities(*roots, item_name)
    rgb_raw, depth_raw = sample["rgb"], sample["depth_raw"]
    label = map_deliver_label(sample["label_raw"])

    rgb_r = cv2.resize(rgb_raw, (size, size), interpolation=cv2.INTER_LINEAR)
    depth_r = cv2.resize(depth_raw, (size, size), interpolation=cv2.INTER_LINEAR)
    label_r = torch.nn.functional.interpolate(
        torch.from_numpy(label)[None, None].float(), size=(size, size), mode="nearest"
    )[0, 0].long()

    rgb = (rgb_r.astype(np.float64) / 255.0 - np.asarray(config.norm_mean)) / np.asarray(config.norm_std)
    depth = np.stack([depth_r] * 3, axis=-1).astype(np.float64) / 255.0
    depth = (depth - 0.48) / 0.28

    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)[None]).float().to(device)
    modal_t = torch.from_numpy(depth.transpose(2, 0, 1)[None]).float().to(device)
    label_t = label_r[None].to(device)
    support_t = torch.ones(1, 1, size, size, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    from models.builder import EncoderDecoder as segmodel
    model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
    model.to(device)
    model.train()
    use_support = sched.supports_depth_support(model)
    if not use_support:
        raise SystemExit("model does not accept depth_support; DeLiVER protocol requires it")

    with torch.no_grad():
        logits = model(rgb_t, modal_t, depth_support=support_t)
    if tuple(logits.shape) != (1, DELIVER_NUM_CLASSES, size, size):
        raise AssertionError("unexpected logits shape %s" % (tuple(logits.shape),))

    params_list = group_weight([], model, nn.BatchNorm2d, float(config.lr))
    groups_before = len(params_list)
    params_before = sum(len(group["params"]) for group in params_list)
    params_list, added = complete_optimizer_groups(model, params_list, float(config.lr))
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    geo_names = [name for name in trainable if "Geo.weight" in name]
    if not geo_names or not set(geo_names) <= set(added):
        raise AssertionError("Geo.weight tensors were not all added by the completion helper: %s" % geo_names)
    optimizer = torch.optim.AdamW(params_list, lr=float(config.lr), betas=(0.9, 0.999),
                                  weight_decay=float(config.weight_decay))
    before = {name: param.detach().clone() for name, param in model.named_parameters()
              if name in set(added)}

    optimizer.zero_grad(set_to_none=True)
    loss = sched.forward_loss(model, rgb_t, modal_t, label_t, support_t, use_support)
    loss_value = float(loss.detach())
    if not bool(torch.isfinite(loss).all()):
        raise AssertionError("non-finite loss: %r" % loss_value)
    loss.backward()
    grad_report = sched.grad_finiteness_report(model)
    if not grad_report["all_finite"]:
        raise AssertionError("non-finite gradients: %s" % grad_report["nonfinite"][:5])
    optimizer.step()

    deltas, changed = sched.parameter_delta_report(
        {name: param for name, param in model.named_parameters() if name in before}, before
    )
    # ``parameter_delta_report`` compares the current parameters against the
    # pre-step copies, so a nonzero delta proves the previously ungrouped tensors
    # really receive optimizer updates.
    if changed != len(before):
        missed = sorted(name for name, row in deltas.items() if not row["changed"])
        raise AssertionError("previously ungrouped tensors did not update: %s" % missed[:10])

    # SegMetrics check: 25x25 confusion matrix and ignore pixels not counted.
    metrics = sched.SegMetrics(DELIVER_NUM_CLASSES, int(config.background), torch.device("cpu"))
    labels = torch.tensor([[[0, 1], [DELIVER_IGNORE, 2]]], dtype=torch.long)
    probs = torch.zeros(1, DELIVER_NUM_CLASSES, 2, 2)
    for row in range(2):
        for col in range(2):
            cls = 0 if labels[0, row, col] == DELIVER_IGNORE else int(labels[0, row, col])
            probs[0, cls, row, col] = 1.0
    metrics.update(probs, labels)
    if tuple(metrics.hist.shape) != (DELIVER_NUM_CLASSES, DELIVER_NUM_CLASSES):
        raise AssertionError("confusion matrix shape %s" % (tuple(metrics.hist.shape),))
    if float(metrics.hist.sum()) != 3.0:
        raise AssertionError("ignore pixel was counted: hist sum=%s" % float(metrics.hist.sum()))
    metric_result = metrics.compute()
    expected_miou = round(3 / DELIVER_NUM_CLASSES * 100, 2)
    if metric_result["miou"] != expected_miou:
        raise AssertionError("miou %s != %s" % (metric_result["miou"], expected_miou))

    report = {
        "config": args.config,
        "device": str(device),
        "protocol_note": "raw-depth protocol, NOT the official HHA RGB-D protocol",
        "sample": item_name,
        "input_size": size,
        "logits_shape": list(logits.shape),
        "num_classes": DELIVER_NUM_CLASSES,
        "loss": loss_value,
        "loss_finite": True,
        "grads_all_finite": bool(grad_report["all_finite"]),
        "grad_num_trainable_tensors": grad_report["num_trainable"],
        "grad_num_with_grad": grad_report["num_with_grad"],
        "grad_no_grad_count": len(grad_report["no_grad"]),
        "grad_no_grad_examples": grad_report["no_grad"][:10],
        "optimizer_groups_before_completion": groups_before,
        "optimizer_params_before_completion": params_before,
        "optimizer_groups_after_completion": len(params_list),
        "optimizer_params_after_completion": sum(len(g["params"]) for g in params_list),
        "ungrouped_added_count": len(added),
        "ungrouped_added_names": added,
        "geo_weight_count": len(geo_names),
        "updated_after_one_step": int(changed),
        "updated_names_expected": len(before),
        "metrics_check": {
            "hist_shape": list(metrics.hist.shape),
            "hist_sum": float(metrics.hist.sum()),
            "miou": metric_result["miou"],
            "expected_miou": expected_miou,
            "ignore_pixels_excluded": True,
        },
        "resumable_checkpoint_written": False,
        "note": "CPU smoke only: no GPU memory/throughput claim and no training checkpoint",
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "smoke-report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print("[smoke] logits=%s loss=%.6f finite grads=%s trainable=%d with_grad=%d"
          % (report["logits_shape"], loss_value, report["grads_all_finite"],
             report["grad_num_trainable_tensors"], report["grad_num_with_grad"]))
    print("[smoke] optimizer groups %d(%d params) -> %d(%d params); added %d ungrouped tensors, %d of them Geo.weight"
          % (groups_before, params_before, report["optimizer_groups_after_completion"],
             report["optimizer_params_after_completion"], len(added), len(geo_names)))
    print("[smoke] after one AdamW step %d/%d previously-ungrouped tensors changed; metrics miou=%s (expected %s)"
          % (changed, len(before), metric_result["miou"], expected_miou))
    print("[smoke] wrote %s (no resumable checkpoint; this is not a training artifact)" % report_path)
    return report


# --------------------------------------------------------------------------- #
# Helpers / CLI
# --------------------------------------------------------------------------- #
def _load_config(module_name):
    return deepcopy(getattr(import_module(module_name), "C"))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def build_parser():
    parser = argparse.ArgumentParser(description="DeLiVER RGB-D dataset layer (RGB + raw depth/ + 25 classes)")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="enumerate img/*/<split>/*/*.png once and write the split manifests")
    prepare.add_argument("--root", required=True, help="DeLiVER dataset root (contains img/, depth/, semantic/)")
    prepare.add_argument("--split-dir", required=True, help="manifest output directory")
    prepare.add_argument("--case", default=None,
                         help="optional substring case filter; writes <split>-<case>.txt and fails on an empty subset")
    prepare.set_defaults(func=cmd_prepare)

    preview = sub.add_parser("preview", help="read a few real samples, assert the format and render figures")
    preview.add_argument("--config", required=True, help="config module, e.g. local_configs.research.DFormerv2_S_DeLiVER")
    preview.add_argument("--limit", type=int, default=6, help="maximum number of samples to read")
    preview.add_argument("--figures", type=int, default=3, help="number of preview figures to render")
    preview.add_argument("--out", default="outputs/deliver-preparation")
    preview.set_defaults(func=cmd_preview)

    smoke = sub.add_parser("smoke", help="one CPU forward/backward/AdamW step on a real sample")
    smoke.add_argument("--config", required=True, help="config module, e.g. local_configs.research.DFormerv2_S_DeLiVER")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--size", type=int, default=128)
    smoke.add_argument("--out", default="outputs/deliver-preparation")
    smoke.set_defaults(func=cmd_smoke)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
