"""Observation-based data layer for the ODG study.

This module is a *thin* layer on top of the author's segmentation pipeline
(``utils/dataloader/RGBXDataset.py`` and ``utils/transforms.py``).  It never
modifies the author loader, model or training entry point.  It only adds:

* ``ObservationDataset`` -- the same path resolution and file reading as
  ``RGBXDataset``, but it additionally builds a per-pixel depth *support* mask
  and forwards four synchronised arrays ``(rgb, gt, depth, support)`` to the
  preprocessor.
* ``ObservationTrainPre`` / ``ObservationValPre`` -- author-compatible
  normalisation and augmentation (scale, flip, crop, pad) applied simultaneously
  to ``rgb``, ``gt``, ``depth`` and ``support``.  Depth keeps the author read,
  normalisation and padding; the support mask is resized with nearest-neighbour,
  padded with ``0`` and is never dilated by a bilinear threshold.
* ``soft_depth_histogram`` / ``depth_bin_centers`` / ``depth_bin_edges`` -- 16
  fixed bins expressed in the author normalisation domain, used by the preview
  CLI and available for analysis.  The ODG model must build its per-stage
  distributions from observations *before* stage compression, not by
  histogramming an already single-valued stage depth map.
* ``sunrgbd_setting`` -- a convenience builder for the ``setting`` dict consumed
  by ``ObservationDataset`` / ``RGBXDataset``.
* A small CLI (``--prepare-split``, ``--check-pairs``, ``--preview``,
  ``--check-source``) for the long-lived data-preparation workflow.

Representation protocol (agreed with the caller)
------------------------------------------------
* Local ``Depth`` PNGs are native ``uint16`` but the author loader reads them
  with ``cv2.IMREAD_GRAYSCALE`` (the high 8 bits) and normalises with
  ``(D8/255 - 0.48) / 0.28``.  This layer keeps that read and normalisation.
* A *natural* raw zero is **not** treated as a physically missing measurement.
  The support domain is ``image domain ∩ finite values ∩ (optional) artificial
  mask``.  It describes the representation protocol, not sensor confidence, so
  it does not invent physical missing points from grey zeros.  Locations with
  empty support must lead to a zero depth bias, which the model handles.
* The artificial mask (when used) is supplied by an upper layer through the
  setting key ``support_mask_root``: a directory holding same-named validity
  images where ``0`` removes a point from the support and a non-zero value keeps
  it.  No mask directory is configured by default.

Run the CLI from the repository root, e.g.::

    python research/data.py --prepare-split --data-root D:/0Project/dataset
    python research/data.py --check-pairs   --data-root D:/0Project/dataset
    python research/data.py --preview       --data-root D:/0Project/dataset
    python research/data.py --check-source  --data-root D:/0Project/dataset
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Make the repository root importable when the module is executed directly
# (``python research/data.py``), while keeping the author-relative imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.dataloader.RGBXDataset import RGBXDataset, get_path  # noqa: E402
from utils.transforms import (  # noqa: E402
    generate_random_crop_pos,
    normalize,
    random_crop_pad_to_shape,
)

__all__ = [
    "ObservationDataset",
    "ObservationTrainPre",
    "ObservationValPre",
    "depth_bin_edges",
    "depth_bin_centers",
    "soft_depth_histogram",
    "sunrgbd_setting",
    "DEFAULT_NUM_BINS",
    "DEPTH_NORM_MIN",
    "DEPTH_NORM_MAX",
    "DEPTH_RAW_ZERO_NORM",
]

# --------------------------------------------------------------------------- #
# Depth representation constants (author SUN protocol)
# --------------------------------------------------------------------------- #
# The author reads depth as 8-bit grey in [0, 255] and normalises with
# ``(D8 / 255 - 0.48) / 0.28`` (see ``utils/transforms.normalize`` and
# ``dataloader.TrainPre/ValPre`` with ``x_is_single_channel=True``).
DEFAULT_NUM_BINS = 16
DEPTH_NORM_MEAN = 0.48
DEPTH_NORM_STD = 0.28
DEPTH_NORM_MIN = (0.0 / 255.0 - DEPTH_NORM_MEAN) / DEPTH_NORM_STD  # -1.714286
DEPTH_NORM_MAX = (255.0 / 255.0 - DEPTH_NORM_MEAN) / DEPTH_NORM_STD  # 1.857143
# Normalised value of a *raw* zero (author ValPre pads the raw depth with 0
# before normalisation, so valid raw zeros also land here).
DEPTH_RAW_ZERO_NORM = (0.0 - DEPTH_NORM_MEAN) / DEPTH_NORM_STD  # -1.714286

SUNRGBD_CLASS_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor_mat", "clothes",
    "ceiling", "books", "fridge", "tv", "paper", "towel", "shower_curtain",
    "box", "whiteboard", "person", "night_stand", "toilet", "sink", "lamp",
    "bathtub", "bag",
]


# --------------------------------------------------------------------------- #
# Bin helpers (preview / analysis).  Bin centres live in the normalised domain.
# --------------------------------------------------------------------------- #
def depth_bin_edges(num_bins=DEFAULT_NUM_BINS):
    """Evenly spaced bin edges covering the author normalised depth range."""
    return np.linspace(DEPTH_NORM_MIN, DEPTH_NORM_MAX, num_bins + 1, dtype=np.float64)


def depth_bin_centers(num_bins=DEFAULT_NUM_BINS):
    """Bin centres in the author normalised depth domain."""
    edges = depth_bin_edges(num_bins)
    return 0.5 * (edges[:-1] + edges[1:])


def soft_depth_histogram(depth, support=None, num_bins=DEFAULT_NUM_BINS):
    """Soft 16-bin depth histogram over the support domain.

    Each value is assigned linearly to its two nearest bin centres; values
    outside the outermost centres fall entirely into the closest edge bin.

    Parameters
    ----------
    depth : array-like
        Normalised depth, ``(H, W)`` or ``(C, H, W)`` (first channel used).
    support : array-like, optional
        ``0/1`` support mask aligned with ``depth``.  Non-finite depth is always
        excluded.
    num_bins : int
        Number of fixed bins.

    Returns
    -------
    centers : ``(num_bins,)`` ndarray
    prob : ``(num_bins,)`` ndarray
        Occupancy normalised to sum to 1 (all zeros when the support is empty).
    valid : int
        Number of support pixels that contributed.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[0]
    finite = np.isfinite(d)
    if support is None:
        mask = finite
    else:
        s = np.asarray(support)
        if s.ndim == 3:
            s = s[0]
        mask = (s > 0) & finite

    centers = depth_bin_centers(num_bins)
    prob = np.zeros(num_bins, dtype=np.float64)
    values = d[mask]
    valid = int(values.size)
    if valid:
        step = centers[1] - centers[0]
        idx = np.clip((values - centers[0]) / step, 0.0, num_bins - 1)
        lo = np.floor(idx).astype(np.int64)
        hi = np.minimum(lo + 1, num_bins - 1)
        frac = idx - lo
        np.add.at(prob, lo, 1.0 - frac)
        np.add.at(prob, hi, frac)
        total = prob.sum()
        if total > 0:
            prob /= total
    return centers, prob, valid


# --------------------------------------------------------------------------- #
# Setting builder
# --------------------------------------------------------------------------- #
def sunrgbd_setting(data_root):
    """Build the ``setting`` dict for ``ObservationDataset`` on SUN RGB-D."""
    root = Path(data_root) / "SUNRGBD"
    return {
        "rgb_root": str(root / "RGB"),
        "rgb_format": ".jpg",
        "gt_root": str(root / "labels"),
        "gt_format": ".png",
        "transform_gt": True,
        "x_root": str(root / "Depth"),
        "x_format": ".png",
        "x_single_channel": True,
        "class_names": SUNRGBD_CLASS_NAMES,
        "train_source": str(root / "train.txt"),
        "eval_source": str(root / "test.txt"),
        "dataset_name": "SUNRGBD",
        "backbone": "DFormerv2_S",
    }


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ObservationDataset(RGBXDataset):
    """``RGBXDataset`` that also returns a depth support mask.

    Interface (kept stable for the training task)::

        ObservationDataset(setting, split_name, preprocess=None, file_length=None)

    ``__getitem__`` returns the author-compatible dict plus ``depth_support``::

        {
            "data":         float32 (3, H, W),  # author-normalised RGB
            "label":        int64   (H, W),     # author gt (ignore=255)
            "modal_x":      float32 (3, H, W),  # author-normalised depth (3ch)
            "depth_support": float32 (1, H, W), # 1 = observation, 0 = padding/invalid
            "fn":           str,                # rgb path (author convention)
            "n":            int,                # dataset length
        }
    """

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super().__init__(setting, split_name, preprocess, file_length)
        # Optional artificial-removal mask directory (validity convention: 0 removes).
        self._support_mask_root = setting.get("support_mask_root", None)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        path_dict = get_path(
            self.dataset_name,
            self._rgb_path,
            self._rgb_format,
            self._x_path,
            self._x_format,
            self._gt_path,
            self._gt_format,
            self.x_modal,
            item_name,
        )

        # This observation layer is written for a single scalar depth modality.
        if list(self.x_modal) != ["d"]:
            raise NotImplementedError(
                "ObservationDataset only supports x_modal=['d'], got %r" % (self.x_modal,)
            )

        if self.dataset_name == "SUNRGBD" and self.backbone.startswith("DFormerv2"):
            rgb_mode = "RGB"
        else:
            rgb_mode = "BGR"
        rgb = self._open_image(path_dict["rgb_path"], rgb_mode)

        gt = self._open_image(path_dict["gt_path"], cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt)

        # Author depth read: 8-bit grey (high byte of the native uint16 PNG),
        # then replicated to three channels.  Kept identical to RGBXDataset.
        depth = self._open_image(path_dict["d_path"], cv2.IMREAD_GRAYSCALE)
        depth = cv2.merge([depth, depth, depth])

        support = self._load_support(path_dict["d_path"])
        if self._support_mask_root:
            # An explicit deletion changes the common input as well as its support.
            depth = np.where(support[..., None] > 0, depth, 0).astype(depth.dtype)

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
            fn=str(path_dict["rgb_path"]),
            n=len(self._file_names),
        )

    def _load_support(self, depth_path):
        """Build the depth support mask at native resolution.

        Support = finite values in the image domain, optionally intersected with
        an externally supplied artificial mask.  Natural raw zeros are *kept*
        (they are not treated as physical missing points).

        The depth file is read once here with ``IMREAD_UNCHANGED`` (a second,
        small read next to the author's 8-bit read in ``__getitem__``) so that
        non-finite native values could be detected; for the ``uint16`` SUN PNGs
        finite is always true.
        """
        raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError("cannot read depth for support: %s" % depth_path)
        if raw.ndim == 3:
            raw = raw[..., 0]
        support = np.isfinite(raw).astype(np.float32)

        if self._support_mask_root:
            mask_path = Path(self._support_mask_root) / Path(depth_path).name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError("cannot read support mask: %s" % mask_path)
            if mask.shape != support.shape:
                raise ValueError(
                    "support mask %s shape %s != depth %s"
                    % (mask_path, mask.shape, support.shape)
                )
            support = support * (mask > 0).astype(np.float32)

        return support


# --------------------------------------------------------------------------- #
# Preprocessing (author-compatible, synchronised over rgb/gt/depth/support)
# --------------------------------------------------------------------------- #
def _random_mirror4(rgb, gt, depth, support):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        depth = cv2.flip(depth, 1)
        support = cv2.flip(support, 1)
    return rgb, gt, depth, support


def _random_scale4(rgb, gt, depth, support, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_LINEAR)
    # Support uses nearest-neighbour only: never a bilinear threshold.
    support = cv2.resize(support, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return rgb, gt, depth, support, scale


class ObservationTrainPre(object):
    """Author ``TrainPre`` extended to four synchronised arrays.

    Signature matches the author preprocess::

        ObservationTrainPre(norm_mean, norm_std, sign=False, config=None)

    Depth follows the author rule: when ``sign`` (i.e. ``x_is_single_channel``)
    is true it is normalised with ``[0.48]*3`` / ``[0.28]*3``; otherwise it uses
    ``norm_mean`` / ``norm_std``.  Crop/pad add ``0`` for rgb, depth and support
    (after normalisation), and ``255`` for gt -- as in the author ``TrainPre``.
    """

    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, depth, support):
        rgb, gt, depth, support = _random_mirror4(rgb, gt, depth, support)
        if self.config.train_scale_array is not None:
            rgb, gt, depth, support, _ = _random_scale4(
                rgb, gt, depth, support, self.config.train_scale_array
            )

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        if self.sign:
            depth = normalize(depth, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])
        else:
            depth = normalize(depth, self.norm_mean, self.norm_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_depth, _ = random_crop_pad_to_shape(depth, crop_pos, crop_size, 0)
        p_support, _ = random_crop_pad_to_shape(support, crop_pos, crop_size, 0)

        # Defensive: support must stay 0 wherever the transformed depth is not
        # finite.  This cannot re-introduce support in padded regions.
        p_support = p_support * np.isfinite(p_depth[..., 0]).astype(np.float32)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_depth = p_depth.transpose(2, 0, 1)
        p_support = p_support[None, ...]

        return p_rgb, p_gt, p_depth, p_support


class ObservationValPre(object):
    """Author ``ValPre`` extended to four synchronised arrays.

    Signature matches the author preprocess::

        ObservationValPre(norm_mean, norm_std, sign=False, config=None)

    When ``config.pad`` is true the raw arrays are padded to 531x730 *before*
    normalisation (author convention), so the depth padding value is a raw ``0``
    (normalised to ``-0.48/0.28``).  The support is padded with ``0``.
    """

    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, depth, support):
        if self.config.pad:
            rgb = cv2.copyMakeBorder(
                rgb, 0, 531 - rgb.shape[0], 0, 730 - rgb.shape[1],
                cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0),
            )
            gt = cv2.copyMakeBorder(
                gt, 0, 531 - gt.shape[0], 0, 730 - gt.shape[1],
                cv2.BORDER_CONSTANT, value=(255,),
            )
            depth = cv2.copyMakeBorder(
                depth, 0, 531 - depth.shape[0], 0, 730 - depth.shape[1],
                cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0),
            )
            support = cv2.copyMakeBorder(
                support, 0, 531 - support.shape[0], 0, 730 - support.shape[1],
                cv2.BORDER_CONSTANT, value=0.0,
            )

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        depth = normalize(depth, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])

        support = support * np.isfinite(depth[..., 0]).astype(np.float32)

        return rgb.transpose(2, 0, 1), gt, depth.transpose(2, 0, 1), support[None, ...]


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def _read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def _write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _default_splits_dir(seed):
    return Path(__file__).resolve().parent / "splits" / ("sunrgbd_seed%d" % seed)


def _split_files(splits_dir):
    return {
        "train": Path(splits_dir) / "train-dev.txt",
        "val": Path(splits_dir) / "dev.txt",
    }


def _dev_split(train_lines, seed, dev_size):
    """Deterministic image-level split of the official train list.

    No scene grouping is used: the SUN RGB-D package in this repository does not
    ship a scene/sequence grouping file, so the split is a fixed image-level
    partition.  Indices are shuffled with ``random.Random(seed)`` (Mersenne
    Twister, stable across Python versions); the selected indices are then
    re-sorted so both output files keep the original ``train.txt`` line order.
    """
    n = len(train_lines)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    dev_idx = sorted(indices[:dev_size])
    train_idx = sorted(indices[dev_size:])
    return [train_lines[i] for i in train_idx], [train_lines[i] for i in dev_idx]


def _config_for_preprocess(pad=True, image_height=480, image_width=480):
    class _Cfg(object):
        pass

    cfg = _Cfg()
    cfg.pad = pad
    cfg.image_height = image_height
    cfg.image_width = image_width
    cfg.train_scale_array = None
    return cfg


def cmd_prepare_split(args):
    data_root = Path(args.data_root)
    sun = data_root / "SUNRGBD"
    official_train = sun / "train.txt"
    official_test = sun / "test.txt"

    train_lines = _read_lines(official_train)
    test_lines = _read_lines(official_test) if official_test.exists() else []
    n = len(train_lines)
    print("[prepare-split] official train.txt: %d lines" % n)
    print("[prepare-split] official test.txt : %d lines (unchanged)" % len(test_lines))
    if n != 5285:
        print("[prepare-split] WARNING: expected 5285 official train lines, got %d" % n)
    if args.dev_size > n:
        raise SystemExit("dev_size %d > train size %d" % (args.dev_size, n))

    train_dev, dev = _dev_split(train_lines, args.seed, args.dev_size)
    splits_dir = Path(args.splits_dir) if args.splits_dir else _default_splits_dir(args.seed)
    targets = _split_files(splits_dir)
    _write_lines(targets["train"], train_dev)
    _write_lines(targets["val"], dev)

    print("[prepare-split] seed=%d dev_size=%d" % (args.seed, args.dev_size))
    print("[prepare-split] train-dev.txt: %d lines -> %s" % (len(train_dev), targets["train"]))
    print("[prepare-split] dev.txt      : %d lines -> %s" % (len(dev), targets["val"]))

    readme = Path(splits_dir) / "README.md"
    if not readme.exists():
        readme.write_text(
            "# SUN RGB-D development split (seed %d)\n\n"
            "Derived from the official `SUNRGBD/train.txt` (%d lines).  The\n"
            "official `train.txt` and `test.txt` are left unchanged.\n\n"
            "- `dev.txt`      : %d lines (fixed development/validation set)\n"
            "- `train-dev.txt`: %d lines (development training set)\n\n"
            "Algorithm: image-level partition, no scene grouping (the local SUN\n"
            "RGB-D package ships no scene/sequence grouping file).  Indices are\n"
            "shuffled with `random.Random(%d)` in Python's Mersenne Twister;\n"
            "the selected indices are re-sorted so both files keep the original\n"
            "`train.txt` line order.  Lines keep the author two-column format\n"
            "`RGB/<name>.jpg labels/<name>.png`.  `dev.txt` and `train-dev.txt`\n"
            "are disjoint and their union is the official train list.\n"
            % (args.seed, n, len(dev), len(train_dev), args.seed),
            encoding="utf-8",
        )
        print("[prepare-split] wrote %s" % readme)


def _pair_check(label, list_path, setting, sample=0):
    if not Path(list_path).exists():
        print("[check-pairs] %-10s SKIP (missing %s)" % (label, list_path))
        return None
    lines = _read_lines(list_path)
    missing_rgb = missing_depth = missing_gt = 0
    first_missing = []
    for item in lines:
        paths = get_path(
            setting["dataset_name"],
            setting["rgb_root"], setting["rgb_format"],
            setting["x_root"], setting["x_format"],
            setting["gt_root"], setting["gt_format"],
            ["d"], item,
        )
        ok = True
        if not os.path.exists(paths["rgb_path"]):
            missing_rgb += 1
            ok = False
        if not os.path.exists(paths["d_path"]):
            missing_depth += 1
            ok = False
        if not os.path.exists(paths["gt_path"]):
            missing_gt += 1
            ok = False
        if not ok and len(first_missing) < 5:
            first_missing.append(item)
    print(
        "[check-pairs] %-10s lines=%d  missing rgb=%d depth=%d gt=%d"
        % (label, len(lines), missing_rgb, missing_depth, missing_gt)
    )
    for item in first_missing:
        print("[check-pairs]   missing example: %s" % item)

    # Read a few real files to report shapes/dtypes without touching all data.
    for item in lines[:sample]:
        paths = get_path(
            setting["dataset_name"],
            setting["rgb_root"], setting["rgb_format"],
            setting["x_root"], setting["x_format"],
            setting["gt_root"], setting["gt_format"],
            ["d"], item,
        )
        rgb = cv2.imread(paths["rgb_path"], cv2.IMREAD_UNCHANGED)
        raw = cv2.imread(paths["d_path"], cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(paths["gt_path"], cv2.IMREAD_GRAYSCALE)
        print(
            "[check-pairs]   sample %s | rgb %s %s | depth %s %s | gt %s %s"
            % (
                Path(paths["rgb_path"]).name,
                rgb.shape, rgb.dtype,
                raw.shape, raw.dtype,
                gt.shape, gt.dtype,
            )
        )
    return lines


def cmd_check_pairs(args):
    data_root = Path(args.data_root)
    sun = data_root / "SUNRGBD"
    setting = sunrgbd_setting(data_root)

    official_train = _pair_check("train.txt", sun / "train.txt", setting)
    official_test = _pair_check("test.txt", sun / "test.txt", setting)

    splits_dir = Path(args.splits_dir) if args.splits_dir else _default_splits_dir(args.seed)
    targets = _split_files(splits_dir)
    train_dev = _pair_check("train-dev.txt", targets["train"], setting)
    dev = _pair_check("dev.txt", targets["val"], setting)

    if official_train is not None and train_dev is not None and dev is not None:
        official_set = set(official_train)
        train_dev_set = set(train_dev)
        dev_set = set(dev)
        print(
            "[check-pairs] partition: |train-dev|=%d |dev|=%d overlap=%d "
            "union==official_train: %s"
            % (
                len(train_dev_set),
                len(dev_set),
                len(train_dev_set & dev_set),
                (train_dev_set | dev_set) == official_set,
            )
        )
    if official_test is not None:
        print("[check-pairs] official test set untouched (read-only check)")



def _png_ihdr_size(path):
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    return h, w


def _jpeg_size(path):
    with open(path, "rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = handle.read(1)
            while byte and byte != b"\xff":
                byte = handle.read(1)
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                return None
            code = marker[0]
            if code in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                handle.read(2)  # length
                handle.read(1)  # precision
                h = int.from_bytes(handle.read(2), "big")
                w = int.from_bytes(handle.read(2), "big")
                return h, w
            length = int.from_bytes(handle.read(2), "big")
            handle.seek(length - 2, 1)


def _image_size(path):
    ext = Path(path).suffix.lower()
    if ext == ".png":
        return _png_ihdr_size(path)
    if ext in (".jpg", ".jpeg"):
        return _jpeg_size(path)
    return None


def _crc32_file(path, chunk=1 << 20):
    import zlib

    crc = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def _inspect_zip(zip_path, sun_dir, sample=3):
    """Read-only inspection of the local SUNRGBD.zip (no extraction).

    Reports size, top-level layout and, for a few members, whether the on-disk
    extracted file matches the archive member (stored CRC vs on-disk CRC).
    """
    import zipfile

    size = zip_path.stat().st_size
    print("[check-source]   zip size: %d bytes (%.2f GiB)" % (size, size / (1024 ** 3)))
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        tops = sorted({n.split("/")[0] for n in names})
        print("[check-source]   zip members: %d, top-level: %s" % (len(names), tops))
        print("[check-source]   zip first names: %s" % names[:6])
        # Sample a few depth/rgb/label members and compare with on-disk files.
        picks = [n for n in names if n.endswith((".png", ".jpg"))][:sample * 3]
        compared = 0
        for name in picks:
            info = zf.getinfo(name)
            base = Path(name).name
            # on-disk layout is SUNRGBD/{RGB,Depth,labels}/<base>
            on_disk = None
            for sub in ("RGB", "Depth", "labels"):
                cand = sun_dir / sub / base
                if cand.exists():
                    on_disk = cand
                    break
            if on_disk is None:
                continue
            same = (_crc32_file(on_disk) == info.CRC) and (on_disk.stat().st_size == info.file_size)
            print("[check-source]   member %-28s (%d B) == on-disk: %s"
                  % (base, info.file_size, same))
            compared += 1
            if compared >= sample:
                break
        if compared == 0:
            print("[check-source]   no common members found to compare")

        # Explicitly compare the official split lists if the archive ships them.
        nameset = set(names)
        for member in ("SUNRGBD/train.txt", "SUNRGBD/test.txt"):
            if member not in nameset:
                print("[check-source]   member %-28s not in archive" % member)
                continue
            info = zf.getinfo(member)
            on_disk = sun_dir / Path(member).name
            if not on_disk.exists():
                print("[check-source]   member %-28s on-disk absent" % member)
                continue
            same = (_crc32_file(on_disk) == info.CRC) and (on_disk.stat().st_size == info.file_size)
            print("[check-source]   member %-28s (%d B) == on-disk: %s"
                  % (member, info.file_size, same))


def cmd_check_source(args):
    data_root = Path(args.data_root)
    sun = data_root / "SUNRGBD"
    print("[check-source] dataset root: %s" % data_root)
    if not data_root.exists():
        print("[check-source] dataset root does not exist")

    print("[check-source] SUNRGBD top level:")
    if sun.exists():
        for p in sorted(sun.iterdir()):
            kind = "dir" if p.is_dir() else "file"
            size = "" if p.is_dir() else " %d bytes" % p.stat().st_size
            print("[check-source]   %-24s %s%s" % (p.name, kind, size))
    else:
        print("[check-source]   SUNRGBD directory not found")

    candidates = [
        data_root / "SUNRGBD.zip",
        data_root / "downloads" / "SUNRGBD.zip",
        sun / "README.md",
        sun / "dataset_meta.json",
        sun / "SUNRGBD.zip",
    ]
    print("[check-source] candidate source/download records (no full scan):")
    for c in candidates:
        if c.exists():
            print("[check-source]   FOUND %s (%d bytes)" % (c, c.stat().st_size))
        else:
            print("[check-source]   absent %s" % c)


    root_zip = data_root / "SUNRGBD.zip"
    if root_zip.exists():
        print("[check-source] inspecting %s (read-only, sample=%d):" % (root_zip, args.zip_samples))
        _inspect_zip(root_zip, sun, sample=args.zip_samples)


def cmd_check_samples(args):
    """Run a handful of real samples through both preprocessors and assert invariants."""
    data_root = Path(args.data_root)
    setting = sunrgbd_setting(data_root)
    splits_dir = Path(args.splits_dir) if args.splits_dir else _default_splits_dir(args.seed)
    targets = _split_files(splits_dir)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_setting = dict(setting)
    train_setting["train_source"] = str(targets["train"])
    train_pre = ObservationTrainPre(mean, std, sign=True,
                                    config=_config_for_preprocess(pad=True))
    train_ds = ObservationDataset(train_setting, "train", train_pre)

    val_setting = dict(setting)
    val_setting["eval_source"] = str(targets["val"])
    val_pre = ObservationValPre(mean, std, sign=True,
                                config=_config_for_preprocess(pad=True))
    val_ds = ObservationDataset(val_setting, "val", val_pre)

    def _report(tag, ds, expect_hw):
        n = len(ds)
        count = min(args.check_samples, n)
        picks = np.linspace(0, n - 1, count).round().astype(int)
        for idx in picks:
            item = ds[int(idx)]
            rgb = item["data"]
            gt = item["label"]
            dep = item["modal_x"]
            sup = item["depth_support"]
            assert tuple(rgb.shape) == (3, expect_hw[0], expect_hw[1]), rgb.shape
            assert tuple(dep.shape) == (3, expect_hw[0], expect_hw[1]), dep.shape
            assert tuple(sup.shape) == (1, expect_hw[0], expect_hw[1]), sup.shape
            assert tuple(gt.shape) == (expect_hw[0], expect_hw[1]), gt.shape
            sup_np = sup.numpy()
            uniq = np.unique(sup_np)
            assert np.all(np.isin(uniq, [0.0, 1.0])), uniq
            dep_np = dep.numpy()
            finite = np.isfinite(dep_np[0])
            assert np.all(sup_np[0][~finite] == 0), "support where depth not finite"
            assert rgb.dtype == torch.float32 and dep.dtype == torch.float32
            assert sup.dtype == torch.float32 and gt.dtype == torch.int64
            assert bool((((gt >= 0) & (gt < 37)) | (gt == 255)).all()), torch.unique(gt)
            assert bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(dep).all())
            centers, prob, valid = soft_depth_histogram(dep_np[0], sup_np[0], DEFAULT_NUM_BINS)
            assert abs(prob.sum() - 1.0) < 1e-9 or valid == 0
            ignore = int((gt == 255).sum())
            print(
                "[check-samples] %-6s idx=%-4d rgb%s dep%s sup%s gt%s "
                "support=%.4f classes=%d ignore_px=%d hist_sum=%.4f"
                % (tag, int(idx), tuple(rgb.shape), tuple(dep.shape), tuple(sup.shape),
                   tuple(gt.shape), float(sup_np.mean()), int(len(np.unique(gt))),
                   ignore, float(prob.sum()))
            )

    _report("train", train_ds, (480, 480))
    _report("val", val_ds, (531, 730))

    # Validation padding: padded band must be raw-zero depth and zero support.
    n = len(val_ds)
    count = min(args.check_samples, n)
    picks = np.linspace(0, n - 1, count).round().astype(int)
    for idx in picks:
        sample = val_ds[int(idx)]
        fn = sample["fn"]
        item = "RGB/" + Path(fn).name
        depth_path = get_path(
            setting["dataset_name"],
            setting["rgb_root"], setting["rgb_format"],
            setting["x_root"], setting["x_format"],
            setting["gt_root"], setting["gt_format"], ["d"], item,
        )["d_path"]
        native = _image_size(depth_path)
        nh, nw = native
        dep = sample["modal_x"][0].numpy()
        sup = sample["depth_support"][0].numpy()
        band = np.ones_like(sup, dtype=bool)
        band[:nh, :nw] = False
        if band.any():
            assert np.all(sup[band] == 0), "padding support not zero"
            assert np.allclose(dep[band], DEPTH_RAW_ZERO_NORM, atol=1e-5), \
                "val depth padding is not raw-zero normalised"
            print("[check-samples] val idx=%d native=%dx%d pad_band_px=%d raw0_ok=True"
                  % (int(idx), nh, nw, int(band.sum())))


def cmd_preview(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_root = Path(args.data_root)
    setting = sunrgbd_setting(data_root)

    splits_dir = Path(args.splits_dir) if args.splits_dir else _default_splits_dir(args.seed)
    targets = _split_files(splits_dir)
    setting = dict(setting)
    setting["eval_source"] = str(targets["val"])

    pre = ObservationValPre([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                            sign=True, config=_config_for_preprocess(pad=True))
    dataset = ObservationDataset(setting, "val", pre)

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "odg-preparation" / "data-preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(dataset)
    count = min(args.num_samples, n)
    picks = np.linspace(0, n - 1, count).round().astype(int)
    centers = depth_bin_centers(DEFAULT_NUM_BINS)

    summary = []
    for idx in picks:
        item = dataset[int(idx)]
        rgb = item["data"].numpy()
        depth = item["modal_x"].numpy()
        gt = item["label"].numpy()
        support = item["depth_support"].numpy()
        fn = item["fn"]

        rgb_disp = rgb.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb_disp = np.clip(rgb_disp, 0.0, 1.0)

        supp2d = support[0]
        depth2d = depth[0].copy()
        depth_disp = np.where(supp2d > 0, depth2d, np.nan)

        _, prob, valid = soft_depth_histogram(depth2d, supp2d, DEFAULT_NUM_BINS)
        valid_frac = float((supp2d > 0).mean())
        d_in = depth2d[(supp2d > 0) & np.isfinite(depth2d)]
        if d_in.size:
            dmin, dmax = float(d_in.min()), float(d_in.max())
        else:
            dmin = dmax = float("nan")

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        axes[0].imshow(rgb_disp)
        axes[0].set_title("RGB (author-normalised, denorm)")
        axes[1].imshow(depth_disp, cmap="viridis")
        axes[1].set_title("depth (author norm, support only)")
        axes[2].imshow(np.where(gt == 255, 0, gt), cmap="nipy_spectral", vmin=0, vmax=36)
        axes[2].set_title("label (ignore=255 shown as 0)")
        axes[3].bar(centers, prob, width=(centers[1] - centers[0]) * 0.9,
                    color="#4c72b0")
        axes[3].axvline(DEPTH_RAW_ZERO_NORM, color="crimson", ls="--", lw=1,
                        label="raw-zero -> %.3f" % DEPTH_RAW_ZERO_NORM)
        axes[3].set_title("16-bin soft depth distribution")
        axes[3].set_xlabel("normalised depth bin centre")
        axes[3].set_ylabel("occupancy p")
        axes[3].legend(fontsize=7)
        for ax in axes[:3]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("dev sample %d/%d  %s" % (int(idx), n - 1,
                     Path(fn).name), fontsize=9)
        fig.tight_layout()
        out_png = out_dir / ("preview_%s.png" % Path(fn).stem)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)

        line = ("idx=%d fn=%s support_frac=%.4f valid_px=%d depth_norm[min=%.4f max=%.4f] "
                "classes=%d p=%s" % (int(idx), Path(fn).name, valid_frac, valid, dmin, dmax,
                                     int(len(np.unique(gt))),
                                     np.array2string(prob, precision=3, max_line_width=200)))
        print("[preview] %s" % line)
        summary.append(line + " png=%s" % out_png.name)

    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("[preview] wrote %d figures + summary.txt to %s" % (count, out_dir))


def build_parser():
    parser = argparse.ArgumentParser(description="ODG observation data layer / preparation CLI")
    parser.add_argument("--data-root", default=os.environ.get("DFORMER_DATASET_ROOT"),
                        help="parent directory containing SUNRGBD/ (default: $DFORMER_DATASET_ROOT)")
    parser.add_argument("--splits-dir", default=None,
                        help="split output dir (default: research/splits/sunrgbd_seed<seed>)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dev-size", type=int, default=528)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--zip-samples", type=int, default=3,
                        help="number of SUNRGBD.zip members to compare with on-disk files")
    parser.add_argument("--output-dir", default=None,
                        help="preview output dir (default: outputs/odg-preparation/data-preview)")
    parser.add_argument("--prepare-split", action="store_true",
                        help="write research/splits/sunrgbd_seed<seed>/{train-dev,dev}.txt")
    parser.add_argument("--check-pairs", action="store_true",
                        help="check all RGB/Depth/label path pairings for the lists")
    parser.add_argument("--check-samples", type=int, default=0,
                        help="run N real samples through TrainPre and ValPre and assert invariants")
    parser.add_argument("--preview", action="store_true",
                        help="render a few RGB/depth/label/16-bin previews")
    parser.add_argument("--check-source", action="store_true",
                        help="shallow report of download records / package origin")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root (or $DFORMER_DATASET_ROOT) is required")
    if not any([args.prepare_split, args.check_pairs, args.preview,
                args.check_source, args.check_samples]):
        parser.error("nothing to do: pass at least one of --prepare-split/--check-pairs/"
                     "--preview/--check-source/--check-samples")
    if args.prepare_split:
        cmd_prepare_split(args)
    if args.check_pairs:
        cmd_check_pairs(args)
    if args.check_samples:
        cmd_check_samples(args)
    if args.preview:
        cmd_preview(args)
    if args.check_source:
        cmd_check_source(args)


if __name__ == "__main__":
    main()
