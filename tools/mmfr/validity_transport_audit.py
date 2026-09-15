#!/usr/bin/env python3
"""CPU-only audit of how ``MMFR-A2-train-integration-v2`` corruption transports or
manufactures Depth validity.

Question this tool answers
--------------------------
``MMFR-A2-train-integration-v2`` supervises the Depth reliability channel with

    R_D^sup(p) = V_D^pre(p) * R_D^syn(p),

where ``V_D^pre(p)`` is ``raw Depth > 0`` *before* corruption and ``R_D^syn`` is the A1
v2 synthetic reliability target (``utils/dataloader/multimodal_failure_v2.py``). The
batch helper (``utils/dataloader/mmfr_training_v2.py``) keeps four pixel populations
separate and closes its census exactly:

    post_corruption_invalid_pixels + newly_valid_pixels
        == natural_invalid_pixels + synthetic_missing_pixels.

That identity is a *bookkeeping* statement. This tool measures what the four populations
actually contain, per corruption kind and per severity, and shows that a closed census
does not by itself make the supervised target semantically correct:

* ``gaussian_noise``, ``blur`` and ``misalignment`` can turn a natively invalid
  (``raw Depth == 0``) pixel into a nonzero one. Those pixels are handed to the
  segmentation backbone as real Depth while the reliability target for them is pinned to
  ``0`` by ``V_D^pre`` -- the target says "missing", the model input says "measurement";
* ``misalignment`` additionally moves Depth *content* spatially while validity stays
  attached to the fixed original coordinate, so a pixel can be valid because of what
  landed on it and another pixel can become invalid because its content was carried away.

The tool records the numbers, states the two candidate resolutions as an explicit
``decision_required`` block, and deliberately selects neither.

What is measured
----------------
1. Per ``(fixture, kind, severity)``: ``V_pre``/``V_post`` pixel counts, the four
   populations (``natural_invalid``, ``synthetic_missing``, ``newly_valid``,
   ``implicit_quality``), ``R_syn`` and ``R_sup`` mean/median/exact-zero fraction, and the
   input value change statistics.
2. A dedicated misalignment section with the realized ``dx``/``dy``, the reference
   displacement, the four transport relations, and the before/after validity agreement
   ratio.
3. A ``decision_required`` block with the two options and the measured evidence for each.

Method and its boundaries
-------------------------
* The ``(kind, severity)`` sweep applies exactly one spec per run through the public
  ``apply_failures`` entry point, with a **fresh** RNG rebuilt for every run from
  ``numpy.random.default_rng(numpy.random.SeedSequence([FIXED_SEED]))``. Sharing one seed
  across all runs means the same underlying random stream is transported through every
  kind, which is what makes the cross-kind comparison meaningful.
* Applying a spec at an exact severity is not reachable through
  ``build_mmfr_training_batch_v2``, because that helper draws its own kinds and severities
  from the curriculum. The sweep therefore replays the helper's own per-sample pipeline
  (pad zeroing -> ``apply_failures`` -> ``R_sup`` composition). ``anchor_check`` proves
  bit for bit that the replay is the helper: the helper's reported seed words are used to
  rebuild its generator, the ``p_clean`` draw and the curriculum draw are replayed, and
  the resulting corrupted Depth, ``depth_valid_pre``, ``depth_valid_post`` and supervised
  target are compared against the helper's own outputs.
* All pixel censuses are restricted to the geometry-valid region
  ``valid = ~(rgb_pad & depth_pad)`` -- the same region the helper uses. The synthetic
  crop/pad is not an observation, and the helper forces it to exact ``0`` in
  ``raw_depth`` and to the neutral target ``1``.
* Fixtures are (a) real ``train-dev`` MUSeg Depth crops resized to ``480 x 640`` with
  ``INTER_NEAREST`` and (b) one deterministic synthetic gradient/texture control. Both
  carry an injected natively invalid block and a ``32``-column / ``32``-row exact-zero
  pad, so ``natural_invalid`` is guaranteed non-empty.
* CPU only. No GPU, no training, no evaluation, no cloud resource, no network, and no
  mutation of any repository file outside the declared output directory.

Exit code is ``0`` when every internal consistency check passes and ``1`` otherwise.
The exit code reports measurement integrity only; it carries no opinion on the
``decision_required`` block.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloader.mmfr_training import (  # noqa: E402
    CLEAN_PROBABILITY,
    CORRUPTION_SEED,
    DEPTH_PLANE_MEAN,
    DEPTH_PLANE_STD,
    MAX_SPECS,
    make_sample_generator,
    normalized_to_uint8,
    uint8_to_normalized,
)
from utils.dataloader.mmfr_training_v2 import (  # noqa: E402
    CORRUPTION_BASIS,
    DEPTH_RELIABILITY_INDEX,
    PROTOCOL_ID as A2_PROTOCOL_ID,
    TARGET_COMPOSITION,
    build_mmfr_training_batch_v2,
    sample_depth_failure_specs_v2,
)
import utils.dataloader.mmfr_training_v2 as mmfr_v2_module  # noqa: E402
from utils.dataloader.multimodal_failure_v2 import (  # noqa: E402
    FAILURE_KINDS,
    PROTOCOL_ID as A1_PROTOCOL_ID,
    SEVERITY_ENCODING,
    FailureSpec,
    apply_failures,
)
import utils.dataloader.multimodal_failure_v2 as mmfr_a1_module  # noqa: E402

# --------------------------------------------------------------------------------------
# Audit configuration. Everything that changes a number is declared here and recorded.
# --------------------------------------------------------------------------------------

#: One seed for the whole sweep. Every ``(fixture, kind, severity)`` run rebuilds its RNG
#: from this seed, so all runs share the same underlying random stream.
FIXED_SEED = 20260915

SEVERITY_GRID: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
#: ``entire_missing`` is a whole-modality failure; only full severity is meaningful and
#: the frozen curriculum only ever samples it at ``1.0``.
ENTIRE_MISSING_SEVERITY = 1.0
ENTIRE_MISSING_ONLY_SEVERITY = True

SWEEP_KINDS: Tuple[str, ...] = (
    "gaussian_noise",
    "blur",
    "quantization",
    "misalignment",
    "spatial_dropout",
    "entire_missing",
)

CROP_HEIGHT = 480
CROP_WIDTH = 640
PAD_ROWS = 32
PAD_COLS = 32
#: Injected natively invalid Depth block, ``rows 10:20`` x ``cols 30:80``, set to ``0``.
INJECTED_INVALID_BLOCK = (slice(10, 20), slice(30, 80))
#: Deterministic resize target: Depth with ``INTER_NEAREST``, RGB with ``INTER_LINEAR``.
DEPTH_RESIZE_INTERPOLATION = "cv2.INTER_NEAREST"
RGB_RESIZE_INTERPOLATION = "cv2.INTER_LINEAR"

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
DEPTH_MEAN = (DEPTH_PLANE_MEAN,)
DEPTH_STD = (DEPTH_PLANE_STD,)

SPLIT_FILE = REPO_ROOT / "data" / "splits" / "MUSeg" / "dev-v1" / "train-dev.txt"
REAL_SAMPLE_COUNT = 3
REAL_SAMPLE_SCAN_LIMIT = 12

#: Frozen A2 config facts used only to position the anchor run in the heavy curriculum
#: stage: ``C.nepochs = 500``, ``C.niters_per_epoch = 1277 // 10 + 1 = 128``.
ANCHOR_EPOCH = 400
ANCHOR_NEPOCHS = 500
ANCHOR_NITERS_PER_EPOCH = 128
ANCHOR_ITERATIONS = 32

SYNTHETIC_FIXTURE_ID = "synthetic-gradient-texture-480x640"
SYNTHETIC_SAMPLE_ID = "synthetic/mmfr-validity-transport/480x640-gradient-texture-v1"

DATA_ROOT = Path(
    os.environ.get(
        "DFORMER_DATA_ROOT",
        str(REPO_ROOT.parent / "dataset"),
    )
)
DATASET_DIR = DATA_ROOT / "MUSeg_DFormer"

OUTPUT_DIR = REPO_ROOT / "outputs" / "mmfr-a2-v2-validity-transport"
REPORT_PATH = OUTPUT_DIR / "validity-transport-audit.json"

STRUCTURAL_KINDS: Tuple[str, ...] = ("entire_missing", "spatial_dropout")
#: Kinds whose output is produced by re-sampling intensities instead of by zeroing.
INTENSITY_KINDS: Tuple[str, ...] = ("gaussian_noise", "blur", "quantization")
SPATIAL_KINDS: Tuple[str, ...] = ("misalignment",)

DEPTH_FIXTURE_RULE = (
    "uint8 Depth = clip(rint(base + texture)) with u = y/(H-1), v = x/(W-1), "
    "base = 40 + 150*(0.55*u + 0.45*v) and "
    "texture = 25*sin(2*pi*7*v)*cos(2*pi*5*u) + 10*sin(2*pi*23*v + 1.3)*sin(2*pi*19*u); "
    "deterministic (no RNG), value range strictly inside (0, 255) before the injected "
    "invalid block"
)
RGB_FIXTURE_RULE = (
    "uint8 RGB = clip(rint(channel)) with u = y/(H-1), v = x/(W-1), "
    "base = 30 + 150*(0.5*u + 0.5*v) and texture = 30*sin(2*pi*9*v + 0.4)*cos(2*pi*6*u); "
    "channel0 = base + texture, channel1 = 0.85*base + 20 + texture, "
    "channel2 = 200 - 0.7*base + texture; deterministic (no RNG), value range strictly "
    "inside (0, 255)"
)

BOOKKEEPING_STATEMENT = (
    "A closed invalidity census is not evidence that the supervised target is "
    "semantically correct. The A2 v2 census identity "
    "(post_corruption_invalid + newly_valid == natural_invalid + synthetic_missing) is a "
    "balance over pixel counts; it holds for gaussian_noise and misalignment precisely "
    "because newly_valid was introduced to absorb pixels whose validity state changed. It "
    "says nothing about whether a natively invalid pixel that corruption turned nonzero "
    "should be handed to the segmentation backbone as a measurement, nor whether Depth "
    "validity should be transported spatially together with Depth content."
)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def distribution(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "exact_zero_fraction": float(np.count_nonzero(array == 0.0) / float(array.size)),
        "exact_one_fraction": float(np.count_nonzero(array == 1.0) / float(array.size)),
    }


def repo_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = completed.stdout.strip()
    return head or None


def make_sweep_rng() -> np.random.Generator:
    """Fresh generator for one ``(fixture, kind, severity)`` run, rebuilt from the seed."""
    return np.random.default_rng(np.random.SeedSequence([FIXED_SEED]))


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _grids(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    u = (np.arange(height, dtype=np.float64) / float(height - 1))[:, None]
    v = (np.arange(width, dtype=np.float64) / float(width - 1))[None, :]
    return u, v


def build_synthetic_depth(height: int, width: int) -> np.ndarray:
    u, v = _grids(height, width)
    base = 40.0 + 150.0 * (0.55 * u + 0.45 * v)
    texture = 25.0 * np.sin(2.0 * np.pi * 7.0 * v) * np.cos(2.0 * np.pi * 5.0 * u) + 10.0 * np.sin(
        2.0 * np.pi * 23.0 * v + 1.3
    ) * np.sin(2.0 * np.pi * 19.0 * u)
    return np.clip(np.rint(base + texture), 0.0, 255.0).astype(np.uint8)


def build_synthetic_rgb(height: int, width: int) -> np.ndarray:
    u, v = _grids(height, width)
    base = 30.0 + 150.0 * (0.5 * u + 0.5 * v)
    texture = 30.0 * np.sin(2.0 * np.pi * 9.0 * v + 0.4) * np.cos(2.0 * np.pi * 6.0 * u)
    channels = [base + texture, 0.85 * base + 20.0 + texture, 200.0 - 0.7 * base + texture]
    stacked = np.stack([np.clip(np.rint(channel), 0.0, 255.0) for channel in channels], axis=-1)
    return stacked.astype(np.uint8)


def build_pad_mask(height: int, width: int) -> np.ndarray:
    """Right ``PAD_COLS`` columns and bottom ``PAD_ROWS`` rows, the declared pad region."""
    pad = np.zeros((height, width), dtype=bool)
    pad[height - PAD_ROWS :, :] = True
    pad[:, width - PAD_COLS :] = True
    return pad


def finalize_fixture(
    fixture_id: str,
    source: str,
    role: str,
    sample_id: str | None,
    raw_depth_decoded: np.ndarray,
    raw_rgb_decoded: np.ndarray,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Inject the invalid block and pad, derive the normalized tensors, and census them.

    The decoded Depth is resized with ``INTER_NEAREST`` and, when the decoder returns
    three channels, the first channel (index ``0``) is used. The decoded RGB is resized
    with ``INTER_LINEAR``. Both arrays are then cropped to ``PAD_ROWS``/``PAD_COLS`` of
    exact zeros, and the natively invalid block is forced to ``0``.
    """
    height, width = CROP_HEIGHT, CROP_WIDTH
    if raw_depth_decoded.ndim == 3:
        depth_source_channel = 0
        raw_depth_decoded = np.ascontiguousarray(raw_depth_decoded[:, :, 0])
    else:
        depth_source_channel = None
    if raw_rgb_decoded.ndim == 2:
        raw_rgb_decoded = np.repeat(raw_rgb_decoded[:, :, None], 3, axis=2)
    raw_rgb_decoded = np.ascontiguousarray(raw_rgb_decoded[:, :, :3])

    depth = np.ascontiguousarray(
        cv2.resize(raw_depth_decoded, (width, height), interpolation=cv2.INTER_NEAREST)
    ).copy()
    rgb = np.ascontiguousarray(
        cv2.resize(raw_rgb_decoded, (width, height), interpolation=cv2.INTER_LINEAR)
    ).copy()
    if depth.ndim != 2 or depth.dtype != np.uint8:
        raise RuntimeError(f"fixture {fixture_id}: resized Depth must be 2D uint8, got {depth.shape}/{depth.dtype}")
    if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"fixture {fixture_id}: resized RGB must be HxWx3 uint8, got {rgb.shape}/{rgb.dtype}")

    pad = build_pad_mask(height, width)
    invalid_pre_resize = int(np.count_nonzero(depth == 0))
    depth[INJECTED_INVALID_BLOCK] = 0
    depth[pad] = 0
    rgb[pad] = 0
    invalid_after_injection = int(np.count_nonzero(depth == 0))

    depth_norm = uint8_to_normalized(depth[:, :, None], DEPTH_MEAN, DEPTH_STD)[:, :, 0]
    rgb_norm = uint8_to_normalized(rgb, RGB_MEAN, RGB_STD).transpose(2, 0, 1)
    # The synthetic crop/pad is not an observation: it is exact normalized zero in both
    # modalities, exactly as the frozen training transform leaves it.
    depth_norm[pad] = 0.0
    rgb_norm[:, pad] = 0.0
    valid = ~(np.all(rgb_norm == 0.0, axis=0) & (depth_norm == 0.0))
    # RGB normalization maps no uint8 value exactly onto 0 for any ImageNet channel mean,
    # so the geometry mask must be exactly the declared pad complement.
    if not np.array_equal(~valid, pad):
        raise RuntimeError(f"fixture {fixture_id}: geometry valid mask is not the pad complement")

    # The normalized fixture must round trip back to the very bytes the helper will see,
    # inside the geometry-valid region. The pad is excluded on purpose: normalized zero is
    # the channel mean and inverse-normalizes to a non-zero byte, which is exactly why the
    # helper zeroes the pad explicitly before any corruption.
    recovered_depth = normalized_to_uint8(depth_norm[None], DEPTH_MEAN, DEPTH_STD)[0]
    recovered_rgb = normalized_to_uint8(rgb_norm, RGB_MEAN, RGB_STD).transpose(1, 2, 0)
    depth_round_trip_exact = bool(np.array_equal(recovered_depth[valid], depth[valid]))
    rgb_round_trip_exact = bool(np.array_equal(recovered_rgb[valid], rgb[valid]))
    if not (depth_round_trip_exact and rgb_round_trip_exact):
        raise RuntimeError(f"fixture {fixture_id}: normalized fixture does not round trip to its own bytes")
    depth_pad_inverse_normalizes_to_mean = bool(
        np.array_equal(
            recovered_depth[pad],
            np.full(int(np.count_nonzero(pad)), int(round(DEPTH_MEAN[0] * 255.0)), dtype=np.uint8),
        )
    )

    depth_valid_pre = (depth > 0) & valid
    manifest = {
        "fixture_id": fixture_id,
        "source": source,
        "role": role,
        "sample_id": sample_id,
        "crop_shape": [height, width],
        "pad_rows": PAD_ROWS,
        "pad_cols": PAD_COLS,
        "pad_pixels": int(np.count_nonzero(pad)),
        "pad_is_exact_zero_in_raw_and_normalized": True,
        "depth_resize_interpolation": DEPTH_RESIZE_INTERPOLATION,
        "rgb_resize_interpolation": RGB_RESIZE_INTERPOLATION,
        "depth_decoded_channels": provenance.get("depth_decoded_channels"),
        "depth_source_channel_index": depth_source_channel,
        "depth_decoded_shape": provenance.get("depth_decoded_shape"),
        "rgb_decoded_shape": provenance.get("rgb_decoded_shape"),
        "injected_invalid_block": {
            "rows": [INJECTED_INVALID_BLOCK[0].start, INJECTED_INVALID_BLOCK[0].stop],
            "cols": [INJECTED_INVALID_BLOCK[1].start, INJECTED_INVALID_BLOCK[1].stop],
            "value": 0,
            "pixels": int(
                (INJECTED_INVALID_BLOCK[0].stop - INJECTED_INVALID_BLOCK[0].start)
                * (INJECTED_INVALID_BLOCK[1].stop - INJECTED_INVALID_BLOCK[1].start)
            ),
        },
        "invalid_depth_pixels_before_injection": invalid_pre_resize,
        "invalid_depth_pixels_after_injection": invalid_after_injection,
        "depth_crop_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        "rgb_crop_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "depth_crop_min": int(depth.min()),
        "depth_crop_max": int(depth.max()),
        "depth_crop_zero_fraction": float(np.count_nonzero(depth == 0) / float(height * width)),
        "valid_pixels": int(np.count_nonzero(valid)),
        "natively_valid_pixels_in_valid_region": int(np.count_nonzero(depth_valid_pre)),
        "natively_invalid_pixels_in_valid_region": int(np.count_nonzero(valid & ~depth_valid_pre)),
        "invalid_pixels_whole_grid_including_pad": int(np.count_nonzero(depth == 0)),
        "native_valid_fraction_of_valid": float(depth_valid_pre.mean()),
        "census_region_note": (
            "the census counts below and in the sweep are restricted to the geometry-valid "
            "region; the synthetic pad is excluded and is not a native sensor observation"
        ),
        "depth_normalization_round_trip_exact": depth_round_trip_exact,
        "rgb_normalization_round_trip_exact": rgb_round_trip_exact,
        "round_trip_region": "geometry-valid region only; the pad is excluded",
        "depth_pad_inverse_normalizes_to_channel_mean": depth_pad_inverse_normalizes_to_mean,
        "depth_pad_inverse_normalized_byte": int(round(DEPTH_MEAN[0] * 255.0)),
        "provenance": dict(provenance),
    }
    return {
        "fixture_id": fixture_id,
        "role": role,
        "source": source,
        "sample_id": sample_id,
        "depth_raw": depth,
        "rgb_raw": rgb,
        "depth_norm": depth_norm,
        "rgb_norm": rgb_norm,
        "pad": pad,
        "valid": valid,
        "depth_valid_pre": depth_valid_pre,
        "manifest": manifest,
    }


def load_real_fixtures() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fixtures: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"split file not found: {SPLIT_FILE}")
    entries = [line.strip() for line in SPLIT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    for relative in entries[:REAL_SAMPLE_SCAN_LIMIT]:
        if len(fixtures) >= REAL_SAMPLE_COUNT:
            break
        stem = Path(relative).stem
        rgb_path = DATASET_DIR / relative
        depth_candidates = sorted((DATASET_DIR / "Depth").glob(f"{stem}.*"))
        if not rgb_path.exists() or not depth_candidates:
            skipped.append(
                {
                    "sample_id": relative,
                    "reason": "missing RGB or Depth file",
                    "rgb_path": str(rgb_path),
                    "depth_candidates": [str(path) for path in depth_candidates],
                }
            )
            continue
        depth_path = depth_candidates[0]
        depth_decoded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb_decoded = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        if depth_decoded is None or rgb_decoded is None:
            skipped.append(
                {
                    "sample_id": relative,
                    "reason": "cv2.imread returned None",
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                }
            )
            continue
        provenance = {
            "depth_file": str(depth_path),
            "depth_file_sha256": sha256_file(depth_path),
            "depth_file_bytes": int(depth_path.stat().st_size),
            "depth_decoded_shape": list(depth_decoded.shape),
            "depth_decoded_dtype": str(depth_decoded.dtype),
            "depth_decoded_channels": 1 if depth_decoded.ndim == 2 else int(depth_decoded.shape[2]),
            "depth_decoded_min": int(depth_decoded.min()),
            "depth_decoded_max": int(depth_decoded.max()),
            "depth_zero_fraction_native_grid": float(np.count_nonzero(depth_decoded == 0) / float(depth_decoded.size)),
            "rgb_file": str(rgb_path),
            "rgb_file_sha256": sha256_file(rgb_path),
            "rgb_file_bytes": int(rgb_path.stat().st_size),
            "rgb_decoded_shape": list(rgb_decoded.shape),
            "rgb_decoded_dtype": str(rgb_decoded.dtype),
            "split_file": str(SPLIT_FILE.relative_to(REPO_ROOT)).replace("\\", "/"),
            "split_entry": relative,
        }
        fixtures.append(
            finalize_fixture(
                fixture_id=f"real-train-dev-{len(fixtures) + 1}-{stem}",
                source="real_museg_train_dev",
                role="real Depth crop from the frozen A2 train-dev split",
                sample_id=f"RGB/{Path(relative).name}",
                raw_depth_decoded=depth_decoded,
                raw_rgb_decoded=rgb_decoded,
                provenance=provenance,
            )
        )
    if not fixtures:
        raise RuntimeError("no real train-dev fixture could be constructed")
    return fixtures, skipped


def build_synthetic_fixture() -> Dict[str, Any]:
    depth = build_synthetic_depth(CROP_HEIGHT, CROP_WIDTH)
    rgb = build_synthetic_rgb(CROP_HEIGHT, CROP_WIDTH)
    if int(depth.min()) <= 0 or int(depth.max()) >= 255:
        raise RuntimeError("synthetic Depth fixture must stay strictly inside (0, 255) before injection")
    if int(rgb.min()) <= 0 or int(rgb.max()) >= 255:
        raise RuntimeError("synthetic RGB fixture must stay strictly inside (0, 255) before injection")
    provenance = {
        "depth_generation_rule": DEPTH_FIXTURE_RULE,
        "rgb_generation_rule": RGB_FIXTURE_RULE,
        "rng": "none: closed-form deterministic construction",
        "depth_decoded_shape": list(depth.shape),
        "depth_decoded_channels": 1,
        "rgb_decoded_shape": list(rgb.shape),
    }
    return finalize_fixture(
        fixture_id=SYNTHETIC_FIXTURE_ID,
        source="deterministic_synthetic",
        role="smooth gradient plus local texture control fixture",
        sample_id=SYNTHETIC_SAMPLE_ID,
        raw_depth_decoded=depth,
        raw_rgb_decoded=rgb,
        provenance=provenance,
    )


# --------------------------------------------------------------------------------------
# Per-(fixture, kind, severity) sweep
# --------------------------------------------------------------------------------------


def census_fixture_arrays(
    fixture: Mapping[str, Any],
    depth_pre_uint8: np.ndarray,
    depth_post_uint8: np.ndarray,
) -> Dict[str, Any]:
    """Split the population exactly as the A2 v2 helper does, plus the derived identities."""
    valid = fixture["valid"]
    pre_positive = depth_pre_uint8 > 0
    post_positive = depth_post_uint8 > 0
    pre_valid = valid & pre_positive
    post_valid = valid & post_positive
    natural_invalid = int(np.count_nonzero(valid & ~pre_positive))
    synthetic_missing = int(np.count_nonzero(valid & pre_positive & ~post_positive))
    newly_valid = int(np.count_nonzero(valid & ~pre_positive & post_positive))
    post_corruption_invalid = int(np.count_nonzero(valid & ~post_positive))
    implicit_quality = int(
        np.count_nonzero(valid & pre_positive & post_positive & (depth_pre_uint8 != depth_post_uint8))
    )
    closure_left = post_corruption_invalid + newly_valid
    closure_right = natural_invalid + synthetic_missing
    v_post_identity = int(np.count_nonzero(pre_valid)) - synthetic_missing + newly_valid
    changed = int(np.count_nonzero(valid & (depth_pre_uint8 != depth_post_uint8)))
    if changed:
        absolute_change = np.abs(
            depth_post_uint8[valid & (depth_pre_uint8 != depth_post_uint8)].astype(np.int32)
            - depth_pre_uint8[valid & (depth_pre_uint8 != depth_post_uint8)].astype(np.int32)
        )
        mean_abs_change_uint8_changed = float(absolute_change.mean())
    else:
        mean_abs_change_uint8_changed = 0.0
    whole_grid_pre = int(np.count_nonzero(depth_pre_uint8 > 0))
    whole_grid_post = int(np.count_nonzero(depth_post_uint8 > 0))
    return {
        "V_pre_pixels": int(np.count_nonzero(pre_valid)),
        "V_post_pixels": int(np.count_nonzero(post_valid)),
        "V_pre_pixels_whole_grid": whole_grid_pre,
        "V_post_pixels_whole_grid_kernel_output": whole_grid_post,
        "V_pre_fraction_of_valid": float(pre_valid.mean()),
        "V_post_fraction_of_valid": float(post_valid.mean()),
        "populations": {
            "valid_pixels": int(np.count_nonzero(valid)),
            "natural_invalid": natural_invalid,
            "synthetic_missing": synthetic_missing,
            "newly_valid": newly_valid,
            "implicit_quality": implicit_quality,
            "post_corruption_invalid": post_corruption_invalid,
            "closure_left": closure_left,
            "closure_right": closure_right,
            "closure_identity_holds": closure_left == closure_right,
            "closure_identity": (
                "post_corruption_invalid + newly_valid == natural_invalid + synthetic_missing"
            ),
            "V_post_identity_holds": v_post_identity == int(np.count_nonzero(post_valid)),
            "V_post_identity": "V_post == V_pre - synthetic_missing + newly_valid",
        },
        "input_change": {
            "changed_pixels": changed,
            "changed_pixels_fraction_of_valid": float(changed) / float(np.count_nonzero(valid)),
            "mean_abs_change_uint8_over_changed_pixels": mean_abs_change_uint8_changed,
            "mean_abs_change_normalized_over_changed_pixels": mean_abs_change_uint8_changed / 255.0,
            "mean_abs_change_uint8_over_valid_pixels": float(
                np.abs(
                    depth_post_uint8[valid].astype(np.float64) - depth_pre_uint8[valid].astype(np.float64)
                ).mean()
            ),
        },
    }


def run_sweep_record(fixture: Mapping[str, Any], kind: str, severity: float) -> Dict[str, Any]:
    """Apply one spec at one exact severity through the public A1 entry point."""
    rng = make_sweep_rng()
    depth_pre_uint8 = fixture["depth_raw"]
    rgb_pre_uint8 = fixture["rgb_raw"]
    spec = FailureSpec(modality="depth", kind=kind, severity=float(severity))
    result = apply_failures(rgb_pre_uint8, depth_pre_uint8, (spec,), rng)
    module_record = dict(result.metadata["specs"][0])
    depth_post_uint8 = np.ascontiguousarray(result.depth)
    R_syn = np.ascontiguousarray(result.reliability[DEPTH_RELIABILITY_INDEX], dtype=np.float32)

    valid = fixture["valid"]
    depth_valid_pre = fixture["depth_valid_pre"]
    depth_valid_post = (depth_post_uint8 > 0) & valid
    R_sup = np.where(depth_valid_pre, R_syn, np.float32(0.0)).astype(np.float32)

    census = census_fixture_arrays(fixture, depth_pre_uint8, depth_post_uint8)
    R_syn_valid = R_syn[valid]
    R_sup_valid = R_sup[valid]
    record: Dict[str, Any] = {
        "fixture": fixture["fixture_id"],
        "fixture_source": fixture["source"],
        "kind": kind,
        "severity": float(severity),
        "severity_encoding": SEVERITY_ENCODING,
        "corruption_parameter": dict(module_record.get("corruption_parameter", {})),
        "depth_valid_pre_matches_replay": bool(
            np.array_equal(depth_valid_pre, (depth_pre_uint8 > 0) & valid)
        ),
        "depth_valid_post_matches_replay": bool(
            np.array_equal(depth_valid_post, (depth_post_uint8 > 0) & valid)
        ),
        "V_pre_pixels": census["V_pre_pixels"],
        "V_post_pixels": census["V_post_pixels"],
        "V_pre_pixels_whole_grid": census["V_pre_pixels_whole_grid"],
        "V_post_pixels_whole_grid_kernel_output": census["V_post_pixels_whole_grid_kernel_output"],
        "V_pre_fraction_of_valid": census["V_pre_fraction_of_valid"],
        "V_post_fraction_of_valid": census["V_post_fraction_of_valid"],
        "populations": census["populations"],
        "R_syn": distribution(R_syn_valid),
        "R_sup": distribution(R_sup_valid),
        "R_syn_exact_zero_pixels": int(np.count_nonzero(R_syn_valid == 0.0)),
        "R_sup_exact_zero_pixels": int(np.count_nonzero(R_sup_valid == 0.0)),
        "R_syn_unique_value_count": int(np.unique(R_syn_valid).size),
        "R_sup_equals_V_pre_times_R_syn_bitwise": bool(
            np.array_equal(
                R_sup,
                np.where(depth_valid_pre, R_syn, np.float32(0.0)).astype(np.float32),
            )
        ),
        "input_change": census["input_change"],
        "module_record": to_jsonable(module_record),
    }
    if kind in STRUCTURAL_KINDS:
        unique = {float(value) for value in np.unique(R_syn_valid)}
        record["structural_target_is_binary"] = bool(unique.issubset({0.0, 1.0}))
    if kind == "entire_missing":
        record["entire_missing_zeroes_all_valid_pixels"] = bool(
            int(np.count_nonzero(depth_post_uint8[valid])) == 0
        )
    record["populations"]["newly_valid_is_zero_for_this_kind"] = census["populations"]["newly_valid"] == 0
    return record


# --------------------------------------------------------------------------------------
# Misalignment transport audit
# --------------------------------------------------------------------------------------


def misalignment_destination_mask(height: int, width: int, dy: int, dx: int) -> np.ndarray:
    """Rows/cols that receive transported content; the complement is set to zero."""
    dst = np.zeros((height, width), dtype=bool)
    dst[
        slice(max(0, dy), height - max(0, -dy)),
        slice(max(0, dx), width - max(0, -dx)),
    ] = True
    return dst


def audit_misalignment(
    fixture: Mapping[str, Any],
    severity: float,
    depth_post_uint8: np.ndarray,
    sweep_record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Count the four transport relations and the before/after validity agreement."""
    height, width = fixture["depth_raw"].shape
    valid = fixture["valid"]
    depth_pre_uint8 = fixture["depth_raw"]
    parameter = dict(sweep_record["corruption_parameter"])
    dy = int(parameter["dy_px"])
    dx = int(parameter["dx_px"])

    dst = misalignment_destination_mask(height, width, dy, dx)
    module_invalid_pixels = int(sweep_record["module_record"]["invalid_pixels"])
    geometry_matches_module = int(np.count_nonzero(~dst)) == module_invalid_pixels

    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    source_rows = np.clip(rows - dy, 0, height - 1)
    source_cols = np.clip(cols - dx, 0, width - 1)
    source_value = depth_pre_uint8[source_rows, source_cols]
    source_is_valid = source_value > 0
    origin_is_valid = depth_pre_uint8 > 0
    post_is_nonzero = depth_post_uint8 > 0

    in_dst = valid & dst
    out_dst = valid & ~dst

    relation_1 = int(np.count_nonzero(in_dst & source_is_valid & origin_is_valid))
    relation_2 = int(np.count_nonzero(in_dst & source_is_valid & ~origin_is_valid))
    relation_3 = int(np.count_nonzero(in_dst & ~source_is_valid & post_is_nonzero))
    relation_4 = int(np.count_nonzero(out_dst & origin_is_valid))
    relation_5 = int(np.count_nonzero(out_dst & ~origin_is_valid))

    agreement_mask = valid & (origin_is_valid == post_is_nonzero)
    disagreement_mask = valid & (origin_is_valid != post_is_nonzero)
    manufactured = int(np.count_nonzero(valid & ~origin_is_valid & post_is_nonzero))
    destroyed = int(np.count_nonzero(valid & origin_is_valid & ~post_is_nonzero))
    census_newly_valid = int(sweep_record["populations"]["newly_valid"])
    census_synthetic_missing = int(sweep_record["populations"]["synthetic_missing"])

    valid_pixels = int(np.count_nonzero(valid))
    return {
        "fixture": fixture["fixture_id"],
        "fixture_source": fixture["source"],
        "severity": float(severity),
        "realized_displacement": {
            "dy_px": dy,
            "dx_px": dx,
            "displacement_px": float(parameter["displacement_px"]),
            "reference_displacement_px": float(parameter["reference_displacement_px"]),
            "reference_displacement_definition": (
                "misalignment_reference_displacement(H, W) = "
                "hypot(MISALIGN_MAX_SHIFT_FRACTION*H, MISALIGN_MAX_SHIFT_FRACTION*W)"
            ),
            "max_shift_y_px": float(parameter["max_shift_y_px"]),
            "max_shift_x_px": float(parameter["max_shift_x_px"]),
            "shift_fraction_of_axis": float(parameter["shift_fraction_of_axis"]),
            "displacement_over_reference": float(parameter["displacement_px"])
            / float(parameter["reference_displacement_px"]),
            "normalized_burden_in_bounds": float(sweep_record["module_record"]["normalized_burden"]),
        },
        "transport_carrier": (
            "destination pixel p receives the source value at p - (dy, dx); every pixel "
            "outside the destination region is set to 0. Validity below is always evaluated "
            "at the fixed original coordinate p, which is what the frozen implementation does."
        ),
        "transport_relations": {
            "source_valid__target_originally_valid": relation_1,
            "source_valid__target_originally_invalid__now_nonzero": relation_2,
            "source_invalid__target_nonzero": relation_3,
            "out_of_bounds_zeroed__originally_valid": relation_4,
            "out_of_bounds_zeroed__originally_invalid": relation_5,
            "source_valid__transported_pixels_total": relation_1 + relation_2,
        },
        "validity_agreement": {
            "valid_pixels": valid_pixels,
            "agreement_pixels": int(np.count_nonzero(agreement_mask)),
            "disagreement_pixels": int(np.count_nonzero(disagreement_mask)),
            "agreement_fraction": float(np.count_nonzero(agreement_mask)) / float(valid_pixels),
            "disagreement_fraction": float(np.count_nonzero(disagreement_mask)) / float(valid_pixels),
            "manufactured_validity_pixels": manufactured,
            "destroyed_validity_pixels": destroyed,
            "manufactured_validity_definition": "invalid at p before corruption, nonzero at p after",
            "destroyed_validity_definition": "valid at p before corruption, zero at p after",
        },
        "reconciliation": {
            "census_newly_valid_pixels": census_newly_valid,
            "transport_relation_2": relation_2,
            "relation_2_equals_census_newly_valid": relation_2 == census_newly_valid,
            "manufactured_equals_relation_2": manufactured == relation_2,
            "census_synthetic_missing_pixels": census_synthetic_missing,
            "destroyed_equals_census_synthetic_missing": destroyed == census_synthetic_missing,
            "geometry_destination_mask_matches_module_invalid_pixels": geometry_matches_module,
            "module_invalid_pixels": module_invalid_pixels,
            "geometry_invalid_pixels": int(np.count_nonzero(~dst)),
        },
        "semantic_note": (
            "This is not a bookkeeping error. The invalidity census closes exactly "
            "(verified: post_corruption_invalid + newly_valid == natural_invalid + "
            "synthetic_missing), and relation 2 reconciles with the census newly_valid count. "
            "The open question is semantic: misalignment transports Depth *content* "
            "spatially, while validity stays attached to the fixed original coordinate. "
            "Whether Depth *validity* should be transported together with Depth content is a "
            "protocol decision, and this tool does not make it."
        ),
    }


# --------------------------------------------------------------------------------------
# Anchor: prove the replay is the A2 v2 helper's own pipeline
# --------------------------------------------------------------------------------------


def run_anchor_check(fixture: Mapping[str, Any]) -> Dict[str, Any]:
    """Rebuild the helper's RNG from its own seed words and compare outputs bit for bit."""
    height, width = fixture["depth_raw"].shape
    rgb_input = torch.from_numpy(np.ascontiguousarray(fixture["rgb_norm"], dtype=np.float32)[None])
    depth_input = torch.from_numpy(
        np.ascontiguousarray(
            np.repeat(fixture["depth_norm"][None, None], 3, axis=1), dtype=np.float32
        )
    )
    sample_id = fixture["sample_id"]
    rows: List[Dict[str, Any]] = []
    corrupted_checked = 0
    clean_checked = 0
    kinds_seen: List[str] = []
    failures: List[str] = []
    for iteration in range(ANCHOR_ITERATIONS):
        batch = build_mmfr_training_batch_v2(
            rgb_input,
            depth_input,
            [sample_id],
            epoch=ANCHOR_EPOCH,
            iteration=iteration,
            niters_per_epoch=ANCHOR_NITERS_PER_EPOCH,
            nepochs=ANCHOR_NEPOCHS,
            rgb_mean=RGB_MEAN,
            rgb_std=RGB_STD,
            corruption_seed=CORRUPTION_SEED,
            p_clean=CLEAN_PROBABILITY,
            max_specs=MAX_SPECS,
        )
        metadata = dict(batch["metadata"][0])
        valid = fixture["valid"]
        helper_raw_depth_uint8 = np.rint(
            batch["raw_depth"][0, 0].numpy().astype(np.float64) * 255.0
        ).astype(np.uint8)
        helper_pre = batch["depth_valid_pre"][0, 0].numpy()
        helper_post = batch["depth_valid_post"][0, 0].numpy()
        helper_target_depth = batch["reliability_target"][0, DEPTH_RELIABILITY_INDEX].numpy()
        helper_target_rgb_unique = {float(value) for value in np.unique(batch["reliability_target"][0, 0].numpy())}

        row: Dict[str, Any] = {
            "fixture": fixture["fixture_id"],
            "epoch": ANCHOR_EPOCH,
            "iteration": iteration,
            "curriculum_progress": float(metadata["curriculum_progress"]),
            "clean": bool(metadata["clean"]),
            "sample_id": metadata["sample_id"],
            "seed_words": list(metadata["seed_words"]),
            "helper_specs": [
                {"kind": record["kind"], "severity": float(record["severity"])}
                for record in metadata["specs"]
            ],
            "helper_census": {
                key: int(metadata[key])
                for key in (
                    "valid_pixels",
                    "natural_invalid_pixels",
                    "synthetic_missing_pixels",
                    "post_corruption_invalid_pixels",
                    "newly_valid_pixels",
                    "implicit_quality_pixels",
                )
            },
        }

        if metadata["clean"]:
            clean_checked += 1
            # A clean sample reuses the caller's normalized Depth tensor bit for bit.
            clean_depth_exact = bool(
                torch.equal(batch["depth"][0, 0], depth_input[0, 0])
            )
            clean_pre_equals_input = bool(
                np.array_equal(helper_pre, fixture["depth_valid_pre"])
            )
            clean_target_is_pre_validity = bool(
                np.array_equal(
                    helper_target_depth,
                    np.where(valid, fixture["depth_valid_pre"].astype(np.float32), 1.0).astype(np.float32),
                )
            )
            row.update(
                {
                    "clean_depth_tensor_bit_identical_to_input": clean_depth_exact,
                    "clean_depth_valid_pre_equals_fixture": clean_pre_equals_input,
                    "clean_supervised_target_is_pre_validity": clean_target_is_pre_validity,
                    "clean_raw_depth_recovery_max_abs_diff": int(
                        np.abs(
                            helper_raw_depth_uint8.astype(np.int32)
                            - fixture["depth_raw"].astype(np.int32)
                        ).max()
                    ),
                    "clean_rgb_target_channel_values": sorted(helper_target_rgb_unique),
                }
            )
            if not (clean_depth_exact and clean_pre_equals_input and clean_target_is_pre_validity):
                failures.append(f"clean iteration {iteration}: clean no-op property failed")
            rows.append(row)
            continue

        corrupted_checked += 1
        replay_rng = make_sample_generator(metadata["seed_words"])
        replay_is_clean = bool(float(replay_rng.random()) < float(CLEAN_PROBABILITY))
        replay_specs, _ = sample_depth_failure_specs_v2(
            float(metadata["curriculum_progress"]), replay_rng, max_specs=MAX_SPECS
        )
        replay = apply_failures(
            np.ascontiguousarray(fixture["rgb_raw"]),
            np.ascontiguousarray(fixture["depth_raw"]),
            replay_specs,
            replay_rng,
        )
        replay_depth_uint8 = np.ascontiguousarray(replay.depth)
        replay_R_syn = np.ascontiguousarray(replay.reliability[DEPTH_RELIABILITY_INDEX], dtype=np.float32)
        replay_R_sup = np.where(fixture["depth_valid_pre"], replay_R_syn, np.float32(0.0)).astype(np.float32)
        expected_target_depth = np.where(valid, replay_R_sup, np.float32(1.0)).astype(np.float32)

        specs_match = [
            (spec.kind, float(spec.severity)) for spec in replay_specs
        ] == [
            (record["kind"], float(record["severity"])) for record in metadata["specs"]
        ]
        clean_flag_matches = replay_is_clean == bool(metadata["clean"])
        depth_matches = bool(
            np.array_equal(replay_depth_uint8[valid], helper_raw_depth_uint8[valid])
        )
        depth_max_abs_diff_outside_valid = int(
            np.abs(
                replay_depth_uint8[~valid].astype(np.int32)
                - helper_raw_depth_uint8[~valid].astype(np.int32)
            ).max()
        )
        pre_matches = bool(np.array_equal(helper_pre, (fixture["depth_raw"] > 0) & valid))
        post_matches = bool(
            np.array_equal(helper_post, (helper_raw_depth_uint8 > 0) & valid)
        )
        target_matches = bool(np.array_equal(helper_target_depth, expected_target_depth))
        kinds_seen.extend(sorted({spec.kind for spec in replay_specs}))
        row.update(
            {
                "replay_is_clean": replay_is_clean,
                "clean_flag_matches_helper": clean_flag_matches,
                "replay_specs_match_helper": specs_match,
                "corrupted_depth_matches_helper_on_valid_region": depth_matches,
                "corrupted_depth_max_abs_diff_on_pad_region": depth_max_abs_diff_outside_valid,
                "corrupted_depth_pad_note": (
                    "the pad difference is expected and is not a mismatch: the helper forces "
                    "raw_depth to exact 0 on the geometry pad after corruption, while the raw "
                    "replay result keeps whatever the kernel left there"
                ),
                "depth_valid_pre_matches_replay": pre_matches,
                "depth_valid_post_matches_replay": post_matches,
                "supervised_target_equals_V_pre_times_R_syn": target_matches,
                "rgb_target_channel_values": sorted(helper_target_rgb_unique),
            }
        )
        if not (
            clean_flag_matches
            and specs_match
            and depth_matches
            and pre_matches
            and post_matches
            and target_matches
        ):
            failures.append(f"corrupted iteration {iteration}: helper replay mismatch")
        rows.append(row)

    return {
        "purpose": (
            "Prove that the direct replay used by the (kind, severity) sweep is bit for bit "
            "the A2 v2 helper's own pipeline, so the sweep measures the frozen training path "
            "and not a paraphrase of it."
        ),
        "method": (
            "Call build_mmfr_training_batch_v2 on the fixture, rebuild its per-sample "
            "generator from the seed words it reports, replay the p_clean draw and the "
            "curriculum spec draw, re-apply apply_failures, and compare the corrupted Depth, "
            "depth_valid_pre, depth_valid_post and supervised target against the helper's own "
            "tensors."
        ),
        "curriculum_position": {
            "epoch": ANCHOR_EPOCH,
            "iterations": ANCHOR_ITERATIONS,
            "nepochs": ANCHOR_NEPOCHS,
            "niters_per_epoch": ANCHOR_NITERS_PER_EPOCH,
            "stage": "heavy (progress >= 2/3), the only stage that offers entire_missing",
        },
        "fixture": fixture["fixture_id"],
        "iterations_scanned": ANCHOR_ITERATIONS,
        "corrupted_samples_checked": corrupted_checked,
        "clean_samples_checked": clean_checked,
        "kinds_observed": sorted({kind for kind in kinds_seen}),
        "all_passed": not failures,
        "failures": failures,
        "rows": rows,
        "scope_note": (
            "This anchor only compares the corruption stage of the helper. It does not load "
            "a model, does not run a forward or backward pass, and does not read the "
            "official test split."
        ),
    }


# --------------------------------------------------------------------------------------
# Decision block
# --------------------------------------------------------------------------------------


def build_decision_required(
    sweep: Sequence[Mapping[str, Any]],
    misalignment_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Write the open question and both candidate resolutions; deliberately choose none."""
    newly_valid_by_kind: Dict[str, Dict[str, Any]] = {}
    implicit_quality_by_kind: Dict[str, Dict[str, Any]] = {}
    for kind in SWEEP_KINDS:
        kind_records = [record for record in sweep if record["kind"] == kind]
        measured = [
            {
                "fixture": record["fixture"],
                "severity": record["severity"],
                "newly_valid_pixels": record["populations"]["newly_valid"],
                "implicit_quality_pixels": record["populations"]["implicit_quality"],
                "V_pre_pixels": record["V_pre_pixels"],
                "synthetic_missing_pixels": record["populations"]["synthetic_missing"],
                "post_corruption_invalid_pixels": record["populations"]["post_corruption_invalid"],
            }
            for record in kind_records
        ]
        newly = [row["newly_valid_pixels"] for row in measured]
        implicit = [row["implicit_quality_pixels"] for row in measured]
        newly_valid_by_kind[kind] = {
            "measured": measured,
            "max_newly_valid_pixels": max(newly) if newly else 0,
            "min_newly_valid_pixels": min(newly) if newly else 0,
            "severities_with_newly_valid_pixels": sorted(
                {row["severity"] for row in measured if row["newly_valid_pixels"] > 0}
            ),
            "manufactures_spurious_nonzero_depth": bool(any(value > 0 for value in newly)),
        }
        implicit_quality_by_kind[kind] = {
            "max_implicit_quality_pixels": max(implicit) if implicit else 0,
            "min_implicit_quality_pixels": min(implicit) if implicit else 0,
        }

    spurious_kinds = sorted(
        kind for kind, entry in newly_valid_by_kind.items() if entry["manufactures_spurious_nonzero_depth"]
    )
    clean_kinds = sorted(
        kind for kind, entry in newly_valid_by_kind.items() if not entry["manufactures_spurious_nonzero_depth"]
    )
    real_fixture_spurious = {
        kind: [
            row
            for row in newly_valid_by_kind[kind]["measured"]
            if row["fixture"].startswith("real-train-dev") and row["newly_valid_pixels"] > 0
        ]
        for kind in spurious_kinds
    }

    transport_table = [
        {
            "fixture": record["fixture"],
            "severity": record["severity"],
            "dy_px": record["realized_displacement"]["dy_px"],
            "dx_px": record["realized_displacement"]["dx_px"],
            "reference_displacement_px": record["realized_displacement"]["reference_displacement_px"],
            "source_valid__target_originally_invalid__now_nonzero": record["transport_relations"][
                "source_valid__target_originally_invalid__now_nonzero"
            ],
            "source_invalid__target_nonzero": record["transport_relations"]["source_invalid__target_nonzero"],
            "out_of_bounds_zeroed__originally_valid": record["transport_relations"][
                "out_of_bounds_zeroed__originally_valid"
            ],
            "agreement_fraction": record["validity_agreement"]["agreement_fraction"],
            "disagreement_fraction": record["validity_agreement"]["disagreement_fraction"],
        }
        for record in misalignment_records
    ]

    return {
        "status": "open",
        "decision_made_by_this_tool": False,
        "selection": None,
        "requested_of": "senior model reviewer",
        "note": (
            "This block states the question and the two candidate resolutions with their "
            "measured support. It records no preference, no ranking and no default. The "
            "implemented behaviour is described only as a fact."
        ),
        "question": (
            "When an intensity-space corruption is applied to a natively invalid Depth pixel "
            "(raw Depth == 0 before corruption), may the corruption produce a nonzero "
            "(spurious) Depth value at that pixel?"
        ),
        "why_it_matters": [
            "The A2 v2 supervised target is R_depth_sup(p) = depth_valid_pre(p) * "
            "R_depth_synthetic(p), so on a natively invalid pixel the supervised target is "
            "exactly 0 regardless of what R_depth_synthetic reports.",
            "The same pixel can still be handed to the segmentation backbone as a nonzero "
            "Depth value. The target says 'missing' while the model input says "
            "'measurement', and the reliability loss never constrains that input value.",
            "Measured here, the effect is large on the sparse real MUSeg Depth crop: "
            "gaussian_noise manufactures up to "
            f"{newly_valid_by_kind['gaussian_noise']['max_newly_valid_pixels']} nonzero Depth "
            "pixels per 480x640 crop that were natively 0, i.e. it rewrites validity over a "
            "large part of the natively invalid area.",
            "The census identity closes exactly with or without this behaviour, so batch "
            "telemetry alone cannot detect the difference; it is a semantics question, not a "
            "counting question.",
        ],
        "measured_evidence": {
            "kinds_that_manufacture_spurious_nonzero_depth": spurious_kinds,
            "kinds_that_never_manufacture_spurious_nonzero_depth": clean_kinds,
            "newly_valid_by_kind": newly_valid_by_kind,
            "implicit_quality_by_kind": implicit_quality_by_kind,
            "real_fixture_spurious_measurements": real_fixture_spurious,
            "definitions": {
                "newly_valid_pixels": (
                    "pixels inside the geometry-valid region with raw Depth == 0 before "
                    "corruption and raw Depth > 0 after; the helper records them separately "
                    "because gaussian_noise and misalignment can produce them"
                ),
                "implicit_quality_pixels": (
                    "pixels that stay nonzero but change value, which the binary validity mask "
                    "cannot see"
                ),
            },
        },
        "options": [
            {
                "id": "A",
                "name": "native invalidity is an absorbing sentinel",
                "statement": (
                    "A natively invalid Depth pixel (raw Depth == 0 before corruption) is "
                    "always treated as missing. Intensity corruptions -- gaussian_noise, blur "
                    "and quantization -- act only on Depth pixels that already carry a valid "
                    "measurement and are not allowed to create a nonzero Depth value on a "
                    "natively invalid pixel. Operationally the corruption output is masked "
                    "back to zero on the complement of V_D^pre, so the model input and the "
                    "reliability target agree on which pixels are missing."
                ),
                "consequences_if_chosen": [
                    "newly_valid_pixels becomes structurally 0 for every intensity kind, and "
                    "the census identity loses its fourth population.",
                    "The noisy Depth handed to the backbone contains no value at a pixel the "
                    "reliability target already calls missing.",
                    "The perturbation a natively invalid pixel receives is removed, so the "
                    "corrupted input is no longer 'the clean measurement plus noise' at those "
                    "pixels.",
                ],
                "measured_support": {
                    "magnitude_that_would_be_removed": {
                        kind: {
                            "max_newly_valid_pixels": newly_valid_by_kind[kind][
                                "max_newly_valid_pixels"
                            ],
                            "min_newly_valid_pixels": newly_valid_by_kind[kind][
                                "min_newly_valid_pixels"
                            ],
                            "severities": newly_valid_by_kind[kind][
                                "severities_with_newly_valid_pixels"
                            ],
                        }
                        for kind in spurious_kinds
                    },
                    "quantization_already_behaves_as_the_absorbing_variant": (
                        "quantization maps raw Depth 0 exactly onto 0 for every severity, so "
                        "it never manufactures spurious Depth; the option would make "
                        "gaussian_noise and blur consistent with it"
                    ),
                },
            },
            {
                "id": "B",
                "name": "spurious-measurement corruption is named explicitly",
                "statement": (
                    "Corruption is allowed to create nonzero Depth on natively invalid pixels, "
                    "but that behaviour must be given its own explicit name -- "
                    "'spurious-measurement corruption' -- and must not remain implicit inside "
                    "the ordinary gaussian_noise or blur kinds. Naming it implies at least: an "
                    "explicit kind or flag that marks the behaviour, explicit per-batch "
                    "reporting of the manufactured pixels, and an explicit statement that the "
                    "supervised target ignores them while the segmentation input contains "
                    "them."
                ),
                "consequences_if_chosen": [
                    "The existing gaussian_noise and blur kinds are reclassified; either they "
                    "gain a variant that is explicitly named spurious-measurement, or the new "
                    "kind is introduced alongside them.",
                    "newly_valid_pixels stays a first-class, explicitly reported quantity "
                    "rather than a diagnostic count.",
                    "Any future claim about what the reliability head has learned must state "
                    "that it was trained with spurious Depth present in its input.",
                ],
                "measured_support": {
                    "magnitude_that_would_be_named": {
                        kind: {
                            "max_newly_valid_pixels": newly_valid_by_kind[kind][
                                "max_newly_valid_pixels"
                            ],
                            "min_newly_valid_pixels": newly_valid_by_kind[kind][
                                "min_newly_valid_pixels"
                            ],
                            "severities": newly_valid_by_kind[kind][
                                "severities_with_newly_valid_pixels"
                            ],
                        }
                        for kind in spurious_kinds
                    },
                    "kinds_that_would_keep_their_current_name": clean_kinds,
                    "implicit_quality_scale": {
                        kind: implicit_quality_by_kind[kind]["max_implicit_quality_pixels"]
                        for kind in SWEEP_KINDS
                    },
                },
            },
        ],
        "misalignment_subquestion": {
            "status": "open",
            "decision_made_by_this_tool": False,
            "selection": None,
            "question": (
                "Should Depth validity be spatially transported together with Depth content "
                "under misalignment?"
            ),
            "why_it_is_a_separate_question": (
                "Option A and option B only govern intensity-space corruption. Misalignment "
                "moves content instead of re-sampling it, so even an 'absorbing sentinel' "
                "rule does not by itself say which pixel is valid afterwards."
            ),
            "measured_evidence": {
                "transport_table": transport_table,
                "reconciliation": [
                    {
                        "fixture": record["fixture"],
                        "severity": record["severity"],
                        "relation_2": record["reconciliation"]["transport_relation_2"],
                        "census_newly_valid_pixels": record["reconciliation"][
                            "census_newly_valid_pixels"
                        ],
                        "relation_2_equals_census_newly_valid": record["reconciliation"][
                            "relation_2_equals_census_newly_valid"
                        ],
                        "destroyed_equals_census_synthetic_missing": record["reconciliation"][
                            "destroyed_equals_census_synthetic_missing"
                        ],
                        "geometry_destination_mask_matches_module": record["reconciliation"][
                            "geometry_destination_mask_matches_module_invalid_pixels"
                        ],
                    }
                    for record in misalignment_records
                ],
                "relations": {
                    "source_valid__target_originally_valid": (
                        "transported content lands on a pixel that was already valid"
                    ),
                    "source_valid__target_originally_invalid__now_nonzero": (
                        "the source of newly_valid_pixels: the destination coordinate was "
                        "natively invalid and now carries transported depth"
                    ),
                    "source_invalid__target_nonzero": (
                        "structurally 0 in the frozen implementation, because each destination "
                        "pixel receives exactly one source value and a zero source yields a "
                        "zero destination"
                    ),
                    "out_of_bounds_zeroed__originally_valid": (
                        "depth was carried out of the grid and the pixel is now zero, so its "
                        "validity was destroyed without any sensor event"
                    ),
                },
            },
            "options": [
                {
                    "id": "MID-A",
                    "name": "validity is transported with the content",
                    "statement": (
                        "Depth validity moves with Depth content: for a destination pixel p "
                        "inside the destination region, V_post(p) = V_pre(p - (dy, dx)), and a "
                        "pixel whose content was carried out of the grid becomes invalid. The "
                        "validity mask is then the transported validity mask, not the "
                        "post-corruption value test at the fixed original coordinate."
                    ),
                    "measured_support": {
                        "transported_valid_pixels_per_fixture_severity": [
                            {
                                "fixture": record["fixture"],
                                "severity": record["severity"],
                                "transported_pixels": record["transport_relations"][
                                    "source_valid__transported_pixels_total"
                                ],
                            }
                            for record in misalignment_records
                        ]
                    },
                },
                {
                    "id": "MID-B",
                    "name": "validity stays attached to the original coordinate",
                    "statement": (
                        "Validity is derived only from the post-corruption value at the fixed "
                        "original coordinate, V_post(p) = raw_post(p) > 0, which is the "
                        "currently implemented behaviour. Content moved into a previously "
                        "invalid pixel counts as newly valid, and pixels whose content was "
                        "carried away count as zeroed."
                    ),
                    "measured_support": {
                        "manufactured_validity_pixels": [
                            {
                                "fixture": record["fixture"],
                                "severity": record["severity"],
                                "manufactured_validity_pixels": record["validity_agreement"][
                                    "manufactured_validity_pixels"
                                ],
                                "destroyed_validity_pixels": record["validity_agreement"][
                                    "destroyed_validity_pixels"
                                ],
                                "agreement_fraction": record["validity_agreement"][
                                    "agreement_fraction"
                                ],
                            }
                            for record in misalignment_records
                        ]
                    },
                },
            ],
            "semantic_note": (
                "The misalignment numbers are internally consistent: the transport relations "
                "reconcile exactly with the helper's census, and the destination geometry "
                "reproduces the module's own invalid-pixel mask. The question is therefore not "
                "whether the accounting is right, but whether 'a nonzero value now sits at "
                "this pixel' is the same claim as 'this pixel carries a valid measurement'."
            ),
        },
    }


# --------------------------------------------------------------------------------------
# Checks and report
# --------------------------------------------------------------------------------------


def collect_checks(
    fixtures: Sequence[Mapping[str, Any]],
    sweep: Sequence[Mapping[str, Any]],
    misalignment_records: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    violations: List[str] = []

    def add(name: str, passed: bool, detail: Mapping[str, Any], failure: str) -> None:
        entry = {"pass": bool(passed), **dict(detail)}
        checks[name] = entry
        if not passed:
            violations.append(failure)

    checks: Dict[str, Dict[str, Any]] = {}

    closure_failures = [
        f"{record['fixture']}/{record['kind']}@{record['severity']}: census identity failed"
        for record in sweep
        if not record["populations"]["closure_identity_holds"]
    ]
    add(
        "census_closure_identity_holds",
        not closure_failures,
        {
            "records_checked": len(sweep),
            "identity": "post_corruption_invalid + newly_valid == natural_invalid + synthetic_missing",
            "failures": closure_failures,
        },
        "census closure identity failed for at least one record",
    )

    v_post_failures = [
        f"{record['fixture']}/{record['kind']}@{record['severity']}: V_post identity failed"
        for record in sweep
        if not record["populations"]["V_post_identity_holds"]
    ]
    add(
        "V_post_identity_holds",
        not v_post_failures,
        {
            "records_checked": len(sweep),
            "identity": "V_post == V_pre - synthetic_missing + newly_valid",
            "failures": v_post_failures,
        },
        "V_post identity failed for at least one record",
    )

    target_failures = [
        f"{record['fixture']}/{record['kind']}@{record['severity']}: R_sup != V_pre * R_syn"
        for record in sweep
        if not record["R_sup_equals_V_pre_times_R_syn_bitwise"]
    ]
    add(
        "R_sup_is_bitwise_V_pre_times_R_syn",
        not target_failures,
        {
            "records_checked": len(sweep),
            "identity": "R_sup(p) = depth_valid_pre(p) * R_syn(p)",
            "failures": target_failures,
        },
        "R_sup composition is not bitwise V_pre * R_syn for at least one record",
    )

    mask_failures = [
        f"{record['fixture']}/{record['kind']}@{record['severity']}: validity mask mismatch"
        for record in sweep
        if not (record["depth_valid_pre_matches_replay"] and record["depth_valid_post_matches_replay"])
    ]
    add(
        "depth_validity_masks_match_replay",
        not mask_failures,
        {"records_checked": len(sweep), "failures": mask_failures},
        "replayed validity masks disagree with the recorded masks",
    )

    structural_failures = [
        f"{record['fixture']}/{record['kind']}@{record['severity']}: target not binary"
        for record in sweep
        if record["kind"] in STRUCTURAL_KINDS and not record.get("structural_target_is_binary", False)
    ]
    structural_failures += [
        f"{record['fixture']}/entire_missing: not every valid pixel is zeroed"
        for record in sweep
        if record["kind"] == "entire_missing"
        and not record.get("entire_missing_zeroes_all_valid_pixels", False)
    ]
    add(
        "structural_kinds_have_binary_targets",
        not structural_failures,
        {"kinds": list(STRUCTURAL_KINDS), "failures": structural_failures},
        "structural failure kinds produced a non-binary reliability target",
    )

    misalign_failures: List[str] = []
    for record in misalignment_records:
        label = f"{record['fixture']}/misalignment@{record['severity']}"
        recon = record["reconciliation"]
        if not recon["relation_2_equals_census_newly_valid"]:
            misalign_failures.append(f"{label}: transport relation 2 != census newly_valid")
        if not recon["manufactured_equals_relation_2"]:
            misalign_failures.append(f"{label}: manufactured validity != relation 2")
        if not recon["destroyed_equals_census_synthetic_missing"]:
            misalign_failures.append(f"{label}: destroyed validity != census synthetic_missing")
        if not recon["geometry_destination_mask_matches_module_invalid_pixels"]:
            misalign_failures.append(f"{label}: destination geometry != module invalid-pixel mask")
        if record["transport_relations"]["source_invalid__target_nonzero"] != 0:
            misalign_failures.append(
                f"{label}: a zero source produced a nonzero destination, which the frozen "
                "single-assignment carrier cannot do"
            )
    add(
        "misalignment_transport_relations_reconcile",
        not misalign_failures,
        {"records_checked": len(misalignment_records), "failures": misalign_failures},
        "misalignment transport relations do not reconcile with the census",
    )

    anchor_failures: List[str] = []
    for fixture_id, entry in anchor.items():
        for failure in entry["failures"]:
            anchor_failures.append(f"anchor[{fixture_id}]: {failure}")
    anchor_all_passed = all(entry["all_passed"] for entry in anchor.values()) and bool(anchor)
    add(
        "sweep_replay_is_the_helper_pipeline",
        anchor_all_passed,
        {
            "fixtures_anchored": [entry["fixture"] for entry in anchor.values()],
            "iterations_scanned_per_fixture": ANCHOR_ITERATIONS,
            "iterations_scanned_total": sum(entry["iterations_scanned"] for entry in anchor.values()),
            "corrupted_samples_checked": sum(
                entry["corrupted_samples_checked"] for entry in anchor.values()
            ),
            "clean_samples_checked": sum(entry["clean_samples_checked"] for entry in anchor.values()),
            "kinds_observed": sorted(
                {kind for entry in anchor.values() for kind in entry["kinds_observed"]}
            ),
            "failure_count": len(anchor_failures),
            "failures": anchor_failures[:8],
        },
        "the sweep replay is not bit for bit the A2 v2 helper pipeline",
    )

    fixture_failures = [
        fixture["fixture_id"]
        for fixture in fixtures
        if not (
            fixture["manifest"]["depth_normalization_round_trip_exact"]
            and fixture["manifest"]["rgb_normalization_round_trip_exact"]
        )
    ]
    natural_invalid_failures = [
        fixture["fixture_id"]
        for fixture in fixtures
        if fixture["manifest"]["natively_invalid_pixels_in_valid_region"] <= 0
    ]
    add(
        "fixtures_round_trip_and_have_native_invalidity",
        not fixture_failures and not natural_invalid_failures,
        {
            "fixtures": [fixture["fixture_id"] for fixture in fixtures],
            "round_trip_failures": fixture_failures,
            "fixtures_without_native_invalidity": natural_invalid_failures,
        },
        "a fixture failed the normalization round trip or has no natively invalid pixel",
    )

    add(
        "sweep_covers_the_declared_grid",
        len(sweep) == len(fixtures) * (len(SWEEP_KINDS) * len(SEVERITY_GRID) - (len(SEVERITY_GRID) - 1)),
        {
            "records": len(sweep),
            "expected": len(fixtures) * (len(SWEEP_KINDS) * len(SEVERITY_GRID) - (len(SEVERITY_GRID) - 1)),
            "kinds": list(SWEEP_KINDS),
            "severities": list(SEVERITY_GRID),
            "entire_missing_severities": [ENTIRE_MISSING_SEVERITY],
        },
        "the sweep does not cover the declared (fixture, kind, severity) grid",
    )

    return checks, violations


def print_table(rows: Sequence[Mapping[str, Any]], header: str) -> None:
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        print(row["line"])


def main() -> int:
    if SEVERITY_ENCODING != "single":
        raise RuntimeError("A1 v2 must encode severity exactly once")
    unknown = [kind for kind in SWEEP_KINDS if kind not in FAILURE_KINDS]
    if unknown:
        raise RuntimeError(f"sweep kinds are not A1 v2 failure kinds: {unknown}")

    real_fixtures, skipped = load_real_fixtures()
    fixtures = real_fixtures + [build_synthetic_fixture()]

    sweep: List[Dict[str, Any]] = []
    for fixture in fixtures:
        for kind in SWEEP_KINDS:
            severities = (ENTIRE_MISSING_SEVERITY,) if kind == "entire_missing" else SEVERITY_GRID
            for severity in severities:
                sweep.append(run_sweep_record(fixture, kind, float(severity)))

    sweep_index = {(record["fixture"], record["kind"], record["severity"]): record for record in sweep}
    misalignment_records: List[Dict[str, Any]] = []
    for fixture in fixtures:
        for severity in SEVERITY_GRID:
            record = sweep_index[(fixture["fixture_id"], "misalignment", float(severity))]
            rng = make_sweep_rng()
            result = apply_failures(
                fixture["rgb_raw"],
                fixture["depth_raw"],
                (FailureSpec(modality="depth", kind="misalignment", severity=float(severity)),),
                rng,
            )
            misalignment_records.append(
                audit_misalignment(
                    fixture,
                    float(severity),
                    np.ascontiguousarray(result.depth),
                    record,
                )
            )

    anchor = {fixture["fixture_id"]: run_anchor_check(fixture) for fixture in fixtures}

    checks, violations = collect_checks(fixtures, sweep, misalignment_records, anchor)
    for fixture_id, entry in anchor.items():
        if not entry["all_passed"]:
            failures = entry["failures"] or ["anchor produced no failure detail"]
            for failure in failures:
                violations.append(f"anchor[{fixture_id}]: {failure}")

    decision = build_decision_required(sweep, misalignment_records)

    exit_code = 0 if not violations else 1
    script_path = Path(__file__).resolve()
    report = {
        "audit": (
            "MMFR-A2-train-integration-v2 validity transport audit: how Depth corruption "
            "transports or manufactures Depth validity, measured per corruption kind"
        ),
        "protocol": {
            "corruption_basis": A1_PROTOCOL_ID,
            "train_integration": A2_PROTOCOL_ID,
            "corruption_basis_module_supersedes": mmfr_a1_module.SUPERSEDES,
            "severity_encoding": SEVERITY_ENCODING,
            "target_composition": TARGET_COMPOSITION,
            "supervised_channels": list(mmfr_v2_module.SUPERVISED_CHANNELS),
            "target_upstream_of_the_batch": "A1 returns R_D^syn; the helper forms R_D^sup",
        },
        "repository": {
            "root": str(REPO_ROOT),
            "head": repo_head(),
            "head_declared_by_caller": "0c653a9",
            "head_note": (
                "the head recorded here is read from the local checkout at run time; the "
                "caller declared 0c653a9 for this audit"
            ),
        },
        "script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
        },
        "modules_under_test": [
            {
                "path": str(Path(mmfr_a1_module.__file__).resolve()),
                "sha256": sha256_file(Path(mmfr_a1_module.__file__).resolve()),
                "role": "corruption basis: apply_failures, FailureSpec, FAILURE_KINDS, MISSING_BURDEN",
                "protocol_id": A1_PROTOCOL_ID,
            },
            {
                "path": str(Path(mmfr_v2_module.__file__).resolve()),
                "sha256": sha256_file(Path(mmfr_v2_module.__file__).resolve()),
                "role": (
                    "batch helper: build_mmfr_training_batch_v2, depth_valid_pre, "
                    "depth_valid_post, supervised target composition"
                ),
                "protocol_id": A2_PROTOCOL_ID,
            },
        ],
        "fixed_seed": FIXED_SEED,
        "rng_construction": (
            "numpy.random.default_rng(numpy.random.SeedSequence([FIXED_SEED])), rebuilt for "
            "every (fixture, kind, severity) run, so all runs share one underlying random stream"
        ),
        "severity_grid": list(SEVERITY_GRID),
        "entire_missing_severity": ENTIRE_MISSING_SEVERITY,
        "entire_missing_only_full_severity": ENTIRE_MISSING_ONLY_SEVERITY,
        "kinds": list(SWEEP_KINDS),
        "kind_taxonomy": {
            "structural_zeroing": list(STRUCTURAL_KINDS),
            "intensity_resampling": list(INTENSITY_KINDS),
            "spatial_content_transport": list(SPATIAL_KINDS),
            "note": (
                "structural_zeroing can only remove validity; intensity_resampling and "
                "spatial_content_transport can add it on pixels that were natively invalid, "
                "which is what this audit measures"
            ),
        },
        "geometry": {
            "crop_height": CROP_HEIGHT,
            "crop_width": CROP_WIDTH,
            "pad_rows": PAD_ROWS,
            "pad_cols": PAD_COLS,
            "pad_is_exact_zero_in_raw_and_normalized": True,
            "injected_native_invalid_block": {
                "rows": [INJECTED_INVALID_BLOCK[0].start, INJECTED_INVALID_BLOCK[0].stop],
                "cols": [INJECTED_INVALID_BLOCK[1].start, INJECTED_INVALID_BLOCK[1].stop],
                "value": 0,
            },
            "depth_resize_interpolation": DEPTH_RESIZE_INTERPOLATION,
            "rgb_resize_interpolation": RGB_RESIZE_INTERPOLATION,
            "depth_source_channel_rule": (
                "if the decoded Depth is three-channel, channel index 0 (the first channel) is "
                "used; the MUSeg Depth assets in this audit decode as single-channel 2D uint8"
            ),
            "geometry_valid_region": "valid = ~(rgb_pad & depth_pad), identical to the helper",
        },
        "normalization": {
            "rgb_mean": list(RGB_MEAN),
            "rgb_std": list(RGB_STD),
            "depth_mean": list(DEPTH_MEAN),
            "depth_std": list(DEPTH_STD),
            "padding_normalized_value": 0.0,
        },
        "sweep_method": {
            "how": (
                "one spec per run at the exact severity, through the public apply_failures entry "
                "point, with the pad zeroed before corruption and R_sup formed as "
                "depth_valid_pre * R_syn"
            ),
            "why_not_the_batch_helper": (
                "build_mmfr_training_batch_v2 draws its own kinds and severities from the "
                "curriculum, so it cannot place a run at an exact severity"
            ),
            "how_it_is_validated": (
                "anchor_check replays the helper's own RNG stream and compares corrupted Depth, "
                "depth_valid_pre, depth_valid_post and the supervised target bit for bit"
            ),
            "region": "all censuses are restricted to the geometry-valid region",
        },
        "fixtures": [fixture["manifest"] for fixture in fixtures],
        "fixtures_skipped": skipped,
        "sweep": sweep,
        "misalignment_transport_audit": {
            "why_this_section": (
                "misalignment is the only kind that moves Depth content instead of re-sampling "
                "or zeroing it, so it is the only kind where validity could in principle be "
                "transported together with the content"
            ),
            "semantic_statement": (
                "The invalidity census closing exactly is not evidence that the target "
                "semantics are correct. For misalignment this is not a bookkeeping error: it "
                "is the semantic question of whether Depth validity should be spatially "
                "transported together with Depth content."
            ),
            "relations_definition": {
                "source_valid__target_originally_valid": (
                    "destination pixel in the destination region, source value > 0, and the "
                    "destination coordinate was already > 0 before corruption"
                ),
                "source_valid__target_originally_invalid__now_nonzero": (
                    "destination coordinate was 0 before corruption and now carries a nonzero "
                    "transported value; this is the origin of newly_valid_pixels"
                ),
                "source_invalid__target_nonzero": (
                    "source value 0 but destination nonzero; structurally impossible for the "
                    "frozen single-assignment carrier and reported to prove that"
                ),
                "out_of_bounds_zeroed__originally_valid": (
                    "pixel outside the destination region that was > 0 before corruption and is "
                    "0 afterwards"
                ),
            },
            "region": "geometry-valid region, i.e. the pad is excluded",
            "records": misalignment_records,
        },
        "anchor_check": anchor,
        "decision_required": decision,
        "checks": checks,
        "conclusions": {
            "bookkeeping_closure_does_not_imply_target_semantics_are_correct": BOOKKEEPING_STATEMENT,
            "census_closure_holds_for_every_measured_record": checks["census_closure_identity_holds"]["pass"],
            "measured_records": len(sweep),
            "kinds_that_manufacture_spurious_nonzero_depth": decision["measured_evidence"][
                "kinds_that_manufacture_spurious_nonzero_depth"
            ],
            "kinds_that_never_manufacture_spurious_nonzero_depth": decision["measured_evidence"][
                "kinds_that_never_manufacture_spurious_nonzero_depth"
            ],
            "misalignment_disagreement_fraction_range": [
                min(record["validity_agreement"]["disagreement_fraction"] for record in misalignment_records),
                max(record["validity_agreement"]["disagreement_fraction"] for record in misalignment_records),
            ],
            "open_decisions": [
                "decision_required.options: A (native invalidity is an absorbing sentinel) or B "
                "(spurious-measurement corruption is named explicitly)",
                "decision_required.misalignment_subquestion: whether validity is transported "
                "with content",
            ],
            "this_tool_selects_no_option": True,
            "violations": violations,
            "violation_count": len(violations),
        },
        "limits": [
            "CPU only; no model is loaded, no forward or backward pass is run, no GPU is used.",
            "The sweep is a controlled replay of the corruption stage, not a training run.",
            "Only one spec is applied per sweep run; the frozen curriculum composes up to two "
            "Depth specs in the moderate and heavy stages, and composition effects are not "
            "measured here.",
            "The real fixtures are 480x640 crops of three train-dev samples, not a dataset-level "
            "statistic.",
            "official_test_included = false: the official test split was not read.",
        ],
        "official_test_included": False,
        "exit_code": exit_code,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    report_sha256 = sha256_file(REPORT_PATH)

    print("=" * 120)
    print("MMFR-A2-train-integration-v2 validity transport audit (CPU only)")
    print("=" * 120)
    print(f"corruption basis     : {A1_PROTOCOL_ID}   severity_encoding={SEVERITY_ENCODING!r}")
    print(f"train integration    : {A2_PROTOCOL_ID}")
    print(f"target composition   : {TARGET_COMPOSITION}")
    print(f"FIXED_SEED           : {FIXED_SEED}   (fresh RNG per (fixture, kind, severity) run)")
    print(f"script sha256        : {report['script']['sha256']}")
    print(f"python/numpy/opencv  : {sys.version.split()[0]} / {np.__version__} / {cv2.__version__}")
    print()
    print("[fixtures]")
    for fixture in fixtures:
        manifest = fixture["manifest"]
        print(
            f"  {manifest['fixture_id']:<52} source={manifest['source']:<24} "
            f"V_pre={manifest['natively_valid_pixels_in_valid_region']:>6} "
            f"natively_invalid={manifest['natively_invalid_pixels_in_valid_region']:>6} "
            f"valid={manifest['valid_pixels']:>6} pad={manifest['pad_pixels']:>6}"
        )
        print(
            f"      sample_id={manifest['sample_id']}  "
            f"depth_zero_fraction={manifest['depth_crop_zero_fraction']:.4f}  "
            f"depth_sha256={manifest['depth_crop_sha256'][:16]}..."
        )

    print()
    print("[sweep: populations and targets per (kind, severity), geometry-valid region]")
    header = (
        f"  {'fixture':<30}{'kind':<17}{'sev':>5}{'V_pre':>8}{'V_post':>8}{'nat_inv':>9}"
        f"{'syn_miss':>10}{'new_val':>9}{'impl_q':>8}{'R_syn_mu':>10}{'R_sup_mu':>10}"
        f"{'Rsup0%':>8}{'chg_px':>9}"
    )
    print(header)
    print("-" * len(header))
    for record in sweep:
        populations = record["populations"]
        short_fixture = record["fixture"][:29]
        print(
            f"  {short_fixture:<30}{record['kind']:<17}{record['severity']:>5.2f}"
            f"{record['V_pre_pixels']:>8}{record['V_post_pixels']:>8}{populations['natural_invalid']:>9}"
            f"{populations['synthetic_missing']:>10}{populations['newly_valid']:>9}"
            f"{populations['implicit_quality']:>8}{record['R_syn']['mean']:>10.6f}"
            f"{record['R_sup']['mean']:>10.6f}"
            f"{record['R_sup']['exact_zero_fraction'] * 100.0:>7.2f}%"
            f"{record['input_change']['changed_pixels']:>9}"
        )

    print()
    print("[misalignment transport audit]")
    header2 = (
        f"  {'fixture':<30}{'sev':>5}{'dy':>5}{'dx':>5}{'ref_disp':>10}"
        f"{'srcV->tgtV':>11}{'srcV->newV':>11}{'srcI->nz':>10}{'oob_zeroed':>11}"
        f"{'agree%':>8}{'disagree%':>11}"
    )
    print(header2)
    print("-" * len(header2))
    for record in misalignment_records:
        relations = record["transport_relations"]
        displacement = record["realized_displacement"]
        print(
            f"  {record['fixture'][:29]:<30}{record['severity']:>5.2f}{displacement['dy_px']:>5}"
            f"{displacement['dx_px']:>5}{displacement['reference_displacement_px']:>10.3f}"
            f"{relations['source_valid__target_originally_valid']:>11}"
            f"{relations['source_valid__target_originally_invalid__now_nonzero']:>11}"
            f"{relations['source_invalid__target_nonzero']:>10}"
            f"{relations['out_of_bounds_zeroed__originally_valid']:>11}"
            f"{record['validity_agreement']['agreement_fraction'] * 100.0:>7.2f}%"
            f"{record['validity_agreement']['disagreement_fraction'] * 100.0:>10.2f}%"
        )

    print()
    print("[anchor: the sweep replay must be the A2 v2 helper's own pipeline]")
    for fixture_id, entry in anchor.items():
        print(
            f"  {fixture_id:<52} iterations={entry['iterations_scanned']:>3} "
            f"corrupted={entry['corrupted_samples_checked']:>3} clean={entry['clean_samples_checked']:>3} "
            f"all_passed={entry['all_passed']} kinds={entry['kinds_observed']}"
        )

    print()
    print("[decision_required] status = open, selection = none (this tool chooses nothing)")
    print(f"  question: {decision['question']}")
    for option in decision["options"]:
        print(f"  option {option['id']} ({option['name']})")
        print(f"    {option['statement']}")
    subquestion = decision["misalignment_subquestion"]
    print(f"  misalignment subquestion: {subquestion['question']}")
    for option in subquestion["options"]:
        print(f"    option {option['id']} ({option['name']})")
        print(f"      {option['statement']}")

    print()
    print("[checks]")
    for name in sorted(checks):
        print(f"  {name}: {'PASS' if checks[name]['pass'] else 'FAIL'}")
    print()
    print("[conclusions]")
    print(f"  bookkeeping_closure_holds_for_every_record: {checks['census_closure_identity_holds']['pass']}")
    print(f"  measured_records: {len(sweep)}")
    print(
        "  kinds that manufacture spurious nonzero Depth: "
        f"{decision['measured_evidence']['kinds_that_manufacture_spurious_nonzero_depth']}"
    )
    print(
        "  kinds that never manufacture spurious nonzero Depth: "
        f"{decision['measured_evidence']['kinds_that_never_manufacture_spurious_nonzero_depth']}"
    )
    print("  census closure does NOT imply the target semantics are correct; see decision_required.")
    print(f"  violations: {len(violations)}")
    for violation in violations[:12]:
        print(f"    - {violation}")
    print()
    print(f"JSON report : {REPORT_PATH}")
    print(f"SHA-256     : {report_sha256}")
    print(f"official_test_included: False")
    print()
    print("OVERALL: " + ("PASS" if exit_code == 0 else "FAIL"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
