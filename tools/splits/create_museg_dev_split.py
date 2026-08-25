"""Create the frozen MUSeg group-aware development split candidate.

This program deliberately accepts only protocol MUSEG-DEV-SPLIT-PROTOCOL-1 input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PROTOCOL_ID = "MUSEG-DEV-SPLIT-PROTOCOL-1"
ALGORITHM_VERSION = "museg-group-best-improvement-v1"
SCHEMA_VERSION = "museg-dev-split-manifest-v1"
SEED_STRING = "DFormer/MUSeg/dev-split/MUSegDevSplit-v1"
SEED_HASH = "466e94a814b9c3ea0dd5021fcf98aeab07bc215d5878c59d2b7727ed0d0c6569"
SEED = 1181652136
TARGET = 319
TRAIN_HASH = "6ff78af2621e32bf0320aea606674a81c5bae21889ad3a3ff0109a9d1d398123"
TEST_HASH = "12d9834215fcbfe696ad88321539c224850ff6fb66a01f48a02b1df478f48a4b"
TRAIN_COUNT, TRAIN_GROUPS, TEST_COUNT, TEST_GROUPS = 1595, 958, 1576, 957
MINES = tuple(f"{n:02d}" for n in range(1, 7))
PATH_RE = re.compile(r"RGB/([^/]+)\.jpg\Z")
DEPTH_BINS = (Fraction(0), Fraction(1,2), Fraction(3,4), Fraction(9,10), Fraction(19,20), Fraction(99,100), Fraction(1))
LUMA_BINS = tuple(Fraction(n, 8) for n in range(9))

@dataclass(frozen=True)
class Sample:
    path: str
    group: str
    mine: str
    pixels: int
    presence: tuple[int, ...]
    class_pixels: tuple[int, ...]
    background: int
    depth_valid: int
    y_num: int
    depth_bin: int
    luma_bin: int
    class_count_bin: int


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"

def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")

def bin_index(value: Fraction, edges: tuple[Fraction, ...]) -> int:
    for index in range(len(edges) - 1):
        if value < edges[index + 1] or index == len(edges) - 2:
            return index
    raise AssertionError("bin value outside range")

def class_bin(count: int) -> int:
    return 0 if count == 0 else count if count <= 4 else 5 if count <= 6 else 6 if count <= 9 else 7

def parse_path(line: str) -> tuple[str, str]:
    match = PATH_RE.fullmatch(line)
    if not match or "\\" in line or line != line.strip() or "/./" in line or "/../" in line:
        raise ValueError(f"invalid official list path: {line!r}")
    pieces = match.group(1).split("-")
    if len(pieces) < 4 or any(not item for item in pieces) or pieces[0] not in MINES:
        raise ValueError(f"invalid stem/group/mine: {line!r}")
    return "-".join(pieces[:4]), pieces[0]

def parse_official_bytes(raw: bytes) -> tuple[str, ...]:
    if raw.startswith(b"\xef\xbb\xbf") or not raw:
        raise ValueError("official list must be non-BOM nonempty UTF-8")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("official list is not UTF-8") from error
    if b"\r" in raw.replace(b"\r\n", b""):
        raise ValueError("official list contains bare CR")
    ending = "\r\n" if b"\r\n" in raw else "\n"
    if not raw.endswith(ending.encode()) or (ending == "\r\n" and b"\n" in raw.replace(b"\r\n", b"")):
        raise ValueError("official list has mixed or missing line endings")
    lines = text[:-len(ending)].split(ending)
    if not lines or any(not line for line in lines) or len(lines) != len(set(lines)):
        raise ValueError("official list contains empty or duplicate lines")
    for line in lines: parse_path(line)
    return tuple(lines)

def group_order(group: str) -> tuple[str, bytes]:
    digest = hashlib.sha256(str(SEED).encode("ascii") + b"\n" + utf8_key(group)).hexdigest()
    return digest, utf8_key(group)

def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def git_object_is_commit(repo: Path, value: str | None) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        return False
    try:
        kind = subprocess.check_output(
            ["git", "cat-file", "-t", value], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return kind == "commit"


def tool_sources(repo: Path) -> dict[str, str]:
    logical_names = (
        "tools/splits/create_museg_dev_split.py",
        "tools/splits/audit_museg_splits.py",
    )
    return {name: sha256_bytes((repo / name).read_bytes()) for name in logical_names}

def jpeg_implementation() -> str:
    return next((line.strip() for line in cv2.getBuildInformation().splitlines() if line.strip().startswith("JPEG:")), "unknown")

def verify_runtime() -> dict[str, str]:
    opencv = importlib.metadata.version("opencv-python")
    numpy_version = np.__version__
    jpeg = jpeg_implementation()
    if opencv != "4.13.0.92" or numpy_version != "2.3.5" or "3.1.2-70" not in jpeg:
        raise ValueError(f"frozen decoder dependency mismatch: opencv={opencv}, numpy={numpy_version}, jpeg={jpeg}")
    return {"opencv_python": opencv, "cv2": cv2.__version__, "numpy": numpy_version, "jpeg": jpeg}

def inventory(root: Path, paths: Iterable[str], directory: str) -> dict[str, Any]:
    records: list[bytes] = []
    total = 0
    for rgb in paths:
        stem = rgb[len("RGB/"):-4]
        rel = f"{directory}/{stem}.{ 'jpg' if directory == 'RGB' else 'png'}"
        file = root / rel
        if not file.is_file(): raise ValueError(f"missing modality: {rel}")
        content = file.read_bytes(); total += len(content)
        records.append(rel.encode() + b"\0" + str(len(content)).encode() + b"\0" + sha256_bytes(content).encode() + b"\n")
    return {"sha256": sha256_bytes(b"".join(sorted(records))), "files": len(records), "bytes": total,
            "serialization": "relative_path_utf8 || NUL || decimal_size_ascii || NUL || lowercase_file_sha256_ascii || LF"}

def load_samples(root: Path, paths: Iterable[str]) -> list[Sample]:
    result=[]
    for path in paths:
        group,mine=parse_path(path); stem=path[4:-4]
        rgb=cv2.imread(str(root/path), cv2.IMREAD_COLOR)
        depth=cv2.imread(str(root/f"Depth16/{stem}.png"), cv2.IMREAD_UNCHANGED)
        label=cv2.imread(str(root/f"Label/{stem}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or label is None: raise ValueError(f"unreadable modality: {path}")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or depth.dtype != np.uint16 or depth.ndim != 2 or label.dtype != np.uint8 or label.ndim != 2:
            raise ValueError(f"invalid modality dtype/dimension: {path}")
        if rgb.shape[:2] != depth.shape or depth.shape != label.shape or int(label.max()) > 15:
            raise ValueError(f"invalid modality dimensions/label range: {path}")
        pixels=int(label.size); presence=tuple(int(np.any(label == c)) for c in range(1,16)); cp=tuple(int(np.count_nonzero(label==c)) for c in range(1,16))
        valid=int(np.count_nonzero(depth)); b,g,r=(rgb[...,0].astype(np.int64),rgb[...,1].astype(np.int64),rgb[...,2].astype(np.int64)); yn=int((299*r+587*g+114*b).sum())
        result.append(Sample(path,group,mine,pixels,presence,cp,int(np.all(label==0)),valid,yn,bin_index(Fraction(valid,pixels),DEPTH_BINS),bin_index(Fraction(yn,1000*255*pixels),LUMA_BINS),class_bin(sum(presence))))
    return result

def totals(samples: Iterable[Sample]) -> dict[str, Any]:
    result = {"images": 0, "pixels": 0, "mine": {m: 0 for m in MINES}, "presence": [0] * 15, "class_pixels": [0] * 15, "background": 0, "depth_valid": 0, "y_num": 0, "depth_hist": [0] * 6, "luma_hist": [0] * 8, "class_hist": [0] * 8}
    for sample in samples:
        result["images"] += 1; result["pixels"] += sample.pixels; result["mine"][sample.mine] += 1
        result["background"] += sample.background; result["depth_valid"] += sample.depth_valid; result["y_num"] += sample.y_num
        result["depth_hist"][sample.depth_bin] += 1; result["luma_hist"][sample.luma_bin] += 1; result["class_hist"][sample.class_count_bin] += 1
        for index in range(15): result["presence"][index] += sample.presence[index]; result["class_pixels"][index] += sample.class_pixels[index]
    return result

def zero_totals() -> dict[str, Any]:
    return {"images": 0, "pixels": 0, "mine": {m: 0 for m in MINES}, "presence": [0] * 15, "class_pixels": [0] * 15, "background": 0, "depth_valid": 0, "y_num": 0, "depth_hist": [0] * 6, "luma_hist": [0] * 8, "class_hist": [0] * 8}

def add_totals(left: dict[str, Any], right: dict[str, Any], sign: int = 1) -> dict[str, Any]:
    """Naive/reference nested-total update used by tests and manifest generation."""
    return {"images": left["images"] + sign * right["images"], "pixels": left["pixels"] + sign * right["pixels"], "mine": {m: left["mine"][m] + sign * right["mine"][m] for m in MINES}, "presence": [left["presence"][i] + sign * right["presence"][i] for i in range(15)], "class_pixels": [left["class_pixels"][i] + sign * right["class_pixels"][i] for i in range(15)], "background": left["background"] + sign * right["background"], "depth_valid": left["depth_valid"] + sign * right["depth_valid"], "y_num": left["y_num"] + sign * right["y_num"], "depth_hist": [left["depth_hist"][i] + sign * right["depth_hist"][i] for i in range(6)], "luma_hist": [left["luma_hist"][i] + sign * right["luma_hist"][i] for i in range(8)], "class_hist": [left["class_hist"][i] + sign * right["class_hist"][i] for i in range(8)]}

def aggregate_groups(groups: dict[str, list[Sample]]) -> dict[str, dict[str, Any]]:
    return {group: totals(samples) for group, samples in groups.items()}

def e(v: int, t: int) -> Fraction: return Fraction(abs(5*v-t), max(t,1))
def d(a:int,b:int,c:int,dd:int)->Fraction: return Fraction(abs(a*dd-c*b), b*dd) if b and dd else Fraction(0)
def score_from_totals(val: dict[str, Any], all_total: dict[str, Any]) -> tuple[Fraction, ...]:
    """Clear reference implementation of the frozen six-level objective."""
    mean=lambda values: sum(values, Fraction(0)) / len(values)
    parts=[d(val['depth_valid'],val['pixels'],all_total['depth_valid'],all_total['pixels']),mean([e(val['depth_hist'][i],all_total['depth_hist'][i]) for i in range(6)]),d(val['y_num'],1000*255*val['pixels'],all_total['y_num'],1000*255*all_total['pixels']),mean([e(val['luma_hist'][i],all_total['luma_hist'][i]) for i in range(8)]),mean([e(val['class_hist'][i],all_total['class_hist'][i]) for i in range(8)])]
    return mean([e(val['mine'][m],all_total['mine'][m]) for m in MINES]),mean([e(val['presence'][i],all_total['presence'][i]) for i in range(15)]),mean([e(val['class_pixels'][i],all_total['class_pixels'][i]) for i in range(15)]),e(val['background'],all_total['background']),mean(parts),Fraction(abs(val['images']-TARGET),TARGET)


def fixed_bins() -> dict[str, list[dict[str, Any]]]:
    """Return the protocol's canonical, machine-auditable bin definitions."""
    def intervals(edges: tuple[Fraction, ...]) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "lower": fraction_text(edges[index]),
                "upper": fraction_text(edges[index + 1]),
                "lower_closed": True,
                "upper_closed": index == len(edges) - 2,
            }
            for index in range(len(edges) - 1)
        ]
    return {
        "depth16_valid_ratio": intervals(DEPTH_BINS),
        "rgb_normalized_mean_luma": intervals(LUMA_BINS),
        "foreground_class_count": [
            {"index": 0, "values": [0]},
            {"index": 1, "values": [1]},
            {"index": 2, "values": [2]},
            {"index": 3, "values": [3]},
            {"index": 4, "values": [4]},
            {"index": 5, "values": [5, 6]},
            {"index": 6, "values": [7, 8, 9]},
            {"index": 7, "values": [10, 11, 12, 13, 14, 15]},
        ],
    }


def statistic_definitions() -> dict[str, Any]:
    """Return complete frozen definitions for every manifest statistic."""
    return {
        "label": {
            "decode": "cv2.IMREAD_UNCHANGED; two-dimensional uint8",
            "allowed_values": list(range(16)),
            "background_label": 0,
            "foreground_classes": list(range(1, 16)),
            "image_presence": "1 iff at least one pixel equals class c; classes 1..15",
            "pixel_count": "number of pixels equal to class c; classes 1..15",
            "all_background_image": "1 iff every Label pixel equals 0",
            "foreground_class_count": "number of present classes among 1..15",
        },
        "depth16": {
            "decode": "cv2.IMREAD_UNCHANGED; two-dimensional uint16",
            "valid_pixel": "depth16 > 0",
            "per_image_valid_ratio": "valid_pixel_count / total_pixel_count as exact Fraction",
            "quantized_depth_directory_used": False,
        },
        "rgb_luma": {
            "decode": "cv2.IMREAD_COLOR uint8 BGR, explicitly reordered to RGB",
            "integer_numerator": "Y_num=299*R+587*G+114*B",
            "per_image_normalized_mean": "sum(Y_num)/(1000*255*pixel_count) as exact Fraction",
            "icc_gamma_resize_normalization": False,
        },
        "class_count": {
            "definition": "number of foreground classes 1..15 with image_presence=1",
            "range": [0, 15],
        },
        "bins": fixed_bins(),
    }


def objective_details(val: dict[str, Any], total: dict[str, Any]) -> dict[str, Any]:
    """Serialize every exact objective component, including all histogram bins."""
    mean = lambda values: sum(values, Fraction(0)) / len(values)
    mine = {mine: e(val["mine"][mine], total["mine"][mine]) for mine in MINES}
    presence = {str(index + 1): e(val["presence"][index], total["presence"][index]) for index in range(15)}
    pixels = {str(index + 1): e(val["class_pixels"][index], total["class_pixels"][index]) for index in range(15)}
    depth_hist = [e(val["depth_hist"][index], total["depth_hist"][index]) for index in range(6)]
    luma_hist = [e(val["luma_hist"][index], total["luma_hist"][index]) for index in range(8)]
    class_hist = [e(val["class_hist"][index], total["class_hist"][index]) for index in range(8)]
    aux = {
        "depth16_total_valid_ratio": d(val["depth_valid"], val["pixels"], total["depth_valid"], total["pixels"]),
        "depth16_valid_ratio_histogram": mean(depth_hist),
        "rgb_total_normalized_mean_luma": d(val["y_num"], 255000 * val["pixels"], total["y_num"], 255000 * total["pixels"]),
        "rgb_normalized_mean_luma_histogram": mean(luma_hist),
        "foreground_class_count_histogram": mean(class_hist),
    }
    six = score_from_totals(val, total)
    return {
        "six_tuple": [fraction_text(value) for value in six],
        "levels": {
            "mine": fraction_text(mean(mine.values())),
            "presence": fraction_text(mean(presence.values())),
            "pixel": fraction_text(mean(pixels.values())),
            "background": fraction_text(e(val["background"], total["background"])),
            "aux": fraction_text(mean(aux.values())),
            "size": fraction_text(Fraction(abs(val["images"] - TARGET), TARGET)),
        },
        "subitems": {
            "mine": {key: fraction_text(value) for key, value in mine.items()},
            "presence": {key: fraction_text(value) for key, value in presence.items()},
            "pixel": {key: fraction_text(value) for key, value in pixels.items()},
            "background": fraction_text(e(val["background"], total["background"])),
            "aux": {key: fraction_text(value) for key, value in aux.items()},
            "depth16_valid_ratio_histogram_bins": [fraction_text(value) for value in depth_hist],
            "rgb_normalized_mean_luma_histogram_bins": [fraction_text(value) for value in luma_hist],
            "foreground_class_count_histogram_bins": [fraction_text(value) for value in class_hist],
        },
    }


def mine_side_counts(groups: dict[str, list[Sample]], selected: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"train_dev": {}, "val_dev": {}}
    for side, side_groups in (("train_dev", set(groups) - selected), ("val_dev", selected)):
        for mine in MINES:
            mine_groups = [group for group in side_groups if groups[group][0].mine == mine]
            result[side][mine] = {
                "images": sum(len(groups[group]) for group in mine_groups),
                "groups": len(mine_groups),
            }
    return result


def set_relationships(
    official_train: Iterable[str], official_test: Iterable[str], train_dev: Iterable[str], val_dev: Iterable[str]
) -> dict[str, Any]:
    """Describe closure, duplicates, and every required sample/group intersection."""
    ot, ox, td, vd = tuple(official_train), tuple(official_test), tuple(train_dev), tuple(val_dev)
    sample_sets = {"official_train": set(ot), "official_test": set(ox), "train_dev": set(td), "val_dev": set(vd)}
    group_sets = {name: {parse_path(path)[0] for path in paths} for name, paths in (("official_train", ot), ("official_test", ox), ("train_dev", td), ("val_dev", vd))}

    def overlap(left: set[str], right: set[str]) -> dict[str, Any]:
        items = sorted(left & right, key=utf8_key)
        return {"count": len(items), "items": items}

    def closure(left: set[str], right: set[str], expected: set[str]) -> dict[str, Any]:
        actual = left | right
        missing = sorted(expected - actual, key=utf8_key)
        unexpected = sorted(actual - expected, key=utf8_key)
        return {"closed": not missing and not unexpected, "missing_count": len(missing), "missing": missing, "unexpected_count": len(unexpected), "unexpected": unexpected}

    sample_duplicates = {"official_train": len(ot) - len(set(ot)), "official_test": len(ox) - len(set(ox)), "train_dev": len(td) - len(set(td)), "val_dev": len(vd) - len(set(vd))}
    return {
        "samples": {
            "duplicates": sample_duplicates,
            "dev_union_vs_official_train": closure(sample_sets["train_dev"], sample_sets["val_dev"], sample_sets["official_train"]),
            "train_dev__val_dev": overlap(sample_sets["train_dev"], sample_sets["val_dev"]),
            "official_train__official_test": overlap(sample_sets["official_train"], sample_sets["official_test"]),
            "train_dev__official_test": overlap(sample_sets["train_dev"], sample_sets["official_test"]),
            "val_dev__official_test": overlap(sample_sets["val_dev"], sample_sets["official_test"]),
        },
        "groups": {
            "dev_union_vs_official_train": closure(group_sets["train_dev"], group_sets["val_dev"], group_sets["official_train"]),
            "train_dev__val_dev": overlap(group_sets["train_dev"], group_sets["val_dev"]),
            "official_train__official_test": overlap(group_sets["official_train"], group_sets["official_test"]),
            "train_dev__official_test": overlap(group_sets["train_dev"], group_sets["official_test"]),
            "val_dev__official_test": overlap(group_sets["val_dev"], group_sets["official_test"]),
        },
    }


def rare_class_list(samples: list[Sample]) -> list[dict[str, Any]]:
    total = totals(samples)
    present_groups = {
        class_id: len({sample.group for sample in samples if sample.presence[class_id - 1]})
        for class_id in range(1, 16)
    }
    return [
        {
            "class_id": class_id,
            "images": total["presence"][class_id - 1],
            "groups": present_groups[class_id],
            "pixels": total["class_pixels"][class_id - 1],
            "cannot_bilateral_cover": present_groups[class_id] == 1,
        }
        for class_id in range(1, 16)
        if total["presence"][class_id - 1] < 5 or present_groups[class_id] < 2
    ]


def threshold_evaluation(samples: list[Sample], val_total: dict[str, Any], total: dict[str, Any], groups: dict[str, list[Sample]]) -> tuple[dict[str, Any], list[str]]:
    """Evaluate review-only anomaly thresholds; these never make the candidate invalid."""
    objective = objective_details(val_total, total)
    group_presence = {class_id: len({sample.group for sample in samples if sample.presence[class_id - 1]}) for class_id in range(1, 16)}
    presence_zero = []
    presence_errors = []
    pixel_errors = []
    warnings: list[str] = []
    for class_id in range(1, 16):
        applicable = total["presence"][class_id - 1] >= 5 and group_presence[class_id] >= 2
        zero_triggered = applicable and val_total["presence"][class_id - 1] == 0
        presence_value = e(val_total["presence"][class_id - 1], total["presence"][class_id - 1])
        pixel_value = e(val_total["class_pixels"][class_id - 1], total["class_pixels"][class_id - 1])
        presence_triggered = presence_value > Fraction(1, 2)
        pixel_triggered = pixel_value > Fraction(1, 2)
        presence_zero.append({"class_id": class_id, "applicable": applicable, "triggered": zero_triggered, "official_train_images": total["presence"][class_id - 1], "official_train_groups": group_presence[class_id], "val_images": val_total["presence"][class_id - 1]})
        presence_errors.append({"class_id": class_id, "value": fraction_text(presence_value), "threshold": "1/2", "operator": ">", "triggered": presence_triggered})
        pixel_errors.append({"class_id": class_id, "value": fraction_text(pixel_value), "threshold": "1/2", "operator": ">", "triggered": pixel_triggered})
        if zero_triggered: warnings.append(f"class_{class_id:02d}_eligible_presence_zero")
        if presence_triggered: warnings.append(f"class_{class_id:02d}_presence_error_gt_1/2:{fraction_text(presence_value)}")
        if pixel_triggered: warnings.append(f"class_{class_id:02d}_pixel_error_gt_1/2:{fraction_text(pixel_value)}")
    background_value = e(val_total["background"], total["background"])
    background = {"value": fraction_text(background_value), "threshold": "1/2", "operator": ">", "triggered": background_value > Fraction(1, 2)}
    if background["triggered"]: warnings.append(f"all_background_error_gt_1/2:{fraction_text(background_value)}")
    aux = []
    aux_values = objective["subitems"]["aux"]
    for name in ("depth16_total_valid_ratio", "depth16_valid_ratio_histogram", "rgb_total_normalized_mean_luma", "rgb_normalized_mean_luma_histogram", "foreground_class_count_histogram"):
        value = Fraction(aux_values[name])
        triggered = value > Fraction(1, 3)
        aux.append({"name": name, "value": fraction_text(value), "threshold": "1/3", "operator": ">", "triggered": triggered})
        if triggered: warnings.append(f"aux_{name}_gt_1/3:{fraction_text(value)}")
    deviation = abs(val_total["images"] - TARGET)
    limit = max(5, max(len(group_samples) for group_samples in groups.values()))
    size = {"value": deviation, "threshold": limit, "operator": ">", "triggered": deviation > limit, "target_val_images": TARGET, "val_images": val_total["images"]}
    if size["triggered"]: warnings.append(f"val_image_deviation_gt_limit:{deviation}>{limit}")
    checks = {
        "eligible_class_presence_zero": presence_zero,
        "class_presence_error_gt_1_2": presence_errors,
        "class_pixel_error_gt_1_2": pixel_errors,
        "all_background_error_gt_1_2": background,
        "aux_component_error_gt_1_3": aux,
        "val_image_deviation_gt_limit": size,
    }
    triggered = bool(warnings)
    return {"requires_sol_review": triggered, "review_status": "required" if triggered else "not_required", "checks": checks}, warnings


def hard_constraint_checks(
    relationships: dict[str, Any],
    mine_counts: dict[str, Any],
    selected_matches_optimizer: bool,
    *,
    official_train_identity: bool = True,
    official_test_identity: bool = True,
    frozen_parameters: bool = True,
    modalities_valid: bool = True,
    mine_group_minimum: bool = True,
    optimizer_converged: bool = True,
) -> dict[str, bool]:
    """Record every frozen precondition and membership hard constraint explicitly."""
    samples, groups = relationships["samples"], relationships["groups"]
    checks = {
        "official_train_frozen_identity": official_train_identity,
        "official_test_frozen_identity": official_test_identity,
        "frozen_seed_and_target": frozen_parameters,
        "all_train_modalities_valid": modalities_valid,
        "each_mine_has_at_least_two_official_train_groups": mine_group_minimum,
        "group_unsplit": groups["train_dev__val_dev"]["count"] == 0,
        "sample_set_closed": samples["dev_union_vs_official_train"]["closed"] and samples["duplicates"]["train_dev"] == 0 and samples["duplicates"]["val_dev"] == 0 and samples["train_dev__val_dev"]["count"] == 0,
        "group_set_closed": groups["dev_union_vs_official_train"]["closed"] and groups["train_dev__val_dev"]["count"] == 0,
        "official_test_samples_isolated": samples["official_train__official_test"]["count"] == 0 and samples["train_dev__official_test"]["count"] == 0 and samples["val_dev__official_test"]["count"] == 0,
        "official_test_groups_isolated": groups["official_train__official_test"]["count"] == 0 and groups["train_dev__official_test"]["count"] == 0 and groups["val_dev__official_test"]["count"] == 0,
        "mine_bilateral_images": all(mine_counts[side][mine]["images"] > 0 for side in ("train_dev", "val_dev") for mine in MINES),
        "mine_bilateral_groups": all(mine_counts[side][mine]["groups"] > 0 for side in ("train_dev", "val_dev") for mine in MINES),
        "fixed_unique_candidate_membership": selected_matches_optimizer,
        "optimizer_converged": optimizer_converged,
    }
    checks["all_pass"] = all(checks.values())
    return checks
def score(chosen: set[str], groups: dict[str,list[Sample]], all_total: dict[str,Any]) -> tuple[Fraction,...]:
    vectors=aggregate_groups(groups); val=zero_totals()
    for group in chosen: val=add_totals(val,vectors[group])
    return score_from_totals(val,all_total)
def feasible(chosen:set[str],groups:dict[str,list[Sample]])->bool:
    return all(any(groups[group][0].mine==mine for group in chosen) and any(groups[group][0].mine==mine for group in groups if group not in chosen) for mine in MINES)
def feasible_totals(val: dict[str, Any], all_total: dict[str, Any]) -> bool:
    return all(val['mine'][mine] > 0 and val['mine'][mine] < all_total['mine'][mine] for mine in MINES)

# Flat vector layout. Every slot is an unbounded Python int:
# images, pixels, mines[6], presence[15], class_pixels[15], background,
# depth_valid, y_num, depth_hist[6], luma_hist[8], class_count_hist[8].
V_IMAGES, V_PIXELS = 0, 1
V_MINE = slice(2, 8)
V_PRESENCE = slice(8, 23)
V_CLASS_PIXELS = slice(23, 38)
V_BACKGROUND, V_DEPTH_VALID, V_Y_NUM = 38, 39, 40
V_DEPTH_HIST = slice(41, 47)
V_LUMA_HIST = slice(47, 55)
V_CLASS_HIST = slice(55, 63)
VECTOR_SIZE = 63
FlatVector = tuple[int, ...]
Move = tuple[int, int, int]  # type rank, removed group index (-1), added group index (-1)

def flatten_totals(value: dict[str, Any]) -> FlatVector:
    return tuple([value['images'], value['pixels'], *(value['mine'][m] for m in MINES), *value['presence'], *value['class_pixels'], value['background'], value['depth_valid'], value['y_num'], *value['depth_hist'], *value['luma_hist'], *value['class_hist']])

def aggregate_group_vectors(groups: dict[str, list[Sample]], order: list[str]) -> list[FlatVector]:
    return [flatten_totals(totals(groups[group])) for group in order]

def add_vectors(left: FlatVector, right: FlatVector, sign: int = 1) -> FlatVector:
    return tuple(a + sign*b for a,b in zip(left,right))

def move_vector(current: FlatVector, vectors: list[FlatVector], move: Move) -> FlatVector:
    _, removed, added = move
    return tuple(value - (vectors[removed][i] if removed >= 0 else 0) + (vectors[added][i] if added >= 0 else 0) for i,value in enumerate(current))

def flat_score(val: FlatVector, total: FlatVector) -> tuple[Fraction, ...]:
    """Exact flat-vector score, intentionally mirroring score_from_totals."""
    mean=lambda values: sum(values, Fraction(0)) / len(values)
    aux=(d(val[V_DEPTH_VALID],val[V_PIXELS],total[V_DEPTH_VALID],total[V_PIXELS]),
         mean([e(val[i],total[i]) for i in range(V_DEPTH_HIST.start,V_DEPTH_HIST.stop)]),
         d(val[V_Y_NUM],255000*val[V_PIXELS],total[V_Y_NUM],255000*total[V_PIXELS]),
         mean([e(val[i],total[i]) for i in range(V_LUMA_HIST.start,V_LUMA_HIST.stop)]),
         mean([e(val[i],total[i]) for i in range(V_CLASS_HIST.start,V_CLASS_HIST.stop)]))
    return (mean([e(val[i],total[i]) for i in range(V_MINE.start,V_MINE.stop)]),
            mean([e(val[i],total[i]) for i in range(V_PRESENCE.start,V_PRESENCE.stop)]),
            mean([e(val[i],total[i]) for i in range(V_CLASS_PIXELS.start,V_CLASS_PIXELS.stop)]),
            e(val[V_BACKGROUND],total[V_BACKGROUND]), mean(aux),
            Fraction(abs(val[V_IMAGES]-TARGET),TARGET))

@dataclass
class OptimizationStats:
    enumerated_moves: int = 0
    legal_moves: int = 0
    full_fraction_scores: int = 0
    integer_level_keys: int = 0
    aux_fraction_keys: int = 0

def _integer_weights(total: FlatVector, indices: range) -> tuple[int, ...]:
    """Scale a sum of E terms to an order-equivalent integer exactly."""
    import math
    denominators=[max(total[i],1) for i in indices]
    common=1
    for denominator in denominators: common=math.lcm(common,denominator)
    return tuple(common//denominator for denominator in denominators)

def _integer_error_key(val: FlatVector, total: FlatVector, indices: range, weights: tuple[int,...]) -> int:
    return sum(abs(5*val[i]-total[i])*weight for i,weight in zip(indices,weights))

def _aux_key(val: FlatVector, total: FlatVector) -> Fraction:
    mean=lambda values: sum(values,Fraction(0))/len(values)
    return (d(val[V_DEPTH_VALID],val[V_PIXELS],total[V_DEPTH_VALID],total[V_PIXELS])
            + mean([e(val[i],total[i]) for i in range(V_DEPTH_HIST.start,V_DEPTH_HIST.stop)])
            + d(val[V_Y_NUM],255000*val[V_PIXELS],total[V_Y_NUM],255000*total[V_PIXELS])
            + mean([e(val[i],total[i]) for i in range(V_LUMA_HIST.start,V_LUMA_HIST.stop)])
            + mean([e(val[i],total[i]) for i in range(V_CLASS_HIST.start,V_CLASS_HIST.stop)]))

def _fast_keys(val: FlatVector, total: FlatVector, weights: tuple[tuple[int,...],...], stats: OptimizationStats | None = None) -> tuple[int,int,int,int,Fraction,int]:
    if stats: stats.integer_level_keys += 4; stats.aux_fraction_keys += 1
    return (_integer_error_key(val,total,range(V_MINE.start,V_MINE.stop),weights[0]),
            _integer_error_key(val,total,range(V_PRESENCE.start,V_PRESENCE.stop),weights[1]),
            _integer_error_key(val,total,range(V_CLASS_PIXELS.start,V_CLASS_PIXELS.stop),weights[2]),
            abs(5*val[V_BACKGROUND]-total[V_BACKGROUND]), _aux_key(val,total),
            abs(val[V_IMAGES]-TARGET))

def _candidate_level(current: FlatVector, vectors: list[FlatVector], move: Move, total: FlatVector, indices: range, weights: tuple[int,...]) -> int:
    _,removed,added=move
    result=0
    for i,weight in zip(indices,weights):
        value=current[i]-(vectors[removed][i] if removed>=0 else 0)+(vectors[added][i] if added>=0 else 0)
        result += abs(5*value-total[i])*weight
    return result

def _candidate_aux(current: FlatVector, vectors: list[FlatVector], move: Move, total: FlatVector) -> Fraction:
    return _aux_key(move_vector(current,vectors,move),total)

def _moves(selected: list[int], train: list[int]):
    for group in train: yield (0,-1,group)
    for group in selected: yield (1,group,-1)
    for out in selected:
        for inn in train: yield (2,out,inn)

def _move_tie(move: Move, order: list[str]) -> tuple[int,int,int,bytes,bytes]:
    move_type,out,inn=move
    return (move_type,out,inn,utf8_key(order[out]) if out>=0 else b'',utf8_key(order[inn]) if inn>=0 else b'')

def _move_feasible(current: FlatVector, vectors: list[FlatVector], move: Move, total: FlatVector) -> bool:
    _,removed,added=move
    for i in range(V_MINE.start,V_MINE.stop):
        value=current[i]-(vectors[removed][i] if removed>=0 else 0)+(vectors[added][i] if added>=0 else 0)
        if value <= 0 or value >= total[i]: return False
    return True

def _initial_flat(groups: dict[str,list[Sample]]) -> tuple[list[str],list[FlatVector],FlatVector,set[int],FlatVector]:
    order=sorted(groups,key=group_order); vectors=aggregate_group_vectors(groups,order)
    total=tuple(sum(vector[i] for vector in vectors) for i in range(VECTOR_SIZE))
    chosen:set[int]=set(); val=(0,)*VECTOR_SIZE
    for mine_index,mine in enumerate(MINES):
        candidates=[index for index,vector in enumerate(vectors) if vector[V_MINE.start+mine_index]]
        if len(candidates)<2: raise ValueError(f'mine {mine} has fewer than two groups')
        chosen.add(candidates[0]); val=add_vectors(val,vectors[candidates[0]])
    for index in range(len(order)):
        if index in chosen: continue
        candidate=add_vectors(val,vectors[index])
        if all(0<candidate[i]<total[i] for i in range(V_MINE.start,V_MINE.stop)) and abs(candidate[V_IMAGES]-TARGET)<abs(val[V_IMAGES]-TARGET):
            chosen.add(index);val=candidate
    return order,vectors,total,chosen,val

def _initial_naive(groups:dict[str,list[Sample]]) -> tuple[list[str],dict[str,dict[str,Any]],dict[str,Any],set[str],dict[str,Any]]:
    """Reference construction of the protocol's single deterministic initial state."""
    order=sorted(groups,key=group_order); vectors=aggregate_groups(groups); all_total=zero_totals()
    for vector in vectors.values(): all_total=add_totals(all_total,vector)
    chosen:set[str]=set(); val=zero_totals()
    for mine in MINES:
        candidates=[group for group in order if vectors[group]['mine'][mine]]
        if len(candidates)<2: raise ValueError(f'mine {mine} has fewer than two groups')
        chosen.add(candidates[0]); val=add_totals(val,vectors[candidates[0]])
    for group in order:
        if group in chosen: continue
        candidate=add_totals(val,vectors[group])
        if feasible_totals(candidate,all_total) and abs(candidate['images']-TARGET)<abs(val['images']-TARGET):
            chosen.add(group);val=candidate
    return order,vectors,all_total,chosen,val

def enforce_search_guard(accepted: int) -> None:
    """Reject an additional improving move after 10,000 accepted moves."""
    if accepted >= 10000:
        raise ValueError("local search exceeded 10000 accepted moves")


def optimize_naive(groups:dict[str,list[Sample]], *, trace:list[tuple[int,str|None,str|None]]|None=None, stats:OptimizationStats|None=None)->tuple[set[str],int,str]:
    """Allocation-heavy reference optimizer matching the frozen protocol literally."""
    order,vectors,all_total,chosen,val=_initial_naive(groups)
    accepted=0; rank={group:index for index,group in enumerate(order)}
    while True:
        current=score_from_totals(val,all_total)
        if stats: stats.full_fraction_scores += 1
        best=None; train=[group for group in order if group not in chosen]; selected=[group for group in order if group in chosen]
        for move_type in range(3):
            moves=((None,group) for group in train) if move_type==0 else ((group,None) for group in selected) if move_type==1 else ((out,inn) for out in selected for inn in train)
            for out,inn in moves:
                if stats: stats.enumerated_moves += 1
                candidate=add_totals(val,vectors[inn]) if move_type==0 else add_totals(val,vectors[out],-1) if move_type==1 else add_totals(add_totals(val,vectors[out],-1),vectors[inn])
                if not feasible_totals(candidate,all_total): continue
                if stats: stats.legal_moves += 1; stats.full_fraction_scores += 1
                candidate_score=score_from_totals(candidate,all_total)
                if candidate_score>=current: continue
                tie=(move_type,rank[out] if out else -1,rank[inn] if inn else -1,utf8_key(out or ''),utf8_key(inn or ''))
                item=(candidate_score,tie,out,inn,candidate)
                if best is None or (item[0],item[1])<(best[0],best[1]): best=item
        if best is None:return chosen,accepted,'no_strict_improvement'
        enforce_search_guard(accepted)
        _,_,out,inn,val=best
        if out is not None: chosen.remove(out)
        if inn is not None: chosen.add(inn)
        if trace is not None: trace.append((0 if out is None else 1 if inn is None else 2,out,inn))
        accepted+=1

def optimize(groups:dict[str,list[Sample]], *, trace:list[tuple[int,str|None,str|None]]|None=None, stats:OptimizationStats|None=None)->tuple[set[str],int,str]:
    """Exact low-allocation, lexicographically filtered best-improvement search."""
    order,vectors,total,chosen,val=_initial_flat(groups)
    ranges=(range(V_MINE.start,V_MINE.stop),range(V_PRESENCE.start,V_PRESENCE.stop),range(V_CLASS_PIXELS.start,V_CLASS_PIXELS.stop))
    weights=tuple(_integer_weights(total,indices) for indices in ranges)
    accepted=0
    while True:
        current_keys=_fast_keys(val,total,weights,stats); best_keys=current_keys; best_move:Move|None=None; best_tie=None
        train=[i for i in range(len(order)) if i not in chosen]; selected=[i for i in range(len(order)) if i in chosen]
        for move in _moves(selected,train):
            if stats: stats.enumerated_moves += 1
            if not _move_feasible(val,vectors,move,total): continue
            if stats: stats.legal_moves += 1; stats.integer_level_keys += 1
            key0=_candidate_level(val,vectors,move,total,ranges[0],weights[0])
            if key0>best_keys[0]: continue
            if stats: stats.integer_level_keys += 1
            key1=_candidate_level(val,vectors,move,total,ranges[1],weights[1])
            if key0==best_keys[0] and key1>best_keys[1]: continue
            if stats: stats.integer_level_keys += 1
            key2=_candidate_level(val,vectors,move,total,ranges[2],weights[2])
            if (key0,key1)==best_keys[:2] and key2>best_keys[2]: continue
            _,removed,added=move
            background=val[V_BACKGROUND]-(vectors[removed][V_BACKGROUND] if removed>=0 else 0)+(vectors[added][V_BACKGROUND] if added>=0 else 0)
            key3=abs(5*background-total[V_BACKGROUND])
            if stats: stats.integer_level_keys += 1
            if (key0,key1,key2)==best_keys[:3] and key3>best_keys[3]: continue
            if stats: stats.aux_fraction_keys += 1
            key4=_candidate_aux(val,vectors,move,total)
            if (key0,key1,key2,key3)==best_keys[:4] and key4>best_keys[4]: continue
            images=val[V_IMAGES]-(vectors[removed][V_IMAGES] if removed>=0 else 0)+(vectors[added][V_IMAGES] if added>=0 else 0)
            keys=(key0,key1,key2,key3,key4,abs(images-TARGET)); tie=_move_tie(move,order)
            if keys<best_keys or (keys==best_keys and best_move is not None and tie<best_tie):
                best_keys=keys;best_move=move;best_tie=tie
        if best_move is None:return {order[i] for i in chosen},accepted,'no_strict_improvement'
        enforce_search_guard(accepted)
        val=move_vector(val,vectors,best_move);_,removed,added=best_move
        if removed>=0:chosen.remove(removed)
        if added>=0:chosen.add(added)
        if trace is not None: trace.append((best_move[0],order[removed] if removed>=0 else None,order[added] if added>=0 else None))
        accepted+=1

def output_lines(groups:dict[str,list[Sample]], selected:set[str])->bytes:
    return ("\n".join(s.path for g in sorted(selected,key=group_order) for s in sorted(groups[g],key=lambda x:utf8_key(x.path)))+"\n").encode()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-val-images", type=int, default=TARGET)
    args = parser.parse_args(argv)
    if args.seed != SEED or args.target_val_images != TARGET:
        parser.error("only frozen --seed 1181652136 and --target-val-images 319 are accepted")
    if args.output_root.exists():
        parser.error("output root already exists; refusing overwrite")

    train_raw, test_raw = args.official_train.read_bytes(), args.official_test.read_bytes()
    train_paths, test_paths = parse_official_bytes(train_raw), parse_official_bytes(test_raw)
    train_groups = {parse_path(path)[0] for path in train_paths}
    test_groups = {parse_path(path)[0] for path in test_paths}
    if (sha256_bytes(train_raw), len(train_paths), len(train_groups)) != (TRAIN_HASH, TRAIN_COUNT, TRAIN_GROUPS) or (sha256_bytes(test_raw), len(test_paths), len(test_groups)) != (TEST_HASH, TEST_COUNT, TEST_GROUPS):
        raise ValueError("official input hash/count/group mismatch")
    if set(train_paths) & set(test_paths) or train_groups & test_groups:
        raise ValueError("official train/test overlap")

    runtime = verify_runtime()
    repo = Path(__file__).resolve().parents[2]
    commit = git_commit(repo)
    if not git_object_is_commit(repo, commit):
        raise ValueError("generator requires an existing full lowercase Git commit object")
    sources = tool_sources(repo)
    samples = load_samples(args.dataset_root, train_paths)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)
    selected, moves, reason = optimize(groups)
    train_dev_paths = tuple(sample.path for group in sorted(set(groups) - selected, key=group_order) for sample in sorted(groups[group], key=lambda item: utf8_key(item.path)))
    val_dev_paths = tuple(sample.path for group in sorted(selected, key=group_order) for sample in sorted(groups[group], key=lambda item: utf8_key(item.path)))
    train_bytes = ("\n".join(train_dev_paths) + "\n").encode("utf-8")
    val_bytes = ("\n".join(val_dev_paths) + "\n").encode("utf-8")
    files = {"train-dev.txt": train_bytes, "val-dev.txt": val_bytes, "official-test.txt": test_raw}

    total = totals(samples)
    val_total = totals(sample for group in selected for sample in groups[group])
    train_total = totals(sample for group in set(groups) - selected for sample in groups[group])
    relationships = set_relationships(train_paths, test_paths, train_dev_paths, val_dev_paths)
    per_mine = mine_side_counts(groups, selected)
    thresholds, warnings = threshold_evaluation(samples, val_total, total, groups)
    hard = hard_constraint_checks(relationships, per_mine, True)
    if not hard["all_pass"]:
        raise ValueError("generated membership violates frozen hard constraints")

    inventory_value = {name: inventory(args.dataset_root, train_paths, name) for name in ("RGB", "Depth16", "Label")}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "seed_source_string": SEED_STRING,
        "seed_source_sha256": SEED_HASH,
        "seed_uint32_big_endian": SEED,
        "target_val_images": TARGET,
        "generator_git_commit": commit,
        "tool_sources": sources,
        "generator_command": "python tools/splits/create_museg_dev_split.py --dataset-root <dataset-root> --official-train <official-train> --official-test <official-test> --output-root <output-root> --seed 1181652136 --target-val-images 319",
        "dependencies": runtime,
        "official": {
            "train": {"logical_name": "train.txt", "sha256": sha256_bytes(train_raw), "samples": len(train_paths), "groups": len(train_groups)},
            "test": {"logical_name": "test.txt", "sha256": sha256_bytes(test_raw), "samples": len(test_paths), "groups": len(test_groups)},
            "sample_overlap": relationships["samples"]["official_train__official_test"],
            "group_overlap": relationships["groups"]["official_train__official_test"],
        },
        "inventory": inventory_value,
        "outputs": {
            name: {
                "sha256": sha256_bytes(content),
                "samples": len(train_dev_paths) if name == "train-dev.txt" else len(val_dev_paths) if name == "val-dev.txt" else len(test_paths),
                "groups": len({parse_path(path)[0] for path in train_dev_paths}) if name == "train-dev.txt" else len({parse_path(path)[0] for path in val_dev_paths}) if name == "val-dev.txt" else len(test_groups),
                "byte_policy": "official_input_byte_for_byte_copy" if name == "official-test.txt" else "utf8_no_bom_lf_one_terminal_newline",
            }
            for name, content in files.items()
        },
        "set_relationships": relationships,
        "group_key_rule": "first four ASCII hyphen-separated stem segments",
        "mine_rule": "first stem segment; 01..06",
        "mine_counts": per_mine,
        "statistic_definitions": statistic_definitions(),
        "statistics": {"official_train": total, "train_dev": train_total, "val_dev": val_total},
        "objective": objective_details(val_total, total),
        "local_search": {"accepted_steps": moves, "stop_reason": reason},
        "rare_classes": rare_class_list(samples),
        "threshold_checks": thresholds,
        "warnings": warnings,
        "hard_constraint_checks": hard,
        "candidate_status": "candidate",
        "user_gate_a": {"status": "pending", "signed_by": None, "signature_reference": None},
        "audit_report": "audit-report.json",
    }

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root: Path | None = None
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent))
        for name, content in files.items():
            (temporary_root / name).write_bytes(content)
        (temporary_root / "manifest.json").write_bytes(canonical_json(manifest))
        from audit_museg_splits import audit
        report = audit(args.dataset_root, args.official_train, args.official_test, temporary_root, write=True)
        if not report["pass"]:
            raise ValueError("generated audit failed")
        if args.output_root.exists():
            raise FileExistsError("output root appeared during generation; refusing overwrite")
        temporary_root.rename(args.output_root)
        temporary_root = None
    except BaseException:
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError) as error: print(f'error: {error}',file=sys.stderr);raise SystemExit(2)