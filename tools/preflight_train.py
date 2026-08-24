#!/usr/bin/env python3
"""Validate a DFormer training environment before allocating a full run."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = "local_configs.MUSeg.DFormerv2_S_4090"
REQUIRED_PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "tensorboardX": "tensorboardX",
    "easydict": "easydict",
    "timm": "timm",
    "mmengine": "mmengine",
}


class Preflight:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"[ERROR] {message}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="importable training config module")
    parser.add_argument("--sample-count", type=int, default=3, help="samples checked in each split")
    parser.add_argument("--allow-no-gpu", action="store_true", help="warn instead of failing when CUDA is unavailable")
    parser.add_argument(
        "--allow-other-gpu",
        action="store_true",
        help="allow a CUDA device other than an RTX 4090 with at least 20 GiB (development only)",
    )
    parser.add_argument("--skip-pretrained", action="store_true", help="skip the pretrained weight check")
    parser.add_argument(
        "--swanlab-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="SwanLab mode that the subsequent training command will use",
    )
    return parser.parse_args(argv)


def check_packages(result: Preflight) -> None:
    for module_name, distribution_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            result.error(f"required package is unavailable: {distribution_name} ({module_name})")
            continue
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            version = "version unknown"
        result.ok(f"package {distribution_name}: {version}")


def load_config(module_name: str, result: Preflight):
    try:
        config = getattr(importlib.import_module(module_name), "C")
    except Exception as exc:
        result.error(f"cannot import config {module_name}: {exc}")
        return None
    result.ok(f"config imported: {module_name}")
    return config


def _sample_lines(lines: list[str], count: int) -> list[str]:
    if not lines or count <= 0:
        return []
    count = min(count, len(lines))
    if count == 1:
        return [lines[0]]
    indexes = [round(index * (len(lines) - 1) / (count - 1)) for index in range(count)]
    return [lines[index] for index in indexes]


def _paths_for_entry(config, entry: str) -> tuple[Path, Path, Path]:
    stem = Path(entry).stem
    return (
        Path(config.rgb_root_folder) / f"{stem}{config.rgb_format}",
        Path(config.x_root_folder) / f"{stem}{config.x_format}",
        Path(config.gt_root_folder) / f"{stem}{config.gt_format}",
    )


def check_dataset(config, result: Preflight, sample_count: int) -> None:
    dataset_root = Path(config.dataset_path)
    if not dataset_root.is_dir():
        result.error(f"dataset directory is missing: {dataset_root}")
        return
    result.ok(f"dataset directory: {dataset_root}")

    for label, root in (
        ("RGB", Path(config.rgb_root_folder)),
        ("depth", Path(config.x_root_folder)),
        ("label", Path(config.gt_root_folder)),
    ):
        if root.is_dir():
            result.ok(f"{label} directory: {root}")
        else:
            result.error(f"{label} directory is missing: {root}")

    for split_name, source_attr, expected_attr in (
        ("train", "train_source", "num_train_imgs"),
        ("validation", "eval_source", "num_eval_imgs"),
    ):
        source = Path(getattr(config, source_attr))
        if not source.is_file():
            result.error(f"{split_name} split is missing: {source}")
            continue
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            result.error(f"{split_name} split is empty: {source}")
            continue
        expected = getattr(config, expected_attr, None)
        if expected is not None and len(lines) != int(expected):
            result.error(f"{split_name} split has {len(lines)} entries; config expects {expected}")
        else:
            result.ok(f"{split_name} split entries: {len(lines)}")

        for entry in _sample_lines(lines, sample_count):
            sample_paths = _paths_for_entry(config, entry)
            missing = [str(path) for path in sample_paths if not path.is_file()]
            if missing:
                result.error(f"sample {entry!r} is incomplete: {', '.join(missing)}")
                continue
            try:
                from PIL import Image

                sizes = []
                for path in sample_paths:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        sizes.append(image.size)
            except Exception as exc:
                result.error(f"sample {entry!r} cannot be decoded: {exc}")
            else:
                if len(set(sizes)) != 1:
                    result.error(f"sample {entry!r} has mismatched RGB/depth/label sizes: {sizes}")
                else:
                    result.ok(f"sample triplet decoded at {sizes[0]}: {entry}")


def check_pretrained(config, result: Preflight, skip: bool) -> None:
    if skip:
        result.warn("pretrained weight check skipped by request")
        return
    pretrained = getattr(config, "pretrained_model", None)
    if not pretrained:
        result.error("config does not define a pretrained model")
        return
    path = Path(pretrained)
    if path.is_file():
        result.ok(f"pretrained weights: {path} ({path.stat().st_size:,} bytes)")
    else:
        result.error(
            f"pretrained weights are missing: {path}; set DFORMER_PRETRAINED to the DFormerv2-S weight file"
        )


def check_output(config, result: Preflight) -> None:
    output = Path(config.log_dir)
    existing_parent = output
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        result.error(f"output path has no existing parent directory: {output}")
    elif not os.access(existing_parent, os.W_OK):
        result.error(f"nearest existing output parent is not writable: {existing_parent}")
    else:
        result.ok(f"nearest existing output parent is writable: {existing_parent} (target: {output})")


def validate_gpu_target(name: str, total_memory: int) -> str | None:
    """Return why a CUDA device is unsuitable for the planned 4090 run."""
    memory_gib = total_memory / 1024**3
    if "RTX 4090" not in name.upper():
        return f"CUDA device is not an RTX 4090: {name}"
    if memory_gib < 20.0:
        return f"RTX 4090 reports only {memory_gib:.1f} GiB; expected at least 20 GiB"
    return None


def check_cuda(result: Preflight, allow_no_gpu: bool, allow_other_gpu: bool = False) -> None:
    try:
        import torch
    except Exception as exc:
        result.error(f"cannot import PyTorch for CUDA check: {exc}")
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        message = f"CUDA GPU unavailable (torch {torch.__version__}, Python {platform.python_version()})"
        if allow_no_gpu:
            result.warn(message)
        else:
            result.error(message)
        return

    props = torch.cuda.get_device_properties(0)
    memory_gib = props.total_memory / 1024**3
    target_error = validate_gpu_target(props.name, props.total_memory)
    if target_error and not allow_other_gpu:
        result.error(target_error + "; use --allow-other-gpu only for development checks")
        return
    if target_error:
        result.warn(target_error + " (allowed by --allow-other-gpu)")
    result.ok(
        f"CUDA device 0: {props.name}, capability {props.major}.{props.minor}, {memory_gib:.1f} GiB"
    )


def check_swanlab(result: Preflight, mode: str) -> None:
    if mode == "disabled":
        result.ok("SwanLab disabled; package and credentials are not required")
        return
    if importlib.util.find_spec("swanlab") is None:
        result.error("SwanLab is unavailable; install requirements-monitoring.txt")
        return
    try:
        version = metadata.version("swanlab")
    except metadata.PackageNotFoundError:
        version = "version unknown"
    result.ok(f"SwanLab package: {version}; requested mode: {mode}")
    if mode == "online":
        if os.environ.get("SWANLAB_API_KEY"):
            result.ok("SWANLAB_API_KEY is present (value not displayed)")
        else:
            result.warn("SWANLAB_API_KEY is absent; online init must use an existing SwanLab login or will fail fast")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = Preflight()
    print(f"DFormer preflight with Python {sys.version.split()[0]}")
    check_packages(result)
    config = load_config(args.config, result)
    if config is not None:
        check_dataset(config, result, args.sample_count)
        check_pretrained(config, result, args.skip_pretrained)
        check_output(config, result)
    check_cuda(result, args.allow_no_gpu, args.allow_other_gpu)
    check_swanlab(result, args.swanlab_mode)
    print(f"Preflight summary: {len(result.errors)} error(s), {len(result.warnings)} warning(s)")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
