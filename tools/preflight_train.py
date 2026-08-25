#!/usr/bin/env python3
"""Statically audit a versioned MUSeg training protocol before GPU allocation."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.museg_protocol import ProtocolError, file_sha256, load_protocol, write_json

DEFAULT_CONFIG = "local_configs.MUSeg.DFormerv2_S_4090"
REQUIRED_PACKAGES = {
    "torch": "torch", "numpy": "numpy", "cv2": "opencv-python", "PIL": "Pillow",
    "tensorboardX": "tensorboardX", "easydict": "easydict", "timm": "timm", "mmengine": "mmengine",
}


class Preflight:
    """Legacy-compatible console collector used by focused package/GPU checks."""
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


@dataclass
class AuditReport:
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    config_summary: dict[str, Any] = field(default_factory=dict)

    def ok(self, code: str, message: str, **details: Any) -> None:
        self.checks.append({"level": "ok", "code": code, "message": message, "details": details})

    def error(self, code: str, message: str, **details: Any) -> None:
        item = {"level": "error", "code": code, "message": message, "details": details}
        self.errors.append(item)
        self.checks.append(item)

    def warn(self, code: str, message: str, **details: Any) -> None:
        item = {"level": "warning", "code": code, "message": message, "details": details}
        self.warnings.append(item)
        self.checks.append(item)

    def to_dict(self, protocol=None) -> dict[str, Any]:
        return {
            "schema_version": "museg-preflight-report-v1", "pass": not self.errors,
            "protocol_id": protocol.protocol_id if protocol else None,
            "protocol_manifest": str(protocol.path) if protocol else None,
            "protocol_manifest_sha256": protocol.manifest_sha256 if protocol else None,
            "phase": protocol.phase if protocol else None, "checks": self.checks,
            "errors": self.errors, "warnings": self.warnings,
            "config_summary": self.config_summary, "environment": self.environment,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest")
    parser.add_argument("--report", help="structured JSON report destination")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--static-only", action="store_true", help="skip all CUDA/GPU calls")
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--swanlab-mode", choices=("disabled", "offline", "online"), default=None)
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
        ("train", "train_source", "num_train_imgs"), ("validation", "val_source", "num_eval_imgs"),
    ):
        value = getattr(config, source_attr, None)
        if value is None:
            continue
        source = Path(value)
        if not source.is_file():
            result.error(f"{split_name} split is missing: {source}")
            continue
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
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
    path = Path(getattr(config, "pretrained_model", ""))
    if path.is_file():
        result.ok(f"pretrained weights: {path} ({path.stat().st_size:,} bytes)")
    else:
        result.error(f"pretrained weights are missing: {path}")


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
    target_error = validate_gpu_target(props.name, props.total_memory)
    if target_error and not allow_other_gpu:
        result.error(target_error + "; use --allow-other-gpu only for development checks")
        return
    if target_error:
        result.warn(target_error + " (allowed by --allow-other-gpu)")
    result.ok(f"CUDA device 0: {props.name}")


def check_swanlab(result: Preflight, mode: str) -> None:
    if mode == "disabled":
        result.ok("SwanLab disabled; package and credentials are not required")
        return
    if importlib.util.find_spec("swanlab") is None:
        result.error("SwanLab is unavailable; install requirements-monitoring.txt")
        return
    if mode == "online" and not (os.environ.get("SWANLAB_API_KEY") or os.environ.get("SWANLAB_API_KEY_FILE")):
        result.error("online SwanLab requires non-interactive credentials in the environment")
        return
    result.ok(f"SwanLab static requirements available for mode: {mode}")


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        encoding="utf-8",
        errors="surrogateescape",
        stderr=subprocess.STDOUT,
    ).strip()


def _split_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group(entry: str) -> str:
    parts = Path(entry).stem.split("-")
    return "-".join(parts[:4]) if len(parts) >= 4 else Path(entry).stem


def _audit_git(report: AuditReport, protocol, repo_root: Path) -> None:
    try:
        commit = _git_output(repo_root, "rev-parse", "HEAD")
        branch = _git_output(repo_root, "branch", "--show-current")
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
            cwd=repo_root,
            encoding="utf-8",
            errors="surrogateescape",
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        report.error("git_unavailable", "cannot inspect Git repository", error=str(exc))
        return
    paths = [record[3:] if len(record) > 3 else record for record in porcelain.split("\0") if record]
    if paths:
        report.error("git_dirty", "Git worktree is not clean", paths=paths)
    else:
        report.ok("git_clean", "Git worktree is clean")
    required = str(protocol.git.get("required_commit", ""))
    if not required:
        report.error("git_required_commit_missing", "protocol does not declare git.required_commit")
    else:
        ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", required, commit], cwd=repo_root, capture_output=True)
        if ancestry.returncode != 0:
            report.error("git_required_commit_absent", "current commit does not contain required protocol commit", required=required, actual=commit)
        else:
            report.ok("git_commit", "required protocol commit is contained by HEAD", required=required, actual=commit, branch=branch)
    report.environment["git"] = {"branch": branch, "commit": commit, "dirty": bool(paths), "dirty_paths": paths}


def _audit_splits(report: AuditReport, protocol) -> None:
    sample_sets: dict[str, set[str]] = {}
    group_sets: dict[str, set[str]] = {}
    for role, declared in protocol.splits.items():
        path = protocol.split_path(role)
        if not path.is_file():
            report.error("split_missing", f"split {role} is missing", role=role, path=str(path))
            continue
        try:
            lines = _split_lines(path)
        except (OSError, UnicodeError) as exc:
            report.error("split_unreadable", f"split {role} cannot be read", role=role, error=str(exc))
            continue
        samples, groups = set(lines), {_group(line) for line in lines}
        sample_sets[role], group_sets[role] = samples, groups
        if len(lines) != len(samples):
            report.error("split_duplicates", f"split {role} contains duplicate samples", role=role, duplicates=len(lines) - len(samples))
        if len(lines) != int(declared["samples"]):
            report.error("split_sample_count_mismatch", f"split {role} sample count differs", role=role, expected=declared["samples"], actual=len(lines))
        if len(groups) != int(declared["groups"]):
            report.error("split_group_count_mismatch", f"split {role} group count differs", role=role, expected=declared["groups"], actual=len(groups))
        actual_sha = file_sha256(path)
        if actual_sha.lower() != str(declared["sha256"]).lower():
            report.error("split_sha256_mismatch", f"split {role} SHA-256 differs", role=role, expected=declared["sha256"], actual=actual_sha)
        else:
            report.ok("split_identity", f"split {role} identity matches", role=role, path=str(path), samples=len(lines), groups=len(groups), sha256=actual_sha)
    roles = sorted(sample_sets)
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            if "official_train" in {left, right} and ({left, right} & {"train_dev", "val_dev"}):
                continue
            sample_overlap = sorted(sample_sets[left] & sample_sets[right])
            group_overlap = sorted(group_sets[left] & group_sets[right])
            if sample_overlap:
                report.error("split_sample_overlap", f"split sample overlap: {left}/{right}", left=left, right=right, count=len(sample_overlap), items=sample_overlap[:20])
            if group_overlap:
                report.error("split_group_overlap", f"split group overlap: {left}/{right}", left=left, right=right, count=len(group_overlap), items=group_overlap[:20])
    if all(role in sample_sets for role in ("train_dev", "val_dev", "official_train")) and sample_sets["train_dev"] | sample_sets["val_dev"] != sample_sets["official_train"]:
        report.error("split_dev_not_closed", "train-dev and val-dev do not close to official train")


def _audit_phase(report: AuditReport, protocol) -> None:
    train_role = "official_train" if protocol.phase == "official" else "train_dev"
    val_role = None if protocol.phase == "official" else "val_dev"
    if protocol.split_path(train_role) == protocol.split_path("official_test"):
        report.error("phase_role_error", "training source aliases sealed official test", train_role=train_role)
    if val_role and protocol.split_path(val_role) == protocol.split_path("official_test"):
        report.error("phase_role_error", "validation source aliases sealed official test", val_role=val_role)
    if protocol.phase == "official" and not bool(protocol.splits["official_test"].get("sealed_unread")):
        report.error("phase_role_error", "official phase must declare official_test sealed_unread")
    report.ok("phase_roles", "phase roles keep official test out of training", phase=protocol.phase, train_role=train_role, val_role=val_role, test_role="sealed_unread")


def _audit_pretrained(report: AuditReport, protocol) -> None:
    declared = protocol.pretrained
    path = protocol.resolve_declared_path(str(declared["path"]))
    if not path.is_file():
        report.error("pretrained_missing", "pretrained checkpoint is missing", path=str(path))
        return
    size = path.stat().st_size
    if size != int(declared["size_bytes"]):
        report.error("pretrained_size_mismatch", "pretrained checkpoint size differs", expected=declared["size_bytes"], actual=size)
    actual_sha = file_sha256(path)
    if actual_sha.lower() != str(declared["sha256"]).lower():
        report.error("pretrained_sha256_mismatch", "pretrained checkpoint SHA-256 differs", expected=declared["sha256"], actual=actual_sha)
    if size == int(declared["size_bytes"]) and actual_sha.lower() == str(declared["sha256"]).lower():
        report.ok("pretrained_identity", "pretrained checkpoint identity matches", path=str(path), size_bytes=size, sha256=actual_sha)


def _audit_output(report: AuditReport, protocol) -> None:
    root = protocol.output_root
    existing = root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir() or not os.access(existing, os.W_OK):
        report.error(
            "output_parent_unwritable",
            "output root has no writable parent",
            output_root=str(root),
            existing_parent=str(existing),
        )
    else:
        report.ok(
            "output_parent_writable",
            "output root has a writable parent",
            output_root=str(root),
            existing_parent=str(existing),
        )
    for seed in protocol.seeds:
        target = protocol.seed_output_dir(seed)
        if not target.exists():
            continue
        if not target.is_dir():
            report.error(
                "output_collision",
                "seed output target exists and is not a directory",
                seed=seed,
                path=str(target),
            )
        elif any(target.iterdir()):
            report.error(
                "output_collision",
                "seed output directory is non-empty",
                seed=seed,
                path=str(target),
            )


def _environment(static_only: bool) -> dict[str, Any]:
    package_names = set(REQUIRED_PACKAGES.values()) | {"swanlab"}
    packages: dict[str, str | None] = {}
    for distribution in package_names:
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None
    result: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "pytorch": packages.get("torch"), "cuda": None, "cudnn": None, "driver": None, "gpu": None,
    }
    if not static_only:
        try:
            import torch
            result["cuda"] = torch.version.cuda
            result["cudnn"] = torch.backends.cudnn.version()
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                result["gpu"] = {"name": props.name, "total_memory": props.total_memory}
        except Exception as exc:
            result["torch_environment_error"] = str(exc)
        try:
            result["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return result


def audit_protocol(
    protocol,
    *,
    repo_root: str | Path = REPO_ROOT,
    check_git: bool = True,
    static_only: bool = True,
    check_dataset_files: bool = False,
    sample_count: int = 3,
) -> AuditReport:
    report = AuditReport()
    report.environment = _environment(static_only)
    for package in REQUIRED_PACKAGES.values():
        if report.environment["packages"].get(package) is None:
            report.error("package_missing", "required package is unavailable", package=package)
    if check_git:
        _audit_git(report, protocol, Path(repo_root).resolve())
    try:
        imported_config = getattr(importlib.import_module(protocol.config_module), "C")
        report.ok(
            "config_import",
            "training config imports successfully",
            module=protocol.config_module,
            dataset=getattr(imported_config, "dataset_name", None),
            backbone=getattr(imported_config, "backbone", None),
        )
        if check_dataset_files:
            data_check = Preflight()
            check_dataset(imported_config, data_check, sample_count)
            for message in data_check.errors:
                report.error("dataset_error", message)
            for message in data_check.warnings:
                report.warn("dataset_warning", message)
    except Exception as exc:
        report.error("config_import_error", "training config cannot be imported", module=protocol.config_module, error=str(exc))
    _audit_splits(report, protocol)
    _audit_phase(report, protocol)
    _audit_pretrained(report, protocol)
    _audit_output(report, protocol)
    training = protocol.training
    report.config_summary = {
        "config_module": protocol.config_module, "phase": protocol.phase, "seeds": list(protocol.seeds),
        "epochs": training["epochs"], "iterations_per_epoch": training.get("iterations_per_epoch"),
        "warmup_epochs": training.get("warmup_epochs"), "batch_size": training["batch_size"],
        "workers": training["workers"], "eval_start_epoch": training["eval_start_epoch"],
        "eval_interval": training["eval_interval"], "save_interval": training["save_interval"],
    }
    mode = str(protocol.swanlab["mode"])
    if mode in {"offline", "online"} and importlib.util.find_spec("swanlab") is None:
        report.error("swanlab_package_missing", f"{mode} SwanLab package is unavailable")
    if mode == "online":
        key = os.environ.get("SWANLAB_API_KEY")
        key_file_value = os.environ.get("SWANLAB_API_KEY_FILE")
        key_file = Path(key_file_value).expanduser() if key_file_value else None
        key_file_available = bool(key_file and key_file.is_file() and key_file.stat().st_size > 0)
        if not key and not key_file_available:
            report.error("swanlab_credentials_missing", "online SwanLab credentials are unavailable")
        report.ok("swanlab_noninteractive", "launcher requires ExperimentTracker interactive=False")
    else:
        report.ok("swanlab_mode", "SwanLab mode does not require online credentials", mode=mode)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol_manifest:
        try:
            protocol = load_protocol(args.protocol_manifest)
            report = audit_protocol(
                protocol,
                repo_root=args.repo_root,
                check_git=True,
                static_only=args.static_only,
                check_dataset_files=True,
                sample_count=args.sample_count,
            )
            if not args.static_only:
                legacy = Preflight()
                check_cuda(legacy, args.allow_no_gpu, args.allow_other_gpu)
                for message in legacy.errors:
                    report.error("gpu_error", message)
                for message in legacy.warnings:
                    report.warn("gpu_warning", message)
            if args.swanlab_mode and args.swanlab_mode != protocol.swanlab["mode"]:
                report.error("swanlab_mode_mismatch", "CLI SwanLab mode differs from protocol", cli=args.swanlab_mode, protocol=protocol.swanlab["mode"])
            destination = Path(args.report).resolve() if args.report else protocol.run_root / "preflight.json"
            write_json(destination, report.to_dict(protocol))
            print(json.dumps(report.to_dict(protocol), ensure_ascii=False, sort_keys=True))
            return 1 if report.errors else 0
        except (OSError, ProtocolError, subprocess.SubprocessError) as exc:
            failure = {"schema_version": "museg-preflight-report-v1", "pass": False, "errors": [{"level": "error", "code": "protocol_error", "message": str(exc), "details": {}}], "warnings": [], "checks": []}
            if args.report:
                write_json(args.report, failure)
            print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
            return 2
    result = Preflight()
    check_packages(result)
    config = load_config(args.config, result)
    if config is not None:
        check_dataset(config, result, args.sample_count)
        check_pretrained(config, result, args.skip_pretrained)
        check_output(config, result)
    if not args.static_only:
        check_cuda(result, args.allow_no_gpu, args.allow_other_gpu)
    check_swanlab(result, args.swanlab_mode or "disabled")
    print(json.dumps({"schema_version": "dformer-legacy-preflight-v1", "pass": not result.errors, "errors": result.errors, "warnings": result.warnings}, ensure_ascii=False))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
