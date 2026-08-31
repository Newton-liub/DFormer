#!/usr/bin/env python3
"""Audit explicit cloud-storage cleanup candidates without deleting anything."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "museg-cloud-storage-audit-v1"
DEFAULT_CHECKPOINT_CANDIDATES = 9


class StorageAuditError(ValueError):
    """Raised when a cleanup candidate violates a safety boundary."""


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_existing(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise StorageAuditError(f"{label} does not exist or cannot be resolved: {candidate}") from exc


def _resolve_protected(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _assert_safe_storage_root(storage_root: Path) -> None:
    if not storage_root.is_dir():
        raise StorageAuditError(f"storage_root is not a directory: {storage_root}")
    if storage_root == Path(storage_root.anchor):
        raise StorageAuditError("storage_root cannot be a filesystem root")


def _assert_candidate_safe(candidate: Path, storage_root: Path, protected: Sequence[Path]) -> None:
    if candidate == storage_root:
        raise StorageAuditError("cleanup candidate cannot equal storage_root")
    if not _is_relative_to(candidate, storage_root):
        raise StorageAuditError(f"cleanup candidate escapes storage_root: {candidate}")
    for protected_path in protected:
        if _is_relative_to(candidate, protected_path) or _is_relative_to(protected_path, candidate):
            raise StorageAuditError(
                f"cleanup candidate overlaps protected path: candidate={candidate}, protected={protected_path}"
            )


def _inspect_path(path: Path, storage_root: Path) -> dict[str, Any]:
    total_bytes = 0
    file_count = 0
    directory_count = 0
    symlink_count = 0
    latest_mtime = path.lstat().st_mtime

    if path.is_file():
        stat = path.stat()
        return {
            "path": str(path),
            "kind": "file",
            "size_bytes": stat.st_size,
            "file_count": 1,
            "directory_count": 0,
            "symlink_count": 0,
            "latest_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }

    if not path.is_dir():
        raise StorageAuditError(f"cleanup candidate must be a regular file or directory: {path}")

    def raise_walk_error(error: OSError) -> None:
        raise StorageAuditError(f"cannot inspect cleanup candidate: {path}: {error}") from error

    for current_root, directories, files in os.walk(
        path,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current = Path(current_root)
        directory_count += 1
        latest_mtime = max(latest_mtime, current.lstat().st_mtime)
        for name in [*directories, *files]:
            entry = current / name
            stat = entry.lstat()
            latest_mtime = max(latest_mtime, stat.st_mtime)
            if entry.is_symlink():
                symlink_count += 1
                target = entry.resolve(strict=False)
                if not _is_relative_to(target, storage_root):
                    raise StorageAuditError(f"symlink inside candidate escapes storage_root: {entry} -> {target}")
                total_bytes += stat.st_size
            elif entry.is_file():
                file_count += 1
                total_bytes += stat.st_size

    return {
        "path": str(path),
        "kind": "directory",
        "size_bytes": total_bytes,
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "latest_mtime_utc": datetime.fromtimestamp(latest_mtime, timezone.utc).isoformat(),
    }


def audit_storage(
    *,
    storage_root: str | Path,
    candidates: Sequence[str | Path],
    protected_paths: Sequence[str | Path],
    checkpoint_size_bytes: int | None = None,
    checkpoint_candidates: int = DEFAULT_CHECKPOINT_CANDIDATES,
) -> dict[str, Any]:
    """Return a structured, non-destructive audit of exact cleanup candidates."""
    root = _resolve_existing(storage_root, "storage_root")
    _assert_safe_storage_root(root)
    if not candidates:
        raise StorageAuditError("at least one explicit cleanup candidate is required")
    if not protected_paths:
        raise StorageAuditError("at least one explicit protected path is required")
    if checkpoint_candidates <= 0:
        raise StorageAuditError("checkpoint_candidates must be positive")
    if checkpoint_size_bytes is not None and checkpoint_size_bytes <= 0:
        raise StorageAuditError("checkpoint_size_bytes must be positive when provided")

    protected = [_resolve_protected(path) for path in protected_paths]
    resolved_candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_candidate in candidates:
        candidate = _resolve_existing(raw_candidate, "cleanup candidate")
        _assert_candidate_safe(candidate, root, protected)
        if candidate in seen:
            raise StorageAuditError(f"duplicate cleanup candidate: {candidate}")
        seen.add(candidate)
        resolved_candidates.append(candidate)

    records = [_inspect_path(path, root) for path in resolved_candidates]
    reclaimable_bytes = sum(int(record["size_bytes"]) for record in records)
    usage = shutil.disk_usage(root)
    checkpoint_budget_bytes = (
        checkpoint_size_bytes * checkpoint_candidates if checkpoint_size_bytes is not None else None
    )
    projected_free = usage.free + reclaimable_bytes

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "destructive_actions_performed": False,
        "storage_root": str(root),
        "protected_paths": [str(path) for path in protected],
        "candidates": records,
        "summary": {
            "candidate_count": len(records),
            "reclaimable_bytes": reclaimable_bytes,
            "disk_total_bytes": usage.total,
            "disk_used_bytes": usage.used,
            "disk_free_bytes": usage.free,
            "projected_free_after_cleanup_bytes": projected_free,
            "checkpoint_candidates": checkpoint_candidates,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "checkpoint_storage_budget_bytes": checkpoint_budget_bytes,
            "projected_free_covers_checkpoint_storage_only": (
                projected_free >= checkpoint_budget_bytes if checkpoint_budget_bytes is not None else None
            ),
            "budget_scope": "checkpoint files only; excludes datasets, logs, archives, and temporary files",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="exact existing path; repeat as needed")
    parser.add_argument("--protect", action="append", required=True, help="protected path; repeat as needed")
    parser.add_argument("--checkpoint-size-bytes", type=int)
    parser.add_argument("--checkpoint-candidates", type=int, default=DEFAULT_CHECKPOINT_CANDIDATES)
    parser.add_argument("--report", help="optional JSON report destination")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_storage(
            storage_root=args.storage_root,
            candidates=args.candidate,
            protected_paths=args.protect,
            checkpoint_size_bytes=args.checkpoint_size_bytes,
            checkpoint_candidates=args.checkpoint_candidates,
        )
    except (OSError, StorageAuditError) as exc:
        print(f"audit_museg_cloud_storage: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        destination = Path(args.report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
