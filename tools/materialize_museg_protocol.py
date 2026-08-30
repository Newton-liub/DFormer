#!/usr/bin/env python3
"""Materialize an immutable, host-specific MUSeg qualification protocol manifest."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.museg_protocol import (
    FROZEN_AUDIT_PATH,
    FROZEN_AUDIT_SHA256,
    FROZEN_MANIFEST_PATH,
    FROZEN_MANIFEST_SHA256,
    ProtocolError,
    file_sha256,
    load_protocol,
    read_json,
    write_json,
)

TEMPLATE_PATH = REPO_ROOT / "protocols" / "museg-qualification-v1.template.json"
_PLACEHOLDER = "__MATERIALIZED_"


def _git_clean_commit(repo_root: Path) -> str:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "-z", "--untracked-files=all"], cwd=repo_root
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError("cannot inspect the Git worktree before protocol materialization") from exc
    if status:
        raise ProtocolError("protocol materialization requires a clean Git worktree")
    if len(commit) != 40:
        raise ProtocolError("Git did not return a full HEAD commit")
    return commit


def _absolute_existing_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ProtocolError(f"{label} does not exist as a file: {path}")
    return path


def materialize(
    *,
    output: str | Path,
    output_root: str | Path,
    official_train: str | Path,
    pretrained: str | Path,
    batch_size: int,
    swanlab_mode: str,
    swanlab_project: str,
    swanlab_workspace: str,
    template_path: str | Path = TEMPLATE_PATH,
    repo_root: str | Path = REPO_ROOT,
) -> tuple[Path, str]:
    if batch_size <= 0:
        raise ProtocolError("batch_size must be positive")
    if swanlab_mode not in {"disabled", "offline", "online"}:
        raise ProtocolError("swanlab_mode must be disabled, offline, or online")
    if not swanlab_project.strip() or not swanlab_workspace.strip():
        raise ProtocolError("SwanLab project and workspace must be non-empty")
    repo = Path(repo_root).resolve()
    commit = _git_clean_commit(repo)
    template = read_json(Path(template_path).resolve())
    if not isinstance(template, dict):
        raise ProtocolError("protocol template root must be an object")
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ProtocolError(f"protocol output already exists: {target}")
    root = Path(output_root).expanduser().resolve()
    if _PLACEHOLDER in str(root):
        raise ProtocolError("output_root contains an unresolved materialization placeholder")
    official_train_path = _absolute_existing_file(str(official_train), "official_train")
    pretrained_path = _absolute_existing_file(str(pretrained), "pretrained")
    raw: dict[str, Any] = copy.deepcopy(template)
    raw["git"]["required_commit"] = commit
    raw["output_root"] = str(root)
    raw["official_train"]["path"] = str(official_train_path)
    raw["pretrained"] = {
        "path": str(pretrained_path),
        "size_bytes": pretrained_path.stat().st_size,
        "sha256": file_sha256(pretrained_path),
    }
    raw["training"]["batch_size"] = batch_size
    raw["swanlab"] = {
        "mode": swanlab_mode,
        "project": swanlab_project,
        "workspace": swanlab_workspace,
    }
    raw["split_authority"] = {
        "manifest_path": str(FROZEN_MANIFEST_PATH.resolve()),
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "audit_report_path": str(FROZEN_AUDIT_PATH.resolve()),
        "audit_report_sha256": FROZEN_AUDIT_SHA256,
    }
    rendered = str(raw)
    if _PLACEHOLDER in rendered:
        raise ProtocolError("protocol contains unresolved materialization placeholders")
    # Verify all split and weight identities before any new protocol evidence is written.
    temporary = target.with_suffix(target.suffix + ".validation")
    try:
        write_json(temporary, raw)
        protocol = load_protocol(temporary)
        protocol.validate_consumed_splits()
        if protocol.split_path("official_train") != official_train_path:
            raise ProtocolError("materialized official train path did not round-trip")
    finally:
        temporary.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, raw)
    return target, file_sha256(target)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new immutable manifest file")
    parser.add_argument("--output-root", required=True, help="absolute root for qualification run evidence")
    parser.add_argument("--official-train", required=True, help="authoritative MUSeg train.txt")
    parser.add_argument("--pretrained", required=True, help="pretrained DFormerv2-S checkpoint")
    parser.add_argument("--batch-size", required=True, type=int, help="probe batch or approved qualification batch")
    parser.add_argument("--swanlab-mode", choices=("disabled", "offline", "online"), default="online")
    parser.add_argument("--swanlab-project", default="DFormer-liu")
    parser.add_argument("--swanlab-workspace", default="Newton_liub")
    parser.add_argument("--template", default=str(TEMPLATE_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        target, digest = materialize(
            output=args.output,
            output_root=args.output_root,
            official_train=args.official_train,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            swanlab_mode=args.swanlab_mode,
            swanlab_project=args.swanlab_project,
            swanlab_workspace=args.swanlab_workspace,
            template_path=args.template,
        )
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"materialize_museg_protocol: {exc}", file=sys.stderr)
        return 2
    print(f"protocol_manifest={target}\nprotocol_manifest_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())