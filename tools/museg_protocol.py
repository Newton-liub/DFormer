#!/usr/bin/env python3
"""Versioned MUSeg training protocol manifest contract and JSON helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_SCHEMA_VERSION = "museg-training-protocol-v1"
RUN_MANIFEST_SCHEMA_VERSION = "museg-run-manifest-v1"
ENVIRONMENT_SCHEMA_VERSION = "museg-environment-v1"
COMMAND_SCHEMA_VERSION = "museg-command-v1"
_ALLOWED_PHASES = {"qualification", "development", "official"}
_REQUIRED_SPLITS = {"train_dev", "val_dev", "official_train", "official_test"}


class ProtocolError(ValueError):
    """Raised when a protocol manifest is incomplete or internally unsafe."""


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | os.PathLike[str]) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read valid JSON from {source}: {exc}") from exc


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically write deterministic UTF-8 JSON on Windows and Unix."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _resolve_path(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def _require_mapping(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"protocol field {name!r} must be an object")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ProtocolError(f"{field} must be 64 hexadecimal characters")
    return value.lower()


def _require_path_component(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ProtocolError(f"{field} must be one safe path component")
    return value


@dataclass(frozen=True)
class ProtocolManifest:
    path: Path
    raw: Mapping[str, Any]
    manifest_sha256: str

    @property
    def schema_version(self) -> str:
        return str(self.raw["schema_version"])

    @property
    def protocol_id(self) -> str:
        return str(self.raw["protocol_id"])

    @property
    def schedule_version(self) -> str:
        return str(self.raw["schedule_version"])

    @property
    def phase(self) -> str:
        return str(self.raw["phase"])

    @property
    def model(self) -> str:
        return str(self.raw["model"])

    @property
    def config_module(self) -> str:
        return str(self.raw["config_module"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.raw["seeds"])

    @property
    def output_root(self) -> Path:
        return _resolve_path(str(self.raw["output_root"]), self.path.parent)

    @property
    def run_root(self) -> Path:
        return self.output_root / self.protocol_id / self.phase

    @property
    def training(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "training")

    @property
    def swanlab(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "swanlab")

    @property
    def git(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "git")

    @property
    def pretrained(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "pretrained")

    @property
    def splits(self) -> Mapping[str, Mapping[str, Any]]:
        return _require_mapping(self.raw, "splits")  # type: ignore[return-value]

    def resolve_declared_path(self, value: str) -> Path:
        return _resolve_path(value, self.path.parent)

    def split_path(self, role: str) -> Path:
        entry = self.splits.get(role)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ProtocolError(f"split {role!r} has no valid path")
        return self.resolve_declared_path(str(entry["path"]))

    def seed_output_dir(self, seed: int) -> Path:
        return self.run_root / f"seed-{int(seed)}"


def load_protocol(path: str | os.PathLike[str]) -> ProtocolManifest:
    source = Path(path).resolve()
    raw = read_json(source)
    if not isinstance(raw, Mapping):
        raise ProtocolError("protocol manifest root must be an object")
    required = {
        "schema_version", "protocol_id", "schedule_version", "phase", "model",
        "config_module", "git", "seeds", "output_root", "splits", "pretrained",
        "training", "swanlab",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ProtocolError(f"protocol manifest is missing fields: {', '.join(missing)}")
    extra = sorted(set(raw) - required)
    if extra:
        raise ProtocolError(f"protocol manifest has unsupported fields: {', '.join(extra)}")
    if raw["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolError(
            f"unsupported protocol schema {raw['schema_version']!r}; expected {PROTOCOL_SCHEMA_VERSION!r}"
        )
    for name in ("protocol_id", "schedule_version", "model", "config_module", "output_root"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ProtocolError(f"protocol field {name!r} must be a non-empty string")
    _require_path_component(raw["protocol_id"], "protocol_id")
    if raw["phase"] not in _ALLOWED_PHASES:
        raise ProtocolError(f"unsupported phase: {raw['phase']!r}")
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ProtocolError("seeds must be a non-empty non-negative integer array")
    if len(set(seeds)) != len(seeds):
        raise ProtocolError("seeds must be unique")
    splits = _require_mapping(raw, "splits")
    git = _require_mapping(raw, "git")
    if set(git) != {"required_commit"}:
        raise ProtocolError("git supports only required_commit")
    required_commit = git.get("required_commit")
    if not isinstance(required_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", required_commit) is None:
        raise ProtocolError("git.required_commit must be a full 40-character hexadecimal commit")
    missing_splits = sorted(_REQUIRED_SPLITS - set(splits))
    if missing_splits:
        raise ProtocolError(f"protocol is missing split roles: {', '.join(missing_splits)}")
    extra_splits = sorted(set(splits) - _REQUIRED_SPLITS)
    if extra_splits:
        raise ProtocolError(f"protocol has unsupported split roles: {', '.join(extra_splits)}")
    for role in _REQUIRED_SPLITS:
        entry = splits[role]
        if not isinstance(entry, Mapping):
            raise ProtocolError(f"split {role!r} must be an object")
        for field in ("path", "samples", "groups", "sha256"):
            if field not in entry:
                raise ProtocolError(f"split {role!r} is missing {field!r}")
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise ProtocolError(f"split {role!r} path must be a non-empty string")
        if not isinstance(entry["samples"], int) or entry["samples"] < 0:
            raise ProtocolError(f"split {role!r} samples must be non-negative")
        if not isinstance(entry["groups"], int) or entry["groups"] < 0:
            raise ProtocolError(f"split {role!r} groups must be non-negative")
        _require_sha256(entry["sha256"], f"split {role!r} sha256")
    if splits["official_test"].get("sealed_unread") is not True:
        raise ProtocolError("official_test must declare sealed_unread=true")
    pretrained = _require_mapping(raw, "pretrained")
    if set(pretrained) != {"path", "size_bytes", "sha256"}:
        raise ProtocolError("pretrained supports only path, size_bytes, and sha256")
    for field in ("path", "size_bytes", "sha256"):
        if field not in pretrained:
            raise ProtocolError(f"pretrained metadata is missing {field!r}")
    if not isinstance(pretrained["path"], str) or not pretrained["path"]:
        raise ProtocolError("pretrained.path must be a non-empty string")
    if not isinstance(pretrained["size_bytes"], int) or pretrained["size_bytes"] <= 0:
        raise ProtocolError("pretrained.size_bytes must be a positive integer")
    _require_sha256(pretrained["sha256"], "pretrained.sha256")
    training = _require_mapping(raw, "training")
    integer_fields = (
        "epochs", "batch_size", "val_batch_size", "workers", "eval_start_epoch",
        "eval_interval", "save_interval",
    )
    boolean_fields = ("amp", "compile", "syncbn", "sliding", "mst")
    for field in integer_fields + boolean_fields:
        if field not in training:
            raise ProtocolError(f"training metadata is missing {field!r}")
    for field in integer_fields:
        minimum = 0 if field == "workers" else 1
        if isinstance(training[field], bool) or not isinstance(training[field], int) or training[field] < minimum:
            raise ProtocolError(f"training.{field} must be an integer >= {minimum}")
    for field in boolean_fields:
        if not isinstance(training[field], bool):
            raise ProtocolError(f"training.{field} must be boolean")
    if training["eval_start_epoch"] > training["epochs"]:
        raise ProtocolError("training.eval_start_epoch cannot exceed training.epochs")
    swanlab = _require_mapping(raw, "swanlab")
    if swanlab.get("mode") not in {"disabled", "offline", "online"}:
        raise ProtocolError("swanlab.mode must be disabled, offline, or online")
    for field in ("project", "workspace"):
        if not isinstance(swanlab.get(field), str) or not str(swanlab[field]).strip():
            raise ProtocolError(f"swanlab.{field} must be a non-empty string")
    return ProtocolManifest(path=source, raw=raw, manifest_sha256=file_sha256(source))