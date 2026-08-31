#!/usr/bin/env python3
"""Pinned MUSeg stage-01 authority bundle and versioned training protocol helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


PROTOCOL_SCHEMA_VERSION = "museg-training-protocol-v3"
LEGACY_PROTOCOL_SCHEMA_VERSION = "museg-training-protocol-v2"
RUN_MANIFEST_SCHEMA_VERSION = "museg-run-manifest-v3"
ENVIRONMENT_SCHEMA_VERSION = "museg-environment-v1"
COMMAND_SCHEMA_VERSION = "museg-command-v1"
FROZEN_SPLIT_SCHEMA_VERSION = "museg-dev-split-manifest-v1"
FROZEN_SPLIT_PROTOCOL_ID = "MUSEG-DEV-SPLIT-PROTOCOL-1"
FROZEN_AUDIT_SCHEMA_VERSION = "museg-dev-split-audit-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPLIT_ROOT = REPO_ROOT / "data" / "splits" / "MUSeg" / "dev-v1"
FROZEN_MANIFEST_PATH = FROZEN_SPLIT_ROOT / "manifest.json"
FROZEN_AUDIT_PATH = FROZEN_SPLIT_ROOT / "audit-report.json"
FROZEN_MANIFEST_SHA256 = "42233412f432e387cfcffc763724461e2dbc111969a595c714ac12add7bf7b01"
FROZEN_AUDIT_SHA256 = "53ac30aba0230919b994202f37b3571a7b416f9129f27eabf003415721e38055"
_ALLOWED_PHASES = frozenset({"qualification", "development", "official"})
_REQUIRED_OUTPUTS = frozenset({"train-dev.txt", "val-dev.txt", "official-test.txt"})


class ProtocolError(ValueError):
    """Raised when a protocol manifest is incomplete or violates frozen evidence."""


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


def _lines_and_groups(path: Path) -> tuple[int, int]:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"cannot read split {path}: {exc}") from exc
    if len(lines) != len(set(lines)):
        raise ProtocolError(f"split {path} contains duplicate samples")
    groups: set[str] = set()
    for line in lines:
        stem = Path(line).stem
        parts = stem.split("-")
        if len(parts) < 4:
            raise ProtocolError(f"split {path} contains an invalid MUSeg sample: {line!r}")
        groups.add("-".join(parts[:4]))
    return len(lines), len(groups)


def _validate_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        fragments = []
        if missing:
            fragments.append(f"missing {', '.join(missing)}")
        if extra:
            fragments.append(f"unsupported {', '.join(extra)}")
        raise ProtocolError(f"{name} has " + "; ".join(fragments))


@dataclass(frozen=True)
class SplitAuthority:
    manifest_path: Path
    manifest_sha256: str
    audit_report_path: Path
    audit_report_sha256: str
    manifest: Mapping[str, Any]
    audit_report: Mapping[str, Any]
    splits: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, str]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "audit_report_path": str(self.audit_report_path),
            "audit_report_sha256": self.audit_report_sha256,
        }


def _load_split_authority(raw: Mapping[str, Any], base: Path) -> SplitAuthority:
    authority = _require_mapping(raw, "split_authority")
    _validate_exact_keys(
        authority,
        frozenset({"manifest_path", "manifest_sha256", "audit_report_path", "audit_report_sha256"}),
        "split_authority",
    )
    for field in ("manifest_path", "audit_report_path"):
        if not isinstance(authority[field], str) or not authority[field].strip():
            raise ProtocolError(f"split_authority.{field} must be a non-empty string")
    manifest_path = _resolve_path(str(authority["manifest_path"]), base)
    audit_path = _resolve_path(str(authority["audit_report_path"]), base)
    if manifest_path != FROZEN_MANIFEST_PATH.resolve() or audit_path != FROZEN_AUDIT_PATH.resolve():
        raise ProtocolError("split_authority must reference the repository's frozen data/splits/MUSeg/dev-v1 bundle")
    manifest_sha = _require_sha256(authority["manifest_sha256"], "split_authority.manifest_sha256")
    audit_sha = _require_sha256(authority["audit_report_sha256"], "split_authority.audit_report_sha256")
    if manifest_sha != FROZEN_MANIFEST_SHA256 or audit_sha != FROZEN_AUDIT_SHA256:
        raise ProtocolError("split_authority does not pin the approved frozen manifest/audit SHA-256")
    if not manifest_path.is_file() or not audit_path.is_file():
        raise ProtocolError("the approved frozen split authority bundle is unavailable")
    if file_sha256(manifest_path) != manifest_sha or file_sha256(audit_path) != audit_sha:
        raise ProtocolError("the approved frozen split authority bundle SHA-256 does not match")
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    if not isinstance(manifest, Mapping) or not isinstance(audit, Mapping):
        raise ProtocolError("frozen split authority JSON roots must be objects")
    if manifest.get("schema_version") != FROZEN_SPLIT_SCHEMA_VERSION or manifest.get("protocol_id") != FROZEN_SPLIT_PROTOCOL_ID:
        raise ProtocolError("frozen split manifest has an unsupported identity")
    if manifest.get("candidate_status") != "frozen" or manifest.get("user_gate_a", {}).get("status") != "approved":
        raise ProtocolError("frozen split manifest is not Gate-A approved")
    if audit.get("schema_version") != FROZEN_AUDIT_SCHEMA_VERSION or audit.get("pass") is not True:
        raise ProtocolError("frozen split audit is not a passing supported audit report")
    if audit.get("manifest_sha256") != manifest_sha or audit.get("details", {}).get("manifest_sha256") != manifest_sha:
        raise ProtocolError("frozen split audit does not bind the approved manifest SHA-256")
    outputs = manifest.get("outputs")
    official = manifest.get("official")
    if not isinstance(outputs, Mapping) or set(outputs) != _REQUIRED_OUTPUTS or not isinstance(official, Mapping):
        raise ProtocolError("frozen split manifest does not contain the required split evidence")
    derived: dict[str, Mapping[str, Any]] = {}
    for role, filename in (("train_dev", "train-dev.txt"), ("val_dev", "val-dev.txt"), ("official_test", "official-test.txt")):
        record = outputs.get(filename)
        if not isinstance(record, Mapping):
            raise ProtocolError(f"frozen split manifest output {filename!r} is invalid")
        sha = _require_sha256(record.get("sha256"), f"frozen {filename} sha256")
        samples, groups = record.get("samples"), record.get("groups")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0 or isinstance(groups, bool) or not isinstance(groups, int) or groups < 0:
            raise ProtocolError(f"frozen split manifest output {filename!r} has invalid counts")
        derived[role] = {"path": str((manifest_path.parent / filename).resolve()), "samples": samples, "groups": groups, "sha256": sha, "sealed_unread": role == "official_test"}
    official_train = official.get("train")
    if not isinstance(official_train, Mapping):
        raise ProtocolError("frozen split manifest lacks official train evidence")
    train_sha = _require_sha256(official_train.get("sha256"), "frozen official train sha256")
    train_samples, train_groups = official_train.get("samples"), official_train.get("groups")
    if isinstance(train_samples, bool) or not isinstance(train_samples, int) or isinstance(train_groups, bool) or not isinstance(train_groups, int):
        raise ProtocolError("frozen official train evidence has invalid counts")
    derived["official_train"] = {"path": None, "samples": train_samples, "groups": train_groups, "sha256": train_sha}
    audit_counts = audit.get("details", {}).get("counts", {})
    audit_groups = audit.get("details", {}).get("groups", {})
    expected_audit = {
        "train_dev": derived["train_dev"], "val_dev": derived["val_dev"],
        "official_test": derived["official_test"], "official_train": derived["official_train"],
    }
    for role, record in expected_audit.items():
        if audit_counts.get(role) != record["samples"] or audit_groups.get(role) != record["groups"]:
            raise ProtocolError(f"frozen split audit count/group evidence mismatches {role}")
    return SplitAuthority(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        audit_report_path=audit_path,
        audit_report_sha256=audit_sha,
        manifest=MappingProxyType(dict(manifest)),
        audit_report=MappingProxyType(dict(audit)),
        splits=MappingProxyType(derived),
    )


@dataclass(frozen=True)
class ProtocolManifest:
    path: Path
    raw: Mapping[str, Any]
    manifest_sha256: str
    authority: SplitAuthority

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
    def run_kind(self) -> str:
        return str(self.raw.get("run_kind", "qualification"))

    @property
    def simulation(self) -> bool:
        return bool(self.raw.get("simulation", False))

    @property
    def checkpoint_policy(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "checkpoint_policy") if "checkpoint_policy" in self.raw else MappingProxyType({})

    @property
    def optimizer_telemetry(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "optimizer_telemetry") if "optimizer_telemetry" in self.raw else MappingProxyType({})

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
    def input_contract(self) -> Mapping[str, Any]:
        value = self.raw.get("input_contract")
        if isinstance(value, Mapping):
            return value
        # Version-2 manifests predate the explicit field. Preserve their
        # historical runtime behavior while marking the source as legacy.
        return MappingProxyType(
            {
                "channel_order": "BGR",
                "normalization": MappingProxyType(
                    {
                        "identity": "imagenet-rgb-statistics-in-array-order-v1",
                        "mean": (0.485, 0.456, 0.406),
                        "std": (0.229, 0.224, 0.225),
                    }
                ),
                "record_origin": "legacy-v2-museg-default",
            }
        )

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
    def official_train(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "official_train")

    @property
    def splits(self) -> Mapping[str, Mapping[str, Any]]:
        records = {role: dict(record) for role, record in self.authority.splits.items()}
        records["official_train"]["path"] = str(self.resolve_declared_path(str(self.official_train["path"])))
        return records

    def resolve_declared_path(self, value: str) -> Path:
        return _resolve_path(value, self.path.parent)

    def split_path(self, role: str) -> Path:
        entry = self.splits.get(role)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ProtocolError(f"split {role!r} has no valid path")
        return Path(str(entry["path"])).resolve()

    def phase_roles(self) -> tuple[str, str | None, str]:
        return ("official_train", None, "official_test") if self.phase == "official" else ("train_dev", "val_dev", "official_test")

    def seed_output_dir(self, seed: int) -> Path:
        return self.run_root / f"seed-{int(seed)}"

    def authority_identity(self) -> dict[str, str]:
        return self.authority.to_dict()

    def validate_consumed_splits(self) -> None:
        """Verify train/validation and official-train inputs without opening sealed test."""
        for role in ("train_dev", "val_dev", "official_train"):
            entry = self.splits[role]
            path = self.split_path(role)
            if not path.is_file():
                raise ProtocolError(f"authoritative split {role} is missing: {path}")
            samples, groups = _lines_and_groups(path)
            if samples != entry["samples"] or groups != entry["groups"]:
                raise ProtocolError(f"authoritative split {role} count/group mismatch")
            if file_sha256(path) != entry["sha256"]:
                raise ProtocolError(f"authoritative split {role} SHA-256 mismatch")


def load_protocol(path: str | os.PathLike[str]) -> ProtocolManifest:
    source = Path(path).resolve()
    raw = read_json(source)
    if not isinstance(raw, Mapping):
        raise ProtocolError("protocol manifest root must be an object")
    required = {
        "schema_version", "protocol_id", "schedule_version", "phase", "model", "config_module",
        "git", "seeds", "output_root", "split_authority", "official_train", "pretrained", "training", "swanlab",
    }
    schema_version = raw.get("schema_version")
    if schema_version == PROTOCOL_SCHEMA_VERSION:
        required.add("input_contract")
    allowed = required | {
        "run_kind", "simulation", "checkpoint_policy", "optimizer_telemetry",
        *({"input_contract"} if schema_version == LEGACY_PROTOCOL_SCHEMA_VERSION else set()),
    }
    missing = sorted(required - set(raw))
    extra = sorted(set(raw) - allowed)
    if missing or extra:
        fragments = []
        if missing:
            fragments.append(f"missing fields: {', '.join(missing)}")
        if extra:
            fragments.append(f"unsupported fields: {', '.join(extra)}")
        raise ProtocolError("protocol manifest has " + "; ".join(fragments))
    if schema_version not in {PROTOCOL_SCHEMA_VERSION, LEGACY_PROTOCOL_SCHEMA_VERSION}:
        raise ProtocolError(
            f"unsupported protocol schema {schema_version!r}; expected {PROTOCOL_SCHEMA_VERSION!r} "
            f"or legacy {LEGACY_PROTOCOL_SCHEMA_VERSION!r}"
        )
    for name in ("protocol_id", "schedule_version", "model", "config_module", "output_root"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ProtocolError(f"protocol field {name!r} must be a non-empty string")
    _require_path_component(raw["protocol_id"], "protocol_id")
    if raw["phase"] not in _ALLOWED_PHASES:
        raise ProtocolError(f"unsupported phase: {raw['phase']!r}")
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ProtocolError("seeds must be a non-empty non-negative integer array")
    if len(set(seeds)) != len(seeds):
        raise ProtocolError("seeds must be unique")
    if schema_version == PROTOCOL_SCHEMA_VERSION:
        input_contract = _require_mapping(raw, "input_contract")
        _validate_exact_keys(
            input_contract,
            frozenset({"channel_order", "normalization"}),
            "input_contract",
        )
        if input_contract.get("channel_order") not in {"BGR", "RGB"}:
            raise ProtocolError("input_contract.channel_order must be BGR or RGB")
        normalization = input_contract.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ProtocolError("input_contract.normalization must be an object")
        _validate_exact_keys(
            normalization,
            frozenset({"identity", "mean", "std"}),
            "input_contract.normalization",
        )
        if not isinstance(normalization.get("identity"), str) or not str(normalization["identity"]).strip():
            raise ProtocolError("input_contract.normalization.identity must be a non-empty string")
        for field in ("mean", "std"):
            values = normalization.get(field)
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)
            ):
                raise ProtocolError(f"input_contract.normalization.{field} must contain three numbers")
        if any(float(value) <= 0 for value in normalization["std"]):
            raise ProtocolError("input_contract.normalization.std values must be positive")
    run_kind = raw.get("run_kind", "qualification")
    if run_kind not in {"qualification", "standard", "probe", "lifecycle-test"}:
        raise ProtocolError("run_kind must be qualification, standard, probe, or lifecycle-test")
    simulation = raw.get("simulation", False)
    if not isinstance(simulation, bool):
        raise ProtocolError("simulation must be boolean")
    if run_kind == "lifecycle-test" and simulation is not True:
        raise ProtocolError("lifecycle-test protocols must set simulation=true")
    if run_kind != "lifecycle-test" and simulation:
        raise ProtocolError("simulation=true is reserved for lifecycle-test protocols")
    checkpoint_policy = raw.get("checkpoint_policy")
    if checkpoint_policy is not None:
        if not isinstance(checkpoint_policy, Mapping):
            raise ProtocolError("checkpoint_policy must be an object")
        _validate_exact_keys(
            checkpoint_policy,
            frozenset({"selector_geometry", "selector_scale", "selector_flip", "top_k", "retain_latest", "tie_break", "candidate_manifest"}),
            "checkpoint_policy",
        )
        selector_scale = checkpoint_policy.get("selector_scale")
        if (
            checkpoint_policy.get("selector_geometry") != "original-full"
            or isinstance(selector_scale, bool)
            or not isinstance(selector_scale, (int, float))
            or float(selector_scale) != 1.0
            or checkpoint_policy.get("selector_flip") is not False
        ):
            raise ProtocolError("checkpoint_policy must identify original-full scale 1.0 without flip")
        top_k = checkpoint_policy.get("top_k")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 8
            or checkpoint_policy.get("retain_latest") is not True
            or checkpoint_policy.get("tie_break") != "earlier_epoch"
        ):
            raise ProtocolError("checkpoint_policy must retain latest with top_k between 1 and 8 and earlier-epoch tie break")
        if not isinstance(checkpoint_policy.get("candidate_manifest"), str) or not checkpoint_policy["candidate_manifest"].strip():
            raise ProtocolError("checkpoint_policy.candidate_manifest must be a non-empty string")
        _require_path_component(checkpoint_policy["candidate_manifest"], "checkpoint_policy.candidate_manifest")
    telemetry = raw.get("optimizer_telemetry")
    if telemetry is not None:
        if not isinstance(telemetry, Mapping):
            raise ProtocolError("optimizer_telemetry must be an object")
        _validate_exact_keys(telemetry, frozenset({"schema_version", "required_counters", "invariant"}), "optimizer_telemetry")
        if telemetry.get("schema_version") != "museg-optimizer-telemetry-v1" or telemetry.get("required_counters") != ["attempted_steps", "completed_optimizer_steps", "skipped_optimizer_steps"] or telemetry.get("invariant") != "attempted_steps=completed_optimizer_steps+skipped_optimizer_steps":
            raise ProtocolError("optimizer_telemetry does not declare the supported counter invariant")

    git = _require_mapping(raw, "git")
    _validate_exact_keys(git, frozenset({"required_commit"}), "git")
    commit = git.get("required_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise ProtocolError("git.required_commit must be a full 40-character hexadecimal commit")
    official_train = _require_mapping(raw, "official_train")
    _validate_exact_keys(official_train, frozenset({"path"}), "official_train")
    if not isinstance(official_train.get("path"), str) or not str(official_train["path"]).strip():
        raise ProtocolError("official_train.path must be a non-empty string")
    pretrained = _require_mapping(raw, "pretrained")
    _validate_exact_keys(pretrained, frozenset({"path", "size_bytes", "sha256"}), "pretrained")
    if not isinstance(pretrained["path"], str) or not pretrained["path"]:
        raise ProtocolError("pretrained.path must be a non-empty string")
    if isinstance(pretrained["size_bytes"], bool) or not isinstance(pretrained["size_bytes"], int) or pretrained["size_bytes"] <= 0:
        raise ProtocolError("pretrained.size_bytes must be a positive integer")
    _require_sha256(pretrained["sha256"], "pretrained.sha256")
    training = _require_mapping(raw, "training")
    integer_fields = ("epochs", "batch_size", "val_batch_size", "workers", "eval_start_epoch", "eval_interval", "save_interval")
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
    authority = _load_split_authority(raw, source.parent)
    return ProtocolManifest(path=source, raw=raw, manifest_sha256=file_sha256(source), authority=authority)