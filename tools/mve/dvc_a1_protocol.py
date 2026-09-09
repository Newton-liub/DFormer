#!/usr/bin/env python3
"""Validate and materialize the dedicated MUSeg DVC-A1 protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_V1_SCHEMA_VERSION = "museg-dvc-a1-protocol-v1"
PROTOCOL_V1_ID = "DVC-A1-valdev-boundary-zero-v1"
PROTOCOL_V2_SCHEMA_VERSION = "museg-dvc-a1-protocol-v2"
PROTOCOL_V2_ID = "DVC-A1-valdev-boundary-zero-v2"
PROTOCOL_V3_SCHEMA_VERSION = "museg-dvc-a1-protocol-v3"
PROTOCOL_V3_ID = "DVC-A1-valdev-boundary-zero-v3-bgcontext"
V1_MASK_MANIFEST_SHA256 = "60b988b3f9ffaabc5f6540cfccda48ddb5efd4d44ce360691bfaeea047e63f29"
EXPECTED_CHECKPOINT_SHA256 = "f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c"
EXPECTED_SPLIT_SHA256 = "1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / "protocols" / "dvc-a1-valdev-boundary-zero-v1.template.json"
DEFAULT_V2_TEMPLATE = REPO_ROOT / "protocols" / "dvc-a1-valdev-boundary-zero-v2.template.json"
DEFAULT_V3_TEMPLATE = REPO_ROOT / "protocols" / "dvc-a1-valdev-boundary-zero-v3-bgcontext.template.json"
DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "MUSeg" / "dev-v1" / "val-dev.txt"
SOURCE_IDENTITY_PATHS = (
    "tools/mve/dvc_a1_protocol.py",
    "tools/mve/dvc_a1_core.py",
    "tools/mve/run_dvc_a1.py",
    "tools/evaluate_museg_checkpoint.py",
    "tools/prepare_museg.py",
)


class DvcProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class DvcProtocol:
    path: Path
    raw: Mapping[str, Any]
    sha256: str

    @property
    def dataset_root(self) -> Path:
        return Path(str(self.raw["dataset"]["root"])).resolve()

    @property
    def checkpoint_path(self) -> Path:
        return Path(str(self.raw["model"]["checkpoint_path"])).resolve()

    @property
    def split_path(self) -> Path:
        return Path(str(self.raw["split"]["path"])).resolve()

    @property
    def evaluation_path(self) -> Path:
        scope = self.raw.get("evaluation_scope")
        if isinstance(scope, Mapping):
            return Path(str(scope["allowlist_path"])).resolve()
        return self.split_path

    @property
    def source_mask_manifest_path(self) -> Path | None:
        scope = self.raw.get("evaluation_scope")
        if not isinstance(scope, Mapping):
            return None
        return Path(str(scope["source_mask_manifest_path"])).resolve()

    @property
    def evidence_root(self) -> Path:
        return Path(str(self.raw["evidence"]["output_root"])).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DvcProtocolError(f"cannot read protocol JSON {path}: {exc}") from exc


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise DvcProtocolError(f"protocol field {key!r} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], name: str) -> None:
    missing, extra = sorted(required - set(value)), sorted(set(value) - required)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported " + ", ".join(extra))
        raise DvcProtocolError(f"{name} fields invalid: {'; '.join(details)}")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DvcProtocolError(f"{name} must be a lowercase SHA-256")
    return value


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("__MATERIALIZED_") and value.endswith("__")


def _require_path(value: Any, name: str, *, allow_placeholders: bool) -> Path | None:
    if allow_placeholders and _is_placeholder(value):
        return None
    if not isinstance(value, str) or not value:
        raise DvcProtocolError(f"{name} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise DvcProtocolError(f"{name} must be absolute")
    return path.resolve()


def _validate_fixed_semantics(raw: Mapping[str, Any], *, allow_placeholders: bool = False) -> int:
    identity = (str(raw.get("schema_version")), str(raw.get("protocol_id")))
    versions = {
        (PROTOCOL_V1_SCHEMA_VERSION, PROTOCOL_V1_ID): 1,
        (PROTOCOL_V2_SCHEMA_VERSION, PROTOCOL_V2_ID): 2,
        (PROTOCOL_V3_SCHEMA_VERSION, PROTOCOL_V3_ID): 3,
    }
    version = versions.get(identity)
    if version is None:
        raise DvcProtocolError("unexpected DVC-A1 schema or protocol identity")
    if raw.get("research_role") != "paired-development-sensitivity":
        raise DvcProtocolError("DVC-A1 is restricted to paired development sensitivity")
    model = _mapping(raw, "model")
    if model.get("name") != "DFormerv2-S RGB Quick-B0" or model.get("config_module") != "local_configs.MUSeg.DFormerv2_S_4090":
        raise DvcProtocolError("model identity differs from frozen Quick-B0")
    if _sha256(model.get("checkpoint_sha256"), "model.checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise DvcProtocolError("checkpoint SHA-256 differs from frozen epoch 420")
    split = _mapping(raw, "split")
    expected_split = {
        "role": "val_dev",
        "sha256": EXPECTED_SPLIT_SHA256,
        "sample_count": 318,
        "location_group_count": 196,
        "location_group_rule": "first-four-ascii-hyphen-separated-stem-segments",
        "mine_rule": "first-ascii-hyphen-separated-stem-segment",
    }
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            raise DvcProtocolError(f"split.{key} differs from the frozen val-dev contract")
    input_contract = _mapping(raw, "input_contract")
    rgb, depth, label = (_mapping(input_contract, key) for key in ("rgb", "depth", "label"))
    if rgb != {
        "channel_order": "RGB",
        "normalization_identity": "rgb-imagenet-rgb-order-v1",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }:
        raise DvcProtocolError("RGB input contract differs from Quick-B0")
    if depth.get("raw_dtype") != "uint16" or depth.get("invalid_value") != 0 or depth.get("depth_max_raw") != 13932:
        raise DvcProtocolError("Depth16 identity differs from the frozen input contract")
    if label.get("raw_background") != 0 or label.get("foreground_ids") != list(range(1, 16)) or label.get("evaluator_ignore") != 255:
        raise DvcProtocolError("label contract differs from the frozen 15-class mapping")
    evaluator = _mapping(raw, "evaluator")
    expected_evaluator = {
        "identity": f"DVC-A1-msflip-whole-original-grid-v{version}",
        "base_identity": "msflip-whole-original-grid-v1",
        "scales": [0.5, 0.75, 1.0, 1.25, 1.5],
        "horizontal_flip": True,
        "forward_precision": "fp32",
        "logits_fusion": "mean-pre-softmax-fp32",
        "metric_grid": "original-label-grid",
        "batch_size": 1,
    }
    if evaluator != expected_evaluator:
        raise DvcProtocolError("evaluator contract differs from the frozen five-scale flip evaluator")
    corruption = _mapping(raw, "corruption")
    conditions = corruption.get("conditions")
    expected_conditions = [
        {"id": "clean", "domain": "boundary", "dose": 0.0},
        {"id": "boundary-q25", "domain": "boundary", "dose": 0.25},
        {"id": "boundary-q50", "domain": "boundary", "dose": 0.5},
        {"id": "boundary-q75", "domain": "boundary", "dose": 0.75},
        {"id": "nonboundary-q50", "domain": "nonboundary-matched", "dose": 0.5},
    ]
    if conditions != expected_conditions:
        raise DvcProtocolError("corruption conditions differ from DVC-A1")
    expected_corruption = {
        "seed": 20260908,
        "relative_jump_threshold": 0.05,
        "neighbor_pairs": ["horizontal", "vertical"],
        "edge_seed_endpoints": "both",
        "boundary_dilation": {"shape": "square", "kernel_size": 3, "iterations": 1},
        "guard_band": {"metric": "chebyshev", "radius_pixels": 5},
        "ranking": "splitmix64-v1(seed,sample_id,domain,flat_index)",
        "dose_count_rounding": "floor",
        "q75_unconstructable_location_group_fraction_limit": 0.05 if version == 1 else 0.0,
        "nonboundary_shortage_policy": "protocol-blocked",
    }
    for key, expected in expected_corruption.items():
        if corruption.get(key) != expected:
            raise DvcProtocolError(f"corruption.{key} differs from the frozen operation")
    expected_boundary_iou = {
        "class_mode": "one-vs-rest",
        "boundary_side": "inside",
        "distance_metric": "euclidean-l2",
        "distance_ratio_of_image_diagonal": 0.02,
        "distance_rounding": "max(1,round-half-to-even)",
        "ignore_exclusion_radius": "same-as-boundary-distance",
        "both_empty": "exclude-pair",
        "one_empty": 0.0,
        "image_aggregation": "macro-over-defined-foreground-classes",
        "group_aggregation": "equal-mean-over-images",
    }
    if version == 3:
        expected_boundary_iou = {
            **expected_boundary_iou,
            "target_encoding": "foreground-raw-minus-one-background-num-classes",
            "background_context": "valid-one-vs-rest-context",
            "true_ignore_value": 255,
        }
    if _mapping(raw, "boundary_iou") != expected_boundary_iou:
        raise DvcProtocolError("Boundary IoU contract differs from the frozen one-vs-rest definition")
    expected_statistics = {
        "unit": "location-group",
        "bootstrap_algorithm": "numpy-pcg64",
        "bootstrap_seed": 20260908,
        "bootstrap_replicates": 10000,
        "interval": "two-sided-95-percentile-linear",
        "dose_effect": "boundary-q75-minus-clean",
        "specificity_effect": "boundary-q50-minus-nonboundary-q50",
        "mine_role": "descriptive-direction-only",
    }
    if _mapping(raw, "statistics") != expected_statistics:
        raise DvcProtocolError("bootstrap contract differs from the preregistered group procedure")
    expected_adjudication = {
        "supported": {
            "dose_point_max_percentage_points": -2.0,
            "dose_interval_upper_strictly_below": 0.0,
            "specificity_point_strictly_below": 0.0,
            "minimum_negative_mines": 4,
        },
        "not_supported": {
            "dose_point_strictly_above_percentage_points": -1.0,
            "specificity_point_min_percentage_points": 0.0,
        },
    }
    if _mapping(raw, "adjudication") != expected_adjudication:
        raise DvcProtocolError("adjudication thresholds differ from the preregistered joint decision")
    dataset = _mapping(raw, "dataset")
    for key, expected in {
        "rgb_directory": "RGB",
        "depth16_directory": "Depth16",
        "quantized_depth_directory": "Depth",
        "label_directory": "Label",
    }.items():
        if dataset.get(key) != expected:
            raise DvcProtocolError(f"dataset.{key} differs from the frozen MUSeg layout")
    official = _mapping(raw, "official_test")
    if official != {
        "state": "sealed_unread",
        "included": False,
        "reject_identity_substrings": ["official", "test"],
    }:
        raise DvcProtocolError("official test must remain sealed and excluded")
    evidence = _mapping(raw, "evidence")
    if evidence.get("save_logits") is not False or evidence.get("save_predictions") is not False:
        raise DvcProtocolError("DVC-A1 must not persist logits or predictions")
    if version in (2, 3):
        scope = _mapping(raw, "evaluation_scope")
        expected_scope = {
            "role": "read-only-derived-evaluation-allowlist",
            "source_protocol_id": PROTOCOL_V1_ID,
            "source_mask_manifest_sha256": V1_MASK_MANIFEST_SHA256,
            "inclusion_rule": "location-group-has-at-least-one-nonempty-boundary-q75",
            "included_sample_count": 218,
            "included_location_group_count": 138,
            "excluded_location_group_count": 58,
            "q75_zero_sample_count": 31,
            "fully_constructable_location_group_count": 123,
            "partially_constructable_location_group_count": 15,
            "sensitivity_scope": "groups-with-nonempty-boundary-q75-for-every-image",
            "sensitivity_location_group_count": 123,
        }
        for key, expected in expected_scope.items():
            if scope.get(key) != expected:
                raise DvcProtocolError(f"evaluation_scope.{key} differs from the frozen v2 scope")
        _exact_keys(
            scope,
            set(expected_scope) | {"source_mask_manifest_path", "allowlist_path", "allowlist_sha256", "allowlist_summary_sha256"},
            "evaluation_scope",
        )
        for key in ("allowlist_sha256", "allowlist_summary_sha256"):
            value = scope.get(key)
            if not (allow_placeholders and _is_placeholder(value)):
                _sha256(value, f"evaluation_scope.{key}")
    return version


def load_dvc_protocol(path: str | os.PathLike[str], *, allow_placeholders: bool = False) -> DvcProtocol:
    source = Path(path).resolve()
    raw = _read_json(source)
    if not isinstance(raw, Mapping):
        raise DvcProtocolError("protocol root must be an object")
    identity = (raw.get("schema_version"), raw.get("protocol_id"))
    is_scoped = identity in {
        (PROTOCOL_V2_SCHEMA_VERSION, PROTOCOL_V2_ID),
        (PROTOCOL_V3_SCHEMA_VERSION, PROTOCOL_V3_ID),
    }
    root_keys = {
        "schema_version", "protocol_id", "research_role", "model", "split", "dataset",
        "input_contract", "evaluator", "corruption", "boundary_iou", "statistics",
        "adjudication", "official_test", "evidence", "code_identity",
    }
    if is_scoped:
        root_keys.add("evaluation_scope")
    _exact_keys(raw, root_keys, "protocol")
    version = _validate_fixed_semantics(raw, allow_placeholders=allow_placeholders)
    model, split, dataset, evidence = (_mapping(raw, key) for key in ("model", "split", "dataset", "evidence"))
    _require_path(model.get("checkpoint_path"), "model.checkpoint_path", allow_placeholders=allow_placeholders)
    _require_path(split.get("path"), "split.path", allow_placeholders=allow_placeholders)
    _require_path(dataset.get("root"), "dataset.root", allow_placeholders=allow_placeholders)
    _require_path(evidence.get("output_root"), "evidence.output_root", allow_placeholders=allow_placeholders)
    if version in (2, 3):
        scope = _mapping(raw, "evaluation_scope")
        _require_path(scope.get("source_mask_manifest_path"), "evaluation_scope.source_mask_manifest_path", allow_placeholders=allow_placeholders)
        _require_path(scope.get("allowlist_path"), "evaluation_scope.allowlist_path", allow_placeholders=allow_placeholders)
    code_identity = raw.get("code_identity")
    if not (allow_placeholders and _is_placeholder(code_identity)):
        if not isinstance(code_identity, Mapping):
            raise DvcProtocolError("code_identity must be a materialized object")
        _exact_keys(code_identity, {"git_head", "git_status_porcelain", "git_status_sha256", "source_sha256"}, "code_identity")
        if re.fullmatch(r"[0-9a-f]{40}", str(code_identity.get("git_head"))) is None:
            raise DvcProtocolError("code_identity.git_head must be a full commit")
        _sha256(code_identity.get("git_status_sha256"), "code_identity.git_status_sha256")
        sources = code_identity.get("source_sha256")
        if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_IDENTITY_PATHS):
            raise DvcProtocolError("code_identity.source_sha256 does not cover the DVC evaluator chain")
        for name, value in sources.items():
            _sha256(value, f"source_sha256.{name}")
    return DvcProtocol(path=source, raw=raw, sha256=file_sha256(source))


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _derive_v2_allowlist(
    *,
    source_mask_manifest_path: Path,
    split_entries: Sequence[str],
    allowlist_path: Path,
    evidence_root: Path,
    derived_protocol_id: str = PROTOCOL_V2_ID,
    summary_schema_version: str = "museg-dvc-a1-v2-allowlist-summary-v1",
) -> dict[str, Any]:
    source_mask_manifest_path = source_mask_manifest_path.resolve()
    if not source_mask_manifest_path.is_file() or file_sha256(source_mask_manifest_path) != V1_MASK_MANIFEST_SHA256:
        raise DvcProtocolError("v1 mask manifest is missing or has the wrong SHA-256")
    manifest = _read_json(source_mask_manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "museg-dvc-a1-mask-manifest-v1":
        raise DvcProtocolError("source mask manifest does not have the frozen v1 schema")
    if manifest.get("sample_count") != 318 or manifest.get("mask_complete_sample_count") != 318:
        raise DvcProtocolError("source mask manifest does not cover all 318 val-dev samples")
    if manifest.get("location_group_count") != 196 or manifest.get("nonboundary_shortages") != []:
        raise DvcProtocolError("source mask manifest group identity or nonboundary coverage differs from v1")
    rows = manifest.get("samples")
    if not isinstance(rows, list) or len(rows) != 318:
        raise DvcProtocolError("source mask manifest must contain 318 sample rows")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DvcProtocolError("source mask manifest contains a non-object sample row")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in rows_by_id:
            raise DvcProtocolError("source mask manifest sample identities are missing or duplicated")
        rows_by_id[sample_id] = row
    entry_by_id = {Path(entry).stem: entry for entry in split_entries}
    if len(entry_by_id) != 318 or set(rows_by_id) != set(entry_by_id):
        raise DvcProtocolError("source mask manifest samples differ from frozen val-dev")
    excluded_groups = sorted(str(value) for value in manifest.get("q75_unconstructable_location_groups", []))
    if len(excluded_groups) != 58 or len(set(excluded_groups)) != 58:
        raise DvcProtocolError("source mask manifest must identify exactly 58 excluded groups")
    included_rows = [row for row in rows if str(row["location_group"]) not in set(excluded_groups)]
    included_groups = sorted({str(row["location_group"]) for row in included_rows})
    if len(included_rows) != 218 or len(included_groups) != 138:
        raise DvcProtocolError("derived v2 scope is not exactly 138 groups and 218 samples")
    q75_count_by_group: dict[str, list[int]] = {}
    q75_zero_sample_ids: list[str] = []
    for row in included_rows:
        counts = row.get("condition_counts")
        if not isinstance(counts, Mapping):
            raise DvcProtocolError("source mask manifest sample lacks condition counts")
        count = int(counts.get("boundary-q75", -1))
        if count < 0:
            raise DvcProtocolError("source mask manifest contains an invalid boundary-q75 count")
        group = str(row["location_group"])
        q75_count_by_group.setdefault(group, []).append(count)
        if count == 0:
            q75_zero_sample_ids.append(str(row["sample_id"]))
    fully_constructable_groups = sorted(group for group, counts in q75_count_by_group.items() if all(count > 0 for count in counts))
    partially_constructable_groups = sorted(group for group, counts in q75_count_by_group.items() if any(count == 0 for count in counts))
    if len(q75_zero_sample_ids) != 31 or len(fully_constructable_groups) != 123 or len(partially_constructable_groups) != 15:
        raise DvcProtocolError("derived v2 constructability counts differ from the frozen 31/123/15 scope")
    included_sample_ids = {str(row["sample_id"]) for row in included_rows}
    allowlist_entries = [entry for entry in split_entries if Path(entry).stem in included_sample_ids]
    allowlist_path = allowlist_path.resolve()
    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = allowlist_path.with_suffix(allowlist_path.suffix + ".tmp")
    temporary.write_text("\n".join(allowlist_entries) + "\n", encoding="utf-8")
    temporary.replace(allowlist_path)
    summary = {
        "schema_version": summary_schema_version,
        "protocol_id": derived_protocol_id,
        "source_protocol_id": PROTOCOL_V1_ID,
        "source_mask_manifest_path": str(source_mask_manifest_path),
        "source_mask_manifest_sha256": V1_MASK_MANIFEST_SHA256,
        "source_split_sha256": EXPECTED_SPLIT_SHA256,
        "allowlist_path": str(allowlist_path),
        "allowlist_sha256": file_sha256(allowlist_path),
        "official_test_included": False,
        "included_sample_count": len(allowlist_entries),
        "included_location_group_count": len(included_groups),
        "excluded_location_group_count": len(excluded_groups),
        "q75_zero_sample_count": len(q75_zero_sample_ids),
        "fully_constructable_location_group_count": len(fully_constructable_groups),
        "partially_constructable_location_group_count": len(partially_constructable_groups),
        "included_location_groups": included_groups,
        "excluded_location_groups": excluded_groups,
        "q75_zero_sample_ids": sorted(q75_zero_sample_ids),
        "fully_constructable_location_groups": fully_constructable_groups,
        "partially_constructable_location_groups": partially_constructable_groups,
    }
    _atomic_write_json(evidence_root.resolve() / "allowlist-summary.json", summary)
    return summary


def materialize_dvc_protocol(
    *,
    template_path: Path,
    output_path: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    evidence_root: Path,
    split_path: Path = DEFAULT_SPLIT,
    source_mask_manifest_path: Path | None = None,
    allowlist_path: Path | None = None,
) -> DvcProtocol:
    template = load_dvc_protocol(template_path, allow_placeholders=True)
    dataset_root, checkpoint_path, split_path, evidence_root = (
        value.resolve() for value in (dataset_root, checkpoint_path, split_path, evidence_root)
    )
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise DvcProtocolError("epoch 420 checkpoint is missing or has the wrong SHA-256")
    if not split_path.is_file() or file_sha256(split_path) != EXPECTED_SPLIT_SHA256:
        raise DvcProtocolError("val-dev split is missing or has the wrong SHA-256")
    entries = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != 318 or len(set(entries)) != 318:
        raise DvcProtocolError("val-dev must contain 318 unique samples")
    sample_ids = [Path(entry).stem for entry in entries]
    if len({"-".join(sample_id.split("-")[:4]) for sample_id in sample_ids}) != 196:
        raise DvcProtocolError("val-dev must contain 196 location groups")
    if any(not entry.startswith("RGB/") or "official" in entry.lower() or "test" in entry.lower() for entry in entries):
        raise DvcProtocolError("split identity is not an allowed val-dev RGB list")
    directories = template.raw["dataset"]
    for sample_id in sample_ids:
        required = (
            dataset_root / str(directories["rgb_directory"]) / f"{sample_id}.jpg",
            dataset_root / str(directories["depth16_directory"]) / f"{sample_id}.png",
            dataset_root / str(directories["quantized_depth_directory"]) / f"{sample_id}.png",
            dataset_root / str(directories["label_directory"]) / f"{sample_id}.png",
        )
        if any(not path.is_file() for path in required):
            raise DvcProtocolError(f"missing required val-dev modality for {sample_id}")
    version = {
        PROTOCOL_V1_ID: 1,
        PROTOCOL_V2_ID: 2,
        PROTOCOL_V3_ID: 3,
    }[str(template.raw["protocol_id"])]
    allowlist_summary: dict[str, Any] | None = None
    if version in (2, 3):
        if source_mask_manifest_path is None or allowlist_path is None:
            raise DvcProtocolError("scoped DVC-A1 materialization requires source mask manifest and allowlist paths")
        allowlist_summary = _derive_v2_allowlist(
            source_mask_manifest_path=source_mask_manifest_path,
            split_entries=entries,
            allowlist_path=allowlist_path,
            evidence_root=evidence_root,
            derived_protocol_id=PROTOCOL_V2_ID if version == 2 else PROTOCOL_V3_ID,
            summary_schema_version="museg-dvc-a1-v2-allowlist-summary-v1" if version == 2 else "museg-dvc-a1-v3-allowlist-summary-v1",
        )
    status = _git_output("status", "--porcelain=v1")
    source_hashes = {}
    for relative_path in SOURCE_IDENTITY_PATHS:
        source = REPO_ROOT / relative_path
        if not source.is_file():
            raise DvcProtocolError(f"missing DVC evaluator source {relative_path}")
        source_hashes[relative_path] = file_sha256(source)
    raw = copy.deepcopy(dict(template.raw))
    raw["model"]["checkpoint_path"] = str(checkpoint_path)
    raw["split"]["path"] = str(split_path)
    raw["dataset"]["root"] = str(dataset_root)
    raw["evidence"]["output_root"] = str(evidence_root)
    if version in (2, 3) and allowlist_summary is not None:
        raw["evaluation_scope"]["source_mask_manifest_path"] = allowlist_summary["source_mask_manifest_path"]
        raw["evaluation_scope"]["allowlist_path"] = allowlist_summary["allowlist_path"]
        raw["evaluation_scope"]["allowlist_sha256"] = allowlist_summary["allowlist_sha256"]
        raw["evaluation_scope"]["allowlist_summary_sha256"] = file_sha256(evidence_root / "allowlist-summary.json")
    raw["code_identity"] = {
        "git_head": _git_output("rev-parse", "HEAD").strip(),
        "git_status_porcelain": status.splitlines(),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_sha256": source_hashes,
    }
    _atomic_write_json(output_path.resolve(), raw)
    return load_dvc_protocol(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--source-mask-manifest", type=Path)
    parser.add_argument("--allowlist", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = materialize_dvc_protocol(
            template_path=args.template,
            output_path=args.output,
            dataset_root=args.dataset_root,
            checkpoint_path=args.checkpoint,
            evidence_root=args.evidence_root,
            split_path=args.split,
            source_mask_manifest_path=args.source_mask_manifest,
            allowlist_path=args.allowlist,
        )
    except (DvcProtocolError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "protocol-blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "protocol-ready", "path": str(protocol.path), "sha256": protocol.sha256}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
