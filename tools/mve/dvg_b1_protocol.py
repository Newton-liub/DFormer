#!/usr/bin/env python3
"""Validate and materialize the preregistered MUSeg DVG-B1 P0 protocol."""

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

PROTOCOL_SCHEMA_VERSION = "museg-dvg-b1-protocol-v1"
PROTOCOL_ID = "DVG-B1-oracle-gsa-v1"
SOURCE_PROTOCOL_ID = "DVC-A1-valdev-boundary-zero-v3-bgcontext"
EXPECTED_CHECKPOINT_SHA256 = "f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c"
EXPECTED_SPLIT_SHA256 = "1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83"
EXPECTED_SOURCE_PROTOCOL_SHA256 = "d52b3dba2c7a34894b9f4cdf1d8e313e304415d1ea821a191b7753b97be74f5d"
EXPECTED_SOURCE_MASK_MANIFEST_SHA256 = "3da28ae84806c61b0178c9563e811eed9e6f16e3f0f4040ab5753b26abdc9e79"
EXPECTED_SOURCE_ALLOWLIST_SHA256 = "5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89"
EXPECTED_SOURCE_ALLOWLIST_SUMMARY_SHA256 = "d60f7c503fb5b71e4941eeeddfd464819fe3fe9435253d8806c552b98c237ccc"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / "protocols" / "dvg-b1-oracle-gsa-v1.template.json"
DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "MUSeg" / "dev-v1" / "val-dev.txt"
SOURCE_IDENTITY_PATHS = (
    "tools/mve/dvg_b1_protocol.py",
    "protocols/dvg-b1-oracle-gsa-v1.template.json",
    "models/encoders/DFormerv2.py",
    "models/builder.py",
    "tools/evaluate_museg_checkpoint.py",
    "tools/mve/dvc_a1_core.py",
)
PLANNED_RUNTIME_PATHS = (
    "tools/mve/dvg_b1_core.py",
    "tools/mve/run_dvg_b1.py",
)


class DvgProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class DvgProtocol:
    path: Path
    raw: Mapping[str, Any]
    sha256: str


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
        raise DvgProtocolError(f"cannot read JSON {path}: {exc}") from exc


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise DvgProtocolError(f"protocol field {key!r} must be an object")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DvgProtocolError(f"{name} must be a lowercase SHA-256")
    return value


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("__MATERIALIZED_") and value.endswith("__")


def _absolute_path(value: Any, name: str, *, allow_placeholders: bool) -> Path | None:
    if allow_placeholders and _is_placeholder(value):
        return None
    if not isinstance(value, str) or not value:
        raise DvgProtocolError(f"{name} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise DvgProtocolError(f"{name} must be absolute")
    return path.resolve()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _validate_fixed_semantics(raw: Mapping[str, Any], *, allow_placeholders: bool) -> None:
    if raw.get("schema_version") != PROTOCOL_SCHEMA_VERSION or raw.get("protocol_id") != PROTOCOL_ID:
        raise DvgProtocolError("unexpected DVG-B1 schema or protocol identity")
    if raw.get("research_role") != "paired-development-oracle-gating":
        raise DvgProtocolError("DVG-B1 must remain paired development Oracle gating")

    preregistration = _mapping(raw, "preregistration")
    if preregistration.get("status") != "project-preregistered-not-literature-standard":
        raise DvgProtocolError("C must be identified as a project preregistration choice")
    if preregistration.get("results_seen_before_decision") is not False:
        raise DvgProtocolError("C must be frozen before DVG-B1 results are seen")

    model = _mapping(raw, "model")
    if model.get("name") != "DFormerv2-S RGB Quick-B0":
        raise DvgProtocolError("model identity differs from frozen Quick-B0")
    if _sha256(model.get("checkpoint_sha256"), "model.checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise DvgProtocolError("checkpoint SHA-256 differs from frozen epoch 420")

    split = _mapping(raw, "split")
    if split.get("role") != "val_dev_paired_development_evidence":
        raise DvgProtocolError("val-dev must retain paired development evidence responsibility")
    if _sha256(split.get("sha256"), "split.sha256") != EXPECTED_SPLIT_SHA256:
        raise DvgProtocolError("split SHA-256 differs from frozen val-dev")
    if split.get("sample_count") != 318 or split.get("location_group_count") != 196:
        raise DvgProtocolError("split counts differ from frozen val-dev")

    scope = _mapping(raw, "evaluation_scope")
    expected_scope = {
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "source_mask_manifest_sha256": EXPECTED_SOURCE_MASK_MANIFEST_SHA256,
        "source_allowlist_sha256": EXPECTED_SOURCE_ALLOWLIST_SHA256,
        "source_allowlist_summary_sha256": EXPECTED_SOURCE_ALLOWLIST_SUMMARY_SHA256,
        "included_sample_count": 218,
        "included_location_group_count": 138,
        "q75_zero_sample_count": 31,
        "fully_constructable_location_group_count": 123,
        "partially_constructable_location_group_count": 15,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise DvgProtocolError(f"evaluation_scope.{key} differs from frozen DVC-A1 v3 evidence")
    summary_sha = scope.get("allowlist_summary_sha256")
    if not (allow_placeholders and _is_placeholder(summary_sha)):
        _sha256(summary_sha, "evaluation_scope.allowlist_summary_sha256")

    evaluator = _mapping(raw, "evaluator")
    expected_evaluator = {
        "identity": "DVG-B1-msflip-whole-original-grid-v1",
        "base_identity": "msflip-whole-original-grid-v1",
        "scales": [0.5, 0.75, 1.0, 1.25, 1.5],
        "horizontal_flip": True,
        "view_count_per_sample": 10,
        "forward_precision": "fp32",
        "logits_fusion": "mean-pre-softmax-fp32",
        "metric_grid": "original-label-grid",
        "batch_size": 1,
    }
    if evaluator != expected_evaluator:
        raise DvgProtocolError("evaluator differs from the frozen five-scale flip contract")

    reliability = _mapping(raw, "oracle_reliability")
    expected_reliability = {
        "corruption_mask_semantics": "m_raw_1_is_corrupted",
        "reliability_definition": "r_raw=1-m_raw",
        "dtype": "fp32",
        "range": [0.0, 1.0],
        "view_resize": "opencv-inter-linear",
        "operation_order": ["resize-to-scaled-view", "horizontal-flip-if-view", "right-bottom-padding"],
        "padding_reliability": 1.0,
        "stage_aggregation": "opencv-inter-area-continuous-effective-area-ratio",
        "stage_indices": [0, 1, 2, 3],
        "fully_corrupted_token": 0.0,
        "all_trusted_token": 1.0,
        "thresholding_allowed": False,
    }
    if reliability != expected_reliability:
        raise DvgProtocolError("Oracle reliability differs from preregistered A")

    gate = _mapping(raw, "geometry_gate")
    expected_gate = {
        "target": "depth-geometry-contribution-only",
        "pairwise_rule": "query-key-symmetric-continuous-product",
        "full_shape": "[B,1,L,L]",
        "h_shape": "[B,1,W,H,H]",
        "w_shape": "[B,1,H,W,W]",
        "broadcast_axis": "heads-only",
        "spatial_contribution_unchanged": True,
        "qkv_unchanged": True,
        "rotary_encoding_unchanged": True,
        "depth_input_unchanged_after_frozen_corruption": True,
        "decoder_and_logits_postprocessing_unchanged": True,
    }
    if gate != expected_gate:
        raise DvgProtocolError("geometry gate differs from preregistered B")

    noop = _mapping(raw, "noop_contract")
    if noop.get("path") != "original-forward-bypass" or noop.get("stage_output_comparison") != "torch.equal":
        raise DvgProtocolError("no-op must use the original forward bypass and torch.equal")
    if noop.get("final_pre_softmax_logits_comparison") != "torch.equal" or noop.get("tolerance_fallback_allowed") is not False:
        raise DvgProtocolError("no-op logits comparison cannot use a tolerance fallback")

    statistics = _mapping(raw, "statistics")
    expected_statistics = {
        "unit": "location-group",
        "bootstrap_algorithm": "numpy-pcg64",
        "bootstrap_seed": 20260908,
        "bootstrap_replicates": 10000,
        "interval": "two-sided-95-percentile-linear",
        "primary_metric": "Boundary-IoU",
        "auxiliary_metric": "mIoU",
        "mine_role": "descriptive-direction-only",
    }
    if statistics != expected_statistics:
        raise DvgProtocolError("statistics differ from the frozen location-group procedure")

    adjudication = _mapping(raw, "adjudication")
    supported = _mapping(adjudication, "oracle_supported")
    if supported != {
        "all_required": True,
        "boundary_iou_point_min_percentage_points": 0.1,
        "boundary_iou_interval_lower_strictly_above_percentage_points": 0.0,
        "miou_point_min_percentage_points": 0.0,
    }:
        raise DvgProtocolError("oracle-supported rule differs from user-frozen C")
    not_supported = _mapping(adjudication, "oracle_not_supported")
    if not_supported != {
        "any_sufficient": True,
        "boundary_iou_point_below_percentage_points": 0.1,
        "miou_point_below_percentage_points": 0.0,
    }:
        raise DvgProtocolError("oracle-not-supported rule differs from user-frozen C")
    if adjudication.get("rule_status") != "project-preregistered-not-literature-standard":
        raise DvgProtocolError("adjudication must be marked as a project preregistration")

    official = _mapping(raw, "official_test")
    if official != {
        "state": "sealed_unread",
        "included": False,
        "reject_identity_substrings": ["official", "test"],
    }:
        raise DvgProtocolError("official test must remain sealed and excluded")
    gates = _mapping(raw, "execution_gates")
    if gates.get("current_maximum_authorized_stage") != "p0":
        raise DvgProtocolError("materialized protocol cannot authorize stages beyond P0")

    path_fields = (
        (model, "checkpoint_path", "model.checkpoint_path"),
        (split, "path", "split.path"),
        (_mapping(raw, "dataset"), "root", "dataset.root"),
        (_mapping(raw, "evidence"), "output_root", "evidence.output_root"),
        (scope, "source_protocol_path", "evaluation_scope.source_protocol_path"),
        (scope, "source_mask_manifest_path", "evaluation_scope.source_mask_manifest_path"),
        (scope, "source_allowlist_path", "evaluation_scope.source_allowlist_path"),
        (scope, "source_allowlist_summary_path", "evaluation_scope.source_allowlist_summary_path"),
        (scope, "allowlist_summary_path", "evaluation_scope.allowlist_summary_path"),
    )
    for parent, key, name in path_fields:
        _absolute_path(parent.get(key), name, allow_placeholders=allow_placeholders)


def load_dvg_protocol(path: str | os.PathLike[str], *, allow_placeholders: bool = False) -> DvgProtocol:
    source = Path(path).resolve()
    raw = _read_json(source)
    if not isinstance(raw, Mapping):
        raise DvgProtocolError("protocol root must be an object")
    _validate_fixed_semantics(raw, allow_placeholders=allow_placeholders)
    code_identity = raw.get("code_identity")
    if not (allow_placeholders and _is_placeholder(code_identity)):
        if not isinstance(code_identity, Mapping):
            raise DvgProtocolError("code_identity must be a materialized object")
        sources = code_identity.get("source_sha256")
        if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_IDENTITY_PATHS):
            raise DvgProtocolError("code_identity.source_sha256 does not cover the P0 source set")
        for name, value in sources.items():
            _sha256(value, f"code_identity.source_sha256.{name}")
        if code_identity.get("identity_role") != "p0-preregistration-baseline":
            raise DvgProtocolError("code_identity role must remain P0 preregistration baseline")
        if code_identity.get("planned_runtime_paths") != list(PLANNED_RUNTIME_PATHS):
            raise DvgProtocolError("planned DVG-B1 runtime paths differ from the plan")
        if code_identity.get("planned_runtime_paths_present") is not False:
            raise DvgProtocolError("P0 must not claim that P1 runtime sources already exist")
        if re.fullmatch(r"[0-9a-f]{40}", str(code_identity.get("git_head"))) is None:
            raise DvgProtocolError("code_identity.git_head must be a full commit")
        _sha256(code_identity.get("git_status_sha256"), "code_identity.git_status_sha256")
    return DvgProtocol(path=source, raw=raw, sha256=file_sha256(source))


def _require_file_sha256(path: Path, expected: str, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise DvgProtocolError(f"{name} is missing: {resolved}")
    actual = file_sha256(resolved)
    if actual != expected:
        raise DvgProtocolError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return resolved


def _validate_source_evidence(
    *,
    source_protocol_path: Path,
    source_mask_manifest_path: Path,
    source_allowlist_path: Path,
    source_allowlist_summary_path: Path,
    split_path: Path,
) -> list[str]:
    source_protocol = _read_json(source_protocol_path)
    if not isinstance(source_protocol, Mapping) or source_protocol.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise DvgProtocolError("source protocol identity is not the frozen DVC-A1 v3 protocol")
    if _mapping(source_protocol, "official_test").get("included") is not False:
        raise DvgProtocolError("source protocol must exclude official test")

    manifest = _read_json(source_mask_manifest_path)
    if not isinstance(manifest, Mapping):
        raise DvgProtocolError("source mask manifest must be an object")
    expected_manifest = {
        "schema_version": "museg-dvc-a1-mask-manifest-v3",
        "protocol_id": SOURCE_PROTOCOL_ID,
        "protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "official_test_included": False,
        "status": "passed",
        "sample_count": 218,
        "mask_complete_sample_count": 218,
        "location_group_count": 138,
        "q75_unconstructable_location_groups": [],
        "nonboundary_shortages": [],
        "gate_passed": True,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise DvgProtocolError(f"source mask manifest field {key!r} differs from frozen v3 evidence")

    split_entries = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    allowlist_entries = [line.strip() for line in source_allowlist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(split_entries) != 318 or len(set(split_entries)) != 318:
        raise DvgProtocolError("val-dev split must contain 318 unique entries")
    if len(allowlist_entries) != 218 or len(set(allowlist_entries)) != 218:
        raise DvgProtocolError("source allowlist must contain 218 unique entries")
    if not set(allowlist_entries).issubset(set(split_entries)):
        raise DvgProtocolError("source allowlist is not a subset of frozen val-dev")
    if any(not entry.startswith("RGB/") or "official" in entry.lower() or "test" in entry.lower() for entry in allowlist_entries):
        raise DvgProtocolError("source allowlist contains a forbidden identity")
    sample_ids = [Path(entry).stem for entry in allowlist_entries]
    if len({"-".join(sample_id.split("-")[:4]) for sample_id in sample_ids}) != 138:
        raise DvgProtocolError("source allowlist must contain 138 location groups")

    source_summary = _read_json(source_allowlist_summary_path)
    if not isinstance(source_summary, Mapping):
        raise DvgProtocolError("source allowlist summary must be an object")
    expected_summary = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "source_split_sha256": EXPECTED_SPLIT_SHA256,
        "allowlist_sha256": EXPECTED_SOURCE_ALLOWLIST_SHA256,
        "official_test_included": False,
        "included_sample_count": 218,
        "included_location_group_count": 138,
        "q75_zero_sample_count": 31,
        "fully_constructable_location_group_count": 123,
        "partially_constructable_location_group_count": 15,
    }
    for key, expected in expected_summary.items():
        if source_summary.get(key) != expected:
            raise DvgProtocolError(f"source allowlist summary field {key!r} differs from frozen v3 evidence")
    return allowlist_entries


def materialize_dvg_protocol(
    *,
    template_path: Path,
    output_path: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    evidence_root: Path,
    split_path: Path,
    source_protocol_path: Path,
    source_mask_manifest_path: Path,
    source_allowlist_path: Path,
    source_allowlist_summary_path: Path,
) -> DvgProtocol:
    template = load_dvg_protocol(template_path, allow_placeholders=True)
    checkpoint_path = _require_file_sha256(checkpoint_path, EXPECTED_CHECKPOINT_SHA256, "checkpoint")
    split_path = _require_file_sha256(split_path, EXPECTED_SPLIT_SHA256, "val-dev split")
    source_protocol_path = _require_file_sha256(
        source_protocol_path, EXPECTED_SOURCE_PROTOCOL_SHA256, "source DVC-A1 v3 protocol"
    )
    source_mask_manifest_path = _require_file_sha256(
        source_mask_manifest_path, EXPECTED_SOURCE_MASK_MANIFEST_SHA256, "source DVC-A1 v3 mask manifest"
    )
    source_allowlist_path = _require_file_sha256(
        source_allowlist_path, EXPECTED_SOURCE_ALLOWLIST_SHA256, "source DVC-A1 v3 allowlist"
    )
    source_allowlist_summary_path = _require_file_sha256(
        source_allowlist_summary_path,
        EXPECTED_SOURCE_ALLOWLIST_SUMMARY_SHA256,
        "source DVC-A1 v3 allowlist summary",
    )
    allowlist_entries = _validate_source_evidence(
        source_protocol_path=source_protocol_path,
        source_mask_manifest_path=source_mask_manifest_path,
        source_allowlist_path=source_allowlist_path,
        source_allowlist_summary_path=source_allowlist_summary_path,
        split_path=split_path,
    )

    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise DvgProtocolError(f"dataset root is missing: {dataset_root}")
    directories = template.raw["dataset"]
    for entry in allowlist_entries:
        sample_id = Path(entry).stem
        required = (
            dataset_root / str(directories["rgb_directory"]) / f"{sample_id}.jpg",
            dataset_root / str(directories["depth16_directory"]) / f"{sample_id}.png",
            dataset_root / str(directories["quantized_depth_directory"]) / f"{sample_id}.png",
            dataset_root / str(directories["label_directory"]) / f"{sample_id}.png",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise DvgProtocolError(f"missing required allowlisted modality for {sample_id}: {missing[0]}")

    evidence_root = evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    allowlist_summary_path = evidence_root / "allowlist-summary.json"
    allowlist_summary = {
        "schema_version": "museg-dvg-b1-allowlist-summary-v1",
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_protocol_path": str(source_protocol_path),
        "source_protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "source_mask_manifest_path": str(source_mask_manifest_path),
        "source_mask_manifest_sha256": EXPECTED_SOURCE_MASK_MANIFEST_SHA256,
        "source_allowlist_path": str(source_allowlist_path),
        "source_allowlist_sha256": EXPECTED_SOURCE_ALLOWLIST_SHA256,
        "source_allowlist_summary_path": str(source_allowlist_summary_path),
        "source_allowlist_summary_sha256": EXPECTED_SOURCE_ALLOWLIST_SUMMARY_SHA256,
        "source_split_sha256": EXPECTED_SPLIT_SHA256,
        "official_test_included": False,
        "included_sample_count": 218,
        "included_location_group_count": 138,
        "q75_zero_sample_count": 31,
        "fully_constructable_location_group_count": 123,
        "partially_constructable_location_group_count": 15,
    }
    _atomic_write_json(allowlist_summary_path, allowlist_summary)

    status = _git_output("status", "--porcelain=v1")
    source_hashes: dict[str, str] = {}
    for relative_path in SOURCE_IDENTITY_PATHS:
        source = REPO_ROOT / relative_path
        if not source.is_file():
            raise DvgProtocolError(f"missing P0 identity source {relative_path}")
        source_hashes[relative_path] = file_sha256(source)
    if any((REPO_ROOT / relative_path).exists() for relative_path in PLANNED_RUNTIME_PATHS):
        raise DvgProtocolError("P1 runtime source already exists; P0 identity must be reviewed before materialization")

    raw = copy.deepcopy(dict(template.raw))
    raw["model"]["checkpoint_path"] = str(checkpoint_path)
    raw["split"]["path"] = str(split_path)
    raw["dataset"]["root"] = str(dataset_root)
    raw["evidence"]["output_root"] = str(evidence_root)
    scope = raw["evaluation_scope"]
    scope["source_protocol_path"] = str(source_protocol_path)
    scope["source_mask_manifest_path"] = str(source_mask_manifest_path)
    scope["source_allowlist_path"] = str(source_allowlist_path)
    scope["source_allowlist_summary_path"] = str(source_allowlist_summary_path)
    scope["allowlist_summary_path"] = str(allowlist_summary_path)
    scope["allowlist_summary_sha256"] = file_sha256(allowlist_summary_path)
    raw["code_identity"] = {
        "identity_role": "p0-preregistration-baseline",
        "git_head": _git_output("rev-parse", "HEAD").strip(),
        "git_status_porcelain": status.splitlines(),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_sha256": source_hashes,
        "planned_runtime_paths": list(PLANNED_RUNTIME_PATHS),
        "planned_runtime_paths_present": False,
    }
    _atomic_write_json(output_path.resolve(), raw)
    return load_dvg_protocol(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-mask-manifest", type=Path, required=True)
    parser.add_argument("--source-allowlist", type=Path, required=True)
    parser.add_argument("--source-allowlist-summary", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = materialize_dvg_protocol(
            template_path=args.template,
            output_path=args.output,
            dataset_root=args.dataset_root,
            checkpoint_path=args.checkpoint,
            evidence_root=args.evidence_root,
            split_path=args.split,
            source_protocol_path=args.source_protocol,
            source_mask_manifest_path=args.source_mask_manifest,
            source_allowlist_path=args.source_allowlist,
            source_allowlist_summary_path=args.source_allowlist_summary,
        )
    except (DvgProtocolError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "protocol-blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {"status": "protocol-ready", "path": str(protocol.path), "sha256": protocol.sha256},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
