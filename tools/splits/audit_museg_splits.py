"""Independently recompute and audit a MUSeg development split candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def current_tool_sources(repo: Path) -> dict[str, str]:
    logical_names = (
        "tools/splits/create_museg_dev_split.py",
        "tools/splits/audit_museg_splits.py",
    )
    return {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in logical_names}


def _canonical_lf_list(raw: bytes) -> bool:
    return bool(raw) and not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def audit(dataset_root: Path, official_train: Path, official_test: Path, root: Path, write: bool = False) -> dict[str, Any]:
    """Read every authoritative input again; never use manifest values as audit inputs."""
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    manifest_path = root / "manifest.json"
    try:
        from create_museg_dev_split import (
            ALGORITHM_VERSION, PROTOCOL_ID, SCHEMA_VERSION, SEED, SEED_HASH, SEED_STRING,
            TARGET, TEST_COUNT, TEST_GROUPS, TEST_HASH, TRAIN_COUNT, TRAIN_GROUPS, TRAIN_HASH,
            git_object_is_commit, hard_constraint_checks, inventory, load_samples,
            mine_side_counts, objective_details, optimize, output_lines, parse_official_bytes,
            parse_path, rare_class_list, set_relationships, statistic_definitions,
            threshold_evaluation, totals, verify_runtime,
        )

        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw.decode("utf-8"))
        train_raw, test_raw = official_train.read_bytes(), official_test.read_bytes()
        official_train_paths = parse_official_bytes(train_raw)
        official_test_paths = parse_official_bytes(test_raw)
        train_dev_raw = (root / "train-dev.txt").read_bytes()
        val_dev_raw = (root / "val-dev.txt").read_bytes()
        copied_test_raw = (root / "official-test.txt").read_bytes()
        train_dev_paths = parse_official_bytes(train_dev_raw)
        val_dev_paths = parse_official_bytes(val_dev_raw)
        copied_test_paths = parse_official_bytes(copied_test_raw)

        official_train_groups = {parse_path(path)[0] for path in official_train_paths}
        official_test_groups = {parse_path(path)[0] for path in official_test_paths}
        train_dev_groups = {parse_path(path)[0] for path in train_dev_paths}
        val_dev_groups = {parse_path(path)[0] for path in val_dev_paths}

        checks["encoding.manifest_canonical_json"] = manifest_raw == canonical(manifest)
        checks["encoding.train_dev_canonical_lf"] = _canonical_lf_list(train_dev_raw)
        checks["encoding.val_dev_canonical_lf"] = _canonical_lf_list(val_dev_raw)
        checks["encoding.official_test_byte_copy"] = copied_test_raw == test_raw and copied_test_paths == official_test_paths
        actual_files = {path.name for path in root.iterdir() if path.is_file()}
        required_files = {"train-dev.txt", "val-dev.txt", "official-test.txt", "manifest.json"}
        checks["outputs.allowed_file_set"] = required_files <= actual_files and actual_files <= required_files | {"audit-report.json"}

        required_top = {
            "schema_version", "protocol_id", "algorithm_version", "seed_source_string",
            "seed_source_sha256", "seed_uint32_big_endian", "target_val_images",
            "generator_git_commit", "tool_sources", "generator_command", "dependencies", "official",
            "inventory", "outputs", "set_relationships", "group_key_rule", "mine_rule",
            "mine_counts", "statistic_definitions", "statistics", "objective", "local_search",
            "rare_classes", "threshold_checks", "warnings", "hard_constraint_checks",
            "candidate_status", "user_gate_a", "audit_report",
        }
        is_frozen = manifest.get("candidate_status") == "frozen"
        if is_frozen:
            required_top.add("freeze_metadata")
        for field in sorted(required_top):
            checks[f"schema.required.{field}"] = field in manifest
        checks["schema.no_unknown_top_level_fields"] = set(manifest) == required_top
        checks["schema.identity"] = manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("protocol_id") == PROTOCOL_ID and manifest.get("algorithm_version") == ALGORITHM_VERSION
        checks["schema.seed_fields"] = all(field in manifest for field in ("seed_source_string", "seed_source_sha256", "seed_uint32_big_endian", "target_val_images"))
        checks["schema.official_fields"] = set(manifest.get("official", {})) == {"train", "test", "sample_overlap", "group_overlap"} and all(set(manifest.get("official", {}).get(side, {})) == {"logical_name", "sha256", "samples", "groups"} for side in ("train", "test"))
        checks["schema.inventory_fields"] = set(manifest.get("inventory", {})) == {"RGB", "Depth16", "Label"} and all(set(manifest.get("inventory", {}).get(name, {})) == {"sha256", "files", "bytes", "serialization"} for name in ("RGB", "Depth16", "Label"))
        checks["schema.output_fields"] = set(manifest.get("outputs", {})) == {"train-dev.txt", "val-dev.txt", "official-test.txt"} and all(set(manifest.get("outputs", {}).get(name, {})) == {"sha256", "samples", "groups", "byte_policy"} for name in ("train-dev.txt", "val-dev.txt", "official-test.txt"))
        checks["schema.set_relationship_fields"] = set(manifest.get("set_relationships", {})) == {"samples", "groups"}
        checks["schema.sample_relationship_detail_fields"] = set(manifest.get("set_relationships", {}).get("samples", {})) == {"duplicates", "dev_union_vs_official_train", "train_dev__val_dev", "official_train__official_test", "train_dev__official_test", "val_dev__official_test"}
        checks["schema.group_relationship_detail_fields"] = set(manifest.get("set_relationships", {}).get("groups", {})) == {"dev_union_vs_official_train", "train_dev__val_dev", "official_train__official_test", "train_dev__official_test", "val_dev__official_test"}
        checks["schema.mine_count_fields"] = set(manifest.get("mine_counts", {})) == {"train_dev", "val_dev"} and all(set(manifest.get("mine_counts", {}).get(side, {})) == {"01", "02", "03", "04", "05", "06"} and all(set(manifest.get("mine_counts", {}).get(side, {}).get(mine, {})) == {"images", "groups"} for mine in ("01", "02", "03", "04", "05", "06")) for side in ("train_dev", "val_dev"))
        checks["schema.statistic_definition_fields"] = set(manifest.get("statistic_definitions", {})) == {"label", "depth16", "rgb_luma", "class_count", "bins"}
        checks["schema.fixed_bin_fields"] = set(manifest.get("statistic_definitions", {}).get("bins", {})) == {"depth16_valid_ratio", "rgb_normalized_mean_luma", "foreground_class_count"}
        checks["schema.statistics_fields"] = set(manifest.get("statistics", {})) == {"official_train", "train_dev", "val_dev"}
        checks["schema.statistic_aggregate_fields"] = all(set(manifest.get("statistics", {}).get(side, {})) == {"images", "pixels", "mine", "presence", "class_pixels", "background", "depth_valid", "y_num", "depth_hist", "luma_hist", "class_hist"} for side in ("official_train", "train_dev", "val_dev"))
        checks["schema.objective_fields"] = set(manifest.get("objective", {})) == {"six_tuple", "levels", "subitems"}
        checks["schema.objective_level_fields"] = set(manifest.get("objective", {}).get("levels", {})) == {"mine", "presence", "pixel", "background", "aux", "size"}
        checks["schema.objective_subitem_fields"] = set(manifest.get("objective", {}).get("subitems", {})) == {"mine", "presence", "pixel", "background", "aux", "depth16_valid_ratio_histogram_bins", "rgb_normalized_mean_luma_histogram_bins", "foreground_class_count_histogram_bins"}
        checks["schema.rare_classes_list"] = isinstance(manifest.get("rare_classes"), list) and all(set(item) == {"class_id", "images", "groups", "pixels", "cannot_bilateral_cover"} for item in manifest.get("rare_classes", []))
        checks["schema.threshold_fields"] = set(manifest.get("threshold_checks", {})) == {"requires_sol_review", "review_status", "checks"}
        checks["schema.threshold_check_detail_fields"] = set(manifest.get("threshold_checks", {}).get("checks", {})) == {"eligible_class_presence_zero", "class_presence_error_gt_1_2", "class_pixel_error_gt_1_2", "all_background_error_gt_1_2", "aux_component_error_gt_1_3", "val_image_deviation_gt_limit"}
        checks["schema.hard_constraint_fields"] = set(manifest.get("hard_constraint_checks", {})) == {"official_train_frozen_identity", "official_test_frozen_identity", "frozen_seed_and_target", "all_train_modalities_valid", "each_mine_has_at_least_two_official_train_groups", "group_unsplit", "sample_set_closed", "group_set_closed", "official_test_samples_isolated", "official_test_groups_isolated", "mine_bilateral_images", "mine_bilateral_groups", "fixed_unique_candidate_membership", "optimizer_converged", "all_pass"}
        checks["schema.state_fields"] = set(manifest.get("user_gate_a", {})) == {"status", "signed_by", "signature_reference"}
        if is_frozen:
            freeze_metadata = manifest.get("freeze_metadata", {})
            checks["schema.freeze_metadata_fields"] = set(freeze_metadata) == {"source_candidate_manifest_sha256", "signed_by", "signature_reference"}
            checks["schema.freeze_metadata_values"] = (
                isinstance(freeze_metadata.get("source_candidate_manifest_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", freeze_metadata.get("source_candidate_manifest_sha256", "")) is not None
                and freeze_metadata.get("signed_by") == manifest.get("user_gate_a", {}).get("signed_by")
                and freeze_metadata.get("signature_reference") == manifest.get("user_gate_a", {}).get("signature_reference")
            )

        checks["frozen.input_train"] = (hashlib.sha256(train_raw).hexdigest(), len(official_train_paths), len(official_train_groups)) == (TRAIN_HASH, TRAIN_COUNT, TRAIN_GROUPS)
        checks["frozen.input_test"] = (hashlib.sha256(test_raw).hexdigest(), len(official_test_paths), len(official_test_groups)) == (TEST_HASH, TEST_COUNT, TEST_GROUPS)
        checks["frozen.parameters"] = manifest.get("seed_source_string") == SEED_STRING and manifest.get("seed_source_sha256") == SEED_HASH and manifest.get("seed_uint32_big_endian") == SEED and manifest.get("target_val_images") == TARGET
        repo = Path(__file__).resolve().parents[2]
        recorded_commit = manifest.get("generator_git_commit")
        checks["generator.commit"] = git_object_is_commit(repo, recorded_commit)
        expected_sources = current_tool_sources(repo)
        checks["schema.tool_source_fields"] = (
            set(manifest.get("tool_sources", {})) == set(expected_sources)
            and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None for value in manifest.get("tool_sources", {}).values())
        )
        checks["generator.tool_sources"] = manifest.get("tool_sources") == expected_sources
        checks["generator.command"] = manifest.get("generator_command") == "python tools/splits/create_museg_dev_split.py --dataset-root <dataset-root> --official-train <official-train> --official-test <official-test> --output-root <output-root> --seed 1181652136 --target-val-images 319"
        checks["generator.runtime"] = manifest.get("dependencies") == verify_runtime()

        relationships = set_relationships(official_train_paths, official_test_paths, train_dev_paths, val_dev_paths)
        checks["relationships.recomputed"] = manifest.get("set_relationships") == relationships
        checks["relationships.official_isolation"] = relationships["samples"]["official_train__official_test"]["count"] == 0 and relationships["groups"]["official_train__official_test"]["count"] == 0

        samples = load_samples(dataset_root, official_train_paths)
        groups: dict[str, list[Any]] = {}
        for sample in samples:
            groups.setdefault(sample.group, []).append(sample)
        optimized_selected, accepted_steps, stop_reason = optimize(groups)
        selected_matches = val_dev_groups == optimized_selected and train_dev_groups == set(groups) - optimized_selected
        per_mine = mine_side_counts(groups, val_dev_groups)
        train_identity = (hashlib.sha256(train_raw).hexdigest(), len(official_train_paths), len(official_train_groups)) == (TRAIN_HASH, TRAIN_COUNT, TRAIN_GROUPS)
        test_identity = (hashlib.sha256(test_raw).hexdigest(), len(official_test_paths), len(official_test_groups)) == (TEST_HASH, TEST_COUNT, TEST_GROUPS)
        parameters_frozen = manifest.get("seed_source_string") == SEED_STRING and manifest.get("seed_source_sha256") == SEED_HASH and manifest.get("seed_uint32_big_endian") == SEED and manifest.get("target_val_images") == TARGET
        mine_group_minimum = all(sum(1 for group in groups if groups[group][0].mine == mine) >= 2 for mine in ("01", "02", "03", "04", "05", "06"))
        hard = hard_constraint_checks(
            relationships,
            per_mine,
            selected_matches,
            official_train_identity=train_identity,
            official_test_identity=test_identity,
            frozen_parameters=parameters_frozen,
            modalities_valid=True,
            mine_group_minimum=mine_group_minimum,
            optimizer_converged=stop_reason == "no_strict_improvement" and accepted_steps <= 10000,
        )
        for name, value in hard.items():
            checks[f"hard_constraint.{name}"] = value
        checks["hard_constraint.manifest_matches"] = manifest.get("hard_constraint_checks") == hard

        total = totals(samples)
        train_total = totals(sample for group in train_dev_groups for sample in groups[group])
        val_total = totals(sample for group in val_dev_groups for sample in groups[group])
        expected_statistics = {"official_train": total, "train_dev": train_total, "val_dev": val_total}
        checks["statistics.definitions"] = manifest.get("statistic_definitions") == statistic_definitions()
        checks["statistics.aggregates"] = manifest.get("statistics") == expected_statistics
        expected_objective = objective_details(val_total, total)
        checks["statistics.objective_all_subitems"] = manifest.get("objective") == expected_objective
        expected_rare = rare_class_list(samples)
        checks["statistics.rare_classes"] = manifest.get("rare_classes") == expected_rare
        expected_thresholds, expected_warnings = threshold_evaluation(samples, val_total, total, groups)
        checks["thresholds.all_checks"] = manifest.get("threshold_checks") == expected_thresholds
        checks["thresholds.warnings"] = manifest.get("warnings") == expected_warnings

        checks["mines.images_and_groups"] = manifest.get("mine_counts") == per_mine
        checks["rules.group_key"] = manifest.get("group_key_rule") == "first four ASCII hyphen-separated stem segments"
        checks["rules.mine"] = manifest.get("mine_rule") == "first stem segment; 01..06"
        checks["algorithm.membership"] = selected_matches
        checks["algorithm.local_search"] = manifest.get("local_search") == {"accepted_steps": accepted_steps, "stop_reason": stop_reason}
        checks["algorithm.output_order"] = train_dev_raw == output_lines(groups, set(groups) - optimized_selected) and val_dev_raw == output_lines(groups, optimized_selected)

        expected_official = {
            "train": {"logical_name": "train.txt", "sha256": hashlib.sha256(train_raw).hexdigest(), "samples": len(official_train_paths), "groups": len(official_train_groups)},
            "test": {"logical_name": "test.txt", "sha256": hashlib.sha256(test_raw).hexdigest(), "samples": len(official_test_paths), "groups": len(official_test_groups)},
            "sample_overlap": relationships["samples"]["official_train__official_test"],
            "group_overlap": relationships["groups"]["official_train__official_test"],
        }
        checks["official.manifest_values"] = manifest.get("official") == expected_official
        checks["inventory.all_modalities"] = manifest.get("inventory") == {name: inventory(dataset_root, official_train_paths, name) for name in ("RGB", "Depth16", "Label")}
        expected_outputs = {
            "train-dev.txt": {"sha256": hashlib.sha256(train_dev_raw).hexdigest(), "samples": len(train_dev_paths), "groups": len(train_dev_groups), "byte_policy": "utf8_no_bom_lf_one_terminal_newline"},
            "val-dev.txt": {"sha256": hashlib.sha256(val_dev_raw).hexdigest(), "samples": len(val_dev_paths), "groups": len(val_dev_groups), "byte_policy": "utf8_no_bom_lf_one_terminal_newline"},
            "official-test.txt": {"sha256": hashlib.sha256(copied_test_raw).hexdigest(), "samples": len(copied_test_paths), "groups": len({parse_path(path)[0] for path in copied_test_paths}), "byte_policy": "official_input_byte_for_byte_copy"},
        }
        checks["outputs.hashes_counts_groups_policies"] = manifest.get("outputs") == expected_outputs
        gate = manifest.get("user_gate_a", {})
        pending_state = manifest.get("candidate_status") == "candidate" and gate == {"status": "pending", "signed_by": None, "signature_reference": None}
        frozen_state = (
            manifest.get("candidate_status") == "frozen"
            and gate.get("status") == "approved"
            and isinstance(gate.get("signed_by"), str) and bool(gate.get("signed_by"))
            and isinstance(gate.get("signature_reference"), str) and bool(gate.get("signature_reference"))
        )
        checks["state.valid_candidate_or_frozen"] = pending_state or frozen_state
        checks["state.audit_report_logical_name"] = manifest.get("audit_report") == "audit-report.json"

        details = {
            "counts": {"official_train": len(official_train_paths), "train_dev": len(train_dev_paths), "val_dev": len(val_dev_paths), "official_test": len(official_test_paths)},
            "groups": {"official_train": len(official_train_groups), "train_dev": len(train_dev_groups), "val_dev": len(val_dev_groups), "official_test": len(official_test_groups)},
            "manifest_sha256": sha(manifest_path),
            "objective_six_tuple": expected_objective["six_tuple"],
            "rare_class_count": len(expected_rare),
            "requires_sol_review": expected_thresholds["requires_sol_review"],
            "warning_count": len(expected_warnings),
        }
    except Exception as error:
        checks["audit.exception_free"] = False
        details = {"error": str(error), "manifest_sha256": sha(manifest_path) if manifest_path.is_file() else None}

    report = {
        "checks": checks,
        "details": details,
        "manifest_sha256": details.get("manifest_sha256"),
        "pass": bool(checks) and all(checks.values()),
        "schema_version": "museg-dev-split-audit-v1",
    }
    report_path = root / "audit-report.json"
    if write:
        report_path.write_bytes(canonical(report))
    elif report_path.is_file():
        try:
            existing_raw = report_path.read_bytes()
            existing_value = json.loads(existing_raw.decode("utf-8"))
            existing_matches = existing_raw == canonical(existing_value) and existing_raw == canonical(report)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            existing_matches = False
        if not existing_matches:
            report = {**report, "checks": {**report["checks"], "existing_report.matches_recomputed": False}, "pass": False}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args.dataset_root, args.official_train, args.official_test, args.output_root, args.write_report)
    print(canonical(report).decode("utf-8"), end="")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
