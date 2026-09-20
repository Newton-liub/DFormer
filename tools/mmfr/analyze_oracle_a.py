#!/usr/bin/env python3
"""Analyse ``MMFR-Oracle-A`` evidence: tables, deltas, paired bootstrap, win/loss counts.

Reads the per-``(variant, condition)`` ``metrics.json`` files written by
``tools.mmfr.run_oracle_a`` and produces:

* the headline mIoU table and the ``Oracle - Original`` delta table;
* per-class IoU and per-class delta tables;
* per-sample win / tie / loss counts on image mIoU;
* the paired location-group bootstrap of ``Oracle - Original`` mIoU, using the project's
  frozen pre-registered rule (``tools.mve.dvc_a1_core.paired_effect``: group = first four
  filename segments, group value = mean image mIoU, 10000 replicates, seed 20260908);
* a concentration check that the gain is not produced by a handful of images;
* the pre-registered Oracle-A go/no-go decision.

Nothing here changes any evidence file; it only reads them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import tools.evaluate_museg_10condition as E10
from tools.mve.dvc_a1_core import paired_effect

DEFAULT_OUTPUT_DIR = Path("experiments/MMFR_OracleA/eval")
HEADLINE_VARIANTS = ("original", "strict", "aggregated", "geometry_off")
GO_SPATIAL_GAIN = 0.50
GO_CLEAN_TOLERANCE = -0.50
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260908


def _load(output_dir: Path, variant: str, condition: str) -> dict[str, Any]:
    path = output_dir / variant / condition / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_directories() -> dict[str, str]:
    return {
        str(condition.condition_id): str(condition.directory)
        for condition in E10.build_conditions(E10.FROZEN_EVALUATION)
    }


def _per_class_map(payload: Mapping[str, Any]) -> dict[str, float]:
    return {str(entry["name"]): float(entry["iou"]) for entry in payload["metrics_percent"]["per_class"]}


def _sample_deltas(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference_by_id = {record["sample_id"]: record for record in reference["per_sample"]}
    rows: list[dict[str, Any]] = []
    for record in candidate["per_sample"]:
        left = reference_by_id[record["sample_id"]]
        rows.append(
            {
                "sample_id": record["sample_id"],
                "location_group": record["location_group"],
                "miou_original": float(left["miou"]),
                "miou_variant": float(record["miou"]),
                "delta": float(record["miou"]) - float(left["miou"]),
            }
        )
    return rows


def _concentration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = np.sort(np.asarray([float(row["delta"]) for row in rows], dtype=np.float64))
    total = float(deltas.sum())
    positives = deltas[deltas > 0]
    negative_total = float(deltas[deltas < 0].sum())
    count = max(1, int(math.ceil(0.05 * deltas.size)))
    top_share = float(deltas[::-1][:count].sum())
    return {
        "sum_delta_miou_points": total,
        "sum_positive_delta_points": float(positives.sum()),
        "sum_negative_delta_points": negative_total,
        "top_5pct_sample_share_of_sum": (top_share / total) if total != 0 else None,
        "top_5pct_delta_sum": top_share,
        "median_delta": float(np.median(deltas)),
        "mean_delta": float(deltas.mean()),
        "min_delta": float(deltas.min()),
        "max_delta": float(deltas.max()),
    }


def analyse(output_dir: Path, variants: Sequence[str], conditions: Sequence[str]) -> dict[str, Any]:
    directories = _condition_directories()
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in variants:
        for condition in conditions:
            payloads[(variant, condition)] = _load(output_dir, variant, directories[condition])

    table = {
        variant: {
            condition: {
                "miou": float(payloads[(variant, condition)]["metrics_percent"]["miou"]),
                "macc": float(payloads[(variant, condition)]["metrics_percent"]["macc"]),
                "mf1": float(payloads[(variant, condition)]["metrics_percent"]["mf1"]),
            }
            for condition in conditions
        }
        for variant in variants
    }
    deltas = {
        variant: {
            condition: round(table[variant][condition]["miou"] - table["original"][condition]["miou"], 4)
            for condition in conditions
        }
        for variant in variants
        if variant != "original"
    }

    per_class: dict[str, dict[str, dict[str, float]]] = {}
    for variant in variants:
        per_class[variant] = {
            condition: _per_class_map(payloads[(variant, condition)]) for condition in conditions
        }
    per_class_delta = {
        variant: {
            condition: {
                name: round(per_class[variant][condition][name] - per_class["original"][condition][name], 4)
                for name in per_class["original"][condition]
            }
            for condition in conditions
        }
        for variant in variants
        if variant != "original"
    }

    sample_analysis: dict[str, dict[str, Any]] = {}
    for variant in variants:
        if variant == "original":
            continue
        sample_analysis[variant] = {}
        for condition in conditions:
            rows = _sample_deltas(payloads[("original", condition)], payloads[(variant, condition)])
            delta_values = np.asarray([row["delta"] for row in rows], dtype=np.float64)
            group_values: dict[str, dict[str, float]] = {}
            for row in rows:
                # ``paired_effect`` follows the frozen DVC-A1 convention: its inputs are
                # fractions and it multiplies by 100 itself.  mIoU is stored in percent,
                # so convert here to keep the reported interval in percentage points.
                group_values.setdefault(row["location_group"], {})["original"] = row["miou_original"] / 100.0
                group_values.setdefault(row["location_group"], {})[variant] = row["miou_variant"] / 100.0
            bootstrap = paired_effect(
                group_values,
                variant,
                "original",
                bootstrap_seed=BOOTSTRAP_SEED,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
            )
            per_group = np.asarray(
                [value for _group, value in sorted(bootstrap["per_group_percentage_points"].items())],
                dtype=np.float64,
            )
            group_count = max(1, int(math.ceil(0.05 * per_group.size)))
            group_total = float(per_group.sum())
            sample_analysis[variant][condition] = {
                "sample_count": len(rows),
                "win": int(np.count_nonzero(delta_values > 0)),
                "tie": int(np.count_nonzero(delta_values == 0)),
                "loss": int(np.count_nonzero(delta_values < 0)),
                "mean_image_delta": float(delta_values.mean()),
                "median_image_delta": float(np.median(delta_values)),
                "paired_group_bootstrap_miou": {
                    key: value for key, value in bootstrap.items() if key != "per_group_percentage_points"
                },
                "paired_group_count": bootstrap["paired_group_count"],
                "group_level_concentration": {
                    "sum_group_delta_points": group_total,
                    "top_5pct_group_share_of_sum": (float(np.sort(per_group)[::-1][:group_count].sum()) / group_total)
                    if group_total != 0
                    else None,
                    "median_group_delta": float(np.median(per_group)),
                },
                "concentration": _concentration(rows),
            }

    decision = _decide(deltas, sample_analysis)
    return {
        "schema_version": "mmfr-oracle-a-analysis-v1",
        "output_dir": str(output_dir),
        "variants": list(variants),
        "conditions": list(conditions),
        "table_miou": table,
        "delta_miou_vs_original": deltas,
        "per_class_iou": per_class,
        "per_class_iou_delta_vs_original": per_class_delta,
        "sample_level": sample_analysis,
        "bootstrap_rule": {
            "source": "tools.mve.dvc_a1_core.paired_effect (project pre-registered rule)",
            "unit": "location group = first four filename segments",
            "group_value": "mean image mIoU inside the group",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "2.5 / 97.5 percentile, linear interpolation",
        },
        "decision": decision,
        "official_test_included": False,
        "timestamp_utc": E10.now_utc(),
    }


def _decide(deltas: Mapping[str, Mapping[str, float]], sample_analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered Oracle-A screening rule, evaluated only from the measured numbers."""
    spatial = float(deltas.get("strict", {}).get("spatial_dropout@0.75", float("nan")))
    spatial_interval = (
        sample_analysis.get("strict", {})
        .get("spatial_dropout@0.75", {})
        .get("paired_group_bootstrap_miou", {})
        .get("interval95_percentage_points")
    )
    clean = float(deltas.get("strict", {}).get("clean", float("nan")))
    checks = {
        "strict_spatial_gain_at_least_0.50": bool(spatial >= GO_SPATIAL_GAIN),
        "strict_spatial_interval_lower_above_zero": bool(
            spatial_interval is not None and float(spatial_interval[0]) > 0.0
        ),
        "strict_clean_loss_within_0.50": bool(clean >= GO_CLEAN_TOLERANCE),
    }
    best_alternative = max(
        (
            float(values.get("spatial_dropout@0.75", float("-inf")))
            for name, values in deltas.items()
            if name != "original"
        ),
        default=float("-inf"),
    )
    if all(checks.values()):
        verdict = "GO"
    else:
        verdict = "NO-GO"
    return {
        "verdict": verdict,
        "criteria": {
            "spatial_dropout_gain_threshold_points": GO_SPATIAL_GAIN,
            "clean_loss_tolerance_points": GO_CLEAN_TOLERANCE,
            "interval": "paired location-group bootstrap 95% percentile interval must exclude 0",
        },
        "checks": checks,
        "measured": {
            "strict_spatial_dropout_delta": spatial,
            "strict_spatial_dropout_interval95": spatial_interval,
            "strict_clean_delta": clean,
            "best_non_reference_spatial_dropout_delta": best_alternative,
        },
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    conditions = list(analysis["conditions"])
    header = "| Variant | " + " | ".join(conditions) + " |"
    separator = "|---" * (len(conditions) + 1) + "|"
    lines = [header, separator]
    for variant in analysis["variants"]:
        row = [variant]
        for condition in conditions:
            row.append(f"{analysis['table_miou'][variant][condition]['miou']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("| Variant | " + " | ".join(f"d({condition})" for condition in conditions) + " |")
    lines.append(separator)
    for variant, values in analysis["delta_miou_vs_original"].items():
        row = [variant]
        for condition in conditions:
            row.append(f"{values[condition]:+.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", default=",".join(HEADLINE_VARIANTS))
    parser.add_argument("--conditions", default="clean,spatial_dropout@0.75,entire_missing@1.0")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    variants = [token.strip() for token in args.variants.split(",") if token.strip()]
    conditions = [token.strip() for token in args.conditions.split(",") if token.strip()]
    analysis = analyse(args.output_dir, variants, conditions)
    target = args.output_dir / "oracle-a-analysis.json"
    E10.atomic_write_json(target, analysis)
    markdown = render_markdown(analysis)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print()
    print(json.dumps(analysis["decision"], indent=2, ensure_ascii=False))
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
