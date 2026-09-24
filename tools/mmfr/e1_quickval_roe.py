#!/usr/bin/env python3
"""MMFR E1 Batch 1B R-OE-lite Quick-Val: frozen four-condition single-view scoring
with the observable-empty routing inputs that the substitute requires.

Why this file exists
--------------------
``tools/mmfr/e1_quickval.py`` — the frozen Batch 1A/1B-comparison Quick-Val entry, SHA-256
``928c4229d937552e00128e9291b915204291f58c1e80b4b5a005efa9365e4073`` — forwards only
``model(rgb, depth)``.  An R-OE-lite checkpoint always builds
``ObservableEmptyGeometrySubstitute``, so ``models/builder.py::forward`` unconditionally
calls ``_route_roe_modal_x``, which fails closed when ``raw_depth`` and the geometry mask
are missing.  The frozen entry therefore cannot score this candidate at all.

``MMFR/01_research/r_oe_lite_design.md`` §11–§12 fixes what an R-OE-lite Quick-Val must
supply and report: the inference detector reads the *current corrupted* ``raw_depth``
(``uint8 / 255``, ``[B, 1, H, W]``, in ``[0, 1]``) and ``V_geom``; it must not be inferred
from the normalized Depth that reaches the network.  Routing has three outcomes
(``observable-empty-substitute``, ``nonempty-exact-bypass``, ``no-geometry-exact-bypass``),
and every condition must additionally report ``oe_trigger`` coverage, geometry pixels and
raw non-zero pixels.

This module is only that thin entry.  Nothing about the ``original-full`` input contract,
the four condition definitions, the frozen corruption seeding, the FP32/TF32-off forward,
the frozen forward-RNG policy, the logits restore or the metric computation is
reimplemented here: they are imported from ``tools.mmfr.e1_quickval``,
``tools.evaluate_museg_checkpoint``, ``tools.evaluate_museg_10condition`` and
``tools.evaluate_museg_checkpoint_fast`` unchanged.

Geometry mask definition
------------------------
The frozen Quick-Val view is ``original-full``: native resolution, no crop and no pad
(``view_protocol.padding = "none"``).  With no crop/pad there is no geometry-invalid
border, so ``V_geom`` is the unit mask over the whole native image.  That is recorded
explicitly as ``v_geom_definition`` in every artefact and is the only routing input this
entry invents; it follows directly from the frozen view, not from a new research choice.

Status
------
This entry point is **new** and carries no prior frozen qualification.  Every artefact it
writes records ``evaluator_identity="e1-quickval-original-full-single-view-roe-v1"`` and a
``deviation`` block so it can never be mistaken for the frozen ten-view evaluator or for
the frozen Batch 1A Quick-Val entry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import evaluate_museg_checkpoint as EV  # noqa: E402
from tools import evaluate_museg_10condition as E10  # noqa: E402
from tools import evaluate_museg_checkpoint_fast as FAST  # noqa: E402
from tools.mmfr import e1_quickval as QV  # noqa: E402

SCHEMA_VERSION = "mmfr-e1-quickval-roe-single-view-v1"
SUMMARY_SCHEMA_VERSION = "mmfr-e1-quickval-roe-comparison-v1"
EVALUATOR_IDENTITY = "e1-quickval-original-full-single-view-roe-v1"
V_GEOM_DEFINITION = (
    "unit mask at native resolution: the frozen original-full view applies no crop and no "
    "pad, so geometry-invalid pixels do not exist on the evaluation grid"
)

QUICKVAL_CONDITIONS = QV.QUICKVAL_CONDITIONS
HARD_CONDITIONS = QV.HARD_CONDITIONS
EXPECTED_SAMPLE_COUNT = QV.EXPECTED_SAMPLE_COUNT
EXPECTED_EVALUATION_SEED = QV.EXPECTED_EVALUATION_SEED

DEV = {
    "reason": (
        "the frozen R-OE-lite inference contract requires raw_depth and V_geom at the model call, "
        "which the frozen Batch 1A Quick-Val entry does not pass"
    ),
    "frozen_quickval_entry": "tools/mmfr/e1_quickval.py (sha256 928c4229d937552e00128e9291b915204291f58c1e80b4b5a005efa9365e4073)",
    "reused_unchanged": [
        "tools.evaluate_museg_checkpoint (input contract, forward precision, logits restore, metrics)",
        "tools.evaluate_museg_10condition (condition definitions, corruption seeding, sample reader)",
        "tools.evaluate_museg_checkpoint_fast (RNG capture/restore, peak memory)",
        "tools.mmfr.e1_quickval (frozen original-full input assembly and artefact helpers)",
    ],
    "added_by_this_entry": [
        "raw_depth and V_geom arguments on the model call",
        "oe routing telemetry from models.builder.EncoderDecoder.last_roe_route",
    ],
    "v_geom_definition": V_GEOM_DEFINITION,
    "qualification_status": "new-entry-point-no-prior-frozen-qualification",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--candidate", default=None, help="free-form label recorded in the artefacts")
    parser.add_argument("--config", default="local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1B_R_OE")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--conditions", default=",".join(QUICKVAL_CONDITIONS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forward-rng", choices=("reset-per-unit", "progressive"), default="reset-per-unit")
    parser.add_argument("--base-rng-seed", type=int, default=None)
    parser.add_argument("--gap-candidate", default="C0")
    parser.add_argument("--gap-experimental", default="R-OE-lite-v2")
    parser.add_argument("--compare", nargs="+", default=None, help="summary.json paths, in candidate order")
    parser.add_argument("--compare-output", type=Path, default=None)
    return parser.parse_args(argv)


def load_run_config(module_name: str) -> Any:
    return importlib.import_module(module_name).C


def build_roe_inputs(corrupted_depth_uint8: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the two inference routing inputs from the current corrupted Depth.

    ``raw_depth`` is the current corrupted Depth transported to ``[0, 1]`` without any
    reconstruction, and ``V_geom`` is the unit mask of the non-padded ``original-full``
    grid.  No cause, severity or generator metadata is read or inferred.
    """
    if corrupted_depth_uint8.ndim != 2:
        raise RuntimeError(f"corrupted Depth must be 2D, got shape {corrupted_depth_uint8.shape}")
    raw = corrupted_depth_uint8.astype(np.float32) / 255.0
    raw_depth = torch.from_numpy(np.ascontiguousarray(raw))[None, None].to(device=device, dtype=torch.float32)
    geometry = torch.ones_like(raw_depth, dtype=torch.bool)
    return raw_depth, geometry


def evaluate(
    args: argparse.Namespace,
    conditions: Sequence[E10.Condition],
    entries: Sequence[str],
    config: Any,
    checkpoint_sha256: str,
    split_sha256: str,
    channel_order: str,
) -> dict[str, Any]:
    device = torch.device(args.device)
    EV.configure_fp32_forward(device)
    model = EV.load_model(config, args.checkpoint, device)
    if getattr(model, "roe_substitute", None) is None:
        raise SystemExit(
            "this entry scores R-OE-lite candidates only, but the built model has no roe_substitute; "
            "check --config"
        )
    model.eval()
    num_classes = int(config.num_classes)
    background = int(config.background)
    class_names = list(config.class_names)
    evaluation_seed = EXPECTED_EVALUATION_SEED

    base_rng_seed = evaluation_seed if args.base_rng_seed is None else int(args.base_rng_seed)
    torch.manual_seed(base_rng_seed)
    torch.cuda.manual_seed_all(base_rng_seed)

    if args.forward_rng == "reset-per-unit":
        base_rng = FAST._capture_rng(device)
    else:
        base_rng = None

    condition_definition_sha256 = E10.json_sha256([condition.definition() for condition in conditions])

    started_at = dt.datetime.now(dt.timezone.utc)
    condition_payloads: dict[str, dict[str, Any]] = {}

    for completed, condition in enumerate(conditions, start=1):
        hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        per_sample: list[dict[str, Any]] = []
        depth_digests: list[str] = []
        condition_started = time.perf_counter()
        forward_seconds_total = 0.0
        trigger_samples = 0
        substitute_forward_samples = 0
        geometry_pixels_total = 0
        raw_depth_nonzero_pixels_total = 0
        no_geometry_bypass_samples = 0
        empty_route_samples = 0
        FAST._reset_peak_memory(device)

        for entry in entries:
            sample_id, rgb_uint8, depth_uint8, label = E10.read_sample(args.dataset_root, entry, channel_order)
            normalized_id = QV.normalize_sample_id(entry)
            outcome = E10.corrupt_depth(condition, rgb_uint8, depth_uint8, evaluation_seed, normalized_id)
            rgb_tensor, depth_tensor, label_tensor = QV.build_original_full_input(
                rgb_uint8,
                outcome.depth,
                label,
                config.norm_mean,
                config.norm_std,
                device,
            )
            raw_depth_tensor, geometry_tensor = build_roe_inputs(outcome.depth, device)
            if base_rng is not None:
                FAST._restore_rng(base_rng, device)
            unit_started = time.perf_counter()
            with torch.inference_mode():
                logits = model(
                    rgb_tensor,
                    depth_tensor,
                    raw_depth=raw_depth_tensor,
                    reliability_valid_mask=geometry_tensor,
                )
                route = getattr(model, "last_roe_route", None)
                if route is None:
                    raise RuntimeError("R-OE routing telemetry was not recorded for this forward")
                logits = EV.restore_logits_to_metric_grid(logits, label_tensor)
            FAST._sync(device)
            forward_seconds_total += time.perf_counter() - unit_started

            geometry_pixels = int(route["geometry_pixels"].sum().item())
            raw_nonzero_pixels = int(route["raw_depth_nonzero_pixels"].sum().item())
            trigger = bool(route["trigger"].sum().item())
            substitute_forwards = int(route["substitute_forward"])
            has_geometry = geometry_pixels > 0
            if trigger:
                trigger_samples += 1
                substitute_forward_samples += substitute_forwards
            elif not has_geometry:
                no_geometry_bypass_samples += 1
            else:
                empty_route_samples += 1
            geometry_pixels_total += geometry_pixels
            raw_depth_nonzero_pixels_total += raw_nonzero_pixels

            if tuple(int(value) for value in logits.shape[-2:]) != tuple(int(value) for value in label_tensor.shape[-2:]):
                raise RuntimeError("restored logits are not on the original Label grid")
            EV.update_confusion(hist, logits, label_tensor, num_classes, background)
            sample_hist = np.zeros((num_classes, num_classes), dtype=np.int64)
            EV.update_confusion(sample_hist, logits, label_tensor, num_classes, background)
            sample_metrics = EV.metrics_from_confusion(sample_hist, class_names)
            depth_digests.append(outcome.depth_sha256())
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "entry": entry,
                    "miou": sample_metrics["miou"],
                    "macc": sample_metrics["macc"],
                    "mf1": sample_metrics["mf1"],
                    "corrupted_depth_sha256": outcome.depth_sha256(),
                    "validity_state_sha256": outcome.state_sha256(),
                    "strict_no_op": bool(outcome.strict_no_op),
                    "rng_constructed": bool(outcome.rng_constructed),
                    "oe_route": (
                        "observable-empty-substitute"
                        if trigger
                        else ("no-geometry-exact-bypass" if not has_geometry else "nonempty-exact-bypass")
                    ),
                    "oe_trigger": trigger,
                    "oe_geometry_pixels": geometry_pixels,
                    "oe_raw_depth_nonzero_pixels": raw_nonzero_pixels,
                    "oe_substitute_forward": substitute_forwards,
                }
            )
            del rgb_tensor, depth_tensor, label_tensor, logits, outcome, raw_depth_tensor, geometry_tensor, route

        peak = FAST._peak_memory(device)
        metrics = EV.metrics_from_confusion(hist, class_names)
        sample_count = len(per_sample)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "evaluator_identity": EVALUATOR_IDENTITY,
            "deviation": DEV,
            "condition": {
                "condition_id": condition.condition_id,
                "directory": condition.directory,
                "group": condition.group,
                "index": condition.index,
                "specs": condition.spec_records() if hasattr(condition, "spec_records") else [],
            },
            "condition_definition_sha256": condition_definition_sha256,
            "call_index": completed,
            "sample_count": sample_count,
            "completed": sample_count == len(entries),
            "identity": {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "split": str(args.split),
                "split_sha256": split_sha256,
                "split_role": "val_dev",
                "config_module": args.config,
                "candidate": args.candidate,
                "evaluation_seed": evaluation_seed,
                "official_test_included": False,
            },
            "view_protocol": {
                "geometry": "original-full",
                "scale": 1.0,
                "flip": False,
                "view_count": 1,
                "padding": "none",
                "input_contract": "native-resolution RGB/Depth, Depth replicated to three channels",
                "forward_precision": "fp32",
                "metric_grid": "original label grid (no crop, no pad)",
                "forward_rng_policy": args.forward_rng,
                "base_rng_seed": base_rng_seed,
            },
            "roe_inference_contract": {
                "detector_inputs": ["current corrupted raw_depth in [0,1]", "V_geom"],
                "raw_depth_source": "current corrupted Depth uint8 / 255, no reconstruction",
                "raw_depth_shape": "[1, 1, H, W]",
                "v_geom_definition": V_GEOM_DEFINITION,
                "cause_severity_or_generator_metadata_read": False,
                "oe_semantics": "observable-empty",
            },
            "oe_routing": {
                "sample_count": sample_count,
                "trigger_samples": trigger_samples,
                "trigger_fraction": (trigger_samples / sample_count) if sample_count else None,
                "no_geometry_bypass_samples": no_geometry_bypass_samples,
                "nonempty_exact_bypass_samples": empty_route_samples,
                "substitute_forward_samples": substitute_forward_samples,
                "geometry_pixels_total": geometry_pixels_total,
                "raw_depth_nonzero_pixels_total": raw_depth_nonzero_pixels_total,
            },
            "metrics_percent": metrics,
            "confusion_matrix": hist.tolist(),
            "per_sample": per_sample,
            "corrupted_depth_aggregate_sha256": QV.canonical_digest(depth_digests),
            "timings": {
                "condition_seconds": round(time.perf_counter() - condition_started, 6),
                "forward_seconds": round(forward_seconds_total, 6),
            },
            "peak_device_memory": peak,
            "runtime": {
                "device": str(device),
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        target = args.output_dir / condition.directory / "metrics.json"
        QV.atomic_write_json(target, payload)
        condition_payloads[condition.directory] = payload
        print(
            f"[{completed}/{len(conditions)}] {condition.condition_id} miou={metrics['miou']} "
            f"trigger={trigger_samples}/{sample_count} elapsed={payload['timings']['condition_seconds']}s -> {target}",
            flush=True,
        )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "deviation": DEV,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "split": str(args.split),
            "split_sha256": split_sha256,
            "config_module": args.config,
            "candidate": args.candidate,
            "output_dir": str(args.output_dir),
            "official_test_included": False,
        },
        "conditions": {
            name: {
                "miou": payload["metrics_percent"]["miou"],
                "macc": payload["metrics_percent"]["macc"],
                "mf1": payload["metrics_percent"]["mf1"],
                "sample_count": payload["sample_count"],
                "metrics_path": str(args.output_dir / name / "metrics.json"),
                "corrupted_depth_aggregate_sha256": payload["corrupted_depth_aggregate_sha256"],
                "oe_routing": payload["oe_routing"],
            }
            for name, payload in condition_payloads.items()
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
    }
    QV.atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def compare(summary_paths: Sequence[Path], gap_candidate: str, gap_experimental: str) -> dict[str, Any]:
    """Apply the frozen §11 screening thresholds to two summaries (control, experimental).

    The control summary may come from the frozen Batch 1A Quick-Val entry: the four
    conditions, corruption seeding and forward-RNG policy are identical, so the clean and
    per-condition numbers are comparable.  The candidate-specific ``oe_routing`` block is
    reported for the experimental side only and never enters the threshold arithmetic.
    """
    if len(summary_paths) != 2:
        raise SystemExit("--compare requires exactly two summary.json paths (control, experimental)")
    control = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    experimental = json.loads(summary_paths[1].read_text(encoding="utf-8"))
    control_conditions = control["conditions"]
    experimental_conditions = experimental["conditions"]
    shared = [name for name in control_conditions if name in experimental_conditions]
    if not shared:
        raise SystemExit("the two summaries share no condition")
    per_condition = {}
    for name in shared:
        control_miou = float(control_conditions[name]["miou"])
        experimental_miou = float(experimental_conditions[name]["miou"])
        entry = {
            "control_miou": control_miou,
            "experimental_miou": experimental_miou,
            "delta_pp": round(experimental_miou - control_miou, 4),
        }
        if "oe_routing" in experimental_conditions[name]:
            entry["oe_routing"] = experimental_conditions[name]["oe_routing"]
        per_condition[name] = entry
    hard_present = [name for name in HARD_CONDITIONS if name in per_condition]
    if len(hard_present) != len(HARD_CONDITIONS):
        raise SystemExit(
            f"missing hard conditions in the comparison: {sorted(set(HARD_CONDITIONS) - set(hard_present))}"
        )
    control_hard = sum(per_condition[name]["control_miou"] for name in hard_present) / len(hard_present)
    experimental_hard = sum(per_condition[name]["experimental_miou"] for name in hard_present) / len(hard_present)
    delta_f = experimental_hard - control_hard
    clean_delta = per_condition["clean"]["delta_pp"] if "clean" in per_condition else None
    per_hard_ok = all(per_condition[name]["delta_pp"] >= -0.50 for name in hard_present)
    any_hard_stop = any(per_condition[name]["delta_pp"] < -1.00 for name in hard_present)
    if delta_f <= 0.0 or (clean_delta is not None and clean_delta < -0.50) or any_hard_stop:
        verdict = "stop"
    elif delta_f >= 0.50 and (clean_delta is None or clean_delta >= -0.25) and per_hard_ok:
        verdict = "promote"
    else:
        verdict = "inconclusive"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "deviation": DEV,
        "candidates": {
            "control": {
                "label": gap_candidate,
                "summary": str(summary_paths[0]),
                "evaluator_identity": control.get("evaluator_identity"),
                "config_module": control["identity"].get("config_module"),
                "checkpoint_sha256": control["identity"].get("checkpoint_sha256"),
            },
            "experimental": {
                "label": gap_experimental,
                "summary": str(summary_paths[1]),
                "evaluator_identity": experimental.get("evaluator_identity"),
                "config_module": experimental["identity"].get("config_module"),
                "checkpoint_sha256": experimental["identity"].get("checkpoint_sha256"),
            },
        },
        "per_condition": per_condition,
        "hard_conditions": hard_present,
        "m3_hard": {
            "definition": "(entire_missing@1.0 + spatial_dropout@0.75 + misalignment@0.75) / 3",
            "control": round(control_hard, 4),
            "experimental": round(experimental_hard, 4),
            "delta_pp": round(delta_f, 4),
        },
        "gate": {
            "source": "MMFR/01_research/e1_batch1_protocol.md §11 (single-view Quick-Val screening)",
            "scope": "screening only; cannot produce the Batch 1B promote/stop verdict that Main-Val is reserved for",
            "promote_rule": "delta_M3_hard >= +0.50 pp; clean >= -0.25 pp; every hard condition >= -0.50 pp",
            "stop_rule": "delta_M3_hard <= 0; or clean < -0.50 pp; or any hard condition < -1.00 pp",
            "otherwise": "inconclusive",
            "observed": {
                "delta_M3_hard_pp": round(delta_f, 4),
                "clean_delta_pp": clean_delta,
                "hard_deltas_pp": {name: per_condition[name]["delta_pp"] for name in hard_present},
            },
            "verdict": verdict,
        },
        "official_test_included": False,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        report = compare([Path(item) for item in args.compare], args.gap_candidate, args.gap_experimental)
        target = args.compare_output or Path("quickval-comparison.json")
        QV.atomic_write_json(target, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return 0

    missing = [
        name
        for name in (
            "checkpoint",
            "dataset_root",
            "split",
            "output_dir",
            "expected_checkpoint_sha256",
            "expected_split_sha256",
        )
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join('--' + item.replace('_', '-') for item in missing)}")

    config = load_run_config(args.config)
    channel_order = str(getattr(config, "channel_order", "RGB"))

    checkpoint_sha256 = EV.file_sha256(args.checkpoint)
    if checkpoint_sha256.lower() != str(args.expected_checkpoint_sha256).lower():
        raise SystemExit(f"checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, got {checkpoint_sha256}")
    split_sha256 = EV.file_sha256(args.split)
    if split_sha256.lower() != str(args.expected_split_sha256).lower():
        raise SystemExit(f"split SHA-256 mismatch: expected {args.expected_split_sha256}, got {split_sha256}")
    entries = EV.split_entries(args.split)
    if len(entries) != EXPECTED_SAMPLE_COUNT:
        raise SystemExit(f"val-dev split has {len(entries)} samples, expected {EXPECTED_SAMPLE_COUNT}")

    all_conditions = E10.build_conditions(E10.FROZEN_EVALUATION)
    conditions = E10.resolve_conditions(args.conditions, all_conditions)
    print(
        f"E1 R-OE-lite Quick-Val single-view: candidate={args.candidate} "
        f"conditions={[c.condition_id for c in conditions]} samples={len(entries)} "
        f"checkpoint={checkpoint_sha256[:16]}…",
        flush=True,
    )
    evaluate(args, conditions, entries, config, checkpoint_sha256, split_sha256, channel_order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
