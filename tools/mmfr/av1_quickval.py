#!/usr/bin/env python3
"""A-v1's authorized single-pass, four-condition Quick-Val evaluator.

This is a thin runner over the frozen original-full input assembly, corruption,
strict model restore, logits-grid restore, metrics, and RNG helpers. It evaluates
off/full/learned with one shared input and one replayed pre-forward RNG state per
(sample, condition), and records the first-sample strict off-vs-C0 identity review
inside the real evaluation pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import platform
import re
import subprocess
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
from utils.dataloader.mmfr_training import normalize_sample_id  # noqa: E402

SCHEMA_VERSION = "mmfr-a-v1-quickval-four-condition-v1"
EVALUATOR_IDENTITY = "mmfr-a-v1-quickval-original-full-single-view-v1"
CONFIG_MODULE = "local_configs.MUSeg.DFormerv2_S_MMFR_AV1"
EXPECTED_SAMPLE_COUNT = 318
EVALUATION_SEED = 2026091401
BEHAVIORS = ("off", "full", "learned")
CONDITION_IDS = (
    "clean",
    "entire_missing@1.0",
    "spatial_dropout@0.75",
    "misalignment@0.75",
)
HARD_CONDITION_IDS = (
    "entire_missing@1.0",
    "spatial_dropout@0.75",
    "misalignment@0.75",
)
CONTINUATION_RULES = {
    "learned_hard_gain_vs_off": {"operator": ">=", "threshold_pp": 0.50},
    "learned_clean_delta_vs_off": {"operator": ">=", "threshold_pp": -0.20},
    "learned_hard_vs_full": {"operator": ">", "threshold": "full_hard_miou_percent"},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="A-v1 fixed-final checkpoint")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--source-c0-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-c0-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="the frozen 318-entry val-dev split")
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="must not already exist")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def load_config(module_name: str) -> Any:
    return importlib.import_module(module_name).C


def _sha256_argument(value: str, option: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise SystemExit(f"{option} must be a 64-character SHA-256 hex digest")
    return normalized


def _check_file_sha256(path: Path, expected: str, label: str) -> str:
    actual = EV.file_sha256(path)
    if actual.lower() != expected:
        raise SystemExit(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def git_identity() -> dict[str, str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    git_sha = completed.stdout.strip().lower()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", git_sha):
        raise RuntimeError(f"git returned an invalid full commit SHA: {git_sha!r}")
    return {
        "git_sha": git_sha,
        "evaluator_file_sha256": EV.file_sha256(Path(__file__).resolve()),
    }


def frozen_conditions() -> list[E10.Condition]:
    frozen = E10.FROZEN_EVALUATION
    if int(frozen.get("seed", -1)) != EVALUATION_SEED:
        raise RuntimeError("E10 frozen corruption seed differs from the A-v1 evaluation seed")
    if int(frozen.get("sample_count", -1)) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("E10 frozen sample count differs from the A-v1 318-sample contract")
    if (
        E10.FROZEN_CONSTANTS.get("depth_normalization_mean") != 0.48
        or E10.FROZEN_CONSTANTS.get("depth_normalization_std") != 0.28
        or E10.FROZEN_CONSTANTS.get("forward_precision") != "fp32"
        or E10.FROZEN_CONSTANTS.get("tf32_enabled") is not False
    ):
        raise RuntimeError("E10 frozen FP32/depth-normalization constants differ from the A-v1 contract")
    all_conditions = E10.build_conditions(frozen)
    by_id = {condition.condition_id: condition for condition in all_conditions}
    missing = [condition_id for condition_id in CONDITION_IDS if condition_id not in by_id]
    if missing:
        raise RuntimeError(f"frozen corruption definitions are missing required conditions: {missing}")
    conditions = [by_id[condition_id] for condition_id in CONDITION_IDS]
    if tuple(condition.condition_id for condition in conditions) != CONDITION_IDS:
        raise RuntimeError("condition order differs from the A-v1 frozen Quick-Val contract")
    expected_specs = {
        "clean": [],
        "entire_missing@1.0": [{"modality": "depth", "kind": "entire_missing", "severity": 1.0}],
        "spatial_dropout@0.75": [{"modality": "depth", "kind": "spatial_dropout", "severity": 0.75}],
        "misalignment@0.75": [{"modality": "depth", "kind": "misalignment", "severity": 0.75}],
    }
    for condition in conditions:
        if condition.spec_records() != expected_specs[condition.condition_id]:
            raise RuntimeError(
                f"frozen specs for {condition.condition_id} differ from A-v1 protocol: "
                f"{condition.spec_records()}"
            )
    return conditions


def validate_fixed_final(path: Path, source_sha: str) -> dict[str, Any]:
    if path.name != "update-2560.pth":
        raise RuntimeError("Quick-Val accepts only the uniquely named fixed-final update-2560.pth")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("av1_resume_version") != "mmfr-av1-epoch-boundary-v1"
            or state.get("preflight") is not False or state.get("phase") != "gate"
            or state.get("phase_completed") != 640 or state.get("global_optimizer_step") != 2560
            or state.get("attempted") != 640 or state.get("completed") != 640 or state.get("skipped") != 0
            or state.get("protocol", {}).get("source_checkpoint_sha256") != source_sha):
        raise RuntimeError("checkpoint is not the completed formal A-v1 fixed-final identity")
    return {"training_git_commit": state["protocol"]["git_commit"],
            "formal_global_completed": 2560, "gate_completed": 640, "skipped": 0}


def unrounded_miou(hist: np.ndarray) -> float:
    tp = np.diag(hist).astype(np.float64)
    union = hist.sum(0) + hist.sum(1) - tp
    # Identical EV metric arithmetic; keep full precision for strict comparisons.
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0) * 100
    return float(iou.mean())


def verify_model_identity(
    source_model: torch.nn.Module,
    av1_model: torch.nn.Module,
    source_config: Any,
    av1_config: Any,
) -> dict[str, Any]:
    if getattr(source_model, "av1", None) is not None:
        raise RuntimeError("the configured source C0 model unexpectedly has an A-v1 module")
    if getattr(av1_model, "av1", None) is None:
        raise RuntimeError("the fixed-final checkpoint did not build an enabled A-v1 module")
    for attribute in ("num_classes", "background", "class_names", "norm_mean", "norm_std"):
        if getattr(source_config, attribute, None) != getattr(av1_config, attribute, None):
            raise RuntimeError(f"source C0 and A-v1 config differ on {attribute}")

    source_state = source_model.state_dict()
    av1_state = av1_model.state_dict()
    source_names = set(source_state)
    av1_base_names = {name for name in av1_state if not name.startswith("av1.")}
    if not source_names or len(source_names) != 812:
        raise RuntimeError("source C0 must contain exactly the frozen 812 model tensors")
    if source_names != av1_base_names:
        missing = sorted(source_names - av1_base_names)
        unexpected = sorted(av1_base_names - source_names)
        raise RuntimeError(
            "non-A-v1 state keys do not exactly match source C0; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name in sorted(source_names):
        source_tensor = source_state[name]
        av1_tensor = av1_state[name]
        if source_tensor.shape != av1_tensor.shape or source_tensor.dtype != av1_tensor.dtype:
            raise RuntimeError(f"non-A-v1 state shape/dtype mismatch for {name}")
        if not torch.equal(source_tensor, av1_tensor):
            raise RuntimeError(f"non-A-v1 state differs from source C0 at {name}")
    return {
        "status": "exact-match",
        "compared_tensor_count": len(source_names),
        "compared_parameters_and_buffers": True,
        "ignored_prefix": "av1.",
    }


def build_observed_depth_and_support(
    corrupted_depth_uint8: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if corrupted_depth_uint8.ndim != 2 or corrupted_depth_uint8.dtype != np.uint8:
        raise RuntimeError(
            "corrupted Depth must be a 2D uint8 image, "
            f"got dtype={corrupted_depth_uint8.dtype} shape={corrupted_depth_uint8.shape}"
        )
    observed = corrupted_depth_uint8.astype(np.float32) / 255.0
    observed_depth = torch.from_numpy(np.ascontiguousarray(observed))[None, None].to(
        device=device, dtype=torch.float32
    )
    support = torch.ones_like(observed_depth, dtype=torch.bool)
    if observed_depth.ndim != 4 or tuple(observed_depth.shape[:2]) != (1, 1):
        raise RuntimeError("observed Depth must have shape [1, 1, H, W]")
    if not bool(torch.isfinite(observed_depth).all()) or bool(
        ((observed_depth < 0) | (observed_depth > 1)).any()
    ):
        raise RuntimeError("observed Depth must be finite and lie in [0, 1]")
    if not bool(support.all()):
        raise RuntimeError("native original-full evaluation support must be all true")
    return observed_depth, support


def _forward_logits(
    model: torch.nn.Module,
    rgb_tensor: torch.Tensor,
    depth_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    *,
    mode: str,
    observed_depth: torch.Tensor,
    geometry_support: torch.Tensor,
) -> torch.Tensor:
    result = model(
        rgb_tensor,
        depth_tensor,
        av1_mode=mode,
        observed_depth=observed_depth,
        geometry_mask=geometry_support,
    )
    if not isinstance(result, torch.Tensor):
        raise RuntimeError(f"A-v1 {mode} forward did not return a single logits tensor")
    logits = EV.restore_logits_to_metric_grid(result, label_tensor)
    expected_hw = tuple(int(value) for value in label_tensor.shape[-2:])
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise RuntimeError(f"A-v1 {mode} logits have invalid shape {tuple(logits.shape)}")
    if tuple(int(value) for value in logits.shape[-2:]) != expected_hw:
        raise RuntimeError(f"A-v1 {mode} logits are not on the original Label grid")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError(f"A-v1 {mode} produced non-finite logits")
    return logits


def _class_metrics(hist: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    return EV.metrics_from_confusion(hist, class_names)


def evaluate(
    args: argparse.Namespace,
    config: Any,
    source_config: Any,
    conditions: Sequence[E10.Condition],
    entries: Sequence[str],
    checkpoint_sha256: str,
    source_c0_sha256: str,
    split_sha256: str,
    code_identity: Mapping[str, str],
    channel_order: str,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Quick-Val but is unavailable")
    EV.configure_fp32_forward(device)
    # Keep this explicit for CPU and CUDA launches; the frozen run requires TF32 off.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    av1_model = EV.load_model(config, args.checkpoint, device)
    av1_model.eval()
    source_model = EV.load_model(source_config, args.source_c0_checkpoint, device)
    source_model.eval()
    model_identity = verify_model_identity(source_model, av1_model, source_config, config)

    num_classes = int(config.num_classes)
    class_names = list(config.class_names)
    background = int(config.background)
    torch.manual_seed(EVALUATION_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(EVALUATION_SEED)
    base_rng = FAST._capture_rng(device)

    condition_results: dict[str, dict[str, Any]] = {}
    identity_check: dict[str, Any] | None = None
    identity_check_seconds = 0.0
    started_at = dt.datetime.now(dt.timezone.utc)
    run_started = time.perf_counter()

    for condition in conditions:
        aggregate_hist = {behavior: np.zeros((num_classes, num_classes), dtype=np.int64) for behavior in BEHAVIORS}
        gate_values: list[float] = []
        per_sample: list[dict[str, Any]] = []
        forward_seconds = {behavior: 0.0 for behavior in BEHAVIORS}
        condition_started = time.perf_counter()
        FAST._reset_peak_memory(device)

        for entry_index, entry in enumerate(entries):
            sample_id, rgb_uint8, depth_uint8, label = E10.read_sample(args.dataset_root, entry, channel_order)
            normalized_id = normalize_sample_id(entry)
            outcome = E10.corrupt_depth(
                condition,
                rgb_uint8,
                depth_uint8,
                EVALUATION_SEED,
                normalized_id,
            )
            rgb_tensor, depth_tensor, label_tensor = QV.build_original_full_input(
                rgb_uint8,
                outcome.depth,
                label,
                config.norm_mean,
                config.norm_std,
                device,
            )
            observed_depth, geometry_support = build_observed_depth_and_support(outcome.depth, device)
            input_hw = tuple(int(value) for value in label_tensor.shape[-2:])
            if tuple(int(value) for value in rgb_tensor.shape[-2:]) != input_hw:
                raise RuntimeError(f"RGB input grid differs from original Label grid for {sample_id}")
            if tuple(int(value) for value in depth_tensor.shape[-2:]) != input_hw:
                raise RuntimeError(f"normalized Depth grid differs from original Label grid for {sample_id}")
            if tuple(int(value) for value in observed_depth.shape[-2:]) != input_hw:
                raise RuntimeError(f"observed Depth grid differs from original Label grid for {sample_id}")

            sample_record: dict[str, Any] = {
                "sample_id": sample_id,
                "entry": entry,
                "behaviors": {},
                "corrupted_depth_sha256": outcome.depth_sha256(),
                "validity_state_sha256": outcome.state_sha256(),
                "strict_no_op": bool(outcome.strict_no_op),
                "rng_constructed": bool(outcome.rng_constructed),
            }

            if identity_check is None:
                if condition.condition_id != "clean" or entry_index != 0:
                    raise RuntimeError("strict off-vs-C0 review must occur on the first clean evaluation sample")
                identity_started = time.perf_counter()
                FAST._restore_rng(base_rng, device)
                with torch.inference_mode():
                    c0_logits = source_model(rgb_tensor, depth_tensor)
                    if not isinstance(c0_logits, torch.Tensor):
                        raise RuntimeError("separate source C0 forward did not return a logits tensor")
                    c0_logits = EV.restore_logits_to_metric_grid(c0_logits, label_tensor)
                FAST._sync(device)
                c0_seconds = time.perf_counter() - identity_started

                FAST._restore_rng(base_rng, device)
                captured_gate: list[torch.Tensor] = []
                hook = av1_model.av1.gate.register_forward_hook(
                    lambda _module, _inputs, output: captured_gate.append(output.detach().reshape(-1).clone())
                )
                proposal_calls = []
                proposal_hook = av1_model.av1.proposal.register_forward_hook(
                    lambda *_args: proposal_calls.append(1)
                )
                try:
                    off_started = time.perf_counter()
                    with torch.inference_mode():
                        off_logits = _forward_logits(
                            av1_model,
                            rgb_tensor,
                            depth_tensor,
                            label_tensor,
                            mode="off",
                            observed_depth=observed_depth,
                            geometry_support=geometry_support,
                        )
                    FAST._sync(device)
                    off_seconds = time.perf_counter() - off_started
                finally:
                    hook.remove()
                    proposal_hook.remove()
                if captured_gate or proposal_calls:
                    raise RuntimeError("A-v1 off unexpectedly invoked Proposal or Gate")
                if c0_logits.shape != off_logits.shape:
                    raise RuntimeError(
                        f"source C0 and A-v1 off logits have different shapes: "
                        f"{tuple(c0_logits.shape)} vs {tuple(off_logits.shape)}"
                    )
                off_equal = torch.equal(c0_logits, off_logits)
                max_abs_error = float((c0_logits - off_logits).abs().max().item())
                if not off_equal:
                    raise RuntimeError(
                        "strict A-v1 off logits differ from the separate source C0 model on the first sample; "
                        f"max_abs_error={max_abs_error}"
                    )
                identity_check_seconds += c0_seconds
                identity_check = {
                    "status": "exact-match",
                    "sample_id": sample_id,
                    "condition_id": condition.condition_id,
                    "source_c0_logits_equal": True,
                    "max_abs_error": max_abs_error,
                    "rng_policy": "same captured pre-forward RNG state restored before source C0 and A-v1 off",
                    "a_v1_off_hook_gate_calls": 0,
                    "a_v1_off_hook_proposal_calls": 0,
                }
                del c0_logits, source_model
                source_model = None
                mode_logits_for_metrics = {"off": off_logits}
                # This exact off forward is both the identity review and the scored unit.
            else:
                mode_logits_for_metrics = {}

            for behavior in BEHAVIORS:
                if identity_check is not None and condition.condition_id == "clean" and entry_index == 0 and behavior == "off":
                    # The first off pass above was the necessary identity review and is
                    # reused as this unit's scored off prediction.
                    logits = mode_logits_for_metrics["off"]
                    if not isinstance(logits, torch.Tensor):
                        raise RuntimeError("first-sample A-v1 off logits were not retained")
                elif behavior == "off" and condition.condition_id == "clean" and entry_index == 0:
                    raise RuntimeError("first-sample off identity result was lost")
                else:
                    FAST._restore_rng(base_rng, device)
                    captured_gate = []
                    hook = None
                    if behavior == "learned":
                        hook = av1_model.av1.gate.register_forward_hook(
                            lambda _module, _inputs, output: captured_gate.append(output.detach().reshape(-1).clone())
                        )
                    try:
                        unit_started = time.perf_counter()
                        with torch.inference_mode():
                            logits = _forward_logits(
                                av1_model,
                                rgb_tensor,
                                depth_tensor,
                                label_tensor,
                                mode=behavior,
                                observed_depth=observed_depth,
                                geometry_support=geometry_support,
                            )
                        FAST._sync(device)
                        forward_seconds[behavior] += time.perf_counter() - unit_started
                    finally:
                        if hook is not None:
                            hook.remove()
                    if behavior == "learned":
                        if len(captured_gate) != 1 or captured_gate[0].numel() != 1:
                            raise RuntimeError(
                                f"learned pass must emit exactly one gate scalar for {sample_id}; "
                                f"hook_calls={len(captured_gate)}"
                            )
                        gate_tensor = captured_gate[0].to(device="cpu", dtype=torch.float64)
                        if not bool(torch.isfinite(gate_tensor).all()) or bool(
                            ((gate_tensor < 0) | (gate_tensor > 1)).any()
                        ):
                            raise RuntimeError(f"learned gate is non-finite or outside [0, 1] for {sample_id}")
                        gate_value = float(gate_tensor.item())
                        gate_values.append(gate_value)
                        sample_record["behaviors"][behavior] = {"gate_value": gate_value}

                if logits.shape[1] != num_classes:
                    raise RuntimeError(
                        f"{behavior} logits have {logits.shape[1]} classes; expected {num_classes}"
                    )
                EV.update_confusion(aggregate_hist[behavior], logits, label_tensor, num_classes, background)
                sample_hist = np.zeros((num_classes, num_classes), dtype=np.int64)
                EV.update_confusion(sample_hist, logits, label_tensor, num_classes, background)
                sample_record["behaviors"].setdefault(behavior, {}).update(
                    {"confusion_matrix": sample_hist.tolist()}
                )
                del logits, sample_hist

            if identity_check is not None and condition.condition_id == "clean" and entry_index == 0:
                # The identity-review off call was measured once; include its scored-forward
                # time as the off behavior's first observation.
                forward_seconds["off"] += off_seconds
            per_sample.append(sample_record)
            del rgb_tensor, depth_tensor, label_tensor, observed_depth, geometry_support, outcome

        peak = FAST._peak_memory(device)
        behavior_results: dict[str, Any] = {}
        reconciliation: dict[str, Any] = {}
        for behavior in BEHAVIORS:
            summed_per_sample = np.zeros_like(aggregate_hist[behavior])
            for sample_record in per_sample:
                summed_per_sample += np.asarray(
                    sample_record["behaviors"][behavior]["confusion_matrix"], dtype=np.int64
                )
            equal = bool(np.array_equal(aggregate_hist[behavior], summed_per_sample))
            reconciliation[behavior] = {
                "passed": equal,
                "aggregate_equals_sum_of_per_sample_confusions": equal,
                "sample_count": len(per_sample),
            }
            if not equal:
                raise RuntimeError(
                    f"{condition.condition_id}/{behavior}: aggregate confusion differs from summed per-sample confusion"
                )
            metrics = _class_metrics(aggregate_hist[behavior], class_names)
            behavior_results[behavior] = {
                "miou_percent": metrics["miou"],
                "miou_percent_unrounded": unrounded_miou(aggregate_hist[behavior]),
                "metrics_percent": metrics,
                "confusion_matrix": aggregate_hist[behavior].tolist(),
            }

        if len(per_sample) != EXPECTED_SAMPLE_COUNT:
            raise RuntimeError(
                f"{condition.condition_id} completed {len(per_sample)} samples, expected {EXPECTED_SAMPLE_COUNT}"
            )
        if len(gate_values) != EXPECTED_SAMPLE_COUNT:
            raise RuntimeError(
                f"{condition.condition_id} captured {len(gate_values)} learned gate values, "
                f"expected {EXPECTED_SAMPLE_COUNT}"
            )
        if identity_check is None:
            raise RuntimeError("first-sample strict off-vs-C0 identity review was not completed")

        gate_statistics = {
            "count": len(gate_values),
            "mean": float(np.mean(np.asarray(gate_values, dtype=np.float64))),
            "median": float(np.median(np.asarray(gate_values, dtype=np.float64))),
            "per_sample_values_in_per_sample_records": True,
        }
        condition_payload = {
            "schema_version": SCHEMA_VERSION,
            "evaluator_identity": EVALUATOR_IDENTITY,
            "identity": {
                "final_checkpoint": str(args.checkpoint.resolve()),
                "final_checkpoint_sha256": checkpoint_sha256,
                "source_c0_checkpoint": str(args.source_c0_checkpoint.resolve()),
                "source_c0_checkpoint_sha256": source_c0_sha256,
                "split": str(args.split.resolve()),
                "split_sha256": split_sha256,
                "split_role": "val_dev",
                "config_module": CONFIG_MODULE,
                "source_config_module": str(config.mmfr_av1["source_config"]),
                **code_identity,
                "official_test_included": False,
            },
            "condition": {
                **condition.definition(),
                "definition_sha256": E10.json_sha256(condition.definition()),
            },
            "sample_count": len(per_sample),
            "completed": True,
            "view_protocol": {
                "geometry": "original-full",
                "scale": 1.0,
                "flip": False,
                "view_count": 1,
                "padding": "none",
                "input_contract": "native-resolution RGB and normalized three-channel Depth",
                "observed_depth": "current corrupted Depth uint8 / 255, float32 [1,1,H,W], values in [0,1]",
                "geometry_support": "all-true bool [1,1,H,W]; original-full applies no crop or padding",
                "metric_grid": "untouched original Label grid",
                "forward_precision": "fp32",
                "tf32_matmul": False,
                "tf32_cudnn": False,
                "evaluation_seed": EVALUATION_SEED,
                "forward_rng_policy": "reset-per-unit; same captured pre-forward state replayed for off/full/learned",
            },
            "identity_checks": {
                "non_av1_state_matches_source_c0": model_identity,
                "strict_off_matches_separate_source_c0_first_sample": identity_check,
            },
            "behaviors": behavior_results,
            "gate_statistics": gate_statistics,
            "per_sample": per_sample,
            "confusion_reconciliation": reconciliation,
            "timings": {
                "condition_seconds": round(time.perf_counter() - condition_started, 6),
                "forward_seconds_by_behavior": {
                    name: round(seconds, 6) for name, seconds in forward_seconds.items()
                },
                "peak_device_memory": peak,
            },
            "runtime": {
                "device": str(device),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        target = output_dir / condition.directory / "metrics.json"
        QV.atomic_write_json(target, condition_payload)
        condition_results[condition.condition_id] = {
            "directory": condition.directory,
            "metrics_path": str(target.resolve()),
            "sample_count": len(per_sample),
            "behaviors": {
                name: {
                    "miou_percent": behavior_results[name]["miou_percent"],
                    "miou_percent_unrounded": behavior_results[name]["miou_percent_unrounded"],
                    "confusion_matrix": behavior_results[name]["confusion_matrix"],
                }
                for name in BEHAVIORS
            },
            "gate_statistics": gate_statistics,
            "timings": condition_payload["timings"],
            "confusion_reconciliation": reconciliation,
        }
        print(
            f"[{len(condition_results)}/{len(conditions)}] {condition.condition_id} "
            f"off/full/learned={[behavior_results[name]['miou_percent'] for name in BEHAVIORS]} "
            f"gate_mean={gate_statistics['mean']:.6f} -> {target}",
            flush=True,
        )

    if source_model is not None:
        raise RuntimeError("source C0 model was not released after the first-sample strict-off review")
    if identity_check is None:
        raise RuntimeError("strict off-vs-C0 identity review is missing")

    hard_averages: dict[str, dict[str, Any]] = {}
    for behavior in BEHAVIORS:
        hard_mious = {
            condition_id: float(condition_results[condition_id]["behaviors"][behavior]["miou_percent_unrounded"])
            for condition_id in HARD_CONDITION_IDS
        }
        hard_averages[behavior] = {
            "condition_miou_percent": hard_mious,
            "mean_miou_percent": float(sum(hard_mious.values()) / len(HARD_CONDITION_IDS)),
            "basis": "unweighted arithmetic mean of the same EV confusion-derived mIoU, before display rounding",
        }
    off_hard = hard_averages["off"]["mean_miou_percent"]
    full_hard = hard_averages["full"]["mean_miou_percent"]
    learned_hard = hard_averages["learned"]["mean_miou_percent"]
    learned_hard_delta = learned_hard - off_hard
    full_hard_delta = full_hard - off_hard
    learned_full_delta = learned_hard - full_hard
    clean_delta = (
        float(condition_results["clean"]["behaviors"]["learned"]["miou_percent_unrounded"])
        - float(condition_results["clean"]["behaviors"]["off"]["miou_percent_unrounded"])
    )

    requirements = {
        "learned_hard_gain_vs_off": {
            **CONTINUATION_RULES["learned_hard_gain_vs_off"],
            "observed_pp": learned_hard_delta,
            "passed": learned_hard_delta >= 0.50,
        },
        "learned_clean_delta_vs_off": {
            **CONTINUATION_RULES["learned_clean_delta_vs_off"],
            "observed_pp": clean_delta,
            "passed": clean_delta >= -0.20,
        },
        "learned_hard_vs_full": {
            **CONTINUATION_RULES["learned_hard_vs_full"],
            "observed_learned_hard_miou_percent": learned_hard,
            "observed_full_hard_miou_percent": full_hard,
            "observed_delta_pp": learned_full_delta,
            "passed": learned_hard > full_hard,
        },
    }
    promote = all(item["passed"] for item in requirements.values())
    if promote:
        classification = "promote-for-next-review"
        classification_reason = "all three frozen numeric continuation requirements passed"
    elif full_hard_delta > 0 and learned_hard <= full_hard:
        classification = "selector-not-supported"
        classification_reason = (
            "full hard mIoU is above matched off, while learned hard mIoU is not above full; "
            "no additional cutoff was applied"
        )
    elif full_hard_delta <= 0 and learned_hard_delta <= 0:
        classification = "stop"
        classification_reason = (
            "neither full nor learned hard average is above matched off; this exact no-positive-gain "
            "case uses no added tolerance or numeric band"
        )
    else:
        classification = "inconclusive"
        classification_reason = (
            "the frozen continuation requirements did not all pass and the observed directions do not "
            "establish either descriptive non-promote category; retain the measured scores for review"
        )

    finished_at = dt.datetime.now(dt.timezone.utc)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "identity": {
            "final_checkpoint": str(args.checkpoint.resolve()),
            "final_checkpoint_sha256": checkpoint_sha256,
            "source_c0_checkpoint": str(args.source_c0_checkpoint.resolve()),
            "source_c0_checkpoint_sha256": source_c0_sha256,
            "split": str(args.split.resolve()),
            "split_sha256": split_sha256,
            "split_role": "val_dev",
            "dataset_root": str(args.dataset_root.resolve()),
            "config_module": CONFIG_MODULE,
            "source_config_module": str(config.mmfr_av1["source_config"]),
            **code_identity,
            "official_test_included": False,
        },
        "evaluation_contract": {
            "sample_count_required": EXPECTED_SAMPLE_COUNT,
            "sample_count_per_condition": EXPECTED_SAMPLE_COUNT,
            "condition_ids": list(CONDITION_IDS),
            "behaviors": list(BEHAVIORS),
            "geometry": "original-full",
            "scale": 1.0,
            "flip": False,
            "forward_precision": "fp32",
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "evaluation_seed": EVALUATION_SEED,
            "forward_rng_policy": "reset-per-unit; replay one pre-forward RNG state across off/full/learned",
            "official_test_included": False,
        },
        "identity_checks": {
            "non_av1_state_matches_source_c0": model_identity,
            "strict_off_matches_separate_source_c0_first_sample": identity_check,
        },
        "conditions": condition_results,
        "hard_arithmetic": {
            "hard_condition_ids": list(HARD_CONDITION_IDS),
            "definition": "unweighted mean of the three per-condition mIoU values",
            "by_behavior": hard_averages,
            "learned_vs_off_delta_pp": learned_hard_delta,
            "full_vs_off_delta_pp": full_hard_delta,
            "learned_vs_full_delta_pp": learned_full_delta,
            "clean_learned_vs_off_delta_pp": clean_delta,
        },
        "continuation_requirements": requirements,
        "classification": classification,
        "classification_reason": classification_reason,
        "classification_policy": {
            "promote": "only the three numeric continuation requirements above",
            "selector_not_supported": "full hard average > off and learned hard average <= full",
            "stop": "both full and learned hard averages <= matched off; no tolerance band",
            "otherwise": "inconclusive; no additional numeric thresholds are applied",
        },
        "timings": {
            "evaluation_seconds": round(time.perf_counter() - run_started, 6),
            "strict_off_c0_identity_forward_seconds": round(identity_check_seconds, 6),
            "by_condition": {
                condition_id: result["timings"] for condition_id, result in condition_results.items()
            },
        },
        "timestamp_utc": finished_at.isoformat(),
    }
    QV.atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_checkpoint = _sha256_argument(args.expected_checkpoint_sha256, "--expected-checkpoint-sha256")
    expected_source = _sha256_argument(args.expected_source_c0_sha256, "--expected-source-c0-sha256")
    expected_split = _sha256_argument(args.expected_split_sha256, "--expected-split-sha256")

    config = load_config(CONFIG_MODULE)
    av1_contract = getattr(config, "mmfr_av1", None)
    if not isinstance(av1_contract, Mapping) or not av1_contract.get("enabled", False):
        raise RuntimeError("the fixed A-v1 evaluation config is missing an enabled mmfr_av1 contract")
    frozen_source_sha = str(av1_contract.get("source_checkpoint_sha256", "")).lower()
    if expected_source != frozen_source_sha:
        raise SystemExit(
            "--expected-source-c0-sha256 does not match the frozen source C0 identity in the A-v1 config: "
            f"expected-config={frozen_source_sha}, supplied={expected_source}"
        )
    source_config_module = str(av1_contract.get("source_config", ""))
    if not source_config_module:
        raise RuntimeError("the A-v1 config does not identify its source C0 config")
    source_config = load_config(source_config_module)

    checkpoint_sha256 = _check_file_sha256(args.checkpoint, expected_checkpoint, "fixed-final checkpoint")
    source_c0_sha256 = _check_file_sha256(args.source_c0_checkpoint, expected_source, "source C0 checkpoint")
    split_sha256 = _check_file_sha256(args.split, expected_split, "val-dev split")
    entries = EV.split_entries(args.split)
    if len(entries) != EXPECTED_SAMPLE_COUNT:
        raise SystemExit(f"val-dev split has {len(entries)} samples, expected exactly {EXPECTED_SAMPLE_COUNT}")
    conditions = frozen_conditions()
    if len(conditions) != 4:
        raise RuntimeError(f"the A-v1 Quick-Val requires exactly four conditions, got {len(conditions)}")

    if not av1_contract["quickval_contract"]["authorized"]:
        raise RuntimeError("A-v1 four-condition evaluation is not explicitly authorized")
    from tools.mmfr.av1_train import assert_committed
    assert_committed()
    code_identity = {**git_identity(), **validate_fixed_final(args.checkpoint, source_c0_sha256)}
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing run output directory: {output_dir}") from error

    channel_order = str(getattr(config, "channel_order", "RGB"))
    print(
        f"A-v1 Quick-Val: conditions={list(CONDITION_IDS)} samples={len(entries)} "
        f"fixed_final={checkpoint_sha256[:16]}… git={code_identity['git_sha']}",
        flush=True,
    )
    summary = evaluate(
        args,
        config,
        source_config,
        conditions,
        entries,
        checkpoint_sha256,
        source_c0_sha256,
        split_sha256,
        code_identity,
        channel_order,
        output_dir,
    )
    print(
        f"A-v1 Quick-Val complete: classification={summary['classification']} "
        f"learned_hard_delta={summary['hard_arithmetic']['learned_vs_off_delta_pp']:.4f} pp "
        f"clean_delta={summary['hard_arithmetic']['clean_learned_vs_off_delta_pp']:.4f} pp",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
