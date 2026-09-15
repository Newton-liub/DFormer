#!/usr/bin/env python3
"""Short real-training trajectory equivalence for the MMFR-A2 v3 corruption pipeline.

Runs one deterministic AMP+GradScaler trajectory on the real ``train-dev`` split, at the
frozen batch size 10, and records per-successful-update hashes.  ``run`` writes one JSON;
``compare`` checks two JSONs for bitwise equivalence.

Typical use (old vs optimized helper):

    git checkout <old-rev> -- utils/dataloader/mmfr_training_v3.py
    python tools/mmfr/a2_v3_short_trajectory_equivalence.py run --label old  --output .../old.json
    git checkout <new-rev> -- utils/dataloader/mmfr_training_v3.py
    python tools/mmfr/a2_v3_short_trajectory_equivalence.py run --label opt-a --output .../opt-a.json
    python tools/mmfr/a2_v3_short_trajectory_equivalence.py compare --reference .../old.json \
        --candidate .../opt-a.json --output .../equivalence.json

The samples are materialized through fixed per-item preprocessing seeds, so the trajectory
does not depend on DataLoader worker scheduling.  Never reads the official test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mmfr.a2_v3_amp_update_path_isolation import (  # noqa: E402
    build_batch_identity,
    build_model,
    build_optimizer,
    build_step_batch,
    load_train_dataset,
    plan_step,
    reliability_parameter_names,
    seed_everything,
    shared_parameter_hashes,
    shared_parameter_names,
    optimizer_state_hashes,
    snapshot_model_state,
    tensor_sha256,
)
from tools.museg_protocol import write_json  # noqa: E402
from utils.training_checkpoint import optimizer_step_was_applied  # noqa: E402

SCHEMA = "mmfr-a2-v3-short-trajectory-equivalence-v1"


def working_tree_helper_sha256() -> str:
    return hashlib.sha256(
        (REPO_ROOT / "utils" / "dataloader" / "mmfr_training_v3.py").read_bytes()
    ).hexdigest()


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_trajectory(
    *,
    label: str,
    min_successful_updates: int,
    max_attempts: int,
    model_seed: int,
    corruption_workers: int,
    batch_size: int,
    deterministic: bool,
) -> Dict[str, Any]:
    import importlib

    if not torch.cuda.is_available():
        raise SystemExit("this gate requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    # Bitwise cross-process comparison is only meaningful if the GPU kernels themselves are
    # deterministic; cuDNN's default allows nondeterministic algorithm choices, which would
    # otherwise show up as run-to-run drift unrelated to the corruption-pipeline change.
    determinism = {
        "cudnn_deterministic": bool(deterministic),
        "cudnn_benchmark": False,
        "use_deterministic_algorithms": False,
    }
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
            determinism["use_deterministic_algorithms"] = True
        except Exception as exc:  # pragma: no cover - environment dependent
            determinism["use_deterministic_algorithms_error"] = str(exc)
    config = importlib.import_module("local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3").C
    corruption = dict(config.mmfr_a2["corruption"])

    dataset = load_train_dataset(config)
    seed_everything(model_seed)
    model = build_model(config, device)
    initial_state = snapshot_model_state(model)
    initial_shared = shared_parameter_hashes(model, shared_parameter_names(model))
    optimizer = build_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler()
    head_names = reliability_parameter_names(model)
    shared_names = shared_parameter_names(model)

    records: List[Dict[str, Any]] = []
    successful = 0
    attempts = 0
    skipped = 0
    for step in range(max_attempts):
        if successful >= min_successful_updates:
            break
        attempts += 1
        plan = plan_step(config, step, batch_size, 0)
        seed_everything(int(plan["rng_seed"]))
        # build_step_batch materializes the 10 items through fixed per-item
        # preprocessing seeds, so the trajectory is independent of DataLoader workers;
        # the helper itself is called with the production serial path here.
        batch, metadata, samples = build_step_batch(dataset, config, plan, corruption)
        identity = build_batch_identity(batch)
        tensors = {
            key: batch[key].to(device, non_blocking=False)
            for key in (
                "rgb",
                "depth",
                "raw_rgb",
                "raw_depth",
                "reliability_target",
                "valid_mask",
                "depth_valid_post",
                "telemetry_masks",
            )
        }
        labels = batch["labels"].to(device, non_blocking=False)
        auxiliary = {
            "raw_rgb": tensors["raw_rgb"],
            "raw_depth": tensors["raw_depth"],
            "reliability_target": tensors["reliability_target"],
            "reliability_valid_mask": tensors["valid_mask"],
            "depth_valid": tensors["depth_valid_post"],
            "reliability_telemetry_masks": tensors["telemetry_masks"],
        }
        input_hashes = {
            "rgb": tensor_sha256(tensors["rgb"]),
            "depth": tensor_sha256(tensors["depth"]),
            "raw_rgb": tensor_sha256(tensors["raw_rgb"]),
            "raw_depth": tensor_sha256(tensors["raw_depth"]),
            "reliability_target": tensor_sha256(tensors["reliability_target"]),
            "valid_mask": tensor_sha256(tensors["valid_mask"]),
            "depth_valid_post": tensor_sha256(tensors["depth_valid_post"]),
            "telemetry_masks": tensor_sha256(tensors["telemetry_masks"]),
            "labels": tensor_sha256(labels),
        }
        seed_everything(int(plan["rng_seed"]))
        optimizer.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale())
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(tensors["rgb"], tensors["depth"], labels, **auxiliary)
        loss_value = float(loss.detach().item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        applied = bool(optimizer_step_was_applied(scale_before, scale_after))
        optimizer.zero_grad(set_to_none=True)
        if not applied:
            skipped += 1
            continue
        successful += 1
        model_state = {name: tensor.detach().to("cpu").clone() for name, tensor in model.state_dict().items()}
        records.append(
            {
                "update_index": successful,
                "step": int(step),
                "epoch": int(plan["epoch"]),
                "iteration": int(plan["iteration"]),
                "sample_indices": [int(sample["index"]) for sample in samples],
                "sample_ids": [str(record["sample_id"]) for record in metadata],
                "clean_flags": [bool(record["clean"]) for record in metadata],
                "num_specs": [int(record["num_specs"]) for record in metadata],
                "spec_kinds": [[str(kind) for kind in record["spec_kinds"]] for record in metadata],
                "batch_identity_hash": identity and hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "input_hashes": input_hashes,
                "loss": loss_value,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "skipped_before_this_update": skipped,
                "shared_parameters": shared_parameter_hashes(model, shared_names),
                "head_parameters": shared_parameter_hashes(model, head_names),
                "optimizer_state": optimizer_state_hashes(
                    optimizer,
                    {id(parameter): name for name, parameter in model.named_parameters()},
                    list(shared_names) + list(head_names),
                ),
                "model_state_hash": hashlib.sha256(
                    b"".join(
                        name.encode("utf-8") + hashlib.sha256(tensor.numpy().tobytes()).digest()
                        for name, tensor in sorted(model_state.items())
                    )
                ).hexdigest(),
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    return {
        "schema_version": SCHEMA,
        "label": label,
        "scientific_protocol": "MMFR-A2-train-integration-v3",
        "scientific_semantics_changed": False,
        "official_test_included": False,
        "code_identity": {
            "git_commit": git_commit(),
            "mmfr_training_v3_sha256": working_tree_helper_sha256(),
        },
        "determinism": determinism,
        "config": {
            "batch_size": int(batch_size),
            "corruption_workers": int(corruption_workers),
            "p_clean": float(corruption["p_clean"]),
            "max_specs": int(corruption["max_specs"]),
            "corruption_seed": int(corruption["seed"]),
            "model_seed": int(model_seed),
            "device": torch.cuda.get_device_name(device),
            "torch_num_threads": int(torch.get_num_threads()),
        },
        "initial_shared_parameters_hash": hashlib.sha256(
            json.dumps(initial_shared, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "attempts": attempts,
        "successful_updates": successful,
        "skipped_steps": skipped,
        "updates": records,
    }


def compare(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []
    if reference["initial_shared_parameters_hash"] != candidate["initial_shared_parameters_hash"]:
        mismatches.append({"field": "initial_shared_parameters_hash"})
    if reference["successful_updates"] != candidate["successful_updates"]:
        mismatches.append(
            {
                "field": "successful_updates",
                "reference": reference["successful_updates"],
                "candidate": candidate["successful_updates"],
            }
        )
    max_abs_loss_difference = 0.0
    for expected, actual in zip(reference["updates"], candidate["updates"]):
        for field in (
            "sample_ids",
            "sample_indices",
            "clean_flags",
            "num_specs",
            "spec_kinds",
            "batch_identity_hash",
            "input_hashes",
            "scale_before",
            "scale_after",
            "skipped_before_this_update",
            "shared_parameters",
            "head_parameters",
            "optimizer_state",
            "model_state_hash",
        ):
            if expected[field] != actual[field]:
                mismatches.append(
                    {
                        "update_index": expected["update_index"],
                        "field": field,
                        "reference": expected[field] if field != "input_hashes" else "see json",
                        "candidate": actual[field] if field != "input_hashes" else "see json",
                    }
                )
        loss_difference = abs(float(expected["loss"]) - float(actual["loss"]))
        max_abs_loss_difference = max(max_abs_loss_difference, loss_difference)
        if expected["loss"] != actual["loss"]:
            mismatches.append(
                {
                    "update_index": expected["update_index"],
                    "field": "loss",
                    "reference": expected["loss"],
                    "candidate": actual["loss"],
                    "abs_difference": loss_difference,
                }
            )
    return {
        "reference_label": reference["label"],
        "candidate_label": candidate["label"],
        "reference_helper_sha256": reference["code_identity"]["mmfr_training_v3_sha256"],
        "candidate_helper_sha256": candidate["code_identity"]["mmfr_training_v3_sha256"],
        "updates_compared": min(len(reference["updates"]), len(candidate["updates"])),
        "max_abs_loss_difference": max_abs_loss_difference,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:30],
        "pass": not mismatches,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--min-successful-updates", type=int, default=10)
    run.add_argument("--max-attempts", type=int, default=40)
    run.add_argument("--model-seed", type=int, default=772961337)
    run.add_argument("--corruption-workers", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=10)
    run.add_argument(
        "--no-deterministic",
        action="store_true",
        help="leave cuDNN at its default (recorded as a harness limitation)",
    )
    comp = sub.add_parser("compare")
    comp.add_argument("--reference", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.mode == "run":
        payload = run_trajectory(
            label=args.label,
            min_successful_updates=args.min_successful_updates,
            max_attempts=args.max_attempts,
            model_seed=args.model_seed,
            corruption_workers=args.corruption_workers,
            batch_size=args.batch_size,
            deterministic=not args.no_deterministic,
        )
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, payload)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "label": payload["label"],
                    "helper_sha256": payload["code_identity"]["mmfr_training_v3_sha256"],
                    "attempts": payload["attempts"],
                    "successful_updates": payload["successful_updates"],
                    "skipped_steps": payload["skipped_steps"],
                    "final_model_state_hash": payload["updates"][-1]["model_state_hash"] if payload["updates"] else None,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    verdict = compare(reference, candidate)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output_path,
        {
            "schema_version": "mmfr-a2-v3-short-trajectory-comparison-v1",
            "official_test_included": False,
            "verdict": verdict,
        },
    )
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
