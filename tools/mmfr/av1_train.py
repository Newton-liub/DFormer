#!/usr/bin/env python3
"""Bounded A-v1 preflight -> clean formal training -> one matched Quick-Val.

No legacy E1 loop, training validation, best selection, retries or batch overrides.
Full recovery is deliberately limited to the frozen 640-update epoch boundaries.
Use --check-inputs for a lightweight no-model/no-worker/no-CUDA identity check.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch

from utils import mmfr_av1_training as T
from utils.training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION, atomic_save_checkpoint, capture_rng_state,
    restore_rng_state, file_sha256, get_git_commit, load_weights_only_model_state,
    optimizer_step_was_applied, stable_sha256,
)
from tools.mmfr.e1_quickval import atomic_write_json

CONFIG = "local_configs.MUSeg.DFormerv2_S_MMFR_AV1"
SOURCE_SHA = "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a"
SPLITS = {
    "train-dev.txt": (1277, "a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470"),
    "val-dev.txt": (318, "1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83"),
}
RESUME_VERSION = "mmfr-av1-epoch-boundary-v1"


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phase_lr(k, total):
    if not 1 <= k <= total:
        raise ValueError("LR coordinate outside successful-update budget")
    return 3e-5 * (k / 128 if k <= 128 else ((total - k) / (total - 128)) ** 0.9)


def check_inputs(config):
    a = config.mmfr_av1
    expected = {
        "phase_successful_updates": {"proposal": 1920, "gate": 640},
        "required_successful_updates": 2560, "margin": 0.01, "lambda_clean": 0.1,
        "new_module_lr": 3e-5, "weight_decay": 0.01, "train_seed": 772961337,
        "initialization_seed": 2026093000,
        "phase_corruption_seeds": {"proposal": 2026093001, "gate": 2026093002},
        "batch_size": 10, "num_workers": 8, "accumulation_steps": 1,
        "train_resolution": [480, 640], "amp": True, "amp_dtype": "float16",
        "tf32_matmul": True, "tf32_cudnn": True, "p_clean": 0.25,
        "optimizer": "AdamW", "float32_matmul_precision": "high",
        "grad_scaler": {"initial_scale": 1024.0, "growth_factor": 2.0,
                        "backoff_factor": 0.5, "growth_interval": 2000},
        "official_test": "sealed_unread", "validation_enabled": False,
        "phase_data_schedule": {"proposal": {"nepochs": 15, "niters_per_epoch": 128},
                                "gate": {"nepochs": 5, "niters_per_epoch": 128}},
    }
    for key, value in expected.items():
        if a.get(key) != value:
            raise RuntimeError(f"frozen A-v1 contract mismatch: {key}")
    for key, value in {"batch_size": 10, "num_workers": 8, "image_height": 480,
                       "image_width": 640, "accumulation_steps": 1, "seed": 772961337}.items():
        if getattr(config, key) != value:
            raise RuntimeError(f"execution field mismatch: {key}")
    if any(getattr(config, key) is not None for key in ("val_source", "eval_source", "test_source")):
        raise RuntimeError("training Val/test sources must be disabled")
    if config.training_validation_enabled or config.e1_batch1 is not None:
        raise RuntimeError("legacy E1 contract/validation must remain disabled")
    if not (a["formal_training_authorized"] and a["cloud_preflight"]["execution_authorized_this_round"]
            and a["quickval_contract"]["authorized"]):
        raise RuntimeError("2026-10-01 bounded execution authorization not explicit")
    split_meta, entries_by_role = {}, {}
    for filename, (expected_count, expected_sha) in SPLITS.items():
        path = ROOT / "data/splits/MUSeg/dev-v1" / filename
        actual_sha = file_sha256(path)
        entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if actual_sha != expected_sha or len(entries) != expected_count or len(set(entries)) != expected_count:
            raise RuntimeError(f"frozen split identity mismatch: {path}")
        entries_by_role[filename] = set(entries)
        split_meta[filename] = {"path": str(path), "samples": len(entries), "sha256": actual_sha}
    if entries_by_role["train-dev.txt"] & entries_by_role["val-dev.txt"]:
        raise RuntimeError("train-dev/val-dev overlap")
    if Path(config.train_source).resolve() != Path(split_meta["train-dev.txt"]["path"]).resolve():
        raise RuntimeError("train source is not frozen train-dev")
    for field in ("rgb_root_folder", "x_root_folder", "gt_root_folder"):
        if not Path(getattr(config, field)).is_dir():
            raise RuntimeError(f"missing dataset directory: {field}")
    source = Path(a["source_checkpoint"]).resolve()
    if a["source_checkpoint_sha256"] != SOURCE_SHA or file_sha256(source) != SOURCE_SHA:
        raise RuntimeError("C0 source checkpoint identity mismatch")
    return {"git_commit": get_git_commit(ROOT), "source_checkpoint": str(source),
            "source_checkpoint_sha256": SOURCE_SHA, "splits": split_meta,
            "config_module": CONFIG, "contract_sha256": stable_sha256(a),
            "official_test": "sealed_unread", "dataset_path": config.dataset_path}


def assert_committed():
    tracked = ["tools/mmfr/av1_train.py", "tools/mmfr/av1_quickval.py",
               "local_configs/MUSeg/DFormerv2_S_MMFR_AV1.py", "utils/mmfr_av1_training.py",
               "models/builder.py", "models/mmfr_av1.py"]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *tracked],
                                    cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("commit all runtime code/config before model execution")


def build_model(config, identity):
    from models.builder import EncoderDecoder
    seed_training(config.mmfr_av1["train_seed"])
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=torch.nn.SyncBatchNorm, syncbn=True)
    load_info = load_weights_only_model_state(
        identity["source_checkpoint"], model=model, expected_sha256=SOURCE_SHA,
        expected_model_key_count=812, allowed_missing_prefixes=("av1.",))
    if any(torch.count_nonzero(t).item() for t in
           (model.av1.proposal[-1].weight, model.av1.proposal[-1].bias)):
        raise RuntimeError("Proposal initialization is not exactly zero")
    model.cuda()
    # Source RNG is never restored; discard construction draws before train/data stream.
    seed_training(config.mmfr_av1["train_seed"])
    return model, load_info


def frozen_snapshot(model, phase):
    active = f"av1.{phase}."
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith(active)}


def check_frozen(model, phase, reference):
    for name, value in model.state_dict().items():
        if name in reference and not torch.equal(value.detach().cpu(), reference[name]):
            raise RuntimeError(f"frozen tensor changed: {name}")
    for name, parameter in model.named_parameters():
        active = name.startswith(f"av1.{phase}.")
        if parameter.requires_grad != active or (not active and parameter.grad is not None):
            raise RuntimeError(f"trainable/frozen membership changed: {name}")
    if any(module.training for name, module in model.named_modules()
           if name and not name.startswith("av1")):
        raise RuntimeError("C0 module escaped eval mode")


def save_recovery(path, model, optimizer, scaler, phase, completed, identity, config, preflight):
    total = config.mmfr_av1["phase_successful_updates"][phase]
    global_step = completed + (1920 if phase == "gate" else 0)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION, "av1_resume_version": RESUME_VERSION,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "amp_scaler": scaler.state_dict(), "rng_state": capture_rng_state(),
        "phase": phase, "phase_completed": completed, "global_optimizer_step": global_step,
        "attempted": completed, "completed": completed, "skipped": 0,
        "preflight": preflight, "protocol": identity,
        "scheduler": {"completed": completed, "total": total, "warmup": 128,
                      "poly_power": 0.9, "peak_lr": 3e-5,
                      "next_lr": phase_lr(completed + 1, total) if completed < total else None},
        "data_state": {"next_epoch": completed // 128 + 1, "next_iteration": 0,
                       "niters_per_epoch": 128, "num_workers": 8,
                       "epoch_boundary": completed % 128 == 0,
                       "persistent_workers": False, "loader_generator": "main torch CPU RNG",
                       "worker_rng": "fresh workers from saved main torch RNG at next epoch",
                       "augmentation": "unchanged TrainPre/RGBXDataset; no prefetched batch reused",
                       "train_source_sha256": identity["splits"]["train-dev.txt"]["sha256"]},
        "best_val_miou": None, "best_val_epoch": None, "best_tie_break_rule": "strict-greater-keeps-earliest",
        "completed_epoch": completed // 128, "next_epoch": completed // 128 + 1,
    }
    atomic_save_checkpoint(checkpoint, path)


def validate_resume(state, identity, config):
    if (state.get("av1_resume_version") != RESUME_VERSION or state.get("preflight") is not False
            or state.get("protocol") != identity):
        raise RuntimeError("full resume identity mismatch; no weights-only fallback")
    phase = state.get("phase")
    n = state.get("phase_completed")
    total = config.mmfr_av1["phase_successful_updates"].get(phase)
    if total is None or not isinstance(n, int) or not 0 < n <= total or n % 640:
        raise RuntimeError("resume is not a frozen 640-update boundary")
    if state.get("global_optimizer_step") != n + (1920 if phase == "gate" else 0):
        raise RuntimeError("resume global counter mismatch")
    if any(state.get(key) != value for key, value in {"attempted": n, "completed": n, "skipped": 0}.items()):
        raise RuntimeError("resume attempt/completion/skip mismatch")
    expected_schedule = {"completed": n, "total": total, "warmup": 128, "poly_power": 0.9,
                         "peak_lr": 3e-5, "next_lr": phase_lr(n + 1, total) if n < total else None}
    if state.get("scheduler") != expected_schedule:
        raise RuntimeError("resume scheduler mismatch")
    cursor = state.get("data_state", {})
    if (cursor.get("next_epoch") != n // 128 + 1 or cursor.get("next_iteration") != 0
            or cursor.get("epoch_boundary") is not True or cursor.get("persistent_workers") is not False
            or cursor.get("num_workers") != 8 or cursor.get("niters_per_epoch") != 128
            or cursor.get("train_source_sha256") != identity["splits"]["train-dev.txt"]["sha256"]):
        raise RuntimeError("resume data/augmentation boundary mismatch")
    return phase, n


def restore_model_with_source_guard(model, state):
    current = model.state_dict()
    saved = state["model"]
    for name, value in current.items():
        if not name.startswith("av1.") and (name not in saved or not torch.equal(value.detach().cpu(), saved[name])):
            raise RuntimeError(f"resume C0 tensor identity mismatch: {name}")
    model.load_state_dict(saved, strict=True)


def prepare_diagnostic(reference_dir, resume_path, state, identity, config):
    """One explicitly authorized replay; ordinary resume remains same-commit only."""
    original_commit = "91c7f96d3768fde0f478e03a4ccda8c6f19368dd"
    expected_sha = "05714d165925c4549aa1111564bbca7d8b8d9896d337c5b68f0b7bb98590564d"
    original = json.loads((reference_dir / "identity.json").read_text())
    stopped = json.loads((reference_dir / "formal/proposal-stopped.json").read_text())
    parent_identity = state.get("protocol", {})
    if (parent_identity.get("git_commit") != original_commit
            or original.get("git_commit") != original_commit
            or file_sha256(resume_path) != expected_sha
            or Path(stopped.get("latest_recovery", "")).resolve() != resume_path.resolve()
            or stopped.get("attempted") != 1871 or stopped.get("completed") != 1870
            or stopped.get("skipped") != 0 or stopped.get("status") != "stopped"
            or "missing/nonfinite active gradient: av1.proposal.0.weight" not in stopped.get("exception", "")):
        raise RuntimeError("diagnostic is restricted to the verified original 1280/1871 failure")
    if {k: v for k, v in identity.items() if k != "git_commit"} != {
            k: v for k, v in parent_identity.items() if k != "git_commit"}:
        raise RuntimeError("diagnostic input/contract identity changed")
    if any(original.get(k) != v for k, v in parent_identity.items()):
        raise RuntimeError("original run and recovery identity disagree")
    phase, n = validate_resume(state, parent_identity, config)
    if (phase, n) != ("proposal", 1280):
        raise RuntimeError("diagnostic boundary changed")
    allowed = {"tools/mmfr/av1_train.py", "doc/main/MUSeg-current-status.md",
               "doc/main/MUSeg-open-decisions.md", "MMFR/02_evidence/reproducibility_current.json"}
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", original_commit, identity["git_commit"]], cwd=ROOT, text=True).splitlines())
    if not changed <= allowed or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("diagnostic permits only committed runner instrumentation and live evidence changes")
    fields = ("phase", "epoch", "iteration", "attempted", "completed", "skipped", "global_completed",
              "lr", "loss", "finite", "scale_before", "scale_after", "optimizer_applied",
              "batch_size", "ce", "clean_consistency")
    expected = {}
    with (reference_dir / "formal/proposal-steps.jsonl").open() as log:
        for line in log:
            record = json.loads(line)
            k = record["completed"]
            if 1280 < k <= 1870:
                if k in expected or record["attempted"] != k or record["skipped"] != 0:
                    raise RuntimeError("original replay reference is not a unique successful sequence")
                expected[k] = {key: record[key] for key in fields}
    if set(expected) != set(range(1281, 1871)):
        raise RuntimeError("original replay reference is incomplete")
    return {"parent_identity": parent_identity, "parent_checkpoint": str(resume_path.resolve()),
            "parent_checkpoint_sha256": expected_sha, "runtime_commit": identity["git_commit"],
            "changed_files": sorted(changed), "stop_attempt": 1871,
            "expected": expected, "matched_steps": 0,
            "compatibility_scope": "explicit diagnostic-only parent identity; math/RNG/data unchanged; normal resume guard untouched"}


def run_phase(model, config, identity, output, phase, *, preflight, resume=None, diagnostic=None):
    from utils.dataloader.dataloader import get_train_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    optimizer = T.configure_phase(model, phase, lr=3e-5, weight_decay=0.01)
    s = config.mmfr_av1["grad_scaler"]
    scaler = torch.cuda.amp.GradScaler(init_scale=s["initial_scale"], growth_factor=s["growth_factor"],
                                      backoff_factor=s["backoff_factor"], growth_interval=s["growth_interval"])
    completed = 0
    if resume is not None:
        phase_restored, completed = validate_resume(
            resume, diagnostic["parent_identity"] if diagnostic else identity, config)
        if phase_restored != phase:
            raise RuntimeError("resume phase mismatch")
        restore_model_with_source_guard(model, resume)
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["amp_scaler"])
        restore_rng_state(resume["rng_state"])
    reference = frozen_snapshot(model, phase)
    check_frozen(model, phase, reference)
    local_config = copy.deepcopy(config)
    local_config.niters_per_epoch = 128  # data length only; no inherited E1 training loop.
    loader, sampler = get_train_loader(SimpleNamespace(distributed=False), RGBXDataset, local_config)
    if sampler is not None or len(loader) != 128 or loader.persistent_workers:
        raise RuntimeError("unexpected data/sampler contract")
    total = config.mmfr_av1["phase_successful_updates"][phase]
    limit = diagnostic["stop_attempt"] if diagnostic else (3 if preflight else total)
    attempted, skipped = completed, 0
    torch.cuda.reset_peak_memory_stats()
    phase_started = time.perf_counter()
    records = []
    latest_recovery = None
    progress = {"phase": phase, "preflight": preflight, "git_commit": identity["git_commit"]}
    failed_site = None
    if diagnostic:
        progress.update(diagnostic_only=True, parent_commit=diagnostic["parent_identity"]["git_commit"],
                        stop_attempt=diagnostic["stop_attempt"])
    log_path = output / f"{phase}-steps.jsonl"
    try:
        with log_path.open("x", buffering=1) as log:
            for epoch in range(completed // 128 + 1, config.mmfr_av1["phase_data_schedule"][phase]["nepochs"] + 1):
                if completed >= limit:
                    break
                wall_started = time.perf_counter()
                for iteration, data in enumerate(loader):
                    attempted += 1
                    lr = phase_lr(completed + 1, total)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    batch = T.build_phase_batch(config, phase, data["data"], data["modal_x"], data["fn"],
                                                epoch=epoch, iteration=iteration,
                                                niters_per_epoch=128,
                                                nepochs=config.mmfr_av1["phase_data_schedule"][phase]["nepochs"])
                    if tuple(batch["rgb"].shape) != (10, 3, 480, 640):
                        raise RuntimeError("full-size/batch contract violated")
                    for key in ("rgb", "depth", "raw_depth", "valid_mask", "clean_mask"):
                        batch[key] = batch[key].cuda(non_blocking=True)
                    labels = data["label"].cuda(non_blocking=True)
                    torch.cuda.synchronize()
                    update_started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    before = float(scaler.get_scale())
                    diagnose_site = diagnostic is not None and attempted == diagnostic["stop_attempt"]
                    if diagnose_site:
                        failed_site = {"rng_before_forward": capture_rng_state(), "backward_completed": False,
                                       "unscale_completed": False, "batch": batch,
                                       "labels": labels, "sample_ids": list(data["fn"])}
                    # Instrument only the original failure coordinate; no extra forward/backward.
                    with torch.autograd.detect_anomaly(check_nan=True) if diagnose_site else nullcontext():
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            loss, details = T.phase_loss(model, batch, labels, phase=phase,
                                                         margin=0.01, lambda_clean=0.1)
                        if diagnose_site:
                            failed_site.update(loss=float(loss.detach()),
                                               loss_details={key: float(details[key]) for key in
                                                             ("ce", "clean_consistency")})
                        scaler.scale(loss).backward()
                        if diagnose_site:
                            failed_site["backward_completed"] = True
                    scaler.unscale_(optimizer)
                    if diagnose_site:
                        failed_site["unscale_completed"] = True
                    for name, p in model.named_parameters():
                        if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all())):
                            raise RuntimeError(f"missing/nonfinite active gradient: {name}")
                    scaler.step(optimizer)
                    scaler.update()
                    after = float(scaler.get_scale())
                    applied = optimizer_step_was_applied(before, after)
                    skipped += int(not applied)
                    torch.cuda.synchronize()
                    if applied:
                        completed += 1
                    record = {"phase": phase, "epoch": epoch, "iteration": iteration,
                              "attempted": attempted, "completed": completed, "skipped": skipped,
                              "global_completed": completed + (1920 if phase == "gate" and not preflight else 0),
                              "lr": lr, "loss": float(loss.detach()), "finite": bool(torch.isfinite(loss)),
                              "scale_before": before, "scale_after": after, "optimizer_applied": applied,
                              "update_seconds": time.perf_counter() - update_started,
                              "wall_seconds_including_data": time.perf_counter() - wall_started,
                              "batch_size": 10,
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                    for key in ("ce", "bce", "clean_consistency"):
                        if key in details:
                            record[key] = float(details[key])
                    if phase == "gate":
                        u = details["utility"].float().cpu()
                        record.update(utility=u.tolist(), positive=int((u > 0.01).sum()),
                                      negative=int((u < -0.01).sum()), ambiguous=int((u.abs() <= 0.01).sum()),
                                      gate=details["gate"].float().cpu().tolist())
                    if diagnostic and completed in diagnostic["expected"]:
                        expected = diagnostic["expected"][completed]
                        actual = {key: record[key] for key in expected}
                        if actual != expected:
                            atomic_write_json(output / "diagnostic-replay-mismatch.json",
                                              {"completed": completed, "expected": expected, "actual": actual})
                            raise RuntimeError("diagnostic replay diverged from original successful record; stop")
                        diagnostic["matched_steps"] += 1
                        progress["matched_reference_steps"] = diagnostic["matched_steps"]
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    progress.update(attempted=attempted, completed=completed, skipped=skipped,
                                    latest_recovery=latest_recovery)
                    if not applied:
                        raise RuntimeError("GradScaler skipped optimizer step; stopping immediately")
                    if preflight:
                        records.append(record)
                    del batch, labels, data, loss, details
                    wall_started = time.perf_counter()
                    if (preflight and completed == 3) or (diagnostic and attempted >= limit):
                        break  # Bounded preflight/diagnostic never performs an extra update.
                # Full formal epochs exhaust the iterator before checkpointing: no live worker
                # or prefetch state is required at the next-epoch restart boundary.
                if not preflight and not diagnostic and completed % 640 == 0:
                    check_frozen(model, phase, reference)
                    name = ("proposal-update-1920.pth" if phase == "proposal" and completed == 1920
                            else f"update-{completed + (1920 if phase == 'gate' else 0)}.pth")
                    destination = output / "checkpoint" / name
                    save_recovery(destination, model, optimizer, scaler, phase, completed, identity, config, False)
                    latest_recovery = str(destination)
            check_frozen(model, phase, reference)
            if completed != limit or attempted != completed or skipped:
                raise RuntimeError("successful-update budget not exactly completed")
    except BaseException:
        failure_trace = traceback.format_exc()
        if diagnostic:
            grad_summary = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    grad = parameter.grad
                    grad_summary[name] = {"missing": grad is None}
                    if grad is not None:
                        grad_summary[name].update(dtype=str(grad.dtype),
                                                  nan_count=int(torch.isnan(grad).sum()),
                                                  inf_count=int(torch.isinf(grad).sum()))
            artifact = None
            if failed_site is not None:
                artifact = output / "diagnostic-failure-state.pth"
                atomic_save_checkpoint({"artifact_role": "diagnostic-only-nonresumable-not-performance-candidate",
                                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                        "batch": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                                  for k, v in failed_site["batch"].items()},
                                        "labels": failed_site["labels"].detach().cpu(),
                                        "rng_before_forward": failed_site["rng_before_forward"],
                                        "sample_ids": failed_site["sample_ids"],
                                        "parent_checkpoint_sha256": diagnostic["parent_checkpoint_sha256"],
                                        "runtime_commit": identity["git_commit"],
                                        "attempted": attempted, "completed": completed}, artifact)
            atomic_write_json(output / "diagnostic-failure.json",
                              {"diagnostic_only": True, "exception": failure_trace,
                               "attempted": attempted, "completed": completed,
                               "matched_reference_steps": diagnostic["matched_steps"],
                               "loss": failed_site.get("loss") if failed_site else None,
                               "loss_details": failed_site.get("loss_details") if failed_site else None,
                               "sample_ids": failed_site.get("sample_ids") if failed_site else None,
                               "backward_completed": failed_site.get("backward_completed") if failed_site else None,
                               "unscale_completed": failed_site.get("unscale_completed") if failed_site else None,
                               "active_gradients": grad_summary,
                               "failure_state_artifact": str(artifact) if artifact else None,
                               "no_contract_change": True, "no_gate_or_evaluation": True})
        progress.update(status="stopped", attempted=attempted, completed=completed, skipped=skipped,
                        latest_recovery=latest_recovery, exception=failure_trace,
                        recovery_policy="only last completed 640-update full checkpoint; no contaminated-state resume")
        atomic_write_json(output / f"{phase}-stopped.json", progress)
        raise
    result = {**progress, "status": ("diagnostic-limit-reached-no-error" if diagnostic else
                                     "passed" if preflight else "completed"),
              "attempted": attempted, "completed": completed, "skipped": skipped,
              "elapsed_seconds": time.perf_counter() - phase_started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "latest_recovery": latest_recovery}
    for key in ("allocated", "reserved"):
        result[f"peak_{key}_mib"] = result[f"peak_{key}_bytes"] / 2**20
    if preflight:
        result["steps"] = records
        for key in ("update_seconds", "wall_seconds_including_data", "loss"):
            result[f"mean_{key}"] = sum(r[key] for r in records) / len(records)
    atomic_write_json(output / f"{phase}-result.json", result)
    return result


def run_quickval(config, identity, output, final):
    from tools.mmfr.av1_quickval import main as evaluate
    sha = file_sha256(final)
    split = identity["splits"]["val-dev.txt"]
    evaluate(["--checkpoint", str(final), "--expected-checkpoint-sha256", sha,
              "--source-c0-checkpoint", identity["source_checkpoint"],
              "--expected-source-c0-sha256", SOURCE_SHA, "--dataset-root", config.dataset_path,
              "--split", split["path"], "--expected-split-sha256", split["sha256"],
              "--output-dir", str(output / "quickval")])
    return sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true", help="identity only; no model or dataset workers")
    parser.add_argument("--execute", choices=("preflight", "all", "diagnose"), help="explicit GPU start; diagnose never starts Gate/Quick-Val")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="full checkpoint only; ordinary formal resume requires same commit")
    parser.add_argument("--diagnostic-reference-dir", type=Path, help="original stopped run; diagnose only")
    args = parser.parse_args(argv)
    config = copy.deepcopy(importlib.import_module(CONFIG).C)
    if args.source_checkpoint:
        config.mmfr_av1["source_checkpoint"] = str(args.source_checkpoint.resolve())
    identity = check_inputs(config)
    if args.check_inputs:
        if args.execute or args.resume:
            parser.error("--check-inputs cannot execute/resume")
        print(json.dumps({"status": "inputs-verified-only", **identity, "cuda_available": torch.cuda.is_available(),
                          "model_constructed": False, "training_started": False}, indent=2))
        return 0
    if not args.execute or not args.output_dir:
        parser.error("GPU execution requires --execute and --output-dir")
    if args.resume and args.execute not in ("all", "diagnose"):
        parser.error("resume requires all or bounded diagnose")
    if args.execute == "diagnose" and (not args.resume or not args.diagnostic_reference_dir):
        parser.error("diagnose requires the original full recovery and stopped-run reference")
    if args.diagnostic_reference_dir and args.execute != "diagnose":
        parser.error("diagnostic reference is forbidden for ordinary training/resume")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("authorized existing RTX 4090 required; no CPU fallback")
    assert_committed()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "identity.json", {**identity, "argv": sys.argv,
                                               "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.perf_counter()
    try:
        resume = None
        diagnostic = None
        if args.resume:
            resume = torch.load(args.resume, map_location="cpu", weights_only=False)
            if args.execute == "diagnose":
                diagnostic = prepare_diagnostic(args.diagnostic_reference_dir.resolve(), args.resume,
                                                resume, identity, config)
                atomic_write_json(output / "diagnostic-compatibility.json",
                                  {k: v for k, v in diagnostic.items() if k != "expected"})
            else:
                resume_phase, resume_n = validate_resume(resume, identity, config)
                if resume_phase == "gate" and resume_n == 640:
                    raise RuntimeError("formal training already complete; refuse duplicate Quick-Val via resume")
        else:
            model, _ = build_model(config, identity)
            preflight = output / "preflight"
            preflight.mkdir()
            p = run_phase(model, config, identity, preflight, "proposal", preflight=True)
            g = run_phase(model, config, identity, preflight, "gate", preflight=True)
            atomic_write_json(output / "preflight-result.json", {"proposal": p, "gate": g, "status": "passed"})
            del model
            torch.cuda.empty_cache()
            if args.execute == "preflight":
                return 0
        model, load_info = build_model(config, identity)  # Discard every preflight tensor, draw and update.
        formal = output / "formal"
        formal.mkdir()
        if diagnostic:
            run_phase(model, config, identity, formal, "proposal", preflight=False,
                      resume=resume, diagnostic=diagnostic)
            print("Bounded diagnostic reached attempt1871 without error; no Gate/Quick-Val or formal completion claim", flush=True)
            return 0
        results = {}
        if resume is None or resume_phase == "proposal":
            results["proposal"] = run_phase(model, config, identity, formal, "proposal", preflight=False, resume=resume)
            resume = None  # Gate always gets a new optimizer/scaler and phase-local corruption stream.
        results["gate"] = run_phase(model, config, identity, formal, "gate", preflight=False, resume=resume)
        final = formal / "checkpoint/update-2560.pth"
        del model, resume
        torch.cuda.empty_cache()
        sha = file_sha256(final)
        atomic_write_json(output / "training-result.json", {"identity": identity, "source_load": load_info,
                          "phases": results, "global_completed": 2560, "global_skipped": 0,
                          "elapsed_seconds": time.perf_counter() - started, "fixed_final": str(final),
                          "fixed_final_sha256": sha, "status": "completed", "official_test_included": False})
        run_quickval(config, identity, output, final)
        print("A-v1 authorized run completed; stopping before Main-Val/test/additional training", flush=True)
    except BaseException:
        atomic_write_json(output / "stopped.json", {"identity": identity, "exception": traceback.format_exc(),
                          "elapsed_seconds": time.perf_counter() - started,
                          "official_test_included": False, "automatic_retry": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
