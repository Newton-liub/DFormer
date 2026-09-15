#!/usr/bin/env python3
"""Profile the MMFR-A2 v3 corruption training pipeline (implementation identity opt1).

This tool is read-only with respect to the frozen science: it does not modify any
production module.  It runs the real ``train-dev`` batches with the frozen batch size,
the frozen per-sample PCG64 corruption RNG and the frozen corruption distribution, and
measures where the wall clock of one optimization step goes.

Stage buckets (best achievable without editing the frozen helper):

    A  dataloader wait
    B  MMFR v3 corruption helper (whole call)
    D  host-to-device copy of the batch and auxiliary tensors
    EF model forward + segmentation loss + reliability BCE (one model call)
    G  backward
    H  optimizer / GradScaler step
    I  probe-style telemetry hashing (the audit probe's own extra cost)
    J  whole step wall clock

Corruption sub-stages are instrumented by wrapping the frozen kernel functions at
runtime (the ``_KERNELS`` dispatch table and the module-level callables).  Inline stages
inside ``build_mmfr_training_batch_v3`` that cannot be separated without editing it
(validity-state update, R_syn/R_sup construction, telemetry masks, tensor
reconstruction) are additionally attributed through an embedded ``cProfile`` sample.

Nothing here reads the official test split.  The tool never writes into the repository
tree except under ``outputs/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import cProfile
import importlib
import io
import json
import os
import pstats
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.builder import EncoderDecoder as segmodel  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.dataloader.dataloader import get_train_loader  # noqa: E402
from utils.dataloader import mmfr_training_v3 as mmfr_v3  # noqa: E402
from utils.dataloader import multimodal_failure_v3 as mf_v3  # noqa: E402
from utils.init_func import group_weight  # noqa: E402
from utils.lr_policy import WarmUpPolyLR  # noqa: E402
from tools.museg_protocol import write_json  # noqa: E402


# --------------------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------------------


class StageTimers:
    """Accumulate wall-clock durations per stage name."""

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self.samples: Dict[str, List[float]] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.calls[name] = self.calls.get(name, 0) + 1
        self.samples.setdefault(name, []).append(float(seconds))

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, values in self.samples.items():
            out[name] = {
                "calls": int(self.calls[name]),
                "total_seconds": round(float(self.totals[name]), 6),
                "mean_seconds": round(float(statistics.fmean(values)), 6),
                "median_seconds": round(float(statistics.median(values)), 6),
                "p50_seconds": round(_percentile(values, 50), 6),
                "p90_seconds": round(_percentile(values, 90), 6),
                "p95_seconds": round(_percentile(values, 95), 6),
                "max_seconds": round(float(max(values)), 6),
            }
        return out


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class WrappedCallable:
    """Wrap a module-level function so every call is timed into ``timers``."""

    def __init__(self, name: str, func: Callable[..., Any], timers: StageTimers, prefix: str) -> None:
        self._name = name
        self._func = func
        self._timers = timers
        self._prefix = prefix

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._func(*args, **kwargs)
        finally:
            self._timers.add(f"{self._prefix}{self._name}", time.perf_counter() - started)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._func, item)


def instrument_corruption(timers: StageTimers) -> Dict[str, Any]:
    """Runtime instrumentation of the frozen corruption kernels (no source edits)."""
    kinds: List[str] = list(mf_v3.FAILURE_KINDS)
    kind_counts: Dict[str, int] = {kind: 0 for kind in kinds}
    patched: Dict[str, Any] = {}
    for kind in kinds:
        original = mf_v3._KERNELS[kind]
        wrapper = WrappedCallable(f"corruption_kernel.{kind}", original, timers, "")
        mf_v3._KERNELS[kind] = wrapper
        patched[kind] = {"function": getattr(original, "__name__", str(original)), "wrapper": wrapper}
    for name in (
        "apply_failures",
        "sample_failure_specs",
        "curriculum_progress",
    ):
        if hasattr(mf_v3, name):
            setattr(mf_v3, name, WrappedCallable(name, getattr(mf_v3, name), timers, "a1_v3."))
    for name in (
        "sample_depth_failure_specs_v3",
        "build_mmfr_training_batch_v3",
        "apply_failures",
    ):
        setattr(mmfr_v3, name, WrappedCallable(name, getattr(mmfr_v3, name), timers, "a2_v3."))
    for owner, name in (
        (mmfr_v3, "normalized_to_uint8"),
        (mmfr_v3, "uint8_to_normalized"),
        (mmfr_v3, "make_sample_generator"),
        (mmfr_v3, "build_seed_words"),
        (mmfr_v3, "sample_id_words"),
    ):
        setattr(owner, name, WrappedCallable(name, getattr(owner, name), timers, "roundtrip."))
    return {"kernels": patched, "kind_counts": kind_counts}


def count_kinds(metadata: Sequence[Mapping[str, Any]], counters: Dict[str, int]) -> Dict[str, Any]:
    specs_per_batch = 0
    clean = 0
    for record in metadata:
        if record.get("clean"):
            clean += 1
        for spec in record.get("specs") or ():
            kind = spec.get("kind") if isinstance(spec, Mapping) else None
            if kind in counters:
                counters[kind] += 1
            specs_per_batch += 1
    return {"specs_in_batch": specs_per_batch, "clean_samples": clean}


class UtilizationSampler:
    """Sample GPU and CPU utilization on a background thread."""

    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval = float(interval_seconds)
        self.gpu_samples: List[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._handle = None
        self._cpu_before: Tuple[int, int] | None = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None

    def _cpu_snapshot(self) -> Tuple[int, int]:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            parts = [int(value) for value in handle.readline().split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        return idle, sum(parts)

    def start(self) -> None:
        self._cpu_before = self._cpu_snapshot()
        if self._pynvml is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._pynvml is not None and self._handle is not None
        while not self._stop.is_set():
            try:
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.gpu_samples.append(float(utilization.gpu))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        cpu_after = self._cpu_snapshot()
        result: Dict[str, Any] = {"gpu_samples": len(self.gpu_samples)}
        if self.gpu_samples:
            values = np.asarray(self.gpu_samples, dtype=np.float64)
            result.update(
                {
                    "gpu_util_mean_percent": round(float(values.mean()), 3),
                    "gpu_util_median_percent": round(float(np.median(values)), 3),
                    "gpu_util_p95_percent": round(float(np.percentile(values, 95)), 3),
                    "gpu_util_idle_sample_fraction": round(float(np.mean(values <= 1.0)), 6),
                }
            )
        if self._cpu_before is not None:
            idle_delta = cpu_after[0] - self._cpu_before[0]
            total_delta = cpu_after[1] - self._cpu_before[1]
            if total_delta > 0:
                result["cpu_busy_percent"] = round(100.0 * (1.0 - idle_delta / total_delta), 3)
                result["cpu_logical_cores"] = int(os.cpu_count() or 0)
        return result


def cuda_timed(call: Callable[[], Any]) -> Tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = call()
    end.record()
    torch.cuda.synchronize()
    return value, float(start.elapsed_time(end)) / 1000.0


# --------------------------------------------------------------------------------------
# pipeline construction (mirrors utils/train.py without modifying it)
# --------------------------------------------------------------------------------------


class _SingleProcessEngine:
    """Minimal stand-in for ``utils.engine.engine.Engine`` used only by the loader helper.

    The production single-GPU probe runs with ``WORLD_SIZE=1``, i.e. ``distributed=False``;
    the loader helper only reads ``engine.distributed``, so a stub keeps this tool free of
    torchrun assumptions (the real ``Engine`` requires ``LOCAL_RANK`` even when not
    distributed).
    """

    distributed = False
    world_size = 1
    local_rank = 0


def build_pipeline(config_module: str, *, syncbn: bool, amp: bool) -> Dict[str, Any]:
    config = getattr(importlib.import_module(config_module), "C")
    engine = _SingleProcessEngine()
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    norm_layer = nn.SyncBatchNorm if syncbn else nn.BatchNorm2d
    model = segmodel(cfg=config, criterion=criterion, norm_layer=norm_layer, syncbn=syncbn)
    if getattr(model, "reliability_estimator", None) is None:
        raise SystemExit("the profiling config must enable the MMFR reliability head")
    params_list: List[Dict[str, Any]] = []
    group_weight(params_list, model, norm_layer, config.lr)
    optimizer = torch.optim.AdamW(
        params_list, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay
    )
    lr_policy = WarmUpPolyLR(
        config.lr,
        config.lr_power,
        config.nepochs * config.niters_per_epoch,
        config.niters_per_epoch * config.warm_up_epoch,
    )
    model.to(torch.device("cuda"))
    scaler = torch.cuda.amp.GradScaler() if amp else None
    train_loader, _ = get_train_loader(engine, RGBXDataset, config)
    return {
        "config": config,
        "engine": engine,
        "model": model,
        "optimizer": optimizer,
        "lr_policy": lr_policy,
        "scaler": scaler,
        "loader": train_loader,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default="local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3",
        help="training config module to profile",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--cprofile-steps", type=int, default=3)
    parser.add_argument("--sample-interval-ms", type=float, default=5.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-syncbn", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "mmfr-a2-v3-pipeline-profile"),
    )
    parser.add_argument("--tag", default="before", help="label for the produced profile (before/after)")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="measure only the CPU data path and the corruption helper; never builds or touches the GPU",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026091402,
        help="seed for python/numpy/torch so before/after runs consume the same batches",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="override dataloader workers for the CPU-only mode (-1 keeps the config value)",
    )
    args = parser.parse_args(argv)
    if args.measured_steps < 60:
        parser.error("measured steps below 60 are not acceptable for this profile")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cpu_only:
        return _run_cpu_only(args)
    torch.cuda.set_device(0)
    pipeline = build_pipeline(args.config, syncbn=not args.no_syncbn, amp=not args.no_amp)
    config = pipeline["config"]
    model = pipeline["model"]
    optimizer = pipeline["optimizer"]
    lr_policy = pipeline["lr_policy"]
    scaler = pipeline["scaler"]
    loader = pipeline["loader"]

    mmfr_a2 = dict(getattr(config, "mmfr_a2", {}) or {})
    corruption_cfg = dict(mmfr_a2.get("corruption") or {})
    corruption_seed = int(corruption_cfg.get("seed", mmfr_v3.CORRUPTION_SEED))
    p_clean = float(corruption_cfg.get("p_clean", mmfr_v3.CLEAN_PROBABILITY))
    max_specs = int(corruption_cfg.get("max_specs", mmfr_v3.MAX_SPECS))
    nepochs = int(config.nepochs)
    niters_per_epoch = int(config.niters_per_epoch)
    config_tag = {
        "config_module": args.config,
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "amp": not args.no_amp,
        "syncbn": not args.no_syncbn,
        "corruption_seed": corruption_seed,
        "p_clean": p_clean,
        "max_specs": max_specs,
        "nepochs": nepochs,
        "niters_per_epoch": niters_per_epoch,
    }

    timers = StageTimers()
    instrumentation = instrument_corruption(timers)
    kind_counters: Dict[str, int] = instrumentation["kind_counts"]
    runtime = mmfr_v3.build_mmfr_training_batch_v3  # wrapped above

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    total_steps = args.warmup_steps + args.measured_steps
    measured: List[Dict[str, Any]] = []
    kind_totals: Dict[str, int] = {kind: 0 for kind in mf_v3.FAILURE_KINDS}
    spec_totals: List[int] = []
    clean_totals = 0
    iterator = iter(loader)
    sampler = UtilizationSampler(interval_seconds=args.sample_interval_ms / 1000.0)
    sampler.start()
    wall_started = time.perf_counter()

    for step in range(total_steps):
        is_warmup = step < args.warmup_steps
        # spread the measured window over the frozen 500-epoch curriculum so the mix of
        # severity stages is representative instead of early-curriculum only.  Warmup steps
        # stay at epoch 1 and the epoch is always clamped to the frozen epoch count so the
        # frozen poly LR schedule is never evaluated beyond its total iterations.
        measured_index = step - args.warmup_steps
        if measured_index < 0 or args.measured_steps <= 1:
            epoch = 1
        else:
            epoch = 1 + int(measured_index * (nepochs - 1) / (args.measured_steps - 1))
        epoch = max(1, min(nepochs, epoch))
        iteration = step % niters_per_epoch

        step_started = time.perf_counter()
        wait_started = time.perf_counter()
        minibatch = next(iterator)
        wait_seconds = time.perf_counter() - wait_started

        imgs = minibatch["data"]
        gts = minibatch["label"]
        modal_xs = minibatch["modal_x"]
        corrupt_started = time.perf_counter()
        mmfr_batch = runtime(
            imgs,
            modal_xs,
            minibatch.get("fn"),
            epoch=epoch,
            iteration=iteration,
            niters_per_epoch=niters_per_epoch,
            nepochs=nepochs,
            rgb_mean=config.norm_mean,
            rgb_std=config.norm_std,
            corruption_seed=corruption_seed,
            global_rank=0,
            p_clean=p_clean,
            max_specs=max_specs,
            sample_id_root=getattr(config, "dataset_path", None),
        )
        corrupt_seconds = time.perf_counter() - corrupt_started

        counters = count_kinds(mmfr_batch["metadata"], kind_counters)
        spec_totals.append(counters["specs_in_batch"])
        clean_totals += counters["clean_samples"]

        imgs = mmfr_batch["rgb"]
        modal_xs = mmfr_batch["depth"]
        torch.cuda.synchronize()
        h2d_started = time.perf_counter()
        imgs_gpu = imgs.cuda(non_blocking=True)
        gts_gpu = gts.cuda(non_blocking=True)
        modal_xs_gpu = modal_xs.cuda(non_blocking=True)
        auxiliary = {
            "raw_rgb": mmfr_batch["raw_rgb"].cuda(non_blocking=True),
            "raw_depth": mmfr_batch["raw_depth"].cuda(non_blocking=True),
            "reliability_target": mmfr_batch["reliability_target"].cuda(non_blocking=True),
            "reliability_valid_mask": mmfr_batch["valid_mask"].cuda(non_blocking=True),
            "depth_valid": mmfr_batch["depth_valid_post"].cuda(non_blocking=True),
            "reliability_telemetry_masks": mmfr_batch["telemetry_masks"].cuda(non_blocking=True),
        }
        torch.cuda.synchronize()
        h2d_seconds = time.perf_counter() - h2d_started

        lr = lr_policy.get_lr((epoch - 1) * niters_per_epoch + iteration)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        def forward_call() -> torch.Tensor:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(imgs_gpu, modal_xs_gpu, gts_gpu, **auxiliary)

        loss, forward_seconds = cuda_timed(forward_call)
        backward_seconds = 0.0
        step_seconds_gpu = 0.0
        if scaler is not None:
            def backward_call() -> None:
                scaler.scale(loss).backward()

            _, backward_seconds = cuda_timed(backward_call)

            def step_call() -> None:
                scaler.step(optimizer)
                scaler.update()

            _, step_seconds_gpu = cuda_timed(step_call)
        else:
            def backward_call_noscale() -> None:
                loss.backward()

            _, backward_seconds = cuda_timed(backward_call_noscale)

            def step_call_plain() -> None:
                optimizer.step()

            _, step_seconds_gpu = cuda_timed(step_call_plain)

        # probe-style telemetry cost: the audit probe hashes rgb/depth/label every step
        import hashlib

        telemetry_started = time.perf_counter()
        _ = {
            "rgb": hashlib.sha256(imgs_gpu.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "depth": hashlib.sha256(modal_xs_gpu.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "label": hashlib.sha256(gts_gpu.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        }
        telemetry_seconds = time.perf_counter() - telemetry_started

        step_seconds = time.perf_counter() - step_started
        if not is_warmup:
            measured.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "iteration": iteration,
                    "dataloader_wait_seconds": wait_seconds,
                    "corruption_seconds": corrupt_seconds,
                    "h2d_seconds": h2d_seconds,
                    "forward_and_loss_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                    "optimizer_step_seconds": step_seconds_gpu,
                    "telemetry_hash_seconds": telemetry_seconds,
                    "step_wall_seconds": step_seconds,
                    "gpu_busy_seconds": forward_seconds + backward_seconds + step_seconds_gpu,
                    "images_in_batch": int(config.batch_size),
                    "specs_in_batch": counters["specs_in_batch"],
                    "clean_samples": counters["clean_samples"],
                    "loss": float(loss.detach()),
                }
            )

    wall_seconds = time.perf_counter() - wall_started
    utilization = sampler.stop()
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    # embedded cProfile sample over a few fresh steps (relative attribution only)
    profile_text = ""
    if args.cprofile_steps > 0:
        profiler = cProfile.Profile()
        profiler.enable()
        for step in range(args.cprofile_steps):
            minibatch = next(iterator)
            _ = runtime(
                minibatch["data"],
                minibatch["modal_x"],
                minibatch.get("fn"),
                epoch=1 + (step * 17) % nepochs,
                iteration=step,
                niters_per_epoch=niters_per_epoch,
                nepochs=nepochs,
                rgb_mean=config.norm_mean,
                rgb_std=config.norm_std,
                corruption_seed=corruption_seed,
                global_rank=0,
                p_clean=p_clean,
                max_specs=max_specs,
                sample_id_root=getattr(config, "dataset_path", None),
            )
        profiler.disable()
        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer).sort_stats("tottime")
        stats.print_stats(40)
        profile_text = buffer.getvalue()

    def column(name: str) -> Dict[str, float]:
        values = [row[name] for row in measured]
        return {
            "mean": round(float(statistics.fmean(values)), 6),
            "median": round(float(statistics.median(values)), 6),
            "p50": round(_percentile(values, 50), 6),
            "p90": round(_percentile(values, 90), 6),
            "p95": round(_percentile(values, 95), 6),
            "max": round(float(max(values)), 6),
        }

    step_wall = column("step_wall_seconds")
    per_step_summary = {
        name: column(name)
        for name in (
            "dataloader_wait_seconds",
            "corruption_seconds",
            "h2d_seconds",
            "forward_and_loss_seconds",
            "backward_seconds",
            "optimizer_step_seconds",
            "telemetry_hash_seconds",
            "step_wall_seconds",
            "gpu_busy_seconds",
        )
    }
    mean_step = step_wall["mean"]
    mean_gpu_busy = per_step_summary["gpu_busy_seconds"]["mean"]
    mean_corruption = per_step_summary["corruption_seconds"]["mean"]
    mean_wait = per_step_summary["dataloader_wait_seconds"]["mean"]
    images_per_second = float(config.batch_size) / mean_step if mean_step > 0 else float("nan")

    kernel_times = {
        name: payload
        for name, payload in timers.summary().items()
        if name.startswith("corruption_kernel.") or name.startswith("roundtrip.") or name.startswith("a1_v3.") or name.startswith("a2_v3.")
    }
    per_kind = {
        kind: {
            "count": int(kind_counters.get(kind, 0)),
            "total_seconds": kernel_times.get(f"corruption_kernel.{kind}", {}).get("total_seconds"),
            "mean_seconds": kernel_times.get(f"corruption_kernel.{kind}", {}).get("mean_seconds"),
        }
        for kind in mf_v3.FAILURE_KINDS
    }

    report: Dict[str, Any] = {
        "schema_version": "mmfr-a2-v3-pipeline-profile-v1",
        "tag": args.tag,
        "scientific_protocol": mmfr_v3.PROTOCOL_ID,
        "scientific_semantics_changed": False,
        "implementation_identity": "MMFR-A2-v3-pipeline-opt1",
        "official_test_included": False,
        "config": config_tag,
        "measurement": {
            "warmup_steps": args.warmup_steps,
            "measured_steps": len(measured),
            "cprofile_steps": args.cprofile_steps,
            "curriculum_sampling": "epoch spread uniformly over [1, nepochs] across measured steps",
            "wall_seconds_total": round(wall_seconds, 3),
        },
        "throughput": {
            "images_per_second": round(images_per_second, 6),
            "step_wall_seconds": step_wall,
            "seconds_per_step": step_wall["mean"],
        },
        "per_step_stage_seconds": per_step_summary,
        "stage_share_of_step": {
            "dataloader_wait": round(mean_wait / mean_step, 6),
            "corruption": round(mean_corruption / mean_step, 6),
            "h2d": round(per_step_summary["h2d_seconds"]["mean"] / mean_step, 6),
            "forward_and_loss": round(per_step_summary["forward_and_loss_seconds"]["mean"] / mean_step, 6),
            "backward": round(per_step_summary["backward_seconds"]["mean"] / mean_step, 6),
            "optimizer_step": round(per_step_summary["optimizer_step_seconds"]["mean"] / mean_step, 6),
            "telemetry_hash": round(per_step_summary["telemetry_hash_seconds"]["mean"] / mean_step, 6),
        },
        "gpu_compute_fraction_of_step": round(mean_gpu_busy / mean_step, 6),
        "cpu_bound_evidence": {
            "mean_corruption_seconds": round(mean_corruption, 6),
            "mean_gpu_compute_seconds": round(mean_gpu_busy, 6),
            "mean_corruption_minus_gpu_compute_seconds": round(mean_corruption - mean_gpu_busy, 6),
            "corruption_over_gpu_compute_ratio": round(mean_corruption / mean_gpu_busy, 6) if mean_gpu_busy else None,
            "verdict": (
                "pipeline CPU-bound / producer-bound"
                if mean_corruption > mean_gpu_busy
                else "pipeline GPU-compute-bound; do not start a CPU-corruption rewrite"
            ),
        },
        "corruption_internals": {
            "instrumented_stages": kernel_times,
            "per_kind": per_kind,
            "specs_per_batch": {
                "mean": round(float(statistics.fmean(spec_totals)), 6) if spec_totals else None,
                "max": int(max(spec_totals)) if spec_totals else None,
                "total": int(sum(spec_totals)),
            },
            "clean_samples_total": int(clean_totals),
            "not_separately_measurable_without_editing_frozen_helper": [
                "validity-state update",
                "R_syn / R_sup construction",
                "telemetry mask construction",
                "normalization / tensor reconstruction",
            ],
        },
        "utilization": utilization,
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "cprofile_top_functions": profile_text,
        "per_step_records": measured,
        "notes": [
            "forward and segmentation loss and reliability BCE share one model call in this codebase, so E and F are reported together",
            "telemetry_hash_seconds reproduces the audit probe's own per-step SHA-256 cost and is not part of production training",
            "cuDNN benchmark is left at the environment default; no torch.compile, no batch/size/hyperparameter change",
        ],
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / f"profile-{args.tag}.json"
    write_json(profile_path, report)
    summary_path = output_dir / f"profile-summary-{args.tag}.md"
    summary_path.write_text(_render_summary(report), encoding="utf-8")
    print(json.dumps({
        "profile": str(profile_path),
        "summary": str(summary_path),
        "images_per_second": report["throughput"]["images_per_second"],
        "seconds_per_step": round(mean_step, 4),
        "corruption_seconds": round(mean_corruption, 4),
        "gpu_compute_seconds": round(mean_gpu_busy, 4),
        "dataloader_wait_seconds": round(mean_wait, 4),
        "verdict": report["cpu_bound_evidence"]["verdict"],
        "cpu_busy_percent": utilization.get("cpu_busy_percent"),
        "gpu_util_mean_percent": utilization.get("gpu_util_mean_percent"),
        "gpu_util_idle_sample_fraction": utilization.get("gpu_util_idle_sample_fraction"),
    }, indent=2, ensure_ascii=False))
    return 0


def build_loader_only(config_module: str, workers: int = -1):
    """CPU-only loader construction that never touches CUDA."""
    config = getattr(importlib.import_module(config_module), "C")
    if workers >= 0:
        config.num_workers = int(workers)
    loader, _ = get_train_loader(_SingleProcessEngine(), RGBXDataset, config)
    return config, loader


def _run_cpu_only(args: argparse.Namespace) -> int:
    """Time the CPU half of the pipeline (data path + corruption helper) with no GPU use."""
    config, loader = build_loader_only(args.config, args.workers)
    mmfr_a2 = dict(getattr(config, "mmfr_a2", {}) or {})
    corruption_cfg = dict(mmfr_a2.get("corruption") or {})
    corruption_seed = int(corruption_cfg.get("seed", mmfr_v3.CORRUPTION_SEED))
    p_clean = float(corruption_cfg.get("p_clean", mmfr_v3.CLEAN_PROBABILITY))
    max_specs = int(corruption_cfg.get("max_specs", mmfr_v3.MAX_SPECS))
    nepochs = int(config.nepochs)
    niters_per_epoch = int(config.niters_per_epoch)

    timers = StageTimers()
    instrumentation = instrument_corruption(timers)
    kind_counters: Dict[str, int] = instrumentation["kind_counts"]
    # the validation function is a module-level global lookup inside the helper, so it can
    # be timed the same way without editing the frozen helper
    original_validate = getattr(mmfr_v3, "_validate_batch_outputs_v3", None)
    if original_validate is not None:
        setattr(
            mmfr_v3,
            "_validate_batch_outputs_v3",
            WrappedCallable("_validate_batch_outputs_v3", original_validate, timers, "a2_v3."),
        )
    runtime = mmfr_v3.build_mmfr_training_batch_v3

    total_steps = args.warmup_steps + args.measured_steps
    rows: List[Dict[str, Any]] = []
    iterator = iter(loader)
    sampler = UtilizationSampler(interval_seconds=args.sample_interval_ms / 1000.0)
    sampler.start()
    wall_started = time.perf_counter()
    for step in range(total_steps):
        is_warmup = step < args.warmup_steps
        measured_index = step - args.warmup_steps
        if measured_index < 0 or args.measured_steps <= 1:
            epoch = 1
        else:
            epoch = 1 + int(measured_index * (nepochs - 1) / (args.measured_steps - 1))
        epoch = max(1, min(nepochs, epoch))
        iteration = step % niters_per_epoch
        wait_started = time.perf_counter()
        minibatch = next(iterator)
        wait_seconds = time.perf_counter() - wait_started
        started = time.perf_counter()
        outputs = runtime(
            minibatch["data"],
            minibatch["modal_x"],
            minibatch.get("fn"),
            epoch=epoch,
            iteration=iteration,
            niters_per_epoch=niters_per_epoch,
            nepochs=nepochs,
            rgb_mean=config.norm_mean,
            rgb_std=config.norm_std,
            corruption_seed=corruption_seed,
            global_rank=0,
            p_clean=p_clean,
            max_specs=max_specs,
            sample_id_root=getattr(config, "dataset_path", None),
        )
        helper_seconds = time.perf_counter() - started
        counters = count_kinds(outputs["metadata"], kind_counters)
        if not is_warmup:
            rows.append(
                {
                    "epoch": epoch,
                    "iteration": iteration,
                    "dataloader_wait_seconds": wait_seconds,
                    "corruption_seconds": helper_seconds,
                    "batch_size": int(config.batch_size),
                    "specs_in_batch": counters["specs_in_batch"],
                }
            )
    wall_seconds = time.perf_counter() - wall_started
    utilization = sampler.stop()

    def column(name: str) -> Dict[str, float]:
        values = [row[name] for row in rows]
        return {
            "mean": round(float(statistics.fmean(values)), 6),
            "median": round(float(statistics.median(values)), 6),
            "p50": round(_percentile(values, 50), 6),
            "p90": round(_percentile(values, 90), 6),
            "p95": round(_percentile(values, 95), 6),
            "max": round(float(max(values)), 6),
        }

    summary = timers.summary()
    helper_summary = column("corruption_seconds")
    validation_seconds = summary.get("a2_v3._validate_batch_outputs_v3", {})
    helper_calls = int(summary.get("a2_v3.build_mmfr_training_batch_v3", {}).get("calls", len(rows)))
    validation_calls = int(validation_seconds.get("calls", helper_calls)) or helper_calls
    # nested timings: subtract the per-call validation cost from the per-call helper cost
    # to expose the per-sample loop that remains after Opt-A (both are per-call averages).
    validation_per_call = float(validation_seconds.get("total_seconds", 0.0)) / max(validation_calls, 1)
    report: Dict[str, Any] = {
        "schema_version": "mmfr-a2-v3-pipeline-profile-v1",
        "tag": args.tag,
        "scope": "cpu-only: dataloader wait + corruption helper (no model, no CUDA use)",
        "scientific_protocol": mmfr_v3.PROTOCOL_ID,
        "scientific_semantics_changed": False,
        "implementation_identity": "MMFR-A2-v3-pipeline-opt1",
        "official_test_included": False,
        "config": {
            "config_module": args.config,
            "batch_size": int(config.batch_size),
            "num_workers": int(config.num_workers),
            "corruption_seed": corruption_seed,
            "p_clean": p_clean,
            "max_specs": max_specs,
            "nepochs": nepochs,
            "niters_per_epoch": niters_per_epoch,
            "seed": args.seed,
        },
        "measurement": {
            "warmup_steps": args.warmup_steps,
            "measured_steps": len(rows),
            "wall_seconds_total": round(wall_seconds, 3),
            "curriculum_sampling": "epoch spread uniformly over [1, nepochs] across measured steps",
        },
        "corruption_helper_seconds": helper_summary,
        "corruption_helper_images_per_second": round(
            float(config.batch_size) / helper_summary["mean"], 6
        ),
        "validation_seconds_per_call": round(validation_per_call, 6),
        "sample_loop_seconds_per_call_excluding_validation": round(
            helper_summary["mean"] - validation_per_call, 6
        ),
        "utilization": utilization,
        "corruption_internals": {
            "instrumented_stages": summary,
            "per_kind": {
                kind: {
                    "count": int(kind_counters.get(kind, 0)),
                    "total_seconds": summary.get(f"corruption_kernel.{kind}", {}).get("total_seconds"),
                    "mean_seconds": summary.get(f"corruption_kernel.{kind}", {}).get("mean_seconds"),
                }
                for kind in mf_v3.FAILURE_KINDS
            },
        },
        "per_step_records": rows,
        "notes": [
            "CPU-only scope: the GPU forward/backward buckets are not measured here",
            "python/numpy/torch are seeded so before/after runs consume the same batches",
        ],
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / f"profile-cpu-{args.tag}.json"
    write_json(profile_path, report)
    summary_path = output_dir / f"profile-cpu-summary-{args.tag}.md"
    summary_path.write_text(_render_cpu_summary(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "profile": str(profile_path),
                "summary": str(summary_path),
                "scope": report["scope"],
                "corruption_seconds_mean": helper_summary["mean"],
                "corruption_seconds_median": helper_summary["median"],
                "corruption_images_per_second": report["corruption_helper_images_per_second"],
                "validation_seconds_per_call": report["validation_seconds_per_call"],
                "sample_loop_seconds_per_call_excluding_validation": report[
                    "sample_loop_seconds_per_call_excluding_validation"
                ],
                "cpu_busy_percent": utilization.get("cpu_busy_percent"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _render_cpu_summary(report: Mapping[str, Any]) -> str:
    helper = report["corruption_helper_seconds"]
    stages = report["corruption_internals"]["instrumented_stages"]
    lines = [
        f"# MMFR-A2 v3 corruption helper CPU profile ({report['tag']})",
        "",
        f"- scope: {report['scope']}",
        f"- scientific protocol: `{report['scientific_protocol']}` (semantics changed: {report['scientific_semantics_changed']})",
        f"- measured steps: {report['measurement']['measured_steps']} (warmup {report['measurement']['warmup_steps']})",
        f"- corruption helper: **{helper['mean']:.4f} s/batch mean**, "
        f"{helper['median']:.4f} median, {helper['p95']:.4f} P95, {helper['max']:.4f} max",
        f"- corruption helper throughput: **{report['corruption_helper_images_per_second']:.3f} images/s** "
        f"(batch of {report['config']['batch_size']})",
        f"- `_validate_batch_outputs_v3`: **{report['validation_seconds_per_call']:.4f} s/call**",
        f"- per-sample loop excluding validation: "
        f"**{report['sample_loop_seconds_per_call_excluding_validation']:.4f} s/call**",
        f"- CPU busy: {report['utilization'].get('cpu_busy_percent')}%",
        "",
        "## Timed stages (total seconds over the measured window)",
        "",
        "| stage | calls | total s | mean s |",
        "|---|---|---|---|",
    ]
    for name, payload in sorted(
        stages.items(), key=lambda item: -(item[1]["total_seconds"] or 0.0)
    )[:14]:
        lines.append(
            f"| {name} | {payload['calls']} | {payload['total_seconds']} | {payload['mean_seconds']} |"
        )
    lines += [
        "",
        "## Corruption kernels",
        "",
        "| kind | occurrences | total s | mean s |",
        "|---|---|---|---|",
    ]
    for kind, payload in report["corruption_internals"]["per_kind"].items():
        lines.append(
            f"| {kind} | {payload['count']} | {payload['total_seconds']} | {payload['mean_seconds']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_summary(report: Mapping[str, Any]) -> str:
    stages = report["per_step_stage_seconds"]
    share = report["stage_share_of_step"]
    per_kind = report["corruption_internals"]["per_kind"]
    utilization = report["utilization"]
    lines = [
        f"# MMFR-A2 v3 corruption pipeline profile ({report['tag']})",
        "",
        f"- scientific protocol: `{report['scientific_protocol']}` (semantics changed: {report['scientific_semantics_changed']})",
        f"- implementation identity: `{report['implementation_identity']}`",
        f"- official test included: {report['official_test_included']}",
        f"- measured steps: {report['measurement']['measured_steps']} (warmup {report['measurement']['warmup_steps']})",
        f"- throughput: **{report['throughput']['images_per_second']:.3f} images/s**, "
        f"{report['throughput']['seconds_per_step']:.4f} s/step",
        "",
        "## Per-step stage wall clock (seconds)",
        "",
        "| stage | mean | median | P90 | P95 | max | share of step |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in (
        ("dataloader_wait_seconds", "A dataloader wait"),
        ("corruption_seconds", "B corruption helper"),
        ("h2d_seconds", "D host-to-device"),
        ("forward_and_loss_seconds", "E+F forward + loss"),
        ("backward_seconds", "G backward"),
        ("optimizer_step_seconds", "H optimizer/scaler"),
        ("telemetry_hash_seconds", "I probe telemetry hash"),
        ("step_wall_seconds", "J whole step"),
        ("gpu_busy_seconds", "GPU busy (E+F+G+H)"),
    ):
        row = stages[key]
        share_text = ""
        lookup = {
            "dataloader_wait_seconds": share["dataloader_wait"],
            "corruption_seconds": share["corruption"],
            "h2d_seconds": share["h2d"],
            "forward_and_loss_seconds": share["forward_and_loss"],
            "backward_seconds": share["backward"],
            "optimizer_step_seconds": share["optimizer_step"],
            "telemetry_hash_seconds": share["telemetry_hash"],
        }
        if key in lookup:
            share_text = f"{100.0 * lookup[key]:.1f}%"
        elif key == "gpu_busy_seconds":
            share_text = f"{100.0 * report['gpu_compute_fraction_of_step']:.1f}%"
        elif key == "step_wall_seconds":
            share_text = "100.0%"
        lines.append(
            f"| {label} | {row['mean']:.4f} | {row['median']:.4f} | {row['p90']:.4f} | "
            f"{row['p95']:.4f} | {row['max']:.4f} | {share_text} |"
        )
    evidence = report["cpu_bound_evidence"]
    lines += [
        "",
        "## CPU-bound check",
        "",
        f"- mean corruption wall: {evidence['mean_corruption_seconds']:.4f} s",
        f"- mean GPU compute: {evidence['mean_gpu_compute_seconds']:.4f} s",
        f"- difference: {evidence['mean_corruption_minus_gpu_compute_seconds']:.4f} s",
        f"- ratio: {evidence['corruption_over_gpu_compute_ratio']}",
        f"- verdict: **{evidence['verdict']}**",
        f"- CPU busy: {utilization.get('cpu_busy_percent')}% over {utilization.get('cpu_logical_cores')} logical cores",
        f"- GPU util: mean {utilization.get('gpu_util_mean_percent')}%, "
        f"median {utilization.get('gpu_util_median_percent')}%, "
        f"idle-sample fraction {utilization.get('gpu_util_idle_sample_fraction')}",
        "",
        "## Corruption kernels",
        "",
        "| kind | occurrences | total s | mean s |",
        "|---|---|---|---|",
    ]
    for kind, payload in per_kind.items():
        lines.append(
            f"| {kind} | {payload['count']} | {payload['total_seconds']} | {payload['mean_seconds']} |"
        )
    internals = report["corruption_internals"]
    lines += [
        "",
        f"- specs per batch: mean {internals['specs_per_batch']['mean']}, max {internals['specs_per_batch']['max']}, "
        f"total {internals['specs_per_batch']['total']}",
        f"- clean samples: {internals['clean_samples_total']}",
        "- stages not separable without editing the frozen helper: "
        + ", ".join(internals["not_separately_measurable_without_editing_frozen_helper"]),
        "",
        "## Memory",
        "",
        f"- peak allocated: {report['memory']['peak_allocated_bytes']} bytes",
        f"- peak reserved: {report['memory']['peak_reserved_bytes']} bytes",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
