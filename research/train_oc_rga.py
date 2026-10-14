"""OC-RGA single-GPU training entry (research layer).

The author ``utils/train.py`` is untouched.  This entry provides what the OC-RGA
study needs:

* ``--model-mode`` selects ``masked_baseline`` | ``oc_rga`` | ``mean_relation`` |
  ``diag_relation`` | ``direct_post``; the first round only trains
  ``masked_baseline`` and ``oc_rga``, the other three are pre-registered read-out
  controls that share the same budget and initialisation;
* the OC-RGA data protocol (three support markers + the shared degradation):
  inference inputs are exactly ``RGB``, the degraded Depth, ``S`` and ``O``;
* the joint objective ``CE + lambda_g(e) L_cond + lambda_o(e) L_obs``;
* the frozen schedule: 120 epochs, 5 epoch warmup, 50 epoch shared observation
  point.  ``--stop-after-epoch`` is a pause and never compresses the schedule;
* optimizer groups where the new geometry branch runs at lr x2 and its new
  positive-parameter logits are not decayed, with every trainable tensor covered
  exactly once;
* FP16 AMP + GradScaler, FP32 probability maths and a shared FP32 attention
  softmax; gradient accumulation with a single global-norm clip at 1.0;
* full resumable snapshots (model / optimizer / scaler / counters / RNG /
  resolved config / contract) at every epoch plus the pre-registered stage
  points, and a strict contract check on resume;
* ``--smoke-steps N`` runs N real optimizer attempts for the later GPU
  qualification run.  A smoke snapshot is never resumable.

``--help`` and ``--print-schedule`` are CPU-only; training itself is CUDA-only.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from copy import deepcopy
from importlib import import_module

import torch
import torch.nn as nn

from research import oc_rga_data as ocdata
from research import odg_schedule as sched
from research.oc_rga_model import ALL_MODES, build_model_for_mode, build_oc_optimizer_param_groups

CHECKPOINT_FORMAT = "oc-rga-checkpoint-v1"

# Fields that must match on resume in addition to the shared contract fields.
OC_CONTRACT_FIELDS = (
    "model_mode",
    "oc_rank",
    "oc_tau",
    "oc_sigma_ref",
    "loss_lambda_cond_start",
    "loss_lambda_cond_end",
    "loss_lambda_obs_start",
    "loss_lambda_obs_end",
    "loss_lambda_ramp_epochs",
    "loss_prior_weight",
    "loss_huber_delta",
    "loss_holdout_targets",
    "degrade_clean",
    "degrade_local_holes",
    "degrade_pixel_missing",
    "degrade_nonzero",
    "degrade_full_missing",
    "eval_long_side",
    "oc_geometry_lr_multiplier",
)
SHARED_CONTRACT_EXTRA = ("data_protocol", "support_markers")

GEOMETRY_PROBE_NAMES = (
    "geometry.prior.head_factor.weight",
    "geometry.prior.head_factor.bias",
    "geometry.precision.out.weight",
    "geometry.precision.out.bias",
    "geometry.evidence.a_ell.0",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OC-RGA single-GPU research training entry")
    parser.add_argument("--config", required=True,
                        help="config module, e.g. local_configs.research.OC_RGA_DeLiVER")
    parser.add_argument("--model-mode", choices=ALL_MODES, default=None)
    parser.add_argument("--gpus", default=1, type=int, help="this entry is single-GPU; must be 1")
    parser.add_argument("--micro-batch", default=None, type=int)
    parser.add_argument("--accum-steps", default=None, type=int)
    parser.add_argument("--effective-batch", default=None, type=int)
    parser.add_argument("--schedule-epochs", default=None, type=int)
    parser.add_argument("--warmup-epochs", default=None, type=int)
    parser.add_argument("--stop-after-epoch", default=None, type=int,
                        help="pause point; never compresses the schedule")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--val-every", default=None, type=int)
    parser.add_argument("--val-images", default=None, type=int,
                        help="limit the periodic validation to the first N dev images (0 = all)")
    parser.add_argument("--grad-check-every", default=50, type=int,
                        help="full per-parameter gradient finiteness scan period "
                             "(AMP already guards every attempt; 0 = every attempt)")
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke-steps", default=0, type=int)
    parser.add_argument("--print-schedule", default=False, action="store_true")
    parser.add_argument("--check", default=False, action="store_true",
                        help="one real micro-batch through the whole chain on CPU or GPU, then exit")
    parser.add_argument("--leak-check", default=False, action="store_true",
                        help="prove the hold-out targets cannot reach the auxiliary conditioning")
    parser.add_argument("--device", default="cuda")
    return parser


def resolve_config(config, args):
    resolved = {
        "model_mode": args.model_mode or str(getattr(config, "model_mode", "oc_rga")),
        "micro_batch": args.micro_batch if args.micro_batch is not None else int(config.micro_batch),
        "accum_steps": args.accum_steps if args.accum_steps is not None else int(config.accum_steps),
        "effective_batch": args.effective_batch if args.effective_batch is not None else int(config.effective_batch),
        "schedule_epochs": args.schedule_epochs if args.schedule_epochs is not None else int(config.schedule_epochs),
        "warmup_epochs": args.warmup_epochs if args.warmup_epochs is not None else int(config.warm_up_epoch),
        "seed": args.seed if args.seed is not None else int(config.seed),
        "num_workers": args.num_workers if args.num_workers is not None else int(config.num_workers),
        "val_every": args.val_every if args.val_every is not None else int(getattr(config, "val_every", 10)),
        "val_images": args.val_images if args.val_images is not None else int(getattr(config, "val_images", 0) or 0),
        "stop_after_epoch": (args.stop_after_epoch if args.stop_after_epoch is not None
                             else int(getattr(config, "stop_after_epoch", 0) or 0)),
    }
    if resolved["model_mode"] not in ALL_MODES:
        raise SystemExit("model-mode must be one of %s" % (ALL_MODES,))
    if int(resolved["schedule_epochs"]) <= 0:
        raise SystemExit("schedule_epochs must be positive")
    if not (0 <= int(resolved["warmup_epochs"]) <= int(resolved["schedule_epochs"])):
        raise SystemExit("warmup_epochs must be within [0, schedule_epochs]")
    if resolved["micro_batch"] * resolved["accum_steps"] != resolved["effective_batch"]:
        raise SystemExit("micro_batch (%d) * accum_steps (%d) != effective_batch (%d)"
                         % (resolved["micro_batch"], resolved["accum_steps"], resolved["effective_batch"]))
    return resolved


def make_oc_contract(config, resolved, train_source, eval_source, plan, amp):
    contract = sched.make_contract(
        config, "original", train_source, eval_source,
        plan["micro_batch"], plan["accum_steps"], plan["effective_batch"],
        resolved["schedule_epochs"], resolved["warmup_epochs"], amp=bool(amp),
    )
    contract["model_mode"] = str(resolved["model_mode"])
    contract["data_protocol"] = "oc_rga_three_support_markers"
    contract["support_markers"] = "S=image domain, O=available observation, Vref=training reference filter"
    contract["degradation_mix"] = {
        "clean": float(getattr(config, "degrade_clean", 0.40)),
        "local_holes": float(getattr(config, "degrade_local_holes", 0.25)),
        "pixel_missing": float(getattr(config, "degrade_pixel_missing", 0.15)),
        "nonzero": float(getattr(config, "degrade_nonzero", 0.15)),
        "entire_missing": float(getattr(config, "degrade_full_missing", 0.05)),
    }
    for field in OC_CONTRACT_FIELDS:
        contract[field] = _config_value(config, field)
    contract["eval_protocol"] = "long_side_%d_pad32_single_scale_no_flip" % int(getattr(config, "eval_long_side", 640))
    return contract


def _config_value(config, field):
    value = getattr(config, field, None)
    if isinstance(value, (list, tuple)):
        return [float(v) if isinstance(v, (int, float)) else v for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def oc_contract_diff(saved, current):
    diffs = sched.contract_diff(saved, current)
    for field in OC_CONTRACT_FIELDS + SHARED_CONTRACT_EXTRA:
        if saved.get(field) != current.get(field):
            diffs.append("%s: saved=%r current=%r" % (field, saved.get(field), current.get(field)))
    return diffs


def print_schedule(config, resolved, plan, schedule, policy, train_source, eval_source):
    config.model_mode = resolved["model_mode"]
    print("[schedule] dataset=%s mode=%s" % (config.dataset_name, resolved["model_mode"]))
    print("[schedule] train_source=%s" % train_source)
    print("[schedule] eval_source=%s (periodic validation target)" % eval_source)
    print("[schedule] samples/epoch=%d micro_batch=%d accum_steps=%d effective_batch=%d"
          % (plan["num_samples"], plan["micro_batch"], plan["accum_steps"], plan["effective_batch"]))
    print("[schedule] updates/epoch=%d micro_batches/epoch=%d repeated_samples=%d"
          % (plan["updates_per_epoch"], plan["micro_batches_per_epoch"], plan["repeated_samples"]))
    print("[schedule] schedule_epochs=%d warmup_epochs=%d total_updates=%d warmup_updates=%d"
          % (schedule["schedule_epochs"], schedule["warmup_epochs"], schedule["total_updates"],
             schedule["warmup_updates"]))
    print("[schedule] 50-epoch shared observation point; stage snapshots=%s"
          % (tuple(getattr(config, "stage_epochs", ()) or ()),))
    if resolved["stop_after_epoch"]:
        print("[schedule] stop_after_epoch=%d is a pause; total_updates stays %d"
              % (resolved["stop_after_epoch"], schedule["total_updates"]))
    print("[schedule] lr base=%g power=%g geometry multiplier=%g"
          % (float(config.lr), float(config.lr_power), float(getattr(config, "oc_geometry_lr_multiplier", 2.0))))
    for row in sched.schedule_report(policy, schedule["total_updates"], schedule["warmup_updates"]):
        print("[schedule]   attempt=%7d lr=%.8e" % (row["attempt"], row["lr"]))
    print("[schedule] degradation mix=%s" % {
        "clean": getattr(config, "degrade_clean", 0.40),
        "local_holes": getattr(config, "degrade_local_holes", 0.25),
        "pixel_missing": getattr(config, "degrade_pixel_missing", 0.15),
        "nonzero": getattr(config, "degrade_nonzero", 0.15),
        "entire_missing": getattr(config, "degrade_full_missing", 0.05),
    })
    lam_g0, lam_o0 = None, None
    from research.oc_rga_losses import loss_scales

    lam_g0, lam_o0 = loss_scales(0.0, config)
    lam_g1, lam_o1 = loss_scales(float(getattr(config, "loss_lambda_ramp_epochs", 5.0)), config)
    print("[schedule] loss weights: lambda_cond %.4f -> %.4f, lambda_obs %.4f -> %.4f"
          % (lam_g0, lam_g1, lam_o0, lam_o1))
    return 0


def run_check(config, resolved, device, train_source, eval_source):
    """One real micro-batch through data -> model -> joint loss -> optimiser step."""
    from research.oc_rga_losses import compute_oc_losses

    dev = torch.device(device)
    sched.seed_everything(resolved["seed"])
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    model = build_model_for_mode(config, resolved["model_mode"], criterion=criterion).to(dev)
    params_list = build_oc_optimizer_param_groups(
        model, float(config.lr), float(config.weight_decay),
        float(getattr(config, "oc_geometry_lr_multiplier", 2.0)),
    )
    optimizer = torch.optim.AdamW(params_list, lr=float(config.lr), betas=(0.9, 0.999),
                                  weight_decay=float(config.weight_decay))
    coverage = sched.optimizer_parameter_report(model, optimizer)
    loader, plan, num_samples = ocdata.build_train_loader(
        config, train_source, eval_source, resolved["micro_batch"], resolved["accum_steps"],
        resolved["effective_batch"], 0, resolved["seed"], epoch=0,
    )
    batch = next(iter(loader))
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, bundle = model(batch["data"], batch["modal_x"], depth_support=batch["depth_support"],
                           image_support=batch["image_support"], return_geometry=True)
    losses = compute_oc_losses(model, logits, bundle, batch, 0.0)
    if losses["total"].requires_grad:
        losses["total"].backward()
    grads = sched.grad_finiteness_report(model)
    before = {name: param.detach().clone() for name, param in model.named_parameters()
              if name in GEOMETRY_PROBE_NAMES or name.endswith("oc_injection.a_beta")}
    optimizer.step()
    deltas, changed = sched.parameter_delta_report(
        {name: param for name, param in model.named_parameters() if name in before}, before)
    report = {
        "model_mode": resolved["model_mode"],
        "device": str(dev),
        "num_samples": int(num_samples),
        "batch_shapes": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
        "logits_shape": list(logits.shape),
        "losses": {k: v for k, v in losses.items() if isinstance(v, float) or v is None},
        "counts": losses["counts"],
        "geometry_stats": losses["geometry"],
        "grads_all_finite": bool(grads["all_finite"]),
        "grad_nonfinite": grads["nonfinite"][:5],
        "probe_changed": int(changed),
        "probe_total": len(before),
        "probe_deltas": deltas,
        "optimizer_coverage": coverage,
    }
    print("[check] mode=%s logits=%s" % (resolved["model_mode"], tuple(logits.shape)))
    print("[check] losses=%s" % {k: losses[k] for k in ("seg", "cond", "obs", "total")})
    print("[check] counts=%s" % losses["counts"])
    print("[check] grads_finite=%s probes changed=%d/%d" % (grads["all_finite"], changed, len(before)))
    return report


def run_leak_check(config, resolved, device, train_source, eval_source):
    """Design 15.2 item 5: the hold-out targets must vanish from every Depth statistic."""
    from research.oc_rga_losses import holdout_leak_check

    dev = torch.device(device)
    sched.seed_everything(resolved["seed"])
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    model = build_model_for_mode(config, resolved["model_mode"], criterion=criterion).to(dev)
    model.eval()
    loader, _plan, _n = ocdata.build_train_loader(
        config, train_source, eval_source, resolved["micro_batch"], resolved["accum_steps"],
        resolved["effective_batch"], 0, resolved["seed"], epoch=0,
    )
    batch = next(iter(loader))
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        _logits, bundle = model(batch["data"], batch["modal_x"], depth_support=batch["depth_support"],
                                image_support=batch["image_support"], return_geometry=True)
    report = holdout_leak_check(model, bundle, batch, config)
    report["model_mode"] = resolved["model_mode"]
    report["device"] = str(dev)
    report["degradation"] = batch.get("degradation")
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.gpus) != 1:
        raise SystemExit("this entry is single-GPU only; --gpus must be 1")

    config = deepcopy(getattr(import_module(args.config), "C"))
    resolved = resolve_config(config, args)
    config.model_mode = resolved["model_mode"]

    if args.resume:
        run_dir = os.path.dirname(os.path.abspath(str(args.resume)))
        config.log_dir = run_dir
        config.checkpoint_dir = run_dir
        config.log_file = os.path.join(run_dir, "train.log")
        config.val_log_file = os.path.join(run_dir, "validation.log")
        config.swanlab_log_dir = os.path.join(run_dir, "swanlab")

    train_source = str(config.train_dev_source)
    eval_source = str(config.dev_eval_source)
    if not os.path.exists(train_source) and getattr(config, "num_train_imgs", None) is None:
        raise SystemExit("train source not found: %s" % train_source)
    num_samples = sched.manifest_lines(train_source, fallback=getattr(config, "num_train_imgs", None))
    plan = sched.plan_epoch(num_samples, resolved["micro_batch"], resolved["accum_steps"],
                            resolved["effective_batch"])
    policy, schedule = sched.build_schedule(
        plan["updates_per_epoch"], resolved["schedule_epochs"], resolved["warmup_epochs"],
        float(config.lr), float(config.lr_power), resolved["stop_after_epoch"])

    if args.print_schedule:
        return print_schedule(config, resolved, plan, schedule, policy, train_source, eval_source)

    if args.check:
        report = run_check(config, resolved, args.device, train_source, eval_source)
        target = os.path.join(config.log_dir, "chain-check.json")
        sched.write_json(target, report)
        print("[check] wrote %s" % target)
        return 0

    if args.leak_check:
        report = run_leak_check(config, resolved, args.device, train_source, eval_source)
        target = os.path.join(config.log_dir, "holdout-leak-check.json")
        sched.write_json(target, report)
        print("[leak-check] %s; wrote %s" % ("ALL PASS" if report["passed"] else "FAILURES", target))
        return 0 if report["passed"] else 1

    if str(args.device) != "cuda":
        print("REFUSED: training supports --device cuda only; use --print-schedule / --check / --help "
              "for CPU-only checks.", file=sys.stderr)
        return 3
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device available.", file=sys.stderr)
        return 3

    return train(config, resolved, plan, schedule, policy, train_source, eval_source, args)


def train(config, resolved, plan, schedule, policy, train_source, eval_source, args) -> int:
    from research.oc_rga_losses import compute_oc_losses, geometry_statistics
    from research import evaluate_oc_rga as oceval
    from utils.engine.logger import get_logger

    os.makedirs(config.log_dir, exist_ok=True)
    logger = get_logger(config.log_dir, config.log_file)
    device = torch.device(args.device)

    logger.info("model_mode=%s dataset=%s" % (resolved["model_mode"], config.dataset_name))
    logger.info("train_source=%s eval_source=%s" % (train_source, eval_source))
    logger.info("batching: micro_batch=%d accum_steps=%d effective_batch=%d updates/epoch=%d"
                % (plan["micro_batch"], plan["accum_steps"], plan["effective_batch"],
                   plan["updates_per_epoch"]))
    logger.info("schedule: epochs=%d warmup=%d total_updates=%d"
                % (schedule["schedule_epochs"], schedule["warmup_epochs"], schedule["total_updates"]))

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    # Both arms are built from the same seed and the training RNG is reset
    # afterwards, so the extra geometry heads cannot shift the baseline's
    # shuffle or augmentation stream.
    sched.seed_everything(resolved["seed"])
    model = build_model_for_mode(config, resolved["model_mode"], criterion=criterion)
    sched.seed_everything(resolved["seed"])
    model.to(device)

    params_list = build_oc_optimizer_param_groups(
        model, float(config.lr), float(config.weight_decay),
        float(getattr(config, "oc_geometry_lr_multiplier", 2.0)),
    )
    optimizer = torch.optim.AdamW(params_list, lr=float(config.lr), betas=(0.9, 0.999),
                                  weight_decay=float(config.weight_decay),
                                  eps=1e-8)
    coverage = sched.optimizer_parameter_report(model, optimizer)
    logger.info("optimizer coverage: %s" % coverage)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    contract = make_oc_contract(config, resolved, train_source, eval_source, plan, args.amp)
    sched.write_json(os.path.join(config.log_dir, "resolved_config.json"), config)
    sched.write_json(os.path.join(config.log_dir, "args.json"), vars(args))
    sched.write_json(os.path.join(config.log_dir, "contract.json"), contract)

    train_loader, plan, num_samples = ocdata.build_train_loader(
        config, train_source, eval_source, plan["micro_batch"], plan["accum_steps"],
        plan["effective_batch"], resolved["num_workers"], resolved["seed"], epoch=0,
    )
    val_loader = None
    val_len = 0
    if resolved["val_every"] > 0:
        val_loader, val_len = ocdata.build_eval_loader(config, eval_source, train_source,
                                                       num_workers=int(resolved["num_workers"]))
        logger.info("dev validation set: %d images (shared 640 single-scale protocol)" % val_len)

    state = {
        "completed_epochs": 0,
        "attempt_updates": 0,
        "applied_updates": 0,
        "amp_skipped_updates": 0,
        "micro_batches_seen": 0,
        "best_dev_miou": 0.0,
        "model_mode": resolved["model_mode"],
        "num_train_samples": int(num_samples),
        "optimizer_param_report": coverage,
    }
    if args.resume:
        payload = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise SystemExit("not an OC-RGA checkpoint: %s" % args.resume)
        if payload.get("snapshot_kind") not in sched.RESUMABLE_SNAPSHOTS or payload.get("epoch_boundary") is False:
            raise SystemExit("refusing to resume from a non-epoch-boundary snapshot (%r)"
                             % payload.get("snapshot_kind"))
        diffs = oc_contract_diff(payload.get("contract", {}), contract)
        if diffs:
            raise SystemExit("resume contract mismatch (refusing to continue):\n  " + "\n  ".join(diffs))
        model.load_state_dict(payload["model"], strict=True)
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
            moved = sched.move_optimizer_state_to_device(optimizer)
            sched.assert_optimizer_state_on_params(optimizer)
            if moved:
                logger.info("moved %d optimizer state tensors onto the parameter device" % moved)
        if scaler is not None and payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        sched.restore_rng(payload.get("rng"))
        for key in ("completed_epochs", "attempt_updates", "applied_updates",
                    "amp_skipped_updates", "micro_batches_seen"):
            state[key] = int(payload.get(key, state[key]))
        state["best_dev_miou"] = float(payload.get("best_dev_miou", 0.0))
        logger.info("resumed %s at completed_epochs=%d attempt_updates=%d (no re-warmup)"
                    % (args.resume, state["completed_epochs"], state["attempt_updates"]))

    loss_csv = sched.CsvWriter(os.path.join(config.log_dir, "loss_lr.csv"), [
        "epoch", "attempt_update", "applied_update", "amp_skipped", "lr", "param_group_lrs",
        "seg", "cond", "cond_post", "cond_prior", "cond_huber", "obs", "lam_cond", "lam_obs",
        "total", "images_with_targets", "targets", "images_with_obs", "obs_cells",
        "group_lrs", "epoch_seconds",
    ])
    val_csv = sched.CsvWriter(os.path.join(config.log_dir, "validation.csv"), [
        "epoch", "condition", "miou", "macc", "mf1", "num_samples", "realised_ratio",
        "ham_seed", "corruption_seed", "seconds",
    ])
    stats_csv = sched.CsvWriter(os.path.join(config.log_dir, "geometry_stats.csv"),
                                ["epoch", "attempt_update", "json"])

    tracking_run = None
    if bool(getattr(config, "swanlab_enabled", False)) and not args.smoke_steps:
        try:
            from research.tracking import start_tracking
            tracking_run = start_tracking(config, args, _TrackingEngine())
        except Exception as exc:  # SwanLab is optional; never block training
            logger.warning("SwanLab tracking disabled: %s" % exc)
    elif args.smoke_steps:
        logger.info("SwanLab tracking disabled for the smoke run (diagnostic, not an experiment)")

    probes = sched.select_probe_params(model)
    for name in GEOMETRY_PROBE_NAMES:
        if name in dict(model.named_parameters()) and name not in probes:
            probes[name] = dict(model.named_parameters())[name]
    probe_before = {name: param.detach().clone() for name, param in probes.items()}

    failure = None
    smoke_start = time.time()
    smoke_samples = 0
    smoke_stats = {"losses_finite": True, "grads_finite": True, "grad_checks": 0,
                   "amp_skipped_attempts": 0, "nonfinite_grad_params": [],
                   "no_graph_micro_batches": 0, "first_loss_requires_grad": None}

    try:
        for epoch in range(state["completed_epochs"], resolved["schedule_epochs"]):
            model.train()
            train_loader.sampler.set_epoch(epoch)
            epoch_start = time.time()
            micro_loss_sum = 0.0
            epoch_updates = 0
            last = {}
            last_group_lrs = []
            optimizer.zero_grad(set_to_none=True)
            loader_iter = iter(train_loader)
            micro_in_group = 0
            stopped = False
            total_micro = plan["micro_batches_per_epoch"]

            for step in range(total_micro):
                batch = next(loader_iter)
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                epoch_progress = float(epoch) + float(step + 1) / float(total_micro)
                autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                            if args.amp else contextlib.nullcontext())
                with autocast:
                    logits, bundle = model(batch["data"], batch["modal_x"],
                                           depth_support=batch["depth_support"],
                                           image_support=batch["image_support"],
                                           return_geometry=True)
                losses = compute_oc_losses(model, logits, bundle, batch, epoch_progress)
                total = losses["total"]
                if smoke_stats["first_loss_requires_grad"] is None:
                    smoke_stats["first_loss_requires_grad"] = bool(total.requires_grad)
                    logger.info("first micro-batch: total.requires_grad=%s seg=%s cond=%s obs=%s counts=%s"
                                % (total.requires_grad, losses["seg"], losses["cond"],
                                   losses["obs"], losses["counts"]))
                if not bool(torch.isfinite(total).all()):
                    raise TrainingFailure("nonfinite_loss",
                                          "non-finite loss at epoch %d micro-batch %d"
                                          % (epoch + 1, step + 1), epoch=epoch + 1)
                if not total.requires_grad:
                    # An all-ignore crop (or a batch whose geometry branch is
                    # legally unused) contributes a constant.  It must never reach
                    # the GradScaler: an unscaled step would assert.  The whole
                    # accumulation group is dropped in that case.
                    micro_loss_sum += float(total.detach())
                    micro_in_group += 1
                    state["micro_batches_seen"] += 1
                    smoke_samples += int(batch["data"].shape[0])
                    last = losses
                    smoke_stats["no_graph_micro_batches"] += 1
                    if micro_in_group < plan["accum_steps"]:
                        continue
                    micro_loss_sum = 0.0
                    micro_in_group = 0
                    continue
                scaled_loss = total / plan["accum_steps"]
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                micro_loss_sum += float(total.detach())
                micro_in_group += 1
                state["micro_batches_seen"] += 1
                smoke_samples += int(batch["data"].shape[0])
                last = losses

                if micro_in_group < plan["accum_steps"]:
                    continue

                lr = sched.lr_for_attempt(policy, state["attempt_updates"], schedule["total_updates"])
                group_lrs = sched.set_param_group_lr(optimizer, lr)

                # The per-parameter finiteness scan is the known fixed overhead of
                # the previous entries.  With AMP the scaler already guards every
                # attempt against inf/nan; the explicit per-parameter report runs
                # on the first attempt and then periodically.  Without AMP there
                # is no such guard, so it runs every attempt.
                scan_now = (not args.amp) or int(args.grad_check_every) <= 0 or (
                    state["attempt_updates"] % max(1, int(args.grad_check_every)) == 0)
                all_params = [p for group in optimizer.param_groups for p in group["params"]]

                if scaler is not None:
                    scaler.unscale_(optimizer)
                if scan_now:
                    grad_report = sched.grad_finiteness_report(model)
                    smoke_stats["grad_checks"] += 1
                    if not grad_report["all_finite"]:
                        smoke_stats["grads_finite"] = False
                        smoke_stats["nonfinite_grad_params"] = list(grad_report["nonfinite"][:20])
                        if scaler is None:
                            raise TrainingFailure("nonfinite_grad",
                                                  "non-finite gradient without AMP at attempt %d"
                                                  % state["attempt_updates"], epoch=epoch + 1)

                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                if scaler is not None:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    applied = scaler.get_scale() >= scale_before
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    applied = True

                if not applied:
                    smoke_stats["amp_skipped_attempts"] += 1
                state["attempt_updates"] += 1
                state["applied_updates"] += int(applied)
                state["amp_skipped_updates"] += int(not applied)
                epoch_updates += 1
                last_group_lrs = group_lrs
                counts = last.get("counts", {})
                loss_csv.write([
                    epoch + 1, state["attempt_updates"], state["applied_updates"],
                    state["amp_skipped_updates"], lr, last_group_lrs[0] if last_group_lrs else lr,
                    last.get("seg"), last.get("cond"), last.get("cond_post"), last.get("cond_prior"),
                    last.get("cond_huber"), last.get("obs"), last.get("lam_cond"), last.get("lam_obs"),
                    float(last["total"].detach()) if last else 0.0,
                    counts.get("images_with_targets", 0), counts.get("targets", 0),
                    counts.get("images_with_obs", 0), counts.get("obs_cells", 0),
                    ";".join("%.3e" % v for v in last_group_lrs),
                    "%.3f" % (time.time() - epoch_start),
                ])
                micro_loss_sum = 0.0
                micro_in_group = 0

                if args.smoke_steps and state["attempt_updates"] >= int(args.smoke_steps):
                    stopped = True
                    break

            state["completed_epochs"] = epoch if stopped else epoch + 1
            if last:
                stats_csv.write([state["completed_epochs"], state["attempt_updates"],
                                 sched.jsonable(last.get("geometry", {}))])

            # ---- periodic validation (clean, shared 640 protocol) ---------- #
            val_miou = None
            if (val_loader is not None and not args.smoke_steps
                    and (state["completed_epochs"] % resolved["val_every"] == 0
                         or state["completed_epochs"] == resolved["schedule_epochs"])):
                conditions = ["clean"]
                if state["completed_epochs"] == 40:
                    conditions = ["clean"] + list(getattr(config, "robust_conditions_epoch40", ()))
                if state["completed_epochs"] == 50:
                    conditions = list(getattr(config, "robust_conditions_epoch50", ("clean",)))
                results = _run_validation(model, val_loader, config, device, args, resolved, conditions,
                                          state["completed_epochs"], logger, val_csv)
                if "clean" in results:
                    val_miou = float(results["clean"]["miou"])
                    if val_miou > state["best_dev_miou"]:
                        state["best_dev_miou"] = val_miou
                        _save(config, model, optimizer, scaler, state, args, contract,
                              "best-dev.pth", "best")
                logger.info("validation done: %s"
                            % {k: v["miou"] for k, v in results.items()})

            if tracking_run is not None:
                try:
                    from research.tracking import log_epoch
                    log_epoch(tracking_run, state["completed_epochs"], state["attempt_updates"],
                              float(last["total"].detach()) if last else 0.0,
                              last_group_lrs[0] if last_group_lrs else float(config.lr), val_miou)
                except Exception:
                    pass
            logger.info("epoch %d/%d done: attempts=%d applied=%d amp_skipped=%d seg=%s cond=%s obs=%s "
                        "val_miou=%s elapsed=%.1fs"
                        % (state["completed_epochs"], resolved["schedule_epochs"],
                           state["attempt_updates"], state["applied_updates"], state["amp_skipped_updates"],
                           None if not last else round(last.get("seg", 0.0), 5),
                           None if not last else round(last.get("cond", 0.0), 6),
                           None if not last else round(last.get("obs", 0.0), 6),
                           ("%.2f" % val_miou) if val_miou is not None else "n/a",
                           time.time() - epoch_start))

            if not args.smoke_steps:
                _save(config, model, optimizer, scaler, state, args, contract, "last.pth", "epoch")
                for stage in tuple(getattr(config, "stage_epochs", ()) or ()):
                    if state["completed_epochs"] == int(stage):
                        _save(config, model, optimizer, scaler, state, args, contract,
                              "stage-epoch-%d.pth" % int(stage), "stage")
                        logger.info("stage snapshot written at epoch %d" % int(stage))

            if args.smoke_steps:
                break
            if resolved["stop_after_epoch"] and state["completed_epochs"] >= resolved["stop_after_epoch"]:
                logger.info("stage pause after %d epochs; the %d-epoch schedule was not compressed"
                            % (state["completed_epochs"], resolved["schedule_epochs"]))
                break

        if args.smoke_steps:
            elapsed = max(time.time() - smoke_start, sched.EPS)
            peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
            deltas, changed = sched.parameter_delta_report(probes, probe_before)
            report = {
                "smoke_steps": int(args.smoke_steps),
                "model_mode": resolved["model_mode"],
                "attempt_updates": state["attempt_updates"],
                "applied_updates": state["applied_updates"],
                "amp_skipped_updates": state["amp_skipped_updates"],
                "micro_batches_seen": state["micro_batches_seen"],
                "samples": int(smoke_samples),
                "seconds": round(elapsed, 3),
                "samples_per_second": round(smoke_samples / elapsed, 3),
                "peak_memory_mb": round(peak_mb, 1),
                "grad_checks": int(smoke_stats["grad_checks"]),
                "no_graph_micro_batches": int(smoke_stats["no_graph_micro_batches"]),
                "first_loss_requires_grad": smoke_stats["first_loss_requires_grad"],
                "grad_nonfinite_params": list(smoke_stats["nonfinite_grad_params"]),
                "probe_count": len(deltas),
                "probe_changed": int(changed),
                "probe_deltas": deltas,
                "last_losses": {k: v for k, v in (last or {}).items() if isinstance(v, float) or v is None},
                "last_geometry": (last or {}).get("geometry", {}),
            }
            sched.write_json(os.path.join(config.log_dir, "smoke-report.json"), report)
            _save(config, model, optimizer, scaler, state, args, contract, "smoke.pth", "smoke")
            logger.info("smoke report: %s" % report)

        logger.info("training entry finished: completed_epochs=%d attempt_updates=%d"
                    % (state["completed_epochs"], state["attempt_updates"]))
    except TrainingFailure as exc:
        failure = exc
        logger.error("training failed (%s): %s" % (exc.kind, exc))
        sched.write_json(os.path.join(config.log_dir, "failure.json"), {
            "failed": True, "kind": exc.kind, "detail": str(exc), "epoch": exc.epoch,
            "completed_epochs": int(state["completed_epochs"]),
            "attempt_updates": int(state["attempt_updates"]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    finally:
        if tracking_run is not None:
            try:
                tracking_run.finish()
            except Exception:
                pass

    return 4 if failure is not None else 0


def _run_validation(model, val_loader, config, device, args, resolved, conditions, epoch, logger, val_csv):
    from research import evaluate_oc_rga as oceval

    limit = int(resolved.get("val_images", 0) or 0)
    max_batches = limit if limit > 0 else 0
    batch_count = min(limit, len(val_loader)) if limit > 0 else len(val_loader)
    bn_before = sched.batchnorm_running_state(model)
    results = oceval.evaluate_conditions(
        model, val_loader, conditions, config, device, amp=bool(args.amp),
        ham_seed=int(getattr(config, "eval_ham_seed", 12345)),
        corruption_seed=int(getattr(config, "eval_corruption_seed", 12345)),
        max_batches=max_batches, log_fn=logger.info,
        progress_every=max(1, batch_count // 8),
    )
    bn_after = sched.batchnorm_running_state(model)
    if bn_after != bn_before:
        raise RuntimeError("validation changed BatchNorm running statistics (it must run in eval mode): "
                           "%s -> %s" % (bn_before, bn_after))
    for name, result in results.items():
        val_csv.write([int(epoch), name, result["miou"], result["macc"], result["mf1"],
                       result["num_samples"], round(result["realised_ratio"], 6),
                       int(getattr(config, "eval_ham_seed", 12345)),
                       int(getattr(config, "eval_corruption_seed", 12345)), result["seconds"]])
        sched.write_json(os.path.join(config.log_dir, "val_per_class", "epoch-%03d-%s.json" % (int(epoch), name)), {
            "epoch": int(epoch), "condition": name, "split": "dev",
            "model_mode": sched.VALIDATION_MODE,
            "protocol": "long_side_%d_pad32_single_scale_no_flip" % int(getattr(config, "eval_long_side", 640)),
            "batchnorm_running_state": bn_after,
            "miou": result["miou"], "macc": result["macc"], "mf1": result["mf1"],
            "num_samples": result["num_samples"], "class_names": result["class_names"],
            "iou": result["iou"], "acc": result["acc"], "f1": result["f1"],
            "confusion": result["confusion"],
            "realised_ratio": result["realised_ratio"],
        })
    return results


def _save(config, model, optimizer, scaler, state, args, contract, name, kind):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "snapshot_kind": kind,
        "epoch_boundary": kind in sched.RESUMABLE_SNAPSHOTS,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "completed_epochs": int(state["completed_epochs"]),
        "attempt_updates": int(state["attempt_updates"]),
        "applied_updates": int(state["applied_updates"]),
        "amp_skipped_updates": int(state["amp_skipped_updates"]),
        "micro_batches_seen": int(state["micro_batches_seen"]),
        "best_dev_miou": float(state["best_dev_miou"]),
        "model_mode": str(state["model_mode"]),
        "num_train_samples": int(state["num_train_samples"]),
        "contract": contract,
        "config": sched.jsonable(config),
        "args": sched.jsonable(vars(args)),
        "rng": sched.capture_rng(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(config.log_dir, name)
    torch.save(payload, path)
    return path


class TrainingFailure(RuntimeError):
    def __init__(self, kind, detail, epoch=None):
        super().__init__(detail)
        self.kind = str(kind)
        self.epoch = epoch


class _TrackingEngine:
    distributed = False
    local_rank = 0
    world_size = 1


if __name__ == "__main__":
    sys.exit(main())
