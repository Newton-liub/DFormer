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
import json
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
# Every knob that changes the *effective* augmentation, degradation or hold-out
# protocol is listed here, so editing one and resuming is refused instead of
# silently mixing two protocols (audit R5).
OC_CONTRACT_FIELDS = (
    "model_mode",
    "oc_geometry_cell",
    "oc_rank",
    "num_heads_per_stage",
    "oc_tau",
    "oc_sigma_ref",
    "oc_beta_init",
    "oc_ell_min",
    "oc_ell_max",
    "loss_lambda_cond_start",
    "loss_lambda_cond_end",
    "loss_lambda_obs_start",
    "loss_lambda_obs_end",
    "loss_lambda_ramp_epochs",
    "loss_prior_weight",
    "loss_huber_delta",
    "loss_holdout_targets",
    "loss_holdout_sparse_prob",
    "loss_holdout_sparse_low",
    "loss_holdout_sparse_high",
    "degrade_clean",
    "degrade_local_holes",
    "degrade_pixel_missing",
    "degrade_nonzero",
    "degrade_full_missing",
    "degrade_hole_ratio_low",
    "degrade_hole_ratio_high",
    "degrade_local_window_low",
    "degrade_local_window_high",
    "degrade_noise_std_low",
    "degrade_noise_std_high",
    "degrade_shift_area_low",
    "degrade_shift_area_high",
    "degrade_shift_offset_low",
    "degrade_shift_offset_high",
    "degrade_shift_outlier_rate",
    "eval_long_side",
    "oc_geometry_lr_multiplier",
)

# A resume must carry a complete training state; a payload without the formal
# optimizer/scaler/RNG is not a resumable artifact.
RESUME_REQUIRED_FIELDS = ("model", "optimizer", "rng", "contract", "config", "args")
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
                        help="one real accumulation group through the whole chain, then exit "
                             "(non-zero exit status on any failed assertion)")
    parser.add_argument("--check-size", default=0, type=int,
                        help="with --check: use a synthetic batch of this size instead of a "
                             "dataset sample (CPU-runnable small-grid check)")
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


def _synthetic_batch(config, size: int, device, seed: int):
    """Small synthetic batch for a CPU-only chain check (no dataset access)."""
    from research.oc_rga import Z_MAX, Z_MIN, ZERO_FILL

    generator = torch.Generator().manual_seed(int(seed))
    classes = int(config.num_classes)
    rgb = torch.randn(1, 3, size, size, generator=generator)
    image_support = torch.ones(1, 1, size, size)
    observed_mask = (torch.rand(1, 1, size, size, generator=generator) > 0.3).float()
    depth = torch.rand(1, 1, size, size, generator=generator) * (Z_MAX - Z_MIN) + Z_MIN
    modal_x = torch.where(observed_mask > 0, depth, torch.full_like(depth, float(ZERO_FILL)))
    reference_valid = (torch.rand(1, 1, size, size, generator=generator) > 0.05).float() * image_support
    label = torch.randint(0, classes, (1, size, size), generator=generator)
    label[..., : max(1, size // 8), :] = int(config.background)
    return {
        "data": rgb.to(device),
        "modal_x": modal_x.repeat(1, 3, 1, 1).to(device),
        "depth_support": observed_mask.to(device),
        "image_support": image_support.to(device),
        "reference_depth": depth.to(device),
        "reference_valid": reference_valid.to(device),
        "label": label.to(device),
        "fn": "synthetic",
        # Metadata stays tensor-shaped so it matches what the collated loader
        # batch provides.
        "sample_id": torch.zeros(1, dtype=torch.long, device=device),
        "epoch": torch.zeros(1, dtype=torch.long, device=device),
        "repeat_index": torch.zeros(1, dtype=torch.long, device=device),
    }


def run_check(config, resolved, device, train_source, eval_source, amp=True, check_size=0):
    """One real accumulation group through the whole chain, with hard assertions.

    The check mirrors the training step exactly -- FP16 autocast, ``accum_steps``
    micro-batches, GradScaler, unscale, global-norm clip 1.0 and a real optimiser
    step -- and asserts the evidence the audit asks for: finite gradients, a
    **non-zero** gradient on every key new parameter, and a parameter change after
    a step performed with weight decay temporarily disabled (so weight decay alone
    cannot be mistaken for a mechanism gradient).  Any failure raises, and
    ``main --check`` returns a non-zero exit status.
    """
    from research.oc_rga import ZERO_FILL
    from research.oc_rga_losses import compute_oc_losses

    dev = torch.device(device)
    amp = bool(amp) and dev.type == "cuda"
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
    scaler = torch.cuda.amp.GradScaler() if amp else None

    source = "synthetic"
    if int(check_size) > 0:
        batch = _synthetic_batch(config, int(check_size), dev, resolved["seed"])
        num_samples = 1
        plan = sched.plan_epoch(1, 1, int(resolved["accum_steps"]), int(resolved["accum_steps"]))
    else:
        loader, plan, num_samples = ocdata.build_train_loader(
            config, train_source, eval_source, resolved["micro_batch"], resolved["accum_steps"],
            resolved["effective_batch"], 0, resolved["seed"], epoch=0,
        )
        batch = next(iter(loader))
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        source = "dataset"

    model.train()
    accum = int(resolved["accum_steps"])
    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    def next_batch(attempt: int):
        if int(check_size) > 0:
            return _synthetic_batch(config, int(check_size), dev, resolved["seed"] + attempt)
        return batch

    # FP16 AMP legitimately overflows on the first attempts while GradScaler walks
    # the scale down; the check therefore mirrors the training loop and retries,
    # and only demands finite gradients plus a real update on an attempt where the
    # scaler actually applies.
    attempts = 0
    amp_skipped = 0
    last = None
    logits = None
    max_attempts = 16 if amp else 1
    while attempts < max_attempts:
        attempts += 1
        current = next_batch(attempts)
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(accum):
            autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                        if amp else contextlib.nullcontext())
            with autocast:
                logits, bundle = model(current["data"], current["modal_x"],
                                       depth_support=current["depth_support"],
                                       image_support=current["image_support"], return_geometry=True)
            losses = compute_oc_losses(model, logits, bundle, current, 0.0)
            total = losses["total"]
            if not bool(torch.isfinite(total).all()):
                raise SystemExit("[check] FAIL: non-finite loss %r" % float(total.detach()))
            if not total.requires_grad:
                raise SystemExit("[check] FAIL: the joint loss carries no graph")
            scaled = total / accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            last = losses

        if scaler is not None:
            scaler.unscale_(optimizer)
        grads = sched.grad_finiteness_report(model)
        if scaler is not None and not grads["all_finite"]:
            scaler.step(optimizer)   # the scaler detects the infs and skips
            scaler.update()          # and walks the scale down
            amp_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        if not grads["all_finite"]:
            raise SystemExit("[check] FAIL: non-finite gradients without AMP %s"
                             % (grads["nonfinite"][:5],))

        probes = {name: param for name, param in model.named_parameters()
                  if name in GEOMETRY_PROBE_NAMES or name.endswith("oc_injection.a_beta")}
        before = {name: param.detach().clone() for name, param in probes.items()}
        zero_grad = [name for name, param in probes.items()
                     if param.grad is None or float(param.grad.abs().max()) == 0.0]
        if zero_grad:
            raise SystemExit("[check] FAIL: these key parameters have a zero/absent gradient: %s"
                             % (zero_grad,))

        saved_wd = [group.get("weight_decay", 0.0) for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["weight_decay"] = 0.0
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        for group, wd in zip(optimizer.param_groups, saved_wd):
            group["weight_decay"] = wd

        deltas, changed = sched.parameter_delta_report(probes, before)
        unmoved = sorted(name for name, row in deltas.items() if not row["changed"])
        if unmoved:
            raise SystemExit("[check] FAIL: key parameters did not update with weight decay "
                             "disabled: %s" % (unmoved,))
        break
    else:
        raise SystemExit("[check] FAIL: no attempt produced finite gradients after %d tries "
                         "(AMP skipped %d)" % (attempts, amp_skipped))

    geometry = dict(last.get("geometry") or {})
    if "raw_E_sample" in geometry:
        geometry["raw_E_sample"] = geometry["raw_E_sample"].tolist()[:8]
    report = {
        "model_mode": resolved["model_mode"],
        "device": str(dev),
        "amp": bool(amp),
        "batch_source": source,
        "accum_steps": accum,
        "attempts": int(attempts),
        "amp_skipped_attempts": int(amp_skipped),
        "num_samples": int(num_samples),
        "batch_shapes": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
        "logits_shape": list(logits.shape),
        "losses": {k: v for k, v in last.items() if isinstance(v, float) or v is None},
        "counts": last["counts"],
        "geometry_sample": geometry,
        "grads_all_finite": bool(grads["all_finite"]),
        "grad_nonfinite": grads["nonfinite"][:5],
        "key_parameter_grad_max": {name: float(param.grad.abs().max()) for name, param in probes.items()},
        "probe_changed_without_weight_decay": int(changed),
        "probe_total": len(before),
        "probe_deltas": deltas,
        "optimizer_coverage": coverage,
        "weight_decay_disabled_for_probe_step": True,
    }
    print("[check] mode=%s source=%s amp=%s logits=%s" % (resolved["model_mode"], source, amp,
                                                          tuple(logits.shape)))
    print("[check] group losses=%s" % {k: last[k] for k in ("seg", "cond", "obs", "total")})
    print("[check] counts=%s" % last["counts"])
    print("[check] grads finite=%s; key grads max=%s"
          % (grads["all_finite"],
             {n.split(".")[-2] + "." + n.split(".")[-1]: round(v, 6)
              for n, v in report["key_parameter_grad_max"].items()}))
    print("[check] updated %d/%d key parameters with weight decay disabled (attempts=%d, AMP skipped=%d)"
          % (changed, len(before), attempts, amp_skipped))
    report["passed"] = True
    return report


def run_leak_check(config, resolved, device, train_source, eval_source, check_size=0):
    """Design 15.2 item 5: the hold-out targets must vanish from every Depth statistic."""
    from research.oc_rga_losses import holdout_leak_check

    dev = torch.device(device)
    sched.seed_everything(resolved["seed"])
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    model = build_model_for_mode(config, resolved["model_mode"], criterion=criterion).to(dev)
    model.eval()
    source = "synthetic"
    if int(check_size) > 0:
        batch = _synthetic_batch(config, int(check_size), dev, resolved["seed"])
    else:
        loader, _plan, _n = ocdata.build_train_loader(
            config, train_source, eval_source, resolved["micro_batch"], resolved["accum_steps"],
            resolved["effective_batch"], 0, resolved["seed"], epoch=0,
        )
        batch = next(iter(loader))
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        source = "dataset"
    with torch.no_grad():
        _logits, bundle = model(batch["data"], batch["modal_x"], depth_support=batch["depth_support"],
                                image_support=batch["image_support"], return_geometry=True)
    report = holdout_leak_check(model, bundle, batch, config)
    report["model_mode"] = resolved["model_mode"]
    report["device"] = str(dev)
    report["batch_source"] = source
    report["degradation"] = batch.get("degradation")
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.gpus) != 1:
        raise SystemExit("this entry is single-GPU only; --gpus must be 1")

    config = deepcopy(getattr(import_module(args.config), "C"))
    resolved = resolve_config(config, args)
    config.model_mode = resolved["model_mode"]
    # Write the resolved seed back once: the spatial/ degradation/ hold-out streams
    # and the contract all read ``config.seed``, so a non-default ``--seed`` must
    # not produce a mixed protocol where part of the pipeline still uses the
    # config value (audit R5).
    config.seed = int(resolved["seed"])
    config.num_workers = int(resolved["num_workers"])

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
        try:
            report = run_check(config, resolved, args.device, train_source, eval_source,
                               amp=bool(args.amp), check_size=int(args.check_size))
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 1
        target = os.path.join(config.log_dir, "chain-check.json")
        sched.write_json(target, report)
        print("[check] PASS; wrote %s" % target)
        return 0

    if args.leak_check:
        report = run_leak_check(config, resolved, args.device, train_source, eval_source,
                                check_size=int(args.check_size))
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
    from research.oc_rga_losses import compute_oc_losses, geometry_statistics, merge_evidence_stats
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
        missing_state = [k for k in RESUME_REQUIRED_FIELDS if not payload.get(k)]
        if missing_state:
            raise SystemExit("refusing to resume: the snapshot lacks the formal training state %s "
                             "(audit R5)" % missing_state)
        if args.amp and not payload.get("scaler"):
            raise SystemExit("refusing to resume with AMP: the snapshot has no GradScaler state")
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
    stats_csv = sched.CsvWriter(os.path.join(config.log_dir, "geometry_stats.csv"), [
        "epoch", "attempt_update", "saturation_rate", "raw_E_p50", "m_fraction",
        "cholesky_fallback", "omega_quantiles", "s_quantiles", "ell_quantiles", "beta_quantiles",
    ])

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
            epoch_updates = 0
            last = {}
            last_group_lrs = []
            optimizer.zero_grad(set_to_none=True)
            loader_iter = iter(train_loader)
            micro_in_group = 0
            stopped = False
            total_micro = plan["micro_batches_per_epoch"]
            # One update == one accumulation group: the logged loss components are
            # the mean over the group's micro-batches and the logged counts are the
            # group's totals, so the curve really is an effective-batch mean rather
            # than the last micro-batch only (audit R6.5).
            group_sum: Dict[str, float] = {}
            group_n: Dict[str, int] = {}
            group_counts: Dict[str, int] = {}
            epoch_evidence = {"samples": [], "saturated": 0.0, "entries": 0.0, "threshold": 4.0}

            def accumulate(entry) -> None:
                for key in ("seg", "cond", "cond_post", "cond_prior", "cond_huber", "obs", "total"):
                    value = entry.get(key)
                    if value is None:
                        continue
                    group_sum[key] = group_sum.get(key, 0.0) + float(value)
                    group_n[key] = group_n.get(key, 0) + 1
                for key, value in (entry.get("counts") or {}).items():
                    group_counts[key] = group_counts.get(key, 0) + int(value)
                sample = entry.get("geometry") or {}
                if "raw_E_sample" in sample:
                    epoch_evidence["samples"].append(sample["raw_E_sample"])
                    epoch_evidence["saturated"] += float(sample.get("saturated", 0.0))
                    epoch_evidence["entries"] += float(sample.get("entries", 0.0))
                    epoch_evidence["threshold"] = float(sample.get("threshold", 4.0))

            def group_mean(key: str):
                count = group_n.get(key, 0)
                return (group_sum.get(key, 0.0) / count) if count else None

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
                    accumulate(losses)
                    micro_in_group += 1
                    state["micro_batches_seen"] += 1
                    smoke_samples += int(batch["data"].shape[0])
                    last = losses
                    smoke_stats["no_graph_micro_batches"] += 1
                    if micro_in_group < plan["accum_steps"]:
                        continue
                    group_sum.clear()
                    group_n.clear()
                    group_counts.clear()
                    micro_in_group = 0
                    continue
                scaled_loss = total / plan["accum_steps"]
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                accumulate(losses)
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
                loss_csv.write([
                    epoch + 1, state["attempt_updates"], state["applied_updates"],
                    state["amp_skipped_updates"], lr, last_group_lrs[0] if last_group_lrs else lr,
                    group_mean("seg"), group_mean("cond"), group_mean("cond_post"),
                    group_mean("cond_prior"), group_mean("cond_huber"), group_mean("obs"),
                    (last or {}).get("lam_cond"), (last or {}).get("lam_obs"),
                    group_mean("total"),
                    group_counts.get("images_with_targets", 0), group_counts.get("targets", 0),
                    group_counts.get("images_with_obs", 0), group_counts.get("obs_cells", 0),
                    ";".join("%.3e" % v for v in last_group_lrs),
                    "%.3f" % (time.time() - epoch_start),
                ])
                group_sum.clear()
                group_n.clear()
                group_counts.clear()
                micro_in_group = 0

                if args.smoke_steps and state["attempt_updates"] >= int(args.smoke_steps):
                    stopped = True
                    break

            state["completed_epochs"] = epoch if stopped else epoch + 1
            if last:
                # One epoch-level geometry summary: the quantiles come from a
                # bounded sample of the last bundle, while the raw-E and saturation
                # figures accumulate the small detached samples of every
                # micro-batch (audit R6.3/R6.4).
                geometry = geometry_statistics(bundle, model)
                geometry.update(merge_evidence_stats({
                    "raw_E_sample": (torch.cat(epoch_evidence["samples"])
                                     if epoch_evidence["samples"] else None),
                    "saturated": epoch_evidence["saturated"],
                    "entries": epoch_evidence["entries"],
                    "threshold": epoch_evidence["threshold"],
                }))
                geometry["epoch"] = state["completed_epochs"]
                geometry["attempt_updates"] = state["attempt_updates"]
                geometry["micro_batches_seen"] = state["micro_batches_seen"]
                sched.write_json(os.path.join(config.log_dir, "geometry_stats",
                                              "epoch-%03d.json" % state["completed_epochs"]), geometry)
                raw_q = geometry.get("raw_E_quantiles") or []
                stats_csv.write([
                    state["completed_epochs"], state["attempt_updates"],
                    geometry.get("saturation", {}).get("rate"),
                    raw_q[1] if len(raw_q) > 1 else None,
                    geometry.get("m_fraction"), geometry.get("cholesky_fallback"),
                    ";".join("%.4f" % v for v in (geometry.get("omega_quantiles") or [])),
                    ";".join("%.4f" % v for v in (geometry.get("s_quantiles") or [])),
                    json.dumps(sched.jsonable(geometry.get("ell_quantiles") or [])),
                    json.dumps(sched.jsonable(geometry.get("beta_quantiles") or [])),
                ])
                epoch_evidence["samples"] = []

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
