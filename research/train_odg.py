"""Thin single-GPU ODG training entry (research layer).

The author's ``utils/train.py`` is left untouched.  This entry differs in ways
the ODG study needs:

* explicit ``--micro-batch`` / ``--accum-steps``; the optimizer always sees
  exactly ``effective_batch`` (default 16) samples per attempt;
* every trainable tensor reaches the optimizer exactly once: the author's
  ``group_weight`` never matches bare ``nn.Parameter`` attributes, so the decay /
  no-decay groups are built here and verified before the run starts;
* periodic validation reads only the fixed dev split, single scale, no flip, and
  runs in **eval mode**: the model is switched to ``model.eval()`` for the pass,
  back to training mode afterwards, and the pass is rejected if it changed any
  BatchNorm running statistic;
* ``--fulltrain`` is opt-in and disables in-training validation entirely, so the
  official test list can never select the best checkpoint;
* every epoch writes a full checkpoint (model / optimizer / scaler / counters /
  RNG / resolved config / args / contract) plus ``best-dev`` and the 30/100
  stage checkpoints;
* resuming validates the experiment contract and continues the *same* schedule
  without re-warming up;
* ``--stop-after-epoch 30|100`` is a pause: it never compresses the schedule;
* ``--smoke-steps N`` runs N real optimizer attempts for the later GPU
  qualification run (finite loss/grad, parameter change, throughput/memory).
  It is never executed automatically.

This entry never starts training without CUDA; ``--help`` and
``--print-schedule`` are CPU-only.
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

from research import odg_schedule as sched


def build_parser():
    parser = argparse.ArgumentParser(
        description="ODG single-GPU research training entry (DFormerv2-S + HAM)."
    )
    parser.add_argument("--config", required=True, help="config module, e.g. local_configs.research.ODG_SUNRGBD")
    parser.add_argument("--gpus", default=1, type=int, help="this entry is single-GPU; must be 1")
    parser.add_argument("--geometry-mode", choices=("original", "mean", "odg"), default=None,
                        help="override C.geometry_mode")
    parser.add_argument("--micro-batch", default=None, type=int, help="per-forward batch (default C.micro_batch)")
    parser.add_argument("--accum-steps", default=None, type=int, help="gradient accumulation steps")
    parser.add_argument("--effective-batch", default=None, type=int, help="micro_batch*accum_steps target")
    parser.add_argument("--schedule-epochs", default=None, type=int, help="whole-schedule length")
    parser.add_argument("--warmup-epochs", default=None, type=int, help="warmup epochs")
    parser.add_argument("--stop-after-epoch", default=None, type=int, help="stage pause; 0 = full schedule")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--val-every", default=None, type=int, help="periodic dev validation interval")
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--norm-eval", default=None, action=argparse.BooleanOptionalAction,
                        help="freeze BN statistics in train mode; C.norm_eval=False")
    parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--fulltrain", default=False, action="store_true",
                        help="official full train list; disables in-training validation")
    parser.add_argument("--resume", default=None, help="resume the same experiment from a checkpoint")
    parser.add_argument("--smoke-steps", default=0, type=int, help="run N real optimizer attempts, then stop")
    parser.add_argument("--save-predictions", default=0, type=int, help="save N dev predictions per validation")
    parser.add_argument("--print-schedule", default=False, action="store_true",
                        help="print batching and LR schedule numbers on CPU, then exit")
    parser.add_argument("--device", default="cuda")
    return parser


def resolve_config(config, args):
    resolved = {
        "geometry_mode": args.geometry_mode or str(config.geometry_mode),
        "micro_batch": args.micro_batch if args.micro_batch is not None else int(config.micro_batch),
        "accum_steps": args.accum_steps if args.accum_steps is not None else int(config.accum_steps),
        "effective_batch": args.effective_batch if args.effective_batch is not None else int(config.effective_batch),
        "schedule_epochs": args.schedule_epochs if args.schedule_epochs is not None else int(config.schedule_epochs),
        "warmup_epochs": args.warmup_epochs if args.warmup_epochs is not None else int(config.warm_up_epoch),
        "seed": args.seed if args.seed is not None else int(config.seed),
        "num_workers": args.num_workers if args.num_workers is not None else int(config.num_workers),
        "val_every": args.val_every if args.val_every is not None else int(config.val_every),
        "stop_after_epoch": (args.stop_after_epoch if args.stop_after_epoch is not None
                             else int(getattr(config, "stop_after_epoch", 0) or 0)),
        "norm_eval": args.norm_eval if args.norm_eval is not None else bool(getattr(config, "norm_eval", False)),
    }
    if resolved["geometry_mode"] not in ("original", "mean", "odg"):
        raise SystemExit("geometry_mode must be one of original|mean|odg")
    if int(resolved["schedule_epochs"]) <= 0:
        raise SystemExit("schedule_epochs must be positive")
    if not (0 <= int(resolved["warmup_epochs"]) <= int(resolved["schedule_epochs"])):
        raise SystemExit("warmup_epochs must be within [0, schedule_epochs]")
    if resolved["micro_batch"] * resolved["accum_steps"] != resolved["effective_batch"]:
        raise SystemExit(
            "micro_batch (%d) * accum_steps (%d) != effective_batch (%d)"
            % (resolved["micro_batch"], resolved["accum_steps"], resolved["effective_batch"])
        )
    return resolved


def resolve_data_plan(config, args, resolved, require_ready):
    if args.fulltrain:
        train_source, eval_source = sched.resolve_sources(config, True)
        periodic_eval = False
    else:
        if require_ready and not bool(getattr(config, "dev_split_ready", True)):
            raise SystemExit(
                "fixed dev split is not ready for %s; pass --fulltrain or create %s"
                % (config.dataset_name, getattr(config, "dev_split_dir", "the dev split"))
            )
        train_source, eval_source = sched.resolve_sources(config, False)
        periodic_eval = int(resolved["val_every"]) > 0
    if not os.path.exists(train_source) and getattr(config, "num_train_imgs", None) is None:
        raise SystemExit("train source not found: %s" % train_source)
    num_samples = sched.manifest_lines(train_source, fallback=getattr(config, "num_train_imgs", None))
    plan = sched.plan_epoch(num_samples, resolved["micro_batch"], resolved["accum_steps"],
                            resolved["effective_batch"])
    return train_source, eval_source, periodic_eval, plan


def print_schedule(config, resolved, plan, schedule, policy, train_source=None, eval_source=None):
    print("[schedule] dataset=%s geometry=%s" % (config.dataset_name, resolved["geometry_mode"]))
    print("[schedule] train_source=%s" % train_source)
    print("[schedule] eval_source=%s (periodic validation target)" % eval_source)
    print("[schedule] samples/epoch=%d micro_batch=%d accum_steps=%d effective_batch=%d"
          % (plan["num_samples"], plan["micro_batch"], plan["accum_steps"], plan["effective_batch"]))
    print("[schedule] micro_batches/epoch=%d updates/epoch=%d repeated_samples=%d file_length=%d"
          % (plan["micro_batches_per_epoch"], plan["updates_per_epoch"], plan["repeated_samples"],
             plan["file_length"]))
    print("[schedule] schedule_epochs=%d warmup_epochs=%d total_updates=%d warmup_updates=%d"
          % (schedule["schedule_epochs"], schedule["warmup_epochs"], schedule["total_updates"],
             schedule["warmup_updates"]))
    if resolved["stop_after_epoch"]:
        print("[schedule] stop_after_epoch=%d is a pause; total_updates stays %d"
              % (resolved["stop_after_epoch"], schedule["total_updates"]))
    print("[schedule] lr base=%g power=%g" % (float(config.lr), float(config.lr_power)))
    for row in sched.schedule_report(policy, schedule["total_updates"], schedule["warmup_updates"]):
        print("[schedule]   attempt=%7d lr=%.8e" % (row["attempt"], row["lr"]))
    return 0


def build_model(config, criterion):
    from models.builder import EncoderDecoder as segmodel

    return segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)


def rebind_run_dir(config, checkpoint_path):
    """Point the output paths at an existing run directory (used on resume)."""
    run_dir = os.path.dirname(os.path.abspath(str(checkpoint_path)))
    config.log_dir = run_dir
    config.log_dir_link = run_dir
    config.checkpoint_dir = run_dir
    config.tb_dir = os.path.join(run_dir, "tb")
    config.log_file = os.path.join(run_dir, "train.log")
    config.val_log_file = os.path.join(run_dir, "validation.log")
    config.swanlab_log_dir = os.path.join(run_dir, "swanlab")
    return run_dir


def run_validation(model, val_loader, config, device, use_support, amp, out_dir, epoch, args):
    """Periodic dev validation in eval mode.

    Validation must use the trained running statistics, so the model is switched
    to ``model.eval()`` for the pass and back to training mode afterwards.  The
    BatchNorm running statistics are fingerprinted before and after: if the pass
    changed them the run stops instead of silently reporting train-mode numbers.
    """
    hole_ratio = float(getattr(config, "artificial_hole_ratio", 0.0) or 0.0)
    hole_seed = int(getattr(config, "artificial_hole_seed", 12345))
    bn_before = sched.batchnorm_running_state(model)
    with sched.evaluation_mode(model):
        result = sched.evaluate_loader(
            model, val_loader, config, device, use_support, msf=False, flip=False,
            hole_ratio=hole_ratio, hole_seed=hole_seed,
            amp=amp, log_every=max(1, len(val_loader) // 4),
            prediction_dir=(os.path.join(out_dir, "predictions", "epoch-%03d" % epoch)
                            if int(args.save_predictions) > 0 else None),
            prediction_limit=int(args.save_predictions),
        )
    bn_after = sched.batchnorm_running_state(model)
    if bn_after != bn_before:
        raise RuntimeError(
            "dev validation changed BatchNorm running statistics, so it did not run in eval mode: "
            "%s -> %s" % (bn_before, bn_after))
    holes = result.get("artificial_holes", {})
    mode = ("single_scale_no_flip_holes" if hole_ratio > 0.0 else "single_scale_no_flip")
    sched.write_json(os.path.join(out_dir, "val_per_class", "epoch-%03d.json" % epoch), {
        "epoch": epoch, "split": "dev", "mode": mode, "model_mode": sched.VALIDATION_MODE,
        "batchnorm_running_state": bn_after,
        "miou": result["miou"], "macc": result["macc"], "mf1": result["mf1"],
        "num_samples": result["num_samples"], "class_names": list(config.class_names),
        "iou": result["iou"], "acc": result["acc"], "f1": result["f1"],
        "artificial_holes": holes,
    })
    result["mode"] = mode
    result["model_mode"] = sched.VALIDATION_MODE
    result["batchnorm_running_state"] = bn_after
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    if int(args.gpus) != 1:
        raise SystemExit("this entry is single-GPU only; --gpus must be 1")

    config = deepcopy(getattr(import_module(args.config), "C"))
    resolved = resolve_config(config, args)

    if args.print_schedule:
        _train, _eval, _periodic, plan = resolve_data_plan(config, args, resolved, require_ready=False)
        policy, schedule = sched.build_schedule(
            plan["updates_per_epoch"], resolved["schedule_epochs"], resolved["warmup_epochs"],
            float(config.lr), float(config.lr_power), resolved["stop_after_epoch"])
        return print_schedule(config, resolved, plan, schedule, policy, _train, _eval)

    # This training entry is CUDA-only.  There is deliberately no CPU-training
    # switch: on CPU, only --print-schedule / --help are supported.
    if str(args.device) != "cuda":
        print("REFUSED: training supports --device cuda only; use --print-schedule or --help "
              "for CPU-only checks.", file=sys.stderr)
        return 3
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device available. Training is single-GPU only; "
              "use --print-schedule or --help for CPU-only checks.", file=sys.stderr)
        return 3

    train_source, eval_source, periodic_eval, plan = resolve_data_plan(config, args, resolved, require_ready=True)
    if args.resume:
        # Continue the same experiment in place: logs/CSV/checkpoints stay in the
        # resumed run directory instead of a new timestamped one.
        rebind_run_dir(config, args.resume)

    # Author-compatible padding checks for SUNRGBD + DFormerv2.
    pad = bool(args.pad_SUNRGBD)
    if pad and config.dataset_name != "SUNRGBD":
        pad = False
    if pad and not str(config.backbone).startswith("DFormerv2"):
        raise SystemExit("DFormerv1 is not recommended with pad_SUNRGBD")
    if (not pad) and str(config.backbone).startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise SystemExit("DFormerv2 on SUNRGBD requires --pad_SUNRGBD")
    config.pad = pad

    config.geometry_mode = resolved["geometry_mode"]
    config.norm_eval = bool(resolved["norm_eval"])
    config.micro_batch = resolved["micro_batch"]
    config.accum_steps = resolved["accum_steps"]
    config.effective_batch = resolved["effective_batch"]
    config.schedule_epochs = resolved["schedule_epochs"]
    config.warm_up_epoch = resolved["warmup_epochs"]
    config.seed = resolved["seed"]
    config.train_source = train_source
    config.eval_source = eval_source
    config.nepochs = resolved["schedule_epochs"]

    os.makedirs(config.log_dir, exist_ok=True)
    from utils.engine.logger import get_logger
    logger = get_logger(config.log_dir, config.log_file)

    logger.info("geometry_mode=%s norm_eval=%s" % (config.geometry_mode, config.norm_eval))
    logger.info("train_source=%s" % train_source)
    logger.info("eval_source=%s (periodic_eval=%s)" % (eval_source, periodic_eval))
    if args.fulltrain:
        logger.warning("fulltrain mode: in-training periodic validation disabled; official test is "
                       "final-only through research/evaluate_odg.py --split test")
    logger.info("batching: micro_batch=%d accum_steps=%d effective_batch=%d micro_batches/epoch=%d "
                "updates/epoch=%d repeated_samples=%d"
                % (plan["micro_batch"], plan["accum_steps"], plan["effective_batch"],
                   plan["micro_batches_per_epoch"], plan["updates_per_epoch"], plan["repeated_samples"]))

    device = torch.device(args.device)
    sched.seed_everything(resolved["seed"])

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    model = build_model(config, criterion)
    use_support = sched.supports_depth_support(model)
    logger.info("model accepts depth_support: %s" % use_support)
    if config.geometry_mode != "original" and not use_support:
        raise SystemExit("geometry_mode=%s but the model does not accept depth_support" % config.geometry_mode)

    base_lr = float(config.lr)
    # Every trainable tensor must reach the optimizer exactly once.  The author's
    # group_weight() drops bare nn.Parameter attributes (the 29 GeoPriorGen.weight
    # geometry kernels), so the research entry builds the decay / no-decay groups
    # itself and verifies the result before the run starts.
    params_list = sched.build_optimizer_param_groups(model, base_lr, float(config.weight_decay))
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999),
                                      weight_decay=float(config.weight_decay))
    else:
        raise SystemExit("this entry only supports AdamW; got %s" % config.optimizer)
    param_report = sched.optimizer_parameter_report(model, optimizer)
    logger.info("optimizer parameter groups: %s" % param_report)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    policy, schedule = sched.build_schedule(
        plan["updates_per_epoch"], resolved["schedule_epochs"], resolved["warmup_epochs"],
        base_lr, float(config.lr_power), resolved["stop_after_epoch"])
    contract = sched.make_contract(config, config.geometry_mode, train_source, eval_source,
                                   plan["micro_batch"], plan["accum_steps"], plan["effective_batch"],
                                   resolved["schedule_epochs"], resolved["warmup_epochs"],
                                   amp=bool(args.amp))
    logger.info("schedule: total_updates=%d warmup_updates=%d stop_after_epoch=%s"
                % (schedule["total_updates"], schedule["warmup_updates"], resolved["stop_after_epoch"]))

    # Move the model to the target device *before* any optimizer state is loaded,
    # so Adam moments land on the same device as the parameters (a later
    # model.to(cuda) would otherwise leave the optimizer state on CPU).
    model.to(device)

    state = {
        "completed_epochs": 0,
        "attempt_updates": 0,
        "applied_updates": 0,
        "amp_skipped_updates": 0,
        "micro_batches_seen": 0,
        "best_dev_miou": 0.0,
        "geometry_mode": config.geometry_mode,
        "num_train_samples": plan["num_samples"],
        "optimizer_param_report": param_report,
    }

    if args.resume:
        payload = sched.load_training_checkpoint(args.resume, map_location="cpu")
        snapshot_kind = str(payload.get("snapshot_kind", "epoch"))
        if snapshot_kind not in sched.RESUMABLE_SNAPSHOTS or payload.get("epoch_boundary") is False:
            raise SystemExit(
                "refusing to resume from a non-epoch-boundary snapshot (snapshot_kind=%r): %s"
                % (snapshot_kind, args.resume))
        diffs = sched.contract_diff(payload.get("contract", {}), contract)
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
        state["completed_epochs"] = int(payload["completed_epochs"])
        state["attempt_updates"] = int(payload["attempt_updates"])
        state["applied_updates"] = int(payload["applied_updates"])
        state["amp_skipped_updates"] = int(payload["amp_skipped_updates"])
        state["micro_batches_seen"] = int(payload.get("micro_batches_seen", 0))
        state["best_dev_miou"] = float(payload.get("best_dev_miou", 0.0))
        logger.info("resumed %s (snapshot_kind=%s) at completed_epochs=%d attempt_updates=%d (no re-warmup)"
                    % (args.resume, snapshot_kind, state["completed_epochs"], state["attempt_updates"]))

    train_loader, plan = sched.build_train_loader(
        config, train_source, eval_source, plan["micro_batch"], plan["accum_steps"],
        plan["effective_batch"], resolved["num_workers"], resolved["seed"])
    val_loader = None
    if periodic_eval:
        val_loader, val_len = sched.build_eval_loader(
            config, eval_source, train_source, batch_size=1, num_workers=resolved["num_workers"])
        logger.info("dev validation set: %d images (single scale, no flip)" % val_len)

    sched.write_json(os.path.join(config.log_dir, "resolved_config.json"), config)
    sched.write_json(os.path.join(config.log_dir, "args.json"), vars(args))
    sched.write_json(os.path.join(config.log_dir, "contract.json"), contract)

    loss_csv = sched.CsvWriter(os.path.join(config.log_dir, "loss_lr.csv"),
                               ["epoch", "attempt_update", "applied_update", "amp_skipped", "lr",
                                "param_group_lr", "loss", "micro_batches", "epoch_seconds"])
    val_csv = sched.CsvWriter(os.path.join(config.log_dir, "validation.csv"),
                              ["epoch", "mode", "model_mode", "miou", "macc", "mf1", "num_samples",
                               "hole_ratio_requested", "hole_ratio_actual"])

    tracking_run = None
    try:
        from research.tracking import start_tracking
        tracking_run = start_tracking(config, args, _TrackingEngine())
    except Exception as exc:  # SwanLab is optional; never block training
        logger.warning("SwanLab tracking disabled: %s" % exc)

    probes = sched.select_probe_params(model)
    probe_before = {name: param.detach().clone() for name, param in probes.items()}
    logger.info("parameter-change probes (%d): %s" % (len(probes), sorted(probes)))
    smoke_stats = {
        "losses_finite": True,
        "grads_finite": True,
        "grad_checks": 0,
        "nonfinite_grad_params": [],
        "no_grad_params": [],
        "no_grad_params_count": 0,
        "amp_skipped_attempts": 0,
    }
    smoke_start = time.time()
    smoke_samples = 0

    start_epoch = state["completed_epochs"]
    failure = None
    try:
        for epoch in range(start_epoch, resolved["schedule_epochs"]):
            model.train()
            epoch_start = time.time()
            micro_loss_sum = 0.0
            epoch_loss_sum = 0.0
            epoch_updates = 0
            last_update_loss = 0.0
            last_update_lr = base_lr
            optimizer.zero_grad(set_to_none=True)
            loader_iter = iter(train_loader)
            micro_in_group = 0
            stopped = False

            for _ in range(plan["micro_batches_per_epoch"]):
                batch = next(loader_iter)
                data, label, modal_x, support, _fn = sched.unwrap_batch(batch)
                data = data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                modal_x = modal_x.to(device, non_blocking=True)
                if support is not None:
                    support = support.to(device, non_blocking=True)

                autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                            if args.amp else contextlib.nullcontext())
                with autocast:
                    raw_loss = sched.forward_loss(model, data, modal_x, label, support, use_support)
                if not bool(torch.isfinite(raw_loss).all()):
                    smoke_stats["losses_finite"] = False
                    # A non-finite loss must fail immediately: continuing would
                    # poison both the weights and the run record.
                    raise TrainingFailure(
                        "nonfinite_loss",
                        "non-finite loss at epoch %d micro-batch %d (%.6f)"
                        % (epoch + 1, micro_in_group + 1, float(raw_loss.detach())),
                        epoch=epoch + 1)
                loss = raw_loss / plan["accum_steps"]
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro_loss_sum += float(raw_loss.detach())
                micro_in_group += 1
                state["micro_batches_seen"] += 1
                smoke_samples += int(data.shape[0])

                if micro_in_group < plan["accum_steps"]:
                    continue

                # Learning rate is written to every param group *before* the step,
                # and the attempt counter advances on every optimizer attempt.
                lr = sched.lr_for_attempt(policy, state["attempt_updates"], schedule["total_updates"])
                group_lrs = sched.set_param_group_lr(optimizer, lr)

                # Verify the *actual* (unscaled) gradient of every trainable
                # parameter at the optimizer boundary.
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_report = sched.grad_finiteness_report(model)
                smoke_stats["grad_checks"] += 1
                if not grad_report["all_finite"]:
                    smoke_stats["grads_finite"] = False
                    for name in grad_report["nonfinite"]:
                        if name not in smoke_stats["nonfinite_grad_params"]:
                            smoke_stats["nonfinite_grad_params"].append(name)
                    if scaler is None:
                        # Without AMP there is no skip path: stop immediately.
                        raise TrainingFailure(
                            "nonfinite_grad",
                            "non-finite gradient (no AMP) at attempt %d in %s"
                            % (state["attempt_updates"], grad_report["nonfinite"][:5]),
                            epoch=epoch + 1)
                smoke_stats["no_grad_params"] = grad_report["no_grad"]
                smoke_stats["no_grad_params_count"] = len(grad_report["no_grad"])

                if scaler is not None:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    applied = scaler.get_scale() >= scale_before  # False == AMP skipped this attempt
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    applied = True

                if not applied:
                    smoke_stats["amp_skipped_attempts"] += 1
                last_update_loss = micro_loss_sum / plan["accum_steps"]
                epoch_loss_sum += last_update_loss
                epoch_updates += 1
                last_update_lr = group_lrs[0] if group_lrs else lr
                state["attempt_updates"] += 1
                if applied:
                    state["applied_updates"] += 1
                else:
                    state["amp_skipped_updates"] += 1
                loss_csv.write([epoch + 1, state["attempt_updates"], state["applied_updates"],
                                state["amp_skipped_updates"], lr, last_update_lr,
                                last_update_loss, plan["accum_steps"], "%.3f" % (time.time() - epoch_start)])
                micro_loss_sum = 0.0
                micro_in_group = 0

                if args.smoke_steps and state["attempt_updates"] >= int(args.smoke_steps):
                    stopped = True
                    break

            state["completed_epochs"] = epoch if stopped else epoch + 1
            epoch_loss = epoch_loss_sum / epoch_updates if epoch_updates else 0.0

            val_miou = None
            if (periodic_eval and not args.smoke_steps
                    and ((state["completed_epochs"] % resolved["val_every"] == 0)
                         or state["completed_epochs"] == resolved["schedule_epochs"])):
                result = run_validation(model, val_loader, config, device, use_support, args.amp,
                                        config.log_dir, state["completed_epochs"], args)
                holes = result.get("artificial_holes", {})
                val_csv.write([state["completed_epochs"], result.get("mode", "single_scale_no_flip"),
                               result.get("model_mode", sched.VALIDATION_MODE),
                               result["miou"], result["macc"], result["mf1"], result["num_samples"],
                               holes.get("requested_ratio", 0.0), holes.get("actual_ratio", 0.0)])
                logger.info("dev epoch %d: mIoU=%.2f mAcc=%.2f mF1=%.2f (best=%.2f)"
                            % (state["completed_epochs"], result["miou"], result["macc"], result["mf1"],
                               state["best_dev_miou"]))
                val_miou = float(result["miou"])
                if val_miou > state["best_dev_miou"]:
                    state["best_dev_miou"] = val_miou
                    sched.save_training_checkpoint(os.path.join(config.log_dir, "best-dev.pth"),
                                                   model, optimizer, scaler, state, config, args, contract,
                                                   snapshot_kind="best")
                model.train()

            if tracking_run is not None:
                try:
                    from research.tracking import log_epoch
                    log_epoch(tracking_run, state["completed_epochs"], state["attempt_updates"],
                              epoch_loss, last_update_lr, val_miou)
                except Exception:
                    pass
            logger.info("epoch %d/%d done: attempts=%d applied=%d amp_skipped=%d mean_loss=%.6f "
                        "val_miou=%s elapsed=%.1fs"
                        % (state["completed_epochs"], resolved["schedule_epochs"], state["attempt_updates"],
                           state["applied_updates"], state["amp_skipped_updates"], epoch_loss,
                           ("%.2f" % val_miou) if val_miou is not None else "n/a",
                           time.time() - epoch_start))

            if not args.smoke_steps:
                sched.save_training_checkpoint(os.path.join(config.log_dir, "last.pth"),
                                               model, optimizer, scaler, state, config, args, contract,
                                               snapshot_kind="epoch")
                for stage in tuple(getattr(config, "stage_epochs", (30, 100)) or ()):
                    if state["completed_epochs"] == int(stage):
                        sched.save_training_checkpoint(
                            os.path.join(config.log_dir, "stage-epoch-%d.pth" % int(stage)),
                            model, optimizer, scaler, state, config, args, contract,
                            snapshot_kind="stage")

            if args.smoke_steps:
                break
            if resolved["stop_after_epoch"] and state["completed_epochs"] >= resolved["stop_after_epoch"]:
                logger.info("stage pause after %d epochs; schedule was not compressed"
                            % state["completed_epochs"])
                break

        if args.smoke_steps:
            elapsed = max(time.time() - smoke_start, sched.EPS)
            peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
            delta_report, changed = sched.parameter_delta_report(probes, probe_before)
            report = {
                "smoke_steps": int(args.smoke_steps),
                "attempt_updates": state["attempt_updates"],
                "applied_updates": state["applied_updates"],
                "amp_skipped_updates": state["amp_skipped_updates"],
                "micro_batches_seen": state["micro_batches_seen"],
                "samples": int(smoke_samples),
                "seconds": round(elapsed, 3),
                "samples_per_second": round(smoke_samples / elapsed, 3),
                "peak_memory_mb": round(peak_mb, 1),
                "losses_finite": bool(smoke_stats["losses_finite"]),
                "grads_finite": bool(smoke_stats["grads_finite"]),
                "grad_checks": int(smoke_stats["grad_checks"]),
                "nonfinite_grad_params": list(smoke_stats["nonfinite_grad_params"]),
                "amp_skipped_attempts": int(smoke_stats["amp_skipped_attempts"]),
                "no_grad_params_count": int(smoke_stats["no_grad_params_count"]),
                "no_grad_params": list(smoke_stats["no_grad_params"])[:50],
                "probe_count": len(delta_report),
                "probe_changed_count": int(changed),
                "probe_params": delta_report,
                "best_dev_miou": state["best_dev_miou"],
            }
            sched.write_json(os.path.join(config.log_dir, "smoke-report.json"), report)
            sched.save_training_checkpoint(os.path.join(config.log_dir, "smoke.pth"),
                                           model, optimizer, scaler, state, config, args, contract,
                                           snapshot_kind="smoke")
            logger.info("smoke report: %s" % report)

        logger.info("training entry finished: completed_epochs=%d attempt_updates=%d"
                    % (state["completed_epochs"], state["attempt_updates"]))
    except TrainingFailure as exc:
        failure = exc
        logger.error("training failed (%s): %s" % (exc.kind, exc))
        sched.write_json(os.path.join(config.log_dir, "failure.json"), {
            "failed": True,
            "kind": str(exc.kind),
            "detail": str(exc),
            "epoch": exc.epoch,
            "completed_epochs": int(state["completed_epochs"]),
            "attempt_updates": int(state["attempt_updates"]),
            "applied_updates": int(state["applied_updates"]),
            "amp_skipped_updates": int(state["amp_skipped_updates"]),
            "grad_checks": int(smoke_stats["grad_checks"]),
            "nonfinite_grad_params": list(smoke_stats["nonfinite_grad_params"])[:50],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    finally:
        if tracking_run is not None:
            try:
                tracking_run.finish()
            except Exception:
                pass

    if failure is not None:
        return 4
    return 0


class TrainingFailure(RuntimeError):
    """A run-stopping training failure that must write an explicit failed state."""

    def __init__(self, kind, detail, epoch=None):
        super().__init__(detail)
        self.kind = str(kind)
        self.epoch = epoch


class _TrackingEngine:
    """Minimal shim so research/tracking.py can run outside the author Engine."""

    distributed = False
    local_rank = 0
    world_size = 1


if __name__ == "__main__":
    sys.exit(main())
