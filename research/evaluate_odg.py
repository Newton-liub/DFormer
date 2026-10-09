"""Thin ODG evaluation entry (research layer).

Defaults match the study protocol:

* single scale, no flip;
* the fixed dev split (``--split dev``);
* the dataset's explicit ``depth_support`` mask is forwarded to the model;
* ``--msf`` is the only way to run the five-scale + flip protocol, and
  ``--split test`` is the only way to read the official test list.

Fusion for ``--msf`` is identical to the author's ``evaluate_msf``: per-view
softmax maps are summed (in FP32) and only then argmaxed.  Missing depth is
never inferred from ``tensor == 0``; artificial connected holes are an explicit
opt-in (``--hole-ratio`` with a fixed ``--hole-seed``) that deletes observed
support and fills the same depth pixels with the canonical raw-zero normalised
value (``-0.48/0.28``), so every geometry mode sees an identical input.  The
realised deleted fraction of the true support domain is recorded; a
support-only deletion is never reported as missing-depth robustness.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from importlib import import_module

import torch
import torch.nn as nn

from research import odg_schedule as sched


def build_parser():
    parser = argparse.ArgumentParser(description="ODG evaluation entry (DFormerv2-S + HAM).")
    parser.add_argument("--config", required=True, help="config module, e.g. local_configs.research.ODG_SUNRGBD")
    parser.add_argument("--checkpoint", required=True, help="ODG research checkpoint (train_odg.py output)")
    parser.add_argument("--allow-raw", default=False, action="store_true",
                        help="accept a plain state dict / author checkpoint with model weights")
    parser.add_argument("--split", choices=("dev", "test"), default="dev",
                        help="dev = fixed development split; test = official test (final only)")
    parser.add_argument("--eval-source", default=None, help="explicit eval list path (overrides --split)")
    parser.add_argument("--geometry-mode", choices=("original", "mean", "odg"), default=None,
                        help="override the geometry mode recorded in the checkpoint")
    parser.add_argument("--pad_SUNRGBD", default=None, action=argparse.BooleanOptionalAction,
                        help="override padding; default follows the checkpoint/config")
    parser.add_argument("--msf", default=False, action="store_true", help="five-scale + flip (off by default)")
    parser.add_argument("--scales", default=None, type=float, nargs="+",
                        help="scales for --msf (default 0.5 0.75 1.0 1.25 1.5)")
    parser.add_argument("--no-msf-flip", default=False, action="store_true", help="disable flip inside --msf")
    parser.add_argument("--batch-size", default=1, type=int, help="eval micro-batch")
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--hole-ratio", default=0.0, type=float,
                        help="artificial connected-hole ratio on the support mask (0 = off)")
    parser.add_argument("--hole-seed", default=12345, type=int)
    parser.add_argument("--save-predictions", default=0, type=int, help="save N argmax prediction PNGs")
    parser.add_argument("--out", default=None, help="output dir (default: alongside the checkpoint)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", default=False, action="store_true", help="run on CPU for debugging")
    parser.add_argument("--strict", default=True, action=argparse.BooleanOptionalAction)
    return parser


def load_state_dict(path, allow_raw):
    try:
        payload = sched.load_training_checkpoint(path, map_location="cpu")
        return payload["model"], payload
    except ValueError:
        if not allow_raw:
            raise
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            for key in ("model", "state_dict"):
                if key in raw:
                    return raw[key], {}
        return raw, {}


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = deepcopy(getattr(import_module(args.config), "C"))

    if args.device.startswith("cuda") and not torch.cuda.is_available() and not args.allow_cpu:
        print("REFUSED: no CUDA device available. Pass --allow-cpu for a slow debugging run.",
              file=sys.stderr)
        return 3
    if args.allow_cpu:
        args.device = "cpu"

    state_dict, payload = load_state_dict(args.checkpoint, args.allow_raw)
    checkpoint_geometry = payload.get("geometry_mode") if payload else None
    if args.geometry_mode and checkpoint_geometry and args.geometry_mode != checkpoint_geometry:
        print("WARNING: overriding checkpoint geometry_mode=%s with %s"
              % (checkpoint_geometry, args.geometry_mode), file=sys.stderr)
    geometry_mode = args.geometry_mode or checkpoint_geometry or config.geometry_mode
    config.geometry_mode = geometry_mode

    saved = payload.get("contract", {}) if payload else {}
    for field in ("dataset_name", "backbone", "num_classes"):
        if saved.get(field) is not None and str(saved.get(field)) != str(getattr(config, field)):
            print("WARNING: checkpoint %s=%r differs from config %r"
                  % (field, saved.get(field), getattr(config, field)), file=sys.stderr)

    saved_cfg = payload.get("config", {}) if payload else {}
    pad = args.pad_SUNRGBD
    if pad is None:
        pad = bool(saved_cfg.get("pad", False))
    config.pad = bool(pad)
    if args.split == "test":
        print("WARNING: evaluating the official test split; use it for a frozen final checkpoint only.",
              file=sys.stderr)
    eval_source = args.eval_source or (str(config.full_test_source) if args.split == "test"
                                       else str(config.dev_eval_source))
    train_source = str(config.train_dev_source)

    device = torch.device(args.device)
    sched.seed_everything(int(getattr(config, "seed", 12345)))

    from models.builder import EncoderDecoder as segmodel
    model = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    use_support = sched.supports_depth_support(model)
    if geometry_mode != "original" and not use_support:
        raise SystemExit("geometry_mode=%s but the model does not accept depth_support" % geometry_mode)
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(args.strict))
    if missing or unexpected:
        print("[evaluate] non-strict load: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    model.to(device)
    model.eval()
    print("[evaluate] checkpoint=%s geometry=%s device=%s" % (args.checkpoint, geometry_mode, device))

    num_workers = args.num_workers if args.num_workers is not None else int(config.num_workers)
    loader, dataset_len = sched.build_eval_loader(config, eval_source, train_source,
                                                  batch_size=args.batch_size, num_workers=num_workers)
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                       "eval-%s" % time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    scales = tuple(args.scales) if args.scales else (0.5, 0.75, 1.0, 1.25, 1.5)

    print("[evaluate] split=%s source=%s images=%d msf=%s scales=%s flip=%s holes=%s"
          % (args.split, eval_source, dataset_len, args.msf, scales if args.msf else (1.0,),
             (not args.no_msf_flip) if args.msf else False, args.hole_ratio))

    start = time.time()
    result = sched.evaluate_loader(
        model, loader, config, device, use_support,
        msf=bool(args.msf), scales=scales, flip=(not args.no_msf_flip),
        hole_ratio=float(args.hole_ratio), hole_seed=int(args.hole_seed),
        amp=bool(args.amp) and device.type == "cuda",
        log_every=max(1, dataset_len // 10),
        prediction_dir=os.path.join(out_dir, "predictions") if args.save_predictions else None,
        prediction_limit=int(args.save_predictions),
    )
    elapsed = time.time() - start
    holes = result.get("artificial_holes", {})

    print("[evaluate] mIoU=%.2f mAcc=%.2f mF1=%.2f images=%d seconds=%.1f"
          % (result["miou"], result["macc"], result["mf1"], result["num_samples"], elapsed))
    print("[evaluate] per-class IoU: %s" % result["iou"])
    if float(args.hole_ratio) > 0.0:
        print("[evaluate] artificial holes: requested_ratio=%.4f actual_ratio=%.4f deleted=%d/%d "
              "fill=%.6f (mask-only deletions are NOT counted as missing-depth robustness)"
              % (holes.get("requested_ratio", float(args.hole_ratio)), holes.get("actual_ratio", 0.0),
                 holes.get("deleted_pixels", 0), holes.get("support_pixels", 0),
                 sched.resolve_hole_fill_value()))

    record = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "geometry_mode": geometry_mode,
        "split": args.split,
        "eval_source": eval_source,
        "mode": ("five_scale_flip" if (args.msf and not args.no_msf_flip)
                 else ("five_scale" if args.msf else "single_scale_no_flip")),
        "scales": list(scales) if args.msf else [1.0],
        "flip": (not args.no_msf_flip) if args.msf else False,
        "fusion": "sum of per-view softmax probabilities, then argmax",
        "hole_ratio": float(args.hole_ratio),
        "hole_seed": int(args.hole_seed) if args.hole_ratio > 0 else None,
        "hole_fill_value": sched.resolve_hole_fill_value() if args.hole_ratio > 0 else None,
        "hole_actual_ratio": holes.get("actual_ratio") if args.hole_ratio > 0 else None,
        "hole_deleted_pixels": holes.get("deleted_pixels") if args.hole_ratio > 0 else None,
        "hole_support_pixels": holes.get("support_pixels") if args.hole_ratio > 0 else None,
        "num_samples": result["num_samples"],
        "seconds": round(elapsed, 2),
        "miou": result["miou"], "macc": result["macc"], "mf1": result["mf1"],
        "class_names": list(config.class_names),
        "iou": result["iou"], "acc": result["acc"], "f1": result["f1"],
    }
    sched.write_json(os.path.join(out_dir, "eval_result.json"), record)
    csv = sched.CsvWriter(os.path.join(out_dir, "eval_per_class.csv"),
                          ["class_index", "class_name", "iou", "acc", "f1"])
    for i, name in enumerate(config.class_names):
        csv.write([i, name, result["iou"][i], result["acc"][i], result["f1"][i]])
    print("[evaluate] wrote %s" % os.path.join(out_dir, "eval_result.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
