"""OC-RGA independent evaluation entry and the shared 640 condition protocol.

Frozen protocol (design 12.1)::

    read native RGB / Depth / label
      -> scale so the long side is ``eval_long_side`` (aspect preserved)
      -> pad right/bottom to a multiple of 32 (new ``S = O = 0``)
      -> generate the fault in that 640 space
      -> forward
      -> drop the pad, bilinear upsample to the *original* label size
      -> 25-class confusion matrix

The fault is never generated before the resize, so a hidden value can never be
smeared into its neighbourhood by a later interpolation.  Both arms receive the
same fault realisation because it is a deterministic function of
``(corruption_seed, stable sample id, condition)``.

Seven conditions are pre-registered: ``clean``, ``holes25``, ``holes50``,
``holes75``, ``noise020``, ``local_value_shift``, ``entire_missing``.
``local_value_shift`` is a local depth **value** offset (design 12.1 / v0.3
clarification 3); ``localshift`` remains accepted as a legacy CLI alias.  The
difficulty mean is the equal-weight mean of ``holes25/50/75``, ``noise020`` and
``local_value_shift``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research import odg_schedule as sched
from research.oc_rga import Z_MAX, Z_MIN, ZERO_FILL
from research.oc_rga_data import (
    _delete_exactly,
    _local_rectangle,
    _rectangle_holes,
    stream_seed,
)

__all__ = [
    "CONDITIONS",
    "HARD_CONDITIONS",
    "DIFFICULTY_MEAN",
    "apply_condition",
    "condition_report",
    "evaluate_conditions",
    "evaluate_checkpoint",
]

CONDITIONS = ("clean", "holes25", "holes50", "holes75", "noise020", "local_value_shift",
              "entire_missing")
HARD_CONDITIONS = ("holes25", "holes50", "holes75", "noise020", "local_value_shift")
DIFFICULTY_MEAN = HARD_CONDITIONS
# ``localshift`` was the earlier CLI spelling; ``local_value_shift`` is canonical
# because the corruption is a local depth *value* offset, not a spatial shift.
CONDITION_ALIASES = {"localshift": "local_value_shift", "entiremissing": "entire_missing"}
HOLE_RATIOS = {"holes25": 0.25, "holes50": 0.50, "holes75": 0.75}
NOISE_STD = {"noise020": 0.20}
LOCAL_SHIFT_AREA = 0.25
LOCAL_SHIFT_MAGNITUDE = 0.60
LOCAL_SHIFT_OUTLIER_RATE = 0.10


def normalise_condition(name: str) -> str:
    key = str(name).strip().lower()
    key = CONDITION_ALIASES.get(key, key)
    if key not in CONDITIONS:
        raise ValueError("unknown condition %r; known: %s" % (name, CONDITIONS))
    return key


# --------------------------------------------------------------------------- #
# Faults on the 640 padded grid
# --------------------------------------------------------------------------- #
def _hw_plane(tensor: torch.Tensor) -> np.ndarray:
    """Host-side ``[H,W]`` view of a ``[B,1,H,W]`` or ``[1,H,W]`` support tensor."""
    array = tensor.detach().cpu().numpy()
    return array.reshape(-1, array.shape[-2], array.shape[-1])[0]


def apply_condition(name: str, modal_x: torch.Tensor, image_support: torch.Tensor,
                    observation: torch.Tensor, corruption_seed: int, sample_id: int,
                    config) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Return ``(modal_x, O, info)`` with the registered condition applied."""
    condition = normalise_condition(name)
    depth = modal_x.clone()
    obs = observation.clone()
    info = {"condition": condition, "deleted_pixels": 0.0, "support_pixels": float(image_support.sum())}

    if condition == "clean":
        obs = image_support.clone()
        return depth, obs, info

    if condition == "entire_missing":
        obs = torch.zeros_like(obs)
        depth = torch.full_like(depth, float(ZERO_FILL))
        info["deleted_pixels"] = float(image_support.sum())
        return depth, obs, info

    if condition == "noise020":
        rng = np.random.default_rng(stream_seed(corruption_seed, 0, sample_id, 0, "noise020"))
        support = _hw_plane(image_support)
        noise = rng.normal(0.0, NOISE_STD["noise020"], size=support.shape).astype(np.float32)
        noise_t = torch.from_numpy(noise).to(depth.device).view(1, 1, *support.shape)
        support_t = image_support.view(1, 1, *support.shape).to(depth.dtype)
        depth = torch.clamp(depth + noise_t * support_t, float(Z_MIN), float(Z_MAX))
        obs = image_support.clone()
        return depth, obs, info

    if condition == "local_value_shift":
        rng = np.random.default_rng(stream_seed(corruption_seed, 0, sample_id, 0, condition))
        support = (_hw_plane(image_support) > 0)
        height, width = support.shape
        region = _local_rectangle(height, width, LOCAL_SHIFT_AREA, rng) & support
        sign = 1.0 if rng.random() < 0.5 else -1.0
        offset = sign * LOCAL_SHIFT_MAGNITUDE
        shift = np.zeros(support.shape, dtype=np.float32)
        shift[region] = offset
        shift_t = torch.from_numpy(shift).to(depth.device).view(1, 1, height, width)
        depth = torch.clamp(depth + shift_t, float(Z_MIN), float(Z_MAX))

        # The 10% anomalies in the shifted region *replace* the depth value on their
        # own mask (the same convention as the training corruption), instead of
        # being added on top of the offset and then clipped to an encoded endpoint.
        outlier_mask = np.zeros(support.shape, dtype=bool)
        flat_idx = np.flatnonzero(region.reshape(-1))
        if flat_idx.size:
            count = max(1, int(round(LOCAL_SHIFT_OUTLIER_RATE * flat_idx.size)))
            picked = rng.choice(flat_idx, size=count, replace=False)
            outlier_mask.reshape(-1)[picked] = True
        replacement = rng.uniform(Z_MIN, Z_MAX, size=support.shape).astype(np.float32)
        mask_t = torch.from_numpy(outlier_mask).to(depth.device).view(1, 1, height, width)
        replace_t = torch.from_numpy(replacement).to(depth.device).view(1, 1, height, width)
        # [1,3,H,W] with a [1,1,H,W] mask/value pair: all three Depth channels change
        # consistently, and O is untouched because this is non-zero corruption.
        depth = torch.where(mask_t, replace_t, depth)
        obs = image_support.clone()
        info["shift"] = float(offset)
        info["region_pixels"] = float(region.sum())
        info["replaced_pixels"] = float(outlier_mask.sum())
        return depth, obs, info

    if condition in HOLE_RATIOS:
        rng = np.random.default_rng(stream_seed(corruption_seed, 0, sample_id, 0, condition))
        support = (_hw_plane(image_support) > 0).astype(np.float32)
        height, width = support.shape
        target = int(round(HOLE_RATIOS[condition] * float(support.sum())))
        # Rectangle-dominated holes on the 640 input grid: grow local windows until
        # the count is reachable, then peel the union boundary to the exact count.
        deleted = _rectangle_holes(
            support, target, float(min(height, width)),
            float(getattr(config, "degrade_local_window_low", 0.10)),
            float(getattr(config, "degrade_local_window_high", 0.40)),
            rng,
        )
        deleted_t = torch.from_numpy(deleted).to(depth.device).view(1, 1, height, width)
        depth = torch.where(deleted_t, torch.full_like(depth, float(ZERO_FILL)), depth)
        obs = torch.where(deleted_t, torch.zeros_like(obs), obs)
        info["deleted_pixels"] = float(deleted.sum())
        info["realised_ratio"] = float(deleted.sum()) / float(max(support.sum(), 1.0))
        return depth, obs, info

    raise ValueError("condition %r has no implementation" % (name,))


def rectangle_dominance(deleted: np.ndarray) -> Dict[str, float]:
    """How rectangle-shaped a deletion mask is (audit R3 spot check).

    Reports the connected-component count and the fill ratio inside the union of
    each component's bounding box: a rectangle-dominated mask has a fill ratio
    close to 1, while scattered independent pixels give a small one.
    """
    import cv2

    labels_count, labels = cv2.connectedComponents(deleted.astype(np.uint8), connectivity=4)
    total = int(deleted.sum())
    boxes_area = 0
    for label in range(1, labels_count):
        rows, cols = np.nonzero(labels == label)
        if rows.size == 0:
            continue
        boxes_area += int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
    return {
        "components": int(labels_count - 1),
        "deleted_pixels": total,
        "bbox_area": boxes_area,
        "fill_ratio": (total / boxes_area) if boxes_area else 0.0,
    }


def condition_report(metrics: Dict[str, object]) -> Dict[str, object]:
    """Trim a SegMetrics result for the per-condition JSON record."""
    return {
        "miou": metrics["miou"],
        "macc": metrics["macc"],
        "mf1": metrics["mf1"],
        "iou": metrics["iou"],
        "acc": metrics["acc"],
        "f1": metrics["f1"],
        "num_samples": metrics["num_samples"],
        "confusion": metrics["confusion"],
    }


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #
def evaluate_conditions(model, loader, conditions: Sequence[str], config, device,
                        amp: bool = False, ham_seed: int = 12345, corruption_seed: int = 12345,
                        max_batches: int = 0, log_fn: Optional[Callable[[str], None]] = None,
                        progress_every: int = 0) -> Dict[str, Dict[str, object]]:
    """Run every requested condition over the loader and return per-condition metrics.

    The caller owns the train/eval mode switch: this function only reads.  Each
    image's HAM random basis is reseeded from ``ham_seed + stable sample id`` and
    the torch RNG state is restored afterwards, so a paired comparison across
    methods and conditions uses the same sampling state.
    """
    log = log_fn if log_fn is not None else (lambda message: print(message))
    num_classes = int(config.num_classes)
    ignore = int(config.background)
    results: Dict[str, Dict[str, object]] = {}
    was_training = bool(model.training)
    # The whole validation pass is wrapped in a full RNG save/restore (Python,
    # NumPy, CPU torch and every CUDA device), and each image reseeds inside a
    # ``fork_rng`` that covers the target devices.  Evaluating therefore cannot
    # leave the training DropPath / HAM stream at the last image's seed.
    outer_rng = sched.capture_rng()
    model.eval()
    try:
        for condition in conditions:
            name = normalise_condition(condition)
            metrics = sched.SegMetrics(num_classes, ignore, device)
            confusion_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
            samples = 0
            deleted = support_pixels = 0.0
            start = time.time()
            for step, batch in enumerate(loader):
                if max_batches and step >= max_batches:
                    break
                data = batch["data"].to(device, non_blocking=True)
                modal_x = batch["modal_x"].to(device, non_blocking=True)
                image_support = batch["image_support"].to(device, non_blocking=True)
                observation = batch["depth_support"].to(device, non_blocking=True)
                label = batch["label"][0].to(device, non_blocking=True)
                original = [int(v) for v in batch["original_size"][0].tolist()]
                scaled = [int(v) for v in batch["scaled_size"][0].tolist()]
                sample_id = int(batch["sample_id"][0])

                modal_c, obs_c, info = apply_condition(
                    name, modal_x, image_support, observation, corruption_seed, sample_id, config
                )
                deleted += float(info.get("deleted_pixels", 0.0))
                support_pixels += float(info.get("support_pixels", 0.0))

                with torch.random.fork_rng():
                    sched.seed_everything(int(ham_seed) + sample_id)
                    with torch.no_grad():
                        autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                                    if amp and device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))
                        with autocast:
                            logits = model(data, modal_c, depth_support=obs_c, image_support=image_support)

                logits = logits[..., : scaled[0], : scaled[1]]
                if tuple(logits.shape[-2:]) != (original[0], original[1]):
                    logits = F.interpolate(logits, size=(original[0], original[1]),
                                           mode="bilinear", align_corners=False)
                probs = logits.float().softmax(dim=1)
                metrics.update(probs, label.unsqueeze(0))
                samples += 1
                if progress_every and (step + 1) % progress_every == 0:
                    log("[evaluate] %s %d/%d" % (name, step + 1, len(loader)))
            result = metrics.compute()
            result["num_samples"] = int(samples)
            result["confusion"] = metrics.hist.detach().cpu().long().tolist()
            result["seconds"] = round(time.time() - start, 3)
            result["deleted_pixels"] = int(deleted)
            result["support_pixels"] = int(support_pixels)
            result["realised_ratio"] = deleted / support_pixels if support_pixels else 0.0
            result["class_names"] = list(config.class_names)
            results[name] = result
            log("[evaluate] %-15s mIoU=%.4f mAcc=%.4f mF1=%.4f (n=%d, %.1fs)"
                % (name, result["miou"], result["macc"], result["mf1"], samples, result["seconds"]))
    finally:
        sched.restore_rng(outer_rng)
        model.train(was_training)
    return results


def difficulty_mean(results: Dict[str, Dict[str, object]],
                    conditions: Sequence[str] = DIFFICULTY_MEAN) -> Optional[float]:
    present = [c for c in conditions if c in results]
    if len(present) != len(conditions):
        return None
    return round(float(np.mean([results[c]["miou"] for c in present])), 4)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_config(config_name: str, checkpoint: Optional[str] = None):
    config = deepcopy(getattr(import_module(config_name), "C"))
    if checkpoint:
        run_dir = str(Path(checkpoint).resolve().parent)
        config.log_dir = run_dir
        config.checkpoint_dir = run_dir
    return config


def evaluate_checkpoint(config_name: str, checkpoint: str, split: str = "dev",
                        conditions: Sequence[str] = CONDITIONS, eval_long_side: int = 640,
                        ham_seed: int = 12345, corruption_seed: int = 12345,
                        device: str = "cuda", max_batches: int = 0,
                        model_mode: Optional[str] = None, num_workers: int = 0,
                        out_dir: Optional[str] = None, amp: bool = True) -> Dict[str, object]:
    from research.oc_rga_data import build_eval_loader
    from research.oc_rga_model import build_model_for_mode

    config = load_config(config_name, checkpoint)
    config.eval_long_side = int(eval_long_side)

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise SystemExit("not a research checkpoint: %s" % checkpoint)
    contract = payload.get("contract", {})
    mode = str(model_mode or contract.get("model_mode") or getattr(config, "model_mode", "oc_rga"))
    saved = contract.get("model_mode")
    if saved and mode != saved:
        raise SystemExit("checkpoint was trained as %r but %r was requested" % (saved, mode))

    config.seed = int(contract.get("seed", getattr(config, "seed", 12345)))
    eval_source = str(config.full_test_source if split == "test" else config.dev_eval_source)
    train_source = str(contract.get("train_source_path") or config.train_dev_source)

    dev = torch.device(device)
    model = build_model_for_mode(config, mode, criterion=None)
    model.load_state_dict(payload["model"], strict=True)
    model.to(dev)
    restore = sched.restore_decoder_norm_config(model, config)
    print("[evaluate] restored decoder norm config: %s" % restore[:3])

    loader, count = build_eval_loader(config, eval_source, train_source, num_workers=num_workers)
    print("[evaluate] split=%s (%d images) conditions=%s long_side=%d"
          % (split, count, list(conditions), eval_long_side))
    results = evaluate_conditions(model, loader, conditions, config, dev, amp=amp,
                                 ham_seed=int(ham_seed), corruption_seed=int(corruption_seed),
                                 max_batches=int(max_batches), progress_every=max(1, count // 8))
    report = {
        "config": config_name,
        "checkpoint": str(checkpoint),
        "model_mode": mode,
        "split": split,
        "num_images": int(count),
        "conditions": list(conditions),
        "eval_long_side": int(eval_long_side),
        "ham_seed": int(ham_seed),
        "corruption_seed": int(corruption_seed),
        "contract": contract,
        "results": results,
        "difficulty_mean": difficulty_mean(results),
        "protocol_note": "shared 640 long-side single-scale protocol; not the official DeLiVER HHA evaluation",
    }
    target_dir = Path(out_dir or Path(config.log_dir) / "eval")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / ("eval-%s-%s.json" % (split, "-".join(sorted(results))))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sched.jsonable(report), handle, indent=2)
    print("[evaluate] difficulty mean=%s; wrote %s" % (report["difficulty_mean"], path))
    return report


def run_self_check(config_name: str = "local_configs.research.OC_RGA_DeLiVER", device: str = "cpu",
                   limit: int = 3, model_mode: str = "masked_baseline",
                   report_path: Optional[str] = None) -> Dict[str, object]:
    """Protocol-level check of the evaluation entry on a few real images.

    Verifies the resize/pad/un-pad/upsample chain, the confusion-matrix
    bookkeeping, the per-condition marker invariants and the determinism of the
    fault realisation.  It uses a freshly built model (no checkpoint) because the
    object under test is the protocol, not a trained score.
    """
    import os

    from research.oc_rga_data import build_eval_loader
    from research.oc_rga_model import build_model_for_mode

    config = load_config(config_name)
    dev = torch.device(device)
    report: Dict[str, object] = {"config": config_name, "device": str(dev), "checks": {}, "failures": []}

    def record(name, ok, detail):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["failures"].append(name)
        print("[eval-self-check] %-36s %s  %s" % (name, "PASS" if ok else "FAIL", detail))

    loader, count = build_eval_loader(config, config.dev_eval_source, config.train_dev_source, num_workers=0)
    model = build_model_for_mode(config, model_mode, criterion=None).to(dev)
    results = evaluate_conditions(model, loader, CONDITIONS, config, dev, amp=False,
                                 ham_seed=int(getattr(config, "eval_ham_seed", 12345)),
                                 corruption_seed=int(getattr(config, "eval_corruption_seed", 12345)),
                                 max_batches=int(limit))

    record("all_conditions_evaluated", sorted(results) == sorted(CONDITIONS),
           "conditions=%s" % sorted(results))

    batch = None
    for candidate in loader:
        batch = candidate
        break
    data = batch["data"].to(dev)
    modal_x = batch["modal_x"].to(dev)
    support = batch["image_support"].to(dev)
    obs = batch["depth_support"].to(dev)
    original = [int(v) for v in batch["original_size"][0].tolist()]
    scaled = [int(v) for v in batch["scaled_size"][0].tolist()]
    padded = [int(v) for v in batch["padded_size"][0].tolist()]
    sample_id = int(batch["sample_id"][0])

    ok_geometry = (
        data.shape[-2:] == torch.Size(padded)
        and abs(scaled[0] / scaled[1] - original[0] / original[1]) < 0.02
        and max(scaled) == int(getattr(config, "eval_long_side", 640))
        and padded[0] % 32 == 0 and padded[1] % 32 == 0
    )
    record("resize_pad_chain", ok_geometry,
           "native=%s scaled=%s padded=%s" % (original, scaled, padded))

    _d, obs_clean, _i = apply_condition("clean", modal_x, support, obs, 12345, sample_id, config)
    _d, obs_missing, _i = apply_condition("entire_missing", modal_x, support, obs, 12345, sample_id, config)
    _d, obs_holes, _i = apply_condition("holes75", modal_x, support, obs, 12345, sample_id, config)
    _d, obs_noise, _i = apply_condition("noise020", modal_x, support, obs, 12345, sample_id, config)
    record("clean_observation_equals_support", bool(torch.equal(obs_clean, support)),
           "clean leaves O = S")
    record("entire_missing_observation_zero", float(obs_missing.max()) == 0.0,
           "entire_missing sets O = 0")
    record("holes_reduce_observation", float(obs_holes.mean()) < float(support.mean()),
           "holes75 O mean=%.4f < S mean=%.4f" % (float(obs_holes.mean()), float(support.mean())))
    record("noise_keeps_observation", bool(torch.equal(obs_noise, support)),
           "noise020 does not delete observations")

    again_a, _oa, _ia = apply_condition("holes50", modal_x, support, obs, 12345, sample_id, config)
    again_b, _ob, _ib = apply_condition("holes50", modal_x, support, obs, 12345, sample_id, config)
    record("fault_realisation_deterministic", bool(torch.equal(again_a, again_b)),
           "same (seed, sample id, condition) -> identical fault")

    # R3: the holes must be rectangle-dominated at every registered severity (not
    # a scattered independent-pixel mask), the count must be exact, and the final
    # boundary adjustment must stay small.
    support_plane = (_hw_plane(support) > 0).astype(np.float32)
    dominance = {}
    for condition, ratio in sorted(HOLE_RATIOS.items()):
        target = int(round(ratio * float(support_plane.sum())))
        rng = np.random.default_rng(stream_seed(12345, 0, sample_id, 0, "r3-%s" % condition))
        window_stats: Dict[str, float] = {}
        deleted = _rectangle_holes(support_plane, target, float(min(support_plane.shape)),
                                   0.10, 0.40, rng, stats=window_stats)
        scattered = _delete_exactly(np.zeros_like(support_plane, dtype=bool),
                                    support_plane, target, rng)
        rect = rectangle_dominance(deleted)
        scat = rectangle_dominance(scattered)
        dominance[condition] = {
            "requested_ratio": ratio,
            "realised_ratio": float(deleted.sum()) / max(1.0, float(support_plane.sum())),
            "deleted_pixels": int(deleted.sum()),
            "target_pixels": target,
            "rect_components": rect["components"],
            "rect_mean_size": rect["deleted_pixels"] / max(1, rect["components"]),
            "scatter_components": scat["components"],
            "scatter_mean_size": scat["deleted_pixels"] / max(1, scat["components"]),
            "trim_fraction": window_stats.get("trim_fraction", 0.0),
            "windows": window_stats.get("windows", 0.0),
        }
    ok_dominance = all(
        v["deleted_pixels"] == v["target_pixels"]
        and v["rect_mean_size"] > 20.0 * max(1.0, v["scatter_mean_size"])
        and v["trim_fraction"] < 0.05
        for v in dominance.values())
    record("holes_are_rectangle_dominated", ok_dominance,
           "; ".join("%s exact=%d windows=%d rectMean=%.0f vs scatterMean=%.2f trim=%.4f"
                     % (k, v["target_pixels"], v["windows"], v["rect_mean_size"],
                        v["scatter_mean_size"], v["trim_fraction"])
                     for k, v in dominance.items()))
    report["hole_dominance"] = dominance

    # R4: the local value shift replaces the outlier pixels instead of adding to
    # them, leaves O untouched and changes all three Depth channels identically.
    _d, obs_shift, info_shift = apply_condition("local_value_shift", modal_x, support, obs,
                                               12345, sample_id, config)
    record("local_value_shift_keeps_observation", bool(torch.equal(obs_shift, support)),
           "non-zero corruption leaves O = S")
    record("local_value_shift_reports_replacement",
           "replaced_pixels" in info_shift and info_shift["replaced_pixels"] > 0,
           "replaced %d pixels inside the shifted region" % int(info_shift.get("replaced_pixels", 0)))
    record("localshift_alias_still_accepted",
           normalise_condition("localshift") == "local_value_shift",
           "legacy CLI spelling maps to the canonical condition")

    # R7: the evaluation padding band must carry S = 0 (and the RGB padding the
    # standardised zero), so a finite fill value cannot be read as an observation.
    from research.oc_rga_data import pad_to_multiple as pad_np

    tiny_support = pad_np(np.ones((20, 30), dtype=np.float32), 32, 0.0)
    tiny_rgb = pad_np(np.ones((20, 30, 3), dtype=np.float32), 32, 0.0)
    padding_ok = (
        tiny_support.shape == (32, 32)
        and tiny_rgb.shape == (32, 32, 3)
        and float(tiny_support[:20, :30].min()) == 1.0
        and float(tiny_support[20:, :].max()) == 0.0
        and float(tiny_support[:, 30:].max()) == 0.0
        and float(tiny_rgb[20:, :].max()) == 0.0
    )
    record("r7_padding_band_has_zero_support", padding_ok,
           "20x30 -> %s, real region S=1, band S=0" % (tuple(tiny_support.shape),))

    for name, result in results.items():
        confusion = torch.tensor(result["confusion"], dtype=torch.float64)
        if confusion.shape != (int(config.num_classes), int(config.num_classes)):
            record("confusion_shape_%s" % name, False, "%s" % (tuple(confusion.shape),))
            continue
        if confusion.sum().item() <= 0 and result["num_samples"] > 0:
            record("confusion_nonempty_%s" % name, False, "empty confusion for %s" % name)
    record("confusion_shapes_and_support",
           all(torch.tensor(r["confusion"]).shape == (int(config.num_classes), int(config.num_classes))
               and torch.tensor(r["confusion"]).sum().item() > 0 for r in results.values()),
           "25x25 integer confusions with pixels on every evaluated condition")
    record("difficulty_mean_available", difficulty_mean(results) is not None,
           "5-condition difficulty mean = %s" % difficulty_mean(results))

    report["results"] = {k: {"miou": v["miou"], "num_samples": v["num_samples"],
                             "realised_ratio": v["realised_ratio"]} for k, v in results.items()}
    report["difficulty_mean"] = difficulty_mean(results)
    report["failures"] = list(report["failures"])
    report["passed"] = not report["failures"]
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(sched.jsonable(report), handle, indent=2)
        print("[eval-self-check] wrote %s" % report_path)
    print("[eval-self-check] %s (%d checks)" % ("ALL PASS" if report["passed"] else "FAILURES",
                                                len(report["checks"])))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OC-RGA evaluation (shared 640 condition protocol)")
    parser.add_argument("--config", default="local_configs.research.OC_RGA_DeLiVER")
    parser.add_argument("--checkpoint", default=None,
                        help="required unless --self-check is used")
    parser.add_argument("--self-check", action="store_true",
                        help="protocol check on a few real images with a freshly built model")
    parser.add_argument("--limit", type=int, default=3, help="images per condition in --self-check")
    parser.add_argument("--report", default=None)
    parser.add_argument("--split", choices=("dev", "test"), default="dev",
                        help="dev maps to the official val list; test is final-only")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--eval-long-side", type=int, default=640)
    parser.add_argument("--ham-seed", type=int, default=12345)
    parser.add_argument("--corruption-seed", type=int, default=12345)
    parser.add_argument("--model-mode", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args(argv)
    if args.self_check:
        report = run_self_check(config_name=args.config, device=args.device, limit=args.limit,
                                model_mode=args.model_mode or "masked_baseline",
                                report_path=args.report)
        return 0 if report["passed"] else 1
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --self-check is used")
    report = evaluate_checkpoint(
        args.config, args.checkpoint, split=args.split, conditions=args.conditions,
        eval_long_side=args.eval_long_side, ham_seed=args.ham_seed,
        corruption_seed=args.corruption_seed, device=args.device,
        max_batches=args.max_batches, model_mode=args.model_mode,
        num_workers=args.num_workers, out_dir=args.out_dir, amp=args.amp,
    )
    return 0 if report["results"] else 1


if __name__ == "__main__":
    sys.exit(main())
