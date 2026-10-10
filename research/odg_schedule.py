"""ODG research support: batching/schedule math, experiment contract and small
runtime helpers shared by ``research/train_odg.py`` and ``research/evaluate_odg.py``.

Nothing here is imported by the author's ``utils/train.py`` and nothing here
modifies the author model, data pipeline or configs.  The batching/schedule and
contract helpers (sections 1-2) are plain Python; the runtime helpers
(section 3) depend on torch and on the sibling ``research/data.py`` layer.

Key conventions
---------------
* One optimizer *attempt* == one call to ``optimizer.step`` / ``scaler.step``.
  The learning-rate schedule advances by attempt, not by successful update, so
  AMP-skipped steps are never hidden.  AMP skips are counted separately, and at
  every optimizer boundary the *unscaled* gradient of every trainable parameter
  is checked for finiteness.
* ``updates_per_epoch = ceil(num_samples / effective_batch)``; the dataset is
  cycled in the loader so every optimizer attempt accumulates exactly
  ``accum_steps`` full micro-batches (a few trailing samples are repeated).
* ``schedule_epochs`` fixes the whole learning-rate schedule.  A stage stop
  (``stop_after_epochs``) pauses training but never compresses the schedule.
* Artificial evaluation holes delete observed support *and* fill the same depth
  pixels with the canonical raw-zero normalised value (``-0.48/0.28``), so every
  geometry mode receives an identical input; the requested ratio is relative to
  the true support domain and the realised deleted fraction is recorded.
* Multi-scale fusion matches the author's rounding and accumulates per-view
  softmax maps in FP32, even under FP16 autocast.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import inspect
import json
import math
import os
import random
import time

import numpy as np
import torch

from utils.lr_policy import WarmUpPolyLR

DEFAULT_EFFECTIVE_BATCH = 16
CHECKPOINT_FORMAT = "odg-research-checkpoint-v1"
EPS = 1e-12

# Fixed semantics of the research training entry (recorded in the contract).
# ``eval``: periodic dev validation runs with ``model.eval()`` and must not touch
# BatchNorm running statistics.  ``all_trainable``: the optimizer receives every
# trainable tensor exactly once.
VALIDATION_MODE = "eval"
OPTIMIZER_PARAM_SCOPE = "all_trainable"

# Modules whose parameters are never weight-decayed (author convention).
NORM_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.GroupNorm,
    torch.nn.LayerNorm,
)

# Canonical author depth normalisation (``(D8/255 - 0.48) / 0.28``), so a raw
# depth zero maps to ``-0.48 / 0.28`` in the normalised domain.  This is the
# value the author ``ValPre`` writes into the padded band; artificial holes use
# the same canonical fill instead of inventing a new "missing" value.
# Must stay equal to ``research.data.DEPTH_RAW_ZERO_NORM`` (checked at runtime).
DEPTH_NORM_MEAN = 0.48
DEPTH_NORM_STD = 0.28
DEPTH_RAW_ZERO_NORM = (0.0 - DEPTH_NORM_MEAN) / DEPTH_NORM_STD  # -0.48 / 0.28

# A resume is only allowed from a full epoch-boundary snapshot.  ``smoke.pth``
# is taken mid-epoch (after N optimizer attempts) and must never be resumed.
RESUMABLE_SNAPSHOTS = ("epoch", "stage", "best")

# Fields that must stay identical when resuming the *same* experiment.
CONTRACT_FIELDS = (
    "geometry_mode",
    "dataset_name",
    "backbone",
    "decoder",
    "decoder_embed_dim",
    "num_classes",
    "schedule_epochs",
    "warmup_epochs",
    "lr",
    "lr_power",
    "weight_decay",
    "effective_batch",
    "micro_batch",
    "accum_steps",
    "seed",
    "norm_eval",
    "amp",
    "augmentation",
    "train_scale_array",
    "init_checkpoint",
    "image_height",
    "image_width",
    "train_source",
    "eval_source",
    "train_fingerprint",
    "eval_fingerprint",
    # Semantics fixed after the 2026-10-10 review.  These two fields exist so a
    # checkpoint trained before the fix can never be resumed by the fixed code:
    # the old runs validated in train mode and left 29 ``GeoPriorGen.weight``
    # tensors out of the optimizer, so their optimizer state is not comparable.
    "validation_mode",
    "optimizer_param_scope",
)


# --------------------------------------------------------------------------- #
# 1. Batching and learning-rate schedule
# --------------------------------------------------------------------------- #
def ceil_div(numerator, denominator):
    if int(denominator) <= 0:
        raise ValueError("denominator must be positive, got %r" % (denominator,))
    return -(-int(numerator) // int(denominator))


def plan_epoch(num_samples, micro_batch, accum_steps, effective_batch=DEFAULT_EFFECTIVE_BATCH):
    """Iteration counts for one epoch at a fixed effective batch size.

    The loader is fed ``file_length`` samples per epoch so that the author-style
    dataset cycling reproduces the trailing samples needed for the last full
    optimizer attempt.
    """
    num_samples = int(num_samples)
    micro_batch = int(micro_batch)
    accum_steps = int(accum_steps)
    effective_batch = int(effective_batch)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive, got %r" % (num_samples,))
    if micro_batch <= 0 or accum_steps <= 0:
        raise ValueError("micro_batch and accum_steps must be positive")
    if micro_batch * accum_steps != effective_batch:
        raise ValueError(
            "micro_batch (%d) * accum_steps (%d) = %d does not match the "
            "declared effective_batch (%d)" % (micro_batch, accum_steps, micro_batch * accum_steps, effective_batch)
        )
    updates_per_epoch = ceil_div(num_samples, effective_batch)
    micro_batches_per_epoch = updates_per_epoch * accum_steps
    samples_per_epoch = micro_batches_per_epoch * micro_batch
    return {
        "num_samples": num_samples,
        "micro_batch": micro_batch,
        "accum_steps": accum_steps,
        "effective_batch": effective_batch,
        "updates_per_epoch": updates_per_epoch,
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "samples_per_epoch": samples_per_epoch,
        "file_length": samples_per_epoch,
        "repeated_samples": samples_per_epoch - num_samples,
    }


def build_schedule(updates_per_epoch, schedule_epochs, warmup_epochs, start_lr, lr_power, stop_after_epochs=None):
    """Build the author's WarmUpPolyLR over the *whole* schedule.

    ``stop_after_epochs`` is accepted only to document the intent and is
    deliberately **not** used: an early stage stop must not shrink the schedule.
    """
    total_updates = int(updates_per_epoch) * int(schedule_epochs)
    warmup_updates = int(updates_per_epoch) * int(warmup_epochs)
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    policy = WarmUpPolyLR(start_lr, lr_power, total_updates, warmup_updates)
    return policy, {
        "total_updates": total_updates,
        "warmup_updates": warmup_updates,
        "updates_per_epoch": int(updates_per_epoch),
        "schedule_epochs": int(schedule_epochs),
        "warmup_epochs": int(warmup_epochs),
        "stop_after_epochs": None if not stop_after_epochs else int(stop_after_epochs),
    }


def lr_for_attempt(policy, attempt_index, total_updates):
    """LR used *before* the (0-based) attempt's ``optimizer.step``.

    The author loop applies the previous iteration's schedule value, so index 0
    would otherwise give lr=0; we shift by one attempt to get the intended warmup
    shape (first attempt = start_lr/warmup) without changing the total budget.
    """
    index = min(int(attempt_index) + 1, int(total_updates))
    return policy.get_lr(index)


def schedule_report(policy, total_updates, warmup_updates, points=None):
    """Human-checkable (attempt, lr) rows for a CPU-only schedule check."""
    if points is None:
        points = sorted(
            {
                0,
                1,
                max(warmup_updates // 2, 0),
                max(warmup_updates - 1, 0),
                warmup_updates,
                total_updates // 4,
                total_updates // 2,
                (3 * total_updates) // 4,
                total_updates - 1,
            }
        )
    rows = []
    for attempt in points:
        if attempt < 0 or attempt >= max(total_updates, 1):
            continue
        rows.append({"attempt": int(attempt), "lr": lr_for_attempt(policy, attempt, total_updates)})
    return rows


def set_param_group_lr(optimizer, lr):
    """Set every param group to the scheduled learning rate.

    Returns the actual per-group lr values that were written, so they can be
    logged even if a future grouping multiplier is introduced.
    """
    written = []
    for group in optimizer.param_groups:
        multiplier = float(group.get("odg_lr_multiplier", 1.0))
        value = float(lr) * multiplier
        group["lr"] = value
        written.append(value)
    return written


def param_group_lrs(optimizer):
    return [float(g["lr"]) for g in optimizer.param_groups]


# --------------------------------------------------------------------------- #
# 2. Manifest fingerprints, experiment contract, RNG, checkpoint io
# --------------------------------------------------------------------------- #
def file_fingerprint(path):
    """Cheap identity of a split list: byte sha1 plus line count."""
    path = str(path)
    with open(path, "rb") as handle:
        data = handle.read()
    lines = data.count(b"\n")
    if data and not data.endswith(b"\n"):
        lines += 1
    return {"path": path, "lines": int(lines), "sha1": hashlib.sha1(data).hexdigest()}


def manifest_lines(path, fallback=None):
    if os.path.exists(str(path)):
        return file_fingerprint(path)["lines"]
    if fallback is None:
        raise FileNotFoundError("manifest not found: %s" % path)
    return int(fallback)


def make_contract(config, geometry_mode, train_source, eval_source, micro_batch, accum_steps, effective_batch,
                  schedule_epochs, warmup_epochs, amp=True):
    train_fp = file_fingerprint(train_source) if os.path.exists(str(train_source)) else None
    eval_fp = file_fingerprint(eval_source) if os.path.exists(str(eval_source)) else None
    scale_array = getattr(config, "train_scale_array", None)
    init_checkpoint = getattr(config, "pretrained_model", None)
    return {
        "geometry_mode": str(geometry_mode),
        "dataset_name": str(config.dataset_name),
        "backbone": str(config.backbone),
        "decoder": str(getattr(config, "decoder", "")),
        "decoder_embed_dim": int(getattr(config, "decoder_embed_dim", 0) or 0),
        "num_classes": int(config.num_classes),
        "schedule_epochs": int(schedule_epochs),
        "warmup_epochs": int(warmup_epochs),
        "lr": float(config.lr),
        "lr_power": float(config.lr_power),
        "weight_decay": float(config.weight_decay),
        "effective_batch": int(effective_batch),
        "micro_batch": int(micro_batch),
        "accum_steps": int(accum_steps),
        "seed": int(config.seed),
        "norm_eval": bool(getattr(config, "norm_eval", False)),
        "amp": bool(amp),
        "augmentation": "mirror+random_scale+random_crop_pad",
        "train_scale_array": [float(v) for v in scale_array] if scale_array is not None else None,
        # Initialisation source identity is the file name, so a resume works
        # across machines with a different absolute repo root; the full path is
        # kept for the record only.
        "init_checkpoint": os.path.basename(str(init_checkpoint)) if init_checkpoint else None,
        "init_checkpoint_path": str(init_checkpoint) if init_checkpoint else None,
        "image_height": int(config.image_height),
        "image_width": int(config.image_width),
        # Split identity is the file *name* plus its content sha1, so a resume
        # works across machines with different absolute dataset roots while a
        # changed split list is still detected.  Full paths are kept for the record.
        "train_source": os.path.basename(str(train_source)),
        "eval_source": os.path.basename(str(eval_source)),
        "train_fingerprint": train_fp["sha1"] if train_fp else "missing",
        "eval_fingerprint": eval_fp["sha1"] if eval_fp else "missing",
        "train_source_path": str(train_source),
        "eval_source_path": str(eval_source),
        # Fixed validation and optimizer semantics (see CONTRACT_FIELDS).
        "validation_mode": VALIDATION_MODE,
        "optimizer_param_scope": OPTIMIZER_PARAM_SCOPE,
    }


def contract_diff(saved, current):
    diffs = []
    for field in CONTRACT_FIELDS:
        if saved.get(field) != current.get(field):
            diffs.append("%s: saved=%r current=%r" % (field, saved.get(field), current.get(field)))
    return diffs


def seed_everything(seed):
    """Normal fixed seed; intentionally not bit-deterministic."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        tensor = state["torch"]
        torch.set_rng_state(tensor.cpu() if torch.is_tensor(tensor) else tensor)
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(str(path)))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)
    return str(path)


def save_training_checkpoint(path, model, optimizer, scaler, state, config, args, contract,
                             snapshot_kind="epoch"):
    """Save a full training snapshot.

    ``snapshot_kind`` is one of ``epoch`` | ``stage`` | ``best`` | ``smoke``.
    Only epoch-boundary snapshots (everything except ``smoke``) are resumable;
    ``smoke`` is taken after N optimizer attempts inside an epoch.
    """
    if snapshot_kind not in ("epoch", "stage", "best", "smoke"):
        raise ValueError("unknown snapshot_kind %r" % (snapshot_kind,))
    payload = {
        "format": CHECKPOINT_FORMAT,
        "snapshot_kind": str(snapshot_kind),
        "epoch_boundary": str(snapshot_kind) in RESUMABLE_SNAPSHOTS,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "completed_epochs": int(state["completed_epochs"]),
        "attempt_updates": int(state["attempt_updates"]),
        "applied_updates": int(state["applied_updates"]),
        "amp_skipped_updates": int(state["amp_skipped_updates"]),
        "micro_batches_seen": int(state.get("micro_batches_seen", 0)),
        "best_dev_miou": float(state["best_dev_miou"]),
        "geometry_mode": str(state["geometry_mode"]),
        "num_train_samples": int(state.get("num_train_samples", 0)),
        "contract": contract,
        "config": jsonable(config),
        "args": jsonable(vars(args)),
        "rng": capture_rng(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    directory = os.path.dirname(os.path.abspath(str(path)))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, str(path))
    return str(path)


def load_training_checkpoint(path, map_location="cpu"):
    # weights_only=False: our checkpoints carry Python/NumPy RNG state, and all
    # project checkpoints are trusted local/cloud artifacts.
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not an ODG research checkpoint (format mismatch): %s" % path)
    return payload


class CsvWriter:
    """Append-only CSV writer that writes the header only for a new file."""

    def __init__(self, path, header):
        self.path = str(path)
        self.header = list(header)

    def write(self, row):
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        is_new = not os.path.exists(self.path)
        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(self.header)
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# 3. Runtime helpers (torch + research/data.py)
# --------------------------------------------------------------------------- #
def import_data_module(config=None):
    """Import the data module selected by ``config.data_module``.

    A config without ``data_module`` keeps resolving to ``research.data``, so the
    SUN/NYU behaviour is unchanged.  ``local_configs.research.DFormerv2_S_DeLiVER``
    sets ``data_module = "research.deliver"``, which exposes the same
    ``ObservationDataset`` / ``ObservationTrainPre`` / ``ObservationValPre`` names
    the factories below use.
    """
    from importlib import import_module

    module_name = str(getattr(config, "data_module", "research.data") or "research.data")
    return import_module(module_name)


def resolve_sources(config, fulltrain):
    if fulltrain:
        return str(config.full_train_source), str(config.full_test_source)
    return str(config.train_dev_source), str(config.dev_eval_source)


def dataset_setting(config, train_source, eval_source):
    setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": str(train_source),
        "eval_source": str(eval_source),
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }
    mask_root = getattr(config, "support_mask_root", None)
    if mask_root:
        setting["support_mask_root"] = str(mask_root)
    return setting


def build_train_loader(config, train_source, eval_source, micro_batch, accum_steps, effective_batch, num_workers, seed):
    data = import_data_module(config)
    setting = dataset_setting(config, train_source, eval_source)
    num_samples = manifest_lines(train_source, fallback=getattr(config, "num_train_imgs", None))
    plan = plan_epoch(num_samples, micro_batch, accum_steps, effective_batch)

    preprocess = data.ObservationTrainPre(
        config.norm_mean, config.norm_std, bool(config.x_is_single_channel), config
    )
    dataset = data.ObservationDataset(setting, "train", preprocess, plan["file_length"])

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(micro_batch),
        shuffle=True,
        num_workers=int(num_workers),
        drop_last=True,
        pin_memory=True,
        generator=generator,
        persistent_workers=int(num_workers) > 0,
    )
    return loader, plan


def build_eval_loader(config, eval_source, train_source, batch_size, num_workers):
    data = import_data_module(config)
    setting = dataset_setting(config, train_source, eval_source)
    preprocess = data.ObservationValPre(
        config.norm_mean, config.norm_std, bool(config.x_is_single_channel), config
    )
    dataset = data.ObservationDataset(setting, "val", preprocess, None)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
    )
    return loader, len(dataset)


def unwrap_batch(batch):
    """Return (data, label, modal_x, depth_support, fn) from a batch."""
    if isinstance(batch, dict):
        return (
            batch.get("data"),
            batch.get("label"),
            batch.get("modal_x"),
            batch.get("depth_support"),
            batch.get("fn"),
        )
    if isinstance(batch, (list, tuple)) and len(batch) >= 4:
        return batch[0], batch[1], batch[2], batch[3], (batch[4] if len(batch) > 4 else None)
    raise TypeError("unexpected batch container: %r" % type(batch))


def supports_depth_support(model):
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return True
    if "depth_support" in signature.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def decoder_norm_layers(model, norm_layer=torch.nn.BatchNorm2d):
    """Yield ``(name, module)`` for the decoder norm layers the author init writes to."""
    heads = [("decode_head", getattr(model, "decode_head", None)),
             ("aux_head", getattr(model, "aux_head", None))]
    for head_name, head in heads:
        if head is None:
            continue
        for name, module in head.named_modules():
            if isinstance(module, norm_layer):
                yield ("%s.%s" % (head_name, name) if name else head_name, module)


def restore_decoder_norm_config(model, config, norm_layer=torch.nn.BatchNorm2d):
    """Restore the decoder norm numerics that a ``state_dict`` cannot carry.

    ``EncoderDecoder`` only calls ``init_weights`` when a criterion is passed, and
    that call is what writes ``cfg.bn_eps`` / ``cfg.bn_momentum`` into the decoder
    norm layers (``utils/init_func.__init_weight``).  Training therefore trained
    and validated with those values, while ``state_dict`` restores tensors only:
    a fresh evaluation model would silently keep PyTorch's defaults.

    This mirrors *only* the numeric part of the author init, on exactly the same
    modules (``decode_head`` and, when present, ``aux_head``).  It never
    re-initialises weights or affine parameters, never touches running statistics
    and never loads the backbone, so the checkpoint stays authoritative.

    Returns one ``"name: eps=<e> momentum=<m>"`` string per layer for the record.
    """
    restored = []
    for name, module in decoder_norm_layers(model, norm_layer):
        module.eps = float(config.bn_eps)
        module.momentum = float(config.bn_momentum)
        restored.append("%s: eps=%g momentum=%g" % (name, module.eps, module.momentum))
    return restored


def forward_logits(model, rgb, modal_x, depth_support=None, use_support=False):
    if use_support and depth_support is not None:
        return model(rgb, modal_x, depth_support=depth_support)
    return model(rgb, modal_x)


@contextlib.contextmanager
def evaluation_mode(model):
    """Run the block with the model in eval mode, always restoring train mode.

    Periodic dev validation must use the trained running statistics and must not
    update them, so the validation phase is switched to ``model.eval()`` and the
    previous mode is restored afterwards (training mode for the training loop).
    """
    was_training = bool(model.training)
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def batchnorm_running_state(model):
    """Fingerprint of the BatchNorm running statistics of the whole model.

    Used to prove that a validation pass did not update them: with the model in
    eval mode the digest must be identical before and after the pass.
    """
    modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    return {
        "layers": len(modules),
        "batches": sum(int(m.num_batches_tracked.item()) for m in modules
                       if m.num_batches_tracked is not None),
        "running_mean_sum": round(sum(float(m.running_mean.sum()) for m in modules
                                      if m.running_mean is not None), 4),
        "running_var_sum": round(sum(float(m.running_var.sum()) for m in modules
                                     if m.running_var is not None), 4),
    }


def classify_trainable_parameters(model):
    """Split every trainable tensor into ``(decay, no_decay)`` lists of pairs.

    The author's ``utils/init_func.group_weight`` walks ``module.modules()`` and
    tests ``isinstance(m, nn.Parameter)``, which can never match: bare
    ``nn.Parameter`` attributes were therefore silently dropped from the
    optimizer (the 29 ``GeoPriorGen.weight`` geometry kernels).  This classifies
    by name and module type instead, keeping the author's split (biases and
    parameters of norm layers are not decayed, other weights are) while no
    trainable tensor can be left out.
    """
    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = "%s.%s" % (module_name, param_name) if module_name else param_name
            if param_name.endswith("bias") or isinstance(module, NORM_MODULE_TYPES):
                no_decay.append((full_name, param))
            else:
                decay.append((full_name, param))
    return decay, no_decay


def build_optimizer_param_groups(model, lr, weight_decay=None):
    """AdamW parameter groups that contain every trainable tensor exactly once."""
    decay, no_decay = classify_trainable_parameters(model)
    groups = [
        {"params": [p for _, p in decay], "lr": float(lr)},
        {"params": [p for _, p in no_decay], "weight_decay": 0.0, "lr": float(lr)},
    ]
    if weight_decay is not None:
        groups[0]["weight_decay"] = float(weight_decay)
    return groups


def optimizer_parameter_report(model, optimizer):
    """Verify the optimizer holds exactly the trainable tensors, once each.

    Raises instead of returning a report when a tensor is missing, duplicated or
    not trainable, so a silent grouping regression can never start a long run.
    """
    optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
    optimizer_ids = [id(p) for p in optimizer_params]
    unique_ids = set(optimizer_ids)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    trainable_ids = {id(p) for p in trainable.values()}
    report = {
        "trainable_tensors": len(trainable),
        "optimizer_tensors": len(optimizer_params),
        "unique_optimizer_tensors": len(unique_ids),
        "duplicates": len(optimizer_ids) - len(unique_ids),
        "missing": sorted(name for name, p in trainable.items() if id(p) not in unique_ids),
        "not_trainable": len(unique_ids - trainable_ids),
        "group_sizes": [len(group["params"]) for group in optimizer.param_groups],
    }
    if (report["missing"] or report["duplicates"] or report["not_trainable"]
            or report["trainable_tensors"] != len(unique_ids)):
        raise RuntimeError("optimizer parameter grouping is not exact: %s" % report)
    return report


def forward_loss(model, rgb, modal_x, label, depth_support=None, use_support=False):
    if use_support and depth_support is not None:
        return model(rgb, modal_x, label, depth_support=depth_support)
    return model(rgb, modal_x, label)


class SegMetrics:
    """Confusion-matrix accumulator; mirrors the author's metric definitions."""

    def __init__(self, num_classes, ignore_index, device):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.hist = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64, device=device)

    def update(self, probs, labels):
        pred = probs.argmax(dim=1)
        keep = labels != self.ignore_index
        index = labels[keep].long() * self.num_classes + pred[keep].long()
        counts = torch.bincount(index, minlength=self.num_classes ** 2)
        self.hist += counts.view(self.num_classes, self.num_classes).to(self.hist.dtype)

    def compute(self):
        hist = self.hist
        diag = torch.diag(hist)
        row = hist.sum(1)
        col = hist.sum(0)
        iou = torch.where(col + row - diag > 0, diag / (col + row - diag).clamp_min(1.0), torch.zeros_like(diag))
        acc = torch.where(row > 0, diag / row.clamp_min(1.0), torch.zeros_like(diag))
        f1 = torch.where(col + row > 0, 2 * diag / (col + row).clamp_min(1.0), torch.zeros_like(diag))
        return {
            "miou": round(float(iou.mean().item()) * 100, 2),
            "macc": round(float(acc.mean().item()) * 100, 2),
            "mf1": round(float(f1.mean().item()) * 100, 2),
            "iou": [round(float(v) * 100, 2) for v in iou.detach().cpu().tolist()],
            "acc": [round(float(v) * 100, 2) for v in acc.detach().cpu().tolist()],
            "f1": [round(float(v) * 100, 2) for v in f1.detach().cpu().tolist()],
        }


def resolve_hole_fill_value():
    """Canonical raw-zero normalised depth used to fill artificial holes.

    Kept in sync with ``research.data.DEPTH_RAW_ZERO_NORM``; if the data module
    can be imported its value wins, otherwise the identical local constant is
    used.  Raises if the two ever disagree, so holes can never silently fill a
    different value.
    """
    try:
        data = import_data_module()
        value = float(data.DEPTH_RAW_ZERO_NORM)
    except Exception:
        return DEPTH_RAW_ZERO_NORM
    if abs(value - DEPTH_RAW_ZERO_NORM) > 1e-9:
        raise RuntimeError(
            "research.data.DEPTH_RAW_ZERO_NORM (%r) != expected %r"
            % (value, DEPTH_RAW_ZERO_NORM)
        )
    return value


def apply_artificial_holes(support, modal_x, ratio, seed, start_index=0, fill_value=DEPTH_RAW_ZERO_NORM):
    """Remove exactly the requested fraction of the rectangular image support.

    A rectangle with an optional partial final row is a connected hole. Padding
    is never removed or filled. SUN v1 support is rectangular because grayscale
    zero is retained and no natural missing-value convention has been established.
    An irregular support domain is rejected rather than silently changing severity.
    The original and distribution modes receive the SAME filled depth tensor.
    """
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError("Artificial hole ratio must be in [0,1]")
    if support is None or modal_x is None:
        raise ValueError("Artificial holes need both explicit support and depth")
    support, modal_x = support.clone(), modal_x.clone()
    boxes, per_sample = [], []
    total_support = total_deleted = 0
    for b in range(support.shape[0]):
        observed = support[b, 0] > 0
        count = int(observed.sum().item())
        target = int(round(float(ratio) * count))
        box, tail = None, 0
        if target:
            rows = observed.any(dim=1).nonzero().flatten()
            cols = observed.any(dim=0).nonzero().flatten()
            top, bottom = int(rows[0]), int(rows[-1]) + 1
            left, right = int(cols[0]), int(cols[-1]) + 1
            height, width = bottom - top, right - left
            if count != height * width:
                raise ValueError("SUN v1 holes require rectangular image support; irregular masks need an explicit protocol")
            rng = np.random.RandomState([int(seed) & 0x7FFFFFFF, int(start_index) + b])
            min_width = max(1, (target + height - 1) // height)
            max_width = min(width, target)
            hole_w = int(rng.randint(min_width, max_width + 1))
            full_rows, tail = divmod(target, hole_w)
            hole_h = full_rows + bool(tail)
            y0 = top + int(rng.randint(0, height - hole_h + 1))
            x0 = left + int(rng.randint(0, width - hole_w + 1))
            deleted = torch.zeros_like(observed)
            deleted[y0:y0 + full_rows, x0:x0 + hole_w] = True
            if tail:
                deleted[y0 + full_rows, x0:x0 + tail] = True
            support[b, 0][deleted] = 0.0
            modal_x[b][:, deleted] = float(fill_value)
            box = (y0, x0, int(hole_h), hole_w)
        total_support += count
        total_deleted += target
        boxes.append(box)
        per_sample.append({"index": int(start_index) + b, "support_pixels": count,
                           "deleted_pixels": target, "actual_ratio": target / count if count else 0.0,
                           "box": box, "partial_final_row_pixels": tail})
    return support, modal_x, {"requested_ratio": float(ratio), "support_pixels": total_support,
                              "deleted_pixels": total_deleted,
                              "actual_ratio": total_deleted / total_support if total_support else 0.0,
                              "per_sample": per_sample, "boxes": boxes}


def multi_scale_flip_probs(model, data, modal_x, support, use_support, scales, flip, out_size):
    """Five-scale (+ optional flip) fusion by summing per-view softmax maps.

    Fusion is identical to the author's ``evaluate_msf``: the softmax
    probabilities of all views are summed and only then argmaxed.  The pipeline
    matches the author's rounding (``int(scale*H)`` first, then round up to a
    multiple of 32) and the sum is accumulated in FP32, so FP16 autocast can
    never accumulate rounding error across views.  The support mask is resized
    with nearest-neighbour so a scale change never invents fractional validity.
    """
    height, width = out_size
    total = None
    for scale in scales:
        scaled_h = int(float(scale) * height)
        scaled_w = int(float(scale) * width)
        new_h = int(math.ceil(scaled_h / 32)) * 32
        new_w = int(math.ceil(scaled_w / 32)) * 32
        images = torch.nn.functional.interpolate(data, size=(new_h, new_w), mode="bilinear", align_corners=True)
        modal = torch.nn.functional.interpolate(modal_x, size=(new_h, new_w), mode="bilinear", align_corners=True)
        sup = None
        if support is not None:
            sup = torch.nn.functional.interpolate(support, size=(new_h, new_w), mode="nearest")
        logits = forward_logits(model, images, modal, sup, use_support)
        logits = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
        probs = logits.float().softmax(dim=1)
        total = probs if total is None else total + probs

        if flip:
            images_f = torch.flip(images, dims=(3,))
            modal_f = torch.flip(modal, dims=(3,))
            sup_f = None if sup is None else torch.flip(sup, dims=(3,))
            logits = forward_logits(model, images_f, modal_f, sup_f, use_support)
            logits = torch.flip(logits, dims=(3,))
            logits = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
            total = total + logits.float().softmax(dim=1)
    return total


@torch.no_grad()
def evaluate_loader(model, loader, config, device, use_support, msf=False, scales=(0.5, 0.75, 1.0, 1.25, 1.5),
                    flip=False, hole_ratio=0.0, hole_seed=0, amp=False, log_every=0, max_batches=0,
                    prediction_dir=None, prediction_limit=0):
    """Single- or multi-scale evaluation over an observation loader.

    Artificial holes (``hole_ratio`` > 0) remove observed support pixels and fill
    the same depth locations with the canonical raw-zero normalised value, so the
    input is identical for every geometry mode.  The realised deleted fraction of
    the true support domain is reported; a support-only deletion is never counted
    as missing-depth robustness.
    """
    metrics = SegMetrics(int(config.num_classes), int(config.background), device)
    samples = 0
    saved = 0
    total_batches = len(loader)
    hole_stats = {"requested_ratio": float(hole_ratio), "support_pixels": 0, "deleted_pixels": 0,
                  "actual_ratio": 0.0, "batches": 0}
    hole_fill = resolve_hole_fill_value() if float(hole_ratio) > 0.0 else None
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        data, label, modal_x, support, fn = unwrap_batch(batch)
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        modal_x = modal_x.to(device, non_blocking=True)
        if support is not None:
            support = support.to(device, non_blocking=True)
        batch_size, height, width = label.shape
        if float(hole_ratio) > 0.0:
            support, modal_x, info = apply_artificial_holes(
                support, modal_x, hole_ratio, hole_seed, start_index=samples, fill_value=hole_fill)
            hole_stats["batches"] += 1
            hole_stats["support_pixels"] += info["support_pixels"]
            hole_stats["deleted_pixels"] += info["deleted_pixels"]
            print("[evaluate] artificial holes ratio=%.3f seed=%d fill=%.6f deleted=%d/%d (actual=%.4f) boxes=%s"
                  % (float(hole_ratio), int(hole_seed), float(hole_fill), info["deleted_pixels"],
                     info["support_pixels"], info["actual_ratio"], info["boxes"][:2]))

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp
            else contextlib.nullcontext()
        )
        with autocast:
            if msf:
                probs = multi_scale_flip_probs(model, data, modal_x, support, use_support, scales, flip, (height, width))
            else:
                logits = forward_logits(model, data, modal_x, support, use_support)
                if tuple(logits.shape[-2:]) != (height, width):
                    logits = torch.nn.functional.interpolate(
                        logits, size=(height, width), mode="bilinear", align_corners=False
                    )
                # Accumulate/compare probabilities in FP32 even under FP16 autocast.
                probs = logits.float().softmax(dim=1)
        metrics.update(probs, label)
        if prediction_dir and prediction_limit and saved < int(prediction_limit):
            for b in range(batch_size):
                if saved >= int(prediction_limit):
                    break
                save_prediction_png(config.dataset_name, probs[b].argmax(0), fn[b] if fn is not None else "", prediction_dir)
                saved += 1
        samples += batch_size
        if log_every and (step + 1) % int(log_every) == 0:
            print("[evaluate] %d/%d batches" % (step + 1, total_batches))
    result = metrics.compute()
    result["num_samples"] = int(samples)
    result["num_batches"] = int(total_batches)
    result["saved_predictions"] = int(saved)
    if float(hole_ratio) > 0.0:
        hole_stats["actual_ratio"] = (float(hole_stats["deleted_pixels"]) / hole_stats["support_pixels"]
                                      if hole_stats["support_pixels"] else 0.0)
    result["artificial_holes"] = hole_stats
    return result


def save_prediction_png(dataset_name, pred_mask, fn, out_dir):
    """Save an argmax prediction using the author's NYU/SUN colour map."""
    import cv2

    os.makedirs(str(out_dir), exist_ok=True)
    name = os.path.basename(str(fn)) if fn else "sample"
    name = os.path.splitext(name)[0]
    path = os.path.join(str(out_dir), "%s_pred.png" % name)
    mask = pred_mask.detach().cpu().numpy().astype(np.int64)
    palette_path = os.path.join("utils", "nyucmap.npy")
    if str(dataset_name) in ("NYUDepthv2", "SUNRGBD") and os.path.exists(palette_path):
        palette = np.load(palette_path)
        if int(mask.max()) < len(palette):
            colored = palette[mask]
            cv2.imwrite(path, cv2.cvtColor(colored.astype(np.uint8), cv2.COLOR_RGB2BGR))
            return path
    # Fallback: raw label ids scaled for a quick visual check.
    cv2.imwrite(path, (mask * (255 // max(1, int(mask.max()) + 1))).astype(np.uint8))
    return path


def optimizer_state_step(optimizer):
    """Number of optimizer steps applied (Adam/Amsgrad store an int/tensor 'step')."""
    total = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            step = state.get("step")
            if step is None:
                continue
            if torch.is_tensor(step):
                total += int(step.max().item()) if step.numel() > 1 else int(step.item())
            else:
                total += int(step)
    return total


def move_optimizer_state_to_device(optimizer):
    """Move every optimizer state tensor to its own parameter's device.

    ``Optimizer.load_state_dict`` casts state to the device of the parameters
    *at load time*.  If the model is moved to CUDA afterwards, the state stays on
    CPU and the first real step fails.  This makes the state follow the params
    regardless of load order.  Returns the number of tensors moved.
    """
    moved = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            for key, value in state.items():
                if torch.is_tensor(value) and value.device != param.device:
                    state[key] = value.to(param.device)
                    moved += 1
    return moved


def assert_optimizer_state_on_params(optimizer):
    """Fail loudly if any optimizer state tensor is on the wrong device."""
    bad = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            for key, value in state.items():
                if torch.is_tensor(value) and value.device != param.device:
                    bad.setdefault(key, (str(value.device), str(param.device)))
    if bad:
        raise RuntimeError(
            "optimizer state is on the wrong device (key: state_device/param_device): %s" % bad
        )


@torch.no_grad()
def grad_finiteness_report(model):
    """Check every trainable parameter's *actual* (unscaled) gradient.

    One aggregated device sync on the all-finite path; the per-parameter names
    are only resolved when something is non-finite.  Parameters with
    ``requires_grad`` but ``grad is None`` are recorded (a HAM/depth path may
    legitimately not receive gradient) but never treated as an error by
    themselves.
    """
    total = 0
    with_grad = 0
    no_grad = []
    flag = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total += 1
        grad = param.grad
        if grad is None:
            no_grad.append(name)
            continue
        with_grad += 1
        finite = torch.isfinite(grad).all()
        flag = finite if flag is None else (flag & finite)
    all_finite = bool(flag.item()) if flag is not None else True
    nonfinite = []
    if not all_finite:
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                nonfinite.append(name)
    return {
        "all_finite": all_finite,
        "num_trainable": int(total),
        "num_with_grad": int(with_grad),
        "no_grad": no_grad,
        "nonfinite": nonfinite,
    }


# Priority substrings for the parameter-change probe.  Covers the backbone
# attention Q projection and the geometry prior, plus decoder parameters; the
# probe is never a single small parameter.
PROBE_PRIORITY = ("q_proj", "Geo.weight", "Geo.decay", "patch_embed")


def select_probe_params(model, max_probes=8, decoder_probes=3):
    """Pick a small, named, cross-component set of parameters to watch."""
    trainable = {
        name: param for name, param in model.named_parameters()
        if param.requires_grad and param.numel() > 0
    }
    chosen = [name for name in trainable if any(key in name for key in PROBE_PRIORITY)]
    decoder = sorted((n for n in trainable if "decode_head" in n),
                     key=lambda n: trainable[n].numel(), reverse=True)[:decoder_probes]
    chosen.extend(decoder)
    if len(chosen) < max_probes:
        for name in sorted(trainable, key=lambda n: trainable[n].numel(), reverse=True):
            if name not in chosen:
                chosen.append(name)
            if len(chosen) >= max_probes:
                break
    chosen = chosen[:max_probes]
    return {name: trainable[name] for name in chosen}


@torch.no_grad()
def parameter_delta_report(probes, before):
    """Max-abs change per probed parameter since ``before`` (name -> tensor)."""
    report = {}
    changed = 0
    for name, param in probes.items():
        reference = before.get(name)
        if reference is None:
            continue
        delta = float((param.detach() - reference).abs().max().item())
        did_change = delta > 0.0
        changed += int(did_change)
        report[name] = {"numel": int(param.numel()), "max_abs_delta": delta, "changed": did_change}
    return report, changed
