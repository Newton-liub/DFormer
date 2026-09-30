"""A-v1 phase interfaces, not an authorized formal-training runner.

The caller creates one fresh observed batch per step, then this module replays
that batch with the existing project RNG snapshot/restore helpers. No evaluator,
checkpoint selector, data cache, or training loop is introduced.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from utils.init_func import build_e1_optimizer_param_groups
from utils.training_checkpoint import capture_rng_state, restore_rng_state

PHASES = ("proposal", "gate")


def configure_phase(model, phase, *, lr, weight_decay):
    """Freeze all C0 tensors/statistics and optimize exactly one A-v1 branch."""
    if phase not in PHASES or model.av1 is None:
        raise ValueError("A-v1 requires phase proposal or gate and an enabled module")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("invalid optimizer settings")
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    getattr(model.av1, phase).requires_grad_(True)
    model.av1_phase = phase
    model.train(True)  # EncoderDecoder.train keeps every base module in eval.
    prefix = f"av1.{phase}."
    # Reuse the audited Conv/Linear decay classification, but omit empty C0 groups.
    groups = build_e1_optimizer_param_groups(
        model, torch.nn.BatchNorm2d, base_lr=lr, new_lr=lr,
        weight_decay=weight_decay, new_parameter_prefixes=(prefix,),
        expected_geo_weight_count=0,
    )
    groups = [group for group in groups if group["params"]]
    expected = {id(parameter) for name, parameter in model.named_parameters() if name.startswith(prefix)}
    actual = [id(parameter) for group in groups for parameter in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("A-v1 optimizer membership mismatch")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_phase_batch(config, phase, rgb, depth, sample_ids, *, epoch, iteration,
                      niters_per_epoch, nepochs, global_rank=0):
    """Regenerate corruption with an isolated phase seed, never an old cache."""
    from utils.dataloader.mmfr_training_v3 import build_mmfr_training_batch_v3

    if phase not in PHASES:
        raise ValueError("unknown A-v1 phase")
    seeds = config.mmfr_av1["phase_corruption_seeds"]
    if int(seeds["proposal"]) == int(seeds["gate"]):
        raise ValueError("Proposal and Gate corruption streams must be isolated")
    corruption = config.mmfr_a2["corruption"]
    batch = build_mmfr_training_batch_v3(
        rgb, depth, sample_ids, epoch=epoch, iteration=iteration,
        niters_per_epoch=niters_per_epoch, nepochs=nepochs,
        rgb_mean=config.norm_mean, rgb_std=config.norm_std,
        corruption_seed=int(seeds[phase]), global_rank=global_rank,
        p_clean=float(corruption["p_clean"]), max_specs=int(corruption["max_specs"]),
        sample_id_root=config.dataset_path,
    )
    # Metadata is only a loss mask / reproduction record, never a gate input.
    batch["clean_mask"] = torch.tensor([item["clean"] for item in batch["metadata"]], dtype=torch.bool)
    return batch


def per_sample_ce(logits, labels, *, ignore_index=255):
    """Per-image mean over the exact same non-ignored label support."""
    valid = labels != ignore_index
    losses = F.cross_entropy(logits, labels.long(), ignore_index=ignore_index, reduction="none")
    return (losses * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)


def utility_targets(loss_off, loss_full, margin):
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("action-utility margin must be positive and finite")
    utility = (loss_off - loss_full).detach()
    if not torch.isfinite(utility).all():
        raise ValueError("non-finite action utility")
    mask = utility.abs() > margin
    targets = (utility > margin).to(utility.dtype).detach()
    return utility, targets, mask.detach()


def masked_gate_bce(gate, targets, mask):
    """An all-ambiguous batch gives a differentiable exact zero, without 0/0."""
    if not bool(mask.any().item()):
        return gate.sum() * 0.0
    return F.binary_cross_entropy(gate[mask], targets.detach()[mask])


def clean_consistency(reference, logits, labels, clean_mask, *, ignore_index=255):
    valid = (labels != ignore_index).to(logits.dtype)
    pixel_kl = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(reference.detach(), dim=1),
                        reduction="none").sum(1)
    sample_kl = (pixel_kl * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
    if not bool(clean_mask.any().item()):
        return logits.sum() * 0.0
    return sample_kl[clean_mask].mean()


def phase_loss(model, batch, labels, *, phase, margin, lambda_clean):
    """Sequential endpoints have no graphs; only the active branch has a graph.

    off/full/learned each start from the same pre-forward RNG snapshot. The global
    stream advances as one forward, because the final active pass consumes it once.
    """
    if phase not in PHASES or getattr(model, "av1_phase", None) != phase:
        raise ValueError("configure the matching A-v1 optimizer phase first")
    if not math.isfinite(lambda_clean) or lambda_clean < 0:
        raise ValueError("lambda_clean must be finite and nonnegative")
    if labels.requires_grad:
        raise ValueError("segmentation labels must be detached")
    rgb, depth = batch["rgb"], batch["depth"]
    observed, support = batch["raw_depth"], batch["valid_mask"]
    clean = batch["clean_mask"].to(device=labels.device, dtype=torch.bool)
    state = capture_rng_state()

    def forward(mode):
        restore_rng_state(state)
        return model(rgb, depth, av1_mode=mode, observed_depth=observed, geometry_mask=support)

    with torch.no_grad():
        off = forward("off")
        loss_off = per_sample_ce(off, labels)
        if phase == "gate":
            full = forward("full")
            loss_full = per_sample_ce(full, labels)
    if phase == "proposal":
        active = forward("full")
        ce = per_sample_ce(active, labels).mean()
        consistency = clean_consistency(off, active, labels, clean)
        total = ce + lambda_clean * consistency
        details = {"ce": ce.detach(), "clean_consistency": consistency.detach(), "off_detached": not off.requires_grad}
    else:
        utility, targets, mask = utility_targets(loss_off, loss_full, margin)
        # Capture the actual gate used by the learned pass, without a second backbone.
        gate_values = []
        handle = model.av1.gate.register_forward_hook(lambda _module, _args, output: gate_values.append(output.reshape(-1)))
        try:
            active = forward("learned")
        finally:
            handle.remove()
        if len(gate_values) != 1:
            raise RuntimeError("learned pass must produce exactly one per-image gate")
        bce = masked_gate_bce(gate_values[0], targets, mask)
        ce = per_sample_ce(active, labels).mean()
        consistency = clean_consistency(off, active, labels, clean)
        total = bce + ce + lambda_clean * consistency
        details = {
            "ce": ce.detach(), "bce": bce.detach(), "clean_consistency": consistency.detach(),
            "utility": utility, "targets": targets, "mask": mask, "gate": gate_values[0].detach(),
            "off_detached": not off.requires_grad, "full_detached": not full.requires_grad,
        }
    if not torch.isfinite(total):
        raise RuntimeError("non-finite A-v1 phase loss")
    return total, details
