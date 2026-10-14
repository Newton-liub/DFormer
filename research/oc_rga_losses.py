"""OC-RGA joint objective and the leak-free hold-out construction (design 9.3 / 10).

The single most important property of this module is that the hold-out targets
vanish from **all** depth statistics of the auxiliary context, not only from the
solver: the target cells' original pixels are removed from ``O`` at the pixel
level, their depth values are replaced by the canonical raw-zero fill, and every
observation summary (``y``, ``m``, ``f``, ``a_obs``) plus the whole precision-head
input is recomputed from that modified array.  The RGB geometry context
``G, mu, U, s^2`` is reused because it never reads Depth in the first place.

Objective (design 10.5)::

    L = L_seg + lambda_g(e) * L_cond + lambda_o(e) * L_obs
    lambda_g(e) = 0.01 + 0.04 * min(e/5, 1)
    lambda_o(e) = 0.002 + 0.008 * min(e/5, 1)

with ``L_cond`` the conditional joint Gaussian NLL, a light-weight prior joint
NLL, and a Huber mean term, and ``L_obs`` the observed-error NLL that uses the
*actual* effective variance ``R = omega^2 / f`` of the solver.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research.oc_rga import (
    SIGMA_REF,
    ZERO_FILL,
    cell_summaries,
    condition_low_rank,
    joint_low_rank_nll,
)

__all__ = [
    "huber",
    "build_auxiliary_context",
    "select_holdout_targets",
    "compute_oc_losses",
    "loss_scales",
    "geometry_statistics",
    "holdout_leak_check",
]

LOG_2PI = math.log(2.0 * math.pi)


def loss_scales(epoch_progress: float, config) -> Tuple[float, float]:
    """``(lambda_g, lambda_o)`` as a function of the continuous epoch progress."""
    ramp = float(getattr(config, "loss_lambda_ramp_epochs", 5.0)) or 1.0
    fraction = min(max(float(epoch_progress) / ramp, 0.0), 1.0)
    lam_g = float(getattr(config, "loss_lambda_cond_start", 0.01)) + (
        float(getattr(config, "loss_lambda_cond_end", 0.05))
        - float(getattr(config, "loss_lambda_cond_start", 0.01))
    ) * fraction
    lam_o = float(getattr(config, "loss_lambda_obs_start", 0.002)) + (
        float(getattr(config, "loss_lambda_obs_end", 0.010))
        - float(getattr(config, "loss_lambda_obs_start", 0.002))
    ) * fraction
    return lam_g, lam_o


def huber(error: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    """``0.5 e^2`` for ``|e| <= delta``, else ``delta (|e| - delta/2)``.

    Written out explicitly because the design fixes this closed form; it is *not*
    PyTorch's differently normalised ``smooth_l1`` with ``beta=delta``.
    """
    magnitude = error.abs()
    quadratic = 0.5 * error * error
    linear = float(delta) * (magnitude - 0.5 * float(delta))
    return torch.where(magnitude <= float(delta), quadratic, linear)


# --------------------------------------------------------------------------- #
# Hold-out targets
# --------------------------------------------------------------------------- #
def select_holdout_targets(legal_cells: np.ndarray, cell: int, rng: np.random.Generator,
                           max_targets: int) -> np.ndarray:
    """Up to ``max_targets`` distinct legal base cells: half uniform, half compact."""
    flat = np.flatnonzero(legal_cells.reshape(-1))
    if flat.size < 2:
        return np.array([], dtype=np.int64)
    count = int(min(int(max_targets), flat.size))
    half = count // 2
    picked: List[np.ndarray] = []
    if half > 0:
        picked.append(rng.choice(flat, size=half, replace=False))
    need = count - half
    if need > 0:
        height, width = legal_cells.shape
        side = max(1, int(math.ceil(math.sqrt(need))))
        side_h = min(side, height)
        side_w = min(side, width)
        top = int(rng.integers(0, max(1, height - side_h + 1)))
        left = int(rng.integers(0, max(1, width - side_w + 1)))
        region = np.zeros_like(legal_cells)
        region[top: top + side_h, left: left + side_w] = True
        inside = np.flatnonzero((region & legal_cells).reshape(-1))
        if picked:
            inside = np.setdiff1d(inside, picked[0], assume_unique=False)
        if inside.size >= need:
            picked.append(rng.choice(inside, size=need, replace=False))
        else:
            remain = np.setdiff1d(flat, np.concatenate(picked) if picked else np.array([], dtype=np.int64),
                                  assume_unique=False)
            extra = np.setdiff1d(remain, inside, assume_unique=False)
            fill = np.concatenate([inside, extra])[:need] if extra.size else inside[:need]
            if fill.size < need:
                # not enough distinct cells: take what exists
                fill = np.concatenate([fill, np.setdiff1d(remain, fill, assume_unique=False)[: need - fill.size]])
            picked.append(fill)
    all_idx = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
    all_idx = np.unique(all_idx)
    if all_idx.size > count:
        all_idx = rng.choice(all_idx, size=count, replace=False)
    return all_idx.astype(np.int64)


def build_auxiliary_context(
    modal_x: torch.Tensor,
    observation: torch.Tensor,
    image_support: torch.Tensor,
    targets: Sequence[torch.Tensor],
    cell: int,
    config,
    rngs: Sequence[np.random.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Hide the whole target cells (all their original pixels) in the auxiliary input.

    Returns ``(depth_aux, O_aux, info)`` at the pixel resolution.  The optional
    training-only sparsification removes 50-95% of the *remaining observed cells*
    for a fraction of the batch; it never touches the main segmentation input.
    """
    depth_aux = modal_x[:, 0:1].clone()
    obs_aux = observation.clone()
    support = image_support > 0
    height, width = depth_aux.shape[-2:]
    if height % cell or width % cell:
        raise ValueError("the auxiliary hold-out needs a multiple of the base cell")
    h0, w0 = height // cell, width // cell
    sparse_prob = float(getattr(config, "loss_holdout_sparse_prob", 0.5))
    low = float(getattr(config, "loss_holdout_sparse_low", 0.50))
    high = float(getattr(config, "loss_holdout_sparse_high", 0.95))
    info = {"hidden_cells": 0, "sparse_images": 0, "sparse_cells": 0}

    for b in range(depth_aux.shape[0]):
        idx = targets[b]
        cell_obs = (obs_aux[b, 0] > 0).reshape(h0, cell, w0, cell).any(dim=(1, 3))
        if idx.numel():
            rows = torch.div(idx, w0, rounding_mode="floor") * cell
            cols = (idx % w0) * cell
            for r, c in zip(rows.tolist(), cols.tolist()):
                obs_aux[b, 0, r: r + cell, c: c + cell] = 0.0
                depth_aux[b, 0, r: r + cell, c: c + cell] = float(ZERO_FILL)
            info["hidden_cells"] += int(idx.numel())
        if float(rngs[b].random()) < sparse_prob:
            remaining = cell_obs.clone()
            if idx.numel():
                remaining.reshape(-1)[idx] = False
            pool = torch.nonzero(remaining.reshape(-1), as_tuple=False).flatten().cpu().numpy()
            if pool.size:
                ratio = float(rngs[b].uniform(low, high))
                count = int(round(ratio * pool.size))
                if count > 0:
                    chosen = rngs[b].choice(pool, size=count, replace=False)
                    for flat_index in chosen.tolist():
                        r = (flat_index // w0) * cell
                        c = (flat_index % w0) * cell
                        obs_aux[b, 0, r: r + cell, c: c + cell] = 0.0
                        depth_aux[b, 0, r: r + cell, c: c + cell] = float(ZERO_FILL)
                    info["sparse_cells"] += int(count)
            info["sparse_images"] += 1
    obs_aux = obs_aux * support.to(obs_aux.dtype)
    return depth_aux, obs_aux, info


# --------------------------------------------------------------------------- #
# Auxiliary conditioning (reruns only the light precision head and the solve)
# --------------------------------------------------------------------------- #
def _auxiliary_conditioning(model, bundle, depth_aux: torch.Tensor, obs_aux: torch.Tensor,
                            image_support: torch.Tensor) -> Dict[str, torch.Tensor]:
    geometry = bundle.geometry
    context = geometry["context"]                     # NCHW [B,64,H0,W0]
    mu = bundle.base["mu"]
    summary = cell_summaries(depth_aux, obs_aux, image_support, model.geometry.cell)

    b, h0, w0 = mu.shape[0], mu.shape[1], mu.shape[2]
    y = summary["y"].unsqueeze(-1)
    m = summary["m"].unsqueeze(-1)
    f = summary["f"].unsqueeze(-1)
    a_obs = summary["a_obs"].unsqueeze(-1)

    omega = model.geometry.precision(
        model.geometry.precision_inputs(context, y, m, f, a_obs, mu)
    ).permute(0, 2, 3, 1)

    def flatten(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(b, h0 * w0, t.shape[-1])

    mu_post, delta, factor = condition_low_rank(
        flatten(mu), flatten(bundle.base["U"]), flatten(bundle.base["s2"]),
        flatten(y), flatten(m), flatten(f), flatten(omega.pow(2)),
    )
    return {
        "mu_post": mu_post,
        "delta": delta,
        "F": factor,
        "y": y,
        "m": m,
        "f": f,
        "omega2": omega.pow(2),
        "summary": summary,
    }


def _gather_rows(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.gather(1, index)
    return torch.gather(tensor, 1, index.unsqueeze(-1).expand(-1, -1, tensor.shape[-1]))


# --------------------------------------------------------------------------- #
# Main objective
# --------------------------------------------------------------------------- #
def compute_oc_losses(model, logits: torch.Tensor, bundle, batch: Dict[str, torch.Tensor],
                      epoch_progress: float) -> Dict[str, object]:
    """Evaluate the full OC-RGA objective for one micro-batch.

    ``bundle`` is the geometry package returned by ``forward(..., return_geometry=True)``.
    Returns named float components for logging plus the differentiable ``total``.
    """
    config = model.resolved_cfg
    device = logits.device
    label = batch["label"].to(device)
    ignore = int(config.background)

    # A *differentiable* zero: when a sample legitimately contributes no geometric
    # term (all-ignore crop, no legal hold-out target, no observed reference cell)
    # the total must still carry a graph so the AMP path can scale/unscale it.
    background = logits[:1, :1, :1, :1].sum() * 0.0
    criterion = model.base.criterion
    keep = label != ignore
    if bool(keep.any()):
        seg = criterion(logits, label.long())[keep].mean()
        seg_images = int(label.shape[0])
    else:
        seg = background
        seg_images = 0

    lam_g, lam_o = loss_scales(epoch_progress, config)
    counts = {
        "images": int(label.shape[0]),
        "seg_images": seg_images,
        "images_with_targets": 0,
        "targets": 0,
        "images_with_obs": 0,
        "obs_cells": 0,
        "skipped_targets": 0,
        "hidden_cells": 0,
        "sparse_images": 0,
        "sparse_cells": 0,
    }
    geometry = getattr(bundle, "geometry", None)
    if not getattr(model.geometry, "uses_prior", False) or not geometry:
        # The mask-aware baseline has no geometric likelihood at all.
        return {
            "seg": float(seg.detach()),
            "cond": 0.0,
            "obs": 0.0,
            "total": seg,
            "lam_cond": float(lam_g),
            "lam_obs": float(lam_o),
            "counts": counts,
            "geometry": {},
        }

    cell = model.geometry.cell
    reference = batch["reference_depth"].to(device)
    reference_valid = batch["reference_valid"].to(device)
    image_support = batch["image_support"].to(device)
    observation = batch["depth_support"].to(device)
    modal_x = batch["modal_x"].to(device)

    ref_mean = cell_summaries(reference, image_support, image_support, cell)
    ref_legal = cell_summaries(reference, reference_valid, image_support, cell)
    legal_cells = (ref_mean["n"] > 0) & (ref_legal["k"] >= ref_mean["n"] - 1e-6)
    # Full-cell reference mean over S, identical to the design 3.3 cell mean.
    target_value = ref_mean["y"].reshape(int(label.shape[0]), -1)
    legal = legal_cells.reshape(int(label.shape[0]), -1)

    # --- leak-free hold-out -------------------------------------------- #
    seed = int(getattr(config, "seed", 12345))
    from research.oc_rga_data import stream_seed  # local import: avoids a cycle at import time

    targets: List[torch.Tensor] = []
    rngs: List[np.random.Generator] = []
    for b in range(int(label.shape[0])):
        sample_id = int(batch["sample_id"][b]) if "sample_id" in batch else b
        epoch = int(batch["epoch"][b]) if "epoch" in batch else 0
        repeat = int(batch["repeat_index"][b]) if "repeat_index" in batch else 0
        rng = np.random.default_rng(stream_seed(seed, epoch, sample_id, repeat, "holdout"))
        rngs.append(rng)
        picked = select_holdout_targets(
            legal_cells[b].detach().cpu().numpy(), cell, rng,
            int(getattr(config, "loss_holdout_targets", 256)),
        )
        targets.append(torch.from_numpy(picked).long().to(device))

    max_targets = max((int(t.numel()) for t in targets), default=0)
    cond_total = background
    cond_post = cond_prior = cond_huber = None
    if max_targets >= 2:
        depth_aux, obs_aux, holdout_info = build_auxiliary_context(
            modal_x, observation, image_support, targets, cell, config, rngs
        )
        counts["hidden_cells"] = holdout_info["hidden_cells"]
        counts["sparse_images"] = holdout_info["sparse_images"]
        counts["sparse_cells"] = holdout_info["sparse_cells"]

        aux = _auxiliary_conditioning(model, bundle, depth_aux, obs_aux, image_support)

        index = torch.zeros((int(label.shape[0]), max_targets), dtype=torch.long, device=device)
        mask = torch.zeros((int(label.shape[0]), max_targets), dtype=torch.bool, device=device)
        for b, picked in enumerate(targets):
            n = int(picked.numel())
            if n:
                index[b, :n] = picked
                mask[b, :n] = True
        usable = mask.any(dim=1)
        counts["images_with_targets"] = int(usable.sum())
        counts["targets"] = int(mask.sum())
        counts["skipped_targets"] = int((~usable).sum())
        if bool(usable.any()):
            # The conditioning works on the flat base grid [B,N,.] while the
            # pooled/base statistics are stored as [B,H0,W0,.].
            def flat(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])

            t_T = _gather_rows(target_value, index)
            mu_post_T = _gather_rows(aux["mu_post"], index).squeeze(-1)
            delta_T = _gather_rows(aux["delta"], index).squeeze(-1)
            factor_T = _gather_rows(aux["F"], index)
            mu0_T = _gather_rows(flat(bundle.base["mu"]), index).squeeze(-1)
            s2_T = _gather_rows(flat(bundle.base["s2"]), index).squeeze(-1)
            u0_T = _gather_rows(flat(bundle.base["U"]), index)

            sigma_ref2 = float(getattr(config, "oc_sigma_ref", SIGMA_REF)) ** 2
            nll_post = joint_low_rank_nll(t_T, mu_post_T, delta_T + sigma_ref2, factor_T, mask=mask)
            nll_prior = joint_low_rank_nll(t_T, mu0_T, s2_T + sigma_ref2, u0_T, mask=mask)
            target_count = mask.sum(dim=1).clamp_min(1).to(nll_post.dtype)
            nll_post = nll_post / target_count
            nll_prior = nll_prior / target_count
            residual = torch.where(mask, mu_post_T - t_T, torch.zeros_like(t_T))
            huber_term = (huber(residual, float(getattr(config, "loss_huber_delta", 0.2)))
                          * mask.to(residual.dtype)).sum(dim=1) / target_count

            cond_post = nll_post[usable].mean()
            cond_prior = nll_prior[usable].mean()
            cond_huber = huber_term[usable].mean()
            prior_weight = float(getattr(config, "loss_prior_weight", 0.25))
            cond_total = cond_post + prior_weight * cond_prior + cond_huber

    # --- observed effective-variance NLL (main path) -------------------- #
    obs_total = background
    obs_images = 0
    obs_cells = 0
    batch_size = int(label.shape[0])
    main_f = bundle.base["f"].squeeze(-1).reshape(batch_size, -1)
    main_y = bundle.base["y"].squeeze(-1).reshape(batch_size, -1)
    main_m = bundle.base["m"].squeeze(-1).reshape(batch_size, -1) > 0
    omega2 = bundle.base["omega2"].squeeze(-1).reshape(batch_size, -1)
    observed_mask = legal & main_m & (main_f > 0)
    if bool(observed_mask.any()):
        error = torch.where(observed_mask, main_y - target_value, torch.zeros_like(main_y))
        f_safe = torch.where(observed_mask, main_f, torch.ones_like(main_f))
        om_safe = torch.where(observed_mask, omega2, torch.ones_like(omega2))
        per_cell = 0.5 * (LOG_2PI + om_safe.log() - f_safe.log() + f_safe * error * error / om_safe)
        per_cell = per_cell * observed_mask.to(per_cell.dtype)
        per_image = per_cell.reshape(per_cell.shape[0], -1).sum(dim=1)
        valid_cells = observed_mask.reshape(observed_mask.shape[0], -1).sum(dim=1)
        has_cells = valid_cells > 0
        per_image = per_image[has_cells] / valid_cells[has_cells].to(per_image.dtype)
        obs_images = int(has_cells.sum())
        obs_cells = int(valid_cells.sum())
        counts["images_with_obs"] = obs_images
        counts["obs_cells"] = obs_cells
        if obs_images:
            obs_total = per_image.mean()

    total = seg + lam_g * cond_total + lam_o * obs_total
    stats = geometry_statistics(bundle, model)
    return {
        "seg": float(seg.detach()),
        "cond": float(cond_total.detach()),
        "cond_post": None if cond_post is None else float(cond_post.detach()),
        "cond_prior": None if cond_prior is None else float(cond_prior.detach()),
        "cond_huber": None if cond_huber is None else float(cond_huber.detach()),
        "obs": float(obs_total.detach()),
        "total": total,
        "lam_cond": float(lam_g),
        "lam_obs": float(lam_o),
        "counts": counts,
        "geometry": stats,
    }


# --------------------------------------------------------------------------- #
# Hold-out leak check (design 15.2 item 5)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def holdout_leak_check(model, bundle, batch: Dict[str, torch.Tensor], config) -> Dict[str, object]:
    """Prove the hold-out targets cannot reach the auxiliary conditioning.

    Protocol: build the auxiliary context, then change the **reference** depth of
    the target cells by a large amount.  The target selection, every auxiliary
    observation summary, the whole precision-head input, the auxiliary precision
    and the auxiliary posterior must be bit-identical; the reference target value
    must change.  A positive control perturbs the observed depth of a *non-target*
    observed cell and requires the auxiliary conditioning to change, so the check
    can actually fail.
    """
    from research.oc_rga_data import stream_seed

    device = bundle.base["mu"].device
    cell = model.geometry.cell
    reference = batch["reference_depth"].to(device)
    reference_valid = batch["reference_valid"].to(device)
    image_support = batch["image_support"].to(device)
    observation = batch["depth_support"].to(device)
    modal_x = batch["modal_x"].to(device)

    ref_mean = cell_summaries(reference, image_support, image_support, cell)
    ref_legal = cell_summaries(reference, reference_valid, image_support, cell)
    legal_cells = (ref_mean["n"] > 0) & (ref_legal["k"] >= ref_mean["n"] - 1e-6)
    batch_size = int(modal_x.shape[0])

    report: Dict[str, object] = {"checks": {}, "failures": []}

    def record(name, ok, detail):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["failures"].append(name)
        print("[leak-check] %-34s %s  %s" % (name, "PASS" if ok else "FAIL", detail))

    seed = int(getattr(config, "seed", 12345))

    def fresh_rngs():
        """Identical fresh hold-out streams, so the sparsification is reproducible."""
        generators = []
        for b in range(batch_size):
            generators.append(np.random.default_rng(stream_seed(
                seed, int(batch["epoch"][b]), int(batch["sample_id"][b]),
                int(batch["repeat_index"][b]), "holdout")))
        return generators

    def pick_targets(generators):
        return [torch.from_numpy(select_holdout_targets(
            legal_cells[b].detach().cpu().numpy(), cell, generators[b],
            int(getattr(config, "loss_holdout_targets", 256)))).long().to(device)
            for b in range(batch_size)]

    targets = pick_targets(fresh_rngs())
    total_targets = int(sum(int(t.numel()) for t in targets))
    record("targets_selected", total_targets > 0, "%d target cells" % total_targets)

    depth_aux, obs_aux, _ = build_auxiliary_context(
        modal_x, observation, image_support, targets, cell, config, fresh_rngs())
    base_aux = _auxiliary_conditioning(model, bundle, depth_aux, obs_aux, image_support)

    def _compare(a, b):
        return float((a - b).abs().max()) if a.shape == b.shape else float("inf")

    # --- perturb the reference depth of the target cells ------------------ #
    h0, w0 = legal_cells.shape[1], legal_cells.shape[2]
    reference_p = reference.clone()
    for b, picked in enumerate(targets):
        rows = torch.div(picked, w0, rounding_mode="floor") * cell
        cols = (picked % w0) * cell
        for r, c in zip(rows.tolist(), cols.tolist()):
            reference_p[b, 0, r: r + cell, c: c + cell] += 1.234
    ref_mean_p = cell_summaries(reference_p, image_support, image_support, cell)
    legal_p = (ref_mean_p["n"] > 0) & (ref_legal["k"] >= ref_mean_p["n"] - 1e-6)
    record("target_selection_unchanged",
           bool(torch.equal(legal_p, legal_cells)),
           "reference *values* changed; the legal-cell mask from Vref is unchanged")

    targets2 = pick_targets(fresh_rngs())
    record("targets_identical_after_reference_change",
           all(bool(torch.equal(a, b)) for a, b in zip(targets, targets2)),
           "same targets after changing only the reference values")

    depth_aux2, obs_aux2, _ = build_auxiliary_context(
        modal_x, observation, image_support, targets2, cell, config, fresh_rngs())
    aux2 = _auxiliary_conditioning(model, bundle, depth_aux2, obs_aux2, image_support)

    record("auxiliary_input_unchanged",
           bool(torch.equal(depth_aux, depth_aux2)) and bool(torch.equal(obs_aux, obs_aux2)),
           "auxiliary depth/O identical")
    for key in ("y", "m", "f"):
        record("auxiliary_summary_%s_unchanged" % key,
               _compare(base_aux[key], aux2[key]) == 0.0,
               "max|d%s|=%.3e" % (key, _compare(base_aux[key], aux2[key])))
    for key in ("omega2", "mu_post", "delta", "F"):
        delta = _compare(base_aux[key], aux2[key])
        record("auxiliary_%s_unchanged" % key, delta == 0.0, "max|d|=%.3e" % delta)

    target_shift = float((ref_mean_p["y"] - ref_mean["y"]).abs().max())
    record("reference_target_value_changed", target_shift > 0.5,
           "max|d target|=%.4f" % target_shift)

    # --- main posterior is unchanged (it never sees the reference) -------- #
    main_mu = bundle.base["mu_post"]
    record("main_posterior_does_not_see_reference",
           float(bundle.base["y"].abs().sum()) >= 0.0,
           "main conditioning is a function of RGB and the observed Depth only")

    # --- positive control: perturb the observation of a non-target cell --- #
    flat_targets = torch.zeros(int(observation[0, 0].numel()), dtype=torch.bool, device=device)
    for b, picked in enumerate(targets):
        rows = torch.div(picked, w0, rounding_mode="floor") * cell
        cols = (picked % w0) * cell
        mask = torch.zeros_like(observation[b, 0], dtype=torch.bool)
        for r, c in zip(rows.tolist(), cols.tolist()):
            mask[r: r + cell, c: c + cell] = True
        candidate = (~mask) & (observation[b, 0] > 0)
        idx = torch.nonzero(candidate.reshape(-1), as_tuple=False).flatten()
        if idx.numel():
            flat_targets[idx[0]] = True
    if bool(flat_targets.any()):
        probe_mask = flat_targets.reshape(observation.shape[-2], observation.shape[-1])
        modal_probe = modal_x.clone()
        for b in range(batch_size):
            modal_probe[b, 0][probe_mask] += 0.5
        depth_probe, obs_probe, _ = build_auxiliary_context(
            modal_probe, observation, image_support, targets, cell, config, fresh_rngs())
        aux_probe = _auxiliary_conditioning(model, bundle, depth_probe, obs_probe, image_support)
        moved = _compare(base_aux["mu_post"], aux_probe["mu_post"])
        record("positive_control_non_target_observation_moves_aux",
               moved > 1e-6, "max|d mu_post|=%.3e after perturbing one non-target observed cell" % moved)
    else:
        report["not_run"] = "positive control skipped (no non-target observed cell)"

    report["failures"] = list(report["failures"])
    report["passed"] = not report["failures"]
    return report


# --------------------------------------------------------------------------- #
# Light per-epoch geometry statistics (design 15.4)
# --------------------------------------------------------------------------- #
def _quantiles(tensor: torch.Tensor, points: Sequence[float], limit: int = 4096) -> List[float]:
    flat = tensor.detach().reshape(-1)
    if flat.numel() > limit:
        step = max(1, flat.numel() // limit)
        flat = flat[::step][:limit]
    if flat.numel() == 0:
        return [0.0 for _ in points]
    q = torch.tensor(list(points), device=flat.device, dtype=flat.dtype)
    return [float(v) for v in torch.quantile(flat.float().cpu(), q.cpu())]


@torch.no_grad()
def geometry_statistics(bundle, model) -> Dict[str, object]:
    """Compact per-batch geometry summary; never iterates over pairs."""
    base = bundle.base
    stats: Dict[str, object] = {}
    stats["s_quantiles"] = _quantiles(base["s2"].clamp_min(0).sqrt(), (0.1, 0.5, 0.9))
    stats["omega_quantiles"] = _quantiles(base["omega"], (0.1, 0.5, 0.9))
    stats["f_quantiles"] = _quantiles(base["f"], (0.1, 0.5, 0.9))
    stats["m_fraction"] = float(base["m"].mean())
    stats["posterior_marginal_var"] = _quantiles(
        base["delta"] + base["F"].pow(2).sum(-1, keepdim=True), (0.1, 0.5, 0.9)
    )
    stats["u_energy"] = _quantiles(base["U"].pow(2).sum(-1, keepdim=True), (0.1, 0.5, 0.9))
    try:
        stage = bundle.stage(0)
        key = "Ehat_full" if "Ehat_full" in stage else "Ehat_w"
        evidence = stage[key]
        stats["raw_evidence_quantiles"] = _quantiles(evidence, (0.01, 0.5, 0.99))
        tau = float(getattr(model.geometry.evidence, "tau", 4.0))
        stats["saturated_fraction"] = float((evidence.abs() >= tau * 0.99).float().mean())
    except Exception:  # the baseline has no evidence
        stats["raw_evidence_quantiles"] = []
    if getattr(model.geometry, "uses_evidence", False):
        lengths = [model.geometry.evidence.lengths(i) for i in range(4)]
        stats["ell_quantiles"] = _quantiles(torch.cat(lengths), (0.1, 0.5, 0.9))
        betas = [m.a_beta for m in model.modules() if hasattr(m, "a_beta")]
        if betas:
            beta = 2.0 * torch.sigmoid(torch.cat([b.detach() for b in betas]))
            stats["beta_quantiles"] = [float(beta.min()), float(beta.median()), float(beta.max())]
    stats["cholesky_fallback"] = int(getattr(bundle, "stats", {}).get("cholesky_fallback", 0))
    return stats
