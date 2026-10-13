"""Direction-A frozen influence-projection operators (forward-only, FP32).

Mathematical definition (plan section 4.1).  For one attention row let

    S    = qk + Bs                (spatial-only logits)
    p0   = softmax(S)             (reference distribution)
    pD   = softmax(S + Bd)        (full-geometry distribution, signed Bd)

Direction A solves the *reverse* KL projection of ``pD`` onto the box
``{p : sum_j p_j = 1, e^{-tau} p0_j <= p_j <= e^{tau} p0_j}``:

    p* = argmin_{p in box} KL(p || pD)

whose unique solution (strictly convex objective, non-empty box) is

    p*_j = clip(c * pD_j, e^{-tau} p0_j, e^{tau} p0_j),  sum_j p*_j = 1,

with a single common normaliser ``c``.  In the log domain, with
``l0 = log p0``, ``r = log(pD / p0)`` and ``z = log c``:

    log p*_j = l0_j + clamp(z + r_j, -tau, tau)
    f(z)     = sum_j exp(log p*_j) - 1  = 0

``f`` is monotone in ``z`` with the guaranteed bracket
``z_lo = -max_j r_j`` and ``z_hi = -min_j r_j``, so a vectorised log-domain
bisection (no per-row Python loop, no ``.item()`` inside the loop) solves every
row at once.  The result is the true projection: the code never clips a
probability and re-normalises it, and it never clamps the reference ``p0`` to an
epsilon.

The strong baselines implemented here share the same inputs and FP32 protocol:

* ``clip``: bounded signed-bias clip
  ``m = (max_j Bd_j + min_j Bd_j) / 2``, ``Bd' = clamp(Bd - m, -tau/2, tau/2)``;
* ``gate``: per-row analytic bounded gate
  ``R = max_j Bd_j - min_j Bd_j``, ``g = 1 if R == 0 else min(1, tau / R)``,
  applied as ``softmax(S + g * Bd)``.

Both satisfy the same one-sided bound ``|log(p_j / p0_j)| <= tau`` (their bias
row span is at most ``tau``), but neither is the KL-minimal element of the
probability box.  They are legitimate strong controls, not approximations of A.

Everything in this module is frozen forward inference: no parameters, no
gradients, no re-normalisation shortcut.  Inputs are rejected (raise) when they
are non-finite or when ``tau`` is not a finite non-negative scalar.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "MODE_ORIGINAL",
    "MODE_DEPTH_OFF",
    "MODE_KL",
    "MODE_CLIP",
    "MODE_GATE",
    "SUPPORTED_MODES",
    "DEFAULT_ITERATIONS",
    "NORM_TOL",
    "BOUND_TOL",
    "TINY_P0",
    "validate_tau",
    "project_log_ratio",
    "controlled_attention",
    "run_self_check",
]

MODE_ORIGINAL = "original"
MODE_DEPTH_OFF = "depth_off"
MODE_KL = "kl"
MODE_CLIP = "clip"
MODE_GATE = "gate"
SUPPORTED_MODES = (MODE_ORIGINAL, MODE_DEPTH_OFF, MODE_KL, MODE_CLIP, MODE_GATE)

#: Default bisection depth.  Not part of the mathematical definition: it may be
#: raised (32/48) if a tolerance check fails, and the extra cost is recorded.
DEFAULT_ITERATIONS = 24
#: Acceptance targets from the plan (probabilities sum to 1, log-ratio bound).
NORM_TOL = 1e-6
BOUND_TOL = 5e-6
#: "Small reference key" separator used by the online attention summary.
TINY_P0 = 1e-6


def validate_tau(tau):
    """Return ``tau`` as a finite non-negative float, or raise."""
    value = float(tau)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("tau must be a finite non-negative scalar, got %r" % (tau,))
    return value


def _assert_finite(name, tensor):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("%s contains NaN/Inf; refusing to compute an influence projection" % name)


def project_log_ratio(log_p0, r, tau, *, iterations=DEFAULT_ITERATIONS):
    """Solve ``min_p KL(p || pD)`` inside the two-sided density-ratio box.

    Parameters
    ----------
    log_p0 : Tensor
        ``log p0`` of the reference (spatial-only) distribution; the whole
        trailing dimension is treated as the key axis.
    r : Tensor
        ``log(pD / p0)``, already normalised so that ``logsumexp(log_p0 + r)`` is
        zero within float tolerance.
    tau : float
        Log-scale budget of the box (``tau >= 0``).
    iterations : int
        Bisection depth (default :data:`DEFAULT_ITERATIONS`).

    Returns
    -------
    (log_p, t) : tuple of Tensors
        ``log_p = log p*`` (used directly for aggregation) and
        ``t = log(p* / p0)``, both with the same shape as the inputs.
    """
    if log_p0.shape != r.shape:
        raise ValueError(
            "log_p0 and r must share a shape, got %s and %s"
            % (tuple(log_p0.shape), tuple(r.shape))
        )
    if log_p0.dim() < 1:
        raise ValueError("the key dimension must exist; got shape %s" % (tuple(log_p0.shape),))
    tau = validate_tau(tau)
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError("iterations must be >= 1, got %r" % (iterations,))
    if log_p0.numel() == 0:
        return log_p0, torch.zeros_like(r)
    _assert_finite("log_p0", log_p0)
    _assert_finite("r", r)

    if tau == 0.0:
        # The box collapses onto p0 (plan section 4.2 special path).
        return log_p0, torch.zeros_like(r)

    feasible = (r.abs() <= tau).all(dim=-1, keepdim=True)
    if bool(feasible.all()):
        # pD already sits inside the box: the projection returns pD unchanged.
        return log_p0 + r, r

    lo = -r.amax(dim=-1, keepdim=True)
    hi = -r.amin(dim=-1, keepdim=True)
    for _ in range(iterations):
        z = 0.5 * (lo + hi)
        t = torch.clamp(z + r, -tau, tau)
        mass = torch.exp(log_p0 + t).sum(dim=-1, keepdim=True)
        lo = torch.where(mass < 1.0, z, lo)
        hi = torch.where(mass >= 1.0, z, hi)

    z = 0.5 * (lo + hi)
    bounded_t = torch.clamp(z + r, -tau, tau)
    t = torch.where(feasible, r, bounded_t)
    return log_p0 + t, t


def controlled_attention(qk, bs, bd, mode, tau, *, observer=None, meta=None,
                         iterations=DEFAULT_ITERATIONS):
    """Attention probabilities for one attention call under a direction-A mode.

    The trailing dimension of ``qk`` / ``bs`` / ``bd`` is the complete key axis;
    every preceding dimension is an independent row.  ``bs`` and ``bd`` are the
    already-weighted (by the original signed ``GeoPriorGen.weight``) spatial and
    depth parts and broadcast against ``qk``.

    ``mode`` is one of :data:`SUPPORTED_MODES`:

    * ``original``  -- full signed geometry, ``p = softmax(qk + bs + bd)``;
    * ``depth_off`` -- ``p = softmax(qk + bs) = p0`` (reference only);
    * ``kl``        -- the KL projection solved by :func:`project_log_ratio`;
    * ``clip``      -- bounded signed-bias clip strong control;
    * ``gate``      -- per-row analytic bounded-gate strong control.

    Returns the FP32 probability tensor ``p``.  ``observer`` (optional) is called
    with keyword arguments ``meta, l0, r, t, p, tau`` for the online summary.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError("unknown mode %r; expected one of %s" % (mode, ", ".join(SUPPORTED_MODES)))
    tau = validate_tau(tau)
    iterations = int(iterations)

    device_type = qk.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        qk = qk.float()
        bs_f = None if bs is None else bs.float()
        bd_f = None if bd is None else bd.float()
        if qk.dim() < 1:
            raise ValueError("attention logits need at least one dimension")
        logits = qk if bs_f is None else qk + bs_f
        # NaN/Inf anywhere in qk or bs propagates into the sum, so one check over
        # the combined logits plus one over bd covers all three inputs.
        _assert_finite("attention logits (qk + bs)", logits)
        if bd_f is not None:
            _assert_finite("depth bias bd", bd_f)
        log_p0 = torch.log_softmax(logits, dim=-1)

        r_full = None
        if bd_f is not None:
            # Row-constant shift only stabilises the exponential; it does not
            # change the signed relation or the softmax distribution.
            b = bd_f - bd_f.amax(dim=-1, keepdim=True)
            r_full = b - torch.logsumexp(log_p0 + b, dim=-1, keepdim=True)

        if mode == MODE_ORIGINAL:
            if r_full is None:
                raise ValueError("mode 'original' needs the depth bias bd")
            t = r_full
            log_p = log_p0 + r_full
        elif mode == MODE_DEPTH_OFF:
            t = torch.zeros_like(log_p0)
            log_p = log_p0
        elif mode == MODE_KL:
            if r_full is None:
                raise ValueError("mode 'kl' needs the depth bias bd")
            log_p, t = project_log_ratio(log_p0, r_full, tau, iterations=iterations)
        elif mode == MODE_CLIP:
            if bd_f is None:
                raise ValueError("mode 'clip' needs the depth bias bd")
            mid = 0.5 * (bd_f.amax(dim=-1, keepdim=True) + bd_f.amin(dim=-1, keepdim=True))
            bc = torch.clamp(bd_f - mid, -tau / 2.0, tau / 2.0)
            t = bc - torch.logsumexp(log_p0 + bc, dim=-1, keepdim=True)
            log_p = log_p0 + t
        else:  # MODE_GATE
            if bd_f is None:
                raise ValueError("mode 'gate' needs the depth bias bd")
            span = bd_f.amax(dim=-1, keepdim=True) - bd_f.amin(dim=-1, keepdim=True)
            positive = span > 0
            span_safe = torch.where(positive, span, torch.ones_like(span))
            g = torch.minimum(torch.ones_like(span), tau / span_safe)
            g = torch.where(positive, g, torch.ones_like(g))
            bg = g * (bd_f - bd_f.amax(dim=-1, keepdim=True))
            t = bg - torch.logsumexp(log_p0 + bg, dim=-1, keepdim=True)
            log_p = log_p0 + t

        p = torch.exp(log_p)

    if observer is not None:
        observer(meta=meta, l0=log_p0, r=r_full, t=t, p=p, tau=tau)
    return p


def _check(condition, name, detail, results, verbose):
    results.append({"check": name, "ok": bool(condition), "detail": detail})
    if verbose:
        print("[self-check] %-48s %-4s %s" % (name, "OK" if condition else "FAIL", detail))
    if not condition:
        raise AssertionError("influence self-check failed: %s (%s)" % (name, detail))


def run_self_check(*, verbose=True, iterations=DEFAULT_ITERATIONS):
    """Fixed small checks for the frozen operators (no model, no data files).

    Raises ``AssertionError`` on the first failure; returns the list of results.
    """
    results = []
    tau = math.log(2.0)

    # 1. Hand-worked example from the plan: p0=(.5,.3,.2), pD=(.8,.1,.1), tau=ln2
    #    => p*=(.75,.15,.10) with a common factor c=0.9375.  A naive
    #    "clip then renormalise" gives (.8,.15,.1)/1.05 and breaks the lower
    #    bounds, which this check rules out.
    p0 = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    pD = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)
    l0 = torch.log(p0)
    r = torch.log(pD) - l0
    want = torch.tensor([[0.75, 0.15, 0.10]], dtype=torch.float64)
    # The default 24-step bisection carries a bounded FP32/FP64 error
    # ((max r - min r) / 2**25); the mathematical solution is verified with a
    # deep FP64 solve, and the default-depth error is reported next to it.
    log_p, t = project_log_ratio(l0, r, tau, iterations=80)
    got = torch.exp(log_p)
    err = float((got - want).abs().max())
    _check(err < 1e-12, "hand example p*=(0.75,0.15,0.10)",
           "max|p*-want|=%.3e (FP64, 80 iterations)" % err, results, verbose)
    default_err = float((torch.exp(project_log_ratio(l0, r, tau)[0]) - want).abs().max())
    _check(default_err < NORM_TOL, "hand example at the default bisection depth",
           "max|p*-want|=%.3e (default %d iterations)" % (default_err, iterations), results, verbose)
    c_err = abs(math.exp(float(t[0, 0] - r[0, 0])) - 0.9375)
    _check(c_err < 1e-12, "hand example common factor c=0.9375", "|c-0.9375|=%.3e" % c_err, results, verbose)
    naive = torch.tensor([[0.8, 0.15, 0.1]]) / 1.05
    naive_bad = bool((naive[0, 1] < p0[0, 1] * math.exp(-tau)) or (naive[0, 2] < p0[0, 2] * math.exp(-tau)))
    _check(naive_bad, "naive clip+renormalise is infeasible (why A matters)",
           "naive lower-bound violation detected", results, verbose)

    # 2. tau = 0 collapses the box onto p0.
    log_p0, t0 = project_log_ratio(l0, r, 0.0)
    zero_err = float((torch.exp(log_p0) - p0).abs().max()) + float(t0.abs().max())
    _check(zero_err < 1e-12, "tau=0 returns p0", "err=%.3e" % zero_err, results, verbose)

    # 3. Single key: the only normalised r is 0 (p0 == pD == 1); the degenerate
    #    bracket must not raise and must stay a probability.
    one_l0 = torch.zeros(2, 1, dtype=torch.float64)
    one_r = torch.zeros(2, 1, dtype=torch.float64)
    one_log_p, _ = project_log_ratio(one_l0, one_r, tau)
    one_err = float((torch.exp(one_log_p) - 1.0).abs().max())
    _check(one_err < 1e-12, "single key stays normalised", "err=%.3e" % one_err, results, verbose)

    # 4. A row already inside the box returns pD unchanged.
    even_l0 = torch.log(torch.full((1, 4), 0.25, dtype=torch.float64))
    r0 = torch.tensor([[0.1, -0.2, 0.05, -0.05]], dtype=torch.float64)
    feasible_r = r0 - torch.logsumexp(even_l0 + r0, dim=-1, keepdim=True)
    _, t_feasible = project_log_ratio(even_l0, feasible_r, tau)
    f_err = float((t_feasible - feasible_r).abs().max())
    _check(f_err < 1e-12, "feasible row keeps pD", "max|t-r|=%.3e" % f_err, results, verbose)

    # 5. Operator-level semantics on random logits.
    torch.manual_seed(0)
    qk = torch.randn(4, 3, 11, 17) * 2.0
    bs = torch.randn(1, 1, 11, 17) * 0.4
    bd = torch.randn(4, 3, 11, 17) * 6.0  # signed, as the pretrained weights are
    p_ref = torch.softmax(qk + bs, -1)
    p_off = controlled_attention(qk, bs, bd, MODE_DEPTH_OFF, tau)
    off_err = float((p_off - p_ref).abs().max())
    _check(off_err < 1e-6, "controlled depth_off == softmax(qk+bs)", "max err=%.3e" % off_err, results, verbose)

    p_orig = controlled_attention(qk, bs, bd, MODE_ORIGINAL, tau)
    p_full = torch.softmax(qk + bs + bd, -1)
    orig_err = float((p_orig - p_full).abs().max())
    _check(orig_err < 1e-6, "controlled original == softmax(qk+bs+bd)", "max err=%.3e" % orig_err, results, verbose)

    const_bd = torch.full((4, 3, 11, 17), 3.5)
    p_const = controlled_attention(qk, bs, const_bd, MODE_KL, tau)
    const_err = float((p_const - p_ref).abs().max())
    _check(const_err < 1e-5, "constant Bd projects back onto p0", "max err=%.3e" % const_err, results, verbose)

    # 6. Bound, normalisation and finiteness for every projection mode.
    for mode in (MODE_KL, MODE_CLIP, MODE_GATE):
        captured = {}

        def _obs(**kwargs):
            captured.update(kwargs)

        p = controlled_attention(qk, bs, bd, mode, tau, observer=_obs)
        t = captured["t"]
        norm_err = float((p.sum(-1) - 1.0).abs().max())
        over = float((t.abs() - tau).clamp_min(0.0).max())
        finite = bool(torch.isfinite(p).all() and torch.isfinite(t).all())
        _check(
            norm_err <= NORM_TOL and over <= BOUND_TOL and finite,
            "%s normalised and inside [-tau, tau]" % mode,
            "norm=%.3e over=%.3e finite=%s" % (norm_err, over, finite),
            results,
            verbose,
        )

    # 7. The three controls are legitimately different on the same row set; A is
    #    not silently reproducing clip/gate.
    p_kl = controlled_attention(qk, bs, bd, MODE_KL, tau)
    p_clip = controlled_attention(qk, bs, bd, MODE_CLIP, tau)
    p_gate = controlled_attention(qk, bs, bd, MODE_GATE, tau)
    diff_kl_clip = float((p_kl - p_clip).abs().max())
    diff_kl_gate = float((p_kl - p_gate).abs().max())
    _check(
        diff_kl_clip > 1e-4 and diff_kl_gate > 1e-4,
        "kl differs from clip and gate",
        "|kl-clip|=%.3e |kl-gate|=%.3e" % (diff_kl_clip, diff_kl_gate),
        results,
        verbose,
    )

    # 8. FP32 default depth against an FP64 reference solve.
    torch.manual_seed(1)
    l0_64 = torch.log_softmax(torch.randn(3, 129, dtype=torch.float64) * 3.0, -1)
    r_64 = torch.randn(3, 129, dtype=torch.float64) * 2.5
    r_64 = r_64 - torch.logsumexp(l0_64 + r_64, dim=-1, keepdim=True)
    lp64, _ = project_log_ratio(l0_64, r_64, tau, iterations=80)
    lp32, _ = project_log_ratio(l0_64.float(), r_64.float(), tau, iterations=iterations)
    dev = float((torch.exp(lp32.double()) - torch.exp(lp64)).abs().max())
    _check(dev < 1e-5, "FP32 projection tracks an FP64 reference", "max|p32-p64|=%.3e" % dev, results, verbose)

    # 9. Pre-normalised r convention: logsumexp(l0 + r) == 0 for the r we build.
    b = bd - bd.amax(dim=-1, keepdim=True)
    r_norm = b - torch.logsumexp(p_ref.log() + b, dim=-1, keepdim=True)
    lse_err = float(torch.logsumexp(p_ref.log() + r_norm, dim=-1).abs().max())
    _check(lse_err < 1e-5, "r is normalised: logsumexp(log_p0 + r) == 0",
           "max|lse|=%.3e" % lse_err, results, verbose)

    # 10. Non-finite input is refused rather than silently sanitised.
    refused = False
    try:
        bad = qk.clone()
        bad[0, 0, 0, 0] = float("nan")
        controlled_attention(bad, bs, bd, MODE_KL, tau)
    except ValueError:
        refused = True
    _check(refused, "NaN input is rejected", "ValueError raised", results, verbose)

    if verbose:
        print("[self-check] %d/%d checks passed" % (len(results), len(results)))
    return results


if __name__ == "__main__":
    run_self_check()
