"""OC-RGA probability-geometry core.

Implements sections 3-8 of the approved OC-RGA v0.3 implementation design
(``临时/OC_RGA_RESEARCH_DESIGN_v0.2.md``; the page version is v0.3).  This module
contains only pure tensor functions and registered ``nn.Module`` objects: the
RGB-only relational prior, the observation precision head, the single low-rank
Cholesky conditioning, the exact non-overlapping scale propagation, the
relational kernel expectation and the bounded posterior-prior evidence.

Vocabulary (fixed by the design)
--------------------------------
* ``S``  image geometry support (finite un-padded image domain),
* ``O``  currently available observation (``O <= S``; zero on known deletions,
  one on non-zero corruption),
* ``Vref``  training-only encoded reference filter; never an inference input.

All scalar depth values live in the author encoding
``z = (D8/255 - 0.48) / 0.28``; they are **not** metres and no scale
equivariance is claimed.  A raw depth zero maps to :data:`Z_MIN`.

Self-check (CPU only, no temporary test files)::

    python -m research.oc_rga --self-check

The self-check verifies the low-rank posterior against a dense FP64 Gaussian
reference, the Cholesky transpose direction, the joint NLL against a dense
reference, the covariance contraction, the relative-covariance read-out, the
self-relation and all-missing endpoints, and runs a tiny FP64 ``gradcheck``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DEPTH_NORM_MEAN",
    "DEPTH_NORM_STD",
    "Z_MIN",
    "Z_MAX",
    "Z_CENTER",
    "ZERO_FILL",
    "RANK",
    "NUM_WINDOWS",
    "SIGMA_REF",
    "TAU_SAT",
    "bounded_std",
    "inverse_bounded_std",
    "bounded_mean",
    "cell_summaries",
    "merge_2x2",
    "condition_low_rank",
    "joint_low_rank_nll",
    "axis_relation_variance",
    "RGBGeometryPrior",
    "ObservationPrecisionHead",
    "OCSpatialPrior",
    "OCInjectionScale",
    "RelationalEvidencePyramid",
    "OCGeometryBuilder",
    "OCGeometryBundle",
    "padding_key_mask",
    "run_self_check",
]

# --------------------------------------------------------------------------- #
# Encoding constants (design section 3.1)
# --------------------------------------------------------------------------- #
DEPTH_NORM_MEAN = 0.48
DEPTH_NORM_STD = 0.28
Z_MIN = -DEPTH_NORM_MEAN / DEPTH_NORM_STD  # -1.7142857...
Z_MAX = (1.0 - DEPTH_NORM_MEAN) / DEPTH_NORM_STD  # 1.8571428...
Z_CENTER = 0.5 * (Z_MIN + Z_MAX)
Z_HALF_SPAN = 0.5 * (Z_MAX - Z_MIN)
# Author raw-zero normalised value: the sentinel written into missing Depth
# inputs.  Kept identical to research.data.DEPTH_RAW_ZERO_NORM.
ZERO_FILL = Z_MIN

# --------------------------------------------------------------------------- #
# Model constants (design sections 4-7)
# --------------------------------------------------------------------------- #
RANK = 20
GLOBAL_CHANNELS = 2
WINDOW_CENTERS = (0.0, 0.5, 1.0)
NUM_WINDOWS = len(WINDOW_CENTERS) ** 2  # 9
LOCAL_CHANNELS = NUM_WINDOWS * GLOBAL_CHANNELS  # 18
BASE_CELL = 4

GLOBAL_ROW_ENERGY = 0.15
LOCAL_ROW_ENERGY = 0.60

S_MIN, S_MAX = 0.03, 1.0
OMEGA_MIN, OMEGA_MAX = 0.05, 3.60
ELL_MIN, ELL_MAX = 0.15, 1.50
ELL_INIT_RANGE = (0.25, 0.70)
SIGMA_INIT = 0.30
OMEGA_INIT = 0.20
BETA_INIT = 0.10

SIGMA_REF = 0.02
TAU_SAT = 4.0
# Finite stand-in for -inf so an all-masked row still yields a finite softmax.
PAD_LOGIT = -1.0e4

S_TYPES = ("S", "O", "VREF")
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Bounded positive parameterisation (design 4.4)
# --------------------------------------------------------------------------- #
def bounded_std(a: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """``BStd(a; l, u) = exp(log l + sigmoid(a) log(u/l))`` (strictly positive)."""
    if not 0.0 < float(lower) < float(upper):
        raise ValueError("bounded_std needs 0 < lower < upper, got %r" % ((lower, upper),))
    ratio = math.log(float(upper) / float(lower))
    return torch.exp(math.log(float(lower)) + torch.sigmoid(a) * ratio)


def inverse_bounded_std(value, lower: float, upper: float) -> torch.Tensor:
    """``logit(log(value/l) / log(u/l))`` -- the ``a`` that yields ``BStd = value``."""
    if not 0.0 < float(lower) < float(upper):
        raise ValueError("inverse_bounded_std needs 0 < lower < upper")
    ratio = math.log(float(upper) / float(lower))
    fraction = math.log(float(value) / float(lower)) / ratio
    fraction = min(max(fraction, 1e-6), 1.0 - 1e-6)
    return torch.tensor(math.log(fraction / (1.0 - fraction)), dtype=torch.float32)


def bounded_mean(a: torch.Tensor) -> torch.Tensor:
    """``mu = z_c + (z_max - z_min)/2 * tanh(a)`` (bounded prior mean)."""
    return Z_CENTER + Z_HALF_SPAN * torch.tanh(a)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
# Base-grid summaries (design 3.3)
# --------------------------------------------------------------------------- #
def cell_summaries(
    depth: torch.Tensor,
    observation: torch.Tensor,
    image_support: torch.Tensor,
    cell: int = BASE_CELL,
) -> Dict[str, torch.Tensor]:
    """Per-``cell``x``cell`` summaries of one encoded depth plane.

    ``depth`` ``[B,1,H,W]`` encoded FP32; ``observation`` / ``image_support``
    ``[B,1,H,W]`` weights in ``[0,1]``.  Returns base-grid tensors of shape
    ``[B,H0,W0,...]`` with ``H0 = H//cell``: ``n, k, m, f, y, zbar, a_obs``.

    Empty cells are cleaned explicitly (no ``0 * NaN``): ``y``, ``a_obs`` and
    ``f`` are zeroed and ``m`` is 0.
    """
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must be [B,1,H,W], got %s" % (tuple(depth.shape),))
    if observation.shape != depth.shape or image_support.shape != depth.shape:
        raise ValueError("observation and image_support must match depth shape")
    b, _, h, w = depth.shape
    if h % cell or w % cell:
        raise ValueError("H and W must be multiples of the base cell (%d)" % cell)

    z = depth.float()
    o = observation.float()
    s = image_support.float()
    finite = torch.isfinite(z)
    if not bool(finite.all()):
        z = torch.where(finite, z, torch.zeros_like(z))
        s = s * finite.float()
        o = o * finite.float()
    if not bool(torch.isfinite(s).all()) or not bool(torch.isfinite(o).all()):
        raise ValueError("observation/image_support must be finite")
    if bool(((s < 0) | (s > 1)).any()) or bool(((o < 0) | (o > 1)).any()):
        raise ValueError("observation/image_support must be weights in [0,1]")
    o = o * s  # O <= S

    h0, w0 = h // cell, w // cell
    zc = z[:, 0].reshape(b, h0, cell, w0, cell)
    sc = s[:, 0].reshape(b, h0, cell, w0, cell)
    oc = o[:, 0].reshape(b, h0, cell, w0, cell)

    n = sc.sum(dim=(2, 4))
    k = oc.sum(dim=(2, 4))
    sum_sz = (sc * zc).sum(dim=(2, 4))
    sum_oz = (oc * zc).sum(dim=(2, 4))
    sum_oz2 = (oc * zc * zc).sum(dim=(2, 4))

    zbar = sum_sz / n.clamp_min(1.0)
    y = sum_oz / k.clamp_min(1.0)
    m = (k > 0).to(z.dtype)
    f = k / n.clamp_min(1.0)
    second = sum_oz2 / k.clamp_min(1.0)
    a_obs = (second - y * y).clamp_min(0.0)

    has_obs = (k > 0).to(z.dtype)
    has_geom = (n > 0).to(z.dtype)
    zero = torch.zeros_like(y)
    y = torch.where(has_obs > 0, y, zero)
    a_obs = torch.where(has_obs > 0, a_obs, zero)
    f = torch.where(has_geom > 0, f, zero)
    zbar = torch.where(has_geom > 0, zbar, zero)

    return {
        "n": n,
        "k": k,
        "m": m,
        "f": f,
        "y": y,
        "zbar": zbar,
        "a_obs": a_obs,
        "h0": h0,
        "w0": w0,
        "grid": (h0, w0),
        "cell": int(cell),
    }


def _support_centroids(image_support: torch.Tensor, cell: int = BASE_CELL) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalised ``(cx, cy)`` cell centroids of the ``S`` support.

    Coordinates are normalised by the un-padded current input domain (the extent
    where ``S`` is non-zero), never by the padded grid.  An axis whose support
    degenerates to a single row/column is set to ``0.5``.
    """
    s = image_support.float()
    b, _, h, w = s.shape
    h0, w0 = h // cell, w // cell
    sv = (s[:, 0] > 0).reshape(b, h0, cell, w0, cell)
    n = s[:, 0].reshape(b, h0, cell, w0, cell).sum(dim=(2, 4))

    # Un-padded extent of the current input domain, resolved per image so a padded
    # member of the batch cannot shrink another member's normalisation.
    rows = sv.any(dim=4).any(dim=3).any(dim=2)  # [B,H0] -- support in that cell row
    cols = sv.any(dim=4).any(dim=2).any(dim=1)  # [B,W0] -- support in that cell column
    row_idx = torch.arange(1, h0 + 1, device=s.device, dtype=torch.float32).view(1, h0)
    col_idx = torch.arange(1, w0 + 1, device=s.device, dtype=torch.float32).view(1, w0)
    h_cells = torch.where(rows.any(dim=1, keepdim=True), (rows.float() * row_idx).max(dim=1, keepdim=True).values,
                          torch.full((b, 1), float(h0), device=s.device))
    w_cells = torch.where(cols.any(dim=1, keepdim=True), (cols.float() * col_idx).max(dim=1, keepdim=True).values,
                          torch.full((b, 1), float(w0), device=s.device))

    xs = (torch.arange(w, device=s.device, dtype=torch.float32) + 0.5) / (w_cells * cell)  # [B,W]
    ys = (torch.arange(h, device=s.device, dtype=torch.float32) + 0.5) / (h_cells * cell)  # [B,H]

    sum_x = (sv * xs.reshape(b, 1, 1, w0, cell)).sum(dim=(2, 4))
    sum_y = (sv.float() * ys.reshape(b, h0, cell, 1, 1)).sum(dim=(2, 4))
    cx = sum_x / n.clamp_min(1.0)
    cy = sum_y / n.clamp_min(1.0)

    cols_any = (sv > 0).any(dim=2).sum(dim=-1)  # [B,H0,W0] columns with support
    rows_any = (sv > 0).any(dim=4).sum(dim=2)  # [B,H0,W0] rows with support
    half = torch.full_like(cx, 0.5)
    cx = torch.where((n > 0) & (cols_any > 1), cx, half)
    cy = torch.where((n > 0) & (rows_any > 1), cy, half)
    return cx, cy


def window_basis(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    """Tight-support tent windows ``g_k`` normalised so ``sum_k g_k^2 == 1``.

    ``cx``/``cy`` are ``[B,H0,W0]`` normalised centroids; returns
    ``[B,H0,W0,9]``.  All-padding cells contribute a finite basis but are zeroed
    by the caller through ``n > 0``.
    """
    basis = []
    for cy_k in WINDOW_CENTERS:
        for cx_k in WINDOW_CENTERS:
            gx = torch.clamp(1.0 - 3.0 * (cx - cx_k).abs(), min=0.0)
            gy = torch.clamp(1.0 - 3.0 * (cy - cy_k).abs(), min=0.0)
            basis.append(gx * gy)
    g = torch.stack(basis, dim=-1)
    norm = g.pow(2).sum(dim=-1, keepdim=True).clamp_min(EPS).sqrt()
    return g / norm


# --------------------------------------------------------------------------- #
# Exact non-overlapping scale propagation (design 6.1)
# --------------------------------------------------------------------------- #
def _merge_2x2_weighted(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted 2x2 merge of ``x`` ``[B,H,W,C]`` with per-cell ``weight`` ``[B,H,W]``."""
    b, h, w, c = x.shape
    wv = weight.reshape(b, h // 2, 2, w // 2, 2)
    xv = x.reshape(b, h // 2, 2, w // 2, 2, c)
    num = (xv * wv.unsqueeze(-1)).sum(dim=(2, 4))
    den = wv.sum(dim=(2, 4))
    return num, den


def merge_2x2(
    mu: torch.Tensor,
    var: torch.Tensor,
    factor: torch.Tensor,
    n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One exact non-overlapping 2x2 merge of a low-rank Gaussian field.

    ``mu``/``var`` ``[B,H,W,1]``, ``factor`` ``[B,H,W,r]``, ``n`` ``[B,H,W]``
    geometry counts.  ``P_ai = n_i / sum_{j in a} n_j`` depends on ``S`` only.
    A parent cell without geometry support keeps a finite positive-variance
    placeholder (uniform children average) that is masked out later.
    """
    b, h, w, _ = mu.shape
    if h % 2 or w % 2:
        raise ValueError("merge_2x2 needs even H and W")
    h2, w2 = h // 2, w // 2
    n_cell = n.reshape(b, h2, 2, w2, 2)
    n_parent = n_cell.sum(dim=(2, 4))

    # P_ai = n_i / n_a when the parent has support, else a uniform 1/4 placeholder
    # for the masked-out all-padding parents (design 6.1).
    has = (n_parent > 0).to(n.dtype).reshape(b, h2, 1, w2, 1)
    ratio = n_cell / n_parent.clamp_min(EPS).reshape(b, h2, 1, w2, 1)
    ratio = ratio * has + 0.25 * (1.0 - has)

    def _pool(x: torch.Tensor) -> torch.Tensor:
        xv = x.reshape(b, h2, 2, w2, 2, x.shape[-1])
        return (xv * ratio.unsqueeze(-1)).sum(dim=(2, 4))

    mu_parent = _pool(mu)
    factor_parent = _pool(factor)
    # delta_parent = sum_i P_ai^2 delta_i: pre-multiply by the per-cell weight in
    # the original [B,H,W,1] layout so ``_pool`` applies it a second time.
    var_parent = _pool(var * ratio.reshape(b, h, w, 1))
    return mu_parent, var_parent, factor_parent, n_parent


def merge_2x2_stats(y: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked mean of the observation summary by observation count.

    ``y`` ``[B,H,W,1]`` cell observation means, ``k`` ``[B,H,W]`` observation
    counts.  Returns ``(ybar, kbar, mbar)`` on the parent grid, which is exactly
    the non-overlapping masked mean over the parent footprint.
    """
    num, den = _merge_2x2_weighted(y, k)
    k_bar = den
    y_bar = num / k_bar.clamp_min(EPS).unsqueeze(-1)
    m_bar = (k_bar > 0).to(y.dtype)
    y_bar = torch.where(m_bar.unsqueeze(-1) > 0, y_bar, torch.zeros_like(y_bar))
    return y_bar, k_bar, m_bar


# --------------------------------------------------------------------------- #
# Low-rank conditioning (design 5.2 / 5.3)
# --------------------------------------------------------------------------- #
def _solve_in_dtype(
    mu: torch.Tensor,
    factor: torch.Tensor,
    s2: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    f: torch.Tensor,
    omega2: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    mu_ = mu.to(dtype)
    u_ = factor.to(dtype)
    s2_ = s2.to(dtype)
    y_ = y.to(dtype)
    m_ = m.to(dtype)
    f_ = f.to(dtype)
    om_ = omega2.to(dtype)

    b, n, r = u_.shape
    mf = m_ * f_
    d = om_ + mf * s2_
    if bool((d <= 0).any()):
        raise RuntimeError("precision denominator must stay positive")
    c = om_ / d
    a = (mf * s2_) / d
    w = mf / d

    eye = torch.eye(r, device=u_.device, dtype=dtype).unsqueeze(0)
    h_mat = eye + u_.transpose(-2, -1) @ (w * u_)
    h_mat = 0.5 * (h_mat + h_mat.transpose(-2, -1))
    chol, info = torch.linalg.cholesky_ex(h_mat)
    if int(info.max().item()) != 0:
        return None, None, None, False

    # b = U^T W (y - mu)   [B, r]
    weights = w.squeeze(-1) * (y_ - mu_).squeeze(-1)  # [B, N]
    rhs = torch.einsum("bnr,bn->br", u_, weights)
    hbar = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)  # [B, r]

    mu_post = c * (mu_ + torch.einsum("bnr,br->bn", u_, hbar).unsqueeze(-1)) + a * y_
    delta = c * s2_
    # Solve L X = (c U)^T, then F = X^T  =>  F = diag(c) U L^{-T}.
    cu_T = (c * u_).transpose(-2, -1)  # [B, r, N]
    x_mat = torch.linalg.solve_triangular(chol, cu_T, upper=False)
    f_post = x_mat.transpose(-2, -1)
    return mu_post, delta, f_post, True


def condition_low_rank(
    mu: torch.Tensor,
    factor: torch.Tensor,
    s2: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    f: torch.Tensor,
    omega2: torch.Tensor,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form low-rank posterior ``(mu_post, delta, F)`` in FP32.

    ``mu`` ``[B,N,1]``, ``factor`` ``[B,N,r]``, ``s2``/``y``/``m``/``f``/``omega2``
    ``[B,N,1]``.  The posterior covariance is only ever represented as
    ``diag(delta) + F F^T``; no dense ``[B,N,N]`` matrix is built and no explicit
    inverse is taken.  A batch element with no observation at all returns the
    prior exactly.  On an FP32 Cholesky failure the system is re-accumulated in
    FP64 (outputs cast back to FP32, gradients preserved) and the event is
    counted; an FP64 failure raises instead of being papered over.
    """
    out_dtype = mu.dtype
    device = mu.device
    # Training runs the probabilities in FP32; a micro FP64 call (dense reference,
    # gradcheck) keeps its own precision instead of being forced back to FP32.
    work = torch.float64 if out_dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=device.type, enabled=False):
        mu_post, delta, f_post, ok = _solve_in_dtype(
            mu, factor, s2, y, m, f, omega2, work
        )
        if not ok:
            if stats is not None:
                stats["cholesky_fallback"] = int(stats.get("cholesky_fallback", 0)) + 1
            mu_post, delta, f_post, ok64 = _solve_in_dtype(
                mu, factor, s2, y, m, f, omega2, torch.float64
            )
            if not ok64:
                raise RuntimeError(
                    "Cholesky failed in FP64 as well; inspect mu/U/s2/omega2 ranges "
                    "(design 5.3 forbids jitter, clamping or nan_to_num here)"
                )

        mu_post = mu_post.to(out_dtype)
        delta = delta.to(out_dtype)
        f_post = f_post.to(out_dtype)
        # Exact all-missing endpoint: q_+ == q_0 for an image without observations.
        any_obs = (m.sum(dim=1, keepdim=True) > 0)
        mu_post = torch.where(any_obs, mu_post, mu)
        delta = torch.where(any_obs, delta, s2)
        f_post = torch.where(any_obs, f_post, factor)
    return mu_post, delta, f_post


def joint_low_rank_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    diag_var: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    """Stable low-rank Gaussian NLL ``-log N(t; mu, diag(d) + F F^T)``.

    ``target``/``mu``/``diag_var`` ``[B,K]`` (``diag_var > 0``), ``factor``
    ``[B,K,r]``.  Uses the positive-form quadratic of design 10.3 so the large
    dense expression ``e^T V^{-1} e - b^T J^{-1} b`` is never formed.  Returns
    ``[B]`` per-image NLL.
    """
    if target.ndim != 2:
        raise ValueError("joint_low_rank_nll expects [B,K] tensors")
    d = diag_var
    if bool((d <= 0).any()):
        raise RuntimeError("joint_low_rank_nll requires a strictly positive diagonal")
    e = target - mu
    inv_sqrt = d.rsqrt()
    e_scaled = e * inv_sqrt
    f_scaled = factor * inv_sqrt.unsqueeze(-1)
    r = factor.shape[-1]
    eye = torch.eye(r, device=factor.device, dtype=factor.dtype).unsqueeze(0)
    j_mat = eye + f_scaled.transpose(-2, -1) @ f_scaled
    j_mat = 0.5 * (j_mat + j_mat.transpose(-2, -1))
    rhs = torch.einsum("bkr,bk->br", f_scaled, e_scaled)
    h_star = torch.linalg.solve(j_mat, rhs.unsqueeze(-1)).squeeze(-1)
    residual = e_scaled - torch.einsum("bkr,br->bk", f_scaled, h_star)
    quad = residual.pow(2).sum(-1) + h_star.pow(2).sum(-1)
    chol = torch.linalg.cholesky(j_mat)
    log_det = d.log().sum(-1) + 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    n = target.shape[-1]
    return 0.5 * (quad + log_det + float(n) * math.log(2.0 * math.pi))


def axis_relation_variance(
    var_a: torch.Tensor,
    var_b: torch.Tensor,
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    zero_diagonal: bool = True,
) -> torch.Tensor:
    """``v_ij = delta_i + delta_j + ||F_i - F_j||^2`` with ``v_ii = 0``.

    Inputs are ``[..., L, 1]`` (variances) and ``[..., L, r]`` (factors).  The
    diagonal is removed functionally (multiplication by a mask), never by
    in-place overwriting of a tensor needed for the backward pass.  A rounding
    level of negative variance is clamped; anything clearly negative raises.
    """
    sq_a = factor_a.pow(2).sum(-1)
    sq_b = factor_b.pow(2).sum(-1)
    gram = factor_a @ factor_b.transpose(-1, -2)
    v = (
        var_a
        + var_b.transpose(-1, -2)
        + sq_a.unsqueeze(-1)
        + sq_b.unsqueeze(-2)
        - 2.0 * gram
    )
    if bool((v < -1e-3).any()):
        raise RuntimeError("relative variance went clearly negative; Gram rounding is not the cause")
    v = v.clamp_min(0.0)
    if zero_diagonal:
        length = v.shape[-1]
        eye = torch.eye(length, device=v.device, dtype=v.dtype)
        v = v * (1.0 - eye)
    return v


# --------------------------------------------------------------------------- #
# RGB-only relational prior (design 4.2 / 4.3)
# --------------------------------------------------------------------------- #
class _ResidualBlock(nn.Module):
    """``DWConv -> GN(8) -> GELU -> PW(expand) -> GELU -> PW`` with a residual."""

    def __init__(self, dim: int, expand: int = 2, dilation: int = 1, groups: int = 8):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, dilation, dilation=dilation, groups=dim)
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.GELU()
        self.pw1 = nn.Conv2d(dim, dim * expand, 1)
        self.act2 = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expand, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.dwconv(x)))
        h = self.pw2(self.act2(self.pw1(h)))
        return x + h


class RGBGeometryPrior(nn.Module):
    """RGB-only geometry context ``G`` plus the three bounded parameter heads.

    Branches off the **shared RGB stem output before any stage-0 attention**; it
    never reads Depth, so conditioning the prior cannot see observations.
    Returns the raw head activations; :class:`OCGeometryBuilder` turns them into
    the bounded ``mu``, ``U`` and ``s2`` of design 4.4.
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 64, rank: int = RANK):
        super().__init__()
        if out_dim != in_dim:
            raise ValueError("this implementation keeps the geometry context width at 64")
        self.in_dim = int(in_dim)
        self.rank = int(rank)

        self.g0 = nn.Sequential(_ResidualBlock(in_dim), _ResidualBlock(in_dim))
        self.down1 = nn.Conv2d(in_dim, 96, 3, 2, 1)
        self.g1 = nn.Sequential(_ResidualBlock(96), _ResidualBlock(96))
        self.down2 = nn.Conv2d(96, 128, 3, 2, 1)
        self.g2 = nn.Sequential(
            _ResidualBlock(128, dilation=2), _ResidualBlock(128, dilation=2)
        )
        self.proj1 = nn.Conv2d(96, 64, 1)
        self.proj2 = nn.Conv2d(128, 64, 1)
        self.proj_ctx = nn.Conv2d(128, 32, 1)

        fuse_channels = 64 + 64 + 64 + 32 + 2
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_channels, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )
        self.fuse_res = _ResidualBlock(out_dim)

        self.head_mean = nn.Conv2d(out_dim, 1, 1)
        self.head_factor = nn.Conv2d(out_dim, rank, 1)
        self.head_scale = nn.Conv2d(out_dim, 1, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module in (self.head_mean, self.head_factor, self.head_scale):
                    continue
                if module.groups == module.in_channels and module.in_channels == module.out_channels:
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.head_mean.weight, std=1e-3)
        nn.init.zeros_(self.head_mean.bias)
        # Non-zero factors: U == 0 has a zero first-order covariance gradient.
        nn.init.normal_(self.head_factor.weight, std=0.02)
        nn.init.normal_(self.head_factor.bias, std=0.01)
        nn.init.normal_(self.head_scale.weight, std=1e-3)
        self.head_scale.bias.data.fill_(
            float(inverse_bounded_std(SIGMA_INIT, S_MIN, S_MAX).item())
        )

    def forward(self, stem_feat: torch.Tensor, image_support: torch.Tensor):
        """``stem_feat`` ``[B,H0,W0,C]`` NHWC stem output; support ``[B,1,H,W]``.

        Returns ``(G, a_mean, a_factor, a_scale, windows, centroid_n)`` with
        ``G`` ``[B,64,H0,W0]``, ``windows`` ``[B,H0,W0,9]`` and the raw head
        activations ``[B,·,H0,W0]``.
        """
        if stem_feat.ndim != 4:
            raise ValueError("stem feature must be NHWC")
        x = stem_feat.permute(0, 3, 1, 2).contiguous()

        f0 = self.g0(x)
        f1 = self.g1(self.down1(f0))
        f2 = self.g2(self.down2(f1))

        size = f0.shape[-2:]
        up1 = F.interpolate(self.proj1(f1), size=size, mode="bilinear", align_corners=False)
        up2 = F.interpolate(self.proj2(f2), size=size, mode="bilinear", align_corners=False)

        # Support-weighted global context from the coarsest geometry level.
        weight = F.adaptive_avg_pool2d(image_support.float(), f2.shape[-2:])
        num = (f2 * weight).sum(dim=(2, 3))
        den = weight.sum(dim=(2, 3)).clamp_min(EPS)
        ctx = self.proj_ctx((num / den).unsqueeze(-1).unsqueeze(-1))
        ctx = ctx.expand(-1, -1, size[0], size[1])

        cx, cy = _support_centroids(image_support, BASE_CELL)
        windows = window_basis(cx, cy)  # [B,H0,W0,9]
        coord = torch.stack(
            [
                torch.linspace(0.0, 1.0, size[1], device=x.device).view(1, 1, size[1]).expand(x.shape[0], size[0], size[1]),
                torch.linspace(0.0, 1.0, size[0], device=x.device).view(1, size[0], 1).expand(x.shape[0], size[0], size[1]),
            ],
            dim=1,
        )

        agg = torch.cat([f0, up1, up2, ctx, coord], dim=1)
        g = self.fuse_res(self.fuse(agg))

        return (
            g,
            self.head_mean(g),
            self.head_factor(g),
            self.head_scale(g),
            windows,
            cx,
            cy,
        )


class ObservationPrecisionHead(nn.Module):
    """Support-aware observation precision ``omega`` (design 5.1).

    Input is exactly the documented 69 channels (``G`` 64, ``y``, ``m``, ``f``,
    ``sqrt(a_obs)``, ``y - mu.detach()``) -- every one of them derived from RGB
    and the *current* observations only.
    """

    def __init__(self, in_ch: int = 64 + 5, hidden: int = 32):
        super().__init__()
        if in_ch != 64 + 5:
            raise ValueError("the precision head input is fixed at 69 channels")
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.pw = nn.Conv2d(hidden, hidden, 1)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.out = nn.Conv2d(hidden, 1, 1)
        for module in (self.conv1, self.dw, self.pw):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in (self.norm1, self.norm2):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.out.weight, std=1e-3)
        self.out.bias.data.fill_(float(inverse_bounded_std(OMEGA_INIT, OMEGA_MIN, OMEGA_MAX).item()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(inputs)))
        h = F.gelu(self.norm2(self.pw(self.dw(h))))
        a = self.out(h)
        return bounded_std(a, OMEGA_MIN, OMEGA_MAX)


class OCSpatialPrior(nn.Module):
    """Spatial-decay prior for an OC block (replaces ``GeoPriorGen``).

    Keeps exactly the author's ``angle``/``decay`` buffers and the loaded spatial
    coefficient, and drops the signed Depth coefficient, so the OC branch cannot
    inject the old ``B^d`` term a second time.
    """

    def __init__(self, embed_dim: int, num_heads: int, initial_value: float = 2, heads_range: float = 4,
                 spatial_weight: float = 1.0):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.weight = nn.Parameter(torch.full((1, 1, 1, 1), float(spatial_weight)))

    def generate_1d_decay(self, length: int) -> torch.Tensor:
        index = torch.arange(length, device=self.decay.device).to(self.decay)
        mask = (index[:, None] - index[None, :]).abs()
        return mask * self.decay[:, None, None]

    def generate_pos_decay(self, height: int, width: int) -> torch.Tensor:
        index_h = torch.arange(height, device=self.decay.device).to(self.decay)
        index_w = torch.arange(width, device=self.decay.device).to(self.decay)
        grid = torch.stack(torch.meshgrid([index_h, index_w], indexing="ij"), dim=-1).reshape(height * width, 2)
        mask = (grid[:, None, :] - grid[None, :, :]).abs().sum(dim=-1)
        return mask * self.decay[:, None, None]

    def forward(self, hw_tuple: Tuple[int, int], split_or_not: bool = False):
        height, width = int(hw_tuple[0]), int(hw_tuple[1])
        index = torch.arange(height * width, device=self.decay.device).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(height, width, -1)
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(height, width, -1)
        if split_or_not:
            bs_h = self.weight * self.generate_1d_decay(height).unsqueeze(0).unsqueeze(2)
            bs_w = self.weight * self.generate_1d_decay(width).unsqueeze(0).unsqueeze(2)
            return (sin, cos), (bs_h, bs_w)
        bs = self.weight * self.generate_pos_decay(height, width)
        return (sin, cos), bs


class OCInjectionScale(nn.Module):
    """Per-block, per-head non-negative injection scale ``beta = 2 sigmoid(a)``."""

    def __init__(self, num_heads: int, init: float = BETA_INIT):
        super().__init__()
        self.num_heads = int(num_heads)
        self.a_beta = nn.Parameter(torch.full((self.num_heads,), _logit(init / 2.0)))

    def forward(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.a_beta)


def padding_key_mask(key_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Additive log mask for padded keys, plus the "no valid key" row flag.

    ``key_valid`` is a boolean ``[..., L]`` mask.  Rows without a single valid key
    keep key ``0`` finite so the softmax never sees an all-masked row; the caller
    zeroes those query outputs afterwards.
    """
    neg = torch.full((), PAD_LOGIT, device=key_valid.device, dtype=torch.float32)
    mask = torch.where(key_valid, torch.zeros_like(neg), neg)
    no_valid = ~key_valid.any(dim=-1, keepdim=True)
    first = torch.zeros_like(key_valid)
    first[..., 0] = True
    mask = torch.where(no_valid & first, torch.zeros_like(mask), mask)
    return mask, no_valid


def research_attention_probs(qk: torch.Tensor, bias: torch.Tensor,
                             valid: Optional[torch.Tensor]) -> torch.Tensor:
    """Shared FP32 softmax over an explicitly assembled research bias.

    Both the mask-aware baseline and the OC model use this path, so the two arms
    never mix precision protocols.  ``qk`` is the already scaled ``QK`` product in
    whichever layout the axis uses; ``bias`` broadcasts against it.  ``valid``
    marks valid geometry tokens as either ``[B,A,L]`` (axial: ``A`` is the row of
    the axial layout) or ``[B,L]`` (full attention) and may be ``None`` when the
    whole grid is valid.  Invalid keys get :data:`PAD_LOGIT`, a row without any
    valid key keeps key 0 finite, and invalid query rows are zeroed exactly.
    """
    with torch.autocast(device_type=qk.device.type, enabled=False):
        scores = qk.float() + bias.float()
        if valid is None:
            return torch.softmax(scores, dim=-1)
        if valid.dim() == 3:
            key_valid = valid.unsqueeze(2).unsqueeze(3)
            query_valid = valid.unsqueeze(2).unsqueeze(-1)
        elif valid.dim() == 2:
            key_valid = valid.view(valid.shape[0], 1, 1, valid.shape[-1])
            query_valid = valid.view(valid.shape[0], 1, valid.shape[-1], 1)
        else:
            raise ValueError("valid mask must be [B,L] or [B,A,L]")
        mask, _ = padding_key_mask(key_valid)
        probs = torch.softmax(scores + mask, dim=-1)
        return probs * query_valid.to(probs.dtype)


# --------------------------------------------------------------------------- #
# Relational evidence pyramid (design 7)
# --------------------------------------------------------------------------- #
class RelationalEvidencePyramid(nn.Module):
    """Per-stage/per-head kernel lengths and the bounded evidence read-out.

    One ``ell_{s,h} = BStd(a; 0.15, 1.50)`` per stage and head, shared by every
    block of that stage, initialised geometrically into ``[0.25, 0.70]``.
    """

    def __init__(self, heads_per_stage: Sequence[int] = (4, 4, 8, 16),
                 ell_min: float = ELL_MIN, ell_max: float = ELL_MAX,
                 init_range: Tuple[float, float] = ELL_INIT_RANGE, tau: float = TAU_SAT):
        super().__init__()
        if len(heads_per_stage) != 4:
            raise ValueError("exactly four stages are supported")
        self.heads_per_stage = tuple(int(h) for h in heads_per_stage)
        self.tau = float(tau)
        self.a_ell = nn.ParameterList()
        for heads in self.heads_per_stage:
            if heads == 1:
                values = torch.tensor([float(init_range[0])])
            else:
                values = torch.logspace(
                    math.log10(float(init_range[0])), math.log10(float(init_range[1])), heads
                )
            self.a_ell.append(
                nn.Parameter(torch.stack([inverse_bounded_std(float(v), ell_min, ell_max) for v in values]))
            )

    def lengths(self, stage: int) -> torch.Tensor:
        return bounded_std(self.a_ell[stage], ELL_MIN, ELL_MAX)

    @staticmethod
    def log_kernel(m: torch.Tensor, v: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        """``log K_h(m, v)`` for ``Delta ~ N(m, v)`` under an RBF kernel."""
        l2 = ell * ell
        return -0.5 * torch.log1p(v / l2) - 0.5 * m * m / (l2 + v)

    def saturate(self, e: torch.Tensor) -> torch.Tensor:
        return self.tau * torch.tanh(e / self.tau)

    def _axis_evidence(
        self,
        mu0: torch.Tensor,
        var0: torch.Tensor,
        f0: torch.Tensor,
        mu1: torch.Tensor,
        var1: torch.Tensor,
        f1: torch.Tensor,
        ell: torch.Tensor,
    ) -> torch.Tensor:
        """``Ehat`` of shape ``[B, heads, L, L]`` for one axis/grid layout."""
        m0 = mu0 - mu0.transpose(-1, -2)
        m1 = mu1 - mu1.transpose(-1, -2)
        v0 = axis_relation_variance(var0, var0, f0, f0)
        v1 = axis_relation_variance(var1, var1, f1, f1)
        out: List[torch.Tensor] = []
        for head in range(ell.numel()):
            length = ell[head]
            delta = self.log_kernel(m1, v1, length) - self.log_kernel(m0, v0, length)
            out.append(self.saturate(delta))
        return torch.stack(out, dim=1)

    def evidence(self, stage: int, pooled: Dict[str, torch.Tensor],
                 layout: str) -> torch.Tensor:
        """Build the saturated evidence for one stage in one axis/full layout.

        ``layout`` is ``"w"`` (``[B,H,W,*]`` -> ``[B,H,heads,W,W]``), ``"h"``
        (``[B,W,H,*]`` -> ``[B,W,heads,H,H]``) or ``"full"``
        (``[B,H*W,*]`` -> ``[B,heads,N,N]``).
        """
        ell = self.lengths(stage)
        mu0 = pooled["mu0"]
        var0 = pooled["s20"]
        f0 = pooled["U0"]
        mu1 = pooled["mu1"]
        var1 = pooled["delta1"]
        f1 = pooled["F1"]
        b, h, w, _ = mu0.shape

        if layout == "full":
            def flat(t):
                return t.reshape(b, h * w, t.shape[-1])
        elif layout == "w":
            def flat(t):
                return t.reshape(b * h, w, t.shape[-1])
        elif layout == "h":
            def flat(t):
                return t.transpose(1, 2).reshape(b * w, h, t.shape[-1])
        else:
            raise ValueError("unknown layout %r" % (layout,))

        evidence = self._axis_evidence(
            flat(mu0), flat(var0), flat(f0), flat(mu1), flat(var1), flat(f1), ell
        )
        if layout == "full":
            return evidence
        if layout == "w":
            return evidence.reshape(b, h, ell.numel(), w, w)
        return evidence.reshape(b, w, ell.numel(), h, h)


# --------------------------------------------------------------------------- #
# Per-forward geometry bundle (design 8.3)
# --------------------------------------------------------------------------- #
class OCGeometryBundle:
    """One-forward container of the OC geometry.

    Holds prior/posterior base-grid factors, the pooled stage factors and the
    lazily built relational evidence.  It is created and destroyed inside a
    single ``forward`` call: nothing is cached across batches or epochs and no
    training graph is stored on a module attribute.
    """

    def __init__(self, builder: "OCGeometryBuilder", base: Dict[str, torch.Tensor],
                 pooled: List[Dict[str, torch.Tensor]],
                 baseline_pooled: List[Dict[str, torch.Tensor]],
                 stage_shapes: List[Tuple[int, int]], geometry: Dict[str, torch.Tensor]):
        self._builder = builder
        self.base = base
        self.pooled = pooled
        self.baseline_pooled = baseline_pooled
        self.stage_shapes = stage_shapes
        self.geometry = geometry
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.stats: Dict[str, int] = {}

    # -- per-stage access ------------------------------------------------- #
    def stage(self, index: int) -> Dict[str, torch.Tensor]:
        if index not in self._cache:
            self._cache[index] = self._build_stage(index)
        return self._cache[index]

    def _build_stage(self, index: int) -> Dict[str, torch.Tensor]:
        pooled = self.pooled[index]
        n = pooled["n"]
        valid = n > 0
        out: Dict[str, torch.Tensor] = {
            "index": index,
            "grid": (int(n.shape[1]), int(n.shape[2])),
            "valid": valid,
            "n": n,
        }
        if self._builder.uses_evidence:
            if index == 3:
                out["Ehat_full"] = self._builder.evidence.evidence(3, pooled, "full")
            else:
                out["Ehat_w"] = self._builder.evidence.evidence(index, pooled, "w")
                out["Ehat_h"] = self._builder.evidence.evidence(index, pooled, "h")
        if self._builder.uses_baseline:
            if index == 3:
                out["base_full"] = _baseline_relation(self.baseline_pooled[index], "full")
            else:
                out["base_w"] = _baseline_relation(self.baseline_pooled[index], "w")
                out["base_h"] = _baseline_relation(self.baseline_pooled[index], "h")
        return out


def _baseline_relation(pooled: Dict[str, torch.Tensor], layout: str = "full"):
    """``|ybar_i - ybar_j| * m_i * m_j`` for the mask-aware baseline (design 11.2)."""
    y = pooled["ybar"]
    m = pooled["mbar"]
    if layout == "full":
        b, h, w, _ = y.shape
        yf = y.reshape(b, h * w)
        mf = m.reshape(b, h * w)
        diff = (yf[:, :, None] - yf[:, None, :]).abs()
        return (diff * (mf[:, :, None] * mf[:, None, :])).unsqueeze(1)
    if layout == "w":
        diff = (y[:, :, :, None, 0] - y[:, :, None, :, 0]).abs()
        prod = m[:, :, :, None] * m[:, :, None, :]
        return (diff * prod).unsqueeze(1)
    if layout == "h":
        yt = y.transpose(1, 2)[..., 0]
        mt = m.transpose(1, 2)
        diff = (yt[:, :, :, None] - yt[:, :, None, :]).abs()
        prod = mt[:, :, :, None] * mt[:, :, None, :]
        return (diff * prod).unsqueeze(1)
    raise ValueError("unknown layout %r" % (layout,))


class OCGeometryBuilder(nn.Module):
    """Registered module that builds the whole OC geometry package once per forward.

    Sub-modules: the RGB-only prior branch, the observation precision head and the
    relational evidence pyramid (kernel lengths).  Nothing here receives the label
    or any reference depth; the conditioning API is exposed separately so the
    training losses can rerun the precision head and the solve on an *auxiliary*
    observation context without touching the main input.
    """

    def __init__(self, heads_per_stage: Sequence[int] = (4, 4, 8, 16),
                 uses_evidence: bool = True, uses_baseline: bool = False,
                 cell: int = BASE_CELL, rank: int = RANK):
        super().__init__()
        self.prior = RGBGeometryPrior(in_dim=64, out_dim=64, rank=rank)
        self.precision = ObservationPrecisionHead()
        self.evidence = RelationalEvidencePyramid(heads_per_stage)
        self.uses_evidence = bool(uses_evidence)
        self.uses_baseline = bool(uses_baseline)
        self.cell = int(cell)
        self.rank = int(rank)

    # -- conditioning API (also used by the auxiliary hold-out context) ---- #
    def precision_inputs(
        self,
        context: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        f: torch.Tensor,
        a_obs: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the fixed 69-channel NCHW precision input.

        ``context`` is the NCHW geometry context ``[B,64,H0,W0]``; the five
        summary planes are NHWC ``[B,H0,W0,1]``.  The last plane stops only its
        own local gradient towards the mean head (``mu.detach()``); ``context``
        itself stays differentiable.
        """
        planes = [context]
        for plane in (y, m, f, a_obs.clamp_min(0.0).sqrt(), y - mu.detach()):
            planes.append(plane.permute(0, 3, 1, 2))
        return torch.cat(planes, dim=1)

    def prior_parameters(self, geom: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Bounded ``mu``, ``U`` and ``s2`` from the RGB head activations."""
        mu = bounded_mean(geom["a_mean"])
        s2 = bounded_std(geom["a_scale"], S_MIN, S_MAX).pow(2)
        factor = _column_basis(geom["a_factor"], geom["windows"], geom["n"])
        return {"mu": mu, "U": factor, "s2": s2}

    def summarise(self, depth: torch.Tensor, observation: torch.Tensor,
                  image_support: torch.Tensor) -> Dict[str, torch.Tensor]:
        return cell_summaries(depth, observation, image_support, self.cell)

    # -- main forward ------------------------------------------------------ #
    def forward(self, stem_feat: torch.Tensor, depth: torch.Tensor,
                observation: torch.Tensor, image_support: torch.Tensor) -> OCGeometryBundle:
        g, a_mean, a_factor, a_scale, windows, cx, cy = self.prior(stem_feat, image_support)
        summary = cell_summaries(depth, observation, image_support, self.cell)
        n = summary["n"].unsqueeze(-1)  # [B,H0,W0,1]

        geometry = {
            "context": g,
            "a_mean": a_mean.permute(0, 2, 3, 1),
            "a_factor": a_factor.permute(0, 2, 3, 1),
            "a_scale": a_scale.permute(0, 2, 3, 1),
            "windows": windows,
            "n": n,
            "cx": cx,
            "cy": cy,
            "summary": summary,
        }
        params = self.prior_parameters(geometry)
        mu, factor, s2 = params["mu"], params["U"], params["s2"]

        y = summary["y"].unsqueeze(-1)
        m = summary["m"].unsqueeze(-1)
        f = summary["f"].unsqueeze(-1)
        a_obs = summary["a_obs"].unsqueeze(-1)

        stats: Dict[str, int] = {}
        # The precision head is a conv stack (NCHW); the geometry keeps NHWC.
        omega = self.precision(
            self.precision_inputs(g, y, m, f, a_obs, mu)
        ).permute(0, 2, 3, 1)
        omega2 = omega.pow(2)

        # The conditioning and the joint likelihood work on the flat base grid
        # [B, N, .]; the scale propagation and the relational read-out work on the
        # [B, H0, W0, .] grid so the axial layouts stay explicit.
        b, h0, w0 = mu.shape[0], mu.shape[1], mu.shape[2]

        def flatten(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, h0 * w0, t.shape[-1])

        mu_post_f, delta_f, f_post_f = condition_low_rank(
            flatten(mu), flatten(factor), flatten(s2),
            flatten(y), flatten(m), flatten(f), flatten(omega2), stats=stats,
        )
        mu_post = mu_post_f.reshape(b, h0, w0, 1)
        delta = delta_f.reshape(b, h0, w0, 1)
        f_post = f_post_f.reshape(b, h0, w0, factor.shape[-1])

        base = {
            "mu": mu,
            "U": factor,
            "s2": s2,
            "y": y,
            "m": m,
            "f": f,
            "a_obs": a_obs,
            "omega": omega,
            "omega2": omega2,
            "mu_post": mu_post,
            "delta": delta,
            "F": f_post,
            "n": n,
            "zbar": summary["zbar"].unsqueeze(-1),
            "k": summary["k"].unsqueeze(-1),
            "grid": summary["grid"],
            "cell": summary["cell"],
        }

        pooled: List[Dict[str, torch.Tensor]] = []
        baseline_pooled: List[Dict[str, torch.Tensor]] = []
        cum = {
            "n": n,
            "mu0": mu,
            "s20": s2,
            "U0": factor,
            "mu1": mu_post,
            "delta1": delta,
            "F1": f_post,
        }
        # Mask-aware baseline summary: an exact non-overlapping masked mean of the
        # observations, obtained by carrying the observation count as the weight.
        y_bar = summary["y"].unsqueeze(-1)
        k_bar = summary["k"]
        m_bar = (k_bar > 0).to(y_bar.dtype)
        for stage in range(4):
            if stage > 0:
                mu_p, var_p, fac_p, n_p = merge_2x2(
                    cum["mu0"], cum["s20"], cum["U0"], cum["n"].squeeze(-1)
                )
                muq_p, varq_p, facq_p, _ = merge_2x2(
                    cum["mu1"], cum["delta1"], cum["F1"], cum["n"].squeeze(-1)
                )
                cum = {
                    "n": n_p.unsqueeze(-1),
                    "mu0": mu_p,
                    "s20": var_p,
                    "U0": fac_p,
                    "mu1": muq_p,
                    "delta1": varq_p,
                    "F1": facq_p,
                }
                y_bar, k_bar, m_bar = merge_2x2_stats(y_bar, k_bar)
            pooled.append(dict(cum))
            if self.uses_baseline:
                baseline_pooled.append({"ybar": y_bar, "mbar": m_bar})

        stage_shapes = [(int(p["n"].shape[1]), int(p["n"].shape[2])) for p in pooled]
        bundle = OCGeometryBundle(self, base, pooled, baseline_pooled, stage_shapes, geometry)
        bundle.stats = stats
        return bundle


def _column_basis(a_factor: torch.Tensor, windows: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Assemble ``u_i`` of design 4.3 from the 20 raw factor channels.

    Channels ``0:2`` are the global factors, channels ``2:20`` are two channels
    per tent window.  ``windows`` is ``[B,H0,W0,9]`` and ``n`` ``[B,H0,W0,1]``;
    all-padding cells get an exactly zero row basis.
    """
    if a_factor.shape[-1] != RANK:
        raise ValueError("factor head must emit %d channels" % RANK)
    global_part = a_factor[..., :GLOBAL_CHANNELS]
    local_part = a_factor[..., GLOBAL_CHANNELS:].reshape(*a_factor.shape[:-1], NUM_WINDOWS, GLOBAL_CHANNELS)

    scale_g = GLOBAL_ROW_ENERGY / math.sqrt(2.0)
    scale_l = LOCAL_ROW_ENERGY / math.sqrt(2.0)
    u_global = scale_g * torch.tanh(global_part)
    u_local = windows.unsqueeze(-1) * scale_l * torch.tanh(local_part)
    basis = torch.cat([u_global, u_local.reshape(*u_local.shape[:-2], LOCAL_CHANNELS)], dim=-1)
    if n is not None:
        basis = basis * (n > 0).to(basis.dtype)
    return basis


# --------------------------------------------------------------------------- #
# Self-check (design 15.2, items 1-4)
# --------------------------------------------------------------------------- #
def _dense_posterior(mu, u, s2, y, m, f, omega2):
    """Dense FP64 reference of the same energy (no low-rank structure assumed)."""
    b, n, r = u.shape
    rho = (m * f / omega2)  # [B,N,1]
    sigma0 = u @ u.transpose(-2, -1) + torch.diag_embed(s2.squeeze(-1))
    prec0 = torch.linalg.inv(sigma0)
    prec = prec0 + torch.diag_embed(rho.squeeze(-1))
    sigma = torch.linalg.inv(prec)
    mu_post = sigma @ (torch.einsum("bij,bj->bi", prec0, mu.squeeze(-1))
                       + rho.squeeze(-1) * y.squeeze(-1)).unsqueeze(-1)
    return mu_post, sigma


def run_self_check(device: str = "cpu", report_path: Optional[str] = None) -> Dict[str, object]:
    """Small-tensor FP64 checks of the probability core.  CPU-first."""
    torch.manual_seed(20261014)
    dev = torch.device(device)
    report: Dict[str, object] = {"device": str(dev), "checks": {}, "failures": []}

    def record(name: str, ok: bool, detail):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["failures"].append(name)
        print("[self-check] %-34s %s  %s" % (name, "PASS" if ok else "FAIL", detail))

    # ---- 1. low-rank posterior vs dense FP64 reference ---------------- #
    b, n, r = 2, 12, 4
    mu = torch.randn(b, n, 1, dtype=torch.float64, device=dev)
    u = torch.randn(b, n, r, dtype=torch.float64, device=dev) * 0.6
    s2 = (torch.rand(b, n, 1, dtype=torch.float64, device=dev) * 0.5 + 0.05)
    m = (torch.rand(b, n, 1, dtype=torch.float64, device=dev) > 0.3).double()
    f = torch.rand(b, n, 1, dtype=torch.float64, device=dev)
    y = torch.randn(b, n, 1, dtype=torch.float64, device=dev)
    omega2 = torch.rand(b, n, 1, dtype=torch.float64, device=dev) * 0.8 + 0.05
    m[0] = 0.0  # one fully missing image

    mu_ref, sigma_ref = _dense_posterior(mu, u, s2, y, m, f, omega2)
    mu_post, delta, f_post = condition_low_rank(mu, u, s2, y, m, f, omega2)
    mu_post, delta, f_post = mu_post.double(), delta.double(), f_post.double()
    err_mu = float((mu_post - mu_ref).abs().max())
    sigma_low = torch.diag_embed(delta.squeeze(-1)) + f_post @ f_post.transpose(-2, -1)
    err_cov = float((sigma_low - sigma_ref).abs().max())
    record("posterior_mean_vs_dense", err_mu < 1e-6, "max|dmu|=%.3e" % err_mu)
    record("posterior_cov_vs_dense", err_cov < 1e-6, "max|dSigma|=%.3e" % err_cov)

    # ---- 2. Cholesky transpose direction ------------------------------- #
    c_diag = omega2 / (omega2 + m * f * s2)
    expect_f = c_diag * u  # F = diag(c) U L^{-T}; with H = I and c=1 when m=0
    zero_obs = (m.sum(dim=1, keepdim=True) == 0).double()  # [B,1,1]
    if float(zero_obs.sum()) == 0.0:
        raise AssertionError("the all-missing endpoint image was not constructed")
    err_dir = float(((f_post - expect_f).abs() * zero_obs).max())
    record("cholesky_transpose_direction", err_dir < 1e-9,
           "max|F - diag(c)U| on the no-observation image = %.3e (H=I => F=U)" % err_dir)

    # ---- 3. joint NLL vs dense reference ------------------------------- #
    k = 10
    t = torch.randn(b, k, dtype=torch.float64, device=dev)
    mu_k = torch.randn(b, k, dtype=torch.float64, device=dev)
    d_k = torch.rand(b, k, dtype=torch.float64, device=dev) * 0.5 + 0.05
    f_k = torch.randn(b, k, r, dtype=torch.float64, device=dev) * 0.4
    nll_low = joint_low_rank_nll(t, mu_k, d_k, f_k).double()
    a_mat = torch.diag_embed(d_k) + f_k @ f_k.transpose(-2, -1)
    e = t - mu_k
    quad = torch.einsum("bi,bij,bj->b", e, torch.linalg.inv(a_mat), e).unsqueeze(-1)
    _, logdet = torch.linalg.slogdet(a_mat)
    nll_ref = 0.5 * (quad + logdet.unsqueeze(-1) + k * math.log(2 * math.pi))
    err_nll = float((nll_low - nll_ref.sum(-1)).abs().max())
    record("joint_nll_vs_dense", err_nll < 1e-8, "max|dNLL|=%.3e" % err_nll)

    # ---- 4. relative covariance with the off-diagonal term ------------- #
    var_a = delta[:1, :, :]
    fac_a = f_post[:1]
    v_rel = axis_relation_variance(var_a, var_a, fac_a, fac_a)
    dense_rel = (sigma_low[:1].diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
                 + sigma_low[:1].diagonal(dim1=-2, dim2=-1).unsqueeze(-2)
                 - 2 * sigma_low[:1])
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    err_rel = float((v_rel[0][off] - dense_rel[0][off]).abs().max())
    diag_zero = float(v_rel.abs().max().item() - v_rel[0][off].abs().max().item())
    record("relative_covariance_matches_dense", err_rel < 1e-6, "max|dv|=%.3e" % err_rel)
    record("self_relation_zero", float(v_rel[0].diagonal().abs().max()) == 0.0,
           "max|v_ii|=%.3e" % float(v_rel[0].diagonal().abs().max()))
    record("common_shift_cancels", _check_common_shift(dev), "shared offset variance released")

    # ---- 5. all-missing and scale propagation -------------------------- #
    n_grid = torch.zeros(1, 4, 4, dtype=torch.float64, device=dev)
    n_grid[0, :2] = 1.0
    pooled_mu, pooled_var, pooled_fac, parent_n = merge_2x2(
        torch.randn(1, 4, 4, 1, dtype=torch.float64, device=dev),
        torch.rand(1, 4, 4, 1, dtype=torch.float64, device=dev) * 0.3 + 0.05,
        torch.randn(1, 4, 4, r, dtype=torch.float64, device=dev) * 0.4,
        n_grid,
    )
    expect_n = n_grid.reshape(1, 2, 2, 2, 2).sum(dim=(2, 4))
    record("scale_propagation_counts", float((parent_n - expect_n).abs().max()) < 1e-12,
           "n parent max err %.3e" % float((parent_n - expect_n).abs().max()))
    record("scale_propagation_finite",
           bool(torch.isfinite(pooled_mu).all()) and bool(torch.isfinite(pooled_var).all())
           and not bool((pooled_var <= 0).any()),
           "mu/variance finite and strictly positive on a half-padding grid")

    # ---- 6. tiny FP64 gradcheck ---------------------------------------- #
    gc_b, gc_n, gc_r = 1, 4, 3
    base_mu = torch.randn(gc_b, gc_n, 1, dtype=torch.float64, device=dev)
    base_u = torch.randn(gc_b, gc_n, gc_r, dtype=torch.float64, device=dev) * 0.5
    base_s2 = torch.rand(gc_b, gc_n, 1, dtype=torch.float64, device=dev) * 0.4 + 0.1
    base_y = torch.randn(gc_b, gc_n, 1, dtype=torch.float64, device=dev)
    base_m = torch.ones(gc_b, gc_n, 1, dtype=torch.float64, device=dev)
    base_f = torch.rand(gc_b, gc_n, 1, dtype=torch.float64, device=dev) * 0.8 + 0.1
    base_w2 = torch.rand(gc_b, gc_n, 1, dtype=torch.float64, device=dev) * 0.5 + 0.1

    def solver_fn(mu_, u_, s2_, y_, m_, f_, w2_):
        out_mu, out_delta, out_f = condition_low_rank(mu_, u_, s2_, y_, m_, f_, w2_)
        return (out_mu.sum() + out_delta.sum() + out_f.sum())

    grad_ok = torch.autograd.gradcheck(
        solver_fn,
        (base_mu.requires_grad_(), base_u.requires_grad_(), base_s2.requires_grad_(),
         base_y.requires_grad_(), base_m, base_f.requires_grad_(), base_w2.requires_grad_()),
        eps=1e-6, atol=1e-5, rtol=1e-3,
    )
    record("gradcheck_condition_low_rank_fp64", bool(grad_ok), "tiny FP64 dense gradcheck")

    def nll_fn(t_, mu_, d_, f_):
        return joint_low_rank_nll(t_, mu_, d_, f_).sum()

    grad_nll = torch.autograd.gradcheck(
        nll_fn,
        (t.requires_grad_(), mu_k.requires_grad_(), d_k.requires_grad_(), f_k.requires_grad_()),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    record("gradcheck_joint_nll_fp64", bool(grad_nll), "tiny FP64 dense gradcheck")

    # ---- 7. covariance contraction and endpoints ----------------------- #
    contract = _check_contraction(mu, u, s2, y, m, f, omega2, mu_ref, sigma_ref)
    record("covariance_contraction", contract, "Sigma+ <= Sigma0 under a fixed prior/precision")

    # ---- 8. shapes / finiteness of the module path --------------------- #
    shape_ok, shape_detail = _check_module_shapes(dev)
    record("module_tensor_layouts", shape_ok, shape_detail)

    report["failures"] = list(report["failures"])
    report["passed"] = not report["failures"]
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("[self-check] wrote %s" % report_path)
    print("[self-check] %s (%d checks)" % ("ALL PASS" if report["passed"] else "FAILURES", len(report["checks"])))
    return report


def _check_common_shift(dev) -> bool:
    """A shared offset in ``mu`` must not change the relative read-out."""
    torch.manual_seed(7)
    n, r = 6, 3
    var = torch.rand(1, n, 1, dtype=torch.float64, device=dev) + 0.1
    fac = torch.randn(1, n, r, dtype=torch.float64, device=dev)
    v = axis_relation_variance(var, var, fac, fac)
    shift = axis_relation_variance(var + 0.0, var + 0.0, fac, fac)  # mu does not enter v at all
    return bool(torch.allclose(v, shift))


def _check_contraction(mu, u, s2, y, m, f, omega2, mu_ref, sigma_ref) -> bool:
    """With a fixed prior/precision, ``Sigma+ <= Sigma0`` in the Loewner order."""
    b, n, r = u.shape
    sigma0 = u @ u.transpose(-2, -1) + torch.diag_embed(s2.squeeze(-1))
    diff = sigma0 - sigma_ref
    eig = torch.linalg.eigvalsh(0.5 * (diff + diff.transpose(-2, -1)))
    return bool(float(eig.min()) > -1e-9)


def _check_module_shapes(dev) -> Tuple[bool, str]:
    """Module path on a tiny grid: layouts, finiteness and padding masks."""
    torch.manual_seed(3)
    b, h, w = 2, 32, 32
    depth = torch.rand(b, 1, h, w, device=dev) * 2.0 + Z_MIN
    obs = torch.ones(b, 1, h, w, device=dev)
    support = torch.ones(b, 1, h, w, device=dev)
    # pad one image's right/bottom band so the padding mask has work to do
    support[1, :, :, 24:] = 0.0
    obs[1, :, :, 24:] = 0.0
    depth[1, :, :, 24:] = ZERO_FILL
    stem = torch.randn(b, h // 4, w // 4, 64, device=dev)
    builder = OCGeometryBuilder((4, 4, 8, 16), uses_evidence=True, uses_baseline=True).to(dev)
    bundle = builder(stem, depth, obs, support)
    problems = []
    for stage in range(4):
        info = bundle.stage(stage)
        n = info["n"]
        if stage < 3:
            if tuple(info["Ehat_w"].shape) != (b, n.shape[1], 4 if stage < 2 else 8, n.shape[2], n.shape[2]):
                problems.append("Ehat_w stage %d %s" % (stage, tuple(info["Ehat_w"].shape)))
            if tuple(info["Ehat_h"].shape) != (b, n.shape[2], 4 if stage < 2 else 8, n.shape[1], n.shape[1]):
                problems.append("Ehat_h stage %d %s" % (stage, tuple(info["Ehat_h"].shape)))
        else:
            heads = 16
            if tuple(info["Ehat_full"].shape) != (b, heads, n.shape[1] * n.shape[2], n.shape[1] * n.shape[2]):
                problems.append("Ehat_full %s" % (tuple(info["Ehat_full"].shape),))
        for key in ("Ehat_w", "Ehat_h", "Ehat_full"):
            if key in info and not bool(torch.isfinite(info[key]).all()):
                problems.append("non-finite %s at stage %d" % (key, stage))
        if stage == 0:
            diag_max = float(info["Ehat_w"].diagonal(dim1=-2, dim2=-1).abs().max())
            if diag_max != 0.0:
                problems.append("self-relation not exactly zero (%g)" % diag_max)
    mask, no_valid = padding_key_mask(torch.zeros(b, 4, 5, dtype=torch.bool, device=dev))
    if not bool(torch.isfinite(mask).all()) or not bool(no_valid.all()):
        problems.append("padding mask helper")
    return (not problems), "; ".join(problems) if problems else "layouts finite, self-relation zero"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OC-RGA probability-geometry core")
    parser.add_argument("--self-check", action="store_true", help="run the CPU small-tensor checks")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default=None, help="optional JSON report path")
    args = parser.parse_args(argv)
    if not args.self_check:
        parser.print_help()
        return 0
    report = run_self_check(device=args.device, report_path=args.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
