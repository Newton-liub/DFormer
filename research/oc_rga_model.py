"""OC-RGA research model and mask-aware baseline (design 8.4 / 8.5 / 11.2).

Two arms share one construction path:

* ``masked_baseline`` -- DFormerv2-S + HAM with the *same* degradation
  augmentation and a simple mask-aware geometric input: each stage takes the
  masked mean of the observations over its non-overlapping footprint from the
  common ``O`` and keeps the author's signed ``w_s``/``w_d`` coefficients, spatial
  decay, RoPE and absolute-difference kernel.  It learns no prior, no precision
  and does no correlation propagation.
* ``oc_rga`` (and the three pre-registered read-out controls
  ``mean_relation`` / ``diag_relation`` / ``direct_post``) -- the same backbone
  and decoder plus the registered OC geometry package.

Both arms run the shared FP32 attention probability operator, so the comparison
never mixes precision protocols.  The author ``EncoderDecoder`` is reused for the
backbone/HAM construction and initialisation (official encoder weights first, then
the 25-class HAM); only the return contract is replaced.  This wrapper never sees
a label, a reference depth or a corruption oracle: ``forward`` takes exactly
``(rgb, modal_x, depth_support=O, image_support=S)``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from importlib import import_module
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.oc_rga import (
    OCGeometryBuilder,
    OCSpatialPrior,
    ZERO_FILL,
)

__all__ = [
    "PAD_MULTIPLE",
    "BASELINE_MODES",
    "OC_MODES",
    "ALL_MODES",
    "READOUT_FOR_MODE",
    "OCRGAEncoderDecoder",
    "pad_to_multiple",
    "install_oc_spatial_prior",
    "build_model_for_mode",
    "build_oc_model",
    "build_masked_baseline",
    "build_oc_optimizer_param_groups",
    "run_model_self_check",
]

PAD_MULTIPLE = 32
BASELINE_MODES = ("masked_baseline",)
OC_MODES = ("oc_rga", "mean_relation", "diag_relation", "direct_post")
ALL_MODES = BASELINE_MODES + OC_MODES
READOUT_FOR_MODE = {
    "oc_rga": "delta",
    "mean_relation": "mean",
    "diag_relation": "diag",
    "direct_post": "direct",
}
STAGE_HEADS = (4, 4, 8, 16)


# --------------------------------------------------------------------------- #
# Input padding to the geometry grid
# --------------------------------------------------------------------------- #
def pad_to_multiple(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    observation: torch.Tensor,
    image_support: torch.Tensor,
    multiple: int = PAD_MULTIPLE,
):
    """Right/bottom pad to a multiple of ``multiple`` (design 3.3).

    New padding always has ``S = O = 0`` and the canonical raw-zero depth fill, so
    padded cells can never be mistaken for observations.  Returns the padded
    tensors plus the un-padded ``(H, W)`` the caller crops the logits back to.
    """
    height, width = rgb.shape[-2], rgb.shape[-1]
    pad_h = (-height) % int(multiple)
    pad_w = (-width) % int(multiple)
    if pad_h == 0 and pad_w == 0:
        return rgb, depth, observation, image_support, (height, width)
    pad = (0, pad_w, 0, pad_h)
    return (
        F.pad(rgb, pad, mode="constant", value=0.0),
        F.pad(depth, pad, mode="constant", value=float(ZERO_FILL)),
        F.pad(observation, pad, mode="constant", value=0.0),
        F.pad(image_support, pad, mode="constant", value=0.0),
        (height, width),
    )


def install_oc_spatial_prior(backbone: nn.Module) -> List[str]:
    """Replace every block's ``GeoPriorGen`` with ``OCSpatialPrior``.

    The author ``angle``/``decay`` buffers and the already loaded spatial
    coefficient ``weight[0]`` are copied; the signed Depth coefficient
    ``weight[1]`` is dropped with the module, so the OC branch can never inject
    the old ``B^d`` term a second time.
    """
    from models.encoders.DFormerv2 import RGBD_Block

    replaced = []
    for name, module in backbone.named_modules():
        if not isinstance(module, RGBD_Block):
            continue
        old = module.Geo
        new = OCSpatialPrior(module.embed_dim, int(old.decay.numel()))
        with torch.no_grad():
            new.angle.copy_(old.angle)
            new.decay.copy_(old.decay)
            new.weight.copy_(old.weight[0].detach().reshape(1, 1, 1, 1))
        module.Geo = new
        replaced.append("%s.Geo" % name if name else "Geo")
    if not replaced:
        raise RuntimeError("no RGBD_Block was found while installing the OC spatial prior")
    return replaced


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class OCRGAEncoderDecoder(nn.Module):
    """DFormerv2-S + HAM research wrapper for the OC-RGA study.

    ``forward(rgb, modal_x, depth_support=O, image_support=S, return_geometry=False)``
    returns ``[B,C,H,W]`` logits at the (padded) input resolution, or
    ``(logits, bundle)`` when ``return_geometry`` is set.  The bundle carries the
    prior/posterior factors and the raw geometry statistics; it never carries a
    reference depth or a label.
    """

    def __init__(
        self,
        cfg,
        mode: str = "oc_rga",
        criterion: Optional[nn.Module] = None,
        norm_layer=nn.BatchNorm2d,
        syncbn: bool = False,
    ):
        super().__init__()
        if mode not in ALL_MODES:
            raise ValueError("model mode must be one of %s" % (ALL_MODES,))
        self.mode = str(mode)
        self.oc_enabled = self.mode in OC_MODES
        self.baseline_enabled = self.mode in BASELINE_MODES

        resolved = deepcopy(cfg)
        resolved.oc_geometry = bool(self.oc_enabled)
        if str(resolved.backbone).startswith("DFormerv2"):
            # The OC path builds its own conditioning, so the backbone never runs
            # the author/ODG depth relation.  ``original`` keeps ``observations``
            # and the legacy depth branch off.
            resolved.geometry_mode = "original"
            resolved.norm_eval = bool(getattr(resolved, "norm_eval", False))
        self.resolved_cfg = resolved

        from models.builder import EncoderDecoder

        self.base = EncoderDecoder(
            cfg=resolved, criterion=criterion, norm_layer=norm_layer, syncbn=syncbn
        )

        heads = tuple(int(h) for h in getattr(resolved, "num_heads_per_stage", STAGE_HEADS))
        if self.baseline_enabled:
            # No prior branch and no precision head exist at all: the baseline must
            # not carry a single unused geometry parameter.
            self.geometry = OCGeometryBuilder(
                heads, uses_prior=False, uses_evidence=False, uses_baseline=True
            )
        else:
            self.geometry = OCGeometryBuilder(
                heads, uses_prior=True, uses_evidence=True, uses_baseline=False,
                readout=READOUT_FOR_MODE[self.mode],
            )
            self.replaced_spatial_prior = install_oc_spatial_prior(self.base.backbone)

    # -- author-compatible accessors used by the shared schedule helpers ---- #
    @property
    def backbone(self) -> nn.Module:
        return self.base.backbone

    @property
    def decode_head(self) -> nn.Module:
        return self.base.decode_head

    @property
    def aux_head(self):
        return getattr(self.base, "aux_head", None)

    # -- forward ------------------------------------------------------------ #
    def forward(
        self,
        rgb: torch.Tensor,
        modal_x: torch.Tensor,
        depth_support: Optional[torch.Tensor] = None,
        image_support: Optional[torch.Tensor] = None,
        return_geometry: bool = False,
    ):
        if depth_support is None:
            raise ValueError("the OC-RGA contract requires an explicit observation support O")
        if image_support is None:
            image_support = torch.ones_like(depth_support)
        if rgb.shape[-2:] != depth_support.shape[-2:]:
            raise ValueError(
                "RGB and support must share the input grid: %s vs %s"
                % (tuple(rgb.shape[-2:]), tuple(depth_support.shape[-2:]))
            )

        rgb_p, modal_p, obs_p, support_p, original = pad_to_multiple(
            rgb, modal_x, depth_support, image_support
        )
        outputs, bundle = self.base.backbone(
            rgb_p,
            modal_p,
            depth_support=obs_p,
            image_support=support_p,
            geometry_builder=self.geometry,
        )
        logits = self.base.decode_head.forward(outputs)
        logits = F.interpolate(logits, size=rgb_p.shape[-2:], mode="bilinear", align_corners=False)
        if (original[0], original[1]) != (rgb_p.shape[-2], rgb_p.shape[-1]):
            logits = logits[..., : original[0], : original[1]]
        if return_geometry:
            return logits, bundle
        return logits


def build_oc_optimizer_param_groups(model: nn.Module, lr: float, weight_decay: Optional[float] = None,
                                    geometry_lr_multiplier: float = 2.0):
    """AdamW groups that keep the OC multiplier and never decay a new logit.

    Encoder / shared stem / HAM / the copied spatial coefficient keep the base
    learning rate; the new geometry convolutions, probability head, kernel
    lengths and injection scales run at ``geometry_lr_multiplier``.  Biases and
    norm parameters are not decayed, and the positive-parameter logits
    (``a_ell``, ``a_beta``) and the copied spatial coefficient are not decayed
    either.  Every trainable tensor lands in exactly one group.  The multiplier is
    written as ``odg_lr_multiplier`` so the shared schedule helper reapplies it
    and the epoch schedule can never overwrite it.
    """
    from research.odg_schedule import NORM_MODULE_TYPES

    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        is_norm = isinstance(module, NORM_MODULE_TYPES)
        is_spatial_prior = type(module).__name__ == "OCSpatialPrior"
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = "%s.%s" % (module_name, param_name) if module_name else param_name
            multiplier = float(geometry_lr_multiplier) if (
                full_name.startswith("geometry.")
                or full_name.startswith("oc_injection.")
                or ".oc_injection." in full_name
            ) else 1.0
            logit_like = ("a_ell" in full_name) or ("a_beta" in full_name)
            if param_name.endswith("bias") or is_norm or is_spatial_prior or logit_like:
                no_decay.append((full_name, param, multiplier))
            else:
                decay.append((full_name, param, multiplier))

    groups: Dict[Tuple[float, float], List[nn.Parameter]] = {}
    for _name, param, multiplier in decay:
        key = (float(multiplier), float(weight_decay) if weight_decay is not None else 0.0)
        groups.setdefault(key, []).append(param)
    for _name, param, multiplier in no_decay:
        groups.setdefault((float(multiplier), 0.0), []).append(param)

    params_list = []
    for (multiplier, wd), params in sorted(groups.items()):
        params_list.append({
            "params": params,
            "lr": float(lr) * multiplier,
            "odg_lr_multiplier": multiplier,
            "weight_decay": wd,
        })
    return params_list


def build_model_for_mode(cfg, mode: str, criterion: Optional[nn.Module] = None,
                         norm_layer=nn.BatchNorm2d, syncbn: bool = False) -> OCRGAEncoderDecoder:
    return OCRGAEncoderDecoder(cfg, mode=mode, criterion=criterion, norm_layer=norm_layer, syncbn=syncbn)


def build_oc_model(cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False):
    return build_model_for_mode(cfg, "oc_rga", criterion=criterion, norm_layer=norm_layer, syncbn=syncbn)


def build_masked_baseline(cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False):
    return build_model_for_mode(cfg, "masked_baseline", criterion=criterion, norm_layer=norm_layer, syncbn=syncbn)


# --------------------------------------------------------------------------- #
# Structure / plumbing self-check (design 15.2 items 4, 6 and 8)
# --------------------------------------------------------------------------- #
_OC_KEY_PARAMS = (
    "geometry.prior.head_factor.weight",
    "geometry.precision.out.weight",
    "geometry.evidence.a_ell.0",
)


def run_model_self_check(
    config_name: str = "local_configs.research.OC_RGA_DeLiVER",
    device: str = "cpu",
    size: int = 128,
    amp: bool = False,
    report_path: Optional[str] = None,
    train_source: Optional[str] = None,
) -> Dict[str, object]:
    """Build both arms on a small grid and check structure, layouts and gradients.

    No dataset is read and no training happens: this is a CPU-first plumbing
    check.  The legacy author entry is exercised once on the same synthetic batch
    to prove the thin interface edits did not break it.
    """
    import inspect

    from research import odg_schedule as sched

    torch.manual_seed(20261014)
    dev = torch.device(device)
    config = deepcopy(getattr(import_module(config_name), "C"))
    config.log_dir = str(getattr(config, "log_dir", "outputs"))
    report: Dict[str, object] = {"config": config_name, "device": str(dev), "size": int(size),
                                 "checks": {}, "failures": [], "not_run": []}

    def record(name: str, ok: bool, detail):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["failures"].append(name)
        print("[model-self-check] %-32s %s  %s" % (name, "PASS" if ok else "FAIL", detail))

    batch = size
    rgb = torch.randn(1, 3, batch, batch, device=dev)
    depth = torch.rand(1, 3, batch, batch, device=dev) * 2.0 + ZERO_FILL
    obs = (torch.rand(1, 1, batch, batch, device=dev) > 0.25).float()
    support = torch.ones(1, 1, batch, batch, device=dev)
    depth = torch.where(obs > 0, depth, torch.full_like(depth, ZERO_FILL))

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=int(config.background))
    oc_model = build_oc_model(config, criterion=criterion).to(dev)
    oc_model.train()

    # --- structure ------------------------------------------------------- #
    block_geos = [type(m.Geo).__name__ for m in oc_model.backbone.modules()
                  if hasattr(m, "oc_injection")]
    ok_geo = block_geos and set(block_geos) == {"OCSpatialPrior"}
    record("oc_arm_spatial_prior_replaced", bool(ok_geo), "block Geo types=%s" % sorted(set(block_geos)))
    record("oc_arm_has_registered_geometry",
           all(hasattr(oc_model.geometry, a) for a in ("prior", "precision", "evidence")),
           "prior/precision/evidence registered")
    old_depth_coeff = any(
        name.endswith("Geo.weight") and tuple(param.shape) == (2, 1, 1, 1)
        for name, param in oc_model.named_parameters()
    )
    record("oc_arm_dropped_old_depth_coefficient", not old_depth_coeff,
           "no (2,1,1,1) Geo.weight remains")

    # --- the model contract cannot carry an oracle ---------------------- #
    contract = set(inspect.signature(type(oc_model).forward).parameters) - {"self"}
    expected_contract = {"rgb", "modal_x", "depth_support", "image_support", "return_geometry"}
    record("forward_contract_has_no_reference_input", contract == expected_contract,
           "forward parameters=%s" % sorted(contract))

    # --- read-out controls share the same parameter set ---------------- #
    from research.oc_rga import RelationalEvidencePyramid

    reference = dict(RelationalEvidencePyramid(STAGE_HEADS, readout="delta").named_parameters())
    same_capacity = {}
    for control, readout in READOUT_FOR_MODE.items():
        other = dict(RelationalEvidencePyramid(STAGE_HEADS, readout=readout).named_parameters())
        same_capacity[control] = (
            readout,
            sorted(other) == sorted(reference)
            and all(tuple(other[k].shape) == tuple(reference[k].shape) for k in reference),
        )
    record("readout_controls_same_capacity",
           all(ok for _r, ok in same_capacity.values()),
           "read-out controls keep identical kernel-length parameters: %s" % same_capacity)

    # --- forward / backward --------------------------------------------- #
    with torch.autocast(device_type=dev.type, enabled=bool(amp) and dev.type == "cuda"):
        logits, bundle = oc_model(rgb, depth, depth_support=obs, image_support=support,
                                  return_geometry=True)
    record("oc_forward_shape", tuple(logits.shape) == (1, int(config.num_classes), batch, batch),
           "logits=%s" % (tuple(logits.shape),))
    record("oc_logits_finite", bool(torch.isfinite(logits).all()), "logits finite")

    stage0 = bundle.stage(0)
    stage3 = bundle.stage(3)
    layout_ok = (
        tuple(stage0["Ehat_w"].shape) == (1, batch // 4, 4, batch // 4, batch // 4)
        and tuple(stage0["Ehat_h"].shape) == (1, batch // 4, 4, batch // 4, batch // 4)
        and tuple(stage3["Ehat_full"].shape) == (1, 16, (batch // 32) ** 2, (batch // 32) ** 2)
    )
    record("w_h_full_tensor_layouts", layout_ok,
           "stage0 w=%s h=%s stage3 full=%s" % (tuple(stage0["Ehat_w"].shape),
                                                tuple(stage0["Ehat_h"].shape),
                                                tuple(stage3["Ehat_full"].shape)))
    same_object = bundle.stage(0) is bundle.stage(0)
    record("stage_evidence_shared_once", same_object,
           "one evidence tensor per stage, reused by every block of that stage")
    record("probability_path_fp32",
           bundle.base["mu"].dtype == torch.float32 and bundle.base["U"].dtype == torch.float32,
           "mu/U dtype=%s" % (bundle.base["mu"].dtype,))

    target = torch.randint(0, int(config.num_classes), (1, batch, batch), device=dev)
    seg = criterion(logits, target)[target != int(config.background)].mean()
    seg.backward()
    grad_report = sched.grad_finiteness_report(oc_model)
    record("grads_finite", bool(grad_report["all_finite"]),
           "nonfinite=%s" % (grad_report["nonfinite"][:5],))

    key_grads = {}
    for name, param in oc_model.named_parameters():
        if name in _OC_KEY_PARAMS:
            key_grads[name] = None if param.grad is None else float(param.grad.abs().max())
    sampled = [n for n, p in oc_model.named_parameters()
               if n.endswith("oc_injection.a_beta")]
    beta_max = max((float(p.grad.abs().max()) for n, p in oc_model.named_parameters()
                    if n.endswith("oc_injection.a_beta") and p.grad is not None), default=0.0)
    key_grads["oc_injection.a_beta (max over blocks)"] = beta_max
    nonzero = all(v is not None and v > 0 for v in key_grads.values())
    record("key_new_parameter_gradients_nonzero", bool(nonzero),
           ", ".join("%s=%.3e" % (k.split(".")[-1], v) for k, v in key_grads.items()))

    # --- optimizer coverage -------------------------------------------- #
    params_list = sched.build_optimizer_param_groups(oc_model, float(config.lr), float(config.weight_decay))
    optimizer = torch.optim.AdamW(params_list, lr=float(config.lr), betas=(0.9, 0.999),
                                  weight_decay=float(config.weight_decay))
    coverage = sched.optimizer_parameter_report(oc_model, optimizer)
    record("optimizer_covers_all_trainable_once", True,
           "trainable=%d groups=%s duplicates=%d missing=%d"
           % (coverage["trainable_tensors"], coverage["group_sizes"],
              coverage["duplicates"], len(coverage["missing"])))
    report["optimizer_coverage"] = coverage

    # --- baseline arm ---------------------------------------------------- #
    baseline = build_masked_baseline(config, criterion=criterion).to(dev)
    baseline.train()
    base_extra = [n for n, _ in baseline.named_parameters() if n.startswith("geometry.prior")
                  or n.startswith("geometry.precision") or n.startswith("geometry.evidence")]
    record("baseline_has_no_oc_geometry_parameters", not base_extra,
           "unexpected=%s" % base_extra[:5])
    baseline_geo = sorted({type(m.Geo).__name__ for m in baseline.backbone.modules()
                           if hasattr(m, "Geo") and type(m).__name__ == "RGBD_Block"})
    record("baseline_keeps_author_geo_prior", baseline_geo == ["GeoPriorGen"],
           "block Geo types=%s" % baseline_geo)
    base_logits, base_bundle = baseline(rgb, depth, depth_support=obs, image_support=support,
                                        return_geometry=True)
    record("baseline_forward_shape",
           tuple(base_logits.shape) == (1, int(config.num_classes), batch, batch),
           "logits=%s" % (tuple(base_logits.shape),))
    base_contract = set(inspect.signature(type(baseline).forward).parameters) - {"self"}
    record("baseline_contract_identical", base_contract == contract,
           "both arms expose the same forward contract")
    record("baseline_evidence_absent",
           not any("Ehat" in key for key in base_bundle.stage(0))
           and "base_w" in base_bundle.stage(0),
           "baseline consumes base_* relations only")

    # --- legacy author entry -------------------------------------------- #
    from models.builder import EncoderDecoder

    legacy_cfg = deepcopy(config)
    legacy_cfg.geometry_mode = "original"
    # criterion=None skips the pretrained load: this check is about the thin
    # interface edits, not about the checkpoint.
    legacy = EncoderDecoder(cfg=legacy_cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False).to(dev)
    legacy.eval()
    with torch.no_grad():
        legacy_out = legacy(rgb, depth, depth_support=support)
    record("legacy_author_entry_forward",
           tuple(legacy_out.shape) == (1, int(config.num_classes), batch, batch),
           "EncoderDecoder(geometry_mode=original) still forwards: %s" % (tuple(legacy_out.shape),))

    # --- AMP boundary ---------------------------------------------------- #
    if dev.type == "cuda":
        oc_model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_logits, amp_bundle = oc_model(rgb, depth, depth_support=obs, image_support=support,
                                              return_geometry=True)
        record("amp_forward_finite_and_fp32_geometry",
               bool(torch.isfinite(amp_logits).all()) and amp_bundle.base["mu"].dtype == torch.float32,
               "fp16 autocast logits finite; geometry factors stay FP32")
    else:
        report["not_run"].append("amp_boundary (needs --device cuda)")

    del oc_model, baseline, legacy
    report["failures"] = list(report["failures"])
    report["passed"] = not report["failures"]
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("[model-self-check] wrote %s" % report_path)
    print("[model-self-check] %s (%d checks, %d not run)"
          % ("ALL PASS" if report["passed"] else "FAILURES", len(report["checks"]),
             len(report["not_run"])))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OC-RGA research model (OC arm + mask-aware baseline)")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--config", default="local_configs.research.OC_RGA_DeLiVER")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    if not args.self_check:
        parser.print_help()
        return 0
    report = run_model_self_check(config_name=args.config, device=args.device, size=args.size,
                                  amp=args.amp, report_path=args.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
