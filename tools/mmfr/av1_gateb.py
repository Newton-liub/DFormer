"""Bounded A-v1 Gate-B: eight tiny real forwards, one update per phase.

No dataset/split/evaluator is opened. Formal training has no entry here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import utils.mmfr_av1_training as training
from utils.training_checkpoint import capture_rng_state, restore_rng_state, file_sha256, load_weights_only_model_state

CHECK_NAMES = (
    "source_identity", "strict_off", "zero_init", "proposal_optimizer_membership",
    "gate_optimizer_membership", "utility_three_branches", "all_ambiguous",
    "frozen_base", "rng_ham_alignment", "gate_input_safety", "finite_loss", "phase_corruption_isolation",
)


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digests(model, prefix=None, exclude_av1=False):
    return {name: tensor_digest(tensor.reshape(-1)) for name, tensor in model.state_dict().items()
            if (prefix is None or name.startswith(prefix)) and not (exclude_av1 and name.startswith("av1."))}


def rng_digest():
    state = capture_rng_state()
    return {"cpu": tensor_digest(state["torch_cpu"]),
            "cuda": [tensor_digest(item) for item in (state["torch_cuda"] or [])]}


def utility_check():
    off = torch.tensor([1.2, 0.8, 1.0, 1.125, 0.875], requires_grad=True)
    full = torch.ones(5, requires_grad=True)
    utility, targets, mask = training.utility_targets(off, full, 0.125)
    gate = torch.tensor([0.8, 0.2, 0.7, 0.6, 0.4], requires_grad=True)
    bce = training.masked_gate_bce(gate, targets, mask)
    expected = torch.nn.functional.binary_cross_entropy(gate[:2], targets[:2])
    bce.backward()
    three = (mask.tolist() == [True, True, False, False, False]
             and targets.tolist() == [1., 0., 0., 0., 0.]
             and torch.equal(bce, expected) and not utility.requires_grad
             and not targets.requires_grad and off.grad is None and full.grad is None
             and bool(torch.equal(gate.grad[2:], torch.zeros(3))))
    ambiguous_gate = torch.tensor([0.4, 0.6], requires_grad=True)
    _, all_targets, all_mask = training.utility_targets(torch.ones(2), torch.ones(2), 0.125)
    zero_bce = training.masked_gate_bce(ambiguous_gate, all_targets, all_mask)
    logits = torch.tensor([[[[1.]], [[0.]]], [[[0.]], [[1.]]]], requires_grad=True)
    ce = training.per_sample_ce(logits, torch.tensor([[[0]], [[1]]])).mean()
    total = zero_bce + ce
    total.backward()
    all_ok = (not bool(all_mask.any()) and zero_bce.item() == 0.0 and bool(torch.isfinite(total))
              and logits.grad is not None and bool((logits.grad != 0).any()))
    return three, all_ok, {"utility": utility.tolist(), "mask": mask.tolist(), "targets": targets.tolist(),
                           "bce": bce.item(), "all_ambiguous_bce": zero_bce.item(),
                           "all_ambiguous_ce": ce.item(), "endpoint_gradients_absent": off.grad is None and full.grad is None}


def membership(model, optimizer, phase):
    ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    counts = {name: ids.count(id(parameter)) for name, parameter in model.named_parameters()}
    good = all(count == int(name.startswith(f"av1.{phase}.")) for name, count in counts.items())
    good = good and all(parameter.requires_grad == name.startswith(f"av1.{phase}.")
                        for name, parameter in model.named_parameters())
    return good, {"proposal": sorted(set(count for name, count in counts.items() if name.startswith("av1.proposal."))),
                  "gate": sorted(set(count for name, count in counts.items() if name.startswith("av1.gate."))),
                  "base": sorted(set(count for name, count in counts.items() if not name.startswith("av1.")))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "MMFR/02_evidence/mmfr_a_v1_gateb.json")
    args = parser.parse_args()
    config = copy.deepcopy(importlib.import_module("local_configs.MUSeg.DFormerv2_S_MMFR_AV1").C)
    checks = {name: {"status": "BLOCKED", "reason": "not reached"} for name in CHECK_NAMES}
    result = {"identity": config.run_id, "checks": checks, "formal_training_started": False,
              "quickval_started": False, "mainval_started": False, "official_test_included": False,
              "input": "synthetic normalized RGB/Depth and synthetic labels, B=1 H=W=64; no dataset reads",
              "smoke_only_settings": {"margin": 0.01, "lambda_clean": 0.1, "lr": 3e-5,
                                       "weight_decay": 0.01, "proposal_steps": 1, "gate_steps": 1},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}

    def check(name, passed, **details):
        checks[name] = {"status": "PASS" if passed else "FAIL", **details}
        print(f"{name}: {checks[name]['status']}", flush=True)

    try:
        source = config.mmfr_av1["source_checkpoint"]
        actual_hash = file_sha256(source)
        check("source_identity", actual_hash == config.mmfr_av1["source_checkpoint_sha256"], path=source, sha256=actual_hash)
        if checks["source_identity"]["status"] != "PASS":
            raise RuntimeError("C0 source identity mismatch")
        three, all_ok, evidence = utility_check()
        check("utility_three_branches", three, evidence=evidence)
        check("all_ambiguous", all_ok, bce=evidence["all_ambiguous_bce"], ce=evidence["all_ambiguous_ce"])
        if not torch.cuda.is_available():
            raise RuntimeError("The unchanged HAM implementation requires CUDA")
        torch.manual_seed(20260930)
        torch.cuda.manual_seed_all(20260930)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        result["environment"] = {"python": sys.executable, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                                 "precision": "FP32", "tf32": False}
        from models.builder import EncoderDecoder
        source_config = copy.deepcopy(importlib.import_module(config.mmfr_av1["source_config"]).C)
        source_config.pretrained_model = None
        generator = torch.Generator().manual_seed(20260930)
        raw_rgb = torch.randint(1, 256, (1, 3, 64, 64), generator=generator).float() / 255.
        raw_depth = torch.randint(1, 256, (1, 1, 64, 64), generator=generator).float() / 255.
        raw_depth[:, :, ::8, ::8] = 0.
        mean = torch.tensor(config.norm_mean, dtype=torch.float32).reshape(1, 3, 1, 1)
        std = torch.tensor(config.norm_std, dtype=torch.float32).reshape(1, 3, 1, 1)
        rgb = (raw_rgb - mean) / std
        depth = ((raw_depth - 0.48) / 0.28).repeat(1, 3, 1, 1)
        labels = torch.randint(0, config.num_classes, (1, 64, 64), generator=generator).cuda()
        labels[:, :2, :2] = 255
        # One fresh corruption construction in each phase. Clean Phase-1 probe also exercises KL.
        proposal_config = copy.deepcopy(config)
        proposal_config.mmfr_a2["corruption"]["p_clean"] = 1.0
        gate_config = copy.deepcopy(config)
        gate_config.mmfr_a2["corruption"]["p_clean"] = 0.0
        batch_args = dict(epoch=421, iteration=0, niters_per_epoch=128, nepochs=500)
        proposal_batch = training.build_phase_batch(proposal_config, "proposal", rgb, depth, ["av1-gateb-synthetic"], **batch_args)
        gate_batch = training.build_phase_batch(gate_config, "gate", rgb, depth, ["av1-gateb-synthetic"], **batch_args)
        phase_words = {phase: batch["metadata"][0]["seed_words"] for phase, batch in (("proposal", proposal_batch), ("gate", gate_batch))}
        check("phase_corruption_isolation", phase_words["proposal"] != phase_words["gate"]
              and proposal_batch["raw_depth"].data_ptr() != gate_batch["raw_depth"].data_ptr(),
              phase_seed_words=phase_words, cache_reused=False,
              synthetic_probe_p_clean={"proposal": 1.0, "gate": 0.0}, gate_specs=gate_batch["metadata"][0]["specs"])
        for batch in (proposal_batch, gate_batch):
            for key in ("rgb", "depth", "raw_depth", "valid_mask", "clean_mask"):
                batch[key] = batch[key].cuda()
        c0 = EncoderDecoder(cfg=source_config, norm_layer=torch.nn.SyncBatchNorm, syncbn=True)
        result["source_load"] = load_weights_only_model_state(source, model=c0, expected_sha256=actual_hash)
        c0.cuda().eval().requires_grad_(False)
        rng = capture_rng_state()
        with torch.no_grad():
            reference = c0(gate_batch["rgb"], gate_batch["depth"])
        del c0
        torch.cuda.empty_cache()
        model = EncoderDecoder(cfg=config, norm_layer=torch.nn.SyncBatchNorm, syncbn=True)
        result["av1_load"] = load_weights_only_model_state(source, model=model, expected_sha256=actual_hash,
                                                          allowed_missing_prefixes=("av1.",))
        model.cuda().eval()
        result["parameters"] = {name: sum(parameter.numel() for parameter in getattr(model.av1, name).parameters())
                                for name in ("proposal", "gate")}
        result["parameters"]["total_added"] = sum(result["parameters"].values())
        residual_calls, gate_calls = [], []
        h1 = model.av1.proposal.register_forward_hook(lambda *_args: residual_calls.append(1))
        h2 = model.av1.gate.register_forward_hook(lambda *_args: gate_calls.append(1))
        with torch.no_grad():
            restore_rng_state(rng)
            off = model(gate_batch["rgb"], gate_batch["depth"], av1_mode="off")
        h1.remove()
        h2.remove()
        check("strict_off", torch.equal(reference, off) and tensor_digest(reference) == tensor_digest(off)
              and not residual_calls and not gate_calls,
              bitwise_equal=tensor_digest(reference) == tensor_digest(off), max_abs_error=(reference-off).abs().max().item(),
              proposal_calls=len(residual_calls), gate_calls=len(gate_calls))
        with torch.no_grad():
            restore_rng_state(rng)
            full = model(gate_batch["rgb"], gate_batch["depth"], av1_mode="full")
        up = model.av1.proposal[-1]
        check("zero_init", torch.equal(full, off) and tensor_digest(full) == tensor_digest(off)
              and torch.count_nonzero(up.weight).item() == 0 and torch.count_nonzero(up.bias).item() == 0,
              bitwise_equal=tensor_digest(full) == tensor_digest(off), max_abs_error=(full-off).abs().max().item())
        del reference, off, full
        frozen_checks, finite_checks = [], []
        for phase, batch in (("proposal", proposal_batch), ("gate", gate_batch)):
            optimizer = training.configure_phase(model, phase, lr=3e-5, weight_decay=0.01)
            good_membership, members = membership(model, optimizer, phase)
            before_base = state_digests(model, exclude_av1=True)
            frozen_prefix = "av1.gate." if phase == "proposal" else "av1.proposal."
            before_frozen = state_digests(model, frozen_prefix)
            before_active = state_digests(model, f"av1.{phase}.")
            forward_records, nmf_records, label_records = [], [], []
            nmf = model.decode_head.hamburger.ham
            original_bases = nmf._build_bases
            original_ce = training.per_sample_ce

            def bases_probe(_self, *values, **kwargs):
                pre = rng_digest()
                bases = original_bases(*values, **kwargs)
                nmf_records.append({"rng": pre, "initial_bases": tensor_digest(bases)})
                return bases

            def forward_probe(_module, inputs, kwargs):
                forward_records.append({"mode": kwargs["av1_mode"], "rng": rng_digest(),
                                        "rgb": tensor_digest(inputs[0]), "depth": tensor_digest(inputs[1]),
                                        "observed": tensor_digest(kwargs["observed_depth"]),
                                        "observed_ptr": kwargs["observed_depth"].data_ptr()})

            def ce_probe(logits, target, **kwargs):
                label_records.append({"hash": tensor_digest(target), "ptr": target.data_ptr()})
                return original_ce(logits, target, **kwargs)

            nmf._build_bases = types.MethodType(bases_probe, nmf)
            training.per_sample_ce = ce_probe
            handle = model.register_forward_pre_hook(forward_probe, with_kwargs=True)
            try:
                optimizer.zero_grad(set_to_none=True)
                loss, details = training.phase_loss(model, batch, labels, phase=phase, margin=0.01, lambda_clean=0.1)
                loss.backward()
                gradient_ok = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                                  for name, parameter in model.named_parameters() if name.startswith(f"av1.{phase}."))
                frozen_grad_ok = all(parameter.grad is None for name, parameter in model.named_parameters()
                                     if not name.startswith(f"av1.{phase}."))
                optimizer.step()
            finally:
                handle.remove()
                nmf._build_bases = original_bases
                training.per_sample_ce = original_ce
            base_equal = before_base == state_digests(model, exclude_av1=True)
            frozen_equal = before_frozen == state_digests(model, frozen_prefix)
            changed = [name for name, digest in state_digests(model, f"av1.{phase}.").items() if digest != before_active[name]]
            base_eval = all(not module.training for name, module in model.named_modules() if not name.startswith("av1") and name)
            frozen_checks.append(base_equal and base_eval and frozen_grad_ok)
            finite_checks.append(bool(torch.isfinite(loss)) and gradient_ok)
            check(f"{phase}_optimizer_membership", good_membership and frozen_equal and base_equal and bool(changed)
                  and gradient_ok and frozen_grad_ok,
                  membership=members, updated_parameters=changed, frozen_branch_unchanged=frozen_equal,
                  base_parameters_and_buffers_unchanged=base_equal, loss=loss.item(),
                  gradients_finite=gradient_ok, frozen_gradients_absent=frozen_grad_ok)
            if phase == "gate":
                alignment = (len(nmf_records) == 3 and nmf_records[0] == nmf_records[1] == nmf_records[2]
                             and len(forward_records) == 3 and all(
                                 all(record[key] == forward_records[0][key] for key in ("rng", "rgb", "depth", "observed", "observed_ptr"))
                                 for record in forward_records)
                             and len(label_records) == 3 and all(record == label_records[0] for record in label_records)
                             and details["off_detached"] and details["full_detached"])
                check("rng_ham_alignment", alignment, forwards=forward_records, nmf=nmf_records, labels=label_records,
                      off_detached=details["off_detached"], full_detached=details["full_detached"])
                result["gate_step"] = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in details.items()}
        check("frozen_base", all(frozen_checks), parameters_and_all_buffers_exact=True if all(frozen_checks) else False,
              base_eval_in_both_phases=all(frozen_checks))
        check("finite_loss", all(finite_checks))
        signature = str(inspect.signature(model.av1.gate_value))
        from models.mmfr_av1 import depth_stats
        empty_stats = depth_stats(torch.zeros(1, 1, 4, 4, device="cuda"), torch.ones(1, 1, 4, 4, device="cuda", dtype=torch.bool))
        empty_support_stats = depth_stats(torch.zeros(1, 1, 4, 4, device="cuda"), torch.zeros(1, 1, 4, 4, device="cuda", dtype=torch.bool))
        check("gate_input_safety", model.av1.gate[0].in_features == 260
              and bool(torch.isfinite(empty_stats).all()) and torch.equal(empty_support_stats, torch.zeros_like(empty_support_stats)),
              gate_value_signature=signature, input_dim=260,
              features=["pooled stage2 feature 256", "observed zero ratio", "observed nonzero mean", "observed nonzero std", "observed valid nonzero neighbor absolute difference"],
              all_zero_depth_statistics=empty_stats.tolist(), empty_support_statistics=empty_support_stats.tolist(),
              static_review_required=True)
        result["source_unchanged_after_steps"] = file_sha256(source) == actual_hash
        if not result["source_unchanged_after_steps"]:
            check("frozen_base", False, reason="source checkpoint bytes changed")
    except Exception:
        result["exception"] = traceback.format_exc()
        print(result["exception"], flush=True)
    result["status"] = "PASS" if all(checks[name]["status"] == "PASS" for name in CHECK_NAMES) else (
        "FAIL" if any(checks[name]["status"] == "FAIL" for name in CHECK_NAMES) else "BLOCKED")
    result["implementation_sha256"] = {path: file_sha256(ROOT / path) for path in (
        "models/builder.py", "models/mmfr_av1.py", "utils/mmfr_av1_training.py",
        "local_configs/MUSeg/DFormerv2_S_MMFR_AV1.py", "tools/mmfr/av1_gateb.py") if (ROOT / path).is_file()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Gate-B {result['status']}; evidence: {args.output}", flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
