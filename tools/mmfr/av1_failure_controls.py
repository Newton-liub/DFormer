#!/usr/bin/env python3
"""One authorized failure-scene precision comparison; never an optimizer update.

Exactly three proposal loss/backward cases, at most six model forwards and three
backwards. This saved scene is diagnostic-only, not a formal resume checkpoint.
Production HAM, phase loss, configuration and normal resume guards stay unchanged.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch

from tools.mmfr.av1_train import CONFIG, build_model, check_inputs, restore_model_with_source_guard
from tools.mmfr.e1_quickval import atomic_write_json
from utils import mmfr_av1_training as T
from utils.training_checkpoint import file_sha256, restore_rng_state

SCENE_SHA = "6865e8865bd486fac5b9bf06317af768d5a4634e5e4af4ea839a41c174bb0fb5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("commit the diagnostic and current evidence before GPU execution")
    if file_sha256(args.scene) != SCENE_SHA:
        raise RuntimeError("only the directly verified original1871 failure scene is authorized")
    scene = torch.load(args.scene, map_location="cpu", weights_only=False)
    if (scene.get("artifact_role") != "diagnostic-only-nonresumable-not-performance-candidate"
            or (scene.get("attempted"), scene.get("completed")) != (1871, 1870)
            or "av1_resume_version" in scene):
        raise RuntimeError("unexpected scene role/cursor; no formal resume")
    config = copy.deepcopy(importlib.import_module(CONFIG).C)
    config.mmfr_av1["source_checkpoint"] = str(args.source_checkpoint.resolve())
    identity = check_inputs(config)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("authorized existing RTX4090 required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "identity.json", {**identity, "scene_sha256": SCENE_SHA,
                      "scene_runtime_commit": scene["runtime_commit"], "diagnostic_only": True,
                      "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "max_model_forwards": 6, "max_backwards": 3, "optimizer_updates": 0})
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cases, forwards, backwards = [], 0, 0
    nmf = None
    original_forward = None
    handle = None
    try:
        model, _ = build_model(config, identity)
        restore_model_with_source_guard(model, scene)
        unused_optimizer = T.configure_phase(model, "proposal", lr=3e-5, weight_decay=0.01)
        del unused_optimizer  # configure membership only; step() is never called.
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in scene["batch"].items()}
        labels = scene["labels"].cuda()
        if tuple(batch["rgb"].shape) != (10, 3, 480, 640) or tuple(labels.shape) != (10, 480, 640):
            raise RuntimeError("saved batch/label shape changed")
        nmf = model.decode_head.hamburger.ham
        original_forward = nmf.forward

        def count_forward(_module, _args):
            nonlocal forwards
            if forwards >= 6:
                raise RuntimeError("authorized forward budget exceeded")
            forwards += 1

        def local_fp32(x, *args, **kwargs):
            with torch.autocast(device_type="cuda", enabled=False):
                return original_forward(x.float(), *args, **kwargs)

        handle = model.register_forward_pre_hook(count_forward)
        for name, scale, use_fp32 in (("original_amp_scale1024", 1024.0, False),
                                       ("original_amp_scale1", 1.0, False),
                                       ("nmf_local_fp32_scale1024", 1024.0, True)):
            nmf.forward = local_fp32 if use_fp32 else original_forward
            model.zero_grad(set_to_none=True)
            restore_rng_state(scene["rng_before_forward"])
            s = config.mmfr_av1["grad_scaler"]
            scaler = torch.cuda.amp.GradScaler(init_scale=scale, growth_factor=s["growth_factor"],
                                              backoff_factor=s["backoff_factor"],
                                              growth_interval=s["growth_interval"])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, details = T.phase_loss(model, batch, labels, phase="proposal",
                                             margin=0.01, lambda_clean=0.1)
            backwards += 1
            if backwards > 3:
                raise RuntimeError("authorized backward budget exceeded")
            scaler.scale(loss).backward()  # Expected nonfinite gradients are recorded, never stepped.
            torch.cuda.synchronize()
            gradients = {}
            for key, parameter in model.named_parameters():
                if parameter.requires_grad:
                    grad = parameter.grad
                    item = {"missing": grad is None}
                    if grad is not None:
                        finite = torch.isfinite(grad)
                        item.update(nan_count=int(torch.isnan(grad).sum()),
                                    inf_count=int(torch.isinf(grad).sum()), all_finite=bool(finite.all()),
                                    dtype=str(grad.dtype))
                        if bool(finite.any()):
                            item["finite_unscaled_max_abs"] = float(grad.detach()[finite].float().abs().max() / scale)
                    gradients[key] = item
            healthy = all(not v["missing"] and v.get("all_finite", False) for v in gradients.values())
            record = {"case": name, "scale": scale, "nmf_local_fp32": use_fp32,
                      "loss": float(loss.detach()), "ce": float(details["ce"]),
                      "clean_consistency": float(details["clean_consistency"]),
                      "active_gradients_all_finite_and_present": healthy, "active_gradients": gradients,
                      "model_forwards_so_far": forwards, "backwards_so_far": backwards,
                      "optimizer_updates": 0}
            cases.append(record)
            atomic_write_json(output / f"{name}.json", record)
            print(json.dumps(record), flush=True)
            if len(cases) == 1:
                expected = json.loads((args.scene.parent / "diagnostic-failure.json").read_text())
                if (record["loss"] != expected["loss"] or record["ce"] != expected["loss_details"]["ce"]
                        or record["clean_consistency"] != expected["loss_details"]["clean_consistency"]
                        or healthy or not any(v.get("nan_count", 0) for v in gradients.values())):
                    raise RuntimeError("original failure not reproduced exactly; stop precision comparisons")
            if len(cases) == 2 and any(record[k] != cases[0][k] for k in ("loss", "ce", "clean_consistency")):
                raise RuntimeError("scale-only comparison changed forward loss; stop")
            nmf.forward = original_forward
            del loss, details, scaler
        if (forwards, backwards) != (6, 3):
            raise RuntimeError("comparison budget/count mismatch")
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v.detach().cpu(), scene["model"][k])]
        if changed:
            raise RuntimeError(f"diagnostic changed model tensors: {changed}")
        atomic_write_json(output / "controls-result.json", {"status": "completed", "diagnostic_only": True,
                          "runtime_commit": identity["git_commit"], "scene_sha256": SCENE_SHA,
                          "cases": cases, "model_forwards": forwards, "backwards": backwards,
                          "optimizer_updates": 0, "all_model_tensors_unchanged": True,
                          "production_precision_contract_unchanged": True,
                          "formal_recovery_authorized": False, "official_test": "sealed_unread"})
    except BaseException:
        atomic_write_json(output / "stopped.json", {"exception": traceback.format_exc(), "cases": cases,
                          "model_forwards": forwards, "backwards_attempted": backwards,
                          "optimizer_updates": 0, "automatic_retry": False})
        raise
    finally:
        if nmf is not None and original_forward is not None:
            nmf.forward = original_forward
        if handle is not None:
            handle.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
