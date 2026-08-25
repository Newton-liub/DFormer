#!/usr/bin/env python3
"""Run the stage-04 real-model B1 masked-loss regression on a qualification protocol.

The command reads only the frozen official-train list to select one all-background and
one ordinary MUSeg image.  It never opens official-test.txt and writes a compact JSON
report instead of checkpoints or model weights.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import math
import random
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.builder import EncoderDecoder
from tools.museg_protocol import ProtocolError, file_sha256, load_protocol, write_json
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import ValPre
from utils.training_checkpoint import get_git_commit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--output", required=True, help="new JSON evidence path")
    parser.add_argument("--seed", type=int, default=1572378116)
    parser.add_argument("--aux-rate", type=float, default=0.4)
    return parser.parse_args(argv)


def _entry_stem(entry: str) -> str:
    return Path(entry).stem


def select_b1_samples(official_train: Path, label_root: Path) -> dict[str, str]:
    """Return deterministic first all-background and ordinary entries from official train."""
    all_background: str | None = None
    ordinary: str | None = None
    for raw_line in official_train.read_text(encoding="utf-8").splitlines():
        entry = raw_line.strip()
        if not entry:
            continue
        label_path = label_root / f"{_entry_stem(entry)}.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None or label.ndim != 2 or label.dtype != np.uint8:
            raise ProtocolError(f"cannot decode MUSeg uint8 label for B1 selection: {label_path}")
        if bool(np.all(label == 0)):
            if all_background is None:
                all_background = entry
        elif ordinary is None:
            ordinary = entry
        if all_background is not None and ordinary is not None:
            return {"all_background": all_background, "ordinary": ordinary}
    raise ProtocolError("official train does not contain both an all-background and an ordinary sample")


def _dataset_setting(config: Any, train_source: Path) -> dict[str, Any]:
    return {
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
        "val_source": None,
        "test_source": None,
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }


def _sample_tensors(config: Any, official_train: Path, selected: dict[str, str]) -> dict[str, dict[str, torch.Tensor]]:
    dataset = RGBXDataset(
        _dataset_setting(config, official_train),
        "train",
        ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config),
    )
    indices = {entry: index for index, entry in enumerate(dataset._file_names)}
    samples: dict[str, dict[str, torch.Tensor]] = {}
    for role, entry in selected.items():
        try:
            item = dataset[indices[entry]]
        except KeyError as exc:
            raise ProtocolError(f"selected B1 entry disappeared from official train: {entry}") from exc
        samples[role] = {
            "data": item["data"],
            "label": item["label"],
            "modal_x": item["modal_x"],
        }
    return samples


def _make_batch(samples: dict[str, dict[str, torch.Tensor]], roles: tuple[str, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.stack([samples[role][field] for role in roles], dim=0) for field in ("data", "modal_x", "label"))  # type: ignore[return-value]


def _finite_gradients(model: torch.nn.Module) -> tuple[bool, bool, list[str], int]:
    expected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing = [name for name, parameter in expected if parameter.grad is None]
    gradients = [parameter.grad for _, parameter in expected if parameter.grad is not None]
    return (
        bool(expected) and not missing and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients),
        bool(expected) and not missing and all(bool(torch.equal(gradient, torch.zeros_like(gradient))) for gradient in gradients),
        missing,
        len(gradients),
    )


def _run_case(
    *,
    config: Any,
    aux_rate: float,
    scenario: str,
    samples: dict[str, dict[str, torch.Tensor]],
    roles: tuple[str, str],
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = copy.copy(config)
    model_config.aux_rate = aux_rate
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=model_config.background)
    model = EncoderDecoder(cfg=model_config, criterion=criterion, syncbn=False).cuda()
    model.train()
    data, modal_x, label = _make_batch(samples, roles)
    data = data.cuda(non_blocking=True)
    modal_x = modal_x.cuda(non_blocking=True)
    label = label.cuda(non_blocking=True)
    model.zero_grad(set_to_none=True)
    loss = model(data, modal_x, label)
    loss_finite = bool(torch.isfinite(loss).all().item())
    if not loss_finite:
        raise FloatingPointError(f"non-finite B1 loss in {scenario}")
    loss.backward()
    gradients_finite, gradients_zero, missing_gradients, gradient_tensors = _finite_gradients(model)
    if not gradients_finite:
        raise FloatingPointError(
            f"non-finite or missing B1 gradients in {scenario}: {', '.join(missing_gradients)}"
        )
    all_background = all(role == "all_background" for role in roles)
    if all_background and (loss.detach().item() != 0.0 or not gradients_zero):
        raise AssertionError(f"all-background B1 case must produce graph-connected exact zero: {scenario}")
    result = {
        "scenario": scenario,
        "aux_rate": aux_rate,
        "samples": list(roles),
        "loss": float(loss.detach().cpu().item()),
        "loss_finite": loss_finite,
        "gradients_finite": gradients_finite,
        "gradients_exact_zero": gradients_zero,
        "gradient_tensors": gradient_tensors,
        "missing_gradients": missing_gradients,
        "all_background": all_background,
        "pass": True,
    }
    del model, data, modal_x, label, loss
    torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise ProtocolError(f"B1 output already exists: {output}")
    if not torch.cuda.is_available():
        raise ProtocolError("B1 real-model regression requires CUDA")
    if args.aux_rate <= 0:
        raise ProtocolError("--aux-rate must be positive")
    protocol = load_protocol(args.protocol_manifest)
    protocol.validate_consumed_splits()
    repo_root = Path(__file__).resolve().parents[1]
    current_commit = get_git_commit(repo_root)
    if current_commit != str(protocol.git["required_commit"]).lower():
        raise ProtocolError(
            f"B1 requires exact materialized protocol commit {protocol.git['required_commit']}, got {current_commit}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if dirty:
        raise ProtocolError("B1 requires a clean Git working tree")
    official_train = protocol.split_path("official_train")
    pretrained_path = protocol.resolve_declared_path(str(protocol.pretrained["path"]))
    actual_pretrained_sha = file_sha256(pretrained_path)
    if pretrained_path.stat().st_size != int(protocol.pretrained["size_bytes"]) or actual_pretrained_sha.lower() != str(protocol.pretrained["sha256"]).lower():
        raise ProtocolError("B1 pretrained weight identity does not match the materialized protocol")
    config = getattr(importlib.import_module(protocol.config_module), "C")
    config.pretrained_model = str(pretrained_path)
    config.experiment_phase = "qualification"
    selected = select_b1_samples(official_train, Path(config.gt_root_folder))
    samples = _sample_tensors(config, official_train, selected)
    cases = []
    for aux_rate, head_name in ((0.0, "main"), (float(args.aux_rate), "main_aux")):
        cases.append(_run_case(
            config=config,
            aux_rate=aux_rate,
            scenario=f"{head_name}_mixed",
            samples=samples,
            roles=("ordinary", "all_background"),
            seed=args.seed,
        ))
        cases.append(_run_case(
            config=config,
            aux_rate=aux_rate,
            scenario=f"{head_name}_all_background",
            samples=samples,
            roles=("all_background", "all_background"),
            seed=args.seed,
        ))
    payload = {
        "schema_version": "museg-stage04-b1-report-v1",
        "pass": all(item["pass"] for item in cases),
        "protocol_id": protocol.protocol_id,
        "protocol_manifest": str(protocol.path),
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "git_commit": get_git_commit(Path(__file__).resolve().parents[1]),
        "phase": "qualification",
        "official_test_included": False,
        "official_train": {
            "path": str(official_train),
            "sha256": protocol.splits["official_train"]["sha256"],
            "samples": protocol.splits["official_train"]["samples"],
        },
        "pretrained": {
            "path": config.pretrained_model,
            "sha256": file_sha256(config.pretrained_model),
        },
        "seed": args.seed,
        "selected_samples": selected,
        "cases": cases,
    }
    write_json(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())