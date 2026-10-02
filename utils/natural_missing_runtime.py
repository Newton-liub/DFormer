"""Shared runtime primitives for the Natural Missing train/eval entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.builder import EncoderDecoder
from utils.training_checkpoint import CheckpointCompatibilityError


# This is the only legacy checkpoint namespace omitted by the segmentation-only
# model. It is created explicitly by the old C0 builder as an auxiliary head.
_OLD_RELIABILITY_KEYS = frozenset(
    {
        "reliability_estimator.features.gaussian_kernel",
        "reliability_estimator.features.sobel_x",
        "reliability_estimator.features.sobel_y",
        "reliability_estimator.features.laplacian",
        "reliability_estimator.head.net.0.weight",
        "reliability_estimator.head.net.0.bias",
        "reliability_estimator.head.net.2.weight",
        "reliability_estimator.head.net.2.bias",
        "reliability_estimator.head.net.4.weight",
        "reliability_estimator.head.net.4.bias",
    }
)



def build_model(config, device: torch.device | str) -> nn.Module:
    """Build the segmentation-only DFormerv2 path and place it on ``device``.

    The caller must load a compatible C0/new NaturalMissing checkpoint before using
    the model for evaluation or training. Skipping ``init_weights`` avoids a hidden
    dependency on the generic pretraining file when an exact continuation checkpoint
    is the actual initialization source.
    """
    _assert_legacy_branches_off(config)
    syncbn = bool(getattr(config, "syncbn", True))
    norm_layer = nn.SyncBatchNorm if syncbn else nn.BatchNorm2d
    model = EncoderDecoder(
        cfg=config,
        criterion=None,
        norm_layer=norm_layer,
        syncbn=syncbn,
    )
    model.criterion = nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=int(config.background),
    )
    model.to(torch.device(device))
    return model


def _assert_legacy_branches_off(config) -> None:
    """Fail closed if a config enables a frozen auxiliary experiment branch."""
    if float(getattr(config, "aux_rate", 0.0)) != 0.0:
        raise ValueError("NaturalMissing requires the segmentation auxiliary CE head to be disabled")

    for name in (
        "mmfr_a2",
        "mmfr_a2_execution_profile",
        "e1_batch1",
        "e1_batch1b",
        "e1_roe",
        "mmfr_av1",
        "mmfr_f_lite",
        "mmfr_roe",
    ):
        value = getattr(config, name, None)
        if value:
            raise ValueError(f"NaturalMissing must not enable legacy branch config {name}")


def _load_segmentation_state(model: nn.Module, state: Mapping[str, Any]) -> dict[str, list[str]]:
    """Strictly load a state mapping, dropping only old C0 reliability-head tensors."""
    target = model.module if hasattr(model, "module") else model
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint model state must be a mapping")
    if not all(isinstance(name, str) for name in state):
        raise TypeError("checkpoint model state keys must be strings")

    state_dict = dict(state)
    intentionally_dropped = sorted(name for name in state_dict if name in _OLD_RELIABILITY_KEYS)
    if intentionally_dropped and set(intentionally_dropped) != _OLD_RELIABILITY_KEYS:
        raise CheckpointCompatibilityError(
            "C0 reliability auxiliary state is incomplete: "
            f"expected all 10 confirmed keys, got {intentionally_dropped}"
        )
    filtered = {name: value for name, value in state_dict.items() if name not in _OLD_RELIABILITY_KEYS}
    target_state = target.state_dict()
    target_keys = set(target_state)
    filtered_keys = set(filtered)
    missing = sorted(target_keys - filtered_keys)
    unexpected = sorted(filtered_keys - target_keys)
    report = {
        "loaded": sorted(filtered_keys & target_keys),
        "intentionally_dropped": intentionally_dropped,
        "missing": missing,
        "unexpected": unexpected,
    }
    if missing or unexpected:
        raise CheckpointCompatibilityError(
            "NaturalMissing segmentation checkpoint key mismatch: "
            f"missing={missing}, unexpected={unexpected}; "
            f"intentionally_dropped={intentionally_dropped}"
        )

    try:
        target.load_state_dict(filtered, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            "NaturalMissing segmentation checkpoint failed strict tensor loading: "
            f"{exc}; report={report}"
        ) from exc
    return report


def load_segmentation_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, list[str]]:
    """Load a C0 or NaturalMissing checkpoint with strict segmentation-key matching.

    The caller is responsible for verifying the C0 file's externally frozen SHA-256
    identity before calling this function. This loader deliberately does not hash the
    file, so the recovery owner can perform that check once without duplicate I/O.
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"segmentation checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointCompatibilityError(f"cannot read segmentation checkpoint {path}: {exc}") from exc

    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("segmentation checkpoint has no model-state mapping")
    return _load_segmentation_state(model, state)
