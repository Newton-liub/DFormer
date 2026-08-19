"""Loss reductions that remain finite for empty valid-pixel sets."""

from __future__ import annotations

import torch


def safe_masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Average ``values`` over ``valid_mask`` without breaking autograd."""
    if values.shape != valid_mask.shape:
        raise ValueError(f"values and valid_mask must have the same shape: {values.shape} != {valid_mask.shape}")

    selected = values[valid_mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()