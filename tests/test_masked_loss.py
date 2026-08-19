from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


_safe_loss_path = Path(__file__).resolve().parents[1] / "models" / "losses" / "safe_masked_loss.py"
_safe_loss_spec = importlib.util.spec_from_file_location("safe_masked_loss", _safe_loss_path)
if _safe_loss_spec is None or _safe_loss_spec.loader is None:
    raise ImportError(f"unable to load {_safe_loss_path}")
_safe_loss_module = importlib.util.module_from_spec(_safe_loss_spec)
_safe_loss_spec.loader.exec_module(_safe_loss_module)
safe_masked_mean = _safe_loss_module.safe_masked_mean


BACKGROUND = 255


def legacy_masked_mean(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel_loss = torch.nn.functional.cross_entropy(
        logits,
        target,
        reduction="none",
        ignore_index=BACKGROUND,
    )
    return pixel_loss[target != BACKGROUND].mean()


def safe_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel_loss = torch.nn.functional.cross_entropy(
        logits,
        target,
        reduction="none",
        ignore_index=BACKGROUND,
    )
    return safe_masked_mean(pixel_loss, target != BACKGROUND)


class SafeMaskedMeanTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.logits = torch.randn(1, 15, 10, 10, dtype=torch.float64)

    def make_target(self, valid_pixels: int) -> torch.Tensor:
        target = torch.full((1, 10, 10), BACKGROUND, dtype=torch.long)
        if valid_pixels:
            target.view(-1)[:valid_pixels] = torch.arange(valid_pixels) % 15
        return target

    def test_a1_empty_valid_pixels_is_non_finite_with_legacy_reduction(self) -> None:
        target = self.make_target(0)
        loss = legacy_masked_mean(self.logits, target)
        self.assertFalse(torch.isfinite(loss).item())

    def test_a1_non_empty_pixel_counts_are_finite(self) -> None:
        for valid_pixels in (1, 100):
            target = self.make_target(valid_pixels)
            loss = legacy_masked_mean(self.logits, target)
            self.assertTrue(torch.isfinite(loss).item())

    def test_b1_empty_valid_pixels_returns_graph_connected_zero(self) -> None:
        logits = self.logits.clone().requires_grad_()
        target = self.make_target(0)
        loss = safe_loss(logits, target)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_b1_non_empty_loss_and_gradient_match_legacy_reduction(self) -> None:
        target = self.make_target(5)
        legacy_logits = self.logits.clone().requires_grad_()
        safe_logits = self.logits.clone().requires_grad_()

        legacy_loss = legacy_masked_mean(legacy_logits, target)
        safe_loss_value = safe_loss(safe_logits, target)
        legacy_loss.backward()
        safe_loss_value.backward()

        self.assertTrue(torch.allclose(legacy_loss, safe_loss_value, atol=1e-12, rtol=0.0))
        self.assertTrue(torch.allclose(legacy_logits.grad, safe_logits.grad, atol=1e-12, rtol=0.0))

    def test_b1_auxiliary_weight_preserves_non_empty_behavior(self) -> None:
        target = self.make_target(5)
        main_logits = self.logits.clone().requires_grad_()
        aux_logits = (self.logits * 0.5).clone().requires_grad_()
        aux_rate = 0.4

        loss = safe_loss(main_logits, target) + aux_rate * safe_loss(aux_logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(main_logits.grad).all().item())
        self.assertTrue(torch.isfinite(aux_logits.grad).all().item())

    def test_valid_pixel_count_is_local_to_each_mask(self) -> None:
        target_empty = self.make_target(0)
        target_one = self.make_target(1)

        empty_loss = safe_loss(self.logits, target_empty)
        one_pixel_loss = safe_loss(self.logits, target_one)

        self.assertEqual(empty_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(one_pixel_loss).item())


if __name__ == "__main__":
    unittest.main()