from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tools.evaluate_museg_checkpoint import (
    load_model,
    metrics_from_confusion,
    sliding_logits,
    split_entries,
    update_confusion,
)


def test_metrics_emit_miou_macc_mf1_and_per_class() -> None:
    hist = np.asarray([[8, 2], [1, 9]], dtype=np.int64)

    result = metrics_from_confusion(hist, ["first", "second"])

    assert result["miou"] == 73.86
    assert result["macc"] == 85.0
    assert result["mf1"] == 84.96
    assert [row["name"] for row in result["per_class"]] == ["first", "second"]
    assert result["per_class"][0]["target_pixels"] == 10


def test_split_identity_rejects_duplicates_and_official_test_names(tmp_path: Path) -> None:
    good = tmp_path / "val-dev.txt"
    good.write_text("RGB/a.jpg\nRGB/b.jpg\n", encoding="utf-8")
    assert split_entries(good) == ["RGB/a.jpg", "RGB/b.jpg"]

    good.write_text("RGB/a.jpg\nRGB/a.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        split_entries(good)

    good.write_text("RGB/official-test-a.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="official test"):
        split_entries(good)


def test_load_model_skips_separate_pretrained_initialization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _CheckpointOnlyModel(nn.Module):
        def __init__(self, *, cfg: object, criterion: object, norm_layer: object, syncbn: bool) -> None:
            super().__init__()
            captured.update(criterion=criterion, norm_layer=norm_layer, syncbn=syncbn)
            self.weight = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr("tools.evaluate_museg_checkpoint.EncoderDecoder", _CheckpointOnlyModel)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": {"weight": torch.ones(1)}}, checkpoint)

    model = load_model(object(), checkpoint, torch.device("cpu"))

    assert captured["criterion"] is None
    assert captured["norm_layer"] is nn.BatchNorm2d
    assert captured["syncbn"] is False
    torch.testing.assert_close(model.weight, torch.ones(1))
    assert model.training is False


class _PointModel(nn.Module):
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        value = rgb[:, :1] + depth[:, :1]
        return torch.cat((value, -value), dim=1)


def test_sliding_logits_preserve_original_geometry_and_pointwise_values() -> None:
    rgb = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7)
    depth = torch.ones_like(rgb)

    actual = sliding_logits(_PointModel(), rgb, depth, height=4, width=4, stride_rate=0.5)
    expected = _PointModel()(rgb, depth)

    assert actual.shape == (1, 2, 5, 7)
    torch.testing.assert_close(actual, expected)


def test_update_confusion_ignores_background_label() -> None:
    logits = torch.tensor([[[[5.0, 0.0], [0.0, 1.0]], [[0.0, 5.0], [5.0, 0.0]]]])
    labels = torch.tensor([[[0, 1], [255, 0]]])
    hist = np.zeros((2, 2), dtype=np.int64)

    update_confusion(hist, logits, labels, 2)

    np.testing.assert_array_equal(hist, np.asarray([[2, 0], [0, 1]]))
