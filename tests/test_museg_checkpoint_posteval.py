from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from tools.evaluate_museg_checkpoint import (
    MUSegPostEvalDataset,
    geometry_contract,
    load_model,
    metrics_from_confusion,
    restore_logits_to_metric_grid,
    sliding_logits,
    split_entries,
    update_confusion,
)
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import ValPre


def _config(*, channel_order: str = "BGR") -> SimpleNamespace:
    return SimpleNamespace(
        norm_mean=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        norm_std=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        normalization_identity="unit-sentinel-v1",
        channel_order=channel_order,
        train_scale_array=[1.0],
        image_height=480,
        image_width=640,
        pad=False,
    )


def _write_sample(root: Path, sample_id: str = "01-01-01-0001-sentinel") -> str:
    for directory in ("RGB", "Depth", "Label"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    color = np.zeros((5, 7, 3), dtype=np.uint8)
    color[:, :] = [17, 83, 211]
    depth = np.full((5, 7), 127, dtype=np.uint8)
    label = np.ones((5, 7), dtype=np.uint8)
    assert cv2.imwrite(str(root / "RGB" / f"{sample_id}.jpg"), color)
    assert cv2.imwrite(str(root / "Depth" / f"{sample_id}.png"), depth)
    assert cv2.imwrite(str(root / "Label" / f"{sample_id}.png"), label)
    return sample_id


def _production_sample(root: Path, sample_id: str, channel_order: str) -> dict[str, torch.Tensor]:
    split = root / "val-dev.txt"
    split.write_text(f"RGB/{sample_id}.jpg\n", encoding="utf-8")
    config = _config(channel_order=channel_order)
    setting = {
        "rgb_root": str(root / "RGB"),
        "rgb_format": ".jpg",
        "gt_root": str(root / "Label"),
        "gt_format": ".png",
        "transform_gt": True,
        "x_root": str(root / "Depth"),
        "x_format": ".png",
        "x_single_channel": True,
        "class_names": ["foreground"],
        "train_source": str(split),
        "val_source": str(split),
        "test_source": None,
        "dataset_name": "MUSeg_DFormer",
        "backbone": "DFormerv2_S",
        "channel_order": channel_order,
    }
    return RGBXDataset(
        setting,
        "val",
        ValPre(config.norm_mean, config.norm_std, True, config),
    )[0]


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


@pytest.mark.parametrize("channel_order", ["BGR", "RGB"])
def test_posteval_matches_production_valpre_channel_contract(tmp_path: Path, channel_order: str) -> None:
    sample_id = _write_sample(tmp_path)
    production = _production_sample(tmp_path, sample_id, channel_order)
    config = _config(channel_order=channel_order)

    post = MUSegPostEvalDataset(
        tmp_path,
        [f"RGB/{sample_id}.jpg"],
        "original-full",
        config,
        channel_order,
    )[0]

    torch.testing.assert_close(post["rgb"], production["data"])
    torch.testing.assert_close(post["depth"], production["modal_x"])
    torch.testing.assert_close(post["label"], production["label"])
    decoded_bgr = cv2.imread(str(tmp_path / "RGB" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
    expected = decoded_bgr if channel_order == "BGR" else decoded_bgr[:, :, ::-1]
    np.testing.assert_allclose(post["rgb"].numpy(), expected.transpose(2, 0, 1) / 255.0, rtol=0, atol=1e-7)


def test_resize_changes_only_model_input_and_restores_original_metric_grid(tmp_path: Path) -> None:
    sample_id = _write_sample(tmp_path)
    sample = MUSegPostEvalDataset(
        tmp_path,
        [f"RGB/{sample_id}.jpg"],
        "resize-480x640",
        _config(),
        "BGR",
    )[0]

    assert sample["rgb"].shape[-2:] == (480, 640)
    assert sample["depth"].shape[-2:] == (480, 640)
    assert sample["label"].shape == (5, 7)
    logits = torch.randn(1, 2, 480, 640)
    restored = restore_logits_to_metric_grid(logits, sample["label"].unsqueeze(0))
    assert restored.shape == (1, 2, 5, 7)

    contract = geometry_contract("resize-480x640", [(5, 7)])
    assert contract["input_geometry"]["resize_size_hw"] == [480, 640]
    assert contract["metric_geometry"]["name"] == "original-label-grid"
    assert contract["metric_geometry"]["output_sizes_hw"] == [[5, 7]]
