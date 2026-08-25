from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.builder import EncoderDecoder
from tools.preflight_train import Preflight, check_dataset
from tools.qualify_museg_b1 import select_b1_samples
from utils.training_checkpoint import phase_uses_validation


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "local_configs" / "MUSeg" / "DFormerv2_S_4090.py"
_B1_PATH = _REPO_ROOT / "tools" / "qualify_museg_b1.py"


def test_b1_absolute_script_entrypoint_imports_from_arbitrary_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(_B1_PATH), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "real-model B1 masked-loss regression" in completed.stdout


def test_auxiliary_head_consumes_selected_backbone_feature_not_batch_axis() -> None:
    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rgb_features = [
                torch.randn(2, 4, 8, 8),
                torch.randn(2, 5, 4, 4),
                torch.randn(2, 6, 2, 2),
                torch.randn(2, 7, 1, 1),
            ]

        def forward(self, _rgb: torch.Tensor, _modal_x: torch.Tensor):
            return self.rgb_features, [feature.clone() for feature in self.rgb_features]

    class DecodeHead(nn.Module):
        def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
            return features[0][:, :2]

    class AuxHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received: torch.Tensor | None = None

        def forward(self, feature: torch.Tensor) -> torch.Tensor:
            self.received = feature
            return feature[:, :2]

    model = EncoderDecoder.__new__(EncoderDecoder)
    nn.Module.__init__(model)
    model.backbone = Backbone()
    model.decode_head = DecodeHead()
    model.aux_head = AuxHead()
    model.aux_index = 2
    rgb = torch.zeros(2, 3, 8, 8)
    modal_x = torch.zeros(2, 3, 8, 8)

    out, aux = model.encode_decode(rgb, modal_x)

    assert model.aux_head.received is model.backbone.rgb_features[2]
    assert out.shape[-2:] == (8, 8)
    assert aux.shape[-2:] == (8, 8)


def test_qualification_and_development_are_the_only_validation_phases() -> None:
    assert phase_uses_validation("qualification")
    assert phase_uses_validation("development")
    assert not phase_uses_validation("official")


def test_museg_4090_config_declares_frozen_validation_size() -> None:
    source = _CONFIG_PATH.read_text(encoding="utf-8")
    assert "C.num_eval_imgs = 318" in source


def test_dataset_preflight_requires_and_decodes_depth16(tmp_path: Path) -> None:
    root = tmp_path / "MUSeg_DFormer"
    rgb = root / "RGB"
    depth = root / "Depth"
    label = root / "Label"
    depth16 = root / "Depth16"
    for directory in (rgb, depth, label, depth16):
        directory.mkdir(parents=True)
    stem = "01-01-01-0001-sample"
    assert cv2.imwrite(str(rgb / f"{stem}.jpg"), np.full((3, 4, 3), 127, dtype=np.uint8))
    assert cv2.imwrite(str(depth / f"{stem}.png"), np.full((3, 4), 127, dtype=np.uint8))
    assert cv2.imwrite(str(label / f"{stem}.png"), np.zeros((3, 4), dtype=np.uint8))
    assert cv2.imwrite(str(depth16 / f"{stem}.png"), np.full((3, 4), 300, dtype=np.uint16))
    train_source = root / "train.txt"
    train_source.write_text(f"RGB/{stem}.jpg\n", encoding="utf-8")
    config = SimpleNamespace(
        dataset_path=str(root),
        rgb_root_folder=str(rgb),
        rgb_format=".jpg",
        x_root_folder=str(depth),
        x_format=".png",
        gt_root_folder=str(label),
        gt_format=".png",
        train_source=str(train_source),
        val_source=None,
        num_train_imgs=1,
        num_eval_imgs=None,
    )

    result = Preflight()
    check_dataset(config, result, sample_count=1)

    assert result.errors == []


def test_b1_selection_records_first_background_and_ordinary_entries(tmp_path: Path) -> None:
    labels = tmp_path / "Label"
    labels.mkdir()
    background = "01-01-01-0001-background"
    ordinary = "02-01-01-0002-ordinary"
    assert cv2.imwrite(str(labels / f"{background}.png"), np.zeros((2, 2), dtype=np.uint8))
    assert cv2.imwrite(str(labels / f"{ordinary}.png"), np.ones((2, 2), dtype=np.uint8))
    source = tmp_path / "official-train.txt"
    source.write_text(f"RGB/{background}.jpg\nRGB/{ordinary}.jpg\n", encoding="utf-8")

    assert select_b1_samples(source, labels) == {
        "all_background": f"RGB/{background}.jpg",
        "ordinary": f"RGB/{ordinary}.jpg",
    }
