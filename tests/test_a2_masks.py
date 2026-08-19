from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


_script_path = Path(__file__).resolve().parents[1] / "tools" / "mve" / "a2_prepare_masks.py"
_script_spec = importlib.util.spec_from_file_location("a2_prepare_masks", _script_path)
if _script_spec is None or _script_spec.loader is None:
    raise ImportError(f"unable to load {_script_path}")
_script_module = importlib.util.module_from_spec(_script_spec)
sys.modules[_script_spec.name] = _script_module
_script_spec.loader.exec_module(_script_module)
SampleStats = _script_module.SampleStats
build_masks = _script_module.build_masks
select_block_mask = _script_module.select_block_mask


class A2MaskPreparationTest(unittest.TestCase):
    def test_block_mask_selects_exact_target_count_from_valid_pixels(self) -> None:
        valid = np.ones((40, 40), dtype=bool)
        mask = select_block_mask(valid, 600, np.random.default_rng(3), 2)
        self.assertEqual(int(mask.sum()), 600)
        self.assertTrue(np.all(mask[valid] >= 0))

    def test_manifest_stores_portable_relative_mask_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            depth_path = root / "Depth16" / "sample.png"
            depth_path.parent.mkdir()
            cv2.imwrite(str(depth_path), np.full((10, 10), 100, dtype=np.uint16))
            stats = SampleStats(
                sample_id="sample",
                depth16_path=str(depth_path),
                depth_path="depth.png",
                label_path="label.png",
                height=10,
                width=10,
                valid_depth_ratio=1.0,
                foreground_pixels=100,
            )
            output_root = root / "mve"
            build_masks(output_root, [stats], [0.0, 0.3], 7, 1)
            manifest = json.loads((output_root / "a2_screening_manifest.json").read_text(encoding="utf-8"))
            paths = [condition["mask_path"] for condition in manifest["samples"][0]["conditions"]]
            self.assertTrue(all(not Path(path).is_absolute() for path in paths))
            self.assertTrue(all((output_root / path).is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()