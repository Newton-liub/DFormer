#!/usr/bin/env python3
"""CPU-only audit of MMFR-A2 v3 validity transport.

This is a synthetic fixture audit.  It proves that intensity corruptions do not create
new valid measurements and that MID-A applies the same integer translation to Depth and
its explicit validity state.  It never reads a checkpoint or official test data and
never enters the training loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.dataloader import multimodal_failure_v3 as basis
from utils.dataloader.mmfr_training_v3 import CORRUPTION_BASIS, PROTOCOL_ID


class Checks:
    def __init__(self) -> None:
        self.assertions = 0
        self.failures: list[Dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.assertions += 1
        if not condition:
            self.failures.append({"name": name, "detail": detail})


def make_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 48, 56
    rows = np.arange(height, dtype=np.uint16)[:, None]
    cols = np.arange(width, dtype=np.uint16)[None, :]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = ((rows * 3 + cols * 7 + 21) % 240 + 1).astype(np.uint8)
    rgb[:, :, 1] = ((rows * 11 + cols * 2 + 31) % 240 + 1).astype(np.uint8)
    rgb[:, :, 2] = ((rows * 5 + cols * 13 + 41) % 240 + 1).astype(np.uint8)
    depth = ((rows * 7 + cols * 9 + 19) % 220 + 20).astype(np.uint8)
    depth[5:17, 8:22] = np.uint8(0)
    depth[30:38, 38:47] = np.uint8(0)
    geometry = np.ones((height, width), dtype=bool)
    geometry[-4:, :] = False
    geometry[:, -5:] = False
    rgb[~geometry] = np.uint8(0)
    depth[~geometry] = np.uint8(0)
    return rgb, depth, (depth > 0) & geometry


def run_audit() -> Dict[str, Any]:
    checks = Checks()
    rgb, depth, initial_state = make_fixture()
    intensity_records = []
    for index, kind in enumerate(("gaussian_noise", "blur", "quantization")):
        for severity in (0.25, 0.5, 1.0):
            result = basis.apply_failures(
                rgb,
                depth,
                [basis.FailureSpec("depth", kind, severity)],
                np.random.default_rng(1000 + index),
                depth_validity=initial_state,
                validity_mask=np.ones_like(initial_state),
            )
            out_depth = result.depth if result.depth.ndim == 2 else result.depth[:, :, 0]
            newly_valid = int(np.count_nonzero((~initial_state) & (out_depth > 0)))
            state_newly_valid = int(np.count_nonzero((~initial_state) & result.validity_state))
            checks.check(f"{kind}_severity_{severity}_newly_valid_zero", newly_valid == 0, str(newly_valid))
            checks.check(f"{kind}_severity_{severity}_state_newly_valid_zero", state_newly_valid == 0, str(state_newly_valid))
            checks.check(f"{kind}_severity_{severity}_state_unchanged", np.array_equal(result.validity_state, initial_state))
            checks.check(f"{kind}_severity_{severity}_invalid_output_zero", bool(np.all(out_depth[~result.validity_state] == 0)))
            intensity_records.append({"kind": kind, "severity": severity, "newly_valid_pixels": newly_valid, "state_newly_valid_pixels": state_newly_valid})

    # MID-A checks use several deterministic shifts and retain the exact recorded dy/dx.
    alignment_records = []
    for index, severity in enumerate((0.25, 0.5, 1.0)):
        result = basis.apply_failures(
            rgb,
            depth,
            [basis.FailureSpec("depth", "misalignment", severity)],
            np.random.default_rng(2000 + index),
            depth_validity=initial_state,
            validity_mask=np.ones_like(initial_state),
        )
        record = result.metadata["specs"][0]
        dy, dx = int(record["dy"]), int(record["dx"])
        expected_depth = basis._translate(depth, dy, dx)
        expected_state = basis._translate(initial_state, dy, dx)
        expected_depth = basis._depth_finalize(expected_depth, expected_state, np.ones_like(initial_state))
        out_depth = result.depth if result.depth.ndim == 2 else result.depth[:, :, 0]
        checks.check(f"mid_a_severity_{severity}_depth_translation", np.array_equal(out_depth, expected_depth), f"dy={dy},dx={dx}")
        checks.check(f"mid_a_severity_{severity}_state_translation", np.array_equal(result.validity_state, expected_state), f"dy={dy},dx={dx}")
        checks.check(f"mid_a_severity_{severity}_state_matches_output", np.array_equal(result.validity_state, out_depth > 0))
        checks.check(f"mid_a_severity_{severity}_same_integer_shift", record["corruption_parameter"]["transport"] == "integer-translate-depth-and-validity-together")
        alignment_records.append({"severity": severity, "dy": dy, "dx": dx, "newly_valid_pixels": int(np.count_nonzero((~initial_state) & result.validity_state))})

    return {
        "schema_version": "mmfr-a2-v3-validity-transport-audit-v1",
        "protocol_id": PROTOCOL_ID,
        "corruption_basis": CORRUPTION_BASIS,
        "assertions": checks.assertions,
        "failed": len(checks.failures),
        "status": "PASS" if not checks.failures else "FAIL",
        "failures": checks.failures,
        "intensity_records": intensity_records,
        "alignment_records": alignment_records,
        "fixture": {"height": 48, "width": 56, "native_invalid_regions": 2},
        "official_test_included": False,
        "checkpoint_read": False,
        "training_run": False,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = run_audit()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
