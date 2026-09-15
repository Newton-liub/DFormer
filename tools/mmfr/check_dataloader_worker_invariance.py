#!/usr/bin/env python3
"""Hard gate: does the DataLoader worker count change the frozen training input trajectory?

The MMFR-A2 v3 corruption helper runs *after* the DataLoader geometry augmentation, so a
worker-count change can only be treated as an implementation-only execution change if the
tensors that enter the helper are bitwise identical.  This tool records, for N consecutive
batches, the sample ids/order and the SHA-256 of the RGB / Depth / label tensors that arrive
just before ``build_mmfr_training_batch_v3``, then compares two worker counts.

Outputs ``dataloader-worker-count-changes-training-input-trajectory=true/false``.

CPU-only; never reads the official test split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.museg_protocol import write_json  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.dataloader.dataloader import get_train_loader  # noqa: E402

SCHEMA = "mmfr-a2-v3-dataloader-worker-invariance-v1"


class _SingleProcessEngine:
    distributed = False
    world_size = 1
    local_rank = 0


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def record_inputs(config_module: str, workers: int, batches: int, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = getattr(importlib.import_module(config_module), "C")
    config.num_workers = int(workers)
    loader, _ = get_train_loader(_SingleProcessEngine(), RGBXDataset, config)
    rows: List[Dict[str, Any]] = []
    iterator = iter(loader)
    for _ in range(batches):
        minibatch = next(iterator)
        sample_ids = minibatch.get("fn")
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        rows.append(
            {
                "sample_ids": [str(value) for value in (sample_ids or [])],
                "rgb_sha256": _sha256(minibatch["data"]),
                "depth_sha256": _sha256(minibatch["modal_x"]),
                "label_sha256": _sha256(minibatch["label"]),
            }
        )
    return {
        "workers": int(workers),
        "seed": int(seed),
        "batches": batches,
        "rows": rows,
        "all_input_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def compare(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    differences: List[Dict[str, Any]] = []
    for index, (expected, actual) in enumerate(zip(reference["rows"], candidate["rows"])):
        for field in ("sample_ids", "rgb_sha256", "depth_sha256", "label_sha256"):
            if expected[field] != actual[field]:
                differences.append(
                    {
                        "batch_index": index,
                        "field": field,
                        "reference": expected[field] if field == "sample_ids" else expected[field][:16],
                        "candidate": actual[field] if field == "sample_ids" else actual[field][:16],
                    }
                )
                break
        if len(differences) >= 5:
            break
    identical = reference["all_input_sha256"] == candidate["all_input_sha256"]
    return {
        "reference_workers": reference["workers"],
        "candidate_workers": candidate["workers"],
        "batches_compared": min(len(reference["rows"]), len(candidate["rows"])),
        "reference_all_input_sha256": reference["all_input_sha256"],
        "candidate_all_input_sha256": candidate["all_input_sha256"],
        "inputs_bitwise_identical": identical,
        "dataloader_worker_count_changes_training_input_trajectory": not identical,
        "first_differences": differences,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3")
    parser.add_argument("--reference-workers", type=int, default=8, help="the frozen production setting")
    parser.add_argument("--candidate-workers", type=int, default=4)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026091402)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "mmfr-a2-v3-pipeline-profile" / "dataloader-worker-invariance.json"),
    )
    args = parser.parse_args(argv)

    reference = record_inputs(args.config, args.reference_workers, args.batches, args.seed)
    candidate = record_inputs(args.config, args.candidate_workers, args.batches, args.seed)
    verdict = compare(reference, candidate)
    report = {
        "schema_version": SCHEMA,
        "scientific_protocol": "MMFR-A2-train-integration-v3",
        "scientific_semantics_changed": False,
        "official_test_included": False,
        "config_module": args.config,
        "verdict": verdict,
        "reference_trajectory": reference,
        "candidate_trajectory": candidate,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    print(json.dumps({k: v for k, v in verdict.items() if k != "first_differences"}, indent=2, ensure_ascii=False))
    if verdict["first_differences"]:
        print(json.dumps(verdict["first_differences"][:3], indent=2, ensure_ascii=False))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
