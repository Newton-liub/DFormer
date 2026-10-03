"""CPU-only readback of this project's own formal NaturalMissing artifacts.

Loads fixed finals on CPU and checks schema, provenance, counters and the hash
before S1, or against completed S1 hashes on retrieval. It never constructs a
model or performs a forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

C0_SHA = "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a"
STRATEGIES = ("Natural", "Grid", "Replay")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_final(path: Path, strategy: str, git_sha: str, s1_hashes: set[str] | None = None) -> dict:
    _require(path.is_file(), f"missing final: {path}")
    size = path.stat().st_size
    _require(300_000_000 < size < 400_000_000, f"unexpected checkpoint size: {size}")
    sha = _file_sha(path)
    if s1_hashes is not None:
        _require(sha in s1_hashes, "local final hash is absent from completed S1 reports")
    # Only the explicitly named, locally generated and trusted run artifacts.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("schema_version") == "dformer-training-checkpoint-v2", "schema mismatch")
    protocol = checkpoint["protocol"]
    summary = protocol["config_summary"]
    natural = checkpoint["natural_missing"]
    identity = natural["resume_identity"]
    cursor = natural["cursor"]
    _require(protocol["git_commit"] == git_sha, "formal-run Git SHA mismatch")
    _require(protocol["run_id"] == f"NaturalMissing-{strategy}-formal", "run identity mismatch")
    _require(natural["execution_mode"] == "formal" and natural["preflight_only"] is False, "not formal eligible")
    for key, expected in {"strategy": strategy, "mode": "formal", "c0_sha256": C0_SHA,
                          "train_sample_count": 1277, "batch_size": 10, "num_workers": 8,
                          "image_size": [480, 640], "iterations_per_epoch": 128}.items():
        _require(identity.get(key) == expected, f"resume identity mismatch: {key}")
    for key, expected in {"strategy": strategy, "execution_mode": "formal", "c0_sha256": C0_SHA,
                          "successful_update_budget": 2560, "recovery_interval_successful_updates": 640,
                          "base_lr": 1e-6, "weight_decay": 0.01, "batch_size": 10,
                          "num_workers": 8, "amp": True, "amp_dtype": "float16",
                          "syncbn": True, "legacy_auxiliary_training": False,
                          "nmf_training_precision": "fp32_local"}.items():
        _require(summary.get(key) == expected, f"scientific contract mismatch: {key}")
    for key, expected in {"epoch": 21, "batch_position": 0, "successful_updates": 2560,
                          "attempted_steps": 2560, "skipped_steps": 0}.items():
        _require(cursor.get(key) == expected, f"counter/cursor mismatch: {key}")
    _require(checkpoint["global_optimizer_step"] == 2560, "global step mismatch")
    _require(len(checkpoint["model"]) == 802, "model key count mismatch")
    states = checkpoint["optimizer"]["state"]
    steps = sorted({float(state["step"]) for state in states.values()})
    _require(len(states) == 685 and steps == [2560.0], "optimizer state/step mismatch")
    _require(checkpoint["rng_state"] and natural["epoch_rng_state"], "missing RNG state")
    _require(checkpoint["best_val_miou"] is None and checkpoint["best_val_epoch"] is None,
             "unexpected best-val selection")
    report = {"path": str(path.resolve()), "bytes": size, "sha256": sha,
              "schema_version": checkpoint["schema_version"], "model_keys": len(checkpoint["model"]),
              "optimizer_states": len(states), "optimizer_steps": steps,
              "global_optimizer_step": checkpoint["global_optimizer_step"], "cursor": cursor,
              "git_commit": protocol["git_commit"], "config_summary": summary,
              "execution_mode": "formal", "preflight_only": False, "formal_eligible": True,
              "rng_saved": True, "amp_scaler": checkpoint["amp_scaler"], "passed": True}
    del checkpoint
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--formal-sha", required=True)
    args = parser.parse_args()
    root = args.local_root.resolve()
    receipt = json.loads((root / "monitor" / "formal-run-receipt.json").read_text(encoding="utf-8"))
    _require(receipt["status"] == "SUCCEEDED" and receipt["source_commit"] == args.formal_sha,
             "formal receipt was not successful at expected SHA")
    # Find only the four evaluator JSON artifacts, excluding unrelated history.
    reports = []
    for path in (root / "monitor").rglob("*.json"):
        if path.parent.name != "checkpoints":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if "s1_conditions" in value:
            reports.append(value)
    _require(len(reports) == 4, "expected exactly four completed S1 checkpoint reports")
    hashes = set()
    for report in reports:
        _require(report["status"] == "completed" and report["official_test_included"] is False,
                 "invalid S1 status or test boundary")
        _require(report["identity"]["sample_count"] == 318, "S1 sample count mismatch")
        _require(len(report["s1_conditions"]) == 3, "S1 condition count mismatch")
        _require(all(condition["sample_count"] == 318 for condition in report["s1_conditions"].values()),
                 "incomplete S1 condition")
        _require(report["forward_rng_policy"]["counts"]["forward_calls"] == 954,
                 "S1 forward count mismatch")
        hashes.add(report["identity"]["checkpoint_sha256"])
    _require(C0_SHA in hashes and len(hashes) == 4, "S1 C0 identity/four unique hashes mismatch")
    readbacks = {strategy: read_final(root / "checkpoints" / strategy / "update-2560.pth",
                                     strategy, args.formal_sha, hashes) for strategy in STRATEGIES}
    result = {"mode": "CPU_ONLY_NO_MODEL_FORWARD", "s1_views_verified": 3816,
              "formal_sha": args.formal_sha, "checkpoint_readback": readbacks, "passed": True}
    destination = root / "checkpoint-readback.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
