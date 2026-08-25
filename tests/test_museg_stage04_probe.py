from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_museg_probe import main as summarize_probe


def _write_run(tmp_path: Path, *, steps: int = 60, bad_attempt: int | None = None) -> Path:
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "training_result.json").write_text(
        json.dumps({"run_kind": "probe", "completed_optimizer_steps": steps}), encoding="utf-8"
    )
    (run / "run_manifest.json").write_text(
        json.dumps({"protocol_id": "unit", "seed": 7, "run_id": "probe-unit"}), encoding="utf-8"
    )
    (run / "command.json").write_text(
        json.dumps({"argv": ["python", "train.py", "--batch-size", "8"]}), encoding="utf-8"
    )
    (run / "train.exit_code").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
    (run / "launcher.log").write_text("complete\n", encoding="utf-8")
    records = []
    for step in range(1, steps + 1):
        records.append({
            "attempt": bad_attempt - 1 if bad_attempt and step == bad_attempt else step,
            "completed_optimizer_steps": step,
            "optimizer_step_completed": True,
            "step_seconds": 0.5 + step / 1000,
            "images_per_second": 8 / (0.5 + step / 1000),
            "free_mib": 4096.0,
            "free_ratio": 0.5,
            "loss": 1.0,
            "amp_scale": 1024.0,
            "safety_passed": True,
            "allocated_mib": 2048.0,
            "reserved_mib": 3072.0,
        })
    (run / "probe-telemetry.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return run


def test_probe_summary_excludes_warmup_and_requires_exact_steps(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    output = tmp_path / "result.json"
    assert summarize_probe([
        "--run-dir", str(run), "--output", str(output),
        "--required-steps", "60", "--warmup-steps", "10",
        "--min-free-vram-gib", "2", "--min-free-vram-ratio", "0.1",
    ]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["eligible"] is True
    assert result["completion"]["completed_optimizer_steps"] == 60
    assert result["windows"]["stable_steps"] == 50
    assert result["stable_throughput_images_per_second"]["count"] == 50
    assert result["batch_size"] == 8


def test_probe_summary_rejects_incomplete_or_nonmonotonic_evidence(tmp_path: Path) -> None:
    incomplete = _write_run(tmp_path / "incomplete", steps=59)
    incomplete_output = tmp_path / "incomplete-result.json"
    summarize_probe([
        "--run-dir", str(incomplete), "--output", str(incomplete_output),
        "--required-steps", "60", "--warmup-steps", "10",
        "--min-free-vram-gib", "2", "--min-free-vram-ratio", "0.1",
    ])
    assert json.loads(incomplete_output.read_text(encoding="utf-8"))["eligible"] is False

    malformed = _write_run(tmp_path / "malformed", bad_attempt=9)
    malformed_output = tmp_path / "malformed-result.json"
    summarize_probe([
        "--run-dir", str(malformed), "--output", str(malformed_output),
        "--required-steps", "60", "--warmup-steps", "10",
        "--min-free-vram-gib", "2", "--min-free-vram-ratio", "0.1",
    ])
    result = json.loads(malformed_output.read_text(encoding="utf-8"))
    assert result["eligible"] is False
    assert result["anomaly"]["class"] == "evidence"