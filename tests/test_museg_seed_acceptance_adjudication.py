from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.adjudicate_museg_seed_acceptance import adjudicate, main


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "seed-7"
    run.mkdir()
    identity = {
        "protocol_id": "museg-development-long500-v2",
        "protocol_manifest_sha256": "a" * 64,
        "phase": "development",
        "seed": 7,
    }
    _write_json(run / "run_manifest.json", {**identity, "exit_code": 0, "process_exit_code": 0})
    _write_json(
        run / "training_result.json",
        {
            **identity,
            "exit_code": 0,
            "final_epoch": 500,
            "best_val_miou": 52.84,
            "best_val_epoch": 460,
            "completed_optimizer_steps": 63973,
            "attempted_steps": 64000,
            "checkpoint": {"path": "/preserved/checkpoint/latest.pth", "sha256": "b" * 64},
            "official_test_included": False,
        },
    )
    _write_json(
        run / "run_config.json",
        {"data": {"splits": {"test": {"sealed_unread": True}}}},
    )
    lines = []
    validation = []
    best = 0.0
    for epoch in range(1, 501):
        lines.append(
            f"Epoch {epoch}/500 Iter 128/128: lr=1.0000e-05 loss=1.0000 "
            f"total_loss=1.1000\n"
        )
        if epoch % 10 == 0:
            miou = round(epoch / 10.0, 2)
            best = max(best, miou)
            lines.append(f"Epoch {epoch} validation result: mIoU {miou}, best mIoU {best}\n")
            validation.append({"epoch": epoch, "miou": miou, "best_miou": best})
    (run / "train.log").write_text("".join(lines), encoding="utf-8")
    evidence = {
        path.name: _sha(path)
        for path in (
            run / "run_manifest.json",
            run / "training_result.json",
            run / "run_config.json",
            run / "train.log",
        )
    }
    acceptance = tmp_path / "acceptance.json"
    _write_json(
        acceptance,
        {
            "schema_version": "museg-stage05-seed-acceptance-v1",
            "pass": False,
            "errors": ["milestones_complete"],
            "checks": [
                {"name": "identity", "pass": True, "details": None},
                {"name": "milestones_complete", "pass": False, "details": [1, 10]},
                {"name": "validation_epoch_sequence", "pass": True, "details": list(range(10, 501, 10))},
            ],
            "evidence_sha256": evidence,
            "official_test_read": False,
            **identity,
            "best_val_miou": 52.84,
            "best_val_epoch": 460,
            "validation_curve": validation,
        },
    )
    return acceptance, run


def test_adjudication_passes_only_correctable_v1_milestone_failure(tmp_path: Path) -> None:
    acceptance, run = _fixture(tmp_path)

    report = adjudicate(acceptance, run)

    assert report["pass"] is True
    assert report["errors"] == []
    assert report["epoch_end_log_count"] == 500
    assert report["validation_point_count"] == 50
    assert [row["epoch"] for row in report["milestones"]] == [1, 10, 20, 50, 100, 200, 300, 400, 500]
    assert report["original_acceptance"]["sha256"] == _sha(acceptance)


def test_adjudication_rejects_any_evidence_hash_change(tmp_path: Path) -> None:
    acceptance, run = _fixture(tmp_path)
    (run / "train.log").write_text((run / "train.log").read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    report = adjudicate(acceptance, run)

    assert report["pass"] is False
    assert "evidence_sha256_unchanged" in report["errors"]


def test_cli_refuses_to_overwrite_original_acceptance(tmp_path: Path) -> None:
    acceptance, run = _fixture(tmp_path)

    try:
        main(["--acceptance", str(acceptance), "--run-dir", str(run), "--output", str(acceptance)])
    except SystemExit as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("expected immutable input overwrite refusal")
