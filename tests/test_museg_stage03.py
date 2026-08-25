from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.museg_protocol import ProtocolError, load_protocol
from tools.preflight_train import audit_protocol
from tools.run_museg_3seed import main as orchestrate_main
from tools.run_museg_seed import _git, main as run_seed_main
from tools.summarize_museg_runs import main as summarize_main
from utils.experiment_tracker import build_museg_run_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_protocol(tmp_path: Path, *, phase: str = "development", seeds=(11, 22, 33)) -> Path:
    split_dir = tmp_path / "split root #"
    split_dir.mkdir(parents=True)
    entries = {
        "train_dev": ["01-01-01-0001-a", "02-01-01-0002-b"],
        "val_dev": ["03-01-01-0003-c"],
        "official_train": ["01-01-01-0001-a", "02-01-01-0002-b", "03-01-01-0003-c"],
        "official_test": ["04-01-01-0004-d"],
    }
    splits = {}
    for role, lines in entries.items():
        path = split_dir / f"{role}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        splits[role] = {
            "path": str(path),
            "samples": len(lines),
            "groups": len(lines),
            "sha256": _sha(path),
            "sealed_unread": role == "official_test",
        }
    weight = tmp_path / "pretrained #.pth"
    weight.write_bytes(b"fake pretrained weights")
    manifest = {
        "schema_version": "museg-training-protocol-v1",
        "protocol_id": "museg-dev-unit-v1",
        "schedule_version": "schedule-unit-v1",
        "phase": phase,
        "model": "DFormerv2-S",
        "config_module": "local_configs.MUSeg.DFormerv2_S_4090",
        "git": {"required_commit": "0" * 40},
        "seeds": list(seeds),
        "output_root": str(tmp_path / "output root #"),
        "splits": splits,
        "pretrained": {
            "path": str(weight),
            "size_bytes": weight.stat().st_size,
            "sha256": _sha(weight),
        },
        "training": {
            "epochs": 20,
            "batch_size": 8,
            "val_batch_size": 1,
            "workers": 2,
            "eval_start_epoch": 5,
            "eval_interval": 5,
            "save_interval": 5,
            "amp": True,
            "compile": False,
            "syncbn": False,
            "sliding": False,
            "mst": False,
        },
        "swanlab": {"mode": "disabled", "project": "test", "workspace": "test"},
    }
    path = tmp_path / "protocol #.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_fake_trainer(tmp_path: Path) -> Path:
    path = tmp_path / "fake trainer #.py"
    path.write_text(
        "import argparse, hashlib, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--seed',type=int); p.add_argument('--run-id'); "
        "p.add_argument('--output-dir'); p.add_argument('--resume'); p.add_argument('--resume-parent-run-id'); "
        "p.add_argument('--resume-checkpoint-sha256'); p.add_argument('--protocol-id'); "
        "p.add_argument('--protocol-manifest-sha256'); p.add_argument('--experiment-phase'); a,x=p.parse_known_args()\n"
        "out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); "
        "checkpoint=out/'checkpoint'/'latest.pth'; checkpoint.parent.mkdir(parents=True,exist_ok=True); "
        "checkpoint.write_bytes(b'fake checkpoint'); checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest(); "
        "(out/'training_result.json').write_text(json.dumps({"
        "'schema_version':'museg-training-result-v1','protocol_id':a.protocol_id,"
        "'protocol_manifest_sha256':a.protocol_manifest_sha256,'phase':a.experiment_phase,"
        "'seed':a.seed,'run_id':a.run_id,'best_val_miou':0.5,"
        "'best_val_epoch':5,'final_epoch':20,'duration_seconds':1.25,'exit_code':0,"
        "'checkpoint':{'path':str(checkpoint.resolve()),'sha256':checkpoint_sha},"
        "'official_test_included':False,'extra_args':x}),encoding='utf-8')\n",
        encoding="utf-8",
    )
    return path


def test_single_seed_launcher_passes_protocol_arguments_and_writes_json(tmp_path: Path) -> None:
    protocol = _write_protocol(tmp_path)
    trainer = _write_fake_trainer(tmp_path)

    assert run_seed_main([
        "--protocol-manifest", str(protocol), "--seed", "11", "--direct",
        "--train-program", str(trainer), "--python", sys.executable,
    ]) == 0

    run_dir = tmp_path / "output root #" / "museg-dev-unit-v1" / "development" / "seed-11"
    command = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert command["argv"][0:2] == [sys.executable, str(trainer)]
    for value in ("--epochs", "20", "--batch-size", "8", "--seed", "11"):
        assert value in command["argv"]
    assert run["exit_code"] == 0
    assert run["protocol_manifest_sha256"] == _sha(protocol)
    assert (run_dir / "launcher.log").is_file()
    assert json.loads((run_dir / "train.exit_code").read_text(encoding="utf-8"))["exit_code"] == 0


def test_single_seed_rejects_nonempty_output_and_resume_requires_parent_and_sha(tmp_path: Path) -> None:
    protocol = _write_protocol(tmp_path)
    trainer = _write_fake_trainer(tmp_path)
    assert run_seed_main(["--protocol-manifest", str(protocol), "--seed", "11", "--direct", "--train-program", str(trainer)]) == 0
    assert run_seed_main(["--protocol-manifest", str(protocol), "--seed", "11", "--direct", "--train-program", str(trainer)]) != 0

    protocol2 = _write_protocol(tmp_path / "resume")
    checkpoint = tmp_path / "parent checkpoint #.pth"
    checkpoint.write_bytes(b"checkpoint")
    assert run_seed_main([
        "--protocol-manifest", str(protocol2), "--seed", "11", "--direct",
        "--train-program", str(trainer), "--resume", str(checkpoint),
    ]) != 0
    assert run_seed_main([
        "--protocol-manifest", str(protocol2), "--seed", "11", "--direct",
        "--train-program", str(trainer), "--resume", str(checkpoint),
        "--resume-parent-run-id", "parent-11", "--resume-checkpoint-sha256", _sha(checkpoint),
    ]) == 0
    run_dir = tmp_path / "resume" / "output root #" / "museg-dev-unit-v1" / "development" / "seed-11"
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run["resume"]["parent_run_id"] == "parent-11"
    assert run["resume"]["checkpoint_sha256"] == _sha(checkpoint)


def test_orchestrator_is_sequential_and_stops_after_first_failure(tmp_path: Path, monkeypatch) -> None:
    protocol = _write_protocol(tmp_path)
    calls = tmp_path / "calls.jsonl"
    launcher = tmp_path / "fake launcher.py"
    launcher.write_text(
        "import argparse,json,sys,time\n"
        "p=argparse.ArgumentParser();p.add_argument('--protocol-manifest');p.add_argument('--seed',type=int);a,x=p.parse_known_args()\n"
        f"f=open({str(calls)!r},'a',encoding='utf-8');f.write(json.dumps({{'seed':a.seed,'time':time.time()}})+'\\n');f.close()\n"
        "raise SystemExit(7 if a.seed==22 else 0)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    status = orchestrate_main([
        "--protocol-manifest", str(protocol), "--seeds", "11", "22", "33",
        "--seed-launcher", str(launcher), "--python", sys.executable,
    ])
    assert status == 7
    assert [json.loads(line)["seed"] for line in calls.read_text(encoding="utf-8").splitlines()] == [11, 22]
    report = json.loads((tmp_path / "output root #" / "museg-dev-unit-v1" / "development" / "orchestrator.json").read_text(encoding="utf-8"))
    assert report["completed_seeds"] == [11]
    assert report["failed_seed"] == 22


def test_protocol_and_preflight_reject_split_phase_weight_and_output_errors(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path)
    protocol = load_protocol(path)
    baseline = audit_protocol(protocol, repo_root=tmp_path, check_git=False)
    assert [error for error in baseline.errors if error["code"] != "package_missing"] == []

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["splits"]["val_dev"]["sha256"] = "f" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert any(e["code"] == "split_sha256_mismatch" for e in audit_protocol(load_protocol(path), repo_root=tmp_path, check_git=False).errors)

    path = _write_protocol(tmp_path / "official", phase="official")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["splits"]["official_train"] = raw["splits"]["official_test"]
    raw["pretrained"]["size_bytes"] += 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    codes = {e["code"] for e in audit_protocol(load_protocol(path), repo_root=tmp_path, check_git=False).errors}
    assert {"phase_role_error", "pretrained_size_mismatch"} <= codes

    clean = _write_protocol(tmp_path / "collision")
    raw = json.loads(clean.read_text(encoding="utf-8"))
    target = Path(raw["output_root"]) / raw["protocol_id"] / raw["phase"] / "seed-11"
    target.mkdir(parents=True)
    (target / "occupied").write_text("x", encoding="utf-8")
    assert any(e["code"] == "output_collision" for e in audit_protocol(load_protocol(clean), repo_root=tmp_path, check_git=False).errors)


def test_launcher_git_identity_excludes_only_generated_output_paths(tmp_path: Path) -> None:
    if not shutil.which("git"):
        pytest.skip("git unavailable")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("clean", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    generated = repo / "output root #"
    generated.mkdir()
    (generated / "orchestrator.json").write_text("{}", encoding="utf-8")
    assert _git(repo, (generated,))["dirty"] is False

    tracked.write_text("dirty", encoding="utf-8")
    identity = _git(repo, (generated,))
    assert identity["dirty"] is True
    assert identity["dirty_paths"] == ["tracked.txt"]


def test_preflight_reports_dirty_git_as_structured_error(tmp_path: Path) -> None:
    if not shutil.which("git"):
        pytest.skip("git unavailable")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("clean", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    (repo / "tracked.txt").write_text("dirty", encoding="utf-8")
    protocol_path = _write_protocol(tmp_path / "protocol")
    raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw["git"]["required_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    protocol_path.write_text(json.dumps(raw), encoding="utf-8")
    report = audit_protocol(load_protocol(protocol_path), repo_root=repo, check_git=True)
    dirty = next(error for error in report.errors if error["code"] == "git_dirty")
    assert "tracked.txt" in dirty["details"]["paths"]


def test_swanlab_metadata_contract_is_complete() -> None:
    config = SimpleNamespace(
        dataset_name="MUSeg", backbone="DFormerv2_S", x="Depth", x_is_single_channel=True,
        nepochs=20, niters_per_epoch=160, batch_size=8, num_workers=2, optimizer="AdamW",
        lr=6e-5, lr_power=0.9, warm_up_epoch=2, weight_decay=0.01,
        train_scale_array=[1.0], eval_scale_array=[1.0], eval_flip=False,
        eval_start_epoch=5, eval_interval=5, save_interval=5, pretrained_model="weights.pth",
    )
    args = SimpleNamespace(amp=True, compile=False, syncbn=False, sliding=False, mst=False, val_amp=True)
    metadata = build_museg_run_config(
        config=config, args=args, protocol_id="p", schedule_version="s", phase="development",
        run_id="r", seed=11, git_commit="a" * 40, split_metadata={"test": {"sealed_unread": True}},
        output_dir="out", resume_parent="parent", resume_checkpoint_sha256="b" * 64,
        pretrained_sha256="c" * 64, environment={"python": "3.11"}, val_batch_size=1,
    )
    for key in ("protocol", "identity", "data", "schedule", "optimization", "evaluation", "output", "resume", "environment"):
        assert key in metadata
    assert metadata["identity"]["dirty"] is False
    assert metadata["data"]["splits"]["test"]["sealed_unread"] is True
    assert metadata["resume"] == {"parent_run_id": "parent", "checkpoint_sha256": "b" * 64}


def test_preflight_rejects_missing_offline_swanlab_and_file_output_collision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_path = _write_protocol(tmp_path)
    raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw["swanlab"]["mode"] = "offline"
    target = Path(raw["output_root"]) / raw["protocol_id"] / raw["phase"] / "seed-11"
    target.parent.mkdir(parents=True)
    target.write_text("not a directory", encoding="utf-8")
    protocol_path.write_text(json.dumps(raw), encoding="utf-8")

    original_find_spec = __import__("importlib.util", fromlist=["find_spec"]).find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "swanlab":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr("tools.preflight_train.importlib.util.find_spec", fake_find_spec)
    report = audit_protocol(load_protocol(protocol_path), repo_root=tmp_path, check_git=False)
    codes = {item["code"] for item in report.errors}
    assert {"swanlab_package_missing", "output_collision"} <= codes


def test_protocol_rejects_nonhexadecimal_identity_hashes(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path)
    raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw["splits"]["train_dev"]["sha256"] = "z" * 64
    protocol_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProtocolError, match="hexadecimal"):
        load_protocol(protocol_path)


def test_single_seed_launch_error_still_writes_structured_evidence(tmp_path: Path) -> None:
    protocol = _write_protocol(tmp_path)
    missing_python = tmp_path / "missing python #"
    status = run_seed_main([
        "--protocol-manifest", str(protocol), "--seed", "11", "--direct",
        "--python", str(missing_python), "--train-program", "missing trainer #.py",
    ])
    assert status != 0
    run_dir = tmp_path / "output root #" / "museg-dev-unit-v1" / "development" / "seed-11"
    exit_record = json.loads((run_dir / "train.exit_code").read_text(encoding="utf-8"))
    run_record = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert exit_record["exit_code"] == status
    assert run_record["exit_code"] == status
    assert run_record["launch_error"]
    assert (run_dir / "launcher.log").is_file()


def test_orchestrator_forwards_resume_to_only_the_selected_seed(tmp_path: Path) -> None:
    protocol = _write_protocol(tmp_path)
    checkpoint = tmp_path / "resume checkpoint #.pth"
    checkpoint.write_bytes(b"resume")
    calls = tmp_path / "resume-calls.jsonl"
    launcher = tmp_path / "resume fake launcher #.py"
    launcher.write_text(
        "import argparse,json\n"
        "p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--resume');"
        "p.add_argument('--resume-parent-run-id');p.add_argument('--resume-checkpoint-sha256');"
        "a,x=p.parse_known_args()\n"
        f"f=open({str(calls)!r},'a',encoding='utf-8');f.write(json.dumps(vars(a))+'\\n');f.close()\n",
        encoding="utf-8",
    )
    assert orchestrate_main([
        "--protocol-manifest", str(protocol), "--seeds", "11", "22", "33",
        "--seed-launcher", str(launcher), "--python", sys.executable, "--dry-run",
        "--resume-seed", "22", "--resume", str(checkpoint),
        "--resume-parent-run-id", "parent-22", "--resume-checkpoint-sha256", _sha(checkpoint),
    ]) == 0
    records = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert [item["seed"] for item in records] == [11, 22, 33]
    assert records[0]["resume"] is None and records[2]["resume"] is None
    assert records[1]["resume"] == str(checkpoint.resolve())
    assert records[1]["resume_parent_run_id"] == "parent-22"
    assert records[1]["resume_checkpoint_sha256"] == _sha(checkpoint)


def _write_run(protocol, seed: int, *, exit_code: int = 0) -> None:
    run = protocol.seed_output_dir(seed)
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "checkpoint" / "latest.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{seed}".encode("utf-8"))
    checkpoint_sha256 = _sha(checkpoint)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "museg-run-manifest-v1",
                "protocol_id": protocol.protocol_id,
                "protocol_manifest_sha256": protocol.manifest_sha256,
                "phase": protocol.phase,
                "seed": seed,
                "run_id": f"run-{seed}",
                "exit_code": exit_code,
            }
        ),
        encoding="utf-8",
    )
    (run / "training_result.json").write_text(
        json.dumps(
            {
                "schema_version": "museg-training-result-v1",
                "protocol_id": protocol.protocol_id,
                "protocol_manifest_sha256": protocol.manifest_sha256,
                "phase": protocol.phase,
                "seed": seed,
                "run_id": f"run-{seed}",
                "best_val_miou": 0.4 + seed / 1000,
                "best_val_epoch": 5,
                "final_epoch": 20,
                "duration_seconds": 1.0,
                "exit_code": exit_code,
                "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
                "official_test_included": False,
            }
        ),
        encoding="utf-8",
    )


def test_summarizer_success_and_fail_fast_cases(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path)
    protocol = load_protocol(protocol_path)
    root = Path(protocol.run_root)
    for seed in protocol.seeds:
        _write_run(protocol, seed)
    output = tmp_path / "summary #.json"
    assert summarize_main(["--protocol-manifest", str(protocol_path), "--output", str(output)]) == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert [item["seed"] for item in summary["runs"]] == [11, 22, 33]
    assert summary["official_test_included"] is False

    (root / "seed-33" / "training_result.json").unlink()
    assert summarize_main(["--protocol-manifest", str(protocol_path), "--output", str(output)]) != 0
    (root / "seed-33" / "training_result.json").write_text("{broken", encoding="utf-8")
    assert summarize_main(["--protocol-manifest", str(protocol_path), "--output", str(output)]) != 0
    _write_run(protocol, 33, exit_code=9)
    assert summarize_main(["--protocol-manifest", str(protocol_path), "--output", str(output)]) != 0

    duplicate_dir = root / "seed-44"
    duplicate_dir.mkdir()
    (duplicate_dir / "run_manifest.json").write_text(
        (root / "seed-11" / "run_manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert summarize_main(["--protocol-manifest", str(protocol_path), "--output", str(output)]) != 0

    duplicate = json.loads(protocol_path.read_text(encoding="utf-8"))
    duplicate["seeds"] = [11, 11, 33]
    protocol_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ProtocolError, match="unique"):
        load_protocol(protocol_path)


def test_single_seed_rejects_missing_or_mismatched_training_result(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path)
    missing_result_trainer = tmp_path / "no result trainer #.py"
    missing_result_trainer.write_text("raise SystemExit(0)\n", encoding="utf-8")
    assert run_seed_main([
        "--protocol-manifest", str(protocol_path), "--seed", "11", "--direct",
        "--train-program", str(missing_result_trainer), "--python", sys.executable,
    ]) != 0
    run_dir = tmp_path / "output root #" / "museg-dev-unit-v1" / "development" / "seed-11"
    run_record = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_record["exit_code"] != 0
    assert "training_result.json" in run_record["evidence_error"]

    second_protocol_path = _write_protocol(tmp_path / "mismatch")
    mismatch_trainer = _write_fake_trainer(tmp_path / "mismatch")
    text = mismatch_trainer.read_text(encoding="utf-8").replace(
        "'protocol_id':a.protocol_id", "'protocol_id':'wrong-protocol'"
    )
    mismatch_trainer.write_text(text, encoding="utf-8")
    assert run_seed_main([
        "--protocol-manifest", str(second_protocol_path), "--seed", "11", "--direct",
        "--train-program", str(mismatch_trainer), "--python", sys.executable,
    ]) != 0


def test_orchestrator_records_launcher_and_summary_failures(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path / "launch")
    missing_python = tmp_path / "missing orchestrator python #"
    status = orchestrate_main([
        "--protocol-manifest", str(protocol_path), "--python", str(missing_python),
    ])
    assert status != 0
    report_path = (
        tmp_path / "launch" / "output root #" / "museg-dev-unit-v1" /
        "development" / "orchestrator.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["failed_seed"] == 11
    assert report["runs"][0]["launch_error"]

    summary_protocol = _write_protocol(tmp_path / "summary")
    launcher = tmp_path / "successful empty launcher #.py"
    launcher.write_text("raise SystemExit(0)\n", encoding="utf-8")
    status = orchestrate_main([
        "--protocol-manifest", str(summary_protocol), "--seed-launcher", str(launcher),
        "--python", sys.executable,
    ])
    assert status != 0
    summary_report_path = (
        tmp_path / "summary" / "output root #" / "museg-dev-unit-v1" /
        "development" / "orchestrator.json"
    )
    summary_report = json.loads(summary_report_path.read_text(encoding="utf-8"))
    assert summary_report["exit_code"] != 0
    assert summary_report["summary_error"]


def test_summarizer_rejects_cross_protocol_artifacts(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path)
    protocol = load_protocol(protocol_path)
    for seed in protocol.seeds:
        _write_run(protocol, seed)
    run_path = protocol.seed_output_dir(22) / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update({
        "protocol_id": "other-protocol",
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "phase": protocol.phase,
    })
    run_path.write_text(json.dumps(run), encoding="utf-8")
    assert summarize_main([
        "--protocol-manifest", str(protocol_path), "--output", str(tmp_path / "summary.json")
    ]) != 0
