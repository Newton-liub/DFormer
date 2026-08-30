from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import tools.materialize_museg_protocol as materializer
import tools.museg_protocol as museg_protocol
from tools.museg_protocol import ProtocolError, load_protocol
from tools.preflight_train import audit_protocol


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_authority(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "authority"
    root.mkdir()
    entries = {
        "train-dev.txt": ["01-01-01-0001-a"], "val-dev.txt": ["02-01-01-0002-b"],
        "official-test.txt": ["03-01-01-0003-c"],
        "official-train.txt": ["01-01-01-0001-a", "02-01-01-0002-b"],
    }
    records = {}
    for name, lines in entries.items():
        path = root / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        records[name] = {"sha256": _sha(path), "samples": len(lines), "groups": len(lines)}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "museg-dev-split-manifest-v1", "protocol_id": "MUSEG-DEV-SPLIT-PROTOCOL-1",
        "candidate_status": "frozen", "user_gate_a": {"status": "approved"},
        "outputs": {name: {**records[name], "byte_policy": "test"} for name in entries if name != "official-train.txt"},
        "official": {"train": {"logical_name": "train.txt", **records["official-train.txt"]}},
    }), encoding="utf-8")
    audit_path = root / "audit-report.json"
    manifest_sha = _sha(manifest_path)
    audit_path.write_text(json.dumps({
        "schema_version": "museg-dev-split-audit-v1", "pass": True, "manifest_sha256": manifest_sha,
        "details": {"manifest_sha256": manifest_sha, "counts": {
            "train_dev": 1, "val_dev": 1, "official_train": 2, "official_test": 1,
        }, "groups": {"train_dev": 1, "val_dev": 1, "official_train": 2, "official_test": 1}},
    }), encoding="utf-8")
    for module in (museg_protocol, materializer):
        monkeypatch.setattr(module, "FROZEN_SPLIT_ROOT", root, raising=False)
        monkeypatch.setattr(module, "FROZEN_MANIFEST_PATH", manifest_path)
        monkeypatch.setattr(module, "FROZEN_AUDIT_PATH", audit_path)
        monkeypatch.setattr(module, "FROZEN_MANIFEST_SHA256", manifest_sha)
        monkeypatch.setattr(module, "FROZEN_AUDIT_SHA256", _sha(audit_path))
    return root


def _clean_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=path, check=True, capture_output=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_tracked_frozen_split_bytes_match_manifest() -> None:
    root = museg_protocol.FROZEN_SPLIT_ROOT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["outputs"].items():
        assert _sha(root / name) == record["sha256"]


def test_tracked_template_is_not_a_runnable_protocol() -> None:
    with pytest.raises(ProtocolError, match="full 40-character"):
        load_protocol(materializer.TEMPLATE_PATH)


def test_materializer_writes_valid_immutable_protocol(tmp_path: Path, monkeypatch) -> None:
    authority = _patch_authority(tmp_path, monkeypatch)
    repo = tmp_path / "clean repo"
    commit = _clean_repo(repo)
    weight = tmp_path / "weight #.pth"
    weight.write_bytes(b"pretrained")
    target = tmp_path / "generated" / "qualification.json"
    manifest, digest = materializer.materialize(
        output=target,
        output_root=tmp_path / "run outputs #",
        official_train=authority / "official-train.txt",
        pretrained=weight,
        batch_size=4,
        swanlab_mode="online",
        swanlab_project="DFormer-liu",
        swanlab_workspace="Newton_liub",
        repo_root=repo,
    )
    protocol = load_protocol(manifest)
    assert digest == _sha(manifest)
    assert protocol.git["required_commit"] == commit
    assert protocol.training["batch_size"] == 4
    assert protocol.schema_version == "museg-training-protocol-v3"
    assert protocol.input_contract["channel_order"] == "BGR"
    assert protocol.input_contract["normalization"]["identity"] == "imagenet-rgb-statistics-in-array-order-v1"
    assert protocol.authority_identity()["manifest_sha256"] == _sha(authority / "manifest.json")
    report = audit_protocol(protocol, repo_root=repo, check_git=False)
    assert not any(error["code"] == "split_authority_mismatch" for error in report.errors)
    assert "API_KEY" not in manifest.read_text(encoding="utf-8")
    with pytest.raises(ProtocolError, match="already exists"):
        materializer.materialize(
            output=target, output_root=tmp_path / "other", official_train=authority / "official-train.txt",
            pretrained=weight, batch_size=4, swanlab_mode="online", swanlab_project="DFormer-liu",
            swanlab_workspace="Newton_liub", repo_root=repo,
        )


def test_quick_b0_template_materializes_protocol_v3_extensions(tmp_path: Path, monkeypatch) -> None:
    authority = _patch_authority(tmp_path, monkeypatch)
    repo = tmp_path / "clean repo"
    _clean_repo(repo)
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"pretrained")
    target = tmp_path / "generated" / "quick-b0.json"
    template = Path(__file__).parents[1] / "protocols" / "museg-dformerv2-s-rgb-quick-b0-v1.template.json"

    manifest, _ = materializer.materialize(
        output=target,
        output_root=tmp_path / "outputs",
        official_train=authority / "official-train.txt",
        pretrained=weight,
        batch_size=10,
        swanlab_mode="offline",
        swanlab_project="project",
        swanlab_workspace="workspace",
        template_path=template,
        repo_root=repo,
    )
    protocol = load_protocol(manifest)

    assert protocol.run_kind == "standard"
    assert protocol.simulation is False
    assert protocol.seeds == (772961337,)
    assert protocol.checkpoint_policy["top_k"] == 3
    assert protocol.checkpoint_policy["candidate_manifest"] == "checkpoint-candidates.json"
    assert protocol.optimizer_telemetry["schema_version"] == "museg-optimizer-telemetry-v1"
    assert protocol.input_contract["channel_order"] == "RGB"


def test_materializer_rejects_dirty_repository_before_writing(tmp_path: Path, monkeypatch) -> None:
    authority = _patch_authority(tmp_path, monkeypatch)
    repo = tmp_path / "dirty repo"
    _clean_repo(repo)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"pretrained")
    target = tmp_path / "generated" / "qualification.json"
    with pytest.raises(ProtocolError, match="clean Git"):
        materializer.materialize(
            output=target, output_root=tmp_path / "outputs", official_train=authority / "official-train.txt",
            pretrained=weight, batch_size=4, swanlab_mode="offline", swanlab_project="project",
            swanlab_workspace="workspace", repo_root=repo,
        )
    assert not target.exists()


def test_v3_protocol_requires_explicit_input_contract(tmp_path: Path, monkeypatch) -> None:
    authority = _patch_authority(tmp_path, monkeypatch)
    repo = tmp_path / "clean repo"
    _clean_repo(repo)
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"pretrained")
    target = tmp_path / "generated" / "qualification.json"
    materializer.materialize(
        output=target,
        output_root=tmp_path / "outputs",
        official_train=authority / "official-train.txt",
        pretrained=weight,
        batch_size=4,
        swanlab_mode="offline",
        swanlab_project="project",
        swanlab_workspace="workspace",
        repo_root=repo,
    )
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw.pop("input_contract")
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProtocolError, match="input_contract"):
        load_protocol(target)
