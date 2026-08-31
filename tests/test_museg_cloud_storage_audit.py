from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.audit_museg_cloud_storage import StorageAuditError, audit_storage


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    storage = tmp_path / "data-disk"
    protected = storage / "DFormer"
    obsolete = storage / "obsolete-test"
    protected.mkdir(parents=True)
    obsolete.mkdir()
    (obsolete / "result.bin").write_bytes(b"obsolete-result")
    return storage, protected, obsolete


def test_audit_reports_exact_candidate_without_mutation(tmp_path: Path) -> None:
    storage, protected, obsolete = _layout(tmp_path)
    before = sorted(str(path.relative_to(storage)) for path in storage.rglob("*"))

    report = audit_storage(
        storage_root=storage,
        candidates=[obsolete],
        protected_paths=[protected],
        checkpoint_size_bytes=100,
        checkpoint_candidates=9,
    )

    after = sorted(str(path.relative_to(storage)) for path in storage.rglob("*"))
    assert after == before
    assert report["destructive_actions_performed"] is False
    assert report["candidates"][0]["path"] == str(obsolete.resolve())
    assert report["summary"]["reclaimable_bytes"] == len(b"obsolete-result")
    assert report["summary"]["checkpoint_storage_budget_bytes"] == 900


def test_audit_rejects_candidate_outside_storage_root(tmp_path: Path) -> None:
    storage, protected, _ = _layout(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(StorageAuditError, match="escapes storage_root"):
        audit_storage(
            storage_root=storage,
            candidates=[outside],
            protected_paths=[protected],
        )


def test_audit_rejects_protected_path_overlap_in_either_direction(tmp_path: Path) -> None:
    storage, protected, _ = _layout(tmp_path)
    active_run = protected / "outputs" / "active-run"
    active_run.mkdir(parents=True)

    with pytest.raises(StorageAuditError, match="overlaps protected path"):
        audit_storage(
            storage_root=storage,
            candidates=[protected],
            protected_paths=[active_run],
        )
    with pytest.raises(StorageAuditError, match="overlaps protected path"):
        audit_storage(
            storage_root=storage,
            candidates=[active_run],
            protected_paths=[protected],
        )


def test_audit_rejects_storage_root_as_candidate(tmp_path: Path) -> None:
    storage, protected, _ = _layout(tmp_path)

    with pytest.raises(StorageAuditError, match="cannot equal storage_root"):
        audit_storage(
            storage_root=storage,
            candidates=[storage],
            protected_paths=[protected],
        )


def test_audit_rejects_symlink_escape(tmp_path: Path) -> None:
    storage, protected, _ = _layout(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = storage / "outside-link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(StorageAuditError, match="escapes storage_root"):
        audit_storage(
            storage_root=storage,
            candidates=[link],
            protected_paths=[protected],
        )
