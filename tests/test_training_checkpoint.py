from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.dataloader.RGBXDataset import RGBXDataset
from utils.lr_policy import WarmUpPolyLR
from utils.training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointProtocol,
    TrainingSources,
    build_split_metadata,
    atomic_save_checkpoint,
    create_training_checkpoint,
    inspect_training_checkpoint,
    compare_checkpoint_inspections,
    inspect_checkpoint_directory,
    load_training_checkpoint,
    optimizer_step_was_applied,
    prepare_output_directory,
    resolve_training_sources,
    restore_training_state,
    select_best_metric,
    should_evaluate,
    should_save_epoch,
)


def _protocol(**overrides) -> CheckpointProtocol:
    protocol = CheckpointProtocol(
        phase="development",
        run_id="unit-test-run",
        git_commit="0123456789abcdef",
        seed=7,
        model_name="ToyModel",
        optimizer_name="SGD",
        total_epochs=3,
        iterations_per_epoch=1,
        warmup_steps=1,
        poly_power=0.9,
        base_lr=0.1,
        split_metadata={
            "train": {"path": "train.txt", "samples": 2, "sha256": "a" * 64},
            "val": {"path": "val.txt", "samples": 1, "sha256": "b" * 64},
            "test": None,
        },
        config_summary={"backbone": "ToyModel", "optimizer": "SGD", "amp": True},
    )
    return replace(protocol, **overrides)


@pytest.mark.parametrize(
    ("total_epochs", "start", "interval", "expected"),
    [
        (20, 5, 5, [5, 10, 15, 20]),
        (100, 10, 10, list(range(10, 101, 10))),
        (500, 5, 10, list(range(5, 496, 10)) + [500]),
        (3, 1, 1, [1, 2, 3]),
    ],
)
def test_validation_schedule_is_explicit_and_forces_final_epoch(
    total_epochs: int,
    start: int,
    interval: int,
    expected: list[int],
) -> None:
    actual = [
        epoch
        for epoch in range(1, total_epochs + 1)
        if should_evaluate(epoch, total_epochs, start, interval)
    ]
    assert actual == expected


def test_save_schedule_uses_interval_and_forces_final_epoch() -> None:
    assert [epoch for epoch in range(1, 21) if should_save_epoch(epoch, 20, 6)] == [6, 12, 18, 20]
    assert should_save_epoch(1, 1, 100)


def test_optimizer_step_count_only_advances_after_real_update() -> None:
    assert optimizer_step_was_applied(1024.0, 1024.0)
    assert optimizer_step_was_applied(1024.0, 2048.0)
    assert not optimizer_step_was_applied(1024.0, 512.0)


@pytest.mark.parametrize("epochs", [20, 100, 500])
def test_epoch_override_recomputes_warmup_and_poly_lr_boundaries(epochs: int) -> None:
    iterations_per_epoch = 5
    total_iterations = epochs * iterations_per_epoch
    warmup_iterations = 2 * iterations_per_epoch
    policy = WarmUpPolyLR(0.1, 0.9, total_iterations, warmup_iterations)

    assert policy.total_iters == total_iterations
    assert policy.warmup_steps == warmup_iterations
    assert policy.get_lr(0) == pytest.approx(0.0)
    middle = total_iterations // 2
    assert policy.get_lr(middle) == pytest.approx(0.1 * ((1 - middle / total_iterations) ** 0.9))
    final_training_iteration = total_iterations - 1
    assert policy.get_lr(final_training_iteration) == pytest.approx(
        0.1 * ((1 / total_iterations) ** 0.9)
    )


def test_best_metric_uses_strict_improvement_and_rejects_non_finite() -> None:
    assert select_best_metric(None, None, 0.4, 3) == (True, 0.4, 3)
    assert select_best_metric(0.4, 3, 0.5, 4) == (True, 0.5, 4)
    assert select_best_metric(0.5, 4, 0.5, 5) == (False, 0.5, 4)
    assert select_best_metric(0.5, 4, 0.3, 5) == (False, 0.5, 4)
    with pytest.raises(FloatingPointError, match="non-finite"):
        select_best_metric(0.5, 4, float("nan"), 5)


def _run_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    optimizer.param_groups[0]["lr"] = 0.1 * (1.0 - step / 3.0)
    x = torch.tensor([[1.0]])
    target = torch.tensor([[0.25]])
    optimizer.zero_grad()
    loss = torch.square(model(x) - target).mean()
    loss.backward()
    optimizer.step()


def test_checkpoint_roundtrip_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    torch.manual_seed(7)
    uninterrupted = torch.nn.Linear(1, 1)
    resumed_source = torch.nn.Linear(1, 1)
    resumed_source.load_state_dict(uninterrupted.state_dict())
    optimizer_a = torch.optim.SGD(uninterrupted.parameters(), lr=0.1, momentum=0.9)
    optimizer_b = torch.optim.SGD(resumed_source.parameters(), lr=0.1, momentum=0.9)

    for step in range(3):
        _run_step(uninterrupted, optimizer_a, step)
    _run_step(resumed_source, optimizer_b, 0)

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint = create_training_checkpoint(
        model=resumed_source,
        optimizer=optimizer_b,
        scaler=scaler,
        completed_epoch=1,
        global_optimizer_step=1,
        best_val_miou=0.42,
        best_val_epoch=1,
        protocol=_protocol(),
    )
    path = tmp_path / "checkpoint" / "latest.pth"
    atomic_save_checkpoint(checkpoint, path)

    restored_model = torch.nn.Linear(1, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
    restored_scaler = torch.cuda.amp.GradScaler(enabled=False)
    loaded = load_training_checkpoint(path, expected_protocol=_protocol())
    resume = restore_training_state(
        loaded,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
        restore_rng=False,
    )
    for step in range(resume.global_optimizer_step, 3):
        _run_step(restored_model, restored_optimizer, step)

    assert loaded["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert resume.next_epoch == 2
    assert resume.global_optimizer_step == 1
    assert resume.best_val_miou == pytest.approx(0.42)
    assert resume.best_val_epoch == 1
    assert restored_optimizer.param_groups[0]["lr"] == optimizer_a.param_groups[0]["lr"]
    for actual, expected in zip(restored_model.parameters(), uninterrupted.parameters()):
        torch.testing.assert_close(actual, expected)


def test_rng_state_is_restored_to_the_next_draw(tmp_path: Path) -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint = create_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_epoch=1,
        global_optimizer_step=1,
        best_val_miou=None,
        best_val_epoch=None,
        protocol=_protocol(),
    )
    path = tmp_path / "latest.pth"
    atomic_save_checkpoint(checkpoint, path)
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.random()
    np.random.random()
    torch.rand(())
    restore_training_state(
        load_training_checkpoint(path, expected_protocol=_protocol()),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        restore_rng=True,
    )
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "incompatible_protocol",
    [
        _protocol(total_epochs=4),
        _protocol(optimizer_name="AdamW"),
        _protocol(base_lr=0.2),
        _protocol(split_metadata={"train": {"path": "train.txt", "samples": 2, "sha256": "f" * 64}}),
    ],
)
def test_checkpoint_rejects_incompatible_protocol(
    tmp_path: Path,
    incompatible_protocol: CheckpointProtocol,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    path = tmp_path / "latest.pth"
    atomic_save_checkpoint(
        create_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            completed_epoch=1,
            global_optimizer_step=1,
            best_val_miou=None,
            best_val_epoch=None,
            protocol=_protocol(),
        ),
        path,
    )

    with pytest.raises(CheckpointCompatibilityError):
        load_training_checkpoint(path, expected_protocol=incompatible_protocol)


def test_checkpoint_rejects_invalid_best_and_protocol_hash(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint = create_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_epoch=1,
        global_optimizer_step=1,
        best_val_miou=None,
        best_val_epoch=None,
        protocol=_protocol(),
    )
    checkpoint["best_val_miou"] = float("nan")
    checkpoint["best_val_epoch"] = 1
    path = tmp_path / "invalid-best.pth"
    atomic_save_checkpoint(checkpoint, path)
    with pytest.raises(CheckpointCorruptionError, match="best validation"):
        load_training_checkpoint(path, expected_protocol=_protocol())

    checkpoint = create_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_epoch=1,
        global_optimizer_step=1,
        best_val_miou=None,
        best_val_epoch=None,
        protocol=_protocol(),
    )
    checkpoint["protocol"]["config_summary"]["optimizer"] = "tampered"
    path = tmp_path / "invalid-protocol-hash.pth"
    atomic_save_checkpoint(checkpoint, path)
    with pytest.raises(CheckpointCorruptionError, match="config SHA-256"):
        load_training_checkpoint(path, expected_protocol=_protocol())


def test_corrupt_checkpoint_fails_without_replacing_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "latest.pth"
    path.write_bytes(b"truncated")
    with pytest.raises(CheckpointCorruptionError, match="cannot load checkpoint"):
        load_training_checkpoint(path, expected_protocol=_protocol())
    assert path.read_bytes() == b"truncated"
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_directory_rejects_latest_older_than_periodic(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    for name, completed_epoch in (("latest.pth", 1), ("epoch-2.pth", 2)):
        atomic_save_checkpoint(
            create_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                completed_epoch=completed_epoch,
                global_optimizer_step=completed_epoch,
                best_val_miou=None,
                best_val_epoch=None,
                protocol=_protocol(),
            ),
            tmp_path / name,
        )
    with pytest.raises(CheckpointCompatibilityError, match="older"):
        inspect_checkpoint_directory(tmp_path, expected_protocol=_protocol())


def test_development_reads_val_source_and_never_opens_test(tmp_path: Path, monkeypatch) -> None:
    train_source = tmp_path / "train.txt"
    val_source = tmp_path / "val.txt"
    test_source = tmp_path / "test.txt"
    train_source.write_text("train-item\n", encoding="utf-8")
    val_source.write_text("val-item\n", encoding="utf-8")
    test_source.write_text("test-item\n", encoding="utf-8")
    opened: list[str] = []
    real_open = open

    def tracked_open(path, *args, **kwargs):
        opened.append(str(Path(path).resolve()))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracked_open)
    dataset = RGBXDataset.__new__(RGBXDataset)
    dataset._train_source = str(train_source)
    dataset._val_source = str(val_source)
    assert dataset._get_file_names("val") == ["val-item"]
    assert str(val_source.resolve()) in opened
    assert str(test_source.resolve()) not in opened

    sources = resolve_training_sources(
        SimpleNamespace(
            experiment_phase="development",
            train_source=str(train_source),
            val_source=str(val_source),
            test_source=str(test_source),
        )
    )
    assert sources.val_source == str(val_source)
    assert sources.test_source == str(test_source)


def test_official_phase_discards_configured_validation_and_never_falls_back_to_test() -> None:
    sources = resolve_training_sources(
        SimpleNamespace(
            experiment_phase="official",
            train_source="official-train.txt",
            val_source="stale-val-dev.txt",
            test_source="sealed-test.txt",
        )
    )
    assert sources.val_source is None
    assert sources.test_source == "sealed-test.txt"


def test_qualification_requires_explicit_validation_source() -> None:
    with pytest.raises(ValueError, match="qualification phase requires"):
        resolve_training_sources(
            SimpleNamespace(experiment_phase="qualification", train_source="train.txt", test_source="sealed-test.txt")
        )


def test_training_source_must_not_alias_sealed_test() -> None:
    with pytest.raises(ValueError, match="training source must not"):
        resolve_training_sources(
            SimpleNamespace(
                experiment_phase="official",
                train_source="sealed-test.txt",
                test_source="sealed-test.txt",
            )
        )


def test_legacy_config_uses_eval_source_only_as_validation_compatibility() -> None:
    sources = resolve_training_sources(
        SimpleNamespace(train_source="train.txt", eval_source="legacy-val.txt")
    )
    assert sources.phase == "qualification"
    assert sources.val_source == "legacy-val.txt"
    assert sources.test_source is None


def test_split_metadata_uses_declared_sealed_test_without_opening_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    test = tmp_path / "test.txt"
    train.write_text("a\nb\n", encoding="utf-8")
    val.write_text("c\n", encoding="utf-8")
    test.write_text("sealed\n", encoding="utf-8")
    real_open = open

    def reject_test_open(path, *args, **kwargs):
        if Path(path).resolve() == test.resolve():
            raise AssertionError("training metadata must not open the sealed test split")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", reject_test_open)
    metadata = build_split_metadata(
        TrainingSources("development", str(train), str(val), str(test)),
        expected_sha256={"test": "c" * 64},
        expected_samples={"test": 1576},
        read_test_source=False,
    )
    assert metadata["train"]["samples"] == 2
    assert metadata["val"]["samples"] == 1
    assert metadata["test"] == {
        "path": str(test.resolve()),
        "samples": 1576,
        "sha256": "c" * 64,
        "sealed_unread": True,
    }


def test_output_directory_refuses_nonempty_even_when_resuming(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    resume = output / "checkpoint" / "latest.pth"
    resume.parent.mkdir()
    resume.write_bytes(b"placeholder")

    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_output_directory(output, resume_path=resume)

    resumed_output = tmp_path / "resumed-run"
    assert prepare_output_directory(resumed_output, resume_path=resume) == resumed_output.resolve()
    assert (output / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_checkpoint_inspection_hashes_logical_state_and_parent_identity(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    path = tmp_path / "latest.pth"
    atomic_save_checkpoint(
        create_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            completed_epoch=1,
            global_optimizer_step=1,
            best_val_miou=0.3,
            best_val_epoch=1,
            protocol=_protocol(),
        ),
        path,
    )
    parent_protocol = _protocol(run_id="parent-run")
    parent_path = tmp_path / "parent.pth"
    atomic_save_checkpoint(
        create_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            completed_epoch=1,
            global_optimizer_step=1,
            best_val_miou=0.3,
            best_val_epoch=1,
            protocol=parent_protocol,
        ),
        parent_path,
    )
    inspection = inspect_training_checkpoint(path, expected_protocol=_protocol())
    assert inspection["component_sha256"]["model"]
    assert compare_checkpoint_inspections(inspection, inspection) == []
    loaded = load_training_checkpoint(
        parent_path,
        expected_protocol=_protocol(run_id="child-run"),
        expected_checkpoint_run_id="parent-run",
    )
    assert loaded["protocol"]["run_id"] == "parent-run"
    with pytest.raises(CheckpointCompatibilityError, match="run_id"):
        load_training_checkpoint(
            parent_path,
            expected_protocol=_protocol(run_id="child-run"),
            expected_checkpoint_run_id="wrong-parent",
        )


def test_resume_equivalence_comparison_only_ignores_protocol_run_id() -> None:
    continuous = _protocol(run_id="continuous-run").to_dict()
    resumed = _protocol(run_id="resumed-child-run").to_dict()
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "completed_epoch": 3,
        "next_epoch": 4,
        "global_optimizer_step": 3,
        "best_val_miou": 0.5,
        "best_val_epoch": 3,
        "optimizer_lrs": [0.01],
        "protocol": continuous,
        "component_sha256": {"model": "a", "optimizer": "b", "amp_scaler": "c", "rng_state": "d"},
    }
    actual = {**expected, "protocol": resumed}

    assert compare_checkpoint_inspections(expected, actual) == ["protocol"]
    assert compare_checkpoint_inspections(expected, actual, allow_protocol_run_id_mismatch=True) == []

    resumed_with_other_change = {**resumed, "seed": 8}
    assert compare_checkpoint_inspections(
        expected,
        {**actual, "protocol": resumed_with_other_change},
        allow_protocol_run_id_mismatch=True,
    ) == ["protocol"]
