from __future__ import annotations

import datetime
import sys
from types import SimpleNamespace

import pytest

from tools.preflight_train import Preflight, check_cuda, check_output, validate_gpu_target
from utils.experiment_tracker import (
    ExperimentTracker,
    build_run_name,
    gpu_safety_violation,
)


def test_allow_no_gpu_converts_cuda_failure_to_warning(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.1.2",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    result = Preflight()

    check_cuda(result, allow_no_gpu=True)

    assert result.errors == []
    assert len(result.warnings) == 1
    assert "CUDA GPU unavailable" in result.warnings[0]


def test_tracker_logs_explicit_step_and_finishes_once(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_swanlab = SimpleNamespace(
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        init=lambda **kwargs: calls.append(("init", kwargs)),
        log=lambda metrics, step: calls.append(("log", (metrics, step))),
        finish=lambda: calls.append(("finish", None)),
    )
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)
    tracker = ExperimentTracker()

    tracker.start(
        mode="online",
        project="DFormer-liu",
        workspace="Newton_liub",
        name="MUSeg-DFormerv2_S-test",
        log_dir="runs/test",
    )
    tracker.log({"train/loss": 0.5}, step=7)
    tracker.finish()
    tracker.finish()

    assert calls[0][0] == "init"
    init_kwargs = calls[0][1]
    assert init_kwargs["experiment_name"] == "MUSeg-DFormerv2_S-test"
    assert init_kwargs["logdir"] == "runs/test"
    assert init_kwargs["settings"].interactive is False
    assert "name" not in init_kwargs
    assert "log_dir" not in init_kwargs
    assert calls[1] == ("log", ({"train/loss": 0.5}, 7))
    assert [name for name, _ in calls].count("finish") == 1


def test_online_tracker_initialization_failure_is_fail_fast(monkeypatch) -> None:
    def fail_init(**kwargs) -> None:
        raise RuntimeError("authentication failed")

    monkeypatch.setitem(
        sys.modules,
        "swanlab",
        SimpleNamespace(Settings=lambda **kwargs: SimpleNamespace(**kwargs), init=fail_init),
    )

    with pytest.raises(RuntimeError, match="authentication failed"):
        ExperimentTracker().start(mode="online")


def test_non_primary_tracker_never_imports_swanlab(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "swanlab", raising=False)
    tracker = ExperimentTracker()

    tracker.start(mode="online", is_primary=False)
    tracker.log({"ignored": 1}, step=1)
    tracker.finish()

    assert not tracker.enabled


def test_gpu_safety_thresholds_are_disabled_by_default() -> None:
    assert gpu_safety_violation(1, 100) is None


def test_gpu_safety_threshold_reports_absolute_then_ratio_limit() -> None:
    one_gib = 1024**3
    assert "below required 2.00 GiB" in gpu_safety_violation(
        one_gib,
        24 * one_gib,
        min_free_gib=2.0,
        min_free_ratio=0.1,
    )
    assert "below required 0.100" in gpu_safety_violation(
        2 * one_gib,
        24 * one_gib,
        min_free_ratio=0.1,
    )
    assert gpu_safety_violation(3 * one_gib, 24 * one_gib, min_free_gib=2.0, min_free_ratio=0.1) is None


def test_automatic_run_name_contains_dataset_backbone_utc_and_commit() -> None:
    now = datetime.datetime(2026, 8, 24, 12, 34, 56, tzinfo=datetime.timezone.utc)
    assert build_run_name("MUSeg", "DFormerv2_S", now=now, git_commit="abc1234") == (
        "MUSeg-DFormerv2_S-20260824T123456Z-abc1234"
    )


def test_explicit_run_name_takes_priority() -> None:
    assert build_run_name("MUSeg", "DFormerv2_S", explicit_name="chosen") == "chosen"


def test_output_preflight_does_not_create_target(tmp_path) -> None:
    target = tmp_path / "new" / "nested" / "output"
    result = Preflight()

    check_output(SimpleNamespace(log_dir=str(target)), result)

    assert result.errors == []
    assert not target.exists()


def test_gpu_target_requires_4090_and_sufficient_memory() -> None:
    gib = 1024**3
    assert validate_gpu_target("NVIDIA GeForce RTX 4090", 24 * gib) is None
    assert "not an RTX 4090" in validate_gpu_target("NVIDIA GeForce RTX 4080", 24 * gib)
    assert "expected at least 20 GiB" in validate_gpu_target("NVIDIA GeForce RTX 4090", 16 * gib)
