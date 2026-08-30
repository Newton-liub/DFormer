from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.control_museg_lifecycle_test import run_lifecycle_controller
from tools.run_museg_lifecycle_test import generate_lifecycle_evidence


def test_lifecycle_controller_verifies_evidence_and_stops_in_finally(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(list(command))
        if any("run_museg_lifecycle_test" in item for item in command):
            generate_lifecycle_evidence(tmp_path / "evidence", run_id="unit-success")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[-2:] == ["instance", "show"]:
            raise AssertionError("status command should include --status")
        if "show" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"ok": True, "data": {"status": "Stopped"}}), ""
            )
        return subprocess.CompletedProcess(command, 0, json.dumps({"ok": True}), "")

    result = run_lifecycle_controller(
        tmp_path / "evidence",
        run_id="unit-success",
        instance_id="instance-1",
        simulated_exit_code=0,
        schedule_at="+15m",
        compshare="fake-compshare",
        runner=runner,
    )

    assert result["evidence_verified"] is True
    assert result["instance_state"] == "Stopped"
    assert result["status"] == "passed"
    assert (tmp_path / "evidence" / "controller-result.json").is_file()
    assert any("schedule" in call and "set" in call for call in calls)
    assert any("schedule" in call and "show" in call for call in calls)
    assert any("instance" in call and "stop" in call for call in calls)
    assert any("instance" in call and "show" in call for call in calls)


def test_lifecycle_controller_stops_after_simulated_workload_failure(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(list(command))
        if any("run_museg_lifecycle_test" in item for item in command):
            return subprocess.CompletedProcess(command, 7, "", "simulated failure")
        if "show" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"ok": True, "data": {"status": "Stopped"}}), ""
            )
        return subprocess.CompletedProcess(command, 0, json.dumps({"ok": True}), "")

    with pytest.raises(RuntimeError, match="workload failed"):
        run_lifecycle_controller(
            tmp_path / "evidence",
            run_id="unit-failure",
            instance_id="instance-2",
            simulated_exit_code=7,
            schedule_at="+15m",
            compshare="fake-compshare",
            runner=runner,
        )

    assert any("schedule" in call and "set" in call for call in calls)
    assert any("schedule" in call and "show" in call for call in calls)
    assert any("instance" in call and "stop" in call for call in calls)
    assert any("instance" in call and "show" in call for call in calls)
    failed_result = json.loads(
        (tmp_path / "evidence" / "controller-result.json").read_text(encoding="utf-8")
    )
    assert failed_result["status"] == "failed"
