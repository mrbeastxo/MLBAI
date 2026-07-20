from pathlib import Path

import pytest

from backend.automation import macos_scheduler


def test_launch_agent_config_runs_daily_workflow_at_requested_time() -> None:
    config = macos_scheduler.launch_agent_config(6, 30)
    assert config["Label"] == "com.mlbai.daily"
    assert config["StartCalendarInterval"] == {"Hour": 6, "Minute": 30}
    assert config["ProgramArguments"][1:] == [
        "-m",
        "backend.automation.daily_run",
    ]
    assert config["WorkingDirectory"].endswith("MLBAI")
    assert config["RunAtLoad"] is False


def test_launch_agent_config_defaults_to_nine_am_and_pm() -> None:
    config = macos_scheduler.launch_agent_config()

    assert config["StartCalendarInterval"] == [
        {"Hour": 9, "Minute": 0},
        {"Hour": 21, "Minute": 0},
    ]


@pytest.mark.parametrize("hour,minute", [(-1, 0), (24, 0), (6, -1), (6, 60)])
def test_invalid_schedule_times_are_rejected(hour: int, minute: int) -> None:
    with pytest.raises(ValueError):
        macos_scheduler.launch_agent_config(hour, minute)


def test_empty_schedule_is_rejected() -> None:
    with pytest.raises(ValueError):
        macos_scheduler.launch_agent_config([], 0)


def test_install_writes_plist_and_loads_agent(tmp_path, monkeypatch) -> None:
    calls = []
    fake_python = tmp_path / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("")
    monkeypatch.setattr(macos_scheduler, "PYTHON_PATH", fake_python)
    monkeypatch.setattr(macos_scheduler, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(macos_scheduler, "STDOUT_PATH", tmp_path / "logs" / "out.log")
    monkeypatch.setattr(macos_scheduler, "STDERR_PATH", tmp_path / "logs" / "error.log")
    monkeypatch.setattr(
        macos_scheduler,
        "launchctl",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    plist = tmp_path / "LaunchAgents" / "com.mlbai.daily.plist"
    macos_scheduler.install(plist, 7, 15)
    assert plist.is_file()
    assert calls[0][0][0] == "bootout"
    assert calls[1][0][0] == "bootstrap"


def test_uninstall_is_safe_when_plist_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(macos_scheduler, "launchctl", lambda *args, **kwargs: None)
    assert macos_scheduler.uninstall(tmp_path / "missing.plist") is False
