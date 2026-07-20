"""Install and manage the MLBAI daily workflow as a macOS LaunchAgent."""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

LABEL = "com.mlbai.daily"
DEFAULT_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
PYTHON_PATH = PROJECT_ROOT / ".venv" / "bin" / "python"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
STDOUT_PATH = LOG_DIR / "daily_run.log"
STDERR_PATH = LOG_DIR / "daily_run_error.log"


def validate_time(hour: int, minute: int) -> None:
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if not 0 <= minute <= 59:
        raise ValueError("minute must be between 0 and 59")


DEFAULT_HOURS = (9, 21)


def schedule_intervals(
    hours: int | Iterable[int] = DEFAULT_HOURS,
    minute: int = 0,
) -> list[dict[str, int]]:
    normalized_hours = [hours] if isinstance(hours, int) else list(hours)
    if not normalized_hours:
        raise ValueError("at least one schedule hour is required")
    for hour in normalized_hours:
        validate_time(hour, minute)
    return [{"Hour": hour, "Minute": minute} for hour in sorted(set(normalized_hours))]


def launch_agent_config(
    hours: int | Iterable[int] = DEFAULT_HOURS,
    minute: int = 0,
) -> dict[str, Any]:
    intervals = schedule_intervals(hours, minute)
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(PYTHON_PATH),
            "-m",
            "backend.automation.daily_run",
        ],
        "WorkingDirectory": str(PROJECT_ROOT),
        "StartCalendarInterval": intervals[0] if len(intervals) == 1 else intervals,
        "RunAtLoad": False,
        "ProcessType": "Background",
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(STDOUT_PATH),
        "StandardErrorPath": str(STDERR_PATH),
    }


def domain() -> str:
    return f"gui/{os.getuid()}"


def launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def install(plist_path: Path, hours: int | Iterable[int], minute: int) -> None:
    if not PYTHON_PATH.is_file():
        raise FileNotFoundError(
            f"Virtual-environment Python not found: {PYTHON_PATH}. Create .venv first."
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    with plist_path.open("wb") as plist_file:
        plistlib.dump(launch_agent_config(hours, minute), plist_file, sort_keys=False)
    launchctl("bootout", domain(), str(plist_path), check=False)
    launchctl("bootstrap", domain(), str(plist_path))


def uninstall(plist_path: Path) -> bool:
    launchctl("bootout", domain(), str(plist_path), check=False)
    if not plist_path.exists():
        return False
    plist_path.unlink()
    return True


def scheduler_status() -> tuple[bool, str]:
    result = launchctl("print", f"{domain()}/{LABEL}", check=False)
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument(
        "--hour",
        dest="hours",
        type=int,
        action="append",
        help="local run hour; repeat for multiple times (default: 9 and 21)",
    )
    install_parser.add_argument("--minute", type=int, default=0)
    install_parser.add_argument("--plist", type=Path, default=DEFAULT_PLIST)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--plist", type=Path, default=DEFAULT_PLIST)
    uninstall_parser = subparsers.add_parser("uninstall")
    uninstall_parser.add_argument("--plist", type=Path, default=DEFAULT_PLIST)
    args = parser.parse_args()

    try:
        if args.command == "install":
            hours = args.hours or list(DEFAULT_HOURS)
            install(args.plist, hours, args.minute)
            times = " and ".join(f"{hour:02d}:{args.minute:02d}" for hour in hours)
            print(f"MLBAI scheduled for {times} local time daily.")
            print(f"LaunchAgent installed at: {args.plist}")
        elif args.command == "uninstall":
            removed = uninstall(args.plist)
            print("MLBAI scheduler removed." if removed else "MLBAI scheduler was not installed.")
        else:
            active, details = scheduler_status()
            print("MLBAI scheduler is active." if active else "MLBAI scheduler is not active.")
            if active:
                print(f"Configuration: {args.plist}")
                print(f"Daily log: {STDOUT_PATH}")
            elif details:
                print(details)
            if not active:
                raise SystemExit(1)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Scheduler operation failed: {error}") from error


if __name__ == "__main__":
    main()
