"""Build a read-only health snapshot for the MLBAI dashboard."""

from __future__ import annotations

import json
import plistlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from backend.automation.macos_scheduler import DEFAULT_PLIST, STDERR_PATH, STDOUT_PATH
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def latest_run(processed_dir: Path = PROCESSED_DATA_DIR) -> dict[str, Any] | None:
    reports = sorted(processed_dir.glob("daily_run_*.json"), reverse=True)
    for report_path in reports:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def next_run_time(hour: int, minute: int, now: datetime | None = None) -> datetime:
    now = now or datetime.now().astimezone()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def scheduler_info(
    plist_path: Path = DEFAULT_PLIST,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not plist_path.is_file():
        return {"installed": False, "schedule": None, "next_run_local": None}
    try:
        with plist_path.open("rb") as plist_file:
            payload = plistlib.load(plist_file)
        interval = payload["StartCalendarInterval"]
        hour = int(interval["Hour"])
        minute = int(interval["Minute"])
    except (OSError, KeyError, TypeError, ValueError, plistlib.InvalidFileException):
        return {
            "installed": True,
            "schedule": None,
            "next_run_local": None,
            "configuration_valid": False,
        }
    return {
        "installed": True,
        "configuration_valid": True,
        "schedule": f"{hour:02d}:{minute:02d}",
        "next_run_local": next_run_time(hour, minute, now).isoformat(),
    }


def log_info(stdout_path: Path = STDOUT_PATH, stderr_path: Path = STDERR_PATH) -> dict[str, Any]:
    return {
        "output_bytes": stdout_path.stat().st_size if stdout_path.is_file() else 0,
        "error_bytes": stderr_path.stat().st_size if stderr_path.is_file() else 0,
        "has_errors": stderr_path.is_file() and stderr_path.stat().st_size > 0,
    }


def system_health(
    *,
    data_dir: Path = DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    plist_path: Path = DEFAULT_PLIST,
    stdout_path: Path = STDOUT_PATH,
    stderr_path: Path = STDERR_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    run = latest_run(processed_dir)
    return {
        "scheduler": scheduler_info(plist_path, now),
        "last_run": run,
        "logs": log_info(stdout_path, stderr_path),
        "storage": {"data_bytes": directory_size(data_dir)},
        "project_root": str(PROJECT_ROOT),
    }
