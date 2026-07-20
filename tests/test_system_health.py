import json
import os
import plistlib
from datetime import UTC, datetime

from backend.monitoring.system_health import (
    directory_size,
    latest_run,
    scheduler_info,
    system_health,
)


def test_latest_run_skips_corrupt_report(tmp_path) -> None:
    (tmp_path / "daily_run_2026-07-20.json").write_text("not json")
    (tmp_path / "daily_run_2026-07-19.json").write_text(
        json.dumps({"date": "2026-07-19", "status": "success"})
    )
    assert latest_run(tmp_path)["date"] == "2026-07-19"


def test_latest_run_uses_completion_time_not_prediction_date(tmp_path) -> None:
    (tmp_path / "daily_run_2026-07-21.json").write_text(
        json.dumps(
            {
                "date": "2026-07-21",
                "status": "success",
                "finished_at_utc": "2026-07-20T10:01:36Z",
            }
        )
    )
    (tmp_path / "daily_run_2026-07-20.json").write_text(
        json.dumps(
            {
                "date": "2026-07-20",
                "status": "success",
                "finished_at_utc": "2026-07-20T16:05:09Z",
            }
        )
    )

    assert latest_run(tmp_path)["date"] == "2026-07-20"


def test_scheduler_info_reports_next_local_run(tmp_path) -> None:
    plist = tmp_path / "agent.plist"
    with plist.open("wb") as plist_file:
        plistlib.dump({"StartCalendarInterval": {"Hour": 6, "Minute": 0}}, plist_file)
    now = datetime.fromisoformat("2026-07-20T07:00:00+05:30")
    info = scheduler_info(plist, now)
    assert info["installed"] is True
    assert info["schedule"] == "06:00"
    assert info["next_run_local"].startswith("2026-07-21T06:00:00")


def test_scheduler_info_reports_two_daily_runs(tmp_path) -> None:
    plist = tmp_path / "agent.plist"
    with plist.open("wb") as plist_file:
        plistlib.dump(
            {
                "StartCalendarInterval": [
                    {"Hour": 9, "Minute": 0},
                    {"Hour": 21, "Minute": 0},
                ]
            },
            plist_file,
        )
    now = datetime.fromisoformat("2026-07-20T10:00:00+05:30")

    info = scheduler_info(plist, now)

    assert info["schedule"] == "09:00 & 21:00"
    assert info["next_run_local"].startswith("2026-07-20T21:00:00")


def test_system_health_reports_storage_logs_and_last_run(tmp_path) -> None:
    data = tmp_path / "data"
    processed = data / "processed"
    processed.mkdir(parents=True)
    (data / "sample.bin").write_bytes(b"12345")
    (processed / "daily_run_2026-07-20.json").write_text(
        json.dumps({"date": "2026-07-20", "status": "success"})
    )
    stderr = tmp_path / "error.log"
    stderr.write_text("")
    health = system_health(
        data_dir=data,
        processed_dir=processed,
        plist_path=tmp_path / "missing.plist",
        stdout_path=tmp_path / "out.log",
        stderr_path=stderr,
    )
    assert health["last_run"]["status"] == "success"
    assert health["scheduler"]["installed"] is False
    assert health["logs"]["has_errors"] is False
    assert health["storage"]["data_bytes"] == directory_size(data)


def test_successful_retry_resolves_an_older_scheduler_error(tmp_path) -> None:
    data = tmp_path / "data"
    processed = data / "processed"
    processed.mkdir(parents=True)
    stderr = tmp_path / "error.log"
    stderr.write_text("earlier run failed")
    error_time = datetime(2026, 7, 20, 15, 30, tzinfo=UTC).timestamp()
    os.utime(stderr, (error_time, error_time))
    (processed / "daily_run_2026-07-20.json").write_text(
        json.dumps(
            {
                "date": "2026-07-20",
                "status": "success",
                "finished_at_utc": "2026-07-20T16:05:09Z",
            }
        )
    )

    health = system_health(
        data_dir=data,
        processed_dir=processed,
        plist_path=tmp_path / "missing.plist",
        stdout_path=tmp_path / "out.log",
        stderr_path=stderr,
    )

    assert health["logs"]["error_bytes"] > 0
    assert health["logs"]["has_errors"] is False
