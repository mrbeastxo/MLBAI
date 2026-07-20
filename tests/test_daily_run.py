import json
from datetime import UTC, date, datetime

from backend.automation import daily_run
from backend.tracking.prediction_tracker import connect_database, record_predictions


def tracked_prediction(game_id: str, official_date: str) -> dict[str, str]:
    return {
        "game_id": game_id,
        "official_date": official_date,
        "game_time_utc": f"{official_date}T20:00:00Z",
        "away_team": "Away",
        "home_team": "Home",
        "away_win_probability": "0.4",
        "home_win_probability": "0.6",
        "model_lean": "Home",
    }


def test_settle_pending_checks_each_unsettled_past_date(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    before = datetime(2026, 7, 18, 10, tzinfo=UTC)
    record_predictions(connection, [tracked_prediction("1", "2026-07-18")], before)

    def fake_schedule(game_date):
        assert game_date == date(2026, 7, 18)
        return {
            "dates": [{"games": [{
                "gamePk": 1,
                "status": {"abstractGameState": "Final", "detailedState": "Final"},
                "teams": {
                    "away": {"score": 2, "isWinner": False},
                    "home": {"score": 5, "isWinner": True},
                },
            }]}]
        }

    summary = daily_run.settle_pending(
        connection, date(2026, 7, 20), schedule_fetcher=fake_schedule
    )
    assert summary["dates_checked"] == ["2026-07-18"]
    assert summary["settled"] == 1
    assert summary["errors"] == []


def test_daily_run_with_no_games_is_successful_and_restart_safe(tmp_path) -> None:
    output = tmp_path / "processed"
    database = tmp_path / "ledger.sqlite3"
    season_payload = {"dates": []}
    now = datetime(2026, 7, 20, 10, tzinfo=UTC)

    first = daily_run.run_daily(
        date(2026, 7, 20),
        database_path=database,
        output_dir=output,
        now=now,
        season_fetcher=lambda year, through: season_payload,
        schedule_fetcher=lambda game_date: {"dates": []},
    )
    second = daily_run.run_daily(
        date(2026, 7, 20),
        database_path=database,
        output_dir=output,
        now=now,
        season_fetcher=lambda year, through: season_payload,
        schedule_fetcher=lambda game_date: {"dates": []},
    )
    report = json.loads((output / "daily_run_2026-07-20.json").read_text())
    assert first["status"] == second["status"] == "success"
    assert report["scheduled_games"] == 0
    assert (output / "analysis_2026-07-20.json").read_text() == "[]"


def test_dry_run_does_not_write_predictions_to_ledger(tmp_path, monkeypatch) -> None:
    output = tmp_path / "processed"
    database = tmp_path / "ledger.sqlite3"
    feature = {field: "0" for field in daily_run.PREGAME_FIELDS}
    feature.update(
        {
            "game_id": "9",
            "official_date": "2026-07-20",
            "game_time_utc": "2026-07-20T20:00:00Z",
            "away_team_name": "Away",
            "home_team_name": "Home",
        }
    )
    prediction = tracked_prediction("9", "2026-07-20")
    model = tmp_path / "model.joblib"
    model.write_text("placeholder")
    model_report = tmp_path / "report.json"
    model_report.write_text("{}")
    monkeypatch.setattr(daily_run, "build_pregame_rows", lambda payload, run_date: [feature])
    monkeypatch.setattr(daily_run.joblib, "load", lambda path: {})
    monkeypatch.setattr(daily_run, "predict_rows", lambda rows, artifact: [prediction])
    monkeypatch.setattr(daily_run, "explain_rows", lambda rows, artifact, report: [prediction])

    summary = daily_run.run_daily(
        date(2026, 7, 20),
        database_path=database,
        model_path=model,
        model_report_path=model_report,
        output_dir=output,
        now=datetime(2026, 7, 20, 10, tzinfo=UTC),
        dry_run=True,
        season_fetcher=lambda year, through: {},
        schedule_fetcher=lambda game_date: {"dates": []},
    )
    connection = connect_database(database)
    count = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    connection.close()
    assert summary["tracking"]["recorded"] == 0
    assert count == 0
