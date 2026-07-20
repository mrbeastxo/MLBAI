import sqlite3
from datetime import UTC, datetime

import pytest

from backend.tracking.prediction_tracker import (
    connect_database,
    performance_report,
    record_predictions,
    settle_results,
    verify_hash_chain,
    verify_score_projection_hashes,
)


def prediction(home_probability: str = "0.6") -> dict[str, str]:
    away_probability = str(round(1 - float(home_probability), 4))
    return {
        "game_id": "1",
        "official_date": "2026-07-20",
        "game_time_utc": "2026-07-20T20:00:00Z",
        "away_team": "Away",
        "home_team": "Home",
        "away_win_probability": away_probability,
        "home_win_probability": home_probability,
        "model_lean": "Home",
        "away_expected_runs": "3.5",
        "home_expected_runs": "4.5",
        "expected_total_runs": "8.0",
    }


def test_prediction_is_immutable_and_hash_chain_valid(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 7, 20, 10, tzinfo=UTC)
    summary = record_predictions(connection, [prediction()], now)
    assert summary["recorded"] == 1
    assert summary["score_recorded"] == 1
    assert summary["shadow_recorded"] == 0
    assert verify_hash_chain(connection)
    assert verify_score_projection_hashes(connection)
    with pytest.raises(ValueError, match="differs"):
        record_predictions(connection, [prediction("0.7")], now)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute("UPDATE predictions SET home_team = 'Changed' WHERE game_id = '1'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute(
            "UPDATE score_projections SET home_expected_runs = 9 WHERE game_id = '1'"
        )


def test_scheduled_run_preserves_existing_different_prediction(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 7, 20, 10, tzinfo=UTC)
    record_predictions(connection, [prediction()], now)
    summary = record_predictions(
        connection, [prediction("0.7")], now, strict_existing=False
    )
    stored = connection.execute(
        "SELECT home_win_probability FROM predictions WHERE game_id = '1'"
    ).fetchone()
    assert summary["preserved_conflicts"] == 1
    assert stored["home_win_probability"] == 0.6


def test_after_start_is_rejected_and_settled_metrics_are_correct(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    before = datetime(2026, 7, 20, 10, tzinfo=UTC)
    after = datetime(2026, 7, 20, 21, tzinfo=UTC)
    assert record_predictions(connection, [prediction()], before)["recorded"] == 1
    late = prediction()
    late["game_id"] = "2"
    assert record_predictions(connection, [late], after)["skipped_after_start"] == 1
    settle_results(
        connection,
        [{"game_id": "1", "away_score": 2, "home_score": 5, "home_win": 1, "status": "Final"}],
        after,
    )
    report = performance_report(connection)
    assert report["settled_games"] == 1
    assert report["accuracy"] == 1.0
    assert report["brier_score"] == 0.16
    assert report["score_projection_games"] == 1
    assert report["score_mae"] == 1.0
    assert report["score_rmse"] == 1.118
    assert report["total_runs_mae"] == 1.0
    assert report["score_projection_hashes_valid"] is True


def test_score_projection_cannot_change_before_start(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 7, 20, 10, tzinfo=UTC)
    record_predictions(connection, [prediction()], now)
    changed = prediction()
    changed["home_expected_runs"] = "5.0"
    changed["expected_total_runs"] = "8.5"
    with pytest.raises(ValueError, match="Score projection.*differs"):
        record_predictions(connection, [changed], now)
