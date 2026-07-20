import sqlite3
from datetime import UTC, datetime

import pytest

from backend.tracking.prediction_tracker import (
    connect_database,
    performance_report,
    record_predictions,
    settle_results,
    verify_hash_chain,
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
    }


def test_prediction_is_immutable_and_hash_chain_valid(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 7, 20, 10, tzinfo=UTC)
    assert record_predictions(connection, [prediction()], now)["recorded"] == 1
    assert verify_hash_chain(connection)
    with pytest.raises(ValueError, match="differs"):
        record_predictions(connection, [prediction("0.7")], now)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute("UPDATE predictions SET home_team = 'Changed' WHERE game_id = '1'")


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
