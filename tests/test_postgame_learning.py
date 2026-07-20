from datetime import UTC, datetime

from backend.tracking.prediction_tracker import (
    connect_database,
    record_predictions,
    settle_results,
)
from ml.postgame_learning import learning_report, sample_label


def prediction() -> dict[str, str]:
    return {
        "game_id": "1",
        "official_date": "2026-07-20",
        "game_time_utc": "2026-07-20T20:00:00Z",
        "away_team": "Away",
        "home_team": "Home",
        "away_win_probability": "0.4",
        "home_win_probability": "0.6",
        "model_lean": "Home",
    }


def test_learning_report_compares_shadow_and_explains_factors(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    before = datetime(2026, 7, 20, 10, tzinfo=UTC)
    shadow = {"1": {**prediction(), "home_win_probability": "0.55", "away_win_probability": "0.45", "model_version": "team_only_v0.30"}}
    context = {"1": {"strongest_supporting_factors": [{"factor": "starting-pitcher ERA", "log_odds_contribution": 0.2}], "strongest_opposing_factors": []}}
    summary = record_predictions(
        connection,
        [prediction()],
        before,
        shadow_rows=shadow,
        learning_context=context,
    )
    settle_results(
        connection,
        [{"game_id": "1", "away_score": 2, "home_score": 5, "home_win": 1, "status": "Final"}],
    )
    report = learning_report(connection)
    assert summary["shadow_recorded"] == 1
    assert summary["context_recorded"] == 1
    assert report["settled_future_games"] == 1
    assert report["shadow_comparison"]["comparable_games"] == 1
    assert report["factor_diagnostics"][0]["helped"] == 1
    assert report["sample_status"] == "too_early"
    assert report["drift_flag"] is False


def test_sample_labels_require_two_hundred_games_for_action() -> None:
    assert sample_label(49) == "too_early"
    assert sample_label(50) == "early"
    assert sample_label(100) == "limited"
    assert sample_label(200) == "actionable"
