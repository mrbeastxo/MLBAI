from datetime import UTC, datetime

from backend.history.season_results import build_season_results, save_season_results
from backend.tracking.prediction_tracker import (
    connect_database,
    record_predictions,
    settle_results,
)


def final_game(game_id=1):
    return {
        "gamePk": game_id,
        "officialDate": "2026-07-19",
        "season": "2026",
        "gameType": "R",
        "status": {"abstractGameState": "Final", "detailedState": "Final"},
        "venue": {"id": 10, "name": "Park"},
        "teams": {
            "away": {"score": 2, "isWinner": False, "team": {"id": 1, "name": "Away"}},
            "home": {"score": 5, "isWinner": True, "team": {"id": 2, "name": "Home"}},
        },
    }


def prediction():
    return {
        "game_id": "1",
        "official_date": "2026-07-19",
        "game_time_utc": "2026-07-19T20:00:00Z",
        "away_team": "Away",
        "home_team": "Home",
        "away_win_probability": "0.4",
        "home_win_probability": "0.6",
        "model_lean": "Home",
    }


def test_results_distinguish_verified_and_untracked_games(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    record_predictions(
        connection, [prediction()], datetime(2026, 7, 19, 10, tzinfo=UTC)
    )
    settle_results(
        connection,
        [{"game_id": "1", "away_score": 2, "home_score": 5, "home_win": 1, "status": "Final"}],
    )
    payload = {"dates": [{"games": [final_game(1), final_game(2)]}]}
    rows = build_season_results(payload, connection)
    by_id = {str(row["game_id"]): row for row in rows}
    assert by_id["1"]["mlbai_verified"] is True
    assert by_id["1"]["mlbai_correct"] is True
    assert by_id["2"]["mlbai_tracked"] is False
    assert by_id["2"]["mlbai_correct"] is None


def test_final_status_without_scores_is_not_a_completed_result(tmp_path) -> None:
    connection = connect_database(tmp_path / "ledger.sqlite3")
    game = final_game(3)
    game["teams"]["away"]["score"] = None
    rows = build_season_results({"dates": [{"games": [game]}]}, connection)
    assert rows == []


def test_save_season_results_writes_json(tmp_path) -> None:
    path = save_season_results([{"game_id": 1}], 2026, tmp_path)
    assert path.name == "season_results_2026.json"
    assert '"game_id": 1' in path.read_text()
