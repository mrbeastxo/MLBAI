from datetime import date

from backend.data_pipeline.historical_training import (
    build_training_rows,
    completed_regular_games,
)


def _game(game_id: int, day: str, away_score: int, home_score: int) -> dict:
    return {
        "game_id": str(game_id),
        "official_date": day,
        "game_time_utc": f"{day}T20:00:00Z",
        "season": "2025",
        "away_team_id": "1",
        "away_team_name": "Away Club",
        "home_team_id": "2",
        "home_team_name": "Home Club",
        "away_score": away_score,
        "home_score": home_score,
        "home_win": int(home_score > away_score),
    }


def test_training_rows_use_only_prior_dates() -> None:
    games = [_game(1, "2025-04-01", 1, 5), _game(2, "2025-04-02", 4, 2)]
    rows = build_training_rows(games, date(2025, 4, 1), date(2025, 4, 2))

    assert rows[0]["home_games_before"] == 0
    assert rows[0]["home_win_percentage_before"] == 0.0
    assert rows[1]["home_games_before"] == 1
    assert rows[1]["home_win_percentage_before"] == 1.0
    assert rows[1]["away_win_percentage_before"] == 0.0


def test_same_date_games_do_not_leak_between_rows() -> None:
    games = [_game(1, "2025-04-01", 1, 5), _game(2, "2025-04-01", 4, 2)]
    rows = build_training_rows(games, date(2025, 4, 1), date(2025, 4, 1))
    assert rows[0]["home_games_before"] == 0
    assert rows[1]["home_games_before"] == 0


def test_completed_regular_games_filters_nonfinal_and_nonregular() -> None:
    base = {
        "gamePk": 1,
        "gameType": "R",
        "season": "2025",
        "officialDate": "2025-04-01",
        "gameDate": "2025-04-01T20:00:00Z",
        "status": {"abstractGameState": "Final"},
        "teams": {
            "away": {"team": {"id": 1, "name": "Away"}, "score": 1, "isWinner": False},
            "home": {"team": {"id": 2, "name": "Home"}, "score": 2, "isWinner": True},
        },
    }
    scheduled = {**base, "gamePk": 2, "status": {"abstractGameState": "Preview"}}
    postseason = {**base, "gamePk": 3, "gameType": "F"}
    scoreless_placeholder = {
        **base,
        "teams": {
            "away": {"team": {"id": 1, "name": "Away"}},
            "home": {"team": {"id": 2, "name": "Home"}},
        },
    }
    payload = {
        "dates": [
            {"date": "2025-03-31", "games": [scoreless_placeholder]},
            {"date": "2025-04-01", "games": [base, scheduled, postseason]},
        ]
    }
    assert [game["game_id"] for game in completed_regular_games(payload)] == ["1"]
