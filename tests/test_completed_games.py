import csv
from datetime import date

from backend.data_pipeline.completed_games import completed_game_rows, save_results


def test_completed_game_rows_keeps_only_final_games() -> None:
    final_game = {
        "gamePk": 123,
        "officialDate": "2025-07-20",
        "season": "2025",
        "gameType": "R",
        "status": {"abstractGameState": "Final", "detailedState": "Final"},
        "venue": {"id": 10, "name": "Example Park"},
        "teams": {
            "away": {
                "score": 3,
                "isWinner": False,
                "team": {"id": 1, "name": "Away Club"},
            },
            "home": {
                "score": 5,
                "isWinner": True,
                "team": {"id": 2, "name": "Home Club"},
            },
        },
    }
    scheduled_game = {
        "gamePk": 456,
        "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
    }

    rows = completed_game_rows(
        {"dates": [{"date": "2025-07-20", "games": [final_game, scheduled_game]}]}
    )

    assert len(rows) == 1
    assert rows[0]["game_id"] == 123
    assert rows[0]["away_score"] == 3
    assert rows[0]["home_score"] == 5
    assert rows[0]["winner_team"] == "Home Club"


def test_save_results_writes_csv(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "backend.data_pipeline.completed_games.PROCESSED_DATA_DIR", tmp_path
    )
    rows = [
        {
            "game_id": 123,
            "official_date": "2025-07-20",
            "season": "2025",
            "game_type": "R",
            "away_team_id": 1,
            "away_team": "Away Club",
            "away_score": 3,
            "home_team_id": 2,
            "home_team": "Home Club",
            "home_score": 5,
            "winner_team_id": 2,
            "winner_team": "Home Club",
            "venue_id": 10,
            "venue": "Example Park",
            "status": "Final",
        }
    ]

    output_path = save_results(rows, date(2025, 7, 20), date(2025, 7, 20))

    with output_path.open(encoding="utf-8", newline="") as csv_file:
        saved_rows = list(csv.DictReader(csv_file))
    assert saved_rows[0]["winner_team"] == "Home Club"
    assert saved_rows[0]["home_score"] == "5"
