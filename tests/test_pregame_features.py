from datetime import date

from backend.data_pipeline.pregame_features import scheduled_regular_games


def test_scheduled_games_excludes_final_and_wrong_date() -> None:
    def game(game_id: int, day: str, state: str) -> dict:
        return {
            "gamePk": game_id,
            "gameType": "R",
            "officialDate": day,
            "gameDate": f"{day}T20:00:00Z",
            "season": "2026",
            "status": {"abstractGameState": state},
            "teams": {
                "away": {"team": {"id": 1, "name": "Away"}},
                "home": {"team": {"id": 2, "name": "Home"}},
            },
        }
    payload = {
        "dates": [
            {
                "date": "2026-07-20",
                "games": [
                    game(1, "2026-07-20", "Preview"),
                    game(2, "2026-07-20", "Final"),
                    game(3, "2026-08-01", "Preview"),
                ],
            }
        ]
    }
    rows = scheduled_regular_games(payload, date(2026, 7, 20))
    assert [row["game_id"] for row in rows] == ["1"]
