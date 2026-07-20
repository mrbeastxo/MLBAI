from datetime import date

from backend.data_pipeline.mlb_schedule import extract_games, parse_date


def test_parse_date() -> None:
    assert parse_date("2026-07-20") == date(2026, 7, 20)


def test_extract_games() -> None:
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 123,
                        "gameDate": "2026-07-20T23:05:00Z",
                        "status": {"detailedState": "Scheduled"},
                        "teams": {
                            "away": {"team": {"name": "Away Club"}},
                            "home": {"team": {"name": "Home Club"}},
                        },
                    }
                ]
            }
        ]
    }

    assert extract_games(payload) == [
        {
            "game_id": "123",
            "away": "Away Club",
            "home": "Home Club",
            "start_time_utc": "2026-07-20T23:05:00Z",
            "status": "Scheduled",
        }
    ]
