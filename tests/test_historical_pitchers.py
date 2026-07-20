from datetime import date

from backend.data_pipeline.historical_pitchers import (
    build_pitcher_feature_rows,
    completed_game_ids,
    starter_appearances,
)


def _appearance(game_id: str, day: str, side: str, pitcher: str, earned_runs: int) -> dict:
    return {
        "game_id": game_id,
        "official_date": day,
        "side": side,
        "team_id": "1" if side == "away" else "2",
        "pitcher_id": pitcher,
        "pitcher_name": f"Pitcher {pitcher}",
        "outs": 18,
        "earned_runs": earned_runs,
        "hits": 5,
        "walks": 1,
        "strikeouts": 6,
        "home_runs": 1,
        "pitches": 90,
    }


def test_starter_appearances_excludes_relievers() -> None:
    boxscore = {
        "teams": {
            "away": {
                "team": {"id": 1},
                "pitchers": [10, 11],
                "players": {
                    "ID10": {"person": {"fullName": "Starter"}, "stats": {"pitching": {"gamesStarted": 1, "outs": 18}}},
                    "ID11": {"person": {"fullName": "Reliever"}, "stats": {"pitching": {"gamesStarted": 0, "outs": 3}}},
                },
            },
            "home": {"team": {"id": 2}, "pitchers": [], "players": {}},
        }
    }
    rows = starter_appearances(boxscore, 99, date(2025, 4, 1))
    assert len(rows) == 1
    assert rows[0]["pitcher_name"] == "Starter"


def test_pitcher_features_use_only_prior_dates() -> None:
    appearances = [
        _appearance("1", "2025-04-01", "away", "10", 2),
        _appearance("1", "2025-04-01", "home", "20", 4),
        _appearance("2", "2025-04-07", "away", "10", 1),
        _appearance("2", "2025-04-07", "home", "20", 1),
    ]
    rows = build_pitcher_feature_rows(appearances)
    assert rows[0]["away_starter_history_missing"] == 1
    assert rows[1]["away_starter_starts_before"] == 1
    assert rows[1]["away_starter_era_before"] == 3.0
    assert rows[1]["home_starter_era_before"] == 6.0
    assert rows[1]["starter_era_home_minus_away"] == 3.0
    assert rows[1]["away_starter_days_since_last_start"] == 6


def test_completed_game_ids_rejects_makeup_date_outside_range() -> None:
    payload = {
        "dates": [
            {
                "date": "2025-04-01",
                "games": [
                    {
                        "gamePk": 1,
                        "officialDate": "2025-04-01",
                        "status": {"abstractGameState": "Final"},
                    },
                    {
                        "gamePk": 2,
                        "officialDate": "2025-08-09",
                        "status": {"abstractGameState": "Final"},
                    },
                ],
            }
        ]
    }
    assert completed_game_ids(payload, date(2025, 4, 1), date(2025, 4, 15)) == [
        (1, date(2025, 4, 1))
    ]
