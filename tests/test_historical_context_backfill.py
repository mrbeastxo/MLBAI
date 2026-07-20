from datetime import date

from backend.data_pipeline.historical_context_backfill import compact_boxscore


def test_compact_boxscore_keeps_starter_and_reliever_without_raw_payload() -> None:
    boxscore = {
        "teams": {
            "away": {
                "team": {"id": 1},
                "pitchers": [10, 11],
                "players": {
                    "ID10": {
                        "person": {"fullName": "Starter"},
                        "stats": {"pitching": {"gamesStarted": 1, "outs": 18}},
                    },
                    "ID11": {
                        "person": {"fullName": "Reliever"},
                        "stats": {"pitching": {"gamesStarted": 0, "outs": 3}},
                    },
                },
            },
            "home": {"team": {"id": 2}, "pitchers": [], "players": {}},
        }
    }
    record = compact_boxscore(boxscore, 99, date(2025, 4, 1))
    assert record["away_team_id"] == "1"
    assert record["starters"][0]["pitcher_name"] == "Starter"
    assert record["relievers"][0]["pitcher_id"] == "11"
    assert "players" not in record
