from datetime import date

from backend.data_pipeline.historical_bullpens import (
    build_bullpen_feature_rows,
    relief_lines,
)


def test_relief_lines_excludes_starter() -> None:
    boxscore = {
        "teams": {
            "away": {
                "team": {"id": 1},
                "pitchers": [10, 11],
                "players": {
                    "ID10": {"stats": {"pitching": {"gamesStarted": 1}}},
                    "ID11": {"stats": {"pitching": {"gamesStarted": 0, "outs": 3, "numberOfPitches": 18}}},
                },
            },
            "home": {"team": {"id": 2}, "pitchers": [], "players": {}},
        }
    }
    rows = relief_lines(boxscore, "99", date(2025, 4, 1))
    assert len(rows) == 1
    assert rows[0]["pitcher_id"] == "11"
    assert rows[0]["pitches"] == 18


def test_bullpen_features_use_only_prior_dates() -> None:
    line = {
        "game_id": "1",
        "official_date": "2025-04-01",
        "side": "away",
        "team_id": "1",
        "pitcher_id": "11",
        "outs": 3,
        "earned_runs": 1,
        "hits": 1,
        "walks": 0,
        "strikeouts": 1,
        "pitches": 18,
    }
    games = [
        {"game_id": "1", "official_date": date(2025, 4, 1), "away_team_id": "1", "home_team_id": "2", "relief_lines": [line]},
        {"game_id": "2", "official_date": date(2025, 4, 2), "away_team_id": "1", "home_team_id": "2", "relief_lines": []},
    ]
    rows = build_bullpen_feature_rows(games)
    assert rows[0]["away_bullpen_history_missing"] == 1
    assert rows[1]["away_bullpen_history_missing"] == 0
    assert rows[1]["away_bullpen_era_before"] == 9.0
    assert rows[1]["away_bullpen_pitches_yesterday"] == 18
