from datetime import date

from backend.data_pipeline.daily_features import (
    FEATURE_FIELDS,
    build_feature_rows,
    schedule_rows,
)


def test_schedule_rows_extracts_stable_team_ids() -> None:
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 99,
                        "gameDate": "2026-07-20T23:00:00Z",
                        "status": {"detailedState": "Scheduled"},
                        "teams": {
                            "away": {"team": {"id": 1, "name": "Away Club"}},
                            "home": {"team": {"id": 2, "name": "Home Club"}},
                        },
                    }
                ]
            }
        ]
    }
    row = schedule_rows(payload, date(2026, 7, 20))[0]
    assert row["game_id"] == "99"
    assert row["away_team_id"] == "1"
    assert row["home_team_id"] == "2"


def test_build_feature_rows_joins_and_calculates_differences() -> None:
    games = [
        {
            "snapshot_date": "2026-07-20",
            "game_id": "99",
            "game_time_utc": "2026-07-20T23:00:00Z",
            "status": "Scheduled",
            "away_team_id": "1",
            "away_team_name": "Away Club",
            "home_team_id": "2",
            "home_team_name": "Home Club",
        }
    ]
    teams = [
        {"team_id": "1", "winning_percentage": ".400", "run_differential": "-10", "ops": ".700", "pitching_era": "4.50"},
        {"team_id": "2", "winning_percentage": ".600", "run_differential": "20", "ops": ".800", "pitching_era": "3.50"},
    ]
    starters = [
        {"game_id": "99", "side": "away", "pitcher_id": "10", "season_era": "4.00"},
        {"game_id": "99", "side": "home", "pitcher_id": "20", "season_era": "3.00"},
    ]
    bullpens = [
        {"team_id": "1", "workload_index": "100"},
        {"team_id": "2", "workload_index": "70"},
    ]
    row = build_feature_rows(games, teams, starters, bullpens)[0]
    assert list(row) == FEATURE_FIELDS
    assert row["home_team_ops"] == ".800"
    assert row["winning_percentage_home_minus_away"] == 0.2
    assert row["run_differential_home_minus_away"] == 30.0
    assert row["starter_era_home_minus_away"] == -1.0
    assert row["bullpen_workload_home_minus_away"] == -30.0
    assert row["away_starter_missing"] == 0
