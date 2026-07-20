from datetime import date

from backend.data_pipeline.bullpen_workload import (
    build_bullpen_rows,
    relief_appearances,
)


def test_relief_appearances_excludes_starter() -> None:
    boxscore = {
        "teams": {
            "away": {
                "team": {"id": 1, "name": "Away Club"},
                "pitchers": [10, 11],
                "players": {
                    "ID10": {
                        "person": {"fullName": "Starter"},
                        "stats": {"pitching": {"gamesStarted": 1, "numberOfPitches": 90, "outs": 18}},
                    },
                    "ID11": {
                        "person": {"fullName": "Reliever"},
                        "stats": {"pitching": {"gamesStarted": 0, "numberOfPitches": 20, "outs": 3}},
                    },
                },
            },
            "home": {"team": {"id": 2}, "pitchers": [], "players": {}},
        }
    }
    rows = relief_appearances(boxscore, date(2026, 7, 19))
    assert len(rows) == 1
    assert rows[0]["pitcher_name"] == "Reliever"
    assert rows[0]["pitches"] == 20


def test_build_bullpen_rows_calculates_back_to_back_and_index() -> None:
    snapshot_date = date(2026, 7, 20)
    appearances = [
        {"game_date": date(2026, 7, 19), "team_id": 1, "pitcher_id": 11, "pitches": 20, "outs": 3},
        {"game_date": date(2026, 7, 18), "team_id": 1, "pitcher_id": 11, "pitches": 25, "outs": 3},
        {"game_date": date(2026, 7, 17), "team_id": 1, "pitcher_id": 12, "pitches": 16, "outs": 3},
    ]
    rows = build_bullpen_rows(
        {1: "Example Club"}, appearances, {1: {date(2026, 7, 17), date(2026, 7, 18), date(2026, 7, 19)}}, snapshot_date
    )
    row = rows[0]
    assert row["bullpen_pitches_last_3_days"] == 61
    assert row["bullpen_innings_last_3_days"] == "3.0"
    assert row["relievers_back_to_back"] == 1
    assert row["relievers_used_2_of_3_days"] == 1
    assert row["high_workload_relievers"] == 1
    assert row["workload_index"] == 66.5
