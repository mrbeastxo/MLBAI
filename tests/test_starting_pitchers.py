from datetime import date

from backend.data_pipeline.starting_pitchers import (
    PITCHER_FIELDS,
    pitcher_row,
    probable_pitcher_assignments,
)


def test_probable_pitcher_assignments_skips_unannounced_starter() -> None:
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 99,
                        "gameDate": "2026-07-20T23:00:00Z",
                        "teams": {
                            "away": {
                                "team": {"id": 1, "name": "Away Club"},
                                "probablePitcher": {"id": 10, "fullName": "Starter A"},
                            },
                            "home": {"team": {"id": 2, "name": "Home Club"}},
                        },
                    }
                ]
            }
        ]
    }
    rows = probable_pitcher_assignments(payload)
    assert len(rows) == 1
    assert rows[0]["pitcher_id"] == 10
    assert rows[0]["opponent_name"] == "Home Club"


def test_pitcher_row_calculates_recent_workload_before_game_date() -> None:
    assignment = {
        "game_id": 99,
        "game_time_utc": "2026-07-20T23:00:00Z",
        "side": "away",
        "team_id": 1,
        "team_name": "Away Club",
        "opponent_id": 2,
        "opponent_name": "Home Club",
        "pitcher_id": 10,
        "pitcher_name": "Starter A",
    }
    payload = {
        "people": [
            {
                "fullName": "Starter A",
                "currentAge": 27,
                "pitchHand": {"code": "R"},
                "stats": [
                    {
                        "type": {"displayName": "season"},
                        "splits": [{"stat": {"era": "3.20", "whip": "1.10"}}],
                    },
                    {
                        "type": {"displayName": "gameLog"},
                        "splits": [
                            {
                                "date": "2026-07-08",
                                "stat": {"numberOfPitches": 90, "outs": 18},
                            },
                            {
                                "date": "2026-07-14",
                                "stat": {"numberOfPitches": 96, "outs": 20},
                            },
                            {
                                "date": "2026-07-20",
                                "stat": {"numberOfPitches": 100, "outs": 21},
                            },
                        ],
                    },
                ],
            }
        ]
    }

    row = pitcher_row(assignment, payload, date(2026, 7, 20))
    assert list(row) == PITCHER_FIELDS
    assert row["days_rest"] == 6
    assert row["pitches_last_appearance"] == 96
    assert row["pitches_last_3"] == 186
    assert row["innings_last_3"] == "12.2"
