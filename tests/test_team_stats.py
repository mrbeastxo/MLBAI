from datetime import date

from backend.data_pipeline.team_stats import (
    TEAM_STAT_FIELDS,
    combine_team_row,
    season_stat_groups,
    standings_rows,
)


def test_standings_rows_extracts_record_and_splits() -> None:
    payload = {
        "records": [
            {
                "division": {"id": 201},
                "teamRecords": [
                    {
                        "team": {"id": 1, "name": "Example Club"},
                        "season": "2026",
                        "gamesPlayed": 100,
                        "wins": 60,
                        "losses": 40,
                        "winningPercentage": ".600",
                        "runsScored": 500,
                        "runsAllowed": 400,
                        "runDifferential": 100,
                        "records": {
                            "splitRecords": [
                                {"type": "home", "wins": 35, "losses": 15},
                                {"type": "away", "wins": 25, "losses": 25},
                            ]
                        },
                    }
                ],
            }
        ]
    }

    row = standings_rows(payload)[0]
    assert row["team_id"] == 1
    assert row["run_differential"] == 100
    assert row["home_wins"] == 35
    assert row["away_losses"] == 25


def test_season_stat_groups_indexes_stat_blocks() -> None:
    payload = {
        "stats": [
            {
                "group": {"displayName": "hitting"},
                "splits": [{"stat": {"ops": ".800"}}],
            },
            {
                "group": {"displayName": "pitching"},
                "splits": [{"stat": {"era": "3.50"}}],
            },
        ]
    }
    assert season_stat_groups(payload) == {
        "hitting": {"ops": ".800"},
        "pitching": {"era": "3.50"},
    }


def test_combine_team_row_has_stable_columns() -> None:
    standing = {field: None for field in TEAM_STAT_FIELDS}
    standing.update({"team_id": 1, "team_name": "Example Club", "season": "2026"})
    stats_payload = {
        "stats": [
            {
                "group": {"displayName": "hitting"},
                "splits": [{"stat": {"avg": ".250", "ops": ".750"}}],
            },
            {
                "group": {"displayName": "pitching"},
                "splits": [{"stat": {"era": "3.75", "whip": "1.20"}}],
            },
        ]
    }

    row = combine_team_row(standing, stats_payload, date(2026, 7, 20))
    assert list(row) == TEAM_STAT_FIELDS
    assert row["snapshot_date"] == "2026-07-20"
    assert row["ops"] == ".750"
    assert row["pitching_whip"] == "1.20"
