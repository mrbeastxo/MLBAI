from backend.data_pipeline.standings import build_tables


def test_build_tables_flattens_divisions_into_league_rankings() -> None:
    payload = {"records": [{
        "league": {"id": 103},
        "lastUpdated": "2026-07-20T00:00:00Z",
        "teamRecords": [{
            "team": {"id": 1, "name": "Test Club", "abbreviation": "TST", "league": {"id": 103}, "division": {"name": "American League East"}},
            "leagueRank": "2", "wins": 55, "losses": 44,
            "winningPercentage": ".556", "leagueGamesBack": "2.0",
            "runDifferential": 30, "streak": {"streakCode": "W2"},
            "records": {"splitRecords": [{"type": "lastTen", "wins": 7, "losses": 3}]},
        }],
    }]}
    tables = build_tables(payload, 2026)
    team = tables["leagues"][0]["teams"][0]
    assert team["rank"] == 2
    assert team["last_ten"] == "7-3"
    assert team["run_differential"] == 30
