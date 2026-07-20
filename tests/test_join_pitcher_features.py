from backend.data_pipeline.join_pitcher_features import join_pitcher_features


def test_join_pitcher_features_filters_range_and_reports_coverage() -> None:
    training = [
        {"game_id": "1", "official_date": "2025-04-01", "season": "2025"},
        {"game_id": "2", "official_date": "2025-04-02", "season": "2025"},
        {"game_id": "3", "official_date": "2025-04-03", "season": "2025"},
    ]
    pitchers = [
        {
            "game_id": "2",
            "official_date": "2025-04-02",
            "starter_era_home_minus_away": "-1.0",
            # The compact in-memory backfill uses integers; CSV reloads use strings.
            "away_starter_history_missing": 0,
            "home_starter_history_missing": 0,
        },
        {
            "game_id": "3",
            "official_date": "2025-04-03",
            "starter_era_home_minus_away": "0.5",
            "away_starter_history_missing": "1",
            "home_starter_history_missing": "0",
        },
    ]
    rows, report = join_pitcher_features(training, pitchers)
    assert [row["game_id"] for row in rows] == ["2", "3"]
    assert rows[0]["starter_era_home_minus_away"] == "-1.0"
    assert report["matchup_coverage"] == 1.0
    assert report["complete_history_coverage"] == 0.5
