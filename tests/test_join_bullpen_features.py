from backend.data_pipeline.join_bullpen_features import join_bullpen_features


def test_join_bullpen_features_matches_game_id() -> None:
    training = [{"game_id": "1", "official_date": "2025-04-01"}]
    bullpens = [
        {
            "game_id": "1",
            "official_date": "2025-04-01",
            "bullpen_era_home_minus_away": "-1.0",
        }
    ]
    rows, coverage = join_bullpen_features(training, bullpens)
    assert rows[0]["bullpen_era_home_minus_away"] == "-1.0"
    assert coverage == 1.0
