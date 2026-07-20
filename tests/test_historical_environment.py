from backend.data_pipeline.historical_environment import (
    environment_features,
    join_environment,
)


def test_environment_features_parse_official_weather() -> None:
    game = {
        "weather": {"temp": "78", "condition": "Light Rain", "wind": "12 mph, Out To CF"},
        "venue": {"fieldInfo": {"roofType": "Open"}},
    }
    row = environment_features(game)
    assert row["temperature_f"] == "78"
    assert row["wind_speed_mph"] == 12.0
    assert row["wind_out_indicator"] == 1
    assert row["weather_precipitation_indicator"] == 1
    assert row["environment_history_missing"] == 0


def test_join_environment_marks_unmatched_games() -> None:
    rows, coverage = join_environment(
        [{"game_id": "1"}, {"game_id": "2"}],
        [{"game_id": "1", "temperature_f": 70}],
    )
    assert coverage == 0.5
    assert rows[0]["temperature_f"] == 70
    assert rows[1]["environment_history_missing"] == 1
