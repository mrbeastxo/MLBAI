from backend.data_pipeline.environment_context import (
    attach_environment_context,
    environment_row,
    nearest_hour_index,
)


def test_nearest_forecast_hour_matches_first_pitch() -> None:
    times = ["2026-07-20T22:00", "2026-07-20T23:00", "2026-07-21T00:00"]
    assert nearest_hour_index(times, "2026-07-20T23:10:00Z") == 1


def test_environment_row_combines_venue_and_hourly_weather() -> None:
    game = {
        "gamePk": 1,
        "officialDate": "2026-07-20",
        "gameDate": "2026-07-20T23:10:00Z",
        "venue": {
            "id": 10,
            "name": "Test Park",
            "location": {"city": "Test City"},
            "fieldInfo": {"roofType": "Open", "turfType": "Grass"},
        },
    }
    forecast = {
        "hourly": {
            "time": ["2026-07-20T23:00"],
            "temperature_2m": [80.0],
            "precipitation_probability": [20],
            "precipitation": [0.01],
            "weather_code": [1],
            "wind_speed_10m": [8.0],
            "wind_gusts_10m": [12.0],
            "wind_direction_10m": [180],
        }
    }
    row = environment_row(game, forecast)
    assert row["venue_name"] == "Test Park"
    assert row["weather_condition"] == "Mainly clear"
    assert row["temperature_f"] == 80.0
    assert row["roof_type"] == "Open"
    assert row["weather_exposure"] == "Outdoor conditions expected"


def test_environment_context_is_explicitly_non_probability() -> None:
    game = {"game_id": "1", "home_win_probability": 0.6}
    row = {"game_id": "1", "venue_name": "Test Park"}
    enriched, coverage = attach_environment_context([game], [row], [])
    assert enriched[0]["home_win_probability"] == 0.6
    assert enriched[0]["environment_context"]["probability_impact"] == "context_only"
    assert coverage["coverage"] == 1.0
