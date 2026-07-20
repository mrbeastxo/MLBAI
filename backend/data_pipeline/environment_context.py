"""Collect ballpark details and first-pitch weather forecasts."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, SCHEDULE_URL, parse_date

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ENVIRONMENT_FIELDS = [
    "snapshot_date",
    "game_id",
    "game_time_utc",
    "venue_id",
    "venue_name",
    "city",
    "roof_type",
    "turf_type",
    "weather_exposure",
    "temperature_f",
    "weather_condition",
    "precipitation_probability",
    "precipitation_inches",
    "wind_speed_mph",
    "wind_gust_mph",
    "wind_direction_degrees",
    "forecast_hour_utc",
]

WMO_CONDITIONS = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    80: "Light showers",
    81: "Showers",
    82: "Heavy showers",
    95: "Thunderstorms",
    96: "Thunderstorms with hail",
    99: "Severe thunderstorms with hail",
}


def weather_exposure(roof_type: str | None) -> str:
    roof = (roof_type or "").casefold()
    if roof == "open":
        return "Outdoor conditions expected"
    if "retractable" in roof:
        return "Depends on game-time roof decision"
    if roof:
        return "Limited outdoor weather exposure"
    return "Roof exposure unknown"


def fetch_games_with_venues(game_date: date) -> list[dict[str, Any]]:
    response = requests.get(
        SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": game_date.strftime("%m/%d/%Y"),
            "hydrate": "venue(location,fieldInfo)",
        },
        timeout=30,
    )
    response.raise_for_status()
    return [
        game
        for date_entry in response.json().get("dates", [])
        for game in date_entry.get("games", [])
    ]


def fetch_hourly_forecast(
    latitude: float, longitude: float, forecast_date: date
) -> dict[str, Any]:
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": forecast_date.isoformat(),
            "end_date": forecast_date.isoformat(),
            "hourly": (
                "temperature_2m,precipitation_probability,precipitation,weather_code,"
                "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def nearest_hour_index(times: list[str], game_time_utc: str) -> int:
    target = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00")).astimezone(UTC)
    parsed = [datetime.fromisoformat(value).replace(tzinfo=UTC) for value in times]
    return min(range(len(parsed)), key=lambda index: abs(parsed[index] - target))


def environment_row(game: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any]:
    venue = game.get("venue", {})
    field = venue.get("fieldInfo", {})
    location = venue.get("location", {})
    hourly = forecast.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise ValueError("Weather forecast contains no hourly values")
    index = nearest_hour_index(times, game["gameDate"])

    def value(name: str) -> Any:
        values = hourly.get(name, [])
        return values[index] if index < len(values) else None

    weather_code = value("weather_code")
    return {
        "snapshot_date": game.get("officialDate"),
        "game_id": str(game.get("gamePk")),
        "game_time_utc": game.get("gameDate"),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "city": location.get("city"),
        "roof_type": field.get("roofType"),
        "turf_type": field.get("turfType"),
        "weather_exposure": weather_exposure(field.get("roofType")),
        "temperature_f": value("temperature_2m"),
        "weather_condition": WMO_CONDITIONS.get(weather_code, f"Weather code {weather_code}"),
        "precipitation_probability": value("precipitation_probability"),
        "precipitation_inches": value("precipitation"),
        "wind_speed_mph": value("wind_speed_10m"),
        "wind_gust_mph": value("wind_gusts_10m"),
        "wind_direction_degrees": value("wind_direction_10m"),
        "forecast_hour_utc": times[index],
    }


def collect_environment_snapshot(
    game_date: date,
) -> tuple[list[dict[str, Any]], list[str]]:
    games = fetch_games_with_venues(game_date)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    def collect(game: dict[str, Any]) -> dict[str, Any]:
        coordinates = game.get("venue", {}).get("location", {}).get("defaultCoordinates", {})
        latitude, longitude = coordinates.get("latitude"), coordinates.get("longitude")
        if latitude is None or longitude is None:
            raise ValueError("Venue coordinates unavailable")
        game_time = datetime.fromisoformat(game["gameDate"].replace("Z", "+00:00"))
        forecast = fetch_hourly_forecast(latitude, longitude, game_time.date())
        return environment_row(game, forecast)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(collect, game): game for game in games}
        for future in as_completed(futures):
            game = futures[future]
            try:
                rows.append(future.result())
            except (requests.RequestException, ValueError) as error:
                errors.append(f"game {game.get('gamePk')}: {error}")
    return sorted(rows, key=lambda row: (row["game_time_utc"], row["game_id"])), errors


def save_environment_snapshot(rows: list[dict[str, Any]], game_date: date) -> Path:
    path = PROCESSED_DATA_DIR / f"environment_context_{game_date.isoformat()}.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ENVIRONMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_environment_snapshot(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def collect_environment_with_cache(
    game_date: date,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Refresh forecasts, falling back only to a same-date local snapshot."""
    path = PROCESSED_DATA_DIR / f"environment_context_{game_date.isoformat()}.csv"
    try:
        rows, errors = collect_environment_snapshot(game_date)
        if rows:
            save_environment_snapshot(rows, game_date)
        return rows or read_environment_snapshot(path), errors
    except requests.RequestException as error:
        return read_environment_snapshot(path), [f"environment: {error}"]


def attach_environment_context(
    analyses: list[dict[str, Any]], rows: list[dict[str, Any]], errors: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_game = {str(row["game_id"]): row for row in rows}
    output = []
    for game in analyses:
        context = by_game.get(str(game["game_id"]))
        output.append(
            {
                **game,
                "environment_context": (
                    {
                        **context,
                        "weather_exposure": context.get("weather_exposure")
                        or weather_exposure(context.get("roof_type")),
                        "probability_impact": "context_only",
                        "note": "Weather was tested on 9,719 historical games but worsened holdout win-probability quality, so it remains context-only.",
                    }
                    if context
                    else None
                ),
            }
        )
    return output, {
        "scheduled_games": len(analyses),
        "forecast_games": len(rows),
        "coverage": round(len(rows) / len(analyses), 4) if analyses else 0.0,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, default=date.today())
    args = parser.parse_args()
    rows, errors = collect_environment_snapshot(args.date)
    path = save_environment_snapshot(rows, args.date)
    print(f"Environment forecasts: {len(rows)}; errors: {len(errors)}")
    print(f"Saved to: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
