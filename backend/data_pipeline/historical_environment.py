"""Build historical pregame weather and ballpark features from official MLB data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.historical_training import completed_regular_games
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, SCHEDULE_URL

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ENVIRONMENT_FEATURES = [
    "temperature_f",
    "wind_speed_mph",
    "wind_out_indicator",
    "wind_in_indicator",
    "weather_precipitation_indicator",
    "roof_closed_indicator",
    "environment_history_missing",
]


def fetch_historical_environment(season: int) -> dict[str, Any]:
    response = requests.get(
        SCHEDULE_URL,
        params={
            "sportId": 1,
            "season": season,
            "gameType": "R",
            "startDate": f"01/01/{season}",
            "endDate": f"12/31/{season}",
            "hydrate": "weather,venue(fieldInfo)",
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def _indicator(text: str, choices: tuple[str, ...]) -> int:
    lowered = text.casefold()
    return int(any(choice in lowered for choice in choices))


def environment_features(game: dict[str, Any]) -> dict[str, Any]:
    weather = game.get("weather") or {}
    condition = str(weather.get("condition") or "")
    wind = str(weather.get("wind") or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*mph", wind, re.IGNORECASE)
    field = (game.get("venue") or {}).get("fieldInfo") or {}
    roof = str(field.get("roofType") or "")
    temperature = weather.get("temp")
    missing = temperature in (None, "") and not wind and not condition
    return {
        "temperature_f": temperature if temperature not in (None, "") else "",
        "wind_speed_mph": float(match.group(1)) if match else "",
        "wind_out_indicator": _indicator(wind, ("out to", "out toward")),
        "wind_in_indicator": _indicator(wind, ("in from", "in toward")),
        "weather_precipitation_indicator": _indicator(
            condition, ("rain", "drizzle", "shower", "storm")
        ),
        "roof_closed_indicator": _indicator(roof, ("closed", "dome", "indoor")),
        "environment_history_missing": int(missing),
    }


def build_environment_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    games_by_id = {
        str(game.get("gamePk")): game
        for date_entry in payload.get("dates", [])
        for game in date_entry.get("games", [])
    }
    rows = []
    for game in completed_regular_games(payload):
        source = games_by_id.get(game["game_id"], {})
        rows.append(
            {
                "game_id": game["game_id"],
                "official_date": game["official_date"],
                "venue_id": str((source.get("venue") or {}).get("id") or ""),
                **environment_features(source),
            }
        )
    return rows


def save_payload(payload: dict[str, Any], season: int) -> Path:
    path = RAW_DATA_DIR / f"historical_environment_{season}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def save_rows(rows: list[dict[str, Any]], season: int) -> Path:
    path = PROCESSED_DATA_DIR / f"historical_environment_{season}.csv"
    fields = ["game_id", "official_date", "venue_id", *ENVIRONMENT_FEATURES]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def join_environment(
    training_rows: list[dict[str, Any]], environment_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    by_game = {str(row["game_id"]): row for row in environment_rows}
    output = []
    matched = 0
    for training in training_rows:
        context = by_game.get(str(training["game_id"]))
        row = dict(training)
        if context:
            matched += 1
            for field in ENVIRONMENT_FEATURES:
                row[field] = context.get(field, "")
        else:
            for field in ENVIRONMENT_FEATURES:
                row[field] = ""
            row["environment_history_missing"] = 1
        output.append(row)
    return output, round(matched / len(training_rows), 4) if training_rows else 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_joined(rows: list[dict[str, Any]], season: int) -> Path:
    path = PROCESSED_DATA_DIR / f"training_context_environment_{season}.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for season in args.seasons:
        payload = fetch_historical_environment(season)
        save_payload(payload, season)
        environment = build_environment_rows(payload)
        save_rows(environment, season)
        training = read_csv(PROCESSED_DATA_DIR / f"training_pitching_bullpen_{season}.csv")
        joined, coverage = join_environment(training, environment)
        output = write_joined(joined, season)
        print(f"{season}: {len(joined)} games, {coverage:.1%} weather coverage -> {output.name}")


if __name__ == "__main__":
    main()
