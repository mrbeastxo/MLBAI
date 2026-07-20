"""Reconstruct leakage-safe rolling team features for today's scheduled games."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.historical_training import (
    TRAINING_FIELDS,
    build_training_rows,
    completed_regular_games,
    fetch_season_games,
)
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, parse_date

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREGAME_FIELDS = [
    field for field in TRAINING_FIELDS if field not in {"away_score", "home_score", "home_win"}
]


def scheduled_regular_games(
    payload: dict[str, Any], prediction_date: date
) -> list[dict[str, Any]]:
    """Extract unfinished regular-season games on exactly the prediction date."""
    games: dict[str, dict[str, Any]] = {}
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            official_date = game.get("officialDate", date_entry.get("date"))
            if official_date != prediction_date.isoformat() or game.get("gameType") != "R":
                continue
            if game.get("status", {}).get("abstractGameState") == "Final":
                continue
            teams = game.get("teams", {})
            away = teams.get("away", {}).get("team", {})
            home = teams.get("home", {}).get("team", {})
            game_id = str(game.get("gamePk"))
            games[game_id] = {
                "game_id": game_id,
                "official_date": prediction_date.isoformat(),
                "game_time_utc": game.get("gameDate"),
                "season": str(game.get("season")),
                "away_team_id": str(away.get("id")),
                "away_team_name": away.get("name"),
                "home_team_id": str(home.get("id")),
                "home_team_name": home.get("name"),
                # Dummy values satisfy the historical builder but are added only
                # after today's rows exist, so they cannot affect any feature.
                "away_score": 0,
                "home_score": 0,
                "home_win": 0,
            }
    return sorted(games.values(), key=lambda game: game["game_time_utc"])


def build_pregame_rows(payload: dict[str, Any], prediction_date: date) -> list[dict[str, Any]]:
    """Build today's rows using completed games from earlier dates only."""
    history = [
        game
        for game in completed_regular_games(payload)
        if game["official_date"] < prediction_date.isoformat()
    ]
    scheduled = scheduled_regular_games(payload, prediction_date)
    rows = build_training_rows(
        history + scheduled, prediction_date, prediction_date
    )
    return [{field: row.get(field, "") for field in PREGAME_FIELDS} for row in rows]


def save_pregame_rows(rows: list[dict[str, Any]], prediction_date: date) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / f"pregame_features_{prediction_date.isoformat()}.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREGAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, default=date.today())
    args = parser.parse_args()
    try:
        payload = fetch_season_games(args.date.year, args.date)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not download season history: {exc}") from exc
    rows = build_pregame_rows(payload, args.date)
    path = save_pregame_rows(rows, args.date)
    print(f"Built leakage-safe pregame features for {len(rows)} scheduled games.")
    print(f"Pregame features saved to: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
