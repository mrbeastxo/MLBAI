"""Download one day's MLB schedule from the official MLB Stats API."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def fetch_schedule(game_date: date, timeout: int = 20) -> dict[str, Any]:
    """Return the MLB schedule response for ``game_date``."""
    response = requests.get(
        SCHEDULE_URL,
        params={"sportId": 1, "date": game_date.strftime("%m/%d/%Y")},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_games(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Convert the nested API response into a compact list for display."""
    games: list[dict[str, str]] = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            games.append(
                {
                    "game_id": str(game.get("gamePk", "unknown")),
                    "away": teams.get("away", {}).get("team", {}).get("name", "TBD"),
                    "home": teams.get("home", {}).get("team", {}).get("name", "TBD"),
                    "start_time_utc": game.get("gameDate", "TBD"),
                    "status": game.get("status", {}).get("detailedState", "Unknown"),
                }
            )
    return games


def save_raw_schedule(payload: dict[str, Any], game_date: date) -> Path:
    """Save the untouched API payload and return its path."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"schedule_{game_date.isoformat()}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def parse_date(value: str) -> date:
    """Parse an ISO date for the command-line interface."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        type=parse_date,
        default=date.today(),
        help="schedule date in YYYY-MM-DD format (default: today)",
    )
    args = parser.parse_args()

    try:
        payload = fetch_schedule(args.date)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not download the MLB schedule: {exc}") from exc

    games = extract_games(payload)
    output_path = save_raw_schedule(payload, args.date)

    print(f"MLB schedule for {args.date.isoformat()} ({len(games)} games)")
    if games:
        for number, game in enumerate(games, start=1):
            print(
                f"{number:>2}. {game['away']} at {game['home']} | "
                f"{game['status']} | {game['start_time_utc']}"
            )
    else:
        print("No MLB games found for this date.")
    print(f"Raw response saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
