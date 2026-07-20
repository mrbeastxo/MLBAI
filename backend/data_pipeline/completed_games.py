"""Collect completed MLB game results as model-ready CSV rows."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, fetch_schedule, parse_date

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_FIELDS = [
    "game_id",
    "official_date",
    "season",
    "game_type",
    "away_team_id",
    "away_team",
    "away_score",
    "home_team_id",
    "home_team",
    "home_score",
    "winner_team_id",
    "winner_team",
    "venue_id",
    "venue",
    "status",
]


def completed_game_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract final MLB games, ignoring scheduled or postponed games."""
    rows: list[dict[str, Any]] = []

    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {})
            if status.get("abstractGameState") != "Final":
                continue

            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})
            away_team = away.get("team", {})
            home_team = home.get("team", {})
            venue = game.get("venue", {})

            winner = away_team if away.get("isWinner") else home_team
            rows.append(
                {
                    "game_id": game.get("gamePk"),
                    "official_date": game.get("officialDate", date_entry.get("date")),
                    "season": game.get("season"),
                    "game_type": game.get("gameType"),
                    "away_team_id": away_team.get("id"),
                    "away_team": away_team.get("name"),
                    "away_score": away.get("score"),
                    "home_team_id": home_team.get("id"),
                    "home_team": home_team.get("name"),
                    "home_score": home.get("score"),
                    "winner_team_id": winner.get("id"),
                    "winner_team": winner.get("name"),
                    "venue_id": venue.get("id"),
                    "venue": venue.get("name"),
                    "status": status.get("detailedState"),
                }
            )

    return rows


def collect_date_range(start_date: date, end_date: date) -> list[dict[str, Any]]:
    """Fetch and extract completed games for an inclusive date range."""
    rows: list[dict[str, Any]] = []
    current_date = start_date
    while current_date <= end_date:
        rows.extend(completed_game_rows(fetch_schedule(current_date)))
        current_date += timedelta(days=1)
    return rows


def save_results(rows: list[dict[str, Any]], start_date: date, end_date: date) -> Path:
    """Write completed games to a CSV file and return its path."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / (
        f"completed_games_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    )
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", type=parse_date)
    args = parser.parse_args()
    end_date = args.end_date or args.start_date

    if end_date < args.start_date:
        parser.error("--end-date cannot be earlier than --start-date")

    try:
        rows = collect_date_range(args.start_date, end_date)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not download completed games: {exc}") from exc

    output_path = save_results(rows, args.start_date, end_date)
    print(
        f"Collected {len(rows)} completed MLB games from "
        f"{args.start_date.isoformat()} through {end_date.isoformat()}."
    )
    for row in rows:
        print(
            f"- {row['away_team']} {row['away_score']} at "
            f"{row['home_team']} {row['home_score']} | Winner: {row['winner_team']}"
        )
    print(f"Clean results saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
