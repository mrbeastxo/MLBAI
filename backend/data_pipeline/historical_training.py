"""Build leakage-safe historical MLB training rows from prior game results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, SCHEDULE_URL, parse_date

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ROLLING_FIELDS = [
    "games_before",
    "wins_before",
    "win_percentage_before",
    "runs_per_game_before",
    "runs_allowed_per_game_before",
    "run_differential_per_game_before",
    "last_10_games",
    "last_10_win_percentage",
    "last_10_runs_per_game",
    "last_10_runs_allowed_per_game",
    "last_10_run_differential_per_game",
    "venue_games_before",
    "venue_win_percentage_before",
    "days_since_last_game",
    "games_previous_3_days",
]
DIFFERENCE_FIELDS = [
    "win_percentage_home_minus_away",
    "runs_per_game_home_minus_away",
    "runs_allowed_per_game_home_minus_away",
    "run_differential_per_game_home_minus_away",
    "last_10_win_percentage_home_minus_away",
    "last_10_run_differential_home_minus_away",
    "venue_win_percentage_home_minus_away",
    "schedule_load_home_minus_away",
]
TRAINING_FIELDS = (
    [
        "game_id",
        "official_date",
        "game_time_utc",
        "season",
        "away_team_id",
        "away_team_name",
        "home_team_id",
        "home_team_name",
    ]
    + [f"away_{field}" for field in ROLLING_FIELDS]
    + [f"home_{field}" for field in ROLLING_FIELDS]
    + DIFFERENCE_FIELDS
    + ["away_score", "home_score", "home_win"]
)


def fetch_season_games(season: int, end_date: date) -> dict[str, Any]:
    """Fetch regular-season games from January 1 through the requested end date."""
    response = requests.get(
        SCHEDULE_URL,
        params={
            "sportId": 1,
            "gameType": "R",
            "startDate": f"01/01/{season}",
            "endDate": end_date.strftime("%m/%d/%Y"),
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def completed_regular_games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten final regular-season games with the actual result."""
    games: list[dict[str, Any]] = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("gameType") != "R":
                continue
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})
            away_team = away.get("team", {})
            home_team = home.get("team", {})
            games.append(
                {
                    "game_id": str(game.get("gamePk")),
                    "official_date": game.get("officialDate", date_entry.get("date")),
                    "game_time_utc": game.get("gameDate"),
                    "season": str(game.get("season")),
                    "away_team_id": str(away_team.get("id")),
                    "away_team_name": away_team.get("name"),
                    "home_team_id": str(home_team.get("id")),
                    "home_team_name": home_team.get("name"),
                    "away_score": int(away.get("score", 0)),
                    "home_score": int(home.get("score", 0)),
                    "home_win": int(bool(home.get("isWinner"))),
                }
            )
    return sorted(games, key=lambda game: (game["official_date"], game["game_time_utc"]))


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize_history(
    history: list[dict[str, Any]], game_date: date, venue: str
) -> dict[str, Any]:
    """Summarize only games completed before ``game_date``."""
    prior = [game for game in history if game["date"] < game_date]
    last_10 = prior[-10:]
    venue_games = [game for game in prior if game["venue"] == venue]
    recent_cutoff = game_date - timedelta(days=3)
    recent_games = [game for game in prior if game["date"] >= recent_cutoff]

    def metrics(games: list[dict[str, Any]]) -> tuple[float, float, float, float]:
        return (
            _average([float(game["won"]) for game in games]),
            _average([game["runs_for"] for game in games]),
            _average([game["runs_against"] for game in games]),
            _average([game["runs_for"] - game["runs_against"] for game in games]),
        )

    win_pct, runs_for, runs_against, run_diff = metrics(prior)
    last_win_pct, last_runs, last_allowed, last_diff = metrics(last_10)
    last_date = prior[-1]["date"] if prior else None
    return {
        "games_before": len(prior),
        "wins_before": sum(game["won"] for game in prior),
        "win_percentage_before": win_pct,
        "runs_per_game_before": runs_for,
        "runs_allowed_per_game_before": runs_against,
        "run_differential_per_game_before": run_diff,
        "last_10_games": len(last_10),
        "last_10_win_percentage": last_win_pct,
        "last_10_runs_per_game": last_runs,
        "last_10_runs_allowed_per_game": last_allowed,
        "last_10_run_differential_per_game": last_diff,
        "venue_games_before": len(venue_games),
        "venue_win_percentage_before": _average(
            [float(game["won"]) for game in venue_games]
        ),
        "days_since_last_game": (game_date - last_date).days if last_date else "",
        "games_previous_3_days": len(recent_games),
    }


def _copy_prefixed(
    row: dict[str, Any], prefix: str, summary: dict[str, Any]
) -> None:
    for field in ROLLING_FIELDS:
        row[f"{prefix}_{field}"] = summary[field]


def _difference(home: Any, away: Any) -> float:
    return round(float(home) - float(away), 4)


def build_training_rows(
    games: list[dict[str, Any]], start_date: date, end_date: date
) -> list[dict[str, Any]]:
    """Build rows chronologically, updating history only after each whole date."""
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    games_by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        games_by_date[date.fromisoformat(game["official_date"])].append(game)

    output: list[dict[str, Any]] = []
    for game_date in sorted(games_by_date):
        day_games = games_by_date[game_date]

        # Build every row first. This conservative rule prevents a doubleheader's
        # first result from leaking into the second game's pregame features.
        if start_date <= game_date <= end_date:
            for game in day_games:
                away = summarize_history(histories[game["away_team_id"]], game_date, "away")
                home = summarize_history(histories[game["home_team_id"]], game_date, "home")
                row = {key: game[key] for key in TRAINING_FIELDS if key in game}
                _copy_prefixed(row, "away", away)
                _copy_prefixed(row, "home", home)
                row["win_percentage_home_minus_away"] = _difference(
                    home["win_percentage_before"], away["win_percentage_before"]
                )
                row["runs_per_game_home_minus_away"] = _difference(
                    home["runs_per_game_before"], away["runs_per_game_before"]
                )
                row["runs_allowed_per_game_home_minus_away"] = _difference(
                    home["runs_allowed_per_game_before"], away["runs_allowed_per_game_before"]
                )
                row["run_differential_per_game_home_minus_away"] = _difference(
                    home["run_differential_per_game_before"], away["run_differential_per_game_before"]
                )
                row["last_10_win_percentage_home_minus_away"] = _difference(
                    home["last_10_win_percentage"], away["last_10_win_percentage"]
                )
                row["last_10_run_differential_home_minus_away"] = _difference(
                    home["last_10_run_differential_per_game"],
                    away["last_10_run_differential_per_game"],
                )
                row["venue_win_percentage_home_minus_away"] = _difference(
                    home["venue_win_percentage_before"], away["venue_win_percentage_before"]
                )
                row["schedule_load_home_minus_away"] = _difference(
                    home["games_previous_3_days"], away["games_previous_3_days"]
                )
                output.append({field: row.get(field, "") for field in TRAINING_FIELDS})

        # Outcomes become available only after every row for this date is built.
        for game in day_games:
            histories[game["away_team_id"]].append(
                {
                    "date": game_date,
                    "won": 1 - game["home_win"],
                    "runs_for": game["away_score"],
                    "runs_against": game["home_score"],
                    "venue": "away",
                }
            )
            histories[game["home_team_id"]].append(
                {
                    "date": game_date,
                    "won": game["home_win"],
                    "runs_for": game["home_score"],
                    "runs_against": game["away_score"],
                    "venue": "home",
                }
            )
    return output


def save_training_rows(
    rows: list[dict[str, Any]], start_date: date, end_date: date
) -> Path:
    """Write historical features and labels to CSV."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / (
        f"training_games_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    )
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAINING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("--end-date cannot be earlier than --start-date")
    if args.start_date.year != args.end_date.year:
        parser.error("Milestone 7 supports one season per run")
    if args.end_date >= date.today():
        parser.error("--end-date must be before today so every label is final")

    try:
        payload = fetch_season_games(args.start_date.year, args.end_date)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not download historical games: {exc}") from exc

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / (
        f"season_games_{args.start_date.year}_through_{args.end_date.isoformat()}.json"
    )
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    games = completed_regular_games(payload)
    rows = build_training_rows(games, args.start_date, args.end_date)
    output_path = save_training_rows(rows, args.start_date, args.end_date)

    home_wins = sum(row["home_win"] for row in rows)
    home_win_rate = home_wins / len(rows) if rows else 0
    print(f"Built {len(rows)} leakage-safe historical game rows.")
    print(f"Columns per row: {len(TRAINING_FIELDS)}")
    print(f"Observed home win rate: {home_win_rate:.3f}")
    print(f"Training dataset saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
