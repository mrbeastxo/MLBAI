"""Collect probable MLB starters with season performance and recent workload."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import (
    PROJECT_ROOT,
    SCHEDULE_URL,
    parse_date,
)

PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PITCHER_FIELDS = [
    "snapshot_date",
    "game_id",
    "game_time_utc",
    "side",
    "team_id",
    "team_name",
    "opponent_id",
    "opponent_name",
    "pitcher_id",
    "pitcher_name",
    "throws",
    "age",
    "season_games_started",
    "season_innings_pitched",
    "season_wins",
    "season_losses",
    "season_era",
    "season_whip",
    "season_strikeouts",
    "season_walks",
    "season_home_runs_allowed",
    "strikeouts_per_9",
    "walks_per_9",
    "home_runs_per_9",
    "last_appearance_date",
    "days_rest",
    "pitches_last_appearance",
    "pitches_last_3",
    "innings_last_3",
]


def fetch_probable_pitchers(game_date: date, timeout: int = 20) -> dict[str, Any]:
    """Return one day's schedule with probable-pitcher information."""
    response = requests.get(
        SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": game_date.strftime("%m/%d/%Y"),
            "hydrate": "probablePitcher",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def probable_pitcher_assignments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten announced probable starters into one assignment per team."""
    assignments: list[dict[str, Any]] = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            for side, opponent_side in (("away", "home"), ("home", "away")):
                team_entry = teams.get(side, {})
                pitcher = team_entry.get("probablePitcher")
                if not pitcher:
                    continue
                team = team_entry.get("team", {})
                opponent = teams.get(opponent_side, {}).get("team", {})
                assignments.append(
                    {
                        "game_id": game.get("gamePk"),
                        "game_time_utc": game.get("gameDate"),
                        "side": side,
                        "team_id": team.get("id"),
                        "team_name": team.get("name"),
                        "opponent_id": opponent.get("id"),
                        "opponent_name": opponent.get("name"),
                        "pitcher_id": pitcher.get("id"),
                        "pitcher_name": pitcher.get("fullName"),
                    }
                )
    return assignments


def fetch_pitcher_profile(pitcher_id: int, season: int, timeout: int = 20) -> dict[str, Any]:
    """Fetch biography, season totals, and game logs in one request."""
    hydrate = f"stats(group=[pitching],type=[season,gameLog],season={season})"
    response = requests.get(
        f"{PEOPLE_URL}/{pitcher_id}", params={"hydrate": hydrate}, timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def _stat_blocks(person: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        block.get("type", {}).get("displayName", ""): block.get("splits", [])
        for block in person.get("stats", [])
    }


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _innings_from_outs(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def pitcher_row(
    assignment: dict[str, Any], profile_payload: dict[str, Any], snapshot_date: date
) -> dict[str, Any]:
    """Combine an assignment with season stats and pregame workload."""
    person = profile_payload.get("people", [{}])[0]
    blocks = _stat_blocks(person)
    season_splits = blocks.get("season", [])
    season_stats = season_splits[0].get("stat", {}) if season_splits else {}

    previous_appearances = sorted(
        (
            split
            for split in blocks.get("gameLog", [])
            if split.get("date") and date.fromisoformat(split["date"]) < snapshot_date
        ),
        key=lambda split: split["date"],
        reverse=True,
    )
    recent_three = previous_appearances[:3]
    last = recent_three[0] if recent_three else {}
    last_stats = last.get("stat", {})
    last_date = date.fromisoformat(last["date"]) if last.get("date") else None

    row = {
        **assignment,
        "snapshot_date": snapshot_date.isoformat(),
        "pitcher_name": person.get("fullName", assignment.get("pitcher_name")),
        "throws": person.get("pitchHand", {}).get("code"),
        "age": person.get("currentAge"),
        "season_games_started": season_stats.get("gamesStarted"),
        "season_innings_pitched": season_stats.get("inningsPitched"),
        "season_wins": season_stats.get("wins"),
        "season_losses": season_stats.get("losses"),
        "season_era": season_stats.get("era"),
        "season_whip": season_stats.get("whip"),
        "season_strikeouts": season_stats.get("strikeOuts"),
        "season_walks": season_stats.get("baseOnBalls"),
        "season_home_runs_allowed": season_stats.get("homeRuns"),
        "strikeouts_per_9": season_stats.get("strikeoutsPer9Inn"),
        "walks_per_9": season_stats.get("walksPer9Inn"),
        "home_runs_per_9": season_stats.get("homeRunsPer9"),
        "last_appearance_date": last.get("date"),
        "days_rest": (snapshot_date - last_date).days if last_date else None,
        "pitches_last_appearance": last_stats.get("numberOfPitches"),
        "pitches_last_3": sum(
            _integer(split.get("stat", {}).get("numberOfPitches"))
            for split in recent_three
        ),
        "innings_last_3": _innings_from_outs(
            sum(_integer(split.get("stat", {}).get("outs")) for split in recent_three)
        ),
    }
    return {field: row.get(field) for field in PITCHER_FIELDS}


def collect_starting_pitchers(season: int, game_date: date) -> list[dict[str, Any]]:
    """Collect announced starters concurrently, sorted by game and side."""
    assignments = probable_pitcher_assignments(fetch_probable_pitchers(game_date))
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_assignment = {
            executor.submit(fetch_pitcher_profile, item["pitcher_id"], season): item
            for item in assignments
        }
        for future in as_completed(future_to_assignment):
            assignment = future_to_assignment[future]
            rows.append(pitcher_row(assignment, future.result(), game_date))
    side_order = {"away": 0, "home": 1}
    return sorted(rows, key=lambda row: (row["game_id"], side_order[row["side"]]))


def save_starting_pitchers(rows: list[dict[str, Any]], game_date: date) -> Path:
    """Write the starting-pitcher snapshot to CSV and return its path."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"starting_pitchers_{game_date.isoformat()}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PITCHER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--date", type=parse_date, default=date.today())
    args = parser.parse_args()
    if args.date.year != args.season:
        parser.error("--date must fall within --season")

    try:
        rows = collect_starting_pitchers(args.season, args.date)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not download starting-pitcher data: {exc}") from exc

    output_path = save_starting_pitchers(rows, args.date)
    print(f"Collected {len(rows)} announced starters for {args.date.isoformat()}.")
    for row in rows:
        print(
            f"- {row['pitcher_name']} ({row['team_name']}): "
            f"ERA {row['season_era']}, WHIP {row['season_whip']}, "
            f"rest {row['days_rest']} days"
        )
    print(f"Pitcher snapshot saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
