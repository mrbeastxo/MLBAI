"""Collect one season snapshot of MLB team records, hitting, and pitching."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, parse_date

API_BASE_URL = "https://statsapi.mlb.com/api/v1"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEAM_STAT_FIELDS = [
    "snapshot_date",
    "season",
    "team_id",
    "team_name",
    "division_id",
    "games_played",
    "wins",
    "losses",
    "winning_percentage",
    "runs_scored",
    "runs_allowed",
    "run_differential",
    "home_wins",
    "home_losses",
    "away_wins",
    "away_losses",
    "batting_average",
    "on_base_percentage",
    "slugging_percentage",
    "ops",
    "home_runs",
    "stolen_bases",
    "pitching_era",
    "pitching_whip",
    "pitching_strikeouts",
    "pitching_walks",
    "saves",
]


def fetch_json(path: str, params: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    """Request JSON from one official MLB Stats API endpoint."""
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_standings(season: int, snapshot_date: date) -> dict[str, Any]:
    """Return American and National League regular-season standings."""
    return fetch_json(
        "/standings",
        {
            "leagueId": "103,104",
            "season": season,
            "date": snapshot_date.strftime("%m/%d/%Y"),
            "standingsTypes": "regularSeason",
        },
    )


def fetch_team_stats(team_id: int, season: int) -> dict[str, Any]:
    """Return season hitting and pitching totals for one team."""
    return fetch_json(
        f"/teams/{team_id}/stats",
        {"stats": "season", "group": "hitting,pitching", "season": season},
    )


def standings_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten standings records into one partial row per MLB team."""
    rows: list[dict[str, Any]] = []
    for division in payload.get("records", []):
        division_id = division.get("division", {}).get("id")
        for record in division.get("teamRecords", []):
            team = record.get("team", {})
            splits = {
                item.get("type"): item
                for item in record.get("records", {}).get("splitRecords", [])
            }
            home = splits.get("home", {})
            away = splits.get("away", {})
            rows.append(
                {
                    "season": record.get("season"),
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "division_id": division_id,
                    "games_played": record.get("gamesPlayed"),
                    "wins": record.get("wins"),
                    "losses": record.get("losses"),
                    "winning_percentage": record.get("winningPercentage"),
                    "runs_scored": record.get("runsScored"),
                    "runs_allowed": record.get("runsAllowed"),
                    "run_differential": record.get("runDifferential"),
                    "home_wins": home.get("wins"),
                    "home_losses": home.get("losses"),
                    "away_wins": away.get("wins"),
                    "away_losses": away.get("losses"),
                }
            )
    return rows


def season_stat_groups(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a team's season-stat response by hitting and pitching group."""
    groups: dict[str, dict[str, Any]] = {}
    for block in payload.get("stats", []):
        group_name = block.get("group", {}).get("displayName")
        splits = block.get("splits", [])
        if group_name and splits:
            groups[group_name] = splits[0].get("stat", {})
    return groups


def combine_team_row(
    standing: dict[str, Any], stats_payload: dict[str, Any], snapshot_date: date
) -> dict[str, Any]:
    """Combine standings, hitting, and pitching into one stable row."""
    groups = season_stat_groups(stats_payload)
    hitting = groups.get("hitting", {})
    pitching = groups.get("pitching", {})
    row = {
        **standing,
        "snapshot_date": snapshot_date.isoformat(),
        "batting_average": hitting.get("avg"),
        "on_base_percentage": hitting.get("obp"),
        "slugging_percentage": hitting.get("slg"),
        "ops": hitting.get("ops"),
        "home_runs": hitting.get("homeRuns"),
        "stolen_bases": hitting.get("stolenBases"),
        "pitching_era": pitching.get("era"),
        "pitching_whip": pitching.get("whip"),
        "pitching_strikeouts": pitching.get("strikeOuts"),
        "pitching_walks": pitching.get("baseOnBalls"),
        "saves": pitching.get("saves"),
    }
    return {field: row.get(field) for field in TEAM_STAT_FIELDS}


def collect_team_snapshot(season: int, snapshot_date: date) -> list[dict[str, Any]]:
    """Collect all MLB teams concurrently and return rows sorted by name."""
    standings = standings_rows(fetch_standings(season, snapshot_date))
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_standing = {
            executor.submit(fetch_team_stats, standing["team_id"], season): standing
            for standing in standings
        }
        for future in as_completed(future_to_standing):
            standing = future_to_standing[future]
            rows.append(combine_team_row(standing, future.result(), snapshot_date))
    return sorted(rows, key=lambda row: str(row["team_name"]))


def save_team_snapshot(rows: list[dict[str, Any]], snapshot_date: date) -> Path:
    """Write a team snapshot to CSV and return its path."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"team_stats_{snapshot_date.isoformat()}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TEAM_STAT_FIELDS)
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
        rows = collect_team_snapshot(args.season, args.date)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not download MLB team statistics: {exc}") from exc

    output_path = save_team_snapshot(rows, args.date)
    print(f"Collected a {args.season} statistics snapshot for {len(rows)} MLB teams.")
    for row in rows:
        print(
            f"- {row['team_name']}: {row['wins']}-{row['losses']}, "
            f"OPS {row['ops']}, ERA {row['pitching_era']}"
        )
    print(f"Team snapshot saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
