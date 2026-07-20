"""Collect and attach live starting-pitcher and bullpen context."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.bullpen_workload import (
    collect_bullpen_snapshot,
    save_bullpen_snapshot,
)
from backend.data_pipeline.starting_pitchers import (
    PROCESSED_DATA_DIR,
    collect_starting_pitchers,
    save_starting_pitchers,
)


def _cached_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def collect_context_snapshot(
    season: int, game_date: date
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Collect each source independently so one outage cannot block predictions."""
    pitcher_rows: list[dict[str, Any]] = []
    bullpen_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        pitcher_rows = collect_starting_pitchers(season, game_date)
        save_starting_pitchers(pitcher_rows, game_date)
    except requests.RequestException as error:
        errors.append(f"starting_pitchers: {error}")
        pitcher_rows = _cached_rows(
            PROCESSED_DATA_DIR / f"starting_pitchers_{game_date.isoformat()}.csv"
        )
    try:
        bullpen_rows = collect_bullpen_snapshot(season, game_date)
        save_bullpen_snapshot(bullpen_rows, game_date)
    except requests.RequestException as error:
        errors.append(f"bullpen_workload: {error}")
        bullpen_rows = _cached_rows(
            PROCESSED_DATA_DIR / f"bullpen_workload_{game_date.isoformat()}.csv"
        )
    return pitcher_rows, bullpen_rows, errors


def starter_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"announced": False}
    return {
        "announced": True,
        "name": row.get("pitcher_name"),
        "throws": row.get("throws"),
        "era": row.get("season_era"),
        "whip": row.get("season_whip"),
        "strikeouts_per_9": row.get("strikeouts_per_9"),
        "walks_per_9": row.get("walks_per_9"),
        "days_rest": row.get("days_rest"),
        "pitches_last_3": row.get("pitches_last_3"),
    }


def bullpen_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"available": False}
    return {
        "available": True,
        "pitches_last_3_days": row.get("bullpen_pitches_last_3_days"),
        "relievers_back_to_back": row.get("relievers_back_to_back"),
        "high_workload_relievers": row.get("high_workload_relievers"),
        "workload_index": row.get("workload_index"),
    }


def attach_matchup_context(
    analyses: list[dict[str, Any]],
    pitcher_rows: list[dict[str, Any]],
    bullpen_rows: list[dict[str, Any]],
    errors: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pitchers = {
        (str(row.get("game_id")), row.get("side")): row for row in pitcher_rows
    }
    bullpens = {row.get("team_name"): row for row in bullpen_rows}
    announced = available_bullpens = 0
    output: list[dict[str, Any]] = []
    for game in analyses:
        game_id = str(game["game_id"])
        away_starter = starter_summary(pitchers.get((game_id, "away")))
        home_starter = starter_summary(pitchers.get((game_id, "home")))
        away_bullpen = bullpen_summary(bullpens.get(game["away_team"]))
        home_bullpen = bullpen_summary(bullpens.get(game["home_team"]))
        announced += int(away_starter["announced"]) + int(home_starter["announced"])
        available_bullpens += int(away_bullpen["available"]) + int(home_bullpen["available"])
        output.append(
            {
                **game,
                "matchup_context": {
                    "probability_impact": "context_only",
                    "away_starter": away_starter,
                    "home_starter": home_starter,
                    "away_bullpen": away_bullpen,
                    "home_bullpen": home_bullpen,
                    "collection_errors": errors or [],
                    "note": "Displayed for analysis; not used in win probability until historical validation passes.",
                },
            }
        )
    possible = len(analyses) * 2
    coverage = {
        "possible_team_sides": possible,
        "announced_starters": announced,
        "starter_coverage": round(announced / possible, 4) if possible else 0.0,
        "available_bullpens": available_bullpens,
        "bullpen_coverage": round(available_bullpens / possible, 4) if possible else 0.0,
        "errors": errors or [],
    }
    return output, coverage
