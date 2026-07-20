"""Join daily MLB snapshots into one model-ready row per scheduled game."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, parse_date

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TEAM_FEATURES = [
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
STARTER_FEATURES = [
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
BULLPEN_FEATURES = [
    "games_last_3_days",
    "relief_appearances_last_3_days",
    "relievers_used_last_3_days",
    "bullpen_pitches_yesterday",
    "bullpen_pitches_2_days_ago",
    "bullpen_pitches_3_days_ago",
    "bullpen_pitches_last_3_days",
    "bullpen_innings_last_3_days",
    "relievers_back_to_back",
    "relievers_used_2_of_3_days",
    "high_workload_relievers",
    "workload_index",
]
BASE_FIELDS = [
    "snapshot_date",
    "game_id",
    "game_time_utc",
    "status",
    "away_team_id",
    "away_team_name",
    "home_team_id",
    "home_team_name",
]
DIFFERENCE_FIELDS = [
    "winning_percentage_home_minus_away",
    "run_differential_home_minus_away",
    "ops_home_minus_away",
    "team_pitching_era_home_minus_away",
    "starter_era_home_minus_away",
    "bullpen_workload_home_minus_away",
]
QUALITY_FIELDS = ["away_starter_missing", "home_starter_missing"]
FEATURE_FIELDS = (
    BASE_FIELDS
    + [f"away_team_{field}" for field in TEAM_FEATURES]
    + [f"home_team_{field}" for field in TEAM_FEATURES]
    + [f"away_starter_{field}" for field in STARTER_FEATURES]
    + [f"home_starter_{field}" for field in STARTER_FEATURES]
    + [f"away_bullpen_{field}" for field in BULLPEN_FEATURES]
    + [f"home_bullpen_{field}" for field in BULLPEN_FEATURES]
    + DIFFERENCE_FIELDS
    + QUALITY_FIELDS
)


def load_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV snapshot into dictionaries."""
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def schedule_rows(payload: dict[str, Any], snapshot_date: date) -> list[dict[str, Any]]:
    """Flatten the schedule fields required by the feature table."""
    rows: list[dict[str, Any]] = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {}).get("team", {})
            home = teams.get("home", {}).get("team", {})
            rows.append(
                {
                    "snapshot_date": snapshot_date.isoformat(),
                    "game_id": str(game.get("gamePk")),
                    "game_time_utc": game.get("gameDate"),
                    "status": game.get("status", {}).get("detailedState"),
                    "away_team_id": str(away.get("id")),
                    "away_team_name": away.get("name"),
                    "home_team_id": str(home.get("id")),
                    "home_team_name": home.get("name"),
                }
            )
    return rows


def _copy_features(
    destination: dict[str, Any],
    source: dict[str, Any] | None,
    prefix: str,
    fields: list[str],
) -> None:
    source = source or {}
    for field in fields:
        destination[f"{prefix}_{field}"] = source.get(field, "")


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _difference(home: Any, away: Any) -> float | str:
    home_number = _number(home)
    away_number = _number(away)
    if home_number is None or away_number is None:
        return ""
    return round(home_number - away_number, 4)


def build_feature_rows(
    games: list[dict[str, Any]],
    team_rows: list[dict[str, Any]],
    starter_rows: list[dict[str, Any]],
    bullpen_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join sources by MLB IDs and return one stable row per game."""
    teams = {str(row["team_id"]): row for row in team_rows}
    starters = {
        (str(row["game_id"]), row["side"]): row for row in starter_rows
    }
    bullpens = {str(row["team_id"]): row for row in bullpen_rows}
    output: list[dict[str, Any]] = []

    for game in games:
        row = dict(game)
        away_team = teams.get(game["away_team_id"])
        home_team = teams.get(game["home_team_id"])
        away_starter = starters.get((game["game_id"], "away"))
        home_starter = starters.get((game["game_id"], "home"))
        away_bullpen = bullpens.get(game["away_team_id"])
        home_bullpen = bullpens.get(game["home_team_id"])

        if away_team is None or home_team is None:
            raise ValueError(f"Missing team statistics for game {game['game_id']}")
        if away_bullpen is None or home_bullpen is None:
            raise ValueError(f"Missing bullpen statistics for game {game['game_id']}")

        _copy_features(row, away_team, "away_team", TEAM_FEATURES)
        _copy_features(row, home_team, "home_team", TEAM_FEATURES)
        _copy_features(row, away_starter, "away_starter", STARTER_FEATURES)
        _copy_features(row, home_starter, "home_starter", STARTER_FEATURES)
        _copy_features(row, away_bullpen, "away_bullpen", BULLPEN_FEATURES)
        _copy_features(row, home_bullpen, "home_bullpen", BULLPEN_FEATURES)

        row["winning_percentage_home_minus_away"] = _difference(
            row["home_team_winning_percentage"], row["away_team_winning_percentage"]
        )
        row["run_differential_home_minus_away"] = _difference(
            row["home_team_run_differential"], row["away_team_run_differential"]
        )
        row["ops_home_minus_away"] = _difference(
            row["home_team_ops"], row["away_team_ops"]
        )
        row["team_pitching_era_home_minus_away"] = _difference(
            row["home_team_pitching_era"], row["away_team_pitching_era"]
        )
        row["starter_era_home_minus_away"] = _difference(
            row["home_starter_season_era"], row["away_starter_season_era"]
        )
        row["bullpen_workload_home_minus_away"] = _difference(
            row["home_bullpen_workload_index"], row["away_bullpen_workload_index"]
        )
        row["away_starter_missing"] = int(away_starter is None)
        row["home_starter_missing"] = int(home_starter is None)
        output.append({field: row.get(field, "") for field in FEATURE_FIELDS})
    return output


def source_paths(snapshot_date: date) -> dict[str, Path]:
    date_text = snapshot_date.isoformat()
    return {
        "schedule": RAW_DATA_DIR / f"schedule_{date_text}.json",
        "teams": PROCESSED_DATA_DIR / f"team_stats_{date_text}.csv",
        "starters": PROCESSED_DATA_DIR / f"starting_pitchers_{date_text}.csv",
        "bullpens": PROCESSED_DATA_DIR / f"bullpen_workload_{date_text}.csv",
    }


def create_daily_features(snapshot_date: date) -> tuple[list[dict[str, Any]], Path]:
    """Load, validate, join, and save all daily feature sources."""
    paths = source_paths(snapshot_date)
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source snapshots: " + ", ".join(missing))

    schedule_payload = json.loads(paths["schedule"].read_text(encoding="utf-8"))
    rows = build_feature_rows(
        schedule_rows(schedule_payload, snapshot_date),
        load_csv(paths["teams"]),
        load_csv(paths["starters"]),
        load_csv(paths["bullpens"]),
    )
    output_path = PROCESSED_DATA_DIR / f"daily_features_{snapshot_date.isoformat()}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, default=date.today())
    args = parser.parse_args()
    try:
        rows, output_path = create_daily_features(args.date)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Could not build daily features: {exc}") from exc

    missing_starters = sum(
        row["away_starter_missing"] + row["home_starter_missing"] for row in rows
    )
    print(f"Built {len(rows)} model-ready game rows for {args.date.isoformat()}.")
    print(f"Columns per game: {len(FEATURE_FIELDS)}")
    print(f"Missing announced starters: {missing_starters}")
    print(f"Daily feature table saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
