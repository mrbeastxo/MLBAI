"""Archive a season's official results with honest MLBAI tracking labels."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.completed_games import completed_game_rows
from backend.data_pipeline.historical_training import fetch_season_games
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, parse_date
from backend.tracking.prediction_tracker import DEFAULT_DATABASE, connect_database

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def ledger_comparisons(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT p.game_id, p.model_lean, p.away_win_probability,
               p.home_win_probability, r.game_id AS settled_game_id
        FROM predictions p LEFT JOIN results r USING (game_id)
        """
    ).fetchall()
    return {
        str(row["game_id"]): {
            "mlbai_tracked": True,
            "mlbai_verified": row["settled_game_id"] is not None,
            "mlbai_lean": row["model_lean"],
            "mlbai_probability": round(
                max(
                    float(row["home_win_probability"]),
                    float(row["away_win_probability"]),
                ),
                4,
            ),
        }
        for row in rows
    }


def build_season_results(
    payload: dict[str, Any], connection: sqlite3.Connection
) -> list[dict[str, Any]]:
    comparisons = ledger_comparisons(connection)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in completed_game_rows(payload):
        if row["game_type"] != "R":
            continue
        if row["away_score"] is None or row["home_score"] is None:
            continue
        game_id = str(row["game_id"])
        if game_id in seen:
            continue
        seen.add(game_id)
        comparison = comparisons.get(
            game_id,
            {
                "mlbai_tracked": False,
                "mlbai_verified": False,
                "mlbai_lean": None,
                "mlbai_probability": None,
            },
        )
        results.append(
            {
                **row,
                **comparison,
                "mlbai_correct": (
                    comparison["mlbai_lean"] == row["winner_team"]
                    if comparison["mlbai_verified"]
                    else None
                ),
            }
        )
    return sorted(
        results,
        key=lambda item: (item["official_date"], int(item["game_id"])),
        reverse=True,
    )


def save_season_results(
    rows: list[dict[str, Any]], season: int, output_dir: Path = PROCESSED_DATA_DIR
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"season_results_{season}.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--through-date", type=parse_date, default=date.today())
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    if args.through_date.year != args.season:
        parser.error("--through-date must be inside --season")
    try:
        payload = fetch_season_games(args.season, args.through_date)
    except requests.RequestException as error:
        raise SystemExit(f"Could not download season results: {error}") from error
    connection = connect_database(args.database)
    try:
        rows = build_season_results(payload, connection)
    finally:
        connection.close()
    path = save_season_results(rows, args.season)
    verified = sum(row["mlbai_verified"] for row in rows)
    print(f"Archived {len(rows)} completed regular-season games.")
    print(f"Verified MLBAI comparisons: {verified}")
    print(f"Season results saved to: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
