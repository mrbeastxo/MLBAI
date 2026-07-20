"""Backfill compact, leakage-safe starter and bullpen training data."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.historical_bullpens import (
    build_bullpen_feature_rows,
    relief_lines,
)
from backend.data_pipeline.historical_pitchers import (
    BOX_SCORE_DIR,
    build_pitcher_feature_rows,
    completed_game_ids,
    fetch_boxscore,
    starter_appearances,
)
from backend.data_pipeline.historical_training import completed_regular_games
from backend.data_pipeline.join_bullpen_features import join_bullpen_features
from backend.data_pipeline.join_pitcher_features import join_pitcher_features
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
COMPACT_DIR = RAW_DATA_DIR / "pitching_context"


def compact_boxscore(
    boxscore: dict[str, Any], game_id: int, game_date: date
) -> dict[str, Any]:
    """Keep only the pitching fields needed for future feature rebuilds."""
    teams = boxscore.get("teams", {})
    return {
        "game_id": str(game_id),
        "official_date": game_date.isoformat(),
        "away_team_id": str(teams.get("away", {}).get("team", {}).get("id")),
        "home_team_id": str(teams.get("home", {}).get("team", {}).get("id")),
        "starters": starter_appearances(boxscore, game_id, game_date),
        "relievers": relief_lines(boxscore, str(game_id), game_date),
    }


def _fetch_compact(game_id: int, game_date: date) -> dict[str, Any]:
    cached = BOX_SCORE_DIR / f"{game_id}.json"
    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
    else:
        error: requests.RequestException | None = None
        for attempt in range(4):
            try:
                payload = fetch_boxscore(game_id)
                break
            except requests.RequestException as exc:
                error = exc
                if attempt == 3:
                    raise
                time.sleep(0.5 * (2**attempt))
        else:  # pragma: no cover - loop either succeeds or raises
            assert error is not None
            raise error
    return compact_boxscore(payload, game_id, game_date)


def read_compact(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                records[record["game_id"]] = record
    return records


def write_compact(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records.values(), key=lambda item: (item["official_date"], item["game_id"])
    )
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in ordered),
        encoding="utf-8",
    )


def backfill_compact_season(
    season_payload: dict[str, Any], season: int, workers: int = 12
) -> tuple[Path, dict[str, int]]:
    """Download only missing games and persist a compact restartable cache."""
    games = completed_game_ids(
        season_payload, date(season, 1, 1), date(season, 12, 31)
    )
    path = COMPACT_DIR / f"pitching_context_{season}.jsonl"
    records = read_compact(path)
    missing = [(game_id, game_date) for game_id, game_date in games if str(game_id) not in records]
    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_compact, game_id, game_date): game_id
            for game_id, game_date in missing
        }
        for future in as_completed(futures):
            record = future.result()
            records[record["game_id"]] = record
            downloaded += 1
            if downloaded % 100 == 0:
                write_compact(path, records)
    write_compact(path, records)
    return path, {
        "scheduled_games": len(games),
        "reused_games": len(games) - len(missing),
        "downloaded_games": downloaded,
        "cached_games": len(records),
    }


def build_combined_rows(
    season_payload: dict[str, Any], compact_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join team, starter, and bullpen features on the same completed games."""
    from backend.data_pipeline.historical_training import build_training_rows

    games = completed_regular_games(season_payload)
    if not games:
        raise ValueError("Season payload contains no completed regular-season games")
    season = int(games[0]["season"])
    team_rows = build_training_rows(games, date(season, 1, 1), date(season, 12, 31))
    appearances = [item for record in compact_records for item in record["starters"]]
    pitcher_rows = build_pitcher_feature_rows(appearances)
    pitcher_joined, pitcher_report = join_pitcher_features(team_rows, pitcher_rows)
    bullpen_games = [
        {
            "game_id": record["game_id"],
            "official_date": date.fromisoformat(record["official_date"]),
            "away_team_id": record["away_team_id"],
            "home_team_id": record["home_team_id"],
            "relief_lines": record["relievers"],
        }
        for record in compact_records
    ]
    bullpen_rows = build_bullpen_feature_rows(bullpen_games)
    combined, bullpen_coverage = join_bullpen_features(pitcher_joined, bullpen_rows)
    return combined, {
        **pitcher_report,
        "bullpen_coverage": round(bullpen_coverage, 4),
        "combined_games": len(combined),
    }


def save_combined(rows: list[dict[str, Any]], season: int) -> Path:
    import csv

    path = PROCESSED_DATA_DIR / f"training_pitching_bullpen_{season}.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    for season in args.seasons:
        candidates = sorted(RAW_DATA_DIR.glob(f"season_games_{season}_through_*.json"))
        if not candidates:
            raise SystemExit(f"Missing season schedule cache for {season}")
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
        compact_path, download_report = backfill_compact_season(
            payload, season, args.workers
        )
        combined, coverage = build_combined_rows(
            payload, list(read_compact(compact_path).values())
        )
        output = save_combined(combined, season)
        print(f"{season}: {download_report}")
        print(f"{season}: {coverage}")
        print(f"{season}: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
