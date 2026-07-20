"""Join historical starter features to leakage-safe team training rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from backend.data_pipeline.historical_pitchers import PITCHER_FEATURE_FIELDS
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
JOIN_FIELDS = [
    field for field in PITCHER_FEATURE_FIELDS if field not in {"game_id", "official_date"}
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def join_pitcher_features(
    training_rows: list[dict[str, str]], pitcher_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join by unique game ID and report matchup/history coverage."""
    pitcher_by_game: dict[str, dict[str, str]] = {}
    for row in pitcher_rows:
        game_id = row["game_id"]
        if game_id in pitcher_by_game:
            raise ValueError(f"Duplicate pitcher feature row for game {game_id}")
        pitcher_by_game[game_id] = row

    if not pitcher_rows:
        raise ValueError("Pitcher feature file is empty")
    start_date = min(row["official_date"] for row in pitcher_rows)
    end_date = max(row["official_date"] for row in pitcher_rows)
    selected_training = [
        row for row in training_rows if start_date <= row["official_date"] <= end_date
    ]
    joined: list[dict[str, Any]] = []
    matched = 0
    complete_history = 0
    for training in selected_training:
        pitcher = pitcher_by_game.get(training["game_id"])
        output: dict[str, Any] = dict(training)
        if pitcher:
            matched += 1
            for field in JOIN_FIELDS:
                output[field] = pitcher.get(field, "")
            complete_history += int(
                str(pitcher.get("away_starter_history_missing")) == "0"
                and str(pitcher.get("home_starter_history_missing")) == "0"
            )
        else:
            for field in JOIN_FIELDS:
                output[field] = ""
            output["away_starter_history_missing"] = 1
            output["home_starter_history_missing"] = 1
        joined.append(output)

    report = {
        "start_date": start_date,
        "end_date": end_date,
        "training_games_in_range": len(selected_training),
        "matched_pitcher_games": matched,
        "matchup_coverage": round(matched / len(selected_training), 4)
        if selected_training
        else 0.0,
        "games_with_both_starter_histories": complete_history,
        "complete_history_coverage": round(complete_history / len(selected_training), 4)
        if selected_training
        else 0.0,
    }
    return joined, report


def save_joined(rows: list[dict[str, Any]], report: dict[str, Any]) -> Path:
    if not rows:
        raise ValueError("No training games overlap the pitcher feature range")
    path = PROCESSED_DATA_DIR / (
        f"training_with_pitchers_{report['start_date']}_{report['end_date']}.csv"
    )
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--pitchers", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows, report = join_pitcher_features(
            read_csv(args.training), read_csv(args.pitchers)
        )
        output_path = save_joined(rows, report)
    except ValueError as exc:
        raise SystemExit(f"Could not join pitcher features: {exc}") from exc
    print(f"Training games in range: {report['training_games_in_range']}")
    print(f"Pitcher matchup coverage: {report['matchup_coverage']:.1%}")
    print(f"Both-starter history coverage: {report['complete_history_coverage']:.1%}")
    print(f"Joined training data saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
