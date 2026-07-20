"""Join historical bullpen features to pitcher-enhanced training rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from backend.data_pipeline.historical_bullpens import BULLPEN_FEATURE_FIELDS
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
JOIN_FIELDS = [
    field for field in BULLPEN_FEATURE_FIELDS if field not in {"game_id", "official_date"}
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def join_bullpen_features(
    training_rows: list[dict[str, str]], bullpen_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], float]:
    bullpen_by_game = {row["game_id"]: row for row in bullpen_rows}
    if len(bullpen_by_game) != len(bullpen_rows):
        raise ValueError("Bullpen file contains duplicate game IDs")
    joined: list[dict[str, Any]] = []
    matched = 0
    for training in training_rows:
        bullpen = bullpen_by_game.get(training["game_id"])
        output: dict[str, Any] = dict(training)
        if bullpen:
            matched += 1
            for field in JOIN_FIELDS:
                output[field] = bullpen.get(field, "")
        else:
            for field in JOIN_FIELDS:
                output[field] = ""
            output["away_bullpen_history_missing"] = 1
            output["home_bullpen_history_missing"] = 1
        joined.append(output)
    coverage = matched / len(training_rows) if training_rows else 0.0
    return joined, coverage


def save_joined(rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("Training file is empty")
    start_date = min(row["official_date"] for row in rows)
    end_date = max(row["official_date"] for row in rows)
    path = PROCESSED_DATA_DIR / f"training_combined_{start_date}_{end_date}.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--bullpens", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows, coverage = join_bullpen_features(
            read_csv(args.training), read_csv(args.bullpens)
        )
        output_path = save_joined(rows)
    except ValueError as exc:
        raise SystemExit(f"Could not join bullpen features: {exc}") from exc
    print(f"Bullpen game coverage: {coverage:.1%}")
    print(f"Combined training data saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
