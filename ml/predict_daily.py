"""Generate daily MLB win probabilities from pregame feature rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import feature_matrix

MODEL_PATH = PROJECT_ROOT / "models" / "production_model.joblib"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTION_FIELDS = [
    "game_id",
    "official_date",
    "game_time_utc",
    "away_team",
    "home_team",
    "away_win_probability",
    "home_win_probability",
    "model_lean",
    "away_expected_runs",
    "home_expected_runs",
    "expected_total_runs",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def predict_rows(
    rows: list[dict[str, str]], artifact: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return probabilities without converting them into betting advice."""
    features = artifact["features"]
    probabilities = artifact["pipeline"].predict_proba(
        feature_matrix(rows, features)
    )[:, 1]
    output: list[dict[str, Any]] = []
    for row, home_probability in zip(rows, probabilities, strict=True):
        away_probability = 1.0 - float(home_probability)
        home_probability = float(home_probability)
        output.append(
            {
                "game_id": row["game_id"],
                "official_date": row["official_date"],
                "game_time_utc": row["game_time_utc"],
                "away_team": row["away_team_name"],
                "home_team": row["home_team_name"],
                "away_win_probability": round(away_probability, 4),
                "home_win_probability": round(home_probability, 4),
                "model_lean": (
                    row["home_team_name"]
                    if home_probability >= 0.5
                    else row["away_team_name"]
                ),
            }
        )
    return sorted(
        output,
        key=lambda item: max(
            item["home_win_probability"], item["away_win_probability"]
        ),
        reverse=True,
    )


def save_predictions(rows: list[dict[str, Any]], prediction_date: str) -> Path:
    path = PROCESSED_DATA_DIR / f"predictions_{prediction_date}.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    rows = read_rows(args.features)
    if not rows:
        raise SystemExit("No scheduled-game feature rows were found")
    artifact = joblib.load(args.model)
    predictions = predict_rows(rows, artifact)
    path = save_predictions(predictions, rows[0]["official_date"])

    print("Model estimates (not guarantees or betting advice):")
    for prediction in predictions:
        print(
            f"- {prediction['away_team']} at {prediction['home_team']}: "
            f"{prediction['away_win_probability']:.1%} / "
            f"{prediction['home_win_probability']:.1%} | "
            f"lean {prediction['model_lean']}"
        )
    print(f"Predictions saved to: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
