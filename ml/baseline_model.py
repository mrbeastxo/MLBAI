"""Train and evaluate MLBAI's first chronological logistic-regression model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "docs"
FEATURES = [
    "win_percentage_home_minus_away",
    "runs_per_game_home_minus_away",
    "runs_allowed_per_game_home_minus_away",
    "run_differential_per_game_home_minus_away",
    "last_10_win_percentage_home_minus_away",
    "last_10_run_differential_home_minus_away",
    "venue_win_percentage_home_minus_away",
    "schedule_load_home_minus_away",
]
ADVANCED_FEATURES = FEATURES + [
    "last_30_win_percentage_home_minus_away",
    "last_30_run_differential_home_minus_away",
    "pythagorean_expectation_home_minus_away",
    "streak_home_minus_away",
    "elo_rating_home_minus_away",
]
PITCHER_FEATURES = ADVANCED_FEATURES + [
    "starter_era_home_minus_away",
    "starter_whip_home_minus_away",
    "starter_k9_home_minus_away",
    "starter_bb9_home_minus_away",
    "starter_last_5_era_home_minus_away",
    "away_starter_history_missing",
    "home_starter_history_missing",
]
COMBINED_FEATURES = PITCHER_FEATURES + [
    "bullpen_era_home_minus_away",
    "bullpen_whip_home_minus_away",
    "bullpen_k9_home_minus_away",
    "bullpen_pitches_last_3_home_minus_away",
    "bullpen_back_to_back_home_minus_away",
    "away_bullpen_history_missing",
    "home_bullpen_history_missing",
]


def load_training_rows(path: Path) -> list[dict[str, str]]:
    """Read historical rows and sort them chronologically."""
    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return sorted(rows, key=lambda row: (row["official_date"], row["game_time_utc"]))


def chronological_split(
    rows: list[dict[str, str]], train_fraction: float = 0.8
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    """Split by whole dates so no date appears in both sets."""
    unique_dates = sorted({row["official_date"] for row in rows})
    if len(unique_dates) < 5:
        raise ValueError("At least five distinct game dates are required")
    split_index = max(
        1, min(len(unique_dates) - 1, int(len(unique_dates) * train_fraction))
    )
    test_start_date = unique_dates[split_index]
    train_rows = [row for row in rows if row["official_date"] < test_start_date]
    test_rows = [row for row in rows if row["official_date"] >= test_start_date]
    return train_rows, test_rows, test_start_date


def feature_matrix(
    rows: list[dict[str, str]], features: list[str] = FEATURES
) -> np.ndarray:
    """Convert approved pregame feature columns into a numeric matrix."""
    return np.asarray(
        [
            [float(row[field]) if row.get(field, "") != "" else np.nan for field in features]
            for row in rows
        ],
        dtype=float,
    )


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["home_win"]) for row in rows], dtype=int)


def build_pipeline() -> Pipeline:
    """Return an interpretable, reproducible baseline model pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000, random_state=42)),
        ]
    )


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "log_loss": round(float(log_loss(y_true, probabilities, labels=[0, 1])), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
    }


def calibration_bins(
    y_true: np.ndarray, probabilities: np.ndarray, bin_count: int = 5
) -> list[dict[str, Any]]:
    """Summarize predicted versus observed home-win rates in fixed bins."""
    results: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for index in range(bin_count):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bin_count - 1 else probabilities < upper
        )
        if not np.any(mask):
            continue
        results.append(
            {
                "range": f"{lower:.1f}-{upper:.1f}",
                "games": int(mask.sum()),
                "mean_prediction": round(float(probabilities[mask].mean()), 4),
                "observed_home_win_rate": round(float(y_true[mask].mean()), 4),
            }
        )
    return results


def train_and_evaluate(
    rows: list[dict[str, str]], train_fraction: float = 0.8
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit on the past, evaluate on the future, and return a report."""
    train_rows, test_rows, test_start_date = chronological_split(rows, train_fraction)
    x_train, y_train = feature_matrix(train_rows), labels(train_rows)
    x_test, y_test = feature_matrix(test_rows), labels(test_rows)

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    model_probabilities = pipeline.predict_proba(x_test)[:, 1]

    baseline_probability = float(y_train.mean())
    baseline_probabilities = np.full(len(y_test), baseline_probability)
    coefficients = pipeline.named_steps["model"].coef_[0]
    ranked_coefficients = sorted(
        (
            {"feature": feature, "coefficient": round(float(coefficient), 4)}
            for feature, coefficient in zip(FEATURES, coefficients, strict=True)
        ),
        key=lambda item: abs(item["coefficient"]),
        reverse=True,
    )

    report = {
        "model": "logistic_regression",
        "split_method": "chronological_by_whole_date",
        "train_games": len(train_rows),
        "test_games": len(test_rows),
        "train_start_date": train_rows[0]["official_date"],
        "train_end_date": train_rows[-1]["official_date"],
        "test_start_date": test_start_date,
        "test_end_date": test_rows[-1]["official_date"],
        "features": FEATURES,
        "model_metrics": probability_metrics(y_test, model_probabilities),
        "home_rate_baseline_probability": round(baseline_probability, 4),
        "home_rate_baseline_metrics": probability_metrics(y_test, baseline_probabilities),
        "calibration": calibration_bins(y_test, model_probabilities),
        "standardized_coefficients": ranked_coefficients,
    }
    return pipeline, report


def save_artifacts(pipeline: Pipeline, report: dict[str, Any]) -> tuple[Path, Path]:
    """Save the fitted pipeline and its evaluation report."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "baseline_logistic.joblib"
    report_path = REPORT_DIR / "baseline_model_report.json"
    joblib.dump({"pipeline": pipeline, "features": FEATURES}, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction < 1.0:
        parser.error("--train-fraction must be at least 0.5 and below 1.0")

    rows = load_training_rows(args.data)
    if len(rows) < 100:
        raise SystemExit("At least 100 historical games are required for this baseline")
    pipeline, report = train_and_evaluate(rows, args.train_fraction)
    model_path, report_path = save_artifacts(pipeline, report)

    model = report["model_metrics"]
    baseline = report["home_rate_baseline_metrics"]
    print(f"Training games: {report['train_games']}")
    print(f"Held-out games: {report['test_games']} starting {report['test_start_date']}")
    print(
        f"Logistic regression | accuracy {model['accuracy']:.3f} | "
        f"log loss {model['log_loss']:.3f} | Brier {model['brier_score']:.3f}"
    )
    print(
        f"Home-rate baseline  | accuracy {baseline['accuracy']:.3f} | "
        f"log loss {baseline['log_loss']:.3f} | Brier {baseline['brier_score']:.3f}"
    )
    print(f"Model saved to: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Report saved to: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
