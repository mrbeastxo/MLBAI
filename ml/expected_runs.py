"""Train and audit chronological expected-runs models for MLB games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import ADVANCED_FEATURES, feature_matrix, load_training_rows
from ml.multiseason_validation import group_rows_by_season

MODEL_PATH = PROJECT_ROOT / "models" / "expected_runs.joblib"
REPORT_PATH = PROJECT_ROOT / "docs" / "expected_runs_validation_report.json"


def build_run_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=1.0, max_iter=2_000)),
        ]
    )


def run_labels(rows: list[dict[str, str]], side: str) -> np.ndarray:
    return np.asarray([float(row[f"{side}_score"]) for row in rows])


def score_metrics(
    away_actual: np.ndarray,
    home_actual: np.ndarray,
    away_prediction: np.ndarray,
    home_prediction: np.ndarray,
) -> dict[str, float]:
    actual = np.concatenate([away_actual, home_actual])
    prediction = np.clip(np.concatenate([away_prediction, home_prediction]), 0.01, None)
    return {
        "mae": round(float(mean_absolute_error(actual, prediction)), 4),
        "rmse": round(float(mean_squared_error(actual, prediction) ** 0.5), 4),
        "mean_poisson_deviance": round(float(mean_poisson_deviance(actual, prediction)), 4),
    }


def fit_pair(rows: list[dict[str, str]]) -> dict[str, Pipeline]:
    matrix = feature_matrix(rows, ADVANCED_FEATURES)
    away, home = build_run_pipeline(), build_run_pipeline()
    away.fit(matrix, run_labels(rows, "away"))
    home.fit(matrix, run_labels(rows, "home"))
    return {"away": away, "home": home}


def evaluate_pair(
    models: dict[str, Pipeline],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> dict[str, Any]:
    matrix = feature_matrix(test_rows, ADVANCED_FEATURES)
    away_actual, home_actual = run_labels(test_rows, "away"), run_labels(test_rows, "home")
    away_prediction = models["away"].predict(matrix)
    home_prediction = models["home"].predict(matrix)
    away_mean = float(run_labels(train_rows, "away").mean())
    home_mean = float(run_labels(train_rows, "home").mean())
    baseline_away = np.full(len(test_rows), away_mean)
    baseline_home = np.full(len(test_rows), home_mean)
    return {
        "train_games": len(train_rows),
        "test_games": len(test_rows),
        "model_metrics": score_metrics(
            away_actual, home_actual, away_prediction, home_prediction
        ),
        "league_average_baseline": {
            "away_runs": round(away_mean, 4),
            "home_runs": round(home_mean, 4),
            "metrics": score_metrics(
                away_actual, home_actual, baseline_away, baseline_home
            ),
        },
        "mean_projected_total": round(float((away_prediction + home_prediction).mean()), 3),
        "mean_actual_total": round(float((away_actual + home_actual).mean()), 3),
    }


def validate_expected_runs(
    grouped: dict[int, list[dict[str, str]]], test_season: int
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    seasons = sorted(grouped)
    if len(seasons) < 3 or test_season != seasons[-1]:
        raise ValueError("Expected at least three seasons with the newest held out")
    folds = []
    final_metrics = None
    for index, validation_season in enumerate(seasons[1:], start=1):
        train_seasons = seasons[:index]
        train_rows = [row for season in train_seasons for row in grouped[season]]
        models = fit_pair(train_rows)
        metrics = evaluate_pair(models, train_rows, grouped[validation_season])
        fold = {
            "train_seasons": train_seasons,
            "validation_season": validation_season,
            **metrics,
        }
        if validation_season == test_season:
            final_metrics = fold
        else:
            folds.append(fold)
    assert final_metrics is not None
    all_rows = [row for season in seasons for row in grouped[season]]
    production_models = fit_pair(all_rows)
    report = {
        "model": "paired_poisson_regression",
        "validation_method": "expanding_window_by_complete_season",
        "features": ADVANCED_FEATURES,
        "development_folds": folds,
        "final_untouched_test": final_metrics,
        "production_refit": {"seasons": seasons, "games": len(all_rows)},
        "interpretation": "Projected scores are conditional means, not exact score forecasts.",
    }
    return production_models, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, nargs="+")
    parser.add_argument("--test-season", type=int, required=True)
    args = parser.parse_args()
    grouped = group_rows_by_season(args.data)
    models, report = validate_expected_runs(grouped, args.test_season)
    joblib.dump(
        {"models": models, "features": ADVANCED_FEATURES, "model_name": report["model"]},
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    final = report["final_untouched_test"]
    print(f"Untouched {args.test_season}: {final['test_games']} games")
    print(f"Model metrics: {final['model_metrics']}")
    print(f"Baseline metrics: {final['league_average_baseline']['metrics']}")
    print(f"Model saved to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
