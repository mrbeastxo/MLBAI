"""Run expanding-season validation with the newest season held out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import (
    ADVANCED_FEATURES,
    FEATURES,
    build_pipeline,
    calibration_bins,
    feature_matrix,
    labels,
    load_training_rows,
    probability_metrics,
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "docs"


def group_rows_by_season(
    paths: list[Path],
) -> dict[int, list[dict[str, str]]]:
    """Load datasets, reject mixed files, and index rows by season."""
    grouped: dict[int, list[dict[str, str]]] = {}
    for path in paths:
        rows = load_training_rows(path)
        seasons = {int(row["season"]) for row in rows}
        if len(seasons) != 1:
            raise ValueError(f"{path} must contain exactly one season")
        season = seasons.pop()
        if season in grouped:
            raise ValueError(f"Season {season} was supplied more than once")
        grouped[season] = rows
    return dict(sorted(grouped.items()))


def expanding_season_splits(
    grouped: dict[int, list[dict[str, str]]], test_season: int
) -> list[tuple[list[int], int]]:
    """Return expanding validation folds ending with one final test season."""
    seasons = sorted(grouped)
    if test_season not in grouped:
        raise ValueError(f"Test season {test_season} is missing")
    if test_season != seasons[-1]:
        raise ValueError("The test season must be the newest supplied season")
    if len(seasons) < 3:
        raise ValueError("At least three seasons are required")
    return [(seasons[:index], season) for index, season in enumerate(seasons[1:], start=1)]


def evaluate_fold(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    features: list[str] = FEATURES,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
    """Fit one chronological fold and compare against its home-rate baseline."""
    pipeline = build_pipeline()
    x_train, y_train = feature_matrix(train_rows, features), labels(train_rows)
    x_validation, y_validation = feature_matrix(validation_rows, features), labels(validation_rows)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_validation)[:, 1]
    baseline_probability = float(y_train.mean())
    baseline_probabilities = np.full(len(y_validation), baseline_probability)
    metrics = {
        "train_games": len(train_rows),
        "validation_games": len(validation_rows),
        "model_metrics": probability_metrics(y_validation, probabilities),
        "home_rate_baseline_probability": round(baseline_probability, 4),
        "home_rate_baseline_metrics": probability_metrics(
            y_validation, baseline_probabilities
        ),
    }
    return pipeline, metrics, y_validation, probabilities


def run_multiseason_validation(
    grouped: dict[int, list[dict[str, str]]],
    test_season: int,
    features: list[str] = FEATURES,
    feature_set_name: str = "baseline",
) -> tuple[Any, dict[str, Any]]:
    """Evaluate expanding folds and return the untouched-season model/report."""
    split_plan = expanding_season_splits(grouped, test_season)
    validation_folds: list[dict[str, Any]] = []
    final_pipeline = None
    final_y = None
    final_probabilities = None
    final_metrics = None

    for train_seasons, validation_season in split_plan:
        train_rows = [row for season in train_seasons for row in grouped[season]]
        validation_rows = grouped[validation_season]
        pipeline, metrics, y_validation, probabilities = evaluate_fold(
            train_rows, validation_rows, features
        )
        fold = {
            "train_seasons": train_seasons,
            "validation_season": validation_season,
            **metrics,
        }
        if validation_season == test_season:
            final_pipeline = pipeline
            final_y = y_validation
            final_probabilities = probabilities
            final_metrics = metrics
        else:
            validation_folds.append(fold)

    assert final_pipeline is not None
    assert final_y is not None
    assert final_probabilities is not None
    assert final_metrics is not None
    final_train_seasons = [season for season in grouped if season < test_season]

    coefficients = final_pipeline.named_steps["model"].coef_[0]
    report = {
        "model": "logistic_regression",
        "validation_method": "expanding_window_by_complete_season",
        "feature_set": feature_set_name,
        "feature_count": len(features),
        "features": features,
        "model_selection_folds": validation_folds,
        "final_untouched_test": {
            "train_seasons": final_train_seasons,
            "test_season": test_season,
            **final_metrics,
            "calibration": calibration_bins(final_y, final_probabilities),
        },
        "final_standardized_coefficients": sorted(
            (
                {"feature": feature, "coefficient": round(float(coefficient), 4)}
                for feature, coefficient in zip(features, coefficients, strict=True)
            ),
            key=lambda item: abs(item["coefficient"]),
            reverse=True,
        ),
    }
    return final_pipeline, report


def save_artifacts(
    pipeline: Any, report: dict[str, Any], features: list[str], feature_set_name: str
) -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "advanced" if feature_set_name == "advanced" else "multiseason"
    model_path = MODEL_DIR / f"{prefix}_logistic.joblib"
    report_path = REPORT_DIR / f"{prefix}_validation_report.json"
    joblib.dump({"pipeline": pipeline, "features": features}, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--test-season", required=True, type=int)
    parser.add_argument(
        "--feature-set", choices=("baseline", "advanced"), default="baseline"
    )
    args = parser.parse_args()

    try:
        grouped = group_rows_by_season(args.data)
        features = ADVANCED_FEATURES if args.feature_set == "advanced" else FEATURES
        pipeline, report = run_multiseason_validation(
            grouped, args.test_season, features, args.feature_set
        )
    except ValueError as exc:
        raise SystemExit(f"Could not run multi-season validation: {exc}") from exc
    model_path, report_path = save_artifacts(
        pipeline, report, features, args.feature_set
    )

    for fold in report["model_selection_folds"]:
        model = fold["model_metrics"]
        print(
            f"Train {fold['train_seasons']} -> validate {fold['validation_season']}: "
            f"accuracy {model['accuracy']:.3f}, log loss {model['log_loss']:.3f}, "
            f"Brier {model['brier_score']:.3f}"
        )
    final = report["final_untouched_test"]
    model = final["model_metrics"]
    baseline = final["home_rate_baseline_metrics"]
    print(
        f"Untouched {final['test_season']} test | model accuracy {model['accuracy']:.3f}, "
        f"log loss {model['log_loss']:.3f}, Brier {model['brier_score']:.3f}"
    )
    print(
        f"Home-rate baseline          | accuracy {baseline['accuracy']:.3f}, "
        f"log loss {baseline['log_loss']:.3f}, Brier {baseline['brier_score']:.3f}"
    )
    print(f"Model saved to: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Report saved to: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
