"""Select a model on development seasons, then test it once on the newest season."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Avoid platform-specific physical-core probing; candidates are intentionally
# trained deterministically for this small classroom dataset.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import (
    ADVANCED_FEATURES,
    build_pipeline,
    calibration_bins,
    feature_matrix,
    labels,
    probability_metrics,
)
from ml.multiseason_validation import group_rows_by_season

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "docs"
MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting")


def build_candidate(name: str) -> Pipeline:
    """Return one fixed candidate; no untouched-test tuning is allowed."""
    if name == "logistic_regression":
        return build_pipeline()
    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_features="sqrt",
                        min_samples_leaf=20,
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.03,
                        max_iter=300,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        l2_regularization=0.5,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate_candidate(
    name: str,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
) -> tuple[Pipeline, dict[str, float]]:
    pipeline = build_candidate(name)
    pipeline.fit(
        feature_matrix(train_rows, ADVANCED_FEATURES),
        labels(train_rows),
    )
    probabilities = pipeline.predict_proba(
        feature_matrix(validation_rows, ADVANCED_FEATURES)
    )[:, 1]
    return pipeline, probability_metrics(labels(validation_rows), probabilities)


def select_by_average_log_loss(
    development_results: dict[str, list[dict[str, Any]]]
) -> str:
    """Select using development folds only, with deterministic tie-breaking."""
    averages = {
        name: np.mean([fold["metrics"]["log_loss"] for fold in folds])
        for name, folds in development_results.items()
    }
    return min(MODEL_NAMES, key=lambda name: (averages[name], MODEL_NAMES.index(name)))


def compare_models(
    grouped: dict[int, list[dict[str, str]]], test_season: int
) -> tuple[Pipeline, dict[str, Any]]:
    seasons = sorted(grouped)
    if len(seasons) < 4:
        raise ValueError("Four seasons are required for two development folds and a test")
    if test_season != seasons[-1]:
        raise ValueError("The untouched test season must be the newest season")

    development_seasons = seasons[:-1]
    validation_seasons = development_seasons[1:]
    development_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_NAMES
    }
    for validation_season in validation_seasons:
        train_seasons = [season for season in development_seasons if season < validation_season]
        train_rows = [row for season in train_seasons for row in grouped[season]]
        validation_rows = grouped[validation_season]
        for name in MODEL_NAMES:
            _, metrics = evaluate_candidate(name, train_rows, validation_rows)
            development_results[name].append(
                {
                    "train_seasons": train_seasons,
                    "validation_season": validation_season,
                    "train_games": len(train_rows),
                    "validation_games": len(validation_rows),
                    "metrics": metrics,
                }
            )

    selected_name = select_by_average_log_loss(development_results)
    final_train_rows = [row for season in development_seasons for row in grouped[season]]
    test_rows = grouped[test_season]
    selected_pipeline, final_metrics = evaluate_candidate(
        selected_name, final_train_rows, test_rows
    )
    y_train = labels(final_train_rows)
    y_test = labels(test_rows)
    test_probabilities = selected_pipeline.predict_proba(
        feature_matrix(test_rows, ADVANCED_FEATURES)
    )[:, 1]
    baseline_probability = float(y_train.mean())
    baseline_probabilities = np.full(len(y_test), baseline_probability)

    development_summary = {}
    for name, folds in development_results.items():
        development_summary[name] = {
            "folds": folds,
            "mean_accuracy": round(
                float(np.mean([fold["metrics"]["accuracy"] for fold in folds])), 4
            ),
            "mean_log_loss": round(
                float(np.mean([fold["metrics"]["log_loss"] for fold in folds])), 4
            ),
            "mean_brier_score": round(
                float(np.mean([fold["metrics"]["brier_score"] for fold in folds])), 4
            ),
        }

    report = {
        "selection_rule": "lowest_mean_development_log_loss",
        "features": ADVANCED_FEATURES,
        "development_results": development_summary,
        "selected_model": selected_name,
        "untouched_test": {
            "train_seasons": development_seasons,
            "test_season": test_season,
            "train_games": len(final_train_rows),
            "test_games": len(test_rows),
            "selected_model_metrics": final_metrics,
            "home_rate_baseline_probability": round(baseline_probability, 4),
            "home_rate_baseline_metrics": probability_metrics(
                y_test, baseline_probabilities
            ),
            "calibration": calibration_bins(y_test, test_probabilities),
        },
        "untouched_test_policy": (
            "Only the development-fold winner was evaluated on the test season."
        ),
    }
    return selected_pipeline, report


def save_artifacts(pipeline: Pipeline, report: dict[str, Any]) -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "selected_model.joblib"
    report_path = REPORT_DIR / "model_comparison_report.json"
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": ADVANCED_FEATURES,
            "model_name": report["selected_model"],
        },
        model_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--test-season", required=True, type=int)
    args = parser.parse_args()
    try:
        grouped = group_rows_by_season(args.data)
        pipeline, report = compare_models(grouped, args.test_season)
    except ValueError as exc:
        raise SystemExit(f"Could not compare models: {exc}") from exc
    model_path, report_path = save_artifacts(pipeline, report)

    print("Development-fold averages:")
    for name in MODEL_NAMES:
        result = report["development_results"][name]
        print(
            f"- {name}: accuracy {result['mean_accuracy']:.3f}, "
            f"log loss {result['mean_log_loss']:.3f}, "
            f"Brier {result['mean_brier_score']:.3f}"
        )
    final = report["untouched_test"]
    selected = final["selected_model_metrics"]
    baseline = final["home_rate_baseline_metrics"]
    print(f"Selected before test: {report['selected_model']}")
    print(
        f"Untouched {final['test_season']} | accuracy {selected['accuracy']:.3f}, "
        f"log loss {selected['log_loss']:.3f}, Brier {selected['brier_score']:.3f}"
    )
    print(
        f"Home-rate baseline | accuracy {baseline['accuracy']:.3f}, "
        f"log loss {baseline['log_loss']:.3f}, Brier {baseline['brier_score']:.3f}"
    )
    print(f"Selected model saved to: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Report saved to: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
