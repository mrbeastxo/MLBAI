"""Audit probability calibration without silently changing production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import ADVANCED_FEATURES, feature_matrix, labels, probability_metrics
from ml.model_comparison import build_candidate
from ml.multiseason_validation import group_rows_by_season

REPORT_PATH = PROJECT_ROOT / "docs" / "calibration_audit_report.json"
MINIMUM_LOG_LOSS_IMPROVEMENT = 0.001
MAXIMUM_ACCURACY_DROP = 0.005


def logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> dict[str, Any]:
    edges = np.linspace(0, 1, bins + 1)
    rows: list[dict[str, Any]] = []
    weighted_error = 0.0
    maximum_error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        count = int(mask.sum())
        if not count:
            continue
        mean_probability = float(probabilities[mask].mean())
        observed_rate = float(y_true[mask].mean())
        gap = abs(mean_probability - observed_rate)
        weighted_error += count / len(y_true) * gap
        maximum_error = max(maximum_error, gap)
        rows.append(
            {
                "range": f"{lower:.1f}-{upper:.1f}",
                "games": count,
                "mean_probability": round(mean_probability, 4),
                "observed_home_win_rate": round(observed_rate, 4),
                "absolute_gap": round(gap, 4),
            }
        )
    return {
        "expected_calibration_error": round(weighted_error, 4),
        "maximum_calibration_error": round(maximum_error, 4),
        "bins": rows,
    }


def development_calibrator(
    grouped: dict[int, list[dict[str, str]]], test_season: int
) -> tuple[LogisticRegression, list[dict[str, Any]]]:
    development_seasons = sorted(season for season in grouped if season < test_season)
    if len(development_seasons) < 3:
        raise ValueError("At least three development seasons are required")
    probabilities: list[float] = []
    outcomes: list[int] = []
    folds: list[dict[str, Any]] = []
    for validation_season in development_seasons[1:]:
        train_seasons = [season for season in development_seasons if season < validation_season]
        train_rows = [row for season in train_seasons for row in grouped[season]]
        validation_rows = grouped[validation_season]
        pipeline = build_candidate("logistic_regression")
        pipeline.fit(feature_matrix(train_rows, ADVANCED_FEATURES), labels(train_rows))
        fold_probabilities = pipeline.predict_proba(
            feature_matrix(validation_rows, ADVANCED_FEATURES)
        )[:, 1]
        probabilities.extend(fold_probabilities)
        outcomes.extend(labels(validation_rows))
        folds.append(
            {
                "train_seasons": train_seasons,
                "validation_season": validation_season,
                "games": len(validation_rows),
            }
        )
    calibrator = LogisticRegression(C=1_000_000, random_state=42)
    calibrator.fit(logits(np.asarray(probabilities)), np.asarray(outcomes))
    return calibrator, folds


def deployment_decision(
    raw_metrics: dict[str, float], calibrated_metrics: dict[str, float]
) -> dict[str, Any]:
    log_loss_gain = raw_metrics["log_loss"] - calibrated_metrics["log_loss"]
    brier_not_worse = calibrated_metrics["brier_score"] <= raw_metrics["brier_score"]
    accuracy_drop = raw_metrics["accuracy"] - calibrated_metrics["accuracy"]
    gates = {
        "minimum_log_loss_improvement": round(MINIMUM_LOG_LOSS_IMPROVEMENT, 4),
        "observed_log_loss_improvement": round(log_loss_gain, 4),
        "brier_score_not_worse": brier_not_worse,
        "maximum_accuracy_drop": round(MAXIMUM_ACCURACY_DROP, 4),
        "observed_accuracy_drop": round(accuracy_drop, 4),
    }
    deploy = (
        log_loss_gain >= MINIMUM_LOG_LOSS_IMPROVEMENT
        and brier_not_worse
        and accuracy_drop <= MAXIMUM_ACCURACY_DROP
    )
    return {
        "deploy_calibration": deploy,
        "decision": "deploy" if deploy else "reject",
        "reason": (
            "All predeclared probability-quality gates passed."
            if deploy
            else "Candidate did not pass every predeclared gate; production remains unchanged."
        ),
        "gates": gates,
    }


def run_audit(
    grouped: dict[int, list[dict[str, str]]], test_season: int
) -> dict[str, Any]:
    seasons = sorted(grouped)
    if test_season != seasons[-1]:
        raise ValueError("The audit season must be the newest supplied season")
    calibrator, folds = development_calibrator(grouped, test_season)
    train_seasons = [season for season in seasons if season < test_season]
    train_rows = [row for season in train_seasons for row in grouped[season]]
    test_rows = grouped[test_season]
    pipeline = build_candidate("logistic_regression")
    pipeline.fit(feature_matrix(train_rows, ADVANCED_FEATURES), labels(train_rows))
    raw_probabilities = pipeline.predict_proba(
        feature_matrix(test_rows, ADVANCED_FEATURES)
    )[:, 1]
    calibrated_probabilities = calibrator.predict_proba(logits(raw_probabilities))[:, 1]
    y_test = labels(test_rows)
    raw_metrics = probability_metrics(y_test, raw_probabilities)
    calibrated_metrics = probability_metrics(y_test, calibrated_probabilities)
    baseline_probability = float(labels(train_rows).mean())
    baseline_metrics = probability_metrics(
        y_test, np.full(len(y_test), baseline_probability)
    )
    return {
        "method": "platt_scaling_on_development_out_of_fold_predictions",
        "features": ADVANCED_FEATURES,
        "development_folds": folds,
        "calibrator": {
            "intercept": round(float(calibrator.intercept_[0]), 6),
            "slope": round(float(calibrator.coef_[0][0]), 6),
        },
        "audit_season": test_season,
        "audit_games": len(test_rows),
        "raw_model": {
            "metrics": raw_metrics,
            "calibration": calibration_error(y_test, raw_probabilities),
        },
        "calibrated_candidate": {
            "metrics": calibrated_metrics,
            "calibration": calibration_error(y_test, calibrated_probabilities),
        },
        "home_rate_baseline": {
            "probability": round(baseline_probability, 4),
            "metrics": baseline_metrics,
        },
        "deployment": deployment_decision(raw_metrics, calibrated_metrics),
        "policy_note": (
            "The audit season has now informed a deployment decision and must not be "
            "described as untouched in future model changes. A newer season is required "
            "for the next independent final test."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--test-season", required=True, type=int)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    try:
        report = run_audit(group_rows_by_season(args.data), args.test_season)
    except ValueError as error:
        raise SystemExit(f"Could not audit calibration: {error}") from error
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raw = report["raw_model"]["metrics"]
    calibrated = report["calibrated_candidate"]["metrics"]
    print(f"Raw model:        log loss {raw['log_loss']:.4f}, Brier {raw['brier_score']:.4f}, accuracy {raw['accuracy']:.4f}")
    print(f"Calibrated model: log loss {calibrated['log_loss']:.4f}, Brier {calibrated['brier_score']:.4f}, accuracy {calibrated['accuracy']:.4f}")
    print(f"Decision: {report['deployment']['decision'].upper()}")
    print(report["deployment"]["reason"])
    print(f"Audit saved to: {args.report.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
