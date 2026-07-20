"""Audit score-derived win probabilities against the production feature model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import (
    ADVANCED_FEATURES,
    build_pipeline,
    feature_matrix,
    labels,
    probability_metrics,
)
from ml.expected_runs import fit_pair
from ml.multiseason_validation import group_rows_by_season
from ml.outcome_uncertainty import outcome_distribution

REPORT_PATH = PROJECT_ROOT / "docs" / "outcome_uncertainty_audit_report.json"
MINIMUM_LOG_LOSS_IMPROVEMENT = 0.001


def audit_score_probability(
    train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]
) -> dict[str, Any]:
    score_models = fit_pair(train_rows)
    test_matrix = feature_matrix(test_rows, ADVANCED_FEATURES)
    away_runs = score_models["away"].predict(test_matrix)
    home_runs = score_models["home"].predict(test_matrix)
    score_probabilities = np.asarray(
        [
            outcome_distribution(away, home)["score_model_home_win_probability"]
            for away, home in zip(away_runs, home_runs, strict=True)
        ]
    )
    probability_model = build_pipeline()
    probability_model.fit(
        feature_matrix(train_rows, ADVANCED_FEATURES), labels(train_rows)
    )
    production_probabilities = probability_model.predict_proba(test_matrix)[:, 1]
    outcomes = labels(test_rows)
    score_metrics = probability_metrics(outcomes, score_probabilities)
    production_metrics = probability_metrics(outcomes, production_probabilities)
    improvement = round(production_metrics["log_loss"] - score_metrics["log_loss"], 4)
    gates = {
        "log_loss_improves": improvement >= MINIMUM_LOG_LOSS_IMPROVEMENT,
        "brier_score_not_worse": score_metrics["brier_score"] <= production_metrics["brier_score"],
        "accuracy_not_worse": score_metrics["accuracy"] >= production_metrics["accuracy"],
    }
    return {
        "validation_method": "train_past_seasons_test_newest_season",
        "train_games": len(train_rows),
        "test_games": len(test_rows),
        "production_feature_model": production_metrics,
        "score_derived_probability": score_metrics,
        "log_loss_improvement": improvement,
        "thresholds": {"minimum_log_loss_improvement": MINIMUM_LOG_LOSS_IMPROVEMENT},
        "gates": gates,
        "decision": "eligible_for_review" if all(gates.values()) else "analysis_context_only",
        "probability_model_changed": False,
        "reason": (
            "The score-derived probability passed every audit gate but still requires explicit promotion review."
            if all(gates.values())
            else "The score-derived probability did not beat the production feature model; it remains analysis context."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, nargs="+")
    parser.add_argument("--test-season", type=int, required=True)
    args = parser.parse_args()
    grouped = group_rows_by_season(args.data)
    seasons = sorted(grouped)
    if args.test_season != seasons[-1]:
        raise SystemExit("The test season must be the newest supplied season")
    train = [row for season in seasons[:-1] for row in grouped[season]]
    report = audit_score_probability(train, grouped[args.test_season])
    report["train_seasons"] = seasons[:-1]
    report["test_season"] = args.test_season
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Decision: {report['decision'].upper()}")
    print(report["reason"])
    print(f"Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
