"""Validate a live-compatible pitcher/bullpen probability candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import ADVANCED_FEATURES, LIVE_CONTEXT_FEATURES
from ml.multiseason_validation import group_rows_by_season, run_multiseason_validation

REPORT_PATH = PROJECT_ROOT / "docs" / "pitching_probability_candidate_report.json"
MODEL_PATH = PROJECT_ROOT / "models" / "pitching_probability_candidate.joblib"
MINIMUM_GAMES_PER_SEASON = 2_000
MINIMUM_LOG_LOSS_IMPROVEMENT = 0.001
MAXIMUM_ACCURACY_DROP = 0.01


def gate_report(
    advanced: dict[str, Any], candidate: dict[str, Any], season_games: dict[int, int]
) -> dict[str, Any]:
    """Apply predeclared coverage and untouched-season quality gates."""
    advanced_test = advanced["final_untouched_test"]["model_metrics"]
    candidate_test = candidate["final_untouched_test"]["model_metrics"]
    log_loss_improvement = round(
        advanced_test["log_loss"] - candidate_test["log_loss"], 4
    )
    gates = {
        "full_season_coverage": all(
            games >= MINIMUM_GAMES_PER_SEASON for games in season_games.values()
        ),
        "log_loss_improves": log_loss_improvement >= MINIMUM_LOG_LOSS_IMPROVEMENT,
        "brier_score_not_worse": (
            candidate_test["brier_score"] <= advanced_test["brier_score"]
        ),
        "accuracy_not_materially_worse": (
            candidate_test["accuracy"]
            >= advanced_test["accuracy"] - MAXIMUM_ACCURACY_DROP
        ),
    }
    deploy = all(gates.values())
    return {
        "decision": "deploy" if deploy else "context_only",
        "probability_model_changed": False,
        "validation_method": "same_rows_expanding_window_by_complete_season",
        "available_games_by_season": {
            str(season): games for season, games in sorted(season_games.items())
        },
        "thresholds": {
            "minimum_games_per_season": MINIMUM_GAMES_PER_SEASON,
            "minimum_log_loss_improvement": MINIMUM_LOG_LOSS_IMPROVEMENT,
            "maximum_accuracy_drop": MAXIMUM_ACCURACY_DROP,
        },
        "gates": gates,
        "untouched_test": {
            "advanced_model": advanced_test,
            "pitching_bullpen_candidate": candidate_test,
            "log_loss_improvement": log_loss_improvement,
        },
        "reason": (
            "Candidate passed every gate and is eligible for a separately reviewed production promotion."
            if deploy
            else "Candidate did not pass every gate; production probabilities remain unchanged."
        ),
    }


def validate_candidate(
    paths: list[Path], test_season: int
) -> tuple[Any, dict[str, Any]]:
    grouped = group_rows_by_season(paths)
    advanced_pipeline, advanced = run_multiseason_validation(
        grouped, test_season, ADVANCED_FEATURES, "advanced"
    )
    del advanced_pipeline
    candidate_pipeline, candidate = run_multiseason_validation(
        grouped, test_season, LIVE_CONTEXT_FEATURES, "combined"
    )
    report = gate_report(
        advanced, candidate, {season: len(rows) for season, rows in grouped.items()}
    )
    report["advanced_validation"] = advanced
    report["candidate_validation"] = candidate
    return candidate_pipeline, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--test-season", required=True, type=int)
    args = parser.parse_args()
    pipeline, report = validate_candidate(args.data, args.test_season)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": LIVE_CONTEXT_FEATURES,
            "model_name": "logistic_regression_pitching_bullpen_candidate",
            "deployment_status": report["decision"],
        },
        MODEL_PATH,
    )
    print(f"Decision: {report['decision'].upper()}")
    print(report["reason"])
    print(f"Candidate model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Audit report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
