"""Decide whether pitcher and bullpen features may affect production probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

DEFAULT_PITCHER_REPORT = PROJECT_ROOT / "docs" / "pitcher_enhanced_validation_report.json"
DEFAULT_COMBINED_REPORT = PROJECT_ROOT / "docs" / "combined_validation_report.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "pitching_bullpen_deployment_report.json"
MINIMUM_GAMES_PER_SEASON = 1000


def deployment_report(
    pitcher_report: dict[str, Any], combined_report: dict[str, Any]
) -> dict[str, Any]:
    combined_folds = combined_report["model_selection_folds"]
    combined_test = combined_report["final_untouched_test"]
    pitcher_test = pitcher_report["final_untouched_test"]
    season_games = {
        str(fold["validation_season"]): fold["validation_games"]
        for fold in combined_folds
    }
    first_fold = combined_folds[0]
    if len(first_fold.get("train_seasons", [])) == 1:
        season_games[str(first_fold["train_seasons"][0])] = first_fold["train_games"]
    season_games[str(combined_test["test_season"])] = combined_test["validation_games"]
    coverage_pass = all(
        games >= MINIMUM_GAMES_PER_SEASON for games in season_games.values()
    )
    pitcher_quality_pass = (
        pitcher_test["model_metrics"]["log_loss"]
        < pitcher_test["home_rate_baseline_metrics"]["log_loss"]
    )
    combined_quality_pass = (
        combined_test["model_metrics"]["log_loss"]
        < combined_test["home_rate_baseline_metrics"]["log_loss"]
    )
    deploy = coverage_pass and pitcher_quality_pass and combined_quality_pass
    return {
        "decision": "deploy" if deploy else "context_only",
        "probability_model_changed": deploy,
        "minimum_games_per_season": MINIMUM_GAMES_PER_SEASON,
        "available_games_by_validation_season": season_games,
        "gates": {
            "coverage_pass": coverage_pass,
            "pitcher_model_beats_baseline_log_loss": pitcher_quality_pass,
            "combined_model_beats_baseline_log_loss": combined_quality_pass,
        },
        "test_metrics": {
            "pitcher_model_log_loss": pitcher_test["model_metrics"]["log_loss"],
            "pitcher_baseline_log_loss": pitcher_test["home_rate_baseline_metrics"]["log_loss"],
            "combined_model_log_loss": combined_test["model_metrics"]["log_loss"],
            "combined_baseline_log_loss": combined_test["home_rate_baseline_metrics"]["log_loss"],
        },
        "reason": (
            "All coverage and probability-quality gates passed."
            if deploy
            else "Historical coverage and validation are not strong enough; live pitcher and bullpen data is analysis context only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitcher-report", type=Path, default=DEFAULT_PITCHER_REPORT)
    parser.add_argument("--combined-report", type=Path, default=DEFAULT_COMBINED_REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report = deployment_report(
        json.loads(args.pitcher_report.read_text(encoding="utf-8")),
        json.loads(args.combined_report.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Decision: {report['decision'].upper()}")
    print(report["reason"])
    print(f"Report saved to: {args.output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
