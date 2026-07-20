"""Validate and train the Milestones 32-36 context-aware probability model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.data_pipeline.historical_environment import ENVIRONMENT_FEATURES
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import ADVANCED_FEATURES, feature_matrix, labels, probability_metrics
from ml.multiseason_validation import group_rows_by_season

STARTER_FEATURES = [
    "starter_era_home_minus_away",
    "starter_whip_home_minus_away",
    "starter_k9_home_minus_away",
    "starter_bb9_home_minus_away",
]
PRODUCTION_CONTEXT_FEATURES = ADVANCED_FEATURES + STARTER_FEATURES
WEATHER_CANDIDATE_FEATURES = ADVANCED_FEATURES + ENVIRONMENT_FEATURES
COMBINED_CANDIDATE_FEATURES = PRODUCTION_CONTEXT_FEATURES + ENVIRONMENT_FEATURES
REPORT_PATH = PROJECT_ROOT / "docs" / "context_model_upgrade_report.json"
MODEL_PATH = PROJECT_ROOT / "models" / "production_model.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(C=0.01, max_iter=2_000, random_state=42),
            ),
        ]
    )


def evaluate(train: list[dict[str, str]], test: list[dict[str, str]], features: list[str]):
    pipeline = build_pipeline()
    pipeline.fit(feature_matrix(train, features), labels(train))
    probabilities = pipeline.predict_proba(feature_matrix(test, features))[:, 1]
    return pipeline, probability_metrics(labels(test), probabilities)


def validation_report(grouped: dict[int, list[dict[str, str]]]) -> dict[str, Any]:
    seasons = sorted(grouped)
    if len(seasons) < 4:
        raise ValueError("Four complete seasons are required")
    folds = []
    for season in seasons[1:]:
        training = [row for year in seasons if year < season for row in grouped[year]]
        test = grouped[season]
        candidates = {}
        for name, features in {
            "team_only": ADVANCED_FEATURES,
            "starter": PRODUCTION_CONTEXT_FEATURES,
            "weather": WEATHER_CANDIDATE_FEATURES,
            "starter_weather": COMBINED_CANDIDATE_FEATURES,
        }.items():
            _, metrics = evaluate(training, test, features)
            candidates[name] = metrics
        folds.append({"test_season": season, "games": len(test), "models": candidates})

    final = folds[-1]["models"]
    starter_log_loss_wins = sum(
        fold["models"]["starter"]["log_loss"]
        < fold["models"]["team_only"]["log_loss"]
        for fold in folds
    )
    starter_gates = {
        "newest_season_log_loss_improves": final["starter"]["log_loss"]
        < final["team_only"]["log_loss"],
        "newest_season_brier_improves": final["starter"]["brier_score"]
        < final["team_only"]["brier_score"],
        "newest_season_accuracy_not_worse": final["starter"]["accuracy"]
        >= final["team_only"]["accuracy"],
        "majority_of_seasons_improve_log_loss": starter_log_loss_wins
        >= (len(folds) // 2 + 1),
    }
    weather_gates = {
        "newest_season_log_loss_improves": final["weather"]["log_loss"]
        < final["team_only"]["log_loss"],
        "newest_season_brier_improves": final["weather"]["brier_score"]
        < final["team_only"]["brier_score"],
        "combined_beats_validated_starter": final["starter_weather"]["log_loss"]
        < final["starter"]["log_loss"],
    }
    return {
        "milestones": [32, 33, 34, 35, 36],
        "method": "expanding-window validation by complete season",
        "seasons": seasons,
        "games": sum(len(rows) for rows in grouped.values()),
        "folds": folds,
        "starter_deployment": {
            "decision": "deploy" if all(starter_gates.values()) else "reject",
            "gates": starter_gates,
            "features": STARTER_FEATURES,
        },
        "weather_deployment": {
            "decision": "deploy" if all(weather_gates.values()) else "reject",
            "gates": weather_gates,
            "features": ENVIRONMENT_FEATURES,
            "policy": "Weather remains display-only when validation rejects it.",
        },
        "lineup_injury_deployment": {
            "decision": "context_only",
            "reason": "Official historical pregame lineup and injury snapshots are not yet complete enough for leakage-safe validation.",
        },
        "calibration": {
            "decision": "reject",
            "reason": "Development-only Platt scaling worsened 2025 log loss, Brier score, and accuracy.",
        },
        "next_independent_audit_season": 2026,
    }


def train_and_save(paths: list[Path]) -> tuple[dict[str, Any], Path]:
    grouped = group_rows_by_season(paths)
    report = validation_report(grouped)
    if report["starter_deployment"]["decision"] != "deploy":
        raise ValueError("Starter candidate failed its deployment gates")
    rows = [row for season in sorted(grouped) for row in grouped[season]]
    pipeline = build_pipeline()
    pipeline.fit(feature_matrix(rows, PRODUCTION_CONTEXT_FEATURES), labels(rows))
    artifact = {
        "pipeline": pipeline,
        "features": PRODUCTION_CONTEXT_FEATURES,
        "model_name": "regularized_logistic_with_validated_starters",
        "training_seasons": sorted(grouped),
        "training_games": len(rows),
        "validation_report": REPORT_PATH.name,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(artifact, MODEL_PATH)
    return report, MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    args = parser.parse_args()
    report, path = train_and_save(args.data)
    print(f"Starter decision: {report['starter_deployment']['decision'].upper()}")
    print(f"Weather decision: {report['weather_deployment']['decision'].upper()}")
    print(f"Production model: {path.relative_to(PROJECT_ROOT)}")
    print(f"Validation report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
