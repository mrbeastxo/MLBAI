"""Explain daily logistic predictions and attach held-out evidence bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import feature_matrix
from ml.predict_daily import predict_rows, read_rows

DEFAULT_MODEL = PROJECT_ROOT / "models" / "production_model.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "model_comparison_report.json"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_LABELS = {
    "win_percentage_home_minus_away": "season win percentage",
    "runs_per_game_home_minus_away": "season scoring",
    "runs_allowed_per_game_home_minus_away": "runs allowed",
    "run_differential_per_game_home_minus_away": "season run differential",
    "last_10_win_percentage_home_minus_away": "last-10 record",
    "last_10_run_differential_home_minus_away": "last-10 run differential",
    "venue_win_percentage_home_minus_away": "home/away split",
    "schedule_load_home_minus_away": "recent schedule load",
    "last_30_win_percentage_home_minus_away": "last-30 record",
    "last_30_run_differential_home_minus_away": "last-30 run differential",
    "pythagorean_expectation_home_minus_away": "Pythagorean expectation",
    "streak_home_minus_away": "current streak",
    "elo_rating_home_minus_away": "Elo team strength",
}


def certainty_band(probability: float) -> str:
    top_probability = max(probability, 1.0 - probability)
    if top_probability < 0.55:
        return "close"
    if top_probability < 0.60:
        return "slight_lean"
    if top_probability < 0.65:
        return "moderate_lean"
    return "strongest_lean"


def logistic_contributions(
    row: dict[str, str], artifact: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """Return intercept and exact standardized log-odds contributions."""
    pipeline = artifact["pipeline"]
    if "scaler" not in pipeline.named_steps:
        raise ValueError("Exact explanations currently require the logistic pipeline")
    features = artifact["features"]
    matrix = feature_matrix([row], features)
    imputed = pipeline.named_steps["imputer"].transform(matrix)
    scaled = pipeline.named_steps["scaler"].transform(imputed)
    model = pipeline.named_steps["model"]
    contributions = {
        feature: float(value * coefficient)
        for feature, value, coefficient in zip(
            features, scaled[0], model.coef_[0], strict=True
        )
    }
    return float(model.intercept_[0]), contributions


def evidence_for_band(report: dict[str, Any], band: str) -> dict[str, Any]:
    bands = report["untouched_test"].get("certainty_bands", [])
    return next((item for item in bands if item["band"] == band), {})


def evidence_grade(evidence: dict[str, Any]) -> str:
    games = int(evidence.get("games") or 0)
    accuracy = evidence.get("observed_accuracy")
    if games < 30 or accuracy is None:
        return "insufficient_history"
    if accuracy < 0.55:
        return "weak"
    if accuracy < 0.60:
        return "limited"
    return "promising_not_proven"


def explain_rows(
    rows: list[dict[str, str]], artifact: dict[str, Any], report: dict[str, Any]
) -> list[dict[str, Any]]:
    predictions = {item["game_id"]: item for item in predict_rows(rows, artifact)}
    explanations: list[dict[str, Any]] = []
    for row in rows:
        prediction = predictions[row["game_id"]]
        home_probability = prediction["home_win_probability"]
        leans_home = home_probability >= 0.5
        intercept, contributions = logistic_contributions(row, artifact)
        favorable = [
            (feature, value)
            for feature, value in contributions.items()
            if (value > 0) == leans_home and value != 0
        ]
        opposing = [
            (feature, value)
            for feature, value in contributions.items()
            if (value > 0) != leans_home and value != 0
        ]
        favorable.sort(key=lambda item: abs(item[1]), reverse=True)
        opposing.sort(key=lambda item: abs(item[1]), reverse=True)
        band = certainty_band(home_probability)
        evidence = evidence_for_band(report, band)
        missing_features = sum(
            row.get(feature, "") == "" for feature in artifact["features"]
        )

        def describe(items: list[tuple[str, float]], limit: int) -> list[dict[str, Any]]:
            return [
                {
                    "factor": FEATURE_LABELS.get(feature, feature),
                    "feature": feature,
                    "raw_home_minus_away": row.get(feature),
                    "log_odds_contribution": round(value, 4),
                }
                for feature, value in items[:limit]
            ]

        explanations.append(
            {
                **prediction,
                "certainty_band": band,
                "evidence_grade": evidence_grade(evidence),
                "held_out_band_games": evidence.get("games", 0),
                "held_out_band_accuracy": evidence.get("observed_accuracy"),
                "missing_model_features": missing_features,
                "data_quality": "complete" if missing_features == 0 else "incomplete",
                "intercept_log_odds": round(intercept, 4),
                "strongest_supporting_factors": describe(favorable, 3),
                "strongest_opposing_factors": describe(opposing, 2),
                "reliability_note": (
                    "Experimental estimate: the selected model improved held-out "
                    "probability quality only slightly and did not beat baseline accuracy."
                ),
            }
        )
    return sorted(
        explanations,
        key=lambda item: max(
            item["home_win_probability"], item["away_win_probability"]
        ),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows = read_rows(args.features)
    artifact = joblib.load(args.model)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    try:
        explanations = explain_rows(rows, artifact, report)
    except ValueError as exc:
        raise SystemExit(f"Could not explain predictions: {exc}") from exc
    if not explanations:
        raise SystemExit("No games were available to explain")
    output_path = PROCESSED_DATA_DIR / f"analysis_{explanations[0]['official_date']}.json"
    output_path.write_text(json.dumps(explanations, indent=2), encoding="utf-8")

    for item in explanations:
        top_probability = max(
            item["home_win_probability"], item["away_win_probability"]
        )
        factors = ", ".join(
            factor["factor"] for factor in item["strongest_supporting_factors"]
        )
        print(
            f"- {item['model_lean']} {top_probability:.1%} | "
            f"{item['certainty_band']} | evidence {item['evidence_grade']} | "
            f"signals: {factors}"
        )
    print(f"Detailed analysis saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
