"""Generate immutable future-game learning reports and shadow-model comparisons."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from backend.tracking.prediction_tracker import DEFAULT_DATABASE, connect_database
from ml.baseline_model import ADVANCED_FEATURES, feature_matrix, labels
from ml.context_model_upgrade import build_pipeline
from ml.multiseason_validation import group_rows_by_season

SHADOW_MODEL_PATH = PROJECT_ROOT / "models" / "team_only_shadow.joblib"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "postgame_learning_report.json"
MINIMUM_DRIFT_SAMPLE = 100
ACTIONABLE_SAMPLE = 200
REFERENCE_LOG_LOSS = 0.6878


def probability_metrics(rows: list[sqlite3.Row], probability_field: str) -> dict[str, Any]:
    if not rows:
        return {"games": 0, "accuracy": None, "log_loss": None, "brier_score": None}
    correct = 0
    losses = []
    briers = []
    for row in rows:
        probability = float(row[probability_field])
        outcome = int(row["home_win"])
        correct += int((probability >= 0.5) == bool(outcome))
        clipped = min(max(probability, 1e-15), 1 - 1e-15)
        losses.append(-(outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped)))
        briers.append((probability - outcome) ** 2)
    return {
        "games": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "log_loss": round(sum(losses) / len(losses), 4),
        "brier_score": round(sum(briers) / len(briers), 4),
    }


def sample_label(games: int) -> str:
    if games < 50:
        return "too_early"
    if games < 100:
        return "early"
    if games < ACTIONABLE_SAMPLE:
        return "limited"
    return "actionable"


def factor_diagnostics(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT c.factors_json, r.home_win
        FROM prediction_context c JOIN results r USING (game_id)
        """
    ).fetchall()
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"appearances": 0, "helped": 0, "misled": 0})
    for row in rows:
        for factor in json.loads(row["factors_json"]):
            name = str(factor.get("factor") or factor.get("feature") or "unknown")
            contribution = float(factor.get("log_odds_contribution") or 0)
            helped = (contribution > 0) == bool(row["home_win"])
            totals[name]["appearances"] += 1
            totals[name]["helped" if helped else "misled"] += 1
    return sorted(
        ({"factor": factor, **counts} for factor, counts in totals.items()),
        key=lambda item: item["appearances"],
        reverse=True,
    )


def learning_report(connection: sqlite3.Connection) -> dict[str, Any]:
    primary_rows = connection.execute(
        """SELECT p.home_win_probability, p.official_date, r.home_win
        FROM predictions p JOIN results r USING (game_id)
        ORDER BY p.official_date, p.game_time_utc"""
    ).fetchall()
    comparison_rows = connection.execute(
        """SELECT p.home_win_probability AS primary_probability,
        s.home_win_probability AS shadow_probability, p.official_date, r.home_win
        FROM predictions p JOIN shadow_predictions s USING (game_id)
        JOIN results r USING (game_id)
        ORDER BY p.official_date, p.game_time_utc"""
    ).fetchall()
    primary = probability_metrics(primary_rows, "home_win_probability")
    shadow = probability_metrics(comparison_rows, "shadow_probability")
    comparable_primary = probability_metrics(comparison_rows, "primary_probability")
    daily = []
    by_date: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in primary_rows:
        by_date[row["official_date"]].append(row)
    for official_date, rows in sorted(by_date.items(), reverse=True):
        daily.append({"date": official_date, **probability_metrics(rows, "home_win_probability")})
    games = primary["games"]
    recent_rows = primary_rows[-50:]
    recent = probability_metrics(recent_rows, "home_win_probability")
    drift = bool(
        games >= MINIMUM_DRIFT_SAMPLE
        and recent["log_loss"] is not None
        and recent["log_loss"] > REFERENCE_LOG_LOSS + 0.03
    )
    return {
        "model_version": "v0.36",
        "settled_future_games": games,
        "sample_status": sample_label(games),
        "minimum_games_for_drift_checks": MINIMUM_DRIFT_SAMPLE,
        "minimum_games_for_retraining_decisions": ACTIONABLE_SAMPLE,
        "primary_model": primary,
        "shadow_comparison": {
            "comparable_games": len(comparison_rows),
            "primary": comparable_primary,
            "team_only_shadow": shadow,
            "log_loss_advantage": (
                round(shadow["log_loss"] - comparable_primary["log_loss"], 4)
                if comparison_rows else None
            ),
        },
        "recent_50": recent,
        "drift_flag": drift,
        "factor_diagnostics": factor_diagnostics(connection),
        "daily_reports": daily[:30],
        "recommendation": (
            "Collect future locked predictions; the sample is too small for retraining."
            if games < ACTIONABLE_SAMPLE
            else "Review drift and shadow-model evidence before considering retraining."
        ),
    }


def train_shadow_model(paths: list[Path]) -> Path:
    grouped = group_rows_by_season(paths)
    rows = [row for season in sorted(grouped) for row in grouped[season]]
    pipeline = build_pipeline()
    pipeline.fit(feature_matrix(rows, ADVANCED_FEATURES), labels(rows))
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": ADVANCED_FEATURES,
            "model_name": "team_only_shadow_v0.30",
            "training_seasons": sorted(grouped),
            "training_games": len(rows),
        },
        SHADOW_MODEL_PATH,
    )
    return SHADOW_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--train-shadow", type=Path, nargs="+")
    args = parser.parse_args()
    if args.train_shadow:
        print(f"Shadow model: {train_shadow_model(args.train_shadow).relative_to(PROJECT_ROOT)}")
    connection = connect_database(args.database)
    try:
        report = learning_report(connection)
    finally:
        connection.close()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
