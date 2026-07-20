"""Attach expected away/home runs to daily game rows."""

from __future__ import annotations

from typing import Any

from ml.baseline_model import feature_matrix


def predict_scores(
    rows: list[dict[str, str]], artifact: dict[str, Any]
) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    matrix = feature_matrix(rows, artifact["features"])
    away = artifact["models"]["away"].predict(matrix)
    home = artifact["models"]["home"].predict(matrix)
    return {
        row["game_id"]: {
            "away_expected_runs": round(max(0.0, float(away_runs)), 2),
            "home_expected_runs": round(max(0.0, float(home_runs)), 2),
            "expected_total_runs": round(max(0.0, float(away_runs + home_runs)), 2),
        }
        for row, away_runs, home_runs in zip(rows, away, home, strict=True)
    }


def attach_scores(
    games: list[dict[str, Any]], scores: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    return [
        {
            **game,
            **scores.get(str(game["game_id"]), {}),
            "score_projection_note": (
                "Expected runs are model averages, not an exact final-score prediction."
                if str(game["game_id"]) in scores
                else None
            ),
        }
        for game in games
    ]
