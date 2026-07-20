"""Convert expected runs into transparent game-outcome distributions."""

from __future__ import annotations

import math
from typing import Any


def poisson_distribution(mean: float, max_runs: int = 20) -> list[float]:
    """Return 0..max_runs probabilities, placing the tiny upper tail last."""
    mean = max(0.01, float(mean))
    probabilities = [math.exp(-mean)]
    for runs in range(1, max_runs):
        probabilities.append(probabilities[-1] * mean / runs)
    probabilities.append(max(0.0, 1.0 - sum(probabilities)))
    return probabilities


def quantile_range(probabilities: list[float], lower=0.1, upper=0.9) -> list[int]:
    cumulative = 0.0
    low = high = len(probabilities) - 1
    found_low = False
    for runs, probability in enumerate(probabilities):
        cumulative += probability
        if not found_low and cumulative >= lower:
            low, found_low = runs, True
        if cumulative >= upper:
            high = runs
            break
    return [low, high]


def outcome_distribution(
    away_expected_runs: float, home_expected_runs: float
) -> dict[str, Any]:
    """Return exact independent-Poisson summaries without random simulation noise."""
    away = poisson_distribution(away_expected_runs)
    home = poisson_distribution(home_expected_runs)
    away_regulation = home_regulation = tie = 0.0
    for away_runs, away_probability in enumerate(away):
        for home_runs, home_probability in enumerate(home):
            probability = away_probability * home_probability
            if home_runs > away_runs:
                home_regulation += probability
            elif away_runs > home_runs:
                away_regulation += probability
            else:
                tie += probability
    away_mode = max(range(len(away)), key=away.__getitem__)
    home_mode = max(range(len(home)), key=home.__getitem__)
    return {
        "method": "independent_poisson_exact_distribution",
        "most_likely_score": {"away": away_mode, "home": home_mode},
        "away_runs_80_percent_range": quantile_range(away),
        "home_runs_80_percent_range": quantile_range(home),
        "regulation_away_win_probability": round(away_regulation, 4),
        "regulation_home_win_probability": round(home_regulation, 4),
        "extra_innings_probability": round(tie, 4),
        "score_model_home_win_probability": round(home_regulation + 0.5 * tie, 4),
        "probability_impact": "analysis_context_only",
        "note": (
            "Ranges describe model uncertainty. The score-derived probability is "
            "kept separate from the production win model unless validation gates pass."
        ),
    }


def attach_uncertainty(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for game in games:
        away = game.get("away_expected_runs")
        home = game.get("home_expected_runs")
        output.append(
            {
                **game,
                "outcome_uncertainty": (
                    outcome_distribution(float(away), float(home))
                    if away is not None and home is not None
                    else None
                ),
            }
        )
    return output
