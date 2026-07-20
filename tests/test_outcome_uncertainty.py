from ml.outcome_uncertainty import (
    attach_uncertainty,
    outcome_distribution,
    poisson_distribution,
)


def test_poisson_distribution_is_normalized() -> None:
    probabilities = poisson_distribution(4.5)
    assert len(probabilities) == 21
    assert abs(sum(probabilities) - 1.0) < 1e-12
    assert all(probability >= 0 for probability in probabilities)


def test_equal_run_expectations_produce_symmetric_outcome() -> None:
    result = outcome_distribution(4.2, 4.2)
    assert result["score_model_home_win_probability"] == 0.5
    total = (
        result["regulation_away_win_probability"]
        + result["regulation_home_win_probability"]
        + result["extra_innings_probability"]
    )
    assert abs(total - 1.0) < 0.001
    assert result["probability_impact"] == "analysis_context_only"


def test_attach_uncertainty_preserves_game_projection() -> None:
    game = {
        "game_id": "1",
        "away_expected_runs": 3.8,
        "home_expected_runs": 5.0,
    }
    enriched = attach_uncertainty([game])[0]
    assert enriched["away_expected_runs"] == 3.8
    assert enriched["outcome_uncertainty"]["most_likely_score"]["away"] >= 0
    assert enriched["outcome_uncertainty"]["home_runs_80_percent_range"][0] <= 5
