import pytest

from ml.model_comparison import (
    MODEL_NAMES,
    build_candidate,
    select_by_average_log_loss,
)


def test_candidate_registry_builds_every_model() -> None:
    for name in MODEL_NAMES:
        assert build_candidate(name).named_steps["model"] is not None


def test_selection_uses_mean_log_loss_not_accuracy() -> None:
    results = {
        "logistic_regression": [
            {"metrics": {"accuracy": 0.50, "log_loss": 0.60}},
            {"metrics": {"accuracy": 0.50, "log_loss": 0.62}},
        ],
        "random_forest": [
            {"metrics": {"accuracy": 0.70, "log_loss": 0.70}},
            {"metrics": {"accuracy": 0.70, "log_loss": 0.72}},
        ],
        "hist_gradient_boosting": [
            {"metrics": {"accuracy": 0.55, "log_loss": 0.65}},
            {"metrics": {"accuracy": 0.55, "log_loss": 0.66}},
        ],
    }
    assert select_by_average_log_loss(results) == "logistic_regression"


def test_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        build_candidate("future_leaking_super_model")
