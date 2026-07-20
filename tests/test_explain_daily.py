import numpy as np
import pytest

from ml.baseline_model import build_pipeline
from ml.explain_daily import certainty_band, evidence_grade, logistic_contributions


def test_certainty_bands_are_symmetric() -> None:
    assert certainty_band(0.52) == "close"
    assert certainty_band(0.44) == "slight_lean"
    assert certainty_band(0.62) == "moderate_lean"
    assert certainty_band(0.70) == "strongest_lean"


def test_evidence_grade_requires_enough_games() -> None:
    assert evidence_grade({"games": 10, "observed_accuracy": 0.9}) == "insufficient_history"
    assert evidence_grade({"games": 100, "observed_accuracy": 0.53}) == "weak"


def test_logistic_contributions_reconstruct_decision_score() -> None:
    pipeline = build_pipeline()
    x = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    pipeline.fit(x, y)
    artifact = {"pipeline": pipeline, "features": ["feature"]}
    intercept, contributions = logistic_contributions({"feature": "1.0"}, artifact)
    reconstructed = intercept + contributions["feature"]
    assert reconstructed == pytest.approx(pipeline.decision_function([[1.0]])[0])
