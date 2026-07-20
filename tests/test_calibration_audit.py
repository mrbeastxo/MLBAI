import numpy as np

from ml.calibration_audit import calibration_error, deployment_decision


def test_calibration_error_is_zero_for_perfectly_matched_bins() -> None:
    result = calibration_error(
        np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0]), bins=2
    )
    assert result["expected_calibration_error"] == 0.0
    assert sum(row["games"] for row in result["bins"]) == 4


def test_deployment_requires_every_predeclared_gate() -> None:
    raw = {"accuracy": 0.55, "log_loss": 0.6900, "brier_score": 0.2480}
    passes = {"accuracy": 0.548, "log_loss": 0.6889, "brier_score": 0.2478}
    tiny_gain = {"accuracy": 0.548, "log_loss": 0.6895, "brier_score": 0.2478}
    assert deployment_decision(raw, passes)["decision"] == "deploy"
    assert deployment_decision(raw, tiny_gain)["decision"] == "reject"


def test_accuracy_drop_gate_can_reject_probability_improvement() -> None:
    raw = {"accuracy": 0.55, "log_loss": 0.6900, "brier_score": 0.2480}
    candidate = {"accuracy": 0.54, "log_loss": 0.6880, "brier_score": 0.2470}
    assert deployment_decision(raw, candidate)["decision"] == "reject"
