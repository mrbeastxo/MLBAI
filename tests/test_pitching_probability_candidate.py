from ml.pitching_probability_candidate import gate_report


def validation(accuracy=0.55, log_loss=0.69, brier=0.25):
    return {
        "final_untouched_test": {
            "model_metrics": {
                "accuracy": accuracy,
                "log_loss": log_loss,
                "brier_score": brier,
            }
        }
    }


def test_candidate_requires_coverage_and_probability_improvement() -> None:
    advanced = validation()
    candidate = validation(accuracy=0.55, log_loss=0.688, brier=0.249)
    report = gate_report(advanced, candidate, {2022: 2400, 2023: 2400, 2024: 2400})
    assert report["decision"] == "deploy"
    assert report["untouched_test"]["log_loss_improvement"] == 0.002

    assert gate_report(advanced, candidate, {2022: 100, 2023: 2400})["decision"] == "context_only"
    assert gate_report(advanced, validation(log_loss=0.70), {2022: 2400})["decision"] == "context_only"
