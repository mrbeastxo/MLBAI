from ml.context_feature_gate import deployment_report


def report(games=1200, model_loss=0.68, baseline_loss=0.69):
    return {
        "model_selection_folds": [{
            "train_seasons": [2023], "train_games": games,
            "validation_season": 2024, "validation_games": games,
        }],
        "final_untouched_test": {
            "test_season": 2025,
            "validation_games": games,
            "model_metrics": {"log_loss": model_loss},
            "home_rate_baseline_metrics": {"log_loss": baseline_loss},
        },
    }


def test_gate_deploys_only_with_coverage_and_both_quality_wins() -> None:
    assert deployment_report(report(), report())["decision"] == "deploy"
    assert deployment_report(report(), report(games=200))["decision"] == "context_only"
    assert deployment_report(report(model_loss=0.70), report())["decision"] == "context_only"
    assert deployment_report(report(), report(model_loss=0.70))["decision"] == "context_only"
