from ml.context_model_upgrade import STARTER_FEATURES, build_pipeline


def test_context_model_uses_regularized_logistic_pipeline() -> None:
    pipeline = build_pipeline()
    assert pipeline.named_steps["model"].C == 0.01
    assert STARTER_FEATURES == [
        "starter_era_home_minus_away",
        "starter_whip_home_minus_away",
        "starter_k9_home_minus_away",
        "starter_bb9_home_minus_away",
    ]
