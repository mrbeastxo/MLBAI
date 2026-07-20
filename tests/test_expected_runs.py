import numpy as np

from ml.expected_runs import score_metrics
from ml.predict_scores import attach_scores, predict_scores


class FixedModel:
    def __init__(self, value):
        self.value = value

    def predict(self, matrix):
        return np.full(len(matrix), self.value)


def test_score_metrics_reward_closer_run_estimates() -> None:
    actual_away = np.array([3.0, 5.0])
    actual_home = np.array([4.0, 2.0])
    exact = score_metrics(actual_away, actual_home, actual_away, actual_home)
    flat = score_metrics(
        actual_away, actual_home, np.array([4.0, 4.0]), np.array([4.0, 4.0])
    )
    assert exact["mae"] == 0.0
    assert exact["mean_poisson_deviance"] == 0.0
    assert flat["mae"] > exact["mae"]


def test_predict_and_attach_scores() -> None:
    row = {
        "game_id": "1",
        "official_date": "2026-07-20",
        "game_time_utc": "2026-07-20T20:00:00Z",
        "away_team_name": "Away",
        "home_team_name": "Home",
        "strength": "0",
    }
    artifact = {
        "features": ["strength"],
        "models": {"away": FixedModel(4.25), "home": FixedModel(4.75)},
    }
    scores = predict_scores([row], artifact)
    games = attach_scores([{"game_id": "1"}], scores)
    assert games[0]["away_expected_runs"] == 4.25
    assert games[0]["home_expected_runs"] == 4.75
    assert games[0]["expected_total_runs"] == 9.0
    assert "not an exact" in games[0]["score_projection_note"]
