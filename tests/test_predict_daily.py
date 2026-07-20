import numpy as np

from ml.predict_daily import predict_rows


class FakePipeline:
    def predict_proba(self, matrix):
        return np.array([[0.4, 0.6]])


def test_predict_rows_outputs_complementary_probabilities() -> None:
    rows = [
        {
            "game_id": "1",
            "official_date": "2026-07-20",
            "game_time_utc": "2026-07-20T20:00:00Z",
            "away_team_name": "Away",
            "home_team_name": "Home",
            "feature": "1.0",
        }
    ]
    predictions = predict_rows(
        rows, {"features": ["feature"], "pipeline": FakePipeline()}
    )
    assert predictions[0]["away_win_probability"] == 0.4
    assert predictions[0]["home_win_probability"] == 0.6
    assert predictions[0]["model_lean"] == "Home"
