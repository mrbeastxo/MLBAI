import numpy as np

from ml.baseline_model import (
    FEATURES,
    calibration_bins,
    chronological_split,
    feature_matrix,
)


def _row(day: str, value: float = 0.0) -> dict[str, str]:
    row = {
        "official_date": day,
        "game_time_utc": f"{day}T20:00:00Z",
        "home_win": "1",
    }
    row.update({feature: str(value) for feature in FEATURES})
    return row


def test_chronological_split_keeps_dates_together() -> None:
    rows = [
        _row("2025-04-01"),
        _row("2025-04-02"),
        _row("2025-04-03"),
        _row("2025-04-04"),
        _row("2025-04-05"),
        _row("2025-04-05"),
    ]
    train, test, test_start = chronological_split(rows, train_fraction=0.8)
    assert test_start == "2025-04-05"
    assert {row["official_date"] for row in train}.isdisjoint(
        {row["official_date"] for row in test}
    )


def test_feature_matrix_uses_only_approved_features() -> None:
    row = _row("2025-04-01", value=1.5)
    row["home_score"] = "99"
    matrix = feature_matrix([row])
    assert matrix.shape == (1, len(FEATURES))
    assert np.all(matrix == 1.5)


def test_calibration_bins_reports_observed_rate() -> None:
    bins = calibration_bins(np.array([0, 1]), np.array([0.1, 0.9]), bin_count=2)
    assert bins[0]["observed_home_win_rate"] == 0.0
    assert bins[1]["observed_home_win_rate"] == 1.0
