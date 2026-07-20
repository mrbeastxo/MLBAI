from pathlib import Path

import pytest

from ml.multiseason_validation import expanding_season_splits


def test_expanding_season_splits_reserves_newest_season() -> None:
    grouped = {2022: [{}], 2023: [{}], 2024: [{}], 2025: [{}]}
    assert expanding_season_splits(grouped, 2025) == [
        ([2022], 2023),
        ([2022, 2023], 2024),
        ([2022, 2023, 2024], 2025),
    ]


def test_test_season_must_be_newest() -> None:
    with pytest.raises(ValueError, match="newest"):
        expanding_season_splits({2022: [{}], 2023: [{}], 2024: [{}]}, 2023)
