from backend.data_pipeline.lineup_context import (
    attach_lineup_context,
    hitting_summary,
    side_context,
)


def profile(player_id: int, name: str, ops: str, plate_appearances: int) -> dict:
    return {
        "id": player_id,
        "fullName": name,
        "primaryPosition": {"abbreviation": "CF"},
        "batSide": {"code": "L"},
        "stats": [
            {
                "group": {"displayName": "hitting"},
                "splits": [
                    {
                        "stat": {
                            "plateAppearances": plate_appearances,
                            "avg": ".280",
                            "obp": ".360",
                            "slg": ".500",
                            "ops": ops,
                            "homeRuns": 12,
                        }
                    }
                ],
            }
        ],
    }


def test_hitting_summary_uses_season_stats() -> None:
    summary = hitting_summary(profile(10, "Test Hitter", ".860", 200))
    assert summary["name"] == "Test Hitter"
    assert summary["ops"] == 0.86
    assert summary["plate_appearances"] == 200
    assert summary["bats"] == "L"


def test_side_context_distinguishes_confirmed_order_from_roster_watch() -> None:
    profiles = {
        str(player_id): hitting_summary(
            profile(player_id, f"Hitter {player_id}", f".{700 + player_id}", 100)
        )
        for player_id in range(1, 10)
    }
    players = {
        f"ID{player_id}": {
            "person": {"id": player_id},
            "position": {"type": "Infielder"},
        }
        for player_id in range(1, 10)
    }
    confirmed = side_context(
        {"team": {"id": 1, "name": "Club"}, "battingOrder": list(range(1, 10)), "players": players},
        profiles,
    )
    assert confirmed["confirmed"] is True
    assert confirmed["batting_order"][0]["batting_spot"] == 1

    waiting = side_context(
        {"team": {"id": 1, "name": "Club"}, "battingOrder": [], "players": players},
        profiles,
    )
    assert waiting["confirmed"] is False
    assert len(waiting["roster_watch"]) == 3


def test_lineup_context_does_not_change_probability() -> None:
    game = {"game_id": "1", "home_win_probability": 0.61}
    row = {
        "game_id": "1",
        "away": {"confirmed": False},
        "home": {"confirmed": True},
    }
    enriched, coverage = attach_lineup_context([game], [row], [])
    assert enriched[0]["home_win_probability"] == 0.61
    assert enriched[0]["lineup_context"]["probability_impact"] == "context_only"
    assert coverage["confirmed_lineup_sides"] == 1
