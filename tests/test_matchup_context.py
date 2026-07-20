from backend.data_pipeline.matchup_context import attach_matchup_context


def test_context_is_attached_without_changing_probability() -> None:
    analyses = [{
        "game_id": "1",
        "away_team": "Away",
        "home_team": "Home",
        "home_win_probability": 0.6,
        "away_win_probability": 0.4,
    }]
    pitchers = [{
        "game_id": "1", "side": "home", "pitcher_name": "Starter",
        "season_era": "3.20", "season_whip": "1.10",
    }]
    bullpens = [{"team_name": "Away", "workload_index": 90}]
    enriched, coverage = attach_matchup_context(analyses, pitchers, bullpens)
    assert enriched[0]["home_win_probability"] == 0.6
    assert enriched[0]["matchup_context"]["home_starter"]["name"] == "Starter"
    assert enriched[0]["matchup_context"]["probability_impact"] == "context_only"
    assert coverage["starter_coverage"] == 0.5
    assert coverage["bullpen_coverage"] == 0.5


def test_missing_context_is_explicit() -> None:
    analyses = [{
        "game_id": "1", "away_team": "Away", "home_team": "Home",
        "home_win_probability": 0.5, "away_win_probability": 0.5,
    }]
    enriched, coverage = attach_matchup_context(analyses, [], [], ["source unavailable"])
    assert enriched[0]["matchup_context"]["away_starter"]["announced"] is False
    assert enriched[0]["matchup_context"]["collection_errors"] == ["source unavailable"]
    assert coverage["starter_coverage"] == 0.0
