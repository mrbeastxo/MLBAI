from datetime import date

import requests

from backend.data_pipeline import matchup_context
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


def test_collection_reuses_same_date_cache_after_api_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(matchup_context, "PROCESSED_DATA_DIR", tmp_path)
    (tmp_path / "starting_pitchers_2026-07-20.csv").write_text(
        "game_id,side,pitcher_name\n1,away,Cached Starter\n"
    )
    (tmp_path / "bullpen_workload_2026-07-20.csv").write_text(
        "team_name,workload_index\nAway,88\n"
    )

    def unavailable(*args, **kwargs):
        raise requests.ConnectTimeout("temporary outage")

    monkeypatch.setattr(matchup_context, "collect_starting_pitchers", unavailable)
    monkeypatch.setattr(matchup_context, "collect_bullpen_snapshot", unavailable)
    pitchers, bullpens, errors = matchup_context.collect_context_snapshot(
        2026, date(2026, 7, 20)
    )
    assert pitchers[0]["pitcher_name"] == "Cached Starter"
    assert bullpens[0]["workload_index"] == "88"
    assert len(errors) == 2
