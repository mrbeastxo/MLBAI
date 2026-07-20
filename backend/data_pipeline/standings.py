"""Collect current American and National League tables from MLB."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LEAGUES = {103: "American League", 104: "National League"}


def fetch_standings(season: int) -> dict[str, Any]:
    response = requests.get(
        STANDINGS_URL,
        params={
            "leagueId": "103,104",
            "season": season,
            "standingsTypes": "regularSeason",
            "hydrate": "team",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _last_ten(team_record: dict[str, Any]) -> str:
    splits = team_record.get("records", {}).get("splitRecords", [])
    split = next((item for item in splits if item.get("type") == "lastTen"), {})
    return f"{split.get('wins', 0)}-{split.get('losses', 0)}"


def build_tables(payload: dict[str, Any], season: int) -> dict[str, Any]:
    tables: dict[int, dict[str, dict[str, Any]]] = {league_id: {} for league_id in LEAGUES}
    last_updated = None
    for record in payload.get("records", []):
        last_updated = max(filter(None, [last_updated, record.get("lastUpdated")]), default=None)
        for team_record in record.get("teamRecords", []):
            team = team_record.get("team", {})
            league_id = int((team.get("league") or record.get("league") or {}).get("id", 0))
            if league_id not in tables:
                continue
            tables[league_id][str(team.get("id"))] = {
                "rank": int(team_record.get("leagueRank") or 0),
                "team_id": str(team.get("id")),
                "team": team.get("name"),
                "abbreviation": team.get("abbreviation"),
                "division": (team.get("division") or {}).get("name"),
                "wins": int(team_record.get("wins") or 0),
                "losses": int(team_record.get("losses") or 0),
                "winning_percentage": float(team_record.get("winningPercentage") or 0),
                "games_back": team_record.get("leagueGamesBack", "-"),
                "run_differential": int(team_record.get("runDifferential") or 0),
                "streak": (team_record.get("streak") or {}).get("streakCode", "—"),
                "last_ten": _last_ten(team_record),
            }
    return {
        "season": season,
        "last_updated": last_updated,
        "leagues": [
            {
                "league_id": league_id,
                "league": name,
                "teams": sorted(tables[league_id].values(), key=lambda row: row["rank"]),
            }
            for league_id, name in LEAGUES.items()
        ],
    }


def standings_path(season: int) -> Path:
    return PROCESSED_DATA_DIR / f"standings_{season}.json"


def collect_with_cache(season: int) -> tuple[dict[str, Any], list[str]]:
    path = standings_path(season)
    try:
        tables = build_tables(fetch_standings(season), season)
        path.write_text(json.dumps(tables, indent=2), encoding="utf-8")
        return tables, []
    except requests.RequestException as error:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")), [str(error)]
        return {"season": season, "last_updated": None, "leagues": []}, [str(error)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    args = parser.parse_args()
    tables, errors = collect_with_cache(args.season)
    print(f"League tables: {sum(len(item['teams']) for item in tables['leagues'])} teams; errors: {len(errors)}")
    print(f"Saved to: {standings_path(args.season).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
