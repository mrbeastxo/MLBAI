"""Measure each MLB bullpen's workload over the previous three days."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, SCHEDULE_URL, parse_date

API_BASE_URL = "https://statsapi.mlb.com/api/v1"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
BULLPEN_FIELDS = [
    "snapshot_date",
    "team_id",
    "team_name",
    "games_last_3_days",
    "relief_appearances_last_3_days",
    "relievers_used_last_3_days",
    "bullpen_pitches_yesterday",
    "bullpen_pitches_2_days_ago",
    "bullpen_pitches_3_days_ago",
    "bullpen_pitches_last_3_days",
    "bullpen_innings_last_3_days",
    "relievers_back_to_back",
    "relievers_used_2_of_3_days",
    "high_workload_relievers",
    "workload_index",
]


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_active_teams(season: int) -> dict[int, str]:
    """Return active MLB team IDs and names for a season."""
    payload = _get_json(
        f"{API_BASE_URL}/teams", {"sportId": 1, "season": season, "activeStatus": "Y"}
    )
    return {team["id"]: team["name"] for team in payload.get("teams", [])}


def fetch_recent_game_ids(snapshot_date: date) -> list[tuple[int, date]]:
    """Return completed game IDs from the three days before a snapshot."""
    start_date = snapshot_date - timedelta(days=3)
    end_date = snapshot_date - timedelta(days=1)
    payload = _get_json(
        SCHEDULE_URL,
        {
            "sportId": 1,
            "startDate": start_date.strftime("%m/%d/%Y"),
            "endDate": end_date.strftime("%m/%d/%Y"),
        },
    )
    return [
        (game["gamePk"], date.fromisoformat(date_entry["date"]))
        for date_entry in payload.get("dates", [])
        for game in date_entry.get("games", [])
        if game.get("status", {}).get("abstractGameState") == "Final"
    ]


def fetch_boxscore(game_id: int) -> dict[str, Any]:
    """Return the official box score for a completed game."""
    return _get_json(f"{API_BASE_URL}/game/{game_id}/boxscore")


def relief_appearances(
    boxscore: dict[str, Any], game_date: date
) -> list[dict[str, Any]]:
    """Extract only non-starting pitchers from both teams in a box score."""
    appearances: list[dict[str, Any]] = []
    for side in ("away", "home"):
        team_box = boxscore.get("teams", {}).get(side, {})
        team = team_box.get("team", {})
        players = team_box.get("players", {})
        for pitcher_id in team_box.get("pitchers", []):
            player = players.get(f"ID{pitcher_id}", {})
            pitching = player.get("stats", {}).get("pitching", {})
            if not pitching or pitching.get("gamesStarted", 0) != 0:
                continue
            appearances.append(
                {
                    "game_date": game_date,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "pitcher_id": pitcher_id,
                    "pitcher_name": player.get("person", {}).get("fullName"),
                    "pitches": int(pitching.get("numberOfPitches", 0)),
                    "outs": int(pitching.get("outs", 0)),
                }
            )
    return appearances


def _innings_from_outs(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def build_bullpen_rows(
    teams: dict[int, str],
    appearances: list[dict[str, Any]],
    game_dates_by_team: dict[int, set[date]],
    snapshot_date: date,
) -> list[dict[str, Any]]:
    """Aggregate relief appearances into one transparent workload row per team."""
    by_team: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for appearance in appearances:
        by_team[appearance["team_id"]].append(appearance)

    rows: list[dict[str, Any]] = []
    for team_id, team_name in teams.items():
        team_appearances = by_team[team_id]
        pitches_by_day = {
            days_ago: sum(
                item["pitches"]
                for item in team_appearances
                if item["game_date"] == snapshot_date - timedelta(days=days_ago)
            )
            for days_ago in (1, 2, 3)
        }
        pitcher_dates: dict[int, set[date]] = defaultdict(set)
        pitcher_pitches_last_2: dict[int, int] = defaultdict(int)
        for item in team_appearances:
            pitcher_dates[item["pitcher_id"]].add(item["game_date"])
            if item["game_date"] >= snapshot_date - timedelta(days=2):
                pitcher_pitches_last_2[item["pitcher_id"]] += item["pitches"]

        yesterday = snapshot_date - timedelta(days=1)
        two_days_ago = snapshot_date - timedelta(days=2)
        back_to_back = sum(
            yesterday in dates and two_days_ago in dates
            for dates in pitcher_dates.values()
        )
        used_2_of_3 = sum(len(dates) >= 2 for dates in pitcher_dates.values())
        high_workload = sum(
            pitches >= 40 or len(pitcher_dates[pitcher_id]) >= 2
            for pitcher_id, pitches in pitcher_pitches_last_2.items()
        )

        # A descriptive workload index, not a probability or learned rating.
        workload_index = round(
            pitches_by_day[1]
            + 0.5 * pitches_by_day[2]
            + 0.25 * pitches_by_day[3]
            + 20 * back_to_back
            + 10 * used_2_of_3,
            1,
        )
        rows.append(
            {
                "snapshot_date": snapshot_date.isoformat(),
                "team_id": team_id,
                "team_name": team_name,
                "games_last_3_days": len(game_dates_by_team.get(team_id, set())),
                "relief_appearances_last_3_days": len(team_appearances),
                "relievers_used_last_3_days": len(pitcher_dates),
                "bullpen_pitches_yesterday": pitches_by_day[1],
                "bullpen_pitches_2_days_ago": pitches_by_day[2],
                "bullpen_pitches_3_days_ago": pitches_by_day[3],
                "bullpen_pitches_last_3_days": sum(pitches_by_day.values()),
                "bullpen_innings_last_3_days": _innings_from_outs(
                    sum(item["outs"] for item in team_appearances)
                ),
                "relievers_back_to_back": back_to_back,
                "relievers_used_2_of_3_days": used_2_of_3,
                "high_workload_relievers": high_workload,
                "workload_index": workload_index,
            }
        )
    return sorted(rows, key=lambda row: row["team_name"])


def collect_bullpen_snapshot(season: int, snapshot_date: date) -> list[dict[str, Any]]:
    """Fetch recent games and produce workload rows for all active MLB teams."""
    teams = fetch_active_teams(season)
    games = fetch_recent_game_ids(snapshot_date)
    all_appearances: list[dict[str, Any]] = []
    game_dates_by_team: dict[int, set[date]] = defaultdict(set)

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_game = {
            executor.submit(fetch_boxscore, game_id): (game_id, game_date)
            for game_id, game_date in games
        }
        for future in as_completed(future_to_game):
            _, game_date = future_to_game[future]
            boxscore = future.result()
            for side in ("away", "home"):
                team_id = boxscore.get("teams", {}).get(side, {}).get("team", {}).get("id")
                if team_id:
                    game_dates_by_team[team_id].add(game_date)
            all_appearances.extend(relief_appearances(boxscore, game_date))

    return build_bullpen_rows(teams, all_appearances, game_dates_by_team, snapshot_date)


def save_bullpen_snapshot(rows: list[dict[str, Any]], snapshot_date: date) -> Path:
    """Write bullpen workloads to CSV and return the path."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"bullpen_workload_{snapshot_date.isoformat()}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BULLPEN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--date", type=parse_date, default=date.today())
    args = parser.parse_args()
    if args.date.year != args.season:
        parser.error("--date must fall within --season")

    try:
        rows = collect_bullpen_snapshot(args.season, args.date)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not download bullpen workload: {exc}") from exc

    output_path = save_bullpen_snapshot(rows, args.date)
    print(f"Collected three-day bullpen workload for {len(rows)} MLB teams.")
    for row in rows:
        print(
            f"- {row['team_name']}: {row['bullpen_pitches_last_3_days']} pitches, "
            f"{row['relievers_back_to_back']} back-to-back, "
            f"index {row['workload_index']}"
        )
    print(f"Bullpen snapshot saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
