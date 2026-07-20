"""Collect confirmed batting orders and evidence-backed roster watchlists."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, fetch_schedule, parse_date

API_BASE_URL = "https://statsapi.mlb.com/api/v1"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def fetch_boxscore(game_id: int) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/game/{game_id}/boxscore", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_people(person_ids: list[int], season: int) -> list[dict[str, Any]]:
    if not person_ids:
        return []
    response = requests.get(
        f"{API_BASE_URL}/people",
        params={
            "personIds": ",".join(str(person_id) for person_id in person_ids),
            "hydrate": f"stats(group=[hitting],type=[season],season={season})",
        },
        timeout=45,
    )
    response.raise_for_status()
    return response.json().get("people", [])


def hitting_summary(person: dict[str, Any]) -> dict[str, Any]:
    splits = [
        split
        for block in person.get("stats", [])
        if block.get("group", {}).get("displayName") == "hitting"
        for split in block.get("splits", [])
    ]
    stats = splits[0].get("stat", {}) if splits else {}

    def number(key: str) -> float | None:
        try:
            return float(stats[key])
        except (KeyError, TypeError, ValueError):
            return None

    return {
        "player_id": str(person.get("id")),
        "name": person.get("fullName"),
        "position": person.get("primaryPosition", {}).get("abbreviation"),
        "bats": person.get("batSide", {}).get("code"),
        "plate_appearances": int(stats.get("plateAppearances", 0) or 0),
        "avg": number("avg"),
        "obp": number("obp"),
        "slg": number("slg"),
        "ops": number("ops"),
        "home_runs": int(stats.get("homeRuns", 0) or 0),
    }


def hitter_ids(boxscore: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for side in ("away", "home"):
        players = boxscore.get("teams", {}).get(side, {}).get("players", {})
        for player in players.values():
            if player.get("position", {}).get("type") == "Pitcher":
                continue
            player_id = player.get("person", {}).get("id")
            if player_id:
                ids.add(int(player_id))
    return ids


def side_context(
    team_box: dict[str, Any], profiles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    order = [str(player_id) for player_id in team_box.get("battingOrder", [])]
    confirmed = len(order) >= 9
    players = team_box.get("players", {})
    roster_ids = [
        str(player.get("person", {}).get("id"))
        for player in players.values()
        if player.get("position", {}).get("type") != "Pitcher"
        and player.get("person", {}).get("id")
    ]
    roster = [profiles[player_id] for player_id in roster_ids if player_id in profiles]
    eligible = [player for player in roster if player.get("plate_appearances", 0) >= 50]
    leaders = sorted(
        eligible or roster,
        key=lambda player: player.get("ops") if player.get("ops") is not None else -1,
        reverse=True,
    )[:3]
    batting_order = []
    if confirmed:
        for spot, player_id in enumerate(order[:9], start=1):
            player = dict(profiles.get(player_id, {"player_id": player_id, "name": "Unknown"}))
            player["batting_spot"] = spot
            batting_order.append(player)
    return {
        "team_id": str(team_box.get("team", {}).get("id")),
        "team_name": team_box.get("team", {}).get("name"),
        "confirmed": confirmed,
        "confirmed_hitters": len(order),
        "batting_order": batting_order,
        "roster_watch": leaders,
    }


def build_lineup_rows(
    games: list[dict[str, Any]],
    boxscores: dict[str, dict[str, Any]],
    people: list[dict[str, Any]],
    game_date: date,
) -> list[dict[str, Any]]:
    profiles = {summary["player_id"]: summary for summary in map(hitting_summary, people)}
    rows = []
    for game in games:
        game_id = str(game["gamePk"])
        boxscore = boxscores.get(game_id, {})
        teams = boxscore.get("teams", {})
        rows.append(
            {
                "snapshot_date": game_date.isoformat(),
                "game_id": game_id,
                "game_time_utc": game.get("gameDate"),
                "away": side_context(teams.get("away", {}), profiles),
                "home": side_context(teams.get("home", {}), profiles),
            }
        )
    return rows


def collect_lineup_snapshot(
    season: int, game_date: date
) -> tuple[list[dict[str, Any]], list[str]]:
    schedule = fetch_schedule(game_date)
    games = [game for entry in schedule.get("dates", []) for game in entry.get("games", [])]
    boxscores: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_boxscore, game["gamePk"]): game for game in games}
        for future in as_completed(futures):
            game = futures[future]
            try:
                boxscores[str(game["gamePk"])] = future.result()
            except requests.RequestException as error:
                errors.append(f"game {game['gamePk']} boxscore: {error}")
    ids = sorted({player_id for boxscore in boxscores.values() for player_id in hitter_ids(boxscore)})
    people: list[dict[str, Any]] = []
    chunks = [ids[index : index + 50] for index in range(0, len(ids), 50)]
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_people, chunk, season): chunk for chunk in chunks}
        for future in as_completed(futures):
            try:
                people.extend(future.result())
            except requests.RequestException as error:
                errors.append(f"hitter profiles: {error}")
    return build_lineup_rows(games, boxscores, people, game_date), errors


def snapshot_path(game_date: date) -> Path:
    return PROCESSED_DATA_DIR / f"lineup_context_{game_date.isoformat()}.json"


def save_lineup_snapshot(rows: list[dict[str, Any]], game_date: date) -> Path:
    path = snapshot_path(game_date)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def read_lineup_snapshot(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def collect_lineups_with_cache(
    season: int, game_date: date
) -> tuple[list[dict[str, Any]], list[str]]:
    path = snapshot_path(game_date)
    try:
        rows, errors = collect_lineup_snapshot(season, game_date)
        if rows:
            save_lineup_snapshot(rows, game_date)
        return rows or read_lineup_snapshot(path), errors
    except requests.RequestException as error:
        return read_lineup_snapshot(path), [f"lineups: {error}"]


def attach_lineup_context(
    analyses: list[dict[str, Any]], rows: list[dict[str, Any]], errors: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_game = {str(row["game_id"]): row for row in rows}
    confirmed_sides = 0
    output = []
    for game in analyses:
        row = by_game.get(str(game["game_id"]))
        if row:
            confirmed_sides += int(row["away"]["confirmed"]) + int(row["home"]["confirmed"])
        output.append(
            {
                **game,
                "lineup_context": (
                    {
                        **row,
                        "probability_impact": "context_only",
                        "note": (
                            "Confirmed orders come from MLB. Roster watchlists are not predicted lineups, "
                            "and lineups do not affect production probabilities yet."
                        ),
                    }
                    if row
                    else None
                ),
            }
        )
    possible = len(analyses) * 2
    return output, {
        "possible_lineup_sides": possible,
        "confirmed_lineup_sides": confirmed_sides,
        "confirmed_coverage": round(confirmed_sides / possible, 4) if possible else 0.0,
        "game_context_coverage": round(len(rows) / len(analyses), 4) if analyses else 0.0,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--date", type=parse_date, default=date.today())
    args = parser.parse_args()
    rows, errors = collect_lineup_snapshot(args.season, args.date)
    path = save_lineup_snapshot(rows, args.date)
    confirmed = sum(int(row[side]["confirmed"]) for row in rows for side in ("away", "home"))
    print(f"Games: {len(rows)}; confirmed lineup sides: {confirmed}; errors: {len(errors)}")
    print(f"Saved to: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
