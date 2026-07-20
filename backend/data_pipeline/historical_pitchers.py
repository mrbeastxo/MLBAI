"""Cache box scores and reconstruct leakage-safe historical starter features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, SCHEDULE_URL, parse_date

API_BASE_URL = "https://statsapi.mlb.com/api/v1"
BOX_SCORE_DIR = PROJECT_ROOT / "data" / "raw" / "boxscores"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
STARTER_METRICS = [
    "pitcher_id",
    "pitcher_name",
    "starts_before",
    "innings_before",
    "era_before",
    "whip_before",
    "strikeouts_per_9_before",
    "walks_per_9_before",
    "home_runs_per_9_before",
    "last_5_era",
    "pitches_last_start",
    "days_since_last_start",
]
PITCHER_FEATURE_FIELDS = (
    ["game_id", "official_date"]
    + [f"away_starter_{field}" for field in STARTER_METRICS]
    + [f"home_starter_{field}" for field in STARTER_METRICS]
    + [
        "starter_era_home_minus_away",
        "starter_whip_home_minus_away",
        "starter_k9_home_minus_away",
        "starter_bb9_home_minus_away",
        "starter_last_5_era_home_minus_away",
        "away_starter_history_missing",
        "home_starter_history_missing",
    ]
)


def fetch_completed_game_ids(start_date: date, end_date: date) -> list[tuple[int, date]]:
    """Return unique final regular-season games in an inclusive range."""
    response = requests.get(
        SCHEDULE_URL,
        params={
            "sportId": 1,
            "gameType": "R",
            "startDate": start_date.strftime("%m/%d/%Y"),
            "endDate": end_date.strftime("%m/%d/%Y"),
        },
        timeout=60,
    )
    response.raise_for_status()
    return completed_game_ids(response.json(), start_date, end_date)


def completed_game_ids(
    payload: dict[str, Any], start_date: date, end_date: date
) -> list[tuple[int, date]]:
    """Filter rescheduled placeholders using each game's official date."""
    games: dict[int, date] = {}
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("abstractGameState") == "Final":
                official_date = date.fromisoformat(
                    game.get("officialDate", date_entry["date"])
                )
                if start_date <= official_date <= end_date:
                    games[game["gamePk"]] = official_date
    return sorted(games.items(), key=lambda item: (item[1], item[0]))


def fetch_boxscore(game_id: int) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/game/{game_id}/boxscore", timeout=30)
    response.raise_for_status()
    return response.json()


def cache_boxscore(game_id: int) -> tuple[int, bool]:
    """Download one missing box score; return whether a request was needed."""
    BOX_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    path = BOX_SCORE_DIR / f"{game_id}.json"
    if path.exists():
        return game_id, False
    payload = fetch_boxscore(game_id)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return game_id, True


def cache_boxscores(games: list[tuple[int, date]], workers: int = 8) -> tuple[int, int]:
    """Cache box scores concurrently and support safe restart."""
    downloaded = 0
    reused = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(cache_boxscore, game_id) for game_id, _ in games]
        for future in as_completed(futures):
            _, was_downloaded = future.result()
            downloaded += int(was_downloaded)
            reused += int(not was_downloaded)
    return downloaded, reused


def starter_appearances(
    boxscore: dict[str, Any], game_id: int, game_date: date
) -> list[dict[str, Any]]:
    """Extract the actual starting pitcher for each side of one completed game."""
    appearances: list[dict[str, Any]] = []
    for side in ("away", "home"):
        team_box = boxscore.get("teams", {}).get(side, {})
        players = team_box.get("players", {})
        for pitcher_id in team_box.get("pitchers", []):
            player = players.get(f"ID{pitcher_id}", {})
            stats = player.get("stats", {}).get("pitching", {})
            if stats.get("gamesStarted") != 1:
                continue
            appearances.append(
                {
                    "game_id": str(game_id),
                    "official_date": game_date.isoformat(),
                    "side": side,
                    "team_id": str(team_box.get("team", {}).get("id")),
                    "pitcher_id": str(pitcher_id),
                    "pitcher_name": player.get("person", {}).get("fullName"),
                    "outs": int(stats.get("outs", 0)),
                    "earned_runs": int(stats.get("earnedRuns", 0)),
                    "hits": int(stats.get("hits", 0)),
                    "walks": int(stats.get("baseOnBalls", 0)),
                    "strikeouts": int(stats.get("strikeOuts", 0)),
                    "home_runs": int(stats.get("homeRuns", 0)),
                    "pitches": int(stats.get("numberOfPitches", 0)),
                }
            )
            break
    return appearances


def load_cached_appearances(games: list[tuple[int, date]]) -> list[dict[str, Any]]:
    appearances: list[dict[str, Any]] = []
    for game_id, game_date in games:
        path = BOX_SCORE_DIR / f"{game_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        appearances.extend(starter_appearances(payload, game_id, game_date))
    return sorted(
        appearances,
        key=lambda item: (item["official_date"], item["game_id"], item["side"]),
    )


def _rate(numerator: int, outs: int, multiplier: int = 3) -> float | str:
    return round(multiplier * numerator / outs, 3) if outs else ""


def _innings(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def summarize_pitcher(history: list[dict[str, Any]], game_date: date) -> dict[str, Any]:
    """Calculate pitcher totals using only starts before ``game_date``."""
    prior = [start for start in history if date.fromisoformat(start["official_date"]) < game_date]
    outs = sum(start["outs"] for start in prior)
    last_five = prior[-5:]
    last_five_outs = sum(start["outs"] for start in last_five)
    last_start = prior[-1] if prior else None
    return {
        "starts_before": len(prior),
        "innings_before": _innings(outs),
        "era_before": _rate(sum(start["earned_runs"] for start in prior), outs, 27),
        "whip_before": _rate(
            sum(start["walks"] + start["hits"] for start in prior), outs, 3
        ),
        "strikeouts_per_9_before": _rate(
            sum(start["strikeouts"] for start in prior), outs, 27
        ),
        "walks_per_9_before": _rate(sum(start["walks"] for start in prior), outs, 27),
        "home_runs_per_9_before": _rate(
            sum(start["home_runs"] for start in prior), outs, 27
        ),
        "last_5_era": _rate(
            sum(start["earned_runs"] for start in last_five), last_five_outs, 27
        ),
        "pitches_last_start": last_start["pitches"] if last_start else "",
        "days_since_last_start": (
            game_date - date.fromisoformat(last_start["official_date"])
        ).days
        if last_start
        else "",
    }


def _difference(home: Any, away: Any) -> float | str:
    try:
        return round(float(home) - float(away), 4)
    except (TypeError, ValueError):
        return ""


def build_pitcher_feature_rows(
    appearances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build game-level starter features before updating that day's histories."""
    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for appearance in appearances:
        by_date[date.fromisoformat(appearance["official_date"])].append(appearance)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for game_date in sorted(by_date):
        day_appearances = by_date[game_date]
        by_game: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for appearance in day_appearances:
            by_game[appearance["game_id"]][appearance["side"]] = appearance

        for game_id, sides in by_game.items():
            row: dict[str, Any] = {
                "game_id": game_id,
                "official_date": game_date.isoformat(),
            }
            summaries: dict[str, dict[str, Any]] = {}
            for side in ("away", "home"):
                appearance = sides.get(side)
                if appearance:
                    summary = summarize_pitcher(
                        histories[appearance["pitcher_id"]], game_date
                    )
                    summary.update(
                        {
                            "pitcher_id": appearance["pitcher_id"],
                            "pitcher_name": appearance["pitcher_name"],
                        }
                    )
                else:
                    summary = {field: "" for field in STARTER_METRICS}
                summaries[side] = summary
                for field in STARTER_METRICS:
                    row[f"{side}_starter_{field}"] = summary.get(field, "")

            row["starter_era_home_minus_away"] = _difference(
                summaries["home"].get("era_before"), summaries["away"].get("era_before")
            )
            row["starter_whip_home_minus_away"] = _difference(
                summaries["home"].get("whip_before"), summaries["away"].get("whip_before")
            )
            row["starter_k9_home_minus_away"] = _difference(
                summaries["home"].get("strikeouts_per_9_before"),
                summaries["away"].get("strikeouts_per_9_before"),
            )
            row["starter_bb9_home_minus_away"] = _difference(
                summaries["home"].get("walks_per_9_before"),
                summaries["away"].get("walks_per_9_before"),
            )
            row["starter_last_5_era_home_minus_away"] = _difference(
                summaries["home"].get("last_5_era"),
                summaries["away"].get("last_5_era"),
            )
            row["away_starter_history_missing"] = int(
                summaries["away"].get("starts_before", 0) == 0
            )
            row["home_starter_history_missing"] = int(
                summaries["home"].get("starts_before", 0) == 0
            )
            rows.append({field: row.get(field, "") for field in PITCHER_FEATURE_FIELDS})

        # Same-day starts are added only after every matchup row is built.
        for appearance in day_appearances:
            histories[appearance["pitcher_id"]].append(appearance)
    return sorted(rows, key=lambda row: (row["official_date"], row["game_id"]))


def save_pitcher_features(
    rows: list[dict[str, Any]], start_date: date, end_date: date
) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / (
        f"historical_pitchers_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    )
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PITCHER_FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("--end-date cannot be earlier than --start-date")
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be between 1 and 12")

    try:
        games = fetch_completed_game_ids(args.start_date, args.end_date)
        downloaded, reused = cache_boxscores(games, args.workers)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not build box-score archive: {exc}") from exc
    appearances = load_cached_appearances(games)
    rows = build_pitcher_feature_rows(appearances)
    output_path = save_pitcher_features(rows, args.start_date, args.end_date)
    missing = sum(
        row["away_starter_history_missing"] + row["home_starter_history_missing"]
        for row in rows
    )
    print(f"Completed games: {len(games)}")
    print(f"Box scores downloaded: {downloaded}; reused from cache: {reused}")
    print(f"Pregame pitcher matchup rows: {len(rows)}")
    print(f"Starter histories unavailable: {missing}")
    print(f"Pitcher features saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
