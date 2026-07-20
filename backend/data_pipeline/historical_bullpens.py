"""Reconstruct leakage-safe bullpen features from cached MLB box scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from backend.data_pipeline.historical_pitchers import BOX_SCORE_DIR
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
BULLPEN_METRICS = [
    "relief_appearances_before",
    "innings_before",
    "era_before",
    "whip_before",
    "strikeouts_per_9_before",
    "pitches_yesterday",
    "pitches_last_3_days",
    "relievers_back_to_back",
]
BULLPEN_FEATURE_FIELDS = (
    ["game_id", "official_date"]
    + [f"away_bullpen_{field}" for field in BULLPEN_METRICS]
    + [f"home_bullpen_{field}" for field in BULLPEN_METRICS]
    + [
        "bullpen_era_home_minus_away",
        "bullpen_whip_home_minus_away",
        "bullpen_k9_home_minus_away",
        "bullpen_pitches_last_3_home_minus_away",
        "bullpen_back_to_back_home_minus_away",
        "away_bullpen_history_missing",
        "home_bullpen_history_missing",
    ]
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def relief_lines(
    boxscore: dict[str, Any], game_id: str, game_date: date
) -> list[dict[str, Any]]:
    """Extract non-starting pitching lines from both teams."""
    lines: list[dict[str, Any]] = []
    for side in ("away", "home"):
        team_box = boxscore.get("teams", {}).get(side, {})
        team_id = str(team_box.get("team", {}).get("id"))
        players = team_box.get("players", {})
        for pitcher_id in team_box.get("pitchers", []):
            player = players.get(f"ID{pitcher_id}", {})
            stats = player.get("stats", {}).get("pitching", {})
            if not stats or stats.get("gamesStarted", 0) != 0:
                continue
            lines.append(
                {
                    "game_id": game_id,
                    "official_date": game_date.isoformat(),
                    "side": side,
                    "team_id": team_id,
                    "pitcher_id": str(pitcher_id),
                    "outs": int(stats.get("outs", 0)),
                    "earned_runs": int(stats.get("earnedRuns", 0)),
                    "hits": int(stats.get("hits", 0)),
                    "walks": int(stats.get("baseOnBalls", 0)),
                    "strikeouts": int(stats.get("strikeOuts", 0)),
                    "pitches": int(stats.get("numberOfPitches", 0)),
                }
            )
    return lines


def load_cached_games(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Load game/team identities and relief lines from the local cache."""
    games: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        game_id = manifest["game_id"]
        game_date = date.fromisoformat(manifest["official_date"])
        path = BOX_SCORE_DIR / f"{game_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached box score: {path}")
        boxscore = json.loads(path.read_text(encoding="utf-8"))
        teams = boxscore.get("teams", {})
        games.append(
            {
                "game_id": game_id,
                "official_date": game_date,
                "away_team_id": str(teams.get("away", {}).get("team", {}).get("id")),
                "home_team_id": str(teams.get("home", {}).get("team", {}).get("id")),
                "relief_lines": relief_lines(boxscore, game_id, game_date),
            }
        )
    return sorted(games, key=lambda game: (game["official_date"], game["game_id"]))


def _rate(numerator: int, outs: int, multiplier: int) -> float | str:
    return round(multiplier * numerator / outs, 3) if outs else ""


def _innings(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def summarize_bullpen(
    history: list[dict[str, Any]], game_date: date
) -> dict[str, Any]:
    """Summarize relief work strictly before the current official date."""
    prior = [
        line
        for line in history
        if date.fromisoformat(line["official_date"]) < game_date
    ]
    yesterday = game_date - timedelta(days=1)
    three_day_cutoff = game_date - timedelta(days=3)
    recent = [
        line
        for line in prior
        if date.fromisoformat(line["official_date"]) >= three_day_cutoff
    ]
    pitcher_dates: dict[str, set[date]] = defaultdict(set)
    for line in prior:
        pitcher_dates[line["pitcher_id"]].add(date.fromisoformat(line["official_date"]))
    two_days_ago = game_date - timedelta(days=2)
    outs = sum(line["outs"] for line in prior)
    return {
        "relief_appearances_before": len(prior),
        "innings_before": _innings(outs),
        "era_before": _rate(sum(line["earned_runs"] for line in prior), outs, 27),
        "whip_before": _rate(
            sum(line["hits"] + line["walks"] for line in prior), outs, 3
        ),
        "strikeouts_per_9_before": _rate(
            sum(line["strikeouts"] for line in prior), outs, 27
        ),
        "pitches_yesterday": sum(
            line["pitches"]
            for line in prior
            if date.fromisoformat(line["official_date"]) == yesterday
        ),
        "pitches_last_3_days": sum(line["pitches"] for line in recent),
        "relievers_back_to_back": sum(
            yesterday in dates and two_days_ago in dates for dates in pitcher_dates.values()
        ),
    }


def _difference(home: Any, away: Any) -> float | str:
    try:
        return round(float(home) - float(away), 4)
    except (TypeError, ValueError):
        return ""


def build_bullpen_feature_rows(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create pregame team bullpen rows, delaying all same-day updates."""
    games_by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        games_by_date[game["official_date"]].append(game)
    rows: list[dict[str, Any]] = []

    for game_date in sorted(games_by_date):
        day_games = games_by_date[game_date]
        for game in day_games:
            away = summarize_bullpen(histories[game["away_team_id"]], game_date)
            home = summarize_bullpen(histories[game["home_team_id"]], game_date)
            row: dict[str, Any] = {
                "game_id": game["game_id"],
                "official_date": game_date.isoformat(),
            }
            for side, summary in (("away", away), ("home", home)):
                for field in BULLPEN_METRICS:
                    row[f"{side}_bullpen_{field}"] = summary[field]
            row["bullpen_era_home_minus_away"] = _difference(
                home["era_before"], away["era_before"]
            )
            row["bullpen_whip_home_minus_away"] = _difference(
                home["whip_before"], away["whip_before"]
            )
            row["bullpen_k9_home_minus_away"] = _difference(
                home["strikeouts_per_9_before"], away["strikeouts_per_9_before"]
            )
            row["bullpen_pitches_last_3_home_minus_away"] = _difference(
                home["pitches_last_3_days"], away["pitches_last_3_days"]
            )
            row["bullpen_back_to_back_home_minus_away"] = _difference(
                home["relievers_back_to_back"], away["relievers_back_to_back"]
            )
            row["away_bullpen_history_missing"] = int(
                away["relief_appearances_before"] == 0
            )
            row["home_bullpen_history_missing"] = int(
                home["relief_appearances_before"] == 0
            )
            rows.append({field: row.get(field, "") for field in BULLPEN_FEATURE_FIELDS})

        for game in day_games:
            for line in game["relief_lines"]:
                histories[line["team_id"]].append(line)
    return sorted(rows, key=lambda row: (row["official_date"], row["game_id"]))


def save_bullpen_features(rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("No cached games were provided")
    start_date = min(row["official_date"] for row in rows)
    end_date = max(row["official_date"] for row in rows)
    path = PROCESSED_DATA_DIR / f"historical_bullpens_{start_date}_{end_date}.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BULLPEN_FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        games = load_cached_games(read_manifest(args.manifest))
        rows = build_bullpen_feature_rows(games)
        output_path = save_bullpen_features(rows)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Could not build historical bullpens: {exc}") from exc
    missing = sum(
        row["away_bullpen_history_missing"] + row["home_bullpen_history_missing"]
        for row in rows
    )
    print(f"Cached games processed: {len(games)}")
    print(f"Pregame bullpen rows: {len(rows)}")
    print(f"Team-game histories unavailable: {missing}")
    print(f"Bullpen features saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
