"""Record immutable pregame predictions and score them after games finish."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, fetch_schedule, parse_date

DEFAULT_DATABASE = PROJECT_ROOT / "data" / "prediction_ledger.sqlite3"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "tracking_report.json"
PREDICTION_COLUMNS = [
    "game_id",
    "official_date",
    "game_time_utc",
    "away_team",
    "home_team",
    "away_win_probability",
    "home_win_probability",
    "model_lean",
]
SCORE_COLUMNS = [
    "away_expected_runs",
    "home_expected_runs",
    "expected_total_runs",
]


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS predictions (
            game_id TEXT PRIMARY KEY,
            official_date TEXT NOT NULL,
            game_time_utc TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_win_probability REAL NOT NULL CHECK (away_win_probability BETWEEN 0 AND 1),
            home_win_probability REAL NOT NULL CHECK (home_win_probability BETWEEN 0 AND 1),
            model_lean TEXT NOT NULL,
            recorded_at_utc TEXT NOT NULL,
            previous_hash TEXT NOT NULL,
            record_hash TEXT NOT NULL UNIQUE,
            CHECK (ABS(away_win_probability + home_win_probability - 1.0) < 0.0002)
        );
        CREATE TABLE IF NOT EXISTS results (
            game_id TEXT PRIMARY KEY REFERENCES predictions(game_id),
            away_score INTEGER NOT NULL,
            home_score INTEGER NOT NULL,
            home_win INTEGER NOT NULL CHECK (home_win IN (0, 1)),
            status TEXT NOT NULL,
            settled_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS score_projections (
            game_id TEXT PRIMARY KEY REFERENCES predictions(game_id),
            away_expected_runs REAL NOT NULL CHECK (away_expected_runs >= 0),
            home_expected_runs REAL NOT NULL CHECK (home_expected_runs >= 0),
            expected_total_runs REAL NOT NULL CHECK (expected_total_runs >= 0),
            recorded_at_utc TEXT NOT NULL,
            projection_hash TEXT NOT NULL UNIQUE,
            CHECK (ABS(away_expected_runs + home_expected_runs - expected_total_runs) < 0.021)
        );
        CREATE TABLE IF NOT EXISTS shadow_predictions (
            game_id TEXT PRIMARY KEY REFERENCES predictions(game_id),
            model_version TEXT NOT NULL,
            away_win_probability REAL NOT NULL CHECK (away_win_probability BETWEEN 0 AND 1),
            home_win_probability REAL NOT NULL CHECK (home_win_probability BETWEEN 0 AND 1),
            model_lean TEXT NOT NULL,
            recorded_at_utc TEXT NOT NULL,
            CHECK (ABS(away_win_probability + home_win_probability - 1.0) < 0.0002)
        );
        CREATE TABLE IF NOT EXISTS prediction_context (
            game_id TEXT PRIMARY KEY REFERENCES predictions(game_id),
            model_version TEXT NOT NULL,
            factors_json TEXT NOT NULL,
            recorded_at_utc TEXT NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS predictions_no_update
        BEFORE UPDATE ON predictions BEGIN
            SELECT RAISE(ABORT, 'prediction records are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS predictions_no_delete
        BEFORE DELETE ON predictions BEGIN
            SELECT RAISE(ABORT, 'prediction records are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS results_no_update
        BEFORE UPDATE ON results BEGIN
            SELECT RAISE(ABORT, 'result records are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS results_no_delete
        BEFORE DELETE ON results BEGIN
            SELECT RAISE(ABORT, 'result records are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS score_projections_no_update
        BEFORE UPDATE ON score_projections BEGIN
            SELECT RAISE(ABORT, 'score projection records are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS score_projections_no_delete
        BEFORE DELETE ON score_projections BEGIN
            SELECT RAISE(ABORT, 'score projection records are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS shadow_predictions_no_update
        BEFORE UPDATE ON shadow_predictions BEGIN
            SELECT RAISE(ABORT, 'shadow predictions are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS shadow_predictions_no_delete
        BEFORE DELETE ON shadow_predictions BEGIN
            SELECT RAISE(ABORT, 'shadow predictions are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS prediction_context_no_update
        BEFORE UPDATE ON prediction_context BEGIN
            SELECT RAISE(ABORT, 'prediction context is immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS prediction_context_no_delete
        BEFORE DELETE ON prediction_context BEGIN
            SELECT RAISE(ABORT, 'prediction context is immutable');
        END;
        """
    )
    return connection


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def canonical_prediction(row: dict[str, Any], recorded_at: str, previous_hash: str) -> str:
    payload = {column: row[column] for column in PREDICTION_COLUMNS}
    payload["recorded_at_utc"] = recorded_at
    payload["previous_hash"] = previous_hash
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def canonical_score_projection(row: dict[str, Any], recorded_at: str) -> str:
    payload = {"game_id": str(row["game_id"])}
    payload.update({column: float(row[column]) for column in SCORE_COLUMNS})
    payload["recorded_at_utc"] = recorded_at
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _record_score_projection(
    connection: sqlite3.Connection, row: dict[str, Any], recorded_at: str
) -> str:
    """Insert or verify one immutable score projection."""
    if any(row.get(column) in (None, "") for column in SCORE_COLUMNS):
        return "missing"
    normalized = {column: float(row[column]) for column in SCORE_COLUMNS}
    existing = connection.execute(
        "SELECT * FROM score_projections WHERE game_id = ?", (row["game_id"],)
    ).fetchone()
    if existing:
        if any(
            not math.isclose(float(existing[column]), normalized[column], abs_tol=1e-9)
            for column in SCORE_COLUMNS
        ):
            raise ValueError(
                f"Score projection for game {row['game_id']} already exists and differs"
            )
        return "reused"
    payload = {"game_id": row["game_id"], **normalized}
    canonical = canonical_score_projection(payload, recorded_at)
    projection_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    connection.execute(
        "INSERT INTO score_projections VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(row["game_id"]),
            normalized["away_expected_runs"],
            normalized["home_expected_runs"],
            normalized["expected_total_runs"],
            recorded_at,
            projection_hash,
        ),
    )
    return "recorded"


def record_predictions(
    connection: sqlite3.Connection,
    rows: list[dict[str, str]],
    now: datetime | None = None,
    *,
    shadow_rows: dict[str, dict[str, Any]] | None = None,
    learning_context: dict[str, dict[str, Any]] | None = None,
    model_version: str = "v0.36",
    strict_existing: bool = True,
) -> dict[str, int]:
    """Insert only new pregame records; never replace an existing prediction."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    recorded = reused = preserved_conflicts = skipped_after_start = 0
    score_recorded = score_reused = score_skipped_after_start = 0
    shadow_recorded = context_recorded = 0
    latest = connection.execute(
        "SELECT record_hash FROM predictions ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    previous_hash = latest["record_hash"] if latest else "GENESIS"

    for row in rows:
        existing = connection.execute(
            "SELECT * FROM predictions WHERE game_id = ?", (row["game_id"],)
        ).fetchone()
        if existing:
            unchanged = all(str(existing[column]) == str(row[column]) for column in PREDICTION_COLUMNS)
            if not unchanged:
                if strict_existing:
                    raise ValueError(f"Prediction for game {row['game_id']} already exists and differs")
                preserved_conflicts += 1
                continue
            reused += 1
            if now < parse_utc(row["game_time_utc"]):
                score_status = _record_score_projection(
                    connection, row, now.isoformat().replace("+00:00", "Z")
                )
                score_recorded += int(score_status == "recorded")
                score_reused += int(score_status == "reused")
            elif all(row.get(column) not in (None, "") for column in SCORE_COLUMNS):
                score_skipped_after_start += 1
            continue
        if now >= parse_utc(row["game_time_utc"]):
            skipped_after_start += 1
            score_skipped_after_start += int(
                all(row.get(column) not in (None, "") for column in SCORE_COLUMNS)
            )
            continue

        normalized: dict[str, Any] = dict(row)
        normalized["away_win_probability"] = float(row["away_win_probability"])
        normalized["home_win_probability"] = float(row["home_win_probability"])
        recorded_at = now.isoformat().replace("+00:00", "Z")
        canonical = canonical_prediction(normalized, recorded_at, previous_hash)
        record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        connection.execute(
            """
            INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized["game_id"],
                normalized["official_date"],
                normalized["game_time_utc"],
                normalized["away_team"],
                normalized["home_team"],
                normalized["away_win_probability"],
                normalized["home_win_probability"],
                normalized["model_lean"],
                recorded_at,
                previous_hash,
                record_hash,
            ),
        )
        previous_hash = record_hash
        recorded += 1
        score_status = _record_score_projection(connection, row, recorded_at)
        score_recorded += int(score_status == "recorded")
        shadow = (shadow_rows or {}).get(str(row["game_id"]))
        if shadow:
            connection.execute(
                "INSERT INTO shadow_predictions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(row["game_id"]),
                    str(shadow.get("model_version", "team_only_v0.30")),
                    float(shadow["away_win_probability"]),
                    float(shadow["home_win_probability"]),
                    str(shadow["model_lean"]),
                    recorded_at,
                ),
            )
            shadow_recorded += 1
        context = (learning_context or {}).get(str(row["game_id"]))
        if context:
            factors = list(context.get("strongest_supporting_factors", [])) + list(
                context.get("strongest_opposing_factors", [])
            )
            connection.execute(
                "INSERT INTO prediction_context VALUES (?, ?, ?, ?)",
                (str(row["game_id"]), model_version, json.dumps(factors), recorded_at),
            )
            context_recorded += 1
    connection.commit()
    return {
        "recorded": recorded,
        "reused": reused,
        "preserved_conflicts": preserved_conflicts,
        "skipped_after_start": skipped_after_start,
        "score_recorded": score_recorded,
        "score_reused": score_reused,
        "score_skipped_after_start": score_skipped_after_start,
        "shadow_recorded": shadow_recorded,
        "context_recorded": context_recorded,
    }


def verify_hash_chain(connection: sqlite3.Connection) -> bool:
    previous_hash = "GENESIS"
    for row in connection.execute("SELECT * FROM predictions ORDER BY rowid"):
        if row["previous_hash"] != previous_hash:
            return False
        canonical = canonical_prediction(dict(row), row["recorded_at_utc"], previous_hash)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != row["record_hash"]:
            return False
        previous_hash = row["record_hash"]
    return True


def verify_score_projection_hashes(connection: sqlite3.Connection) -> bool:
    for row in connection.execute("SELECT * FROM score_projections ORDER BY game_id"):
        canonical = canonical_score_projection(dict(row), row["recorded_at_utc"])
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != row["projection_hash"]:
            return False
    return True


def final_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            if away.get("score") is None or home.get("score") is None:
                continue
            results.append(
                {
                    "game_id": str(game["gamePk"]),
                    "away_score": int(away["score"]),
                    "home_score": int(home["score"]),
                    "home_win": int(bool(home.get("isWinner"))),
                    "status": game.get("status", {}).get("detailedState", "Final"),
                }
            )
    return results


def settle_results(
    connection: sqlite3.Connection,
    results: list[dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, int]:
    now = (now or datetime.now(UTC)).astimezone(UTC).isoformat().replace("+00:00", "Z")
    settled = ignored_untracked = reused = 0
    for result in results:
        prediction = connection.execute(
            "SELECT game_id FROM predictions WHERE game_id = ?", (result["game_id"],)
        ).fetchone()
        if not prediction:
            ignored_untracked += 1
            continue
        existing = connection.execute(
            "SELECT * FROM results WHERE game_id = ?", (result["game_id"],)
        ).fetchone()
        if existing:
            same = all(existing[key] == result[key] for key in ("away_score", "home_score", "home_win", "status"))
            if not same:
                raise ValueError(f"Settled result for game {result['game_id']} differs")
            reused += 1
            continue
        connection.execute(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)",
            (
                result["game_id"],
                result["away_score"],
                result["home_score"],
                result["home_win"],
                result["status"],
                now,
            ),
        )
        settled += 1
    connection.commit()
    return {"settled": settled, "reused": reused, "ignored_untracked": ignored_untracked}


def performance_report(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT p.*, r.away_score, r.home_score, r.home_win
        FROM predictions p JOIN results r USING (game_id)
        ORDER BY p.official_date, p.game_time_utc
        """
    ).fetchall()
    pending = connection.execute(
        "SELECT COUNT(*) AS count FROM predictions p LEFT JOIN results r USING (game_id) WHERE r.game_id IS NULL"
    ).fetchone()["count"]
    score_rows = connection.execute(
        """
        SELECT s.*, r.away_score, r.home_score
        FROM score_projections s JOIN results r USING (game_id)
        ORDER BY s.game_id
        """
    ).fetchall()
    score_metrics = {
        "score_projection_games": len(score_rows),
        "score_mae": None,
        "score_rmse": None,
        "total_runs_mae": None,
        "score_projection_hashes_valid": verify_score_projection_hashes(connection),
    }
    if score_rows:
        side_errors = [
            error
            for row in score_rows
            for error in (
                float(row["away_expected_runs"]) - int(row["away_score"]),
                float(row["home_expected_runs"]) - int(row["home_score"]),
            )
        ]
        total_errors = [
            float(row["expected_total_runs"])
            - (int(row["away_score"]) + int(row["home_score"]))
            for row in score_rows
        ]
        score_metrics.update(
            {
                "score_mae": round(sum(abs(error) for error in side_errors) / len(side_errors), 4),
                "score_rmse": round(
                    math.sqrt(sum(error**2 for error in side_errors) / len(side_errors)), 4
                ),
                "total_runs_mae": round(
                    sum(abs(error) for error in total_errors) / len(total_errors), 4
                ),
            }
        )
    if not rows:
        return {
            "settled_games": 0,
            "pending_games": pending,
            "accuracy": None,
            "log_loss": None,
            "brier_score": None,
            "hash_chain_valid": verify_hash_chain(connection),
            **score_metrics,
        }
    correct = 0
    losses = []
    briers = []
    for row in rows:
        probability = float(row["home_win_probability"])
        outcome = int(row["home_win"])
        correct += int((probability >= 0.5) == bool(outcome))
        clipped = min(max(probability, 1e-15), 1 - 1e-15)
        losses.append(-(outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped)))
        briers.append((probability - outcome) ** 2)
    return {
        "settled_games": len(rows),
        "pending_games": pending,
        "accuracy": round(correct / len(rows), 4),
        "log_loss": round(sum(losses) / len(losses), 4),
        "brier_score": round(sum(briers) / len(briers), 4),
        "hash_chain_valid": verify_hash_chain(connection),
        **score_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--predictions", required=True, type=Path)
    settle_parser = subparsers.add_parser("settle")
    settle_parser.add_argument("--date", required=True, type=parse_date)
    subparsers.add_parser("report")
    args = parser.parse_args()

    connection = connect_database(args.database)
    try:
        if args.command == "record":
            summary = record_predictions(connection, read_predictions(args.predictions))
            print(
                f"Recorded {summary['recorded']}; reused {summary['reused']}; "
                f"skipped after start {summary['skipped_after_start']}."
            )
        elif args.command == "settle":
            try:
                payload = fetch_schedule(args.date)
            except requests.RequestException as exc:
                raise SystemExit(f"Could not download final results: {exc}") from exc
            summary = settle_results(connection, final_results(payload))
            print(
                f"Settled {summary['settled']}; reused {summary['reused']}; "
                f"ignored untracked {summary['ignored_untracked']}."
            )
        else:
            report = performance_report(connection)
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            print(f"Tracking report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    except ValueError as exc:
        raise SystemExit(f"Tracking refused the operation: {exc}") from exc
    finally:
        connection.close()


if __name__ == "__main__":
    main()
