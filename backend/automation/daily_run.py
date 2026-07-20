"""Run the complete leakage-safe daily prediction and tracking workflow."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import requests

from backend.data_pipeline.historical_training import fetch_season_games
from backend.data_pipeline.mlb_schedule import PROJECT_ROOT, fetch_schedule, parse_date
from backend.data_pipeline.pregame_features import PREGAME_FIELDS, build_pregame_rows
from backend.tracking.prediction_tracker import (
    DEFAULT_DATABASE,
    REPORT_PATH,
    connect_database,
    final_results,
    performance_report,
    record_predictions,
    settle_results,
)
from ml.explain_daily import DEFAULT_REPORT, explain_rows
from ml.predict_daily import MODEL_PATH, PREDICTION_FIELDS, predict_rows

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FetchSchedule = Callable[[date], dict[str, Any]]
FetchSeason = Callable[[int, date], dict[str, Any]]


def pending_dates(connection, before: date) -> list[date]:
    rows = connection.execute(
        """
        SELECT DISTINCT p.official_date
        FROM predictions p LEFT JOIN results r USING (game_id)
        WHERE r.game_id IS NULL AND p.official_date < ?
        ORDER BY p.official_date
        """,
        (before.isoformat(),),
    ).fetchall()
    return [date.fromisoformat(row["official_date"]) for row in rows]


def settle_pending(
    connection,
    run_date: date,
    schedule_fetcher: FetchSchedule = fetch_schedule,
) -> dict[str, Any]:
    """Try every pending past date without blocking today's predictions."""
    totals = {"settled": 0, "reused": 0, "ignored_untracked": 0}
    errors: list[dict[str, str]] = []
    checked: list[str] = []
    for pending_date in pending_dates(connection, run_date):
        checked.append(pending_date.isoformat())
        try:
            summary = settle_results(
                connection, final_results(schedule_fetcher(pending_date))
            )
        except requests.RequestException as error:
            errors.append({"date": pending_date.isoformat(), "error": str(error)})
            continue
        for key in totals:
            totals[key] += summary[key]
    return {"dates_checked": checked, **totals, "errors": errors}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def display_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def run_daily(
    run_date: date,
    *,
    database_path: Path = DEFAULT_DATABASE,
    model_path: Path = MODEL_PATH,
    model_report_path: Path = DEFAULT_REPORT,
    output_dir: Path = PROCESSED_DATA_DIR,
    now: datetime | None = None,
    dry_run: bool = False,
    season_fetcher: FetchSeason = fetch_season_games,
    schedule_fetcher: FetchSchedule = fetch_schedule,
) -> dict[str, Any]:
    """Execute one restart-safe daily run and return its audit summary."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    started_at = now.isoformat().replace("+00:00", "Z")
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"daily_run_{run_date.isoformat()}.json"
    summary: dict[str, Any] = {
        "date": run_date.isoformat(),
        "started_at_utc": started_at,
        "status": "running",
        "dry_run": dry_run,
    }
    connection = connect_database(database_path)
    try:
        summary["settlement"] = settle_pending(
            connection, run_date, schedule_fetcher=schedule_fetcher
        )
        season_payload = season_fetcher(run_date.year, run_date)
        features = build_pregame_rows(season_payload, run_date)
        feature_path = output_dir / f"pregame_features_{run_date.isoformat()}.csv"
        write_csv(feature_path, PREGAME_FIELDS, features)

        predictions: list[dict[str, Any]] = []
        analyses: list[dict[str, Any]] = []
        if features:
            if not model_path.is_file():
                raise FileNotFoundError(f"Production model not found: {model_path}")
            if not model_report_path.is_file():
                raise FileNotFoundError(f"Model report not found: {model_report_path}")
            artifact = joblib.load(model_path)
            model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
            predictions = predict_rows(features, artifact)
            analyses = explain_rows(features, artifact, model_report)

        prediction_path = output_dir / f"predictions_{run_date.isoformat()}.csv"
        analysis_path = output_dir / f"analysis_{run_date.isoformat()}.json"
        write_csv(prediction_path, PREDICTION_FIELDS, predictions)
        write_json(analysis_path, analyses)

        tracking = (
            {"recorded": 0, "reused": 0, "skipped_after_start": 0}
            if dry_run
            else record_predictions(connection, predictions, now)
        )
        report = performance_report(connection)
        write_json(output_dir / REPORT_PATH.name, report)
        summary.update(
            {
                "status": "success",
                "scheduled_games": len(features),
                "predictions_generated": len(predictions),
                "tracking": tracking,
                "performance": report,
                "outputs": {
                    "features": display_path(feature_path),
                    "predictions": display_path(prediction_path),
                    "analysis": display_path(analysis_path),
                },
                "finished_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
        )
    except Exception as error:
        summary.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "finished_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
        )
        write_json(status_path, summary)
        raise
    finally:
        connection.close()
    write_json(status_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, default=date.today())
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate files and metrics without recording new predictions",
    )
    args = parser.parse_args()
    try:
        summary = run_daily(
            args.date,
            database_path=args.database,
            model_path=args.model,
            model_report_path=args.model_report,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, requests.RequestException) as error:
        raise SystemExit(f"Daily run failed: {error}") from error
    print(json.dumps(summary, indent=2))
    print(
        f"Daily run report saved to: "
        f"data/processed/daily_run_{args.date.isoformat()}.json"
    )


if __name__ == "__main__":
    main()
