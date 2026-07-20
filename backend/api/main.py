"""Expose MLBAI predictions and performance through a read-only API."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from backend.tracking.prediction_tracker import connect_database, performance_report

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_REPORT_PATH = PROJECT_ROOT / "docs" / "model_comparison_report.json"
LEDGER_PATH = PROJECT_ROOT / "data" / "prediction_ledger.sqlite3"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class Factor(BaseModel):
    factor: str
    feature: str
    raw_home_minus_away: Any
    log_odds_contribution: float


class GameAnalysis(BaseModel):
    game_id: str
    official_date: str
    game_time_utc: str
    away_team: str
    home_team: str
    away_win_probability: float
    home_win_probability: float
    model_lean: str
    certainty_band: str
    evidence_grade: str
    held_out_band_games: int
    held_out_band_accuracy: float | None
    missing_model_features: int
    data_quality: str
    intercept_log_odds: float
    strongest_supporting_factors: list[Factor]
    strongest_opposing_factors: list[Factor]
    reliability_note: str


class GamesResponse(BaseModel):
    date: str
    count: int
    games: list[GameAnalysis]


class PerformanceResponse(BaseModel):
    settled_games: int
    pending_games: int
    accuracy: float | None
    log_loss: float | None
    brier_score: float | None
    hash_chain_valid: bool


app = FastAPI(
    title="MLBAI API",
    version="0.18.0",
    description="Read-only access to MLBAI game analysis and model tracking.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Data not found: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"Could not read {path.name}") from error


def load_games(game_date: date) -> list[dict[str, Any]]:
    payload = load_json(PROCESSED_DATA_DIR / f"analysis_{game_date.isoformat()}.json")
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Analysis file has an invalid format")
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "MLBAI API", "version": app.version}


@app.get("/api/v1/games", response_model=GamesResponse)
def games(
    game_date: date = Query(default_factory=date.today, alias="date"),
) -> GamesResponse:
    rows = load_games(game_date)
    return GamesResponse(date=game_date.isoformat(), count=len(rows), games=rows)


@app.get("/api/v1/games/{game_id}", response_model=GameAnalysis)
def game(
    game_id: str,
    game_date: date = Query(default_factory=date.today, alias="date"),
) -> GameAnalysis:
    for row in load_games(game_date):
        if str(row.get("game_id")) == game_id:
            return GameAnalysis.model_validate(row)
    raise HTTPException(status_code=404, detail=f"Game {game_id} not found")


@app.get("/api/v1/performance", response_model=PerformanceResponse)
def performance() -> PerformanceResponse:
    if not LEDGER_PATH.is_file():
        raise HTTPException(status_code=404, detail="Prediction ledger not found")
    connection = connect_database(LEDGER_PATH)
    try:
        return PerformanceResponse.model_validate(performance_report(connection))
    finally:
        connection.close()


@app.get("/api/v1/model")
def model() -> dict[str, Any]:
    report = load_json(MODEL_REPORT_PATH)
    return {
        "selected_model": report.get("selected_model"),
        "selection_metric": report.get("selection_metric"),
        "untouched_test": report.get("untouched_test"),
        "reliability_note": (
            "Experimental classroom model. Probabilities are estimates, not guarantees or betting advice."
        ),
    }


# Keep this mount last so the API routes above always take priority.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="dashboard")
