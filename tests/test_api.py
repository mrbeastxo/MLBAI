import json

from fastapi.testclient import TestClient

from backend.api import main
from backend.tracking.prediction_tracker import connect_database


SAMPLE_GAME = {
    "game_id": "123",
    "official_date": "2026-07-20",
    "game_time_utc": "2026-07-20T23:00:00Z",
    "away_team": "Away Club",
    "home_team": "Home Club",
    "away_win_probability": 0.4,
    "home_win_probability": 0.6,
    "model_lean": "Home Club",
    "certainty_band": "moderate",
    "evidence_grade": "limited",
    "held_out_band_games": 20,
    "held_out_band_accuracy": 0.55,
    "missing_model_features": 0,
    "data_quality": "complete",
    "intercept_log_odds": 0.1,
    "strongest_supporting_factors": [],
    "strongest_opposing_factors": [],
    "reliability_note": "Experimental model.",
}


def configure_test_data(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "analysis_2026-07-20.json").write_text(
        json.dumps([SAMPLE_GAME]), encoding="utf-8"
    )
    model_report = tmp_path / "model.json"
    model_report.write_text(
        json.dumps(
            {
                "selected_model": "logistic_regression",
                "selection_metric": "log_loss",
                "untouched_test": {"log_loss": 0.69},
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.sqlite3"
    connect_database(ledger).close()
    monkeypatch.setattr(main, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(main, "MODEL_REPORT_PATH", model_report)
    monkeypatch.setattr(main, "LEDGER_PATH", ledger)


def test_health():
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_is_served():
    response = TestClient(main.app).get("/")
    assert response.status_code == 200
    assert "Every matchup" in response.text
    assert "MLBAI" in response.text


def test_games_and_game_detail(tmp_path, monkeypatch):
    configure_test_data(tmp_path, monkeypatch)
    client = TestClient(main.app)
    listing = client.get("/api/v1/games?date=2026-07-20")
    detail = client.get("/api/v1/games/123?date=2026-07-20")
    assert listing.status_code == 200
    assert listing.json()["count"] == 1
    assert detail.status_code == 200
    assert detail.json()["model_lean"] == "Home Club"


def test_missing_analysis_returns_404(tmp_path, monkeypatch):
    configure_test_data(tmp_path, monkeypatch)
    response = TestClient(main.app).get("/api/v1/games?date=2026-07-19")
    assert response.status_code == 404


def test_performance_and_model(tmp_path, monkeypatch):
    configure_test_data(tmp_path, monkeypatch)
    client = TestClient(main.app)
    performance = client.get("/api/v1/performance")
    model = client.get("/api/v1/model")
    assert performance.status_code == 200
    assert performance.json()["hash_chain_valid"] is True
    assert model.status_code == 200
    assert model.json()["selected_model"] == "logistic_regression"


def test_system_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        main,
        "system_health",
        lambda: {
            "scheduler": {"installed": True, "schedule": "06:00"},
            "last_run": {"status": "success"},
            "logs": {"has_errors": False},
            "storage": {"data_bytes": 100},
            "project_root": "/project",
        },
    )
    response = TestClient(main.app).get("/api/v1/system")
    assert response.status_code == 200
    assert response.json()["scheduler"]["installed"] is True
