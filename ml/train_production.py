"""Refit the development-selected model on all evaluated historical seasons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from backend.data_pipeline.mlb_schedule import PROJECT_ROOT
from ml.baseline_model import ADVANCED_FEATURES, feature_matrix, labels
from ml.model_comparison import build_candidate
from ml.multiseason_validation import group_rows_by_season

MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_SELECTION_REPORT = PROJECT_ROOT / "docs" / "model_comparison_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION_REPORT)
    args = parser.parse_args()

    report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    model_name = report["selected_model"]
    grouped = group_rows_by_season(args.data)
    rows = [row for season in sorted(grouped) for row in grouped[season]]
    pipeline = build_candidate(model_name)
    pipeline.fit(feature_matrix(rows, ADVANCED_FEATURES), labels(rows))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "production_model.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": ADVANCED_FEATURES,
            "model_name": model_name,
            "training_seasons": sorted(grouped),
            "training_games": len(rows),
        },
        path,
    )
    print(f"Refit {model_name} on {len(rows)} games from seasons {sorted(grouped)}.")
    print(f"Production model saved to: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
