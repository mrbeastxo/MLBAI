# MLBAI

MLBAI is a learning-focused MLB analytics project. Version 0.1 collects the
official daily MLB schedule and stores the raw response locally. Predictions
and machine-learning models will be added only after the data pipeline is
reliable and testable.

## Current milestone

- Python 3.12 virtual environment
- Official MLB Stats API schedule collector
- Completed-game results collector with model-ready CSV output
- Thirty-team season snapshots with records, splits, hitting, and pitching
- Human-readable game list in the terminal
- Raw JSON snapshots saved under `data/raw/`
- Starter automated tests

## Quick start (macOS)

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m backend.data_pipeline.mlb_schedule
```

To collect a specific date:

```bash
python -m backend.data_pipeline.mlb_schedule --date 2026-07-20
```

To collect completed games for one date:

```bash
python -m backend.data_pipeline.completed_games --start-date 2025-07-20
```

Or use an inclusive date range:

```bash
python -m backend.data_pipeline.completed_games \
  --start-date 2025-07-18 --end-date 2025-07-20
```

To collect a current team-statistics snapshot:

```bash
python -m backend.data_pipeline.team_stats --season 2026 --date 2026-07-20
```

Snapshots must be handled carefully during model training: a prediction may
only use statistics that were available before that game's start time.

The command prints the games it finds and saves the complete API response to
`data/raw/schedule_YYYY-MM-DD.json`.

## Test

```bash
pytest
```

## Project structure

```text
backend/       Python application and data collectors
data/raw/      Local API snapshots (not committed to Git)
data/processed Future cleaned datasets
docs/          Design and class-presentation notes
frontend/      Future web dashboard
ml/            Future feature engineering and model training
models/        Future trained model artifacts
tests/         Automated tests
```

## Data source

Schedule data is requested from the MLB Stats API at
`https://statsapi.mlb.com/api/v1/schedule`.

## Responsible use

Model outputs will be probabilities, not guarantees. The goal is honest sports
analysis with out-of-sample evaluation, calibration, and transparent reasoning.
