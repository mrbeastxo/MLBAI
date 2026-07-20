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
- Probable starters with season performance, rest, and recent workload
- Three-day bullpen workload derived from official game box scores
- One model-ready daily feature row per scheduled game
- Leakage-safe historical training rows with final outcome labels
- Chronologically evaluated logistic-regression baseline
- Expanding multi-season validation with a fully untouched newest season
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

To collect announced starting pitchers for a game date:

```bash
python -m backend.data_pipeline.starting_pitchers \
  --season 2026 --date 2026-07-20
```

Some teams may be absent when MLB has not announced their probable starter yet.

To measure bullpen workload before a game date:

```bash
python -m backend.data_pipeline.bullpen_workload \
  --season 2026 --date 2026-07-20
```

The workload index is descriptive, not a probability: yesterday's pitches are
weighted most heavily, followed by older pitches and repeat-use penalties.

After collecting all snapshots for a date, combine them with:

```bash
python -m backend.data_pipeline.daily_features --date 2026-07-20
```

The resulting table uses stable MLB IDs, separate `home_` and `away_` features,
and explicit home-minus-away differences. It contains inputs only—no invented
prediction or outcome label.

To build historical training rows for a completed date range:

```bash
python -m backend.data_pipeline.historical_training \
  --start-date 2025-07-18 --end-date 2025-07-20
```

Historical features are reconstructed chronologically from prior results. The
current game's result is attached only as the `home_win` training label, and
same-day games are updated together to prevent doubleheader leakage. Current
team, pitcher, and bullpen snapshots are intentionally not joined to old games.

Build a full completed season and train the baseline:

```bash
python -m backend.data_pipeline.historical_training \
  --start-date 2025-03-27 --end-date 2025-09-28

python -m ml.baseline_model \
  --data data/processed/training_games_2025-03-27_2025-09-28.csv
```

The split is chronological by whole date. Logistic regression is compared with
a constant probability learned from the training period's home-win rate using
accuracy, log loss, Brier score, and calibration bins.

For stronger evaluation, build separate season files and run:

```bash
python -m ml.multiseason_validation \
  --data data/processed/training_games_2022-04-07_2022-10-05.csv \
         data/processed/training_games_2023-03-30_2023-10-01.csv \
         data/processed/training_games_2024-03-20_2024-09-30.csv \
         data/processed/training_games_2025-03-27_2025-09-28.csv \
  --test-season 2025
```

Earlier seasons form expanding validation folds. The newest supplied season is
evaluated once as the untouched final test set.

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
