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
- Advanced leakage-safe Elo, last-30, Pythagorean, and streak features
- Resumable historical box-score archive and pregame starter features
- Pitcher-feature joins with explicit matchup and history coverage
- Historical bullpen performance and workload from cached box scores
- Development-fold model selection with one untouched-season test
- Refit production model and daily rolling win-probability pipeline
- Exact logistic explanations and held-out certainty-band evidence
- Immutable pregame prediction ledger with final-result scoring
- Read-only FastAPI service for analysis, model details, and performance
- Responsive daily dashboard with matchup cards and factor explanations
- Restart-safe one-command daily prediction and settlement workflow
- Native macOS daily scheduling with local logs and lifecycle controls
- Read-only automation health, run history, log, and storage monitoring
- Complete current-season results archive with verified prediction comparisons
- Development-only probability calibration audit with strict deployment gates
- Live starting-pitcher and bullpen context with validation-based deployment gate
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

After regenerating the season files, evaluate the advanced feature set by
adding this option to the same multi-season command:

```bash
--feature-set advanced
```

The baseline feature set remains available for a direct comparison.

Build cached historical starting-pitcher features for a completed range:

```bash
python -m backend.data_pipeline.historical_pitchers \
  --start-date 2025-04-01 --end-date 2025-04-07
```

Every box score is cached by MLB game ID. Re-running the command reuses existing
files, and starter totals are calculated only from starts on earlier dates.

Join an archived pitcher range to its season training file:

```bash
python -m backend.data_pipeline.join_pitcher_features \
  --training data/processed/training_games_2025-03-27_2025-09-28.csv \
  --pitchers data/processed/historical_pitchers_2025-04-01_2025-04-07.csv
```

Use `--feature-set pitcher` with multi-season validation only after equivalent
date ranges have been archived and joined for every supplied season.

Build bullpen features from an existing pitcher-feature manifest, join them,
then use `--feature-set combined` for validation. Bullpen rows include prior
ERA, WHIP, K/9, three-day pitches, and back-to-back reliever usage.

Compare logistic regression, random forest, and histogram gradient boosting on
the full advanced team dataset with:

```bash
python -m ml.model_comparison \
  --data data/processed/training_games_2022-04-07_2022-10-05.csv \
         data/processed/training_games_2023-03-30_2023-10-01.csv \
         data/processed/training_games_2024-03-20_2024-09-30.csv \
         data/processed/training_games_2025-03-27_2025-09-28.csv \
  --test-season 2025
```

Selection uses mean development-fold log loss. Only the winner is evaluated on
the untouched newest season.

After model selection, refit on all evaluated seasons, build today's rolling
features, and generate estimates:

```bash
python -m ml.train_production --data data/processed/training_games_2022-04-07_2022-10-05.csv data/processed/training_games_2023-03-30_2023-10-01.csv data/processed/training_games_2024-03-20_2024-09-30.csv data/processed/training_games_2025-03-27_2025-09-28.csv
python -m backend.data_pipeline.pregame_features --date 2026-07-20
python -m ml.predict_daily --features data/processed/pregame_features_2026-07-20.csv
```

Daily probabilities are model estimates, not guaranteed outcomes or betting
advice. Same-day results are excluded from the rolling feature history.

Explain daily estimates using exact standardized logistic contributions:

```bash
python -m ml.explain_daily \
  --features data/processed/pregame_features_2026-07-20.csv
```

The analysis separates probability strength from held-out evidence, reports
missing inputs, and retains an experimental-model reliability warning.

Record predictions before game time, settle them later, and view performance:

```bash
python -m backend.tracking.prediction_tracker record \
  --predictions data/processed/predictions_2026-07-20.csv
python -m backend.tracking.prediction_tracker settle --date 2026-07-20
python -m backend.tracking.prediction_tracker report
```

Predictions and results are append-only, protected by SQLite triggers, and
linked by SHA-256 hashes. The ledger refuses changed duplicates and late rows.

Start the local API with:

```bash
python -m uvicorn backend.api.main:app --reload
```

Then open `http://127.0.0.1:8000` for the MLBAI dashboard, or
`http://127.0.0.1:8000/docs` for the interactive API page. The API
provides health, dated game analysis, individual game detail, selected-model
information, and prediction-ledger performance. It exposes no write endpoint.

The command prints the games it finds and saves the complete API response to
`data/raw/schedule_YYYY-MM-DD.json`.

Run the complete daily workflow with one command:

```bash
python -m backend.automation.daily_run
```

It settles past tracked games, reconstructs leakage-safe features, generates
probabilities and explanations, records only pregame predictions, refreshes
performance, and writes `data/processed/daily_run_YYYY-MM-DD.json`. Re-running
it is safe: matching immutable predictions are reused, while older locked
predictions from a different model version are preserved and reported without
stopping the rest of the scheduled refresh. Use
`--date YYYY-MM-DD` for a specific date or `--dry-run` to generate outputs
without adding predictions to the ledger.

Schedule that workflow for 9:00 AM and 9:00 PM local time every day on macOS:

```bash
python -m backend.automation.macos_scheduler install
```

Check or remove the schedule with:

```bash
python -m backend.automation.macos_scheduler status
python -m backend.automation.macos_scheduler uninstall
```

Use `install --hour 7 --minute 30` to choose one different local time, or repeat
`--hour` to configure several times. Standard
output and errors are stored under `data/logs/`. The Mac must be awake with an
internet connection; if it is asleep at the scheduled time, macOS normally runs
the missed calendar job after it wakes.

The dashboard system-health section shows the installed schedule, next run,
latest workflow result, local storage use, and whether the scheduler has an
unresolved error newer than the latest successful run. The same read-only snapshot is available at
`/api/v1/system`; it cannot start, stop, or modify the scheduler.

Refresh the current season's completed-game archive manually with:

```bash
python -m backend.history.season_results --season 2026 --through-date 2026-07-20
```

The daily workflow refreshes this archive automatically. The Performance Center
can filter and paginate official results while labeling old games as untracked.
Only predictions present in the immutable pregame ledger and subsequently
settled count as verified MLBAI performance.

Run the calibration audit with:

```bash
python -m ml.calibration_audit \
  --data data/processed/training_games_2022-04-07_2022-10-05.csv \
         data/processed/training_games_2023-03-30_2023-10-01.csv \
         data/processed/training_games_2024-03-20_2024-09-30.csv \
         data/processed/training_games_2025-03-27_2025-09-28.csv \
  --test-season 2025
```

The calibrator is fit only on development out-of-fold predictions. Deployment
requires at least 0.001 lower audit log loss, no worse Brier score, and no more
than a 0.005 accuracy decrease. A rejected candidate never changes production.

Pitcher and bullpen deployment is evaluated with:

```bash
python -m ml.context_feature_gate
```

The current historical pilot does not meet full-season coverage or log-loss
quality gates, so these signals are displayed as analysis context only. The
daily workflow collects announced starters, season pitching stats, rest,
recent workload, and three-day bullpen fatigue without changing the validated
win probability. The dashboard labels this distinction explicitly.

Milestone 26 replaces that pilot with a compact full-season backfill and tests
a live-compatible pitcher/bullpen candidate against the current advanced model
on the exact same games. The newest season remains untouched until the final
test. Deployment eligibility requires at least 2,000 games in every season,
0.001 lower test log loss, no worse Brier score, and no more than a one-point
accuracy decrease. Failing any gate leaves production probabilities unchanged.

```bash
python -m backend.data_pipeline.historical_context_backfill \
  --seasons 2022 2023 2024 2025
python -m ml.pitching_probability_candidate \
  --data data/processed/training_pitching_bullpen_2022.csv \
         data/processed/training_pitching_bullpen_2023.csv \
         data/processed/training_pitching_bullpen_2024.csv \
         data/processed/training_pitching_bullpen_2025.csv \
  --test-season 2025
```

### Milestone 27: expected runs

MLBAI trains separate Poisson-regression models for away and home scoring and
shows their conditional mean as a projected score. Validation remains
chronological by complete season. On the untouched 2025 test, the model reduced
mean absolute run error from 2.5402 to 2.5195 and mean Poisson deviance from
2.4148 to 2.3924 versus league-average home/away scoring baselines. Projected
scores are model averages—not promises of an exact final score.

```bash
python -m ml.expected_runs \
  --data data/processed/training_games_2022-04-07_2022-10-05.csv \
         data/processed/training_games_2023-03-30_2023-10-01.csv \
         data/processed/training_games_2024-03-20_2024-09-30.csv \
         data/processed/training_games_2025-03-27_2025-09-28.csv \
  --test-season 2025
```

### Milestone 28: score-projection accountability

Pregame expected-run projections are stored in their own immutable, hashed
SQLite ledger without changing the original probability hash chain. Once MLB
marks a tracked game final, MLBAI reports per-team score MAE/RMSE and total-runs
MAE through `/api/v1/performance` and the dashboard. Historical games and
projections generated after first pitch are never presented as verified model
performance.

### Milestone 29: outcome uncertainty

Expected runs now produce an exact independent-Poisson score distribution with
a most-likely score, an 80% run range for each team, and the probability that
regulation ends tied. The derived winner probability was audited on 2,428
untouched 2025 games and performed worse than the production feature model
(0.6923 versus 0.6892 log loss), so it is explicitly labeled analysis context
and does not change MLBAI's production win probability.

### Milestone 30: ballpark and weather intelligence

The daily workflow collects official MLB venue metadata—stadium, city, roof,
and playing surface—and matches each park's coordinates and first-pitch time to
an hourly Open-Meteo forecast. Temperature, condition, rain chance, wind, and
gusts are cached for every matchup and displayed as context only. They do not
change production probabilities until equivalent historical weather is
backfilled and validated chronologically.

Weather data: `https://open-meteo.com/` (CC BY 4.0).

### Milestone 31: confirmed lineup intelligence

The daily workflow now checks official MLB boxscores for confirmed batting
orders and displays each hitter's batting side, position, and current-season
AVG, OBP, SLG, OPS, and home runs. Before an order is confirmed, the dashboard
shows a clearly labeled roster watch based on season OPS; it never presents
those players as a predicted lineup. Same-date cached data keeps the section
available during a temporary API failure.

Lineup information is analysis context only. It does not change the production
win probability until equivalent historical lineup data is backfilled and its
effect is proven through chronological out-of-sample testing.

### Milestones 32–36: validated context model

MLBAI backfilled 9,719 regular-season games from 2022–2025 and compared the
same regularized model with team-only, starter, weather, and combined features
using expanding-season validation. Starting-pitcher ERA, WHIP, K/9, and BB/9
improved newest-season log loss, Brier score, and accuracy and won the log-loss
comparison in two of three validation seasons, so those four metrics now affect
daily win probabilities when probable starters are available.

Weather achieved 100% historical coverage but worsened holdout probability
quality, so it remains visible analysis context rather than being forced into
the win model. Bullpen workload, lineups, and injuries also remain context-only
until equivalent historical pregame coverage passes the same leakage-safe
gates. Platt calibration was rejected because it worsened the 2025 audit. The
complete machine-readable decision is in
`docs/context_model_upgrade_report.json`; 2026 tracked predictions are the next
independent audit.

### Milestone 37: postgame learning

Every daily run now writes an immutable pregame factor snapshot and a parallel
team-only shadow prediction for new games. After MLB marks those games final,
MLBAI compares v0.36 with the shadow model using accuracy, log loss, and Brier
score; summarizes daily results; counts which factors helped or misled; and
monitors recent performance for drift. Drift checks remain disabled below 100
future games, and the system refuses to recommend retraining below 200. The
dashboard and `/api/v1/postgame-learning` clearly show when the sample is too
small instead of drawing conclusions early.

### League tables

The daily workflow also collects official American League and National League
standings. The dashboard shows league rank, record, winning percentage, games
back, run differential, last-10 record, and streak. These tables provide season
context and do not directly alter model probabilities.

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
