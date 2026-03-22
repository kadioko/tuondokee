# Sports Betting Value Model

This project helps you answer a simple question:

**Is the bookmaker offering a price that is better than the probability estimated by our model?**

In plain language, the system does this:

- It reads historical match results and bookmaker odds.
- It learns patterns from past matches.
- It estimates the chance of an outcome happening.
- It compares that estimate with the bookmaker's fair price after removing vig.
- It simulates which bets would have been taken and how the bankroll would have changed over time.

This is a research and backtesting tool for soccer markets. It is meant to help you study betting value, not guarantee profit.

## Who This Is For

This README is written for a user who may not be a programmer.

If you want to:

- load your own historical match data
- train a probability model
- test a betting strategy on old matches
- score future matches with the saved model

then this system gives you a structured workflow to do that.

## What the System Can Predict

Right now the project supports these soccer markets:

- **Binary `over_0_5`**
  - `over_0_5`
  - `under_0_5`

- **Binary `over_1_5`**
  - `over_1_5`
  - `under_1_5`

- **Three-way `1x2`**
  - `home`
  - `draw`
  - `away`

- **Binary `over_2_5`**
  - `over_2_5`
  - `under_2_5`

- **Binary `btts`**
  - `btts_yes`
  - `btts_no`

- **Binary `home_win`**
  - `home`
  - `not_home`

## The Big Idea Behind the Model

The system does **not** only ask "who will win?"

Instead, it asks:

- **What probability does the model give to this outcome?**
- **What probability is implied by the bookmaker's price?**
- **After removing bookmaker margin, is the model probability higher than the fair bookmaker probability?**

If the model says an outcome is more likely than the market suggests, that can be a **value bet**.

## What You Need to Provide

You need two CSV files.

### 1. `matches.csv`

This is your match history file.

Required columns:

- `match_id`
- `kickoff_time`
- `league`
- `season`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`

### 2. `odds_snapshots.csv`

This is your bookmaker odds history file.

Required columns:

- `match_id`
- `snapshot_time`
- `bookmaker`
- `market`
- `selection`
- `decimal_odds`
- `is_opening`

## Important Safety Rule: No Data Leakage

This system is designed to avoid a common modeling mistake called **data leakage**.

That means:

- it only uses information available **before** the match starts
- it trains and tests in **time order**
- it never uses future matches to help predict earlier ones

This matters because a betting model can look unrealistically strong if it accidentally learns from the future.

## Quick Start

If you just want to run the sample project as-is, do this from the project root.

Published repository:

- `https://github.com/kadioko/tuondokee`

### 1. Create a virtual environment and install packages

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=src
```

### 2. Run the test suite

```bash
python -m pytest -q
```

### 3. Backfill real history (requires API key)

The sample data ships empty. To populate it with real match history across one or more leagues:

```bash
$env:FOOTBALL_DATA_API_KEY="your_key_here"
python scripts/backfill_history.py --competitions EPL LaLiga Bundesliga UCL UEL --seasons 2023 2024 2025
```

This writes all fetched matches into `data/sample/matches.csv`.

Supported competition codes:

- `EPL`
- `LaLiga`
- `Bundesliga`
- `UCL`
- `UEL`

### 4. Run the full sample workflow

```bash
# Train the 1x2 (match result) model
python -m valuebetting train-model --config config/example_config.json

# Train the over/under 1.5 goals model
python -m valuebetting train-model --config config/over15_config.json

# Train the over/under 0.5 goals model
python -m valuebetting train-model --config config/over05_config.json

# Train the over/under 2.5 goals model
python -m valuebetting train-model --config config/over25_config.json

# Train the BTTS model
python -m valuebetting train-model --config config/btts_config.json

# Backtest the 1x2 model
python -m valuebetting run-backtest --config config/example_config.json

# Score upcoming fixtures
python -m valuebetting score-upcoming --config config/example_config.json

# Generate a combined value report
python -m valuebetting report-value --config config/example_config.json --extra-configs config/over05_config.json config/over15_config.json config/over25_config.json config/btts_config.json
```

## Getting This Week's EPL Thoughts Automatically

The project can now fetch fixtures and odds on its own and turn them into plain-English value opinions.

### What you need

You need API keys for two external services:

- `FOOTBALL_DATA_API_KEY`
  - used for EPL fixtures
- `THE_ODDS_API_KEY`
  - used for live bookmaker odds

Set them in your shell before running the fetch command.

Example on Windows PowerShell:

```bash
$env:FOOTBALL_DATA_API_KEY="your_fixtures_api_key"
$env:THE_ODDS_API_KEY="your_odds_api_key"
```

### End-to-end workflow for this week's fixtures

```bash
# Step 1 — Fetch upcoming fixtures and live odds for multiple competitions
python -m valuebetting fetch-epl-data --config config/example_config.json --date-from 2026-03-20 --date-to 2026-03-27 --competitions EPL LaLiga Bundesliga UCL UEL

# Step 2 — Train (or retrain) the match-result model (1x2)
python -m valuebetting train-model --config config/example_config.json

# Step 3 — Train the over/under 1.5 goals model
python -m valuebetting train-model --config config/over15_config.json

# Step 4 — Train the over/under 0.5 goals model
python -m valuebetting train-model --config config/over05_config.json

# Step 5 — Train the over/under 2.5 goals model
python -m valuebetting train-model --config config/over25_config.json

# Step 6 — Train the BTTS model
python -m valuebetting train-model --config config/btts_config.json

# Step 7 — Generate a combined value report (result + goals + BTTS markets)
python -m valuebetting report-value --config config/example_config.json --extra-configs config/over05_config.json config/over15_config.json config/over25_config.json config/btts_config.json --top-n 20
```

This will:

- fetch fixtures across the competitions you selected into `matches.csv`
- fetch current odds into `odds_snapshots.csv`
- train separate models for match result, over/under 0.5 goals, over/under 1.5 goals, over/under 2.5 goals, and BTTS
- produce a combined report covering **result, goals, and BTTS** bets

The `--extra-configs` flag lets you stack as many extra models as you like into a single report.

## League-Specific Models

You can now train a model on only one or more leagues by adding this to the config file:

```json
"features": {
  "snapshot_minutes_before_kickoff": 60,
  "form_windows": [5, 10],
  "elo_k": 20.0,
  "min_history_matches": 3,
  "league_filters": ["EPL"]
}
```

Examples:

- `league_filters: ["EPL"]`
  - train and score only EPL matches
- `league_filters: ["La Liga"]`
  - build a La Liga-only model
- `league_filters: ["EPL", "Bundesliga"]`
  - keep a narrower multi-league model instead of a fully global one

This affects:

- feature building
- training
- backtesting
- upcoming scoring
- report generation

It is the safest path toward league-specific models without changing the command structure.

Ready-made presets are now included for:

- `config/presets/epl_1x2_config.json`
- `config/presets/laliga_1x2_config.json`
- `config/presets/bundesliga_1x2_config.json`
- `config/presets/ucl_1x2_config.json`
- `config/presets/uel_1x2_config.json`

These give you separate artifact folders for true per-league model runs.

There is now a full preset matrix for each of these leagues and markets:

- `1x2`
- `over_0_5`
- `over_1_5`
- `over_2_5`
- `btts`
- `home_win`

Examples:

- `config/presets/epl_over25_config.json`
- `config/presets/laliga_btts_config.json`
- `config/presets/bundesliga_homewin_config.json`
- `config/presets/ucl_over05_config.json`
- `config/presets/uel_btts_config.json`

## Beginner-Friendly Terminal Workflows

You do **not** need the IDE to run the model.

Two PowerShell scripts are included:

- `scripts/train_all_models.ps1`
  - trains the main 1x2 model plus the extra market models
- `scripts/generate_current_report.ps1`
  - fetches fixtures for the current date window, scores them, and writes the latest report
- `scripts/train_league_models.ps1`
  - trains all supported markets for one league preset set
- `scripts/generate_league_report.ps1`
  - fetches, scores, and reports for one chosen league in `daily` or `weekly` mode

Examples:

```bash
powershell -ExecutionPolicy Bypass -File scripts/train_all_models.ps1
powershell -ExecutionPolicy Bypass -File scripts/generate_current_report.ps1
powershell -ExecutionPolicy Bypass -File scripts/train_league_models.ps1 -League epl
powershell -ExecutionPolicy Bypass -File scripts/generate_league_report.ps1 -League epl -Mode weekly
```

The current report script automatically uses:

- today's date as the start
- a forward-looking window of 7 days by default
- the standard multi-market report command

That means the report updates according to the time period you are in, without needing the IDE.

## Useful Commands

### Train the full global market stack

```bash
powershell -ExecutionPolicy Bypass -File scripts/train_all_models.ps1
```

### Train all models for one league

```bash
powershell -ExecutionPolicy Bypass -File scripts/train_league_models.ps1 -League epl
powershell -ExecutionPolicy Bypass -File scripts/train_league_models.ps1 -League laliga
powershell -ExecutionPolicy Bypass -File scripts/train_league_models.ps1 -League bundesliga
powershell -ExecutionPolicy Bypass -File scripts/train_league_models.ps1 -League ucl
powershell -ExecutionPolicy Bypass -File scripts/train_league_models.ps1 -League uel
```

### Generate a current multi-league report

```bash
powershell -ExecutionPolicy Bypass -File scripts/generate_current_report.ps1
```

### Generate a league-specific daily report

```bash
powershell -ExecutionPolicy Bypass -File scripts/generate_league_report.ps1 -League epl -Mode daily
```

### Generate a league-specific weekly report

```bash
powershell -ExecutionPolicy Bypass -File scripts/generate_league_report.ps1 -League bundesliga -Mode weekly
```

### Output files

- `value_opinions.csv`
  - structured shortlist of value opportunities (one row per opinion)

- `best_bets.csv`
  - a cleaner betting-slip style export with matchup, kickoff, market, best bookmaker, edge, and Kelly stake fields

- `value_opinions.txt`
  - plain-English summary grouped by match, with all scored market views and explicit model opinions such as `VALUE BET`, `WATCHLIST`, or `PASS`

- `value_opinions.html`
  - shareable HTML version of the report for viewing outside the terminal

- `walkforward_predictions.csv`
  - selection-level out-of-sample walkforward predictions used for reliability checks

- `calibration_diagnostics_oos.csv`
  - out-of-sample calibration summary from walkforward predictions

The report tells you, in everyday language, whether a selection looks like a **value bet**, **watchlist**, or **pass**, with the model's probability, the fair market probability, and the edge percentage.

## How to Read `value_opinions.txt`

The new report is designed to be scanned quickly.

### 1. Best bets summary

This is the shortlist at the top.

- It only shows picks that cleared the minimum edge threshold.
- It also requires a minimum reliability score.
- It is the fastest section to read if you just want the strongest ideas.

Example:

```text
- Borussia Dortmund vs Hamburger SV: Under 2.5 Goals @ 2.50 [STRONG]
```

This means the model believes that price is better than the market's fair probability, and the edge is large enough to qualify as a bet.

### 2. Match-by-match breakdown

Each match block shows:

- kickoff time
- bookmaker offering the price
- market and odds
- model opinion (`VALUE BET`, `WATCHLIST`, or `PASS`)
- model probability
- fair market probability
- edge
- suggested stake size

Example:

```text
Kickoff: Sat 21 Mar 17:30
over_2_5: Under 2.5 Goals @ 2.50 with William Hill → VALUE BET [WATCH]
Our model: 44.3% | Market fair: 37.5% | Edge: +6.8%
Model thinks: VALUE BET [WATCH]
Suggested stake: Half stake (12.5 units)
```

### 3. Strength labels

The labels are simple confidence bands based on edge size:

- `STRONG`
  - biggest edge in the report
- `MEDIUM`
  - clear value, but smaller than strong bets
- `WATCH`
  - weaker edge that still qualifies as a bet

These labels do **not** guarantee the outcome will win. They only rank how far the model is above the market's fair probability.

There are **no sure winners** in this system.

What the model can do is:

- rank picks by value
- rank picks by reliability
- reduce weak or noisy suggestions

What it cannot do is promise that a bet will definitely win.

### 4. Suggested stake

The stake line is a simple guide based on the edge band:

- `Full stake`
- `Half stake`
- `Small stake`

This is a convenience layer for readability. It is **not** a promise of profit.

The report also shows:

- `Safe Kelly`
  - a smaller Kelly-based bankroll allocation using your configured fractional Kelly value
- `Aggressive Kelly cap`
  - the uncapped Kelly idea after applying only the max stake cap
- `Market comparison`
  - best price, second-best price when available, and market average odds
- `Reliability score`
  - a combined ranking factor based on edge, calibration, yield, agreement across markets, and bookmaker price quality

Confidence tiers now use **both**:

- edge strength
- out-of-sample calibration/reliability quality

That means a pick with a large edge can still be downgraded if its supporting model has weaker out-of-sample reliability.

### 5. Full market-view report

The combined report can show more than one angle on the same match:

- a **result** opinion, such as `Home Win` or `Away Win`
- a **goals** opinion, such as `Over 0.5 Goals`, `Over 1.5 Goals`, or `Under 2.5 Goals`
- a **BTTS** opinion, such as `BTTS Yes` or `BTTS No`
- a **home_win** opinion, such as `Home Win` or `Draw or Away`

That is why one match can appear with several separate model views if multiple markets were scored.

The text, CSV, and HTML reports now keep one opinion per exact market instead of collapsing all totals into a single shared bucket.

## Publishing

This project is published at:

- `https://github.com/kadioko/tuondokee`

Typical update flow:

```bash
git status
git add .
git commit -m "Describe your update"
git push origin main
```

## Calibration Diagnostics

Each trained model now writes a file called:

- `calibration_diagnostics.csv`
- `calibration_diagnostics_oos.csv`

This file groups predictions into probability bands and shows:

- `sample_count`
  - how many predictions landed in that band
- `mean_predicted_probability`
  - the average probability the model predicted in that band
- `empirical_win_rate`
  - how often those bets/outcomes actually happened in-sample
- `calibration_gap`
  - the difference between predicted probability and observed hit rate

Very simple reading guide:

- small `calibration_gap`
  - better calibration
- large positive gap
  - the model may be overconfident
- large negative gap
  - the model may be underconfident

Calibration matters because a betting model can find value only if its probabilities are trustworthy.

The out-of-sample version is more important when judging whether the model is trustworthy in real use.

## Optional Advanced Input Columns

If your `matches.csv` includes richer pre-match data, the feature builder will now use it automatically when present.

Supported optional pairs include:

- `home_position`, `away_position`
- `home_xg`, `away_xg`
- `home_xga`, `away_xga`
- `home_strength_of_schedule`, `away_strength_of_schedule`

Supported optional passthrough columns include:

- `home_shots_for_last_5`
- `away_shots_for_last_5`
- `home_xpoints_last_5`
- `away_xpoints_last_5`

You do not need these columns for the project to work, but if you have them they will be folded into the model automatically.

## What Each Command Does

### `build-features`

Creates a model-ready dataset from your raw match and odds files.

It builds pre-match features such as:

- recent form
- goals scored and conceded
- rest days
- Elo-style team ratings
- bookmaker consensus prices
- opening-to-latest line movement

### `tune-model`

Searches for better LightGBM settings using time-aware validation.

Use this when you want the system to try different parameter combinations automatically.

### `train-model`

Trains the final model on the historical dataset and saves the model artifacts.

This is the model later used for scoring.

### `run-backtest`

Simulates betting chronologically on historical matches.

The backtest:

- generates walk-forward predictions
- compares model probability to fair bookmaker probability
- places bets only when edge is above your threshold
- applies your staking rules
- tracks bankroll over time

### `score-upcoming`

Uses the saved model to score the matches in the current dataset and writes model probabilities to a CSV.

In a real workflow, you would point this at future or upcoming fixtures with their latest available pre-match inputs.

You can now include upcoming fixtures that do **not** have final scores yet.

For those rows:

- leave `home_goals` blank
- leave `away_goals` blank
- still provide match details and pre-match odds snapshots

The system will:

- exclude those fixtures from training and backtesting
- still build their pre-match features
- include them in `upcoming_scores.csv`

### `fetch-epl-data`

Fetches real fixtures and current market odds and writes them into the configured CSV files.

It is designed so the system can refresh data on its own before scoring.

You can fetch multiple competitions in one run:

```bash
python -m valuebetting fetch-epl-data --config config/example_config.json --date-from 2026-03-20 --date-to 2026-03-27 --competitions EPL LaLiga Bundesliga UCL UEL
```

### `report-value`

Turns scored upcoming fixtures and the latest fair odds into plain-English betting opinions.

Use `--extra-configs` to include additional models (e.g. over/under 1.5 goals and over/under 2.5 goals) in the same report:

```bash
python -m valuebetting report-value --config config/example_config.json --extra-configs config/over15_config.json config/over25_config.json --top-n 20
```

The report groups opinions by match and shows:

- **Result market** — e.g. "Home Win @ 2.40 → VALUE BET"
- **Goals market (1.5)** — e.g. "Over 1.5 Goals @ 1.35 → VALUE BET"
- **Goals market** — e.g. "Over 2.5 Goals @ 1.83 → VALUE BET"
- For each: the model's probability, the bookmaker's fair probability, and the edge

## Config Files

The project ships with three config files:

- `config/example_config.json` — three-way 1x2 (match result) model
- `config/over15_config.json` — binary over/under 1.5 goals model
- `config/over25_config.json` — binary over/under 2.5 goals model

You can create additional configs for other markets (e.g. `home_win`) by copying one of these and changing the `model` and `paths.artifacts_dir` sections.

The most important settings are:

### Market settings

- `target`
  - use `three_way` for `1x2`
  - use `binary` for two-outcome markets

- `binary_market`
  - use `over_1_5` for lower totals betting
  - use `over_2_5` for totals betting
  - use `home_win` for home vs not-home betting

### Betting settings

- `initial_bankroll`
  - starting bankroll for the simulation

- `staking`
  - `flat` means every bet uses the same fixed stake
  - `fractional_kelly` means stake size changes based on the estimated edge

- `minimum_edge`
  - minimum model edge required before a bet is placed

- `max_stake_pct`
  - limits how much of bankroll can be risked on one bet

## Example: Switching to the `home_win` Binary Market

If you want the model to focus on **home win vs not home win**, update the config like this:

```json
"model": {
  "market": "home_win",
  "target": "binary",
  "binary_market": "home_win",
  "calibration": true,
  "n_splits": 4,
  "gap_matches": 0,
  "random_state": 42
}
```

Then rerun:

```bash
python -m valuebetting build-features --config config/example_config.json
python -m valuebetting train-model --config config/example_config.json
python -m valuebetting run-backtest --config config/example_config.json
```

## What Files the System Produces

Outputs are written to the configured `artifacts/` folder.

The most useful files are:

- `features.csv`
  - the training table created from your raw data

- `model.joblib`
  - the saved trained model

- `metadata.json`
  - information about the trained model, classes, and features

- `cv_metrics.csv`
  - validation scores across chronological folds

- `bets.csv`
  - every simulated bet that passed the edge filter

- `bankroll_curve.csv`
  - bankroll progression over time

- `backtest_summary.json`
  - the headline backtest results

- `upcoming_scores.csv`
  - predicted probabilities from the trained model

## How to Read the Backtest Results

The most important metrics are:

- **`bet_count`**
  - how many bets were placed

- **`profit`**
  - total money won or lost in the simulation

- **`roi`**
  - return relative to starting bankroll

- **`yield`**
  - profit divided by total amount staked

- **`hit_rate`**
  - percentage of bets that won

- **`max_drawdown`**
  - worst bankroll drop from a peak

Important:

- a high hit rate does **not** automatically mean a profitable strategy
- a low hit rate can still be profitable if the odds are large enough
- always judge the strategy using both profitability and risk

## Quick Interpretation Guide for `backtest_summary.json`

If you want a fast way to judge whether a strategy looks promising, use this simple checklist:

### 1. Check `bet_count`

- If it is very low, the strategy may not have enough data to trust yet.
- If it is high, you have more evidence, but only if the sample is realistic.

### 2. Check `profit`

- Positive profit means the strategy made money in the backtest.
- Negative profit means it lost money.

Profit alone is not enough. A strategy can show profit with too much risk or too few bets.

### 3. Check `yield`

- This is one of the most useful betting metrics.
- It tells you how much profit was made per unit staked.

Very simple rule of thumb:

- positive `yield` is good
- near-zero `yield` means weak edge
- negative `yield` means the strategy was not profitable

### 4. Check `max_drawdown`

- This tells you how painful the losing stretches were.
- Even a profitable strategy can be hard to follow if drawdown is too large.

If drawdown is severe, the strategy may be too volatile for real use.

### 5. Check `roi` together with `yield`

- `roi` shows return relative to starting bankroll
- `yield` shows efficiency relative to total stake volume

Looking at both gives a better picture than using only one.

### 6. Check `hit_rate` carefully

- Do not treat hit rate as proof of a good model.
- A strategy can win often and still lose money if prices are poor.
- A strategy can win less often and still make money if the odds are high enough.

### Simple layman reading example

If you see something like this:

- positive `profit`
- positive `yield`
- manageable `max_drawdown`
- enough `bet_count`

that usually means the strategy is at least worth deeper investigation.

If you see this instead:

- negative `profit`
- negative `yield`
- large `max_drawdown`

that usually means the strategy is not ready.

## Suggested Workflow for a New User

If you are using the system for the first time, follow this order:

1. Set your API keys (`FOOTBALL_DATA_API_KEY`, `THE_ODDS_API_KEY`).
2. Run `backfill_history.py` with the competitions and seasons you want.
3. Run `fetch-epl-data --competitions ...` to pull this week's fixtures and live odds.
4. Run `train-model` for all configs you want in the report (1x2, over 1.5, over 2.5).
5. Run `report-value --extra-configs config/over15_config.json config/over25_config.json` to get your combined report.
6. Open `value_opinions.txt` — that's your plain-English summary.
7. Optionally run `run-backtest` to see how the model would have performed historically.

If your file also contains future fixtures with blank scores, `score-upcoming` will include them in the output as long as their pre-match odds snapshots are available.

## Project Structure

```text
Tuondoke/
├── config/
│   ├── example_config.json      # 1x2 match-result model config
│   ├── over15_config.json       # over/under 1.5 goals model config
│   └── over25_config.json       # over/under 2.5 goals model config
├── data/
│   └── sample/
│       ├── matches.csv
│       └── odds_snapshots.csv
├── artifacts/                   # 1x2 model outputs
├── artifacts_over15/            # over 1.5 model outputs
├── artifacts_over25/            # over 2.5 model outputs
├── scripts/
│   ├── backfill_history.py      # fetch multi-league, multi-season history from API
│   └── clean_data.py            # remove toy data rows
├── requirements.txt
├── tests/
│   ├── test_backtest.py
│   ├── test_cli.py
│   ├── test_features.py
│   ├── test_fetch_reporting.py
│   ├── test_modeling.py
│   ├── test_odds.py
│   ├── test_upcoming.py
│   └── test_walkforward.py
└── src/
    └── valuebetting/
        ├── backtest.py
        ├── cli.py
        ├── config.py
        ├── data.py
        ├── features.py
        ├── fetch.py
        ├── modeling.py
        ├── odds.py
        ├── reporting.py
        ├── tuning.py
        └── walkforward.py
```

## CI and Testing

This project includes automated tests and a GitHub Actions workflow.

- Local test command:

```bash
python -m pytest -q
```

- CI workflow file:

```text
.github/workflows/ci.yml
```

## Known Limitations

- The football-data.org free tier may restrict which seasons and competitions are available. For deeper history you may need a paid plan or another data source.
- Live odds from the-odds-api.com are only available for upcoming fixtures, not historical ones. Historical matches therefore train on Elo, form, and rest-day features only.
- This is a research starter system, not a finished betting product.
- Real-world deployment would need stronger data coverage, more markets, and stricter operational controls.
- Walk-forward retraining is already chronological, but advanced scheduling logic could still be improved further.

## Final Reminder

This tool helps you study **betting value**.

It does **not** promise profit, and it should not be used blindly with real money. The best use of this project is to test ideas carefully, understand model behavior, and improve decision-making with disciplined research.
