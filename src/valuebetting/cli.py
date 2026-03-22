from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pandas as pd

from valuebetting.backtest import build_prediction_frame, run_backtest
from valuebetting.config import load_config, write_default_config
from valuebetting.fetch import fetch_competition_fixtures, fetch_competition_odds, update_matches_csv, update_odds_csv
from valuebetting.features import build_feature_dataset
from valuebetting.modeling import evaluate_time_series, load_model_artifacts, predict_probabilities, save_train_artifacts, train_final_model
from valuebetting.reporting import generate_value_report, write_value_report
from valuebetting.tuning import tune_lightgbm
from valuebetting.walkforward import generate_walkforward_predictions



def _ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory



def cmd_init_config(args: argparse.Namespace) -> None:
    write_default_config(args.output)
    print(f"Wrote default config to {args.output}")



def cmd_fetch_competition_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    competitions = args.competitions or ["EPL"]
    all_fixtures: list[pd.DataFrame] = []
    all_odds: list[pd.DataFrame] = []
    fetch_summary: list[dict[str, Any]] = []
    for competition in competitions:
        try:
            fixtures = fetch_competition_fixtures(competition, args.date_from, args.date_to, api_key_env=args.fixtures_api_key_env)
            all_fixtures.append(fixtures)
            odds = fetch_competition_odds(fixtures, competition, api_key_env=args.odds_api_key_env, regions=args.regions)
            all_odds.append(odds)
            fetch_summary.append(
                {
                    "competition": competition,
                    "status": "ok",
                    "fixtures_fetched": int(len(fixtures)),
                    "odds_rows_fetched": int(len(odds)),
                }
            )
        except HTTPError as exc:
            fetch_summary.append(
                {
                    "competition": competition,
                    "status": "failed",
                    "error_type": "http_error",
                    "status_code": int(exc.code),
                    "message": "Competition unavailable on current API tier or temporarily blocked.",
                }
            )
        except Exception as exc:
            fetch_summary.append(
                {
                    "competition": competition,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    fixtures = pd.concat(all_fixtures, ignore_index=True) if all_fixtures else pd.DataFrame()
    update_matches_csv(config.paths.matches_csv, fixtures)
    odds = pd.concat(all_odds, ignore_index=True) if all_odds else pd.DataFrame()
    update_odds_csv(config.paths.odds_csv, odds)
    summary = {
        "competitions": competitions,
        "competition_results": fetch_summary,
        "fixtures_fetched": int(len(fixtures)),
        "odds_rows_fetched": int(len(odds)),
        "matches_csv": config.paths.matches_csv,
        "odds_csv": config.paths.odds_csv,
    }
    print(json.dumps(summary, indent=2))



def cmd_build_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    artifacts = build_feature_dataset(
        matches_csv=config.paths.matches_csv,
        odds_csv=config.paths.odds_csv,
        snapshot_minutes_before_kickoff=config.features.snapshot_minutes_before_kickoff,
        form_windows=config.features.form_windows,
        elo_k=config.features.elo_k,
        league_filters=config.features.league_filters,
        include_incomplete_matches=False,
    )
    output_dir = _ensure_dir(config.paths.artifacts_dir)
    artifacts.dataset.to_csv(output_dir / "features.csv", index=False)
    metadata = {
        "feature_columns": artifacts.feature_columns,
        "categorical_columns": artifacts.categorical_columns,
    }
    with (output_dir / "features_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Feature dataset written to {output_dir / 'features.csv'}")



def cmd_tune_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    artifacts = build_feature_dataset(
        matches_csv=config.paths.matches_csv,
        odds_csv=config.paths.odds_csv,
        snapshot_minutes_before_kickoff=config.features.snapshot_minutes_before_kickoff,
        form_windows=config.features.form_windows,
        elo_k=config.features.elo_k,
        league_filters=config.features.league_filters,
        include_incomplete_matches=False,
    )
    best_params, study = tune_lightgbm(
        dataset=artifacts.dataset,
        feature_columns=artifacts.feature_columns,
        categorical_columns=artifacts.categorical_columns,
        target_type=config.model.target,
        binary_market=config.model.binary_market,
        n_splits=config.model.n_splits,
        random_state=config.model.random_state,
        calibration=config.model.calibration,
        n_trials=config.tuning.n_trials,
        timeout_seconds=config.tuning.timeout_seconds,
    )
    output_dir = _ensure_dir(config.paths.artifacts_dir)
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)
    study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)
    print(json.dumps(best_params, indent=2))



def _load_best_params_if_available(artifacts_dir: str | Path) -> dict[str, Any] | None:
    best_params_path = Path(artifacts_dir) / "best_params.json"
    if best_params_path.exists():
        with best_params_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return None



def cmd_train_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    artifacts = build_feature_dataset(
        matches_csv=config.paths.matches_csv,
        odds_csv=config.paths.odds_csv,
        snapshot_minutes_before_kickoff=config.features.snapshot_minutes_before_kickoff,
        form_windows=config.features.form_windows,
        elo_k=config.features.elo_k,
        league_filters=config.features.league_filters,
        include_incomplete_matches=False,
    )
    cv_metrics, folds = evaluate_time_series(
        dataset=artifacts.dataset,
        feature_columns=artifacts.feature_columns,
        categorical_columns=artifacts.categorical_columns,
        target_type=config.model.target,
        binary_market=config.model.binary_market,
        n_splits=config.model.n_splits,
        random_state=config.model.random_state,
        calibration=config.model.calibration,
        params=_load_best_params_if_available(config.paths.artifacts_dir),
    )
    trained = train_final_model(
        dataset=artifacts.dataset,
        feature_columns=artifacts.feature_columns,
        categorical_columns=artifacts.categorical_columns,
        target_type=config.model.target,
        binary_market=config.model.binary_market,
        random_state=config.model.random_state,
        calibration=config.model.calibration,
        params=_load_best_params_if_available(config.paths.artifacts_dir),
    )
    trained.metrics.update(cv_metrics)
    output_dir = _ensure_dir(config.paths.artifacts_dir)
    save_train_artifacts(trained, output_dir)
    pd.DataFrame(folds).to_csv(output_dir / "cv_metrics.csv", index=False)
    print(json.dumps(trained.metrics, indent=2))



def cmd_backtest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    walkforward = generate_walkforward_predictions(
        matches_csv=config.paths.matches_csv,
        odds_csv=config.paths.odds_csv,
        snapshot_minutes_before_kickoff=config.features.snapshot_minutes_before_kickoff,
        form_windows=config.features.form_windows,
        elo_k=config.features.elo_k,
        league_filters=config.features.league_filters,
        target_type=config.model.target,
        binary_market=config.model.binary_market,
        n_splits=config.model.n_splits,
        random_state=config.model.random_state,
        calibration=config.model.calibration,
        params=_load_best_params_if_available(config.paths.artifacts_dir),
    )
    prediction_frame = build_prediction_frame(
        dataset=walkforward.dataset,
        predicted_probabilities=walkforward.probabilities,
        classes=list(walkforward.classes),
        selected_odds=walkforward.selected_odds,
        bookmaker=config.betting.bookmaker,
        binary_market=config.model.binary_market if config.model.target == "binary" else None,
    )
    result = run_backtest(
        prediction_frame=prediction_frame,
        initial_bankroll=config.betting.initial_bankroll,
        staking=config.betting.staking,
        flat_stake=config.betting.flat_stake,
        fractional_kelly_fraction=config.betting.fractional_kelly_fraction,
        minimum_edge=config.betting.minimum_edge,
        max_stake_pct=config.betting.max_stake_pct,
    )
    output_dir = _ensure_dir(config.paths.artifacts_dir)
    walkforward.prediction_frame.to_csv(output_dir / "walkforward_predictions.csv", index=False)
    walkforward.calibration_table.to_csv(output_dir / "calibration_diagnostics_oos.csv", index=False)
    result.bets.to_csv(output_dir / "bets.csv", index=False)
    result.bankroll_curve.to_csv(output_dir / "bankroll_curve.csv", index=False)
    with (output_dir / "backtest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2)
    for name, frame in result.breakdowns.items():
        frame.to_csv(output_dir / f"breakdown_{name}.csv", index=False)
    print(json.dumps(result.summary, indent=2))



def cmd_score_upcoming(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model, metadata = load_model_artifacts(config.paths.artifacts_dir)
    feature_artifacts = build_feature_dataset(
        matches_csv=config.paths.matches_csv,
        odds_csv=config.paths.odds_csv,
        snapshot_minutes_before_kickoff=config.features.snapshot_minutes_before_kickoff,
        form_windows=config.features.form_windows,
        elo_k=config.features.elo_k,
        league_filters=config.features.league_filters,
        include_incomplete_matches=True,
    )
    X = feature_artifacts.dataset[metadata["feature_columns"]]
    probabilities = predict_probabilities(model, X)
    scored = feature_artifacts.dataset[["match_id", "kickoff_time", "league", "season", "home_team", "away_team"]].copy()
    if "is_completed" in feature_artifacts.dataset.columns:
        scored["is_completed"] = feature_artifacts.dataset["is_completed"].astype(int)
    for class_name, idx in zip(metadata["classes"], range(probabilities.shape[1])):
        scored[f"model_prob_{class_name}"] = probabilities[:, idx]
    output_dir = _ensure_dir(config.paths.artifacts_dir)
    scored.to_csv(output_dir / "upcoming_scores.csv", index=False)
    print(f"Scores written to {output_dir / 'upcoming_scores.csv'}")



def cmd_report_value(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    extra_configs = getattr(args, "extra_configs", None) or []
    report_frame, report_text = generate_value_report(config, top_n=args.top_n, extra_config_paths=extra_configs)
    output_dir = _ensure_dir(config.paths.artifacts_dir)
    csv_path, best_bets_path, txt_path, html_path = write_value_report(output_dir, report_frame, report_text)
    print(report_text)
    print(f"\nValue opinions written to {csv_path}, {best_bets_path}, {txt_path}, and {html_path}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sports betting value model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_config = subparsers.add_parser("init-config", help="Write a default JSON config file")
    init_config.add_argument("--output", default="config/example_config.json")
    init_config.set_defaults(func=cmd_init_config)

    fetch_competition = subparsers.add_parser("fetch-epl-data", help="Fetch fixtures and live odds into the configured CSV files")
    fetch_competition.add_argument("--config", required=True)
    fetch_competition.add_argument("--date-from", required=True)
    fetch_competition.add_argument("--date-to", required=True)
    fetch_competition.add_argument("--competitions", nargs="*", default=["EPL"], help="Competitions to fetch: EPL, LaLiga, Bundesliga, UCL, UEL")
    fetch_competition.add_argument("--fixtures-api-key-env", default="FOOTBALL_DATA_API_KEY")
    fetch_competition.add_argument("--odds-api-key-env", default="THE_ODDS_API_KEY")
    fetch_competition.add_argument("--regions", default="uk")
    fetch_competition.set_defaults(func=cmd_fetch_competition_data)

    build_features = subparsers.add_parser("build-features", help="Build leakage-safe feature dataset")
    build_features.add_argument("--config", required=True)
    build_features.set_defaults(func=cmd_build_features)

    tune_model = subparsers.add_parser("tune-model", help="Tune LightGBM with Optuna and time-series CV")
    tune_model.add_argument("--config", required=True)
    tune_model.set_defaults(func=cmd_tune_model)

    train_model = subparsers.add_parser("train-model", help="Train LightGBM model and save artifacts")
    train_model.add_argument("--config", required=True)
    train_model.set_defaults(func=cmd_train_model)

    backtest = subparsers.add_parser("run-backtest", help="Run chronological backtest from saved model artifacts")
    backtest.add_argument("--config", required=True)
    backtest.set_defaults(func=cmd_backtest)

    score = subparsers.add_parser("score-upcoming", help="Score matches using the saved model")
    score.add_argument("--config", required=True)
    score.set_defaults(func=cmd_score_upcoming)

    report = subparsers.add_parser("report-value", help="Create a plain-English value opinion report for upcoming matches")
    report.add_argument("--config", required=True)
    report.add_argument("--extra-configs", nargs="*", default=[], help="Additional model config files (e.g. over_2_5, over_1_5) to include in the report")
    report.add_argument("--top-n", type=int, default=20)
    report.set_defaults(func=cmd_report_value)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
