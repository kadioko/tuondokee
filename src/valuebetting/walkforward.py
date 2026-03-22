from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from valuebetting.backtest import build_prediction_frame
from valuebetting.data import load_matches, load_odds_snapshots, select_snapshot_before_kickoff
from valuebetting.features import build_feature_dataset
from valuebetting.modeling import build_calibration_table, predict_probabilities, train_final_model


@dataclass
class WalkForwardPredictionResult:
    dataset: pd.DataFrame
    probabilities: np.ndarray
    classes: list[str]
    selected_odds: pd.DataFrame
    prediction_frame: pd.DataFrame
    calibration_table: pd.DataFrame



def generate_walkforward_predictions(
    matches_csv: str,
    odds_csv: str,
    snapshot_minutes_before_kickoff: int,
    form_windows: tuple[int, ...],
    elo_k: float,
    target_type: str,
    binary_market: str,
    n_splits: int,
    random_state: int,
    calibration: bool,
    league_filters: tuple[str, ...] = (),
    params: dict[str, Any] | None = None,
) -> WalkForwardPredictionResult:
    artifacts = build_feature_dataset(
        matches_csv=matches_csv,
        odds_csv=odds_csv,
        snapshot_minutes_before_kickoff=snapshot_minutes_before_kickoff,
        form_windows=form_windows,
        elo_k=elo_k,
        league_filters=league_filters,
        include_incomplete_matches=False,
    )
    dataset = artifacts.dataset.copy().reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    probabilities = np.full((len(dataset), 3 if target_type == "three_way" else 2), np.nan)
    classes: list[str] = []
    for train_idx, test_idx in splitter.split(dataset):
        trained = train_final_model(
            dataset=dataset.iloc[train_idx].reset_index(drop=True),
            feature_columns=artifacts.feature_columns,
            categorical_columns=artifacts.categorical_columns,
            target_type=target_type,
            binary_market=binary_market,
            random_state=random_state,
            calibration=calibration,
            params=params,
        )
        fold_probs = predict_probabilities(trained.estimator, dataset.iloc[test_idx][artifacts.feature_columns])
        probabilities[test_idx, : fold_probs.shape[1]] = fold_probs
        classes = trained.classes_

    valid_rows = ~np.isnan(probabilities).any(axis=1)
    filtered_dataset = dataset.loc[valid_rows].reset_index(drop=True)
    filtered_probabilities = probabilities[valid_rows]
    matches = load_matches(matches_csv)
    if league_filters:
        matches = matches.loc[matches["league"].isin(league_filters)].copy()
    odds = load_odds_snapshots(odds_csv)
    if league_filters and not odds.empty:
        match_ids = set(matches["match_id"].astype(str))
        odds = odds.loc[odds["match_id"].astype(str).isin(match_ids)].copy()
    selected_odds = select_snapshot_before_kickoff(matches, odds, snapshot_minutes_before_kickoff)
    selected_odds = selected_odds.loc[selected_odds["match_id"].isin(filtered_dataset["match_id"])].reset_index(drop=True)
    tradable_match_ids = selected_odds["match_id"].drop_duplicates().tolist()
    tradable_mask = filtered_dataset["match_id"].isin(tradable_match_ids).to_numpy()
    filtered_dataset = filtered_dataset.loc[tradable_mask].reset_index(drop=True)
    filtered_probabilities = filtered_probabilities[tradable_mask]
    prediction_frame = build_prediction_frame(
        dataset=filtered_dataset,
        predicted_probabilities=filtered_probabilities,
        classes=list(classes),
        selected_odds=selected_odds,
        bookmaker=None,
        binary_market=binary_market if target_type == "binary" else None,
    )
    if target_type == "three_way":
        y_true = np.select(
            [filtered_dataset["home_goals"] > filtered_dataset["away_goals"], filtered_dataset["home_goals"] == filtered_dataset["away_goals"]],
            ["home", "draw"],
            default="away",
        )
        class_index = {label: idx for idx, label in enumerate(classes)}
        y_encoded = np.asarray([class_index[label] for label in y_true])
    else:
        if binary_market == "over_0_5":
            y_encoded = filtered_dataset["over_0_5"].astype(int).to_numpy()
        elif binary_market == "over_1_5":
            y_encoded = filtered_dataset["over_1_5"].astype(int).to_numpy()
        elif binary_market == "over_2_5":
            y_encoded = filtered_dataset["over_2_5"].astype(int).to_numpy()
        elif binary_market == "btts":
            y_encoded = filtered_dataset["btts"].astype(int).to_numpy()
        else:
            y_encoded = filtered_dataset["home_win"].astype(int).to_numpy()
    calibration_table = build_calibration_table(y_encoded, filtered_probabilities, target_type)
    return WalkForwardPredictionResult(
        dataset=filtered_dataset,
        probabilities=filtered_probabilities,
        classes=classes,
        selected_odds=selected_odds,
        prediction_frame=prediction_frame,
        calibration_table=calibration_table,
    )
