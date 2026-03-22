from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from valuebetting.data import load_matches, load_odds_snapshots, opening_snapshots, select_snapshot_before_kickoff
from valuebetting.odds import compute_fair_probabilities


@dataclass
class FeatureArtifacts:
    dataset: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]



def _elo_expected(home_rating: float, away_rating: float, home_advantage: float = 55.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (((away_rating) - (home_rating + home_advantage)) / 400.0))



def _elo_actual(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0



def _build_match_history_features(matches: pd.DataFrame, form_windows: tuple[int, ...], elo_k: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    team_history: dict[str, list[dict[str, float]]] = defaultdict(list)
    ratings: dict[str, float] = defaultdict(lambda: 1500.0)

    for _, match in matches.sort_values(["kickoff_time", "match_id"]).iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_history = team_history[home_team]
        away_history = team_history[away_team]

        row: dict[str, object] = {
            "match_id": match["match_id"],
            "kickoff_time": match["kickoff_time"],
            "league": match["league"],
            "season": match["season"],
            "home_team": home_team,
            "away_team": away_team,
            "home_elo_pre": ratings[home_team],
            "away_elo_pre": ratings[away_team],
            "elo_diff": ratings[home_team] - ratings[away_team],
            "home_rest_days": np.nan,
            "away_rest_days": np.nan,
        }

        if home_history:
            row["home_rest_days"] = (match["kickoff_time"] - pd.Timestamp(home_history[-1]["kickoff_time"])).days
        if away_history:
            row["away_rest_days"] = (match["kickoff_time"] - pd.Timestamp(away_history[-1]["kickoff_time"])).days

        for window in form_windows:
            home_window = home_history[-window:]
            away_window = away_history[-window:]
            row[f"home_points_last_{window}"] = float(sum(item["points"] for item in home_window)) / max(len(home_window), 1)
            row[f"away_points_last_{window}"] = float(sum(item["points"] for item in away_window)) / max(len(away_window), 1)
            row[f"home_goals_for_last_{window}"] = float(sum(item["goals_for"] for item in home_window)) / max(len(home_window), 1)
            row[f"home_goals_against_last_{window}"] = float(sum(item["goals_against"] for item in home_window)) / max(len(home_window), 1)
            row[f"away_goals_for_last_{window}"] = float(sum(item["goals_for"] for item in away_window)) / max(len(away_window), 1)
            row[f"away_goals_against_last_{window}"] = float(sum(item["goals_against"] for item in away_window)) / max(len(away_window), 1)
            row[f"home_home_points_last_{window}"] = float(sum(item["points"] for item in home_window if item["is_home"] == 1)) / max(sum(item["is_home"] == 1 for item in home_window), 1)
            row[f"away_away_points_last_{window}"] = float(sum(item["points"] for item in away_window if item["is_home"] == 0)) / max(sum(item["is_home"] == 0 for item in away_window), 1)
            row[f"home_clean_sheet_rate_last_{window}"] = float(sum(item["clean_sheet"] for item in home_window)) / max(len(home_window), 1)
            row[f"away_clean_sheet_rate_last_{window}"] = float(sum(item["clean_sheet"] for item in away_window)) / max(len(away_window), 1)
            row[f"home_failed_to_score_rate_last_{window}"] = float(sum(item["failed_to_score"] for item in home_window)) / max(len(home_window), 1)
            row[f"away_failed_to_score_rate_last_{window}"] = float(sum(item["failed_to_score"] for item in away_window)) / max(len(away_window), 1)
            row[f"home_btts_rate_last_{window}"] = float(sum(item["btts"] for item in home_window)) / max(len(home_window), 1)
            row[f"away_btts_rate_last_{window}"] = float(sum(item["btts"] for item in away_window)) / max(len(away_window), 1)
            row[f"home_over_1_5_rate_last_{window}"] = float(sum(item["over_1_5"] for item in home_window)) / max(len(home_window), 1)
            row[f"away_over_1_5_rate_last_{window}"] = float(sum(item["over_1_5"] for item in away_window)) / max(len(away_window), 1)
            row[f"home_over_2_5_rate_last_{window}"] = float(sum(item["over_2_5"] for item in home_window)) / max(len(home_window), 1)
            row[f"away_over_2_5_rate_last_{window}"] = float(sum(item["over_2_5"] for item in away_window)) / max(len(away_window), 1)

        records.append(row)

        if int(match.get("is_completed", 1)) != 1:
            continue

        home_expected = _elo_expected(ratings[home_team], ratings[away_team])
        home_actual = _elo_actual(int(match["home_goals"]), int(match["away_goals"]))
        away_actual = 1.0 - home_actual if home_actual != 0.5 else 0.5

        ratings[home_team] = ratings[home_team] + elo_k * (home_actual - home_expected)
        ratings[away_team] = ratings[away_team] + elo_k * (away_actual - (1.0 - home_expected))

        home_points = 3 if match["home_goals"] > match["away_goals"] else 1 if match["home_goals"] == match["away_goals"] else 0
        away_points = 3 if match["away_goals"] > match["home_goals"] else 1 if match["home_goals"] == match["away_goals"] else 0
        team_history[home_team].append(
            {
                "kickoff_time": match["kickoff_time"],
                "points": float(home_points),
                "goals_for": float(match["home_goals"]),
                "goals_against": float(match["away_goals"]),
                "is_home": 1,
                "clean_sheet": float(int(match["away_goals"] == 0)),
                "failed_to_score": float(int(match["home_goals"] == 0)),
                "btts": float(int(match["home_goals"] > 0 and match["away_goals"] > 0)),
                "over_1_5": float(int((match["home_goals"] + match["away_goals"]) > 1.5)),
                "over_2_5": float(int((match["home_goals"] + match["away_goals"]) > 2.5)),
            }
        )
        team_history[away_team].append(
            {
                "kickoff_time": match["kickoff_time"],
                "points": float(away_points),
                "goals_for": float(match["away_goals"]),
                "goals_against": float(match["home_goals"]),
                "is_home": 0,
                "clean_sheet": float(int(match["home_goals"] == 0)),
                "failed_to_score": float(int(match["away_goals"] == 0)),
                "btts": float(int(match["home_goals"] > 0 and match["away_goals"] > 0)),
                "over_1_5": float(int((match["home_goals"] + match["away_goals"]) > 1.5)),
                "over_2_5": float(int((match["home_goals"] + match["away_goals"]) > 2.5)),
            }
        )

    features = pd.DataFrame(records)
    features["rest_days_diff"] = features["home_rest_days"].fillna(features["home_rest_days"].median()) - features[
        "away_rest_days"
    ].fillna(features["away_rest_days"].median())
    return features



def _build_odds_features(selected_odds: pd.DataFrame, opening_odds: pd.DataFrame) -> pd.DataFrame:
    if selected_odds.empty:
        return pd.DataFrame(columns=["match_id"])

    fair_selected = compute_fair_probabilities(selected_odds)

    consensus = (
        fair_selected.groupby(["match_id", "market", "selection"], as_index=False)
        .agg(
            consensus_fair_prob=("fair_prob", "mean"),
            consensus_decimal_odds=("decimal_odds", "mean"),
        )
    )
    consensus_pivot = consensus.pivot_table(
        index="match_id",
        columns=["market", "selection"],
        values=["consensus_fair_prob", "consensus_decimal_odds"],
    )
    consensus_pivot.columns = ["__".join(col).lower() for col in consensus_pivot.columns]
    consensus_pivot = consensus_pivot.reset_index()

    if opening_odds.empty:
        return consensus_pivot

    fair_opening = compute_fair_probabilities(opening_odds)
    opening_consensus = (
        fair_opening.groupby(["match_id", "market", "selection"], as_index=False)
        .agg(opening_fair_prob=("fair_prob", "mean"))
    )
    opening_pivot = opening_consensus.pivot_table(
        index="match_id", columns=["market", "selection"], values="opening_fair_prob"
    )
    opening_pivot.columns = [f"opening__{'__'.join(col).lower()}" for col in opening_pivot.columns]
    opening_pivot = opening_pivot.reset_index()

    odds_features = consensus_pivot.merge(opening_pivot, on="match_id", how="left")
    for column in list(odds_features.columns):
        if column.startswith("consensus_fair_prob__"):
            suffix = column.replace("consensus_fair_prob__", "")
            opening_column = f"opening__{suffix}"
            if opening_column in odds_features.columns:
                odds_features[f"line_move__{suffix}"] = odds_features[column] - odds_features[opening_column]
    return odds_features


def _build_optional_context_features(matches: pd.DataFrame) -> pd.DataFrame:
    context = matches[["match_id"]].copy()
    paired_columns = [
        ("home_position", "away_position", "position_diff"),
        ("home_xg", "away_xg", "xg_diff"),
        ("home_xga", "away_xga", "xga_diff"),
        ("home_strength_of_schedule", "away_strength_of_schedule", "strength_of_schedule_diff"),
    ]
    for home_column, away_column, diff_column in paired_columns:
        if home_column in matches.columns and away_column in matches.columns:
            context[home_column] = pd.to_numeric(matches[home_column], errors="coerce")
            context[away_column] = pd.to_numeric(matches[away_column], errors="coerce")
            context[diff_column] = context[home_column] - context[away_column]
    passthrough_columns = [
        "home_shots_for_last_5",
        "away_shots_for_last_5",
        "home_xpoints_last_5",
        "away_xpoints_last_5",
    ]
    for column in passthrough_columns:
        if column in matches.columns:
            context[column] = pd.to_numeric(matches[column], errors="coerce")
    return context



def build_feature_dataset(
    matches_csv: str | Path,
    odds_csv: str | Path,
    snapshot_minutes_before_kickoff: int,
    form_windows: tuple[int, ...],
    elo_k: float,
    league_filters: tuple[str, ...] = (),
    include_incomplete_matches: bool = False,
) -> FeatureArtifacts:
    matches = load_matches(matches_csv)
    if league_filters:
        matches = matches.loc[matches["league"].isin(league_filters)].copy()
    odds = load_odds_snapshots(odds_csv)
    if league_filters and not odds.empty:
        match_ids = set(matches["match_id"].astype(str))
        odds = odds.loc[odds["match_id"].astype(str).isin(match_ids)].copy()
    selected_odds = select_snapshot_before_kickoff(matches, odds, snapshot_minutes_before_kickoff)
    opening_odds = opening_snapshots(odds)

    history_features = _build_match_history_features(matches, form_windows=form_windows, elo_k=elo_k)
    odds_features = _build_odds_features(selected_odds, opening_odds)
    context_features = _build_optional_context_features(matches)

    dataset = matches.merge(history_features, on=["match_id", "kickoff_time", "league", "season", "home_team", "away_team"], how="left")
    dataset = dataset.merge(odds_features, on="match_id", how="left")
    dataset = dataset.merge(context_features, on="match_id", how="left")
    dataset = dataset.sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)
    if not include_incomplete_matches:
        dataset = dataset.loc[dataset["is_completed"] == 1].reset_index(drop=True)

    numeric_fill_columns = [column for column in dataset.columns if dataset[column].dtype.kind in "fi"]
    dataset[numeric_fill_columns] = dataset[numeric_fill_columns].fillna(0.0)

    feature_columns = [
        column
        for column in dataset.columns
        if column
        not in {
            "match_id",
            "kickoff_time",
            "home_goals",
            "away_goals",
            "total_goals",
            "home_win",
            "draw",
            "away_win",
            "over_0_5",
            "over_1_5",
            "over_2_5",
            "btts",
            "is_completed",
        }
    ]
    categorical_columns = [column for column in ["league", "season", "home_team", "away_team"] if column in feature_columns]
    return FeatureArtifacts(dataset=dataset, feature_columns=feature_columns, categorical_columns=categorical_columns)
