from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from valuebetting.odds import compute_fair_probabilities


@dataclass
class BacktestResult:
    bets: pd.DataFrame
    bankroll_curve: pd.DataFrame
    summary: dict[str, float]
    breakdowns: dict[str, pd.DataFrame]



def _kelly_fraction(probability: float, odds: float) -> float:
    b = odds - 1.0
    q = 1.0 - probability
    if b <= 0:
        return 0.0
    kelly = (b * probability - q) / b
    return max(0.0, kelly)



def _selection_result(row: pd.Series) -> int:
    selection = row["selection"]
    if selection == "home":
        return int(row["home_goals"] > row["away_goals"])
    if selection == "not_home":
        return int(row["home_goals"] <= row["away_goals"])
    if selection == "draw":
        return int(row["home_goals"] == row["away_goals"])
    if selection == "away":
        return int(row["home_goals"] < row["away_goals"])
    if selection == "over_0_5":
        return int((row["home_goals"] + row["away_goals"]) > 0.5)
    if selection == "under_0_5":
        return int((row["home_goals"] + row["away_goals"]) < 0.5)
    if selection == "over_1_5":
        return int((row["home_goals"] + row["away_goals"]) > 1.5)
    if selection == "under_1_5":
        return int((row["home_goals"] + row["away_goals"]) < 1.5)
    if selection == "over_2_5":
        return int((row["home_goals"] + row["away_goals"]) > 2.5)
    if selection == "under_2_5":
        return int((row["home_goals"] + row["away_goals"]) < 2.5)
    if selection == "btts_yes":
        return int((row["home_goals"] > 0) and (row["away_goals"] > 0))
    if selection == "btts_no":
        return int((row["home_goals"] == 0) or (row["away_goals"] == 0))
    raise ValueError(f"Unsupported selection: {selection}")



def build_prediction_frame(
    dataset: pd.DataFrame,
    predicted_probabilities: np.ndarray,
    classes: list[str],
    selected_odds: pd.DataFrame,
    bookmaker: str | None,
    binary_market: str | None = None,
) -> pd.DataFrame:
    if selected_odds.empty:
        return pd.DataFrame()
    odds_frame = compute_fair_probabilities(selected_odds)
    if odds_frame.empty:
        return pd.DataFrame()
    if bookmaker is not None:
        odds_frame = odds_frame.loc[odds_frame["bookmaker"] == bookmaker].copy()
    odds_frame = odds_frame.drop(columns=["kickoff_time"], errors="ignore")

    scored_dataset = dataset.copy().reset_index(drop=True)
    class_index = {label: idx for idx, label in enumerate(classes)}
    for selection, idx in class_index.items():
        scored_dataset[f"model_prob__{selection}"] = predicted_probabilities[:, idx]

    merged = scored_dataset.merge(odds_frame, on="match_id", how="inner")
    merged = merged.copy()
    merged["model_probability"] = np.nan

    for selection in ["home", "draw", "away"]:
        probability_column = f"model_prob__{selection}"
        if probability_column in merged.columns:
            merged.loc[merged["selection"] == selection, "model_probability"] = merged.loc[
                merged["selection"] == selection, probability_column
            ]

    if "model_prob__1" in merged.columns:
        merged.loc[merged["selection"] == "over_0_5", "model_probability"] = merged.loc[
            merged["selection"] == "over_0_5", "model_prob__1"
        ]
        merged.loc[merged["selection"] == "over_1_5", "model_probability"] = merged.loc[
            merged["selection"] == "over_1_5", "model_prob__1"
        ]
        merged.loc[merged["selection"] == "over_2_5", "model_probability"] = merged.loc[
            merged["selection"] == "over_2_5", "model_prob__1"
        ]
        merged.loc[merged["selection"] == "btts_yes", "model_probability"] = merged.loc[
            merged["selection"] == "btts_yes", "model_prob__1"
        ]
        merged.loc[merged["selection"] == "home", "model_probability"] = merged.loc[
            merged["selection"] == "home", "model_prob__1"
        ]
    if "model_prob__0" in merged.columns:
        merged.loc[merged["selection"] == "under_0_5", "model_probability"] = merged.loc[
            merged["selection"] == "under_0_5", "model_prob__0"
        ]
        merged.loc[merged["selection"] == "under_1_5", "model_probability"] = merged.loc[
            merged["selection"] == "under_1_5", "model_prob__0"
        ]
        merged.loc[merged["selection"] == "under_2_5", "model_probability"] = merged.loc[
            merged["selection"] == "under_2_5", "model_prob__0"
        ]
        merged.loc[merged["selection"] == "btts_no", "model_probability"] = merged.loc[
            merged["selection"] == "btts_no", "model_prob__0"
        ]
        merged.loc[merged["selection"] == "not_home", "model_probability"] = merged.loc[
            merged["selection"] == "not_home", "model_prob__0"
        ]

    if binary_market == "over_0_5":
        merged = merged.loc[merged["market"] == "over_0_5"].copy()
    if binary_market == "over_1_5":
        merged = merged.loc[merged["market"] == "over_1_5"].copy()
    if binary_market == "over_2_5":
        merged = merged.loc[merged["market"] == "over_2_5"].copy()
    if binary_market == "btts":
        merged = merged.loc[merged["market"] == "btts"].copy()
    if binary_market == "home_win":
        merged = merged.loc[merged["market"] == "home_win"].copy()

    merged = merged.dropna(subset=["model_probability"]).copy()
    merged["edge"] = merged["model_probability"] - merged["fair_prob"]
    return merged



def run_backtest(
    prediction_frame: pd.DataFrame,
    initial_bankroll: float,
    staking: str,
    flat_stake: float,
    fractional_kelly_fraction: float,
    minimum_edge: float,
    max_stake_pct: float,
) -> BacktestResult:
    if prediction_frame.empty or "edge" not in prediction_frame.columns:
        bankroll_curve = pd.DataFrame([{"step": 0, "bankroll": initial_bankroll}])
        summary = {
            "bet_count": 0.0,
            "turnover": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "average_odds": 0.0,
            "average_edge": 0.0,
            "final_bankroll": initial_bankroll,
        }
        empty = pd.DataFrame()
        return BacktestResult(bets=empty, bankroll_curve=bankroll_curve, summary=summary, breakdowns={"league": empty, "season": empty, "bookmaker": empty, "market": empty})
    bets = prediction_frame.loc[prediction_frame["edge"] >= minimum_edge].copy()
    bets = bets.sort_values(["kickoff_time", "snapshot_time", "match_id", "market", "selection"]).reset_index(drop=True)

    bankroll = initial_bankroll
    curve_rows: list[dict[str, Any]] = [{"step": 0, "bankroll": bankroll}]
    bet_rows: list[dict[str, Any]] = []
    peak = bankroll
    drawdowns: list[float] = []

    for idx, row in bets.iterrows():
        if staking == "flat":
            stake = min(flat_stake, bankroll * max_stake_pct, bankroll)
        elif staking == "fractional_kelly":
            stake_pct = _kelly_fraction(float(row["model_probability"]), float(row["decimal_odds"])) * fractional_kelly_fraction
            stake = min(bankroll * stake_pct, bankroll * max_stake_pct, bankroll)
        else:
            raise ValueError(f"Unsupported staking method: {staking}")

        if stake <= 0:
            continue

        result = _selection_result(row)
        pnl = stake * (float(row["decimal_odds"]) - 1.0) if result == 1 else -stake
        bankroll += pnl
        peak = max(peak, bankroll)
        drawdowns.append((peak - bankroll) / peak if peak > 0 else 0.0)
        curve_rows.append({"step": idx + 1, "bankroll": bankroll, "kickoff_time": row["kickoff_time"]})
        bet_rows.append(
            {
                "match_id": row["match_id"],
                "kickoff_time": row["kickoff_time"],
                "bookmaker": row["bookmaker"],
                "market": row["market"],
                "selection": row["selection"],
                "league": row["league"],
                "season": row["season"],
                "odds": float(row["decimal_odds"]),
                "fair_probability": float(row["fair_prob"]),
                "model_probability": float(row["model_probability"]),
                "edge": float(row["edge"]),
                "stake": float(stake),
                "result": int(result),
                "pnl": float(pnl),
                "bankroll_after_bet": float(bankroll),
            }
        )

    bets_df = pd.DataFrame(bet_rows)
    bankroll_curve = pd.DataFrame(curve_rows)
    if bets_df.empty:
        summary = {
            "bet_count": 0.0,
            "turnover": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "average_odds": 0.0,
            "average_edge": 0.0,
            "final_bankroll": initial_bankroll,
        }
        empty = pd.DataFrame()
        return BacktestResult(bets=bets_df, bankroll_curve=bankroll_curve, summary=summary, breakdowns={"league": empty, "season": empty, "bookmaker": empty, "market": empty})

    turnover = float(bets_df["stake"].sum())
    profit = float(bets_df["pnl"].sum())
    summary = {
        "bet_count": float(len(bets_df)),
        "turnover": turnover,
        "profit": profit,
        "roi": float(profit / initial_bankroll),
        "yield": float(profit / turnover) if turnover > 0 else 0.0,
        "hit_rate": float(bets_df["result"].mean()),
        "max_drawdown": float(max(drawdowns) if drawdowns else 0.0),
        "average_odds": float(bets_df["odds"].mean()),
        "average_edge": float(bets_df["edge"].mean()),
        "final_bankroll": float(bankroll),
    }

    def summarize_by(column: str) -> pd.DataFrame:
        grouped = bets_df.groupby(column, as_index=False).agg(
            bets=("match_id", "count"),
            turnover=("stake", "sum"),
            profit=("pnl", "sum"),
            hit_rate=("result", "mean"),
            avg_edge=("edge", "mean"),
        )
        grouped["yield"] = grouped["profit"] / grouped["turnover"].replace(0.0, np.nan)
        return grouped.sort_values("profit", ascending=False)

    breakdowns = {name: summarize_by(name) for name in ["league", "season", "bookmaker", "market"]}
    return BacktestResult(bets=bets_df, bankroll_curve=bankroll_curve, summary=summary, breakdowns=breakdowns)
