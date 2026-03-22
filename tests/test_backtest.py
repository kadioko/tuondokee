import numpy as np
import pandas as pd

from valuebetting.backtest import build_prediction_frame, run_backtest



def test_build_prediction_frame_supports_over_0_5_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "match_id": ["M1"],
            "kickoff_time": [pd.Timestamp("2023-08-10 18:00:00")],
            "league": ["EPL"],
            "season": ["2023/2024"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_goals": [1],
            "away_goals": [0],
        }
    )
    probabilities = np.array([[0.10, 0.90]])
    selected_odds = pd.DataFrame(
        [
            {"match_id": "M1", "snapshot_time": pd.Timestamp("2023-08-10 17:00:00"), "bookmaker": "P", "market": "over_0_5", "selection": "over_0_5", "decimal_odds": 1.10},
            {"match_id": "M1", "snapshot_time": pd.Timestamp("2023-08-10 17:00:00"), "bookmaker": "P", "market": "over_0_5", "selection": "under_0_5", "decimal_odds": 8.0},
        ]
    )
    frame = build_prediction_frame(dataset=dataset, predicted_probabilities=probabilities, classes=["0", "1"], selected_odds=selected_odds, bookmaker=None, binary_market="over_0_5")
    assert set(frame["selection"]) == {"over_0_5", "under_0_5"}
    assert frame.loc[frame["selection"] == "over_0_5", "model_probability"].iloc[0] == 0.90
    assert frame.loc[frame["selection"] == "under_0_5", "model_probability"].iloc[0] == 0.10



def test_build_prediction_frame_supports_over_1_5_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "match_id": ["M1"],
            "kickoff_time": [pd.Timestamp("2023-08-10 18:00:00")],
            "league": ["EPL"],
            "season": ["2023/2024"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_goals": [1],
            "away_goals": [1],
        }
    )
    probabilities = np.array([[0.30, 0.70]])
    selected_odds = pd.DataFrame(
        [
            {"match_id": "M1", "snapshot_time": pd.Timestamp("2023-08-10 17:00:00"), "bookmaker": "P", "market": "over_1_5", "selection": "over_1_5", "decimal_odds": 1.55},
            {"match_id": "M1", "snapshot_time": pd.Timestamp("2023-08-10 17:00:00"), "bookmaker": "P", "market": "over_1_5", "selection": "under_1_5", "decimal_odds": 2.45},
        ]
    )
    frame = build_prediction_frame(
        dataset=dataset,
        predicted_probabilities=probabilities,
        classes=["0", "1"],
        selected_odds=selected_odds,
        bookmaker=None,
        binary_market="over_1_5",
    )
    assert set(frame["selection"]) == {"over_1_5", "under_1_5"}
    assert frame.loc[frame["selection"] == "over_1_5", "model_probability"].iloc[0] == 0.70
    assert frame.loc[frame["selection"] == "under_1_5", "model_probability"].iloc[0] == 0.30



def test_build_prediction_frame_supports_btts_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "match_id": ["M1"],
            "kickoff_time": [pd.Timestamp("2023-08-10 18:00:00")],
            "league": ["EPL"],
            "season": ["2023/2024"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_goals": [1],
            "away_goals": [1],
        }
    )
    probabilities = np.array([[0.42, 0.58]])
    selected_odds = pd.DataFrame(
        [
            {"match_id": "M1", "snapshot_time": pd.Timestamp("2023-08-10 17:00:00"), "bookmaker": "P", "market": "btts", "selection": "btts_yes", "decimal_odds": 1.80},
            {"match_id": "M1", "snapshot_time": pd.Timestamp("2023-08-10 17:00:00"), "bookmaker": "P", "market": "btts", "selection": "btts_no", "decimal_odds": 2.00},
        ]
    )
    frame = build_prediction_frame(dataset=dataset, predicted_probabilities=probabilities, classes=["0", "1"], selected_odds=selected_odds, bookmaker=None, binary_market="btts")
    assert set(frame["selection"]) == {"btts_yes", "btts_no"}
    assert frame.loc[frame["selection"] == "btts_yes", "model_probability"].iloc[0] == 0.58
    assert frame.loc[frame["selection"] == "btts_no", "model_probability"].iloc[0] == 0.42



def test_build_prediction_frame_supports_home_win_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "match_id": ["M1"],
            "kickoff_time": [pd.Timestamp("2023-08-10 18:00:00")],
            "league": ["EPL"],
            "season": ["2023/2024"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_goals": [1],
            "away_goals": [0],
        }
    )
    probabilities = np.array([[0.35, 0.65]])
    selected_odds = pd.DataFrame(
        [
            {"match_id": "M1", "snapshot_time": pd.Timestamp("2023-08-10 17:00:00"), "bookmaker": "P", "market": "home_win", "selection": "home", "decimal_odds": 2.1},
            {"match_id": "M1", "snapshot_time": pd.Timestamp("2023-08-10 17:00:00"), "bookmaker": "P", "market": "home_win", "selection": "not_home", "decimal_odds": 1.8},
        ]
    )
    frame = build_prediction_frame(
        dataset=dataset,
        predicted_probabilities=probabilities,
        classes=["0", "1"],
        selected_odds=selected_odds,
        bookmaker=None,
        binary_market="home_win",
    )
    assert set(frame["selection"]) == {"home", "not_home"}
    assert frame.loc[frame["selection"] == "home", "model_probability"].iloc[0] == 0.65
    assert frame.loc[frame["selection"] == "not_home", "model_probability"].iloc[0] == 0.35



def test_run_backtest_places_and_settles_home_win_bet() -> None:
    prediction_frame = pd.DataFrame(
        {
            "match_id": ["M1"],
            "kickoff_time": [pd.Timestamp("2023-08-10 18:00:00")],
            "snapshot_time": [pd.Timestamp("2023-08-10 17:00:00")],
            "bookmaker": ["P"],
            "market": ["home_win"],
            "selection": ["home"],
            "league": ["EPL"],
            "season": ["2023/2024"],
            "decimal_odds": [2.1],
            "fair_prob": [0.48],
            "model_probability": [0.65],
            "edge": [0.17],
            "home_goals": [1],
            "away_goals": [0],
        }
    )
    result = run_backtest(
        prediction_frame=prediction_frame,
        initial_bankroll=1000.0,
        staking="flat",
        flat_stake=25.0,
        fractional_kelly_fraction=0.25,
        minimum_edge=0.03,
        max_stake_pct=0.05,
    )
    assert result.summary["bet_count"] == 1.0
    assert result.bets.iloc[0]["result"] == 1
    assert result.bets.iloc[0]["pnl"] > 0



def test_run_backtest_places_and_settles_btts_bet() -> None:
    prediction_frame = pd.DataFrame(
        {
            "match_id": ["M1"],
            "kickoff_time": [pd.Timestamp("2023-08-10 18:00:00")],
            "snapshot_time": [pd.Timestamp("2023-08-10 17:00:00")],
            "bookmaker": ["P"],
            "market": ["btts"],
            "selection": ["btts_yes"],
            "league": ["EPL"],
            "season": ["2023/2024"],
            "decimal_odds": [1.80],
            "fair_prob": [0.52],
            "model_probability": [0.58],
            "edge": [0.06],
            "home_goals": [1],
            "away_goals": [1],
        }
    )
    result = run_backtest(prediction_frame=prediction_frame, initial_bankroll=1000.0, staking="flat", flat_stake=25.0, fractional_kelly_fraction=0.25, minimum_edge=0.03, max_stake_pct=0.05)
    assert result.summary["bet_count"] == 1.0
    assert result.bets.iloc[0]["result"] == 1
    assert result.bets.iloc[0]["pnl"] > 0



def test_run_backtest_places_and_settles_over_1_5_bet() -> None:
    prediction_frame = pd.DataFrame(
        {
            "match_id": ["M1"],
            "kickoff_time": [pd.Timestamp("2023-08-10 18:00:00")],
            "snapshot_time": [pd.Timestamp("2023-08-10 17:00:00")],
            "bookmaker": ["P"],
            "market": ["over_1_5"],
            "selection": ["over_1_5"],
            "league": ["EPL"],
            "season": ["2023/2024"],
            "decimal_odds": [1.55],
            "fair_prob": [0.60],
            "model_probability": [0.70],
            "edge": [0.10],
            "home_goals": [1],
            "away_goals": [1],
        }
    )
    result = run_backtest(
        prediction_frame=prediction_frame,
        initial_bankroll=1000.0,
        staking="flat",
        flat_stake=25.0,
        fractional_kelly_fraction=0.25,
        minimum_edge=0.03,
        max_stake_pct=0.05,
    )
    assert result.summary["bet_count"] == 1.0
    assert result.bets.iloc[0]["result"] == 1
    assert result.bets.iloc[0]["pnl"] > 0
