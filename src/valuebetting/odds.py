from __future__ import annotations

import numpy as np
import pandas as pd


def decimal_odds_to_implied_prob(odds: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return 1.0 / odds



def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Total implied probability must be positive for vig removal.")
    return prob_a / total, prob_b / total



def remove_vig_three_way(prob_home: float, prob_draw: float, prob_away: float) -> tuple[float, float, float]:
    total = prob_home + prob_draw + prob_away
    if total <= 0:
        raise ValueError("Total implied probability must be positive for vig removal.")
    return prob_home / total, prob_draw / total, prob_away / total



def compute_fair_probabilities(odds_frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"match_id", "market", "bookmaker", "selection", "decimal_odds"}
    missing = required_columns - set(odds_frame.columns)
    if missing:
        raise ValueError(f"Odds frame missing required columns: {sorted(missing)}")

    frame = odds_frame.copy()
    frame["implied_prob"] = decimal_odds_to_implied_prob(frame["decimal_odds"].astype(float))

    fair_rows: list[dict[str, object]] = []
    group_columns = ["match_id", "market", "bookmaker"]
    for group_key, group in frame.groupby(group_columns, sort=False):
        probabilities = dict(zip(group["selection"], group["implied_prob"]))
        if len(probabilities) == 2:
            selections = list(probabilities.keys())
            fair_a, fair_b = remove_vig_two_way(probabilities[selections[0]], probabilities[selections[1]])
            fair_map = {selections[0]: fair_a, selections[1]: fair_b}
        elif len(probabilities) == 3:
            expected = ["home", "draw", "away"]
            if not all(key in probabilities for key in expected):
                raise ValueError(f"Three-way market {group_key} must contain home/draw/away selections.")
            fair_home, fair_draw, fair_away = remove_vig_three_way(
                probabilities["home"], probabilities["draw"], probabilities["away"]
            )
            fair_map = {"home": fair_home, "draw": fair_draw, "away": fair_away}
        else:
            raise ValueError(f"Only two-way and three-way markets are supported, got {len(probabilities)} for {group_key}.")

        for _, row in group.iterrows():
            fair_rows.append(
                {
                    **row.to_dict(),
                    "fair_prob": fair_map[row["selection"]],
                }
            )

    return pd.DataFrame(fair_rows)
