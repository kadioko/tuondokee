import math

import pandas as pd

from valuebetting.odds import compute_fair_probabilities, remove_vig_three_way, remove_vig_two_way


def test_remove_vig_two_way_sums_to_one() -> None:
    fair_a, fair_b = remove_vig_two_way(0.55, 0.50)
    assert math.isclose(fair_a + fair_b, 1.0, rel_tol=1e-9)



def test_remove_vig_three_way_sums_to_one() -> None:
    fair_home, fair_draw, fair_away = remove_vig_three_way(0.50, 0.30, 0.28)
    assert math.isclose(fair_home + fair_draw + fair_away, 1.0, rel_tol=1e-9)



def test_compute_fair_probabilities_handles_two_way_and_three_way_markets() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "M1", "market": "home_win", "bookmaker": "P", "selection": "home", "decimal_odds": 2.0},
            {"match_id": "M1", "market": "home_win", "bookmaker": "P", "selection": "not_home", "decimal_odds": 1.8},
            {"match_id": "M1", "market": "1x2", "bookmaker": "P", "selection": "home", "decimal_odds": 2.4},
            {"match_id": "M1", "market": "1x2", "bookmaker": "P", "selection": "draw", "decimal_odds": 3.2},
            {"match_id": "M1", "market": "1x2", "bookmaker": "P", "selection": "away", "decimal_odds": 2.9},
        ]
    )
    fair = compute_fair_probabilities(frame)
    assert set(fair["selection"]) == {"home", "not_home", "draw", "away"}
    assert math.isclose(fair.loc[fair["market"] == "home_win", "fair_prob"].sum(), 1.0, rel_tol=1e-9)
    assert math.isclose(fair.loc[fair["market"] == "1x2", "fair_prob"].sum(), 1.0, rel_tol=1e-9)
