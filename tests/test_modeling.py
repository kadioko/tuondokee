import pandas as pd

from valuebetting.modeling import prepare_target



def test_prepare_target_supports_over_0_5_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "home_goals": [0, 0, 2],
            "away_goals": [0, 1, 2],
            "home_win": [0, 0, 0],
            "over_0_5": [0, 1, 1],
            "over_1_5": [0, 0, 1],
            "over_2_5": [0, 0, 1],
            "btts": [0, 0, 1],
        }
    )
    y, target_name, classes = prepare_target(dataset, "binary", "over_0_5")
    assert target_name == "over_0_5"
    assert classes == ["0", "1"]
    assert list(y) == [0, 1, 1]



def test_prepare_target_supports_over_1_5_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "home_goals": [1, 0, 2],
            "away_goals": [0, 1, 2],
            "home_win": [1, 0, 0],
            "over_0_5": [1, 1, 1],
            "over_1_5": [0, 0, 1],
            "over_2_5": [0, 0, 1],
            "btts": [0, 0, 1],
        }
    )
    y, target_name, classes = prepare_target(dataset, "binary", "over_1_5")
    assert target_name == "over_1_5"
    assert classes == ["0", "1"]
    assert list(y) == [0, 0, 1]



def test_prepare_target_supports_home_win_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "home_goals": [1, 0, 2],
            "away_goals": [0, 1, 2],
            "home_win": [1, 0, 0],
            "over_0_5": [1, 1, 1],
            "over_1_5": [0, 0, 1],
            "over_2_5": [0, 0, 1],
            "btts": [0, 0, 1],
        }
    )
    y, target_name, classes = prepare_target(dataset, "binary", "home_win")
    assert target_name == "home_win"
    assert classes == ["0", "1"]
    assert list(y) == [1, 0, 0]



def test_prepare_target_supports_over_2_5_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "home_goals": [1, 0, 2],
            "away_goals": [0, 1, 2],
            "home_win": [1, 0, 0],
            "over_0_5": [1, 1, 1],
            "over_1_5": [0, 0, 1],
            "over_2_5": [0, 0, 1],
            "btts": [0, 0, 1],
        }
    )
    y, target_name, classes = prepare_target(dataset, "binary", "over_2_5")
    assert target_name == "over_2_5"
    assert classes == ["0", "1"]
    assert list(y) == [0, 0, 1]



def test_prepare_target_supports_btts_binary_market() -> None:
    dataset = pd.DataFrame(
        {
            "home_goals": [1, 0, 2],
            "away_goals": [0, 1, 2],
            "home_win": [1, 0, 0],
            "over_0_5": [1, 1, 1],
            "over_1_5": [0, 0, 1],
            "over_2_5": [0, 0, 1],
            "btts": [0, 0, 1],
        }
    )
    y, target_name, classes = prepare_target(dataset, "binary", "btts")
    assert target_name == "btts"
    assert classes == ["0", "1"]
    assert list(y) == [0, 0, 1]
