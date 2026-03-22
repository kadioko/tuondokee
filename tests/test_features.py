from pathlib import Path

import pandas as pd

from valuebetting.features import build_feature_dataset

FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'


def test_feature_builder_uses_only_prior_matches() -> None:
    artifacts = build_feature_dataset(
        matches_csv=FIXTURES_DIR / "toy_matches.csv",
        odds_csv=FIXTURES_DIR / "toy_odds_snapshots.csv",
        snapshot_minutes_before_kickoff=60,
        form_windows=(5, 10),
        elo_k=20.0,
    )
    dataset = artifacts.dataset
    first_row = dataset.iloc[0]
    assert first_row["home_points_last_5"] == 0.0
    assert first_row["away_points_last_5"] == 0.0
    assert first_row["home_elo_pre"] == 1500.0
    assert first_row["away_elo_pre"] == 1500.0



def test_feature_builder_adds_richer_btts_and_totals_form_features() -> None:
    artifacts = build_feature_dataset(
        matches_csv=FIXTURES_DIR / "toy_matches.csv",
        odds_csv=FIXTURES_DIR / "toy_odds_snapshots.csv",
        snapshot_minutes_before_kickoff=60,
        form_windows=(5,),
        elo_k=20.0,
    )
    dataset = artifacts.dataset
    arsenal_chelsea = dataset.loc[dataset["match_id"] == "M6"].iloc[0]
    assert "home_btts_rate_last_5" in dataset.columns
    assert "away_clean_sheet_rate_last_5" in dataset.columns
    assert "home_over_2_5_rate_last_5" in dataset.columns
    assert arsenal_chelsea["home_btts_rate_last_5"] == 0.0
    assert arsenal_chelsea["away_clean_sheet_rate_last_5"] == 0.5
    assert arsenal_chelsea["home_over_2_5_rate_last_5"] == 0.5



def test_feature_builder_can_filter_to_specific_leagues(tmp_path: Path) -> None:
    source_matches = pd.read_csv(FIXTURES_DIR / "toy_matches.csv")
    extra_match = pd.DataFrame(
        [
            {
                "match_id": "L1",
                "kickoff_time": "2023-08-20 18:00:00",
                "league": "La Liga",
                "season": "2023/2024",
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "home_goals": 2,
                "away_goals": 1,
            }
        ]
    )
    matches_path = tmp_path / "matches.csv"
    pd.concat([source_matches, extra_match], ignore_index=True).to_csv(matches_path, index=False)

    source_odds = pd.read_csv(FIXTURES_DIR / "toy_odds_snapshots.csv")
    extra_odds = pd.DataFrame(
        [
            {"match_id": "L1", "snapshot_time": "2023-08-20 17:00:00", "bookmaker": "Pinnacle", "market": "1x2", "selection": "home", "decimal_odds": 2.1, "is_opening": 0},
            {"match_id": "L1", "snapshot_time": "2023-08-20 17:00:00", "bookmaker": "Pinnacle", "market": "1x2", "selection": "draw", "decimal_odds": 3.4, "is_opening": 0},
            {"match_id": "L1", "snapshot_time": "2023-08-20 17:00:00", "bookmaker": "Pinnacle", "market": "1x2", "selection": "away", "decimal_odds": 3.2, "is_opening": 0},
        ]
    )
    odds_path = tmp_path / "odds.csv"
    pd.concat([source_odds, extra_odds], ignore_index=True).to_csv(odds_path, index=False)

    filtered = build_feature_dataset(
        matches_csv=matches_path,
        odds_csv=odds_path,
        snapshot_minutes_before_kickoff=60,
        form_windows=(5,),
        elo_k=20.0,
        league_filters=("La Liga",),
    )
    assert set(filtered.dataset["league"]) == {"La Liga"}
    assert set(filtered.dataset["match_id"]) == {"L1"}
