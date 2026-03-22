import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from valuebetting.cli import cmd_build_features, cmd_score_upcoming, cmd_train_model
from valuebetting.data import load_matches
from valuebetting.features import build_feature_dataset



def _write_temp_config(tmp_path: Path, matches_csv: Path, odds_csv: Path) -> Path:
    config = {
        'paths': {
            'matches_csv': str(matches_csv),
            'odds_csv': str(odds_csv),
            'artifacts_dir': str(tmp_path / 'artifacts'),
        },
        'features': {
            'snapshot_minutes_before_kickoff': 60,
            'form_windows': [5, 10],
            'elo_k': 20.0,
            'min_history_matches': 3,
        },
        'model': {
            'market': '1x2',
            'target': 'three_way',
            'binary_market': 'over_2_5',
            'calibration': True,
            'n_splits': 4,
            'gap_matches': 0,
            'random_state': 42,
        },
        'tuning': {
            'enabled': False,
            'n_trials': 2,
            'timeout_seconds': 30,
        },
        'betting': {
            'initial_bankroll': 1000.0,
            'staking': 'flat',
            'flat_stake': 25.0,
            'fractional_kelly_fraction': 0.25,
            'minimum_edge': 0.03,
            'max_stake_pct': 0.05,
            'bookmaker': None,
        },
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    return config_path



def test_load_matches_allows_upcoming_fixtures_without_scores(tmp_path: Path) -> None:
    matches_path = tmp_path / 'matches.csv'
    matches_path.write_text(
        'match_id,kickoff_time,league,season,home_team,away_team,home_goals,away_goals\n'
        'M1,2023-08-01 18:00:00,EPL,2023/2024,A,B,1,0\n'
        'M2,2023-08-02 18:00:00,EPL,2023/2024,C,D,,\n',
        encoding='utf-8',
    )
    matches = load_matches(matches_path)
    assert len(matches) == 2
    assert matches.loc[matches['match_id'] == 'M2', 'is_completed'].iloc[0] == 0
    assert pd.isna(matches.loc[matches['match_id'] == 'M2', 'home_win'].iloc[0])



FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'


def test_build_feature_dataset_can_include_upcoming_incomplete_matches(tmp_path: Path) -> None:
    source_matches = pd.read_csv(FIXTURES_DIR / 'toy_matches.csv')
    upcoming_row = {
        'match_id': 'M11',
        'kickoff_time': '2023-08-20 18:00:00',
        'league': 'EPL',
        'season': '2023/2024',
        'home_team': 'Arsenal',
        'away_team': 'Liverpool',
        'home_goals': '',
        'away_goals': '',
    }
    matches_path = tmp_path / 'matches.csv'
    pd.concat([source_matches, pd.DataFrame([upcoming_row])], ignore_index=True).to_csv(matches_path, index=False)

    source_odds = pd.read_csv(FIXTURES_DIR / 'toy_odds_snapshots.csv')
    additional_odds = pd.DataFrame(
        [
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'home', 'decimal_odds': 2.3, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'draw', 'decimal_odds': 3.4, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'away', 'decimal_odds': 2.9, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'home', 'decimal_odds': 2.2, 'is_opening': 0},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'draw', 'decimal_odds': 3.5, 'is_opening': 0},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'away', 'decimal_odds': 3.0, 'is_opening': 0},
        ]
    )
    odds_path = tmp_path / 'odds.csv'
    pd.concat([source_odds, additional_odds], ignore_index=True).to_csv(odds_path, index=False)

    completed_only = build_feature_dataset(matches_path, odds_path, 60, (5, 10), 20.0, include_incomplete_matches=False)
    with_upcoming = build_feature_dataset(matches_path, odds_path, 60, (5, 10), 20.0, include_incomplete_matches=True)

    assert 'M11' not in set(completed_only.dataset['match_id'])
    assert 'M11' in set(with_upcoming.dataset['match_id'])
    assert with_upcoming.dataset.loc[with_upcoming.dataset['match_id'] == 'M11', 'is_completed'].iloc[0] == 0



def test_score_upcoming_includes_incomplete_fixture_rows(tmp_path: Path) -> None:
    source_matches = pd.read_csv(FIXTURES_DIR / 'toy_matches.csv')
    upcoming_row = {
        'match_id': 'M11',
        'kickoff_time': '2023-08-20 18:00:00',
        'league': 'EPL',
        'season': '2023/2024',
        'home_team': 'Arsenal',
        'away_team': 'Liverpool',
        'home_goals': '',
        'away_goals': '',
    }
    matches_path = tmp_path / 'matches.csv'
    pd.concat([source_matches, pd.DataFrame([upcoming_row])], ignore_index=True).to_csv(matches_path, index=False)

    source_odds = pd.read_csv(FIXTURES_DIR / 'toy_odds_snapshots.csv')
    additional_odds = pd.DataFrame(
        [
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'home', 'decimal_odds': 2.3, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'draw', 'decimal_odds': 3.4, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'away', 'decimal_odds': 2.9, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'home', 'decimal_odds': 2.2, 'is_opening': 0},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'draw', 'decimal_odds': 3.5, 'is_opening': 0},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'away', 'decimal_odds': 3.0, 'is_opening': 0},
        ]
    )
    odds_path = tmp_path / 'odds.csv'
    pd.concat([source_odds, additional_odds], ignore_index=True).to_csv(odds_path, index=False)

    config_path = _write_temp_config(tmp_path, matches_path, odds_path)
    cmd_build_features(Namespace(config=str(config_path)))
    cmd_train_model(Namespace(config=str(config_path)))
    cmd_score_upcoming(Namespace(config=str(config_path)))

    scored = pd.read_csv(tmp_path / 'artifacts' / 'upcoming_scores.csv')
    upcoming_row = scored.loc[scored['match_id'] == 'M11'].iloc[0]
    assert upcoming_row['is_completed'] == 0
    assert 'model_prob_away' in scored.columns
    assert 'model_prob_draw' in scored.columns
    assert 'model_prob_home' in scored.columns
