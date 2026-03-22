import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from valuebetting.cli import cmd_build_features, cmd_report_value, cmd_score_upcoming, cmd_train_model
from valuebetting.config import load_config
from valuebetting.fetch import football_data_matches_to_frame, odds_api_events_to_frame
from valuebetting.reporting import generate_value_report



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
            'bookmaker': 'Pinnacle',
        },
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    return config_path



FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'


def _write_dataset_with_upcoming_fixture(tmp_path: Path) -> tuple[Path, Path]:
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
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'home', 'decimal_odds': 2.4, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'draw', 'decimal_odds': 3.4, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 15:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'away', 'decimal_odds': 2.8, 'is_opening': 1},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'home', 'decimal_odds': 2.2, 'is_opening': 0},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'draw', 'decimal_odds': 3.6, 'is_opening': 0},
            {'match_id': 'M11', 'snapshot_time': '2023-08-20 17:00:00', 'bookmaker': 'Pinnacle', 'market': '1x2', 'selection': 'away', 'decimal_odds': 3.2, 'is_opening': 0},
        ]
    )
    odds_path = tmp_path / 'odds.csv'
    pd.concat([source_odds, additional_odds], ignore_index=True).to_csv(odds_path, index=False)
    return matches_path, odds_path



def test_football_data_matches_to_frame_transforms_payload() -> None:
    payload = {
        'matches': [
            {
                'id': 123,
                'utcDate': '2026-03-22T15:00:00Z',
                'status': 'SCHEDULED',
                'season': {'startDate': '2025-08-01', 'endDate': '2026-05-31'},
                'homeTeam': {'name': 'Arsenal'},
                'awayTeam': {'name': 'Chelsea'},
                'score': {'fullTime': {'home': None, 'away': None}},
            }
        ]
    }
    frame = football_data_matches_to_frame(payload)
    assert list(frame['match_id']) == ['FD_PL_123']
    assert pd.isna(frame.loc[0, 'home_goals'])
    assert frame.loc[0, 'league'] == 'EPL'



def test_football_data_matches_to_frame_supports_other_competitions() -> None:
    payload = {
        'matches': [
            {
                'id': 456,
                'utcDate': '2026-03-22T15:00:00Z',
                'status': 'SCHEDULED',
                'season': {'startDate': '2025-08-01', 'endDate': '2026-05-31'},
                'homeTeam': {'name': 'Real Madrid'},
                'awayTeam': {'name': 'Barcelona'},
                'score': {'fullTime': {'home': None, 'away': None}},
            }
        ]
    }
    frame = football_data_matches_to_frame(payload, competition='LaLiga')
    assert list(frame['match_id']) == ['FD_PD_456']
    assert frame.loc[0, 'league'] == 'La Liga'



def test_odds_api_events_to_frame_transforms_h2h_and_totals() -> None:
    fixtures = pd.DataFrame(
        [
            {
                'match_id': 'FD_PL_123',
                'kickoff_time': pd.Timestamp('2026-03-22 15:00:00'),
                'league': 'EPL',
                'season': '2025/2026',
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'home_goals': None,
                'away_goals': None,
            }
        ]
    )
    payload = [
        {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'commence_time': '2026-03-22T15:00:00Z',
            'bookmakers': [
                {
                    'title': 'Pinnacle',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Arsenal', 'price': 2.2},
                                {'name': 'Draw', 'price': 3.5},
                                {'name': 'Chelsea', 'price': 3.2},
                            ],
                        },
                        {
                            'key': 'totals',
                            'outcomes': [
                                {'name': 'Over', 'price': 1.22, 'point': 1.5},
                                {'name': 'Under', 'price': 4.4, 'point': 1.5},
                                {'name': 'Over', 'price': 1.95, 'point': 2.5},
                                {'name': 'Under', 'price': 1.88, 'point': 2.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
    frame = odds_api_events_to_frame(payload, fixtures, snapshot_time=pd.Timestamp('2026-03-20 12:00:00'))
    assert {'1x2', 'home_win', 'over_1_5', 'over_2_5'} <= set(frame['market'])
    assert {'home', 'draw', 'away', 'not_home', 'over_1_5', 'under_1_5', 'over_2_5', 'under_2_5'} <= set(frame['selection'])



def test_generate_value_report_produces_plain_english_opinions(tmp_path: Path) -> None:
    matches_path, odds_path = _write_dataset_with_upcoming_fixture(tmp_path)
    config_path = _write_temp_config(tmp_path, matches_path, odds_path)
    cmd_build_features(Namespace(config=str(config_path)))
    cmd_train_model(Namespace(config=str(config_path)))
    cmd_score_upcoming(Namespace(config=str(config_path)))

    config = load_config(config_path)
    report_frame, report_text = generate_value_report(config, top_n=5)
    assert not report_frame.empty
    assert "EPL VALUE REPORT" in report_text
    assert 'Arsenal' in report_text and 'Liverpool' in report_text
    assert 'Suggested stake:' in report_text
    assert 'Kickoff:' in report_text
    assert 'Reliability score:' in report_text
    assert report_frame.drop_duplicates(subset=['match_id', 'market_category']).shape[0] == len(report_frame)
    assert (report_frame['recommendation'] == 'bet').all()



def test_cmd_report_value_writes_outputs(tmp_path: Path) -> None:
    matches_path, odds_path = _write_dataset_with_upcoming_fixture(tmp_path)
    config_path = _write_temp_config(tmp_path, matches_path, odds_path)
    cmd_build_features(Namespace(config=str(config_path)))
    cmd_train_model(Namespace(config=str(config_path)))
    cmd_score_upcoming(Namespace(config=str(config_path)))
    cmd_report_value(Namespace(config=str(config_path), top_n=5, extra_configs=[]))
    artifacts_dir = tmp_path / 'artifacts'
    assert (artifacts_dir / 'value_opinions.csv').exists()
    assert (artifacts_dir / 'best_bets.csv').exists()
    assert (artifacts_dir / 'value_opinions.txt').exists()
    assert (artifacts_dir / 'value_opinions.html').exists()
