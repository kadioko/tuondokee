import json
from argparse import Namespace
from pathlib import Path

from valuebetting.cli import (
    build_parser,
    cmd_backtest,
    cmd_build_features,
    cmd_init_config,
    cmd_score_upcoming,
    cmd_train_model,
)



FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'


def _write_temp_config(tmp_path: Path, target: str = 'three_way', binary_market: str = 'over_2_5') -> Path:
    config = {
        'paths': {
            'matches_csv': str(FIXTURES_DIR / 'toy_matches.csv'),
            'odds_csv': str(FIXTURES_DIR / 'toy_odds_snapshots.csv'),
            'artifacts_dir': str(tmp_path / 'artifacts'),
        },
        'features': {
            'snapshot_minutes_before_kickoff': 60,
            'form_windows': [5, 10],
            'elo_k': 20.0,
            'min_history_matches': 3,
        },
        'model': {
            'market': 'home_win' if target == 'binary' else '1x2',
            'target': target,
            'binary_market': binary_market,
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



def test_cmd_init_config_writes_file(tmp_path: Path) -> None:
    output = tmp_path / 'example_config.json'
    cmd_init_config(Namespace(output=str(output)))
    assert output.exists()



def test_build_parser_parses_known_command() -> None:
    parser = build_parser()
    args = parser.parse_args(['build-features', '--config', 'config/example_config.json'])
    assert args.command == 'build-features'
    assert args.config == 'config/example_config.json'



def test_build_parser_parses_fetch_competitions() -> None:
    parser = build_parser()
    args = parser.parse_args([
        'fetch-epl-data',
        '--config', 'config/example_config.json',
        '--date-from', '2026-03-20',
        '--date-to', '2026-03-27',
        '--competitions', 'EPL', 'LaLiga', 'Bundesliga'
    ])
    assert args.command == 'fetch-epl-data'
    assert args.competitions == ['EPL', 'LaLiga', 'Bundesliga']



def test_cli_sample_flow_writes_expected_artifacts(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    cmd_build_features(Namespace(config=str(config_path)))
    cmd_train_model(Namespace(config=str(config_path)))
    cmd_backtest(Namespace(config=str(config_path)))
    cmd_score_upcoming(Namespace(config=str(config_path)))

    artifacts_dir = tmp_path / 'artifacts'
    assert (artifacts_dir / 'features.csv').exists()
    assert (artifacts_dir / 'features_metadata.json').exists()
    assert (artifacts_dir / 'model.joblib').exists()
    assert (artifacts_dir / 'metadata.json').exists()
    assert (artifacts_dir / 'cv_metrics.csv').exists()
    assert (artifacts_dir / 'bets.csv').exists()
    assert (artifacts_dir / 'backtest_summary.json').exists()
    assert (artifacts_dir / 'upcoming_scores.csv').exists()



def test_cli_binary_home_win_flow_writes_outputs(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path, target='binary', binary_market='home_win')
    cmd_build_features(Namespace(config=str(config_path)))
    cmd_train_model(Namespace(config=str(config_path)))
    cmd_backtest(Namespace(config=str(config_path)))

    artifacts_dir = tmp_path / 'artifacts'
    summary_path = artifacts_dir / 'backtest_summary.json'
    metadata_path = artifacts_dir / 'metadata.json'
    assert summary_path.exists()
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    assert metadata['binary_market'] == 'home_win'



def test_cli_binary_over_1_5_flow_writes_outputs(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path, target='binary', binary_market='over_1_5')
    cmd_build_features(Namespace(config=str(config_path)))
    cmd_train_model(Namespace(config=str(config_path)))
    cmd_backtest(Namespace(config=str(config_path)))

    artifacts_dir = tmp_path / 'artifacts'
    summary_path = artifacts_dir / 'backtest_summary.json'
    metadata_path = artifacts_dir / 'metadata.json'
    assert summary_path.exists()
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    assert metadata['binary_market'] == 'over_1_5'
