from pathlib import Path

import numpy as np

from valuebetting.walkforward import generate_walkforward_predictions

FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'


def test_generate_walkforward_predictions_returns_out_of_sample_probabilities() -> None:
    result = generate_walkforward_predictions(
        matches_csv=FIXTURES_DIR / 'toy_matches.csv',
        odds_csv=FIXTURES_DIR / 'toy_odds_snapshots.csv',
        snapshot_minutes_before_kickoff=60,
        form_windows=(5, 10),
        elo_k=20.0,
        target_type='three_way',
        binary_market='over_2_5',
        n_splits=4,
        random_state=42,
        calibration=True,
        params=None,
    )
    assert len(result.dataset) > 0
    assert len(result.dataset) < 10
    assert result.probabilities.shape[0] == len(result.dataset)
    assert result.probabilities.shape[1] == 3
    assert not np.isnan(result.probabilities).any()
    assert sorted(result.classes) == ['away', 'draw', 'home']
    assert set(result.selected_odds['match_id']) == set(result.dataset['match_id'])



def test_generate_walkforward_predictions_supports_home_win_binary_market() -> None:
    result = generate_walkforward_predictions(
        matches_csv=FIXTURES_DIR / 'toy_matches.csv',
        odds_csv=FIXTURES_DIR / 'toy_odds_snapshots.csv',
        snapshot_minutes_before_kickoff=60,
        form_windows=(5, 10),
        elo_k=20.0,
        target_type='binary',
        binary_market='home_win',
        n_splits=4,
        random_state=42,
        calibration=True,
        params=None,
    )
    assert len(result.dataset) > 0
    assert result.probabilities.shape[0] == len(result.dataset)
    assert result.probabilities.shape[1] == 2
    assert not np.isnan(result.probabilities).any()
    assert result.classes == ['0', '1']
    assert (result.selected_odds['market'] == 'home_win').any()
