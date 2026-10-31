"""Unit tests for request-time Adaptive V2 history features."""

import numpy as np

from src.policy.adaptive_v2_features import FEATURE_NAMES, history_features


def test_empty_history_returns_finite_zero_features():
    frame = history_features([[]])
    assert list(frame.columns) == FEATURE_NAMES
    assert frame.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(frame.to_numpy()).all()
    assert frame.iloc[0]["history_length"] == 0
    assert frame.iloc[0]["unique_count"] == 0
    assert frame.iloc[0]["normalized_entropy"] == 0


def test_repeated_history_captures_concentration_and_adjacent_repeats():
    frame = history_features([[7, 7, 7, 7]])
    row = frame.iloc[0]
    assert row["history_length"] == 4
    assert row["unique_count"] == 1
    assert row["unique_ratio"] == 0.25
    assert row["max_item_share"] == 1.0
    assert row["top5_item_share"] == 1.0
    assert row["adjacent_repeat_ratio"] == 1.0
    assert row["max_adjacent_repeat_run"] == 4


def test_diverse_history_has_maximal_entropy():
    frame = history_features([[1, 2, 3, 4, 5]])
    row = frame.iloc[0]
    assert row["unique_count"] == 5
    assert np.isclose(row["normalized_entropy"], 1.0)
    assert np.isclose(row["effective_unique_count"], 5.0)
    assert row["max_item_share"] == 0.2
    assert row["adjacent_repeat_ratio"] == 0.0


def test_recent_novelty_is_relative_to_older_history():
    history = list(range(100)) + list(range(100, 200))
    row = history_features([history]).iloc[0]
    assert row["new_unique_in_last_100"] == 100
    assert row["novelty_rate_last_100"] == 1.0
    assert row["head_tail_overlap_100"] == 0.0


def test_feature_extraction_is_deterministic_and_numeric():
    histories = [[1, 2, 1, 3], [4, 4], []]
    first = history_features(histories)
    second = history_features(histories)
    assert first.equals(second)
    assert np.isfinite(first.to_numpy()).all()
