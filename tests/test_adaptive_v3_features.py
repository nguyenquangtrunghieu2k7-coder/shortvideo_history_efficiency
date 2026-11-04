import numpy as np

from src.policy.adaptive_v3_features import FEATURE_NAMES, history_features


def test_v3_features_ignore_old_prefix_when_suffix_and_length_match():
    suffix = list(range(100, 200))
    first = list(range(100)) + suffix
    second = list(range(1000, 1100)) + suffix

    a = history_features([first])
    b = history_features([second])

    assert list(a.columns) == FEATURE_NAMES
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_v3_features_handle_empty_and_short_histories():
    frame = history_features([[], [1, 1, 2]])
    assert frame.shape == (2, len(FEATURE_NAMES))
    assert frame.loc[0, "history_length"] == 0
    assert frame.loc[1, "history_length"] == 3
    assert frame.loc[1, "unique_last_5"] == 2 / 3


def test_v3_features_are_bounded_to_recent_window():
    history = list(range(10000))
    frame = history_features([history])
    assert frame.loc[0, "history_length"] == 10000
    assert frame.loc[0, "recent_unique_count_100"] == 100
