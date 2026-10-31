import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.policy.adaptive_v2_features import FEATURE_NAMES
from src.policy.train_adaptive_selector_v2 import (
    K_VALUES,
    choose_k,
    make_relative_loss_targets,
    predict_relative_losses,
    target_clip_bounds,
)


def test_relative_targets_use_full_as_zero_reference():
    losses = np.array([
        [0.2, 0.3, 0.4],
        [0.5, 0.4, 0.6],
    ])
    targets = make_relative_loss_targets(losses)

    assert targets.shape == (2, 2)
    np.testing.assert_allclose(targets, [[-0.2, -0.1], [-0.1, -0.2]])


def test_target_clip_bounds_are_per_output_and_finite():
    targets = np.array([
        [-1.0, 0.0],
        [-0.5, 0.2],
        [0.0, 0.4],
        [0.5, 0.6],
        [10.0, 20.0],
    ])
    lower, upper = target_clip_bounds(targets)

    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(lower < upper)


def test_prediction_appends_full_zero_reference():
    x = pd.DataFrame(np.arange(8 * len(FEATURE_NAMES)).reshape(8, -1),
                     columns=FEATURE_NAMES)
    # The predictor estimates one delta for each truncated K; Full is
    # appended by predict_relative_losses as an exact zero reference.
    y = np.column_stack([
        np.linspace(-0.5 + index * 0.01, 0.5 + index * 0.01, len(x))
        for index in range(len(K_VALUES) - 1)
    ])
    model = DummyRegressor(strategy="mean").fit(x, y)
    predicted = predict_relative_losses(
        model,
        x,
        np.full(len(K_VALUES) - 1, -1.0),
        np.full(len(K_VALUES) - 1, 1.0),
    )

    assert predicted.shape == (len(x), len(K_VALUES))
    assert predicted.shape[1] == len(K_VALUES)
    np.testing.assert_array_equal(predicted[:, -1], np.zeros(len(x)))


def test_cost_policy_can_choose_zero_history_when_quality_delta_supports_it():
    # K=0 is the best predicted quality option here; it should remain selectable.
    predicted_delta = np.array([[-0.2, -0.1, 0.0]])
    costs = np.array([[0.0, 0.2, 1.0]])
    selected = choose_k(predicted_delta, costs, penalty=0.01)

    assert selected.tolist() == [0]
