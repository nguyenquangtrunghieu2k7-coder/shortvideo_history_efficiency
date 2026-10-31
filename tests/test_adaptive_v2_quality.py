import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier

from src.policy.adaptive_v2_features import FEATURE_NAMES
from src.policy.train_adaptive_selector_v2 import (
    K_VALUES,
    choose_k_by_acceptance,
    make_acceptance_targets,
    predict_acceptance_probabilities,
)


def test_acceptance_targets_compare_each_k_to_full():
    losses = np.array([
        [0.30, 0.40, 0.32],
        [0.50, 0.40, 0.60],
    ])
    targets = make_acceptance_targets(losses, max_logloss_regret=0.05)

    assert targets.shape == (2, 2)
    np.testing.assert_array_equal(targets, [[1, 0], [1, 1]])


def test_acceptance_predictor_returns_one_probability_per_truncated_k():
    x = pd.DataFrame(
        np.arange(12 * len(FEATURE_NAMES)).reshape(12, -1),
        columns=FEATURE_NAMES,
    )
    y = np.column_stack([
        np.array([(row + k) % 2 for row in range(len(x))])
        for k in range(len(K_VALUES) - 1)
    ])
    model = MultiOutputClassifier(DummyClassifier(strategy="prior"))
    model.fit(x, y)

    probabilities = predict_acceptance_probabilities(model, x)

    assert probabilities.shape == (len(x), len(K_VALUES) - 1)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_constant_output_classifier_is_handled_without_missing_class_error():
    x = pd.DataFrame(
        np.arange(10 * len(FEATURE_NAMES)).reshape(10, -1),
        columns=FEATURE_NAMES,
    )
    y = np.column_stack([
        np.ones(len(x), dtype=int) if i == 0 else np.zeros(len(x), dtype=int)
        for i in range(len(K_VALUES) - 1)
    ])
    model = MultiOutputClassifier(DummyClassifier(strategy="prior"))
    model.fit(x, y)

    probabilities = predict_acceptance_probabilities(model, x)

    np.testing.assert_allclose(probabilities[:, 0], 1.0)
    np.testing.assert_allclose(probabilities[:, 1:], 0.0)


def test_policy_can_choose_k0_when_its_acceptance_probability_is_high():
    probabilities = np.array([
        [0.95, 0.80, 0.70, 0.65, 0.60, 0.55, 0.50],
        [0.10, 0.20, 0.30, 0.40, 0.80, 0.70, 0.60],
    ])
    selected = choose_k_by_acceptance(probabilities, threshold=0.75)

    assert selected.tolist() == [0, 4]  # K=0 for row 1, K=20 for row 2


def test_policy_falls_back_to_full_when_no_truncated_k_is_acceptable():
    probabilities = np.full((3, len(K_VALUES) - 1), 0.2)
    selected = choose_k_by_acceptance(probabilities, threshold=0.8)

    assert selected.tolist() == [len(K_VALUES) - 1] * 3
