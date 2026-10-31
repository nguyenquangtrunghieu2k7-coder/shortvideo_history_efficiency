import numpy as np

from src.policy.adaptive_v4_policy import (
    K_VALUES,
    auc_contribution_uplift_targets,
    choose_actions,
    interaction_matrix,
    stratified_paired_auc_difference_ci,
    evaluate_actions,
    balanced_tradeoff_utility,
)


def test_auc_uplift_targets_are_zero_for_full_reference():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = {
        0: np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        1: np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6]),
        5: np.array([0.3, 0.7, 0.4, 0.6, 0.5, 0.5]),
        10: np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        20: np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        50: np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        100: np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        "Full": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
    }
    uplift = auc_contribution_uplift_targets(y, p)
    assert uplift.shape == (len(y), 7)
    assert np.allclose(uplift[:, 3:], 0.0)


def test_interaction_cost_respects_available_history_length():
    histories = [[1], [1, 2, 3], list(range(150))]
    costs = interaction_matrix(histories)
    assert costs[:, 0].tolist() == [0, 0, 0]
    assert costs[:, 1].tolist() == [1, 1, 1]
    assert costs[:, 2].tolist() == [1, 3, 5]
    assert costs[:, -1].tolist() == [1, 3, 150]


def test_choose_actions_always_has_full_fallback_action():
    histories = [list(range(20)), list(range(100))]
    uplift = np.full((2, 7), -0.1)
    actions = choose_actions(uplift, histories, penalty=0.0, cost_scale=100)
    assert actions.tolist() == [7, 7]


def test_bootstrap_is_paired_and_reproducible():
    y = np.array([0, 1] * 30)
    full = np.tile([0.2, 0.8], 30)
    policy = full.copy()
    low1, high1 = stratified_paired_auc_difference_ci(y, policy, full, repeats=30, random_state=7)
    low2, high2 = stratified_paired_auc_difference_ci(y, policy, full, repeats=30, random_state=7)
    assert low1 == low2 == 0.0
    assert high1 == high2 == 0.0


def test_evaluate_actions_reports_interaction_reduction():
    y = np.array([0, 1, 0, 1])
    histories = [list(range(10)) for _ in y]
    predictions = {k: np.array([0.1, 0.9, 0.2, 0.8]) for k in K_VALUES}
    actions = np.zeros(len(y), dtype=int)
    metrics, _ = evaluate_actions(y, predictions, histories, actions, "K0")
    assert metrics["history_reduction_vs_full"] == 1.0



def test_balanced_utility_rewards_both_quality_and_history_saving():
    q, saving, utility = balanced_tradeoff_utility(
        auc=0.63, full_auc=0.65, history_cost_ratio=0.5,
        max_auc_drop=0.03, quality_weight=0.6, history_cost_weight=0.4,
    )
    assert np.isclose(q, 1.0 - (0.02 / 0.03))
    assert np.isclose(saving, 0.5)
    assert np.isclose(utility, q ** 0.6 * saving ** 0.4)


def test_full_history_has_zero_history_saving_utility():
    q, saving, utility = balanced_tradeoff_utility(
        auc=0.65, full_auc=0.65, history_cost_ratio=1.0,
        max_auc_drop=0.03,
    )
    assert q == 1.0
    assert saving == 0.0
    assert utility == 0.0


def test_balanced_utility_requires_weights_to_sum_to_one():
    import pytest
    with pytest.raises(ValueError):
        balanced_tradeoff_utility(
            auc=0.63, full_auc=0.65, history_cost_ratio=0.5,
            max_auc_drop=0.03, quality_weight=0.6, history_cost_weight=0.5,
        )
