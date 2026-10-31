import numpy as np
import pandas as pd

from src.policy.adaptive_v6_policy import (
    _dominates,
    _pareto_frontier,
    _knee_point,
)


def _row(policy, auc, saving):
    return {
        "policy": policy,
        "auc": auc,
        "history_saving": saving,
        "history_cost_ratio": 1.0 - saving,
    }


def test_pareto_frontier_removes_dominated_candidates():
    candidates = pd.DataFrame([
        _row("A", 0.90, 0.10),
        _row("B", 0.85, 0.05),
        _row("C", 0.80, 0.70),
        _row("D", 0.70, 0.90),
    ])
    frontier, duplicate_count = _pareto_frontier(candidates)
    assert set(frontier["policy"]) == {"A", "C", "D"}
    assert duplicate_count == 0
    assert _dominates(_row("A", 0.90, 0.10), _row("B", 0.85, 0.05)) is True


def test_duplicate_coordinates_are_collapsed_deterministically():
    candidates = pd.DataFrame([
        {**_row("z-policy", 0.90, 0.10), "history_cost_ratio": 0.90},
        {**_row("a-policy", 0.90, 0.10), "history_cost_ratio": 0.90},
        {**_row("middle", 0.80, 0.80), "history_cost_ratio": 0.20},
    ])
    frontier, duplicate_count = _pareto_frontier(candidates)
    assert duplicate_count == 1
    assert "a-policy" in set(frontier["policy"])
    assert "z-policy" not in set(frontier["policy"])


def test_knee_point_selects_bend_when_curve_is_on_either_chord_side():
    # Same geometric bend, mirrored around the endpoint chord.
    for knee_auc in (0.80, 0.62):
        frontier = pd.DataFrame([
            _row("quality-end", 0.90, 0.00),
            _row("knee", knee_auc, 0.80),
            _row("saving-end", 0.60, 1.00),
        ])
        selected, annotated, reason = _knee_point(frontier)
        assert reason is None
        assert selected["policy"] == "knee"
        assert selected["knee_distance"] > 0
        assert annotated["signed_knee_distance"].notna().sum() == 1


def test_knee_point_identifies_bend_in_observed_v5_frontier_shape():
    frontier = pd.DataFrame([
        _row("Full", 0.6485928545, 0.000000),
        _row("lambda-0", 0.6312650896, 0.250790),
        _row("lambda-0011", 0.6286983, 0.268633),
        _row("lambda-0203", 0.6264047, 0.298559),
        _row("lambda-0492", 0.62568894, 0.482877),
        _row("lambda-0661", 0.62297708, 0.574135),
    ])
    selected, annotated, reason = _knee_point(frontier)
    assert reason is None
    assert selected["policy"] == "lambda-0203"
    assert selected["knee_distance"] > annotated["knee_distance"].dropna().median()


def test_knee_point_falls_back_for_nearly_linear_frontier():
    frontier = pd.DataFrame([
        _row("quality-end", 0.90, 0.00),
        _row("middle", 0.80, 0.50),
        _row("saving-end", 0.70, 1.00),
    ])
    selected, annotated, reason = _knee_point(frontier)
    assert selected is None
    assert reason == "frontier_nearly_linear_no_stable_knee"

def test_knee_point_uses_full_fallback_when_frontier_has_too_few_points():
    frontier = pd.DataFrame([
        _row("quality-end", 0.90, 0.00),
        _row("saving-end", 0.60, 1.00),
    ])
    selected, annotated, reason = _knee_point(frontier)
    assert selected is None
    assert reason == "fewer_than_three_distinct_pareto_points"


def test_knee_point_rejects_zero_objective_range():
    frontier = pd.DataFrame([
        _row("a", 0.90, 0.00),
        _row("b", 0.90, 0.50),
        _row("c", 0.90, 1.00),
    ])
    selected, annotated, reason = _knee_point(frontier)
    assert selected is None
    assert reason == "zero_auc_range"
