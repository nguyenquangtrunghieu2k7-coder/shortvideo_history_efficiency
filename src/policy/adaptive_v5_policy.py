"""Pareto knee-point selection for Adaptive History Selector V5.

Candidate generation, uplift prediction, action application, and evaluation are
reused from V4. Only validation policy selection changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from adaptive_v4_policy import *  # noqa: F401,F403
from adaptive_v4_policy import (
    K_VALUES, actions_for_fixed_k, choose_actions, evaluate_actions,
)


def _dominates(a, b, tol=1e-12):
    no_worse = (
        a["auc"] >= b["auc"] - tol
        and a["history_saving"] >= b["history_saving"] - tol
    )
    strictly_better = (
        a["auc"] > b["auc"] + tol
        or a["history_saving"] > b["history_saving"] + tol
    )
    return no_worse and strictly_better


def _deduplicate_coordinates(frame, tol=1e-12):
    """Keep one deterministic representative per numerically equal (AUC, saving)."""
    ordered = frame.sort_values(
        ["auc", "history_cost_ratio", "policy"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    kept = []
    for row in ordered.to_dict("records"):
        duplicate = any(
            abs(row["auc"] - old["auc"]) <= tol
            and abs(row["history_saving"] - old["history_saving"]) <= tol
            for old in kept
        )
        if not duplicate:
            kept.append(row)
    return pd.DataFrame(kept), len(frame) - len(kept)


def _pareto_frontier(feasible):
    unique, duplicate_count = _deduplicate_coordinates(feasible)
    records = unique.to_dict("records")
    frontier = [
        row for i, row in enumerate(records)
        if not any(_dominates(other, row) for j, other in enumerate(records) if i != j)
    ]
    result = pd.DataFrame(frontier).sort_values(
        ["history_saving", "auc", "policy"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return result, duplicate_count


def _knee_point(frontier, full_policy="Fixed-K-Full", tol=1e-12):
    """Return (selected row or None, annotated frontier, reason)."""
    f = frontier.copy()
    f["normalized_history_saving"] = np.nan
    f["normalized_quality"] = np.nan
    f["signed_knee_distance"] = np.nan

    if len(f) < 3:
        return None, f, "fewer_than_three_distinct_pareto_points"

    s_min, s_max = float(f["history_saving"].min()), float(f["history_saving"].max())
    q_min, q_max = float(f["auc"].min()), float(f["auc"].max())
    if s_max - s_min <= tol:
        return None, f, "zero_history_saving_range"
    if q_max - q_min <= tol:
        return None, f, "zero_auc_range"

    f["normalized_history_saving"] = (f["history_saving"] - s_min) / (s_max - s_min)
    f["normalized_quality"] = (f["auc"] - q_min) / (q_max - q_min)

    # Endpoints are selected by the objective definitions, not row order.
    a_idx = f.sort_values(
        ["auc", "history_cost_ratio", "policy"],
        ascending=[False, True, True], kind="mergesort"
    ).index[0]
    b_idx = f.sort_values(
        ["history_saving", "auc", "policy"],
        ascending=[False, False, True], kind="mergesort"
    ).index[0]
    if a_idx == b_idx:
        return None, f, "same_quality_and_saving_endpoint"

    a = f.loc[a_idx]
    b = f.loc[b_idx]
    ax, ay = float(a["normalized_history_saving"]), float(a["normalized_quality"])
    bx, by = float(b["normalized_history_saving"]), float(b["normalized_quality"])
    denominator = float(np.hypot(bx - ax, by - ay))
    if denominator <= tol:
        return None, f, "degenerate_endpoint_chord"

    interior = f.drop(index=[a_idx, b_idx]).copy()
    if interior.empty:
        return None, f, "no_interior_pareto_points"

    signed = (
        (bx - ax) * (interior["normalized_quality"] - ay)
        - (by - ay) * (interior["normalized_history_saving"] - ax)
    ) / denominator
    f.loc[interior.index, "signed_knee_distance"] = signed
    candidates = f.loc[interior.index].copy()
    candidates = candidates[candidates["signed_knee_distance"] > tol]
    if candidates.empty:
        return None, f, "no_interior_point_on_ideal_side_of_chord"

    candidates = candidates.sort_values(
        ["signed_knee_distance", "auc", "history_saving", "policy"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return candidates.iloc[0].to_dict(), f, None


def calibrate_knee_policy(
    y_true, predictions, histories, uplift_predictions,
    max_auc_drop, lambda_steps,
):
    """Select the validation Pareto knee; never reads held-out test data."""
    y = np.asarray(y_true, dtype=np.int8)
    if y.ndim != 1 or len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Validation labels must be a non-empty binary vector.")
    if len(histories) != len(y):
        raise ValueError("Validation histories and labels must have equal length.")
    if not np.isfinite(float(max_auc_drop)) or float(max_auc_drop) < 0:
        raise ValueError("max_auc_drop must be finite and non-negative.")
    if int(lambda_steps) < 2:
        raise ValueError("lambda_steps must be at least 2.")

    full_scores = np.asarray(predictions["Full"], dtype=float)
    full_auc = float(roc_auc_score(y, full_scores))
    full_cost = float(np.mean([len(h) for h in histories]))
    if not np.isfinite(full_cost) or full_cost <= 0:
        raise ValueError("Full-history validation cost must be positive and finite.")

    cost_scale = full_cost
    lambdas = np.concatenate(([0.0], np.geomspace(1e-4, 10.0, int(lambda_steps) - 1)))
    rows = []

    def add_candidate(kind, name, actions, penalty=None, fixed_k=None):
        metrics, _ = evaluate_actions(y, predictions, histories, actions, name)
        auc = float(metrics["auc"])
        cost_ratio = float(metrics["history_cost_ratio"])
        saving = float(1.0 - cost_ratio)
        feasible = bool(auc >= full_auc - float(max_auc_drop) - 1e-12)
        rows.append({
            "policy": name,
            "policy_type": kind,
            "penalty": penalty,
            "fixed_K": fixed_k,
            "auc": auc,
            "auc_delta_vs_full": auc - full_auc,
            "history_cost_ratio": cost_ratio,
            "history_reduction_vs_full": saving,
            "history_saving": saving,
            "mean_history_interactions": float(metrics["mean_history_interactions"]),
            "selection_feasible": feasible,
            "is_pareto_frontier": False,
            "normalized_history_saving": np.nan,
            "normalized_quality": np.nan,
            "signed_knee_distance": np.nan,
            "selected_for_test": False,
        })

    for k in K_VALUES:
        add_candidate("fixed", f"Fixed-K-{k}", actions_for_fixed_k(len(y), k), fixed_k=str(k))

    for penalty in lambdas:
        actions = choose_actions(
            uplift_predictions, histories, float(penalty), cost_scale
        )
        add_candidate("learned", f"V5-lambda-{penalty:.8g}", actions, penalty=float(penalty))

    all_candidates = pd.DataFrame(rows)
    feasible = all_candidates[all_candidates["selection_feasible"]].copy()
    if feasible.empty:
        raise RuntimeError("Full fallback was not feasible; check candidate generation and AUC calculations.")

    frontier, duplicate_count = _pareto_frontier(feasible)
    selected_row, annotated_frontier, fallback_reason = _knee_point(frontier)

    if selected_row is None:
        # Full is the conservative, always-available fallback.
        selected = all_candidates.loc[
            all_candidates["policy"].eq("Fixed-K-Full")
        ].iloc[0].to_dict()
        selected["selection_status"] = "fallback_full"
        selected["fallback_reason"] = fallback_reason
        selected["deployment_mode"] = "fixed_k_fallback"
        selected["selection_objective"] = "validation_pareto_knee_point"
    else:
        selected = selected_row
        selected["selection_status"] = "knee_selected"
        selected["fallback_reason"] = None
        selected["deployment_mode"] = (
            "learned_adaptive" if selected["policy_type"] == "learned"
            else "fixed_k_fallback"
        )
        selected["selection_objective"] = "validation_pareto_knee_point"

    selected["fixed_K"] = selected.get("fixed_K")
    selected["validation_full_auc"] = full_auc
    selected["validation_auc_drop_tolerance"] = float(max_auc_drop)
    selected["validation_feasible_candidate_count"] = int(len(feasible))
    selected["validation_pareto_point_count"] = int(len(frontier))
    selected["validation_duplicate_coordinate_count"] = int(duplicate_count)
    selected["quality_endpoint_policy"] = str(
        frontier.sort_values(["auc", "history_cost_ratio", "policy"],
                             ascending=[False, True, True]).iloc[0]["policy"]
    )
    selected["history_saving_endpoint_policy"] = str(
        frontier.sort_values(["history_saving", "auc", "policy"],
                             ascending=[False, False, True]).iloc[0]["policy"]
    )

    # Copy frontier annotations and mark its selected point.
    frontier_out = annotated_frontier.copy()
    frontier_out["is_pareto_frontier"] = True
    frontier_out["selected_for_test"] = frontier_out["policy"].eq(selected["policy"])
    all_candidates = all_candidates.drop(
        columns=["is_pareto_frontier", "normalized_history_saving",
                 "normalized_quality", "signed_knee_distance", "selected_for_test"]
    ).merge(
        frontier_out[["policy", "is_pareto_frontier", "normalized_history_saving",
                      "normalized_quality", "signed_knee_distance", "selected_for_test"]],
        on="policy", how="left",
    )
    for col in ["is_pareto_frontier", "selected_for_test"]:
        all_candidates[col] = all_candidates[col].fillna(False).astype(bool)
    return selected, all_candidates, frontier_out, cost_scale, full_auc


def apply_selected_v5_policy_from_lengths(
    selected, model, features, history_lengths, cost_scale, feature_names
):
    """Apply a frozen V5 policy using V4's unchanged action semantics."""
    return apply_selected_policy_from_lengths(
        selected, model, features, history_lengths, cost_scale, feature_names
    )
