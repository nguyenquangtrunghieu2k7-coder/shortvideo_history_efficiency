"""Core policy-learning utilities for Adaptive History Selector V3.

V3 learns request-level *AUC contribution uplift* relative to the Full-history
reference. The global AUC constraint is then enforced on a disjoint validation
calibration split using paired bootstrap confidence bounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss


K_VALUES = [0, 1, 5, 10, 20, 50, 100, "Full"]
TRUNCATED_K_VALUES = K_VALUES[:-1]
EPS = 1e-7


def score_matrix(predictions: dict, k_values=K_VALUES) -> np.ndarray:
    matrix = np.column_stack([np.asarray(predictions[k], dtype=float) for k in k_values])
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Predictions must form a finite 2-D matrix.")
    return matrix


def auc_contribution_uplift_targets(y_true, predictions, k_values=K_VALUES):
    """Return per-request one-sided AUC-contribution uplift over Full.

    For a positive request, contribution is the fraction of negative Full
    scores it outranks. For a negative request, it is the fraction of positive
    Full scores that outrank it. Ties count as 0.5. Each truncated action is
    compared with the same Full-reference scores from the same partition.

    This is a lower-variance, ranking-aligned supervised proxy, not an exact
    decomposition of the AUC after multiple requests change actions. The
    latter is evaluated directly during policy calibration.
    """
    y = np.asarray(y_true, dtype=np.int8)
    scores = score_matrix(predictions, k_values)
    if len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Need aligned labels containing both binary classes.")

    positive = y == 1
    negative = y == 0
    full_scores = scores[:, -1]
    positive_full = full_scores[positive]
    negative_full = full_scores[negative]

    def contribution(action_scores):
        result = np.empty(len(y), dtype=np.float64)
        # A positive request is good when its score outranks negative Full scores.
        pos_scores = action_scores[positive, None]
        pos_cmp = (pos_scores > negative_full[None, :]).mean(axis=1)
        pos_ties = (pos_scores == negative_full[None, :]).mean(axis=1)
        result[positive] = pos_cmp + 0.5 * pos_ties
        # A negative request is good when positive Full scores outrank its score.
        neg_scores = action_scores[None, negative]
        neg_cmp = (positive_full[:, None] > neg_scores).mean(axis=0)
        neg_ties = (positive_full[:, None] == neg_scores).mean(axis=0)
        result[negative] = neg_cmp + 0.5 * neg_ties
        return result

    full_contribution = contribution(full_scores)
    uplift = np.column_stack([
        contribution(scores[:, j]) - full_contribution
        for j in range(scores.shape[1] - 1)
    ])
    return uplift


def _candidate_regressors(args):
    return {
        "ExtraTreesRegressor": MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=args.trees,
                min_samples_leaf=args.min_samples_leaf,
                max_features=0.9,
                random_state=args.random_state,
                n_jobs=1,
            ), n_jobs=-1
        ),
        "RandomForestRegressor": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=args.trees,
                min_samples_leaf=args.min_samples_leaf,
                max_features=0.9,
                random_state=args.random_state,
                n_jobs=1,
            ), n_jobs=-1
        ),
        "HistGradientBoostingRegressor": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=max(10, args.min_samples_leaf),
                l2_regularization=1.0,
                random_state=args.random_state,
            ), n_jobs=-1
        ),
        "MeanBaseline": DummyRegressor(strategy="mean"),
    }


def _class_balanced_weights(labels):
    y = np.asarray(labels, dtype=np.int8)
    positive = y == 1
    negative = y == 0
    if not positive.any() or not negative.any():
        raise ValueError("Class-balanced fitting requires both labels.")
    weights = np.where(positive, 0.5 / positive.sum(), 0.5 / negative.sum())
    return weights * (len(weights) / weights.sum())


def fit_auc_uplift_predictor(features, labels, predictions, args, feature_names):
    """Model selection inside development validation; calibration stays separate."""
    y = np.asarray(labels, dtype=np.int8)
    x = features[feature_names].reset_index(drop=True)
    fit_idx, select_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.20,
        random_state=args.random_state + 11,
        stratify=y,
    )
    # Targets are built independently inside each partition. In particular,
    # calibration-partition labels/scores never contribute to fit targets.
    fit_pred = {k: np.asarray(predictions[k])[fit_idx] for k in K_VALUES}
    select_pred = {k: np.asarray(predictions[k])[select_idx] for k in K_VALUES}
    y_fit = auc_contribution_uplift_targets(y[fit_idx], fit_pred)
    y_select = auc_contribution_uplift_targets(y[select_idx], select_pred)
    fit_weights = _class_balanced_weights(y[fit_idx])

    candidates = _candidate_regressors(args)
    rows = []
    for name, model in candidates.items():
        model.fit(x.iloc[fit_idx], y_fit, sample_weight=fit_weights)
        predicted = np.asarray(model.predict(x.iloc[select_idx]), dtype=float)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, y_fit.shape[1])
        # AUC weights positive and negative classes equally; mirror that here.
        absolute_error = np.abs(predicted - y_select)
        positive_select = y[select_idx] == 1
        negative_select = y[select_idx] == 0
        mae_by_k = 0.5 * (
            absolute_error[positive_select].mean(axis=0)
            + absolute_error[negative_select].mean(axis=0)
        )
        # Class-balanced per-request best-action agreement is secondary.
        predicted_best = np.argmax(np.column_stack([predicted, np.zeros(len(predicted))]), axis=1)
        actual_best = np.argmax(np.column_stack([y_select, np.zeros(len(y_select))]), axis=1)
        agreement = 0.5 * (
            np.mean(predicted_best[positive_select] == actual_best[positive_select])
            + np.mean(predicted_best[negative_select] == actual_best[negative_select])
        )
        rows.append({
            "model": name,
            "selection_macro_mae": float(np.mean(mae_by_k)),
            "selection_best_action_agreement": float(agreement),
        })
    table = pd.DataFrame(rows).sort_values(
        ["selection_macro_mae", "selection_best_action_agreement"],
        ascending=[True, False],
    ).reset_index(drop=True)
    selected_name = str(table.iloc[0]["model"])

    # Refit the selected model on all development rows. The policy-calibration
    # subset remains untouched and is used only for lambda/policy selection.
    development_targets = auc_contribution_uplift_targets(y, predictions)
    final_model = candidates[selected_name]
    final_model.fit(x, development_targets, sample_weight=_class_balanced_weights(y))
    return final_model, selected_name, table


def predict_auc_uplift(model, features, feature_names):
    values = np.asarray(model.predict(features[feature_names]), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, len(TRUNCATED_K_VALUES))
    if values.shape != (len(features), len(TRUNCATED_K_VALUES)):
        raise ValueError(f"Unexpected uplift prediction shape: {values.shape}")
    return values


def interaction_matrix_from_lengths(history_lengths, k_values=K_VALUES):
    """Return per-request interaction counts for each action in K_VALUES order."""
    lengths = np.asarray(history_lengths, dtype=np.int64)
    if lengths.ndim != 1 or np.any(lengths < 0):
        raise ValueError("History lengths must be a non-negative 1-D array.")
    columns = []
    for k in k_values:
        if k == 0:
            used = np.zeros_like(lengths)
        elif k == "Full":
            used = lengths.copy()
        else:
            used = np.minimum(lengths, int(k))
        columns.append(used)
    return np.column_stack(columns)


def interaction_matrix(histories, k_values=K_VALUES):
    return interaction_matrix_from_lengths([len(h) for h in histories], k_values)


def choose_actions_from_lengths(uplift_predictions, history_lengths, penalty, cost_scale):
    """Choose K using the shared uplift-minus-normalized-cost rule."""
    uplift = np.column_stack([
        np.asarray(uplift_predictions, dtype=float),
        np.zeros(len(uplift_predictions), dtype=float),  # Full has zero uplift.
    ])
    costs = interaction_matrix_from_lengths(history_lengths).astype(np.float64)
    costs /= max(float(cost_scale), 1.0)
    if uplift.shape != costs.shape:
        raise ValueError(
            f"Uplift/cost action matrices must align; got {uplift.shape} and {costs.shape}."
        )
    utility = uplift - float(penalty) * costs
    return np.argmax(utility, axis=1).astype(np.int64)


def choose_actions(uplift_predictions, histories, penalty, cost_scale):
    """History-list wrapper around the canonical length-based action selector."""
    return choose_actions_from_lengths(
        uplift_predictions, [len(h) for h in histories], penalty, cost_scale
    )


def actions_for_fixed_k(n_rows, k):
    return np.full(n_rows, K_VALUES.index(k), dtype=np.int64)


def selected_scores(predictions, actions):
    matrix = score_matrix(predictions)
    return matrix[np.arange(len(actions)), actions]


def evaluate_actions(y_true, predictions, histories, actions, name):
    y = np.asarray(y_true, dtype=np.int8)
    scores = selected_scores(predictions, actions)
    used = interaction_matrix(histories)[np.arange(len(actions)), actions]
    full_used = interaction_matrix(histories)[:, -1]
    total = int(used.sum())
    full_total = int(full_used.sum())
    return {
        "policy": name,
        "auc": float(roc_auc_score(y, scores)),
        "logloss": float(log_loss(y, np.clip(scores, EPS, 1.0 - EPS), labels=[0, 1])),
        "mean_history_interactions": float(used.mean()),
        "total_history_interactions": total,
        "history_reduction_vs_full": float(1.0 - total / max(full_total, 1)),
        "history_cost_ratio": float(total / max(full_total, 1)),
    }, scores


def stratified_paired_auc_difference_ci(y_true, policy_scores, full_scores,
                                        repeats=200, random_state=42):
    """Paired, class-stratified bootstrap CI for AUC(policy)-AUC(Full)."""
    y = np.asarray(y_true, dtype=np.int8)
    policy_scores = np.asarray(policy_scores, dtype=float)
    full_scores = np.asarray(full_scores, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC bootstrap requires both classes.")
    rng = np.random.default_rng(random_state)
    differences = np.empty(repeats, dtype=np.float64)
    for b in range(repeats):
        sample = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        differences[b] = (
            roc_auc_score(y[sample], policy_scores[sample])
            - roc_auc_score(y[sample], full_scores[sample])
        )
    return float(np.quantile(differences, 0.05)), float(np.quantile(differences, 0.95))


def calibrate_policy(y_true, predictions, histories, uplift_predictions,
                     max_auc_drop, lambda_steps, bootstrap_repeats,
                     random_state=42):
    """Select the highest-quality candidate that passes tuning non-inferiority.

    Both the point-estimate AUC constraint and paired-bootstrap lower-bound
    constraint must pass. Among those candidates, maximize global tuning AUC;
    use lower history cost only as a tie-breaker. Independent certification
    remains the final gate.
    """
    y = np.asarray(y_true, dtype=np.int8)
    full_scores = np.asarray(predictions["Full"], dtype=float)
    full_auc = float(roc_auc_score(y, full_scores))
    mean_full_length = max(float(np.mean([len(h) for h in histories])), 1.0)
    lambdas = np.concatenate(([0.0], np.geomspace(1e-4, 10.0, lambda_steps - 1)))
    candidates = []

    def add_candidate(kind, name, actions, penalty=None, fixed_k=None):
        metrics, scores = evaluate_actions(y, predictions, histories, actions, name)
        ci_low, ci_high = stratified_paired_auc_difference_ci(
            y, scores, full_scores, repeats=bootstrap_repeats,
            random_state=random_state + len(candidates),
        )
        delta = metrics["auc"] - full_auc
        point_feasible = bool(delta >= -float(max_auc_drop))
        ci_feasible = bool(ci_low >= -float(max_auc_drop))
        candidates.append({
            "policy": name,
            "policy_type": kind,
            "penalty": penalty,
            "fixed_K": fixed_k,
            "auc": metrics["auc"],
            "auc_delta_vs_full": delta,
            "auc_delta_ci_low_90": ci_low,
            "auc_delta_ci_high_90": ci_high,
            "history_cost_ratio": metrics["history_cost_ratio"],
            "history_reduction_vs_full": metrics["history_reduction_vs_full"],
            "mean_history_interactions": metrics["mean_history_interactions"],
            "feasible": point_feasible,
            "ci_feasible_on_tuning": ci_feasible,
            "selection_feasible": bool(point_feasible and ci_feasible),
        })

    for k in K_VALUES:
        add_candidate("fixed", f"Fixed-K-{k}", actions_for_fixed_k(len(y), k), fixed_k=str(k))

    for penalty in lambdas:
        actions = choose_actions(
            uplift_predictions, histories, float(penalty), mean_full_length
        )
        add_candidate("learned", f"V3-lambda-{penalty:.8g}", actions, penalty=float(penalty))

    table = pd.DataFrame(candidates)
    eligible = table[table["selection_feasible"]].copy()
    if eligible.empty:
        # Full must pass: its AUC delta and paired difference are exactly zero.
        raise RuntimeError("No policy passed tuning constraints, including Full.")

    # Quality first. History cost is only a tie-breaker, not the objective.
    eligible = eligible.sort_values(
        ["auc", "history_cost_ratio"],
        ascending=[False, True],
    ).reset_index(drop=True)
    selected = eligible.iloc[0].to_dict()
    selected["deployment_mode"] = (
        "learned_adaptive" if selected["policy_type"] == "learned"
        else "fixed_k_fallback"
    )
    table["selected_for_test"] = table["policy"].eq(selected["policy"])
    return selected, table, mean_full_length, full_auc


def apply_heuristic_policy(policy_name, features, history_lengths):
    """Apply a validation-selected transparent heuristic without retuning it."""
    parts = policy_name.split("-")
    lengths = np.asarray(history_lengths, dtype=np.int64)
    if parts[1] == "length":
        cutoff, low_k, high_k = int(parts[2]), int(parts[3]), parts[4]
        values = lengths
    elif parts[1] == "diversity":
        cutoff, low_k, high_k = float(parts[2]), int(parts[3]), parts[4]
        if "recent_unique_ratio_100" not in features:
            raise ValueError("The selected diversity heuristic requires recent_unique_ratio_100.")
        values = features["recent_unique_ratio_100"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown heuristic policy: {policy_name}")

    high_k = _parse_k(high_k)
    return np.where(
        values <= cutoff,
        K_VALUES.index(low_k),
        K_VALUES.index(high_k),
    ).astype(np.int64)


def apply_selected_policy_from_lengths(
    selected, model, features, history_lengths, cost_scale, feature_names
):
    """Apply exactly the policy frozen after validation certification."""
    mode = selected.get("deployment_mode")
    if mode == "fixed_k_fallback":
        return actions_for_fixed_k(len(features), _parse_k(selected["fixed_K"]))
    if mode == "heuristic":
        return apply_heuristic_policy(
            selected["heuristic_name"], features, history_lengths
        )
    if mode != "learned_adaptive":
        raise ValueError(f"Unknown or unfrozen V3 deployment mode: {mode!r}")

    uplift = predict_auc_uplift(model, features, feature_names)
    return choose_actions_from_lengths(
        uplift, history_lengths, float(selected["penalty"]), cost_scale
    )


def apply_selected_policy(selected, model, features, histories, cost_scale, feature_names):
    """History-list wrapper around the canonical frozen-policy application."""
    return apply_selected_policy_from_lengths(
        selected, model, features, [len(h) for h in histories],
        cost_scale, feature_names,
    )


def _parse_k(value):
    if str(value) == "Full":
        return "Full"
    return int(value)
