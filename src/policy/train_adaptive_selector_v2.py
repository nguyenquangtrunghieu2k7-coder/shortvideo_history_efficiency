"""Adaptive History Selector V2: predict per-K quality, then allocate budget.

Protocol
--------
1. Frozen K-specific recommenders produce validation/test probabilities.
2. The supervised target is each truncated K model's per-request log-loss
   difference relative to Full (Full is the exact zero reference).
3. Validation is split into predictor-fit and policy-calibration subsets;
   ExtraTrees and RandomForest are compared on an inner fit-only split.
4. A scalar cost penalty is selected on the held-out calibration subset by
   minimizing calibration log-loss plus a configured normalized history cost.
5. Test is used once for final evaluation.

This evaluates history-interaction budget, not end-to-end latency, money, energy,
or Bloom-FaaS serving cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "policy"))

import train_baseline as baseline  # noqa: E402
from adaptive_v2_features import FEATURE_NAMES, history_features  # noqa: E402


K_VALUES = [0, 1, 5, 10, 20, 50, 100, "Full"]
EPS = 1e-7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Adaptive History Selector V2."
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument(
        "--budget-weight", type=float, default=0.005,
        help="Weight for normalized mean history cost in calibration utility.",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=0.50,
        help="Upper bound of the history-cost penalty grid.",
    )
    parser.add_argument("--lambda-steps", type=int, default=101)
    return parser.parse_args()


def truncate_history(history, k):
    if k == 0:
        return []
    if k == "Full":
        return history
    return history[-int(k):]


def make_batch(dataset, start, end, k):
    items = [dataset[i] for i in range(start, end)]
    for item in items:
        item["history_ids"] = truncate_history(item["history_ids"], k)
    return baseline.collate_batch(items)


def load_frozen_model(k, vocab_size, device):
    checkpoint_path = baseline.MODEL_DIR / baseline.model_filename("full", k)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing frozen recommender checkpoint: {checkpoint_path}. "
            "Train/freeze Phase 8 full-mode K models first."
        )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = baseline.build_model(vocab_size, mode="full").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_for_k(dataset, k, vocab_size, device, batch_size):
    model = load_frozen_model(k, vocab_size, device)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            (
                candidate_ids,
                history_ids,
                history_mask,
                _labels,
                _users,
                _evaluation_ids,
            ) = make_batch(dataset, start, end, k)
            scores = model(
                candidate_ids.to(device),
                history_ids.to(device),
                history_mask.to(device),
            )
            predictions.extend(scores.detach().cpu().numpy().tolist())

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(predictions, dtype=np.float64)


def collect_predictions(dataset, vocab_size, device, batch_size):
    result = {}
    for k in K_VALUES:
        print(f"  Predicting {getattr(dataset, 'split_name', 'split')} K={k} ...")
        result[k] = predict_for_k(dataset, k, vocab_size, device, batch_size)
    return result


def per_row_logloss(y_true, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y_true, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def make_loss_matrix(y_true, predictions):
    return np.column_stack([
        per_row_logloss(y_true, predictions[k]) for k in K_VALUES
    ])


def make_relative_loss_targets(loss_matrix):
    """Per-request logloss deltas against Full; negative means better than Full."""
    losses = np.asarray(loss_matrix, dtype=np.float64)
    return losses[:, :-1] - losses[:, [-1]]


def target_clip_bounds(targets):
    """Robust target bounds learned only from predictor-fit data."""
    return (
        np.quantile(targets, 0.01, axis=0),
        np.quantile(targets, 0.99, axis=0),
    )


def fit_quality_predictor(features, loss_matrix, labels, args):
    """Select among tree ensembles and a mean baseline on an inner split."""
    targets = make_relative_loss_targets(loss_matrix)
    fit_ids, select_ids = train_test_split(
        np.arange(len(targets)), test_size=0.20,
        random_state=args.random_state + 1, stratify=np.asarray(labels),
    )
    candidates = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=args.trees, min_samples_leaf=args.min_samples_leaf,
            max_features=0.9, random_state=args.random_state, n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=args.trees, min_samples_leaf=args.min_samples_leaf,
            max_features=0.9, random_state=args.random_state, n_jobs=-1,
        ),
        "HistGradientBoosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=args.min_samples_leaf, l2_regularization=1.0,
                random_state=args.random_state,
            ),
            n_jobs=-1,
        ),
        "MeanBaseline": DummyRegressor(strategy="mean"),
    }
    rows = []
    for name, model in candidates.items():
        lo, hi = target_clip_bounds(targets[fit_ids])
        model.fit(features.iloc[fit_ids][FEATURE_NAMES],
                  np.clip(targets[fit_ids], lo, hi))
        pred = model.predict(features.iloc[select_ids][FEATURE_NAMES])
        rows.append({
            "model": name,
            "selection_mae": float(mean_absolute_error(targets[select_ids], pred)),
            "selection_rmse": float(np.sqrt(mean_squared_error(targets[select_ids], pred))),
        })
    selection = pd.DataFrame(rows).sort_values(
        ["selection_mae", "selection_rmse"]
    ).reset_index(drop=True)
    name = str(selection.iloc[0]["model"])
    model = candidates[name]
    lo, hi = target_clip_bounds(targets)
    model.fit(features[FEATURE_NAMES], np.clip(targets, lo, hi))
    return model, lo, hi, name, selection


def predict_relative_losses(model, features, lower, upper):
    pred = np.clip(model.predict(features[FEATURE_NAMES]), lower, upper)
    return np.column_stack([pred, np.zeros(len(pred), dtype=np.float64)])


def quality_prediction_diagnostics(actual_loss_matrix, predicted_delta_matrix, split_name):
    actual = make_relative_loss_targets(actual_loss_matrix)
    predicted = np.asarray(predicted_delta_matrix, dtype=np.float64)[:, :-1]
    rows = []
    for index, k in enumerate(K_VALUES[:-1]):
        y, p = actual[:, index], predicted[:, index]
        rows.append({
            "split": split_name, "target": "logloss_K_minus_logloss_Full",
            "K": str(k), "n_requests": len(y),
            "actual_mean_delta_logloss": float(y.mean()),
            "predicted_mean_delta_logloss": float(p.mean()),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
            "better_than_full_sign_accuracy": float(np.mean((y < 0) == (p < 0))),
        })
    return rows


def metric_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def metric_logloss(y_true, scores):
    return float(log_loss(
        y_true,
        np.clip(scores, EPS, 1.0 - EPS),
        labels=[0, 1],
    ))


def history_cost_fraction(histories, scale):
    """Normalize interaction counts by a fixed calibration-set mean length.

    The same scale must be reused at test/inference time so lambda retains
    exactly the same units after calibration.
    """
    lengths = np.asarray([len(history) for history in histories], dtype=np.float64)
    scale = max(float(scale), 1.0)
    costs = []
    for k in K_VALUES:
        if k == 0:
            used = np.zeros_like(lengths)
        elif k == "Full":
            used = lengths.copy()
        else:
            used = np.minimum(lengths, int(k))
        costs.append(used / scale)
    return np.column_stack(costs)


def choose_k(predicted_losses, cost_fraction, penalty):
    objective = predicted_losses + float(penalty) * cost_fraction
    return np.argmin(objective, axis=1)


def selected_scores(predictions, selected_indices):
    score_matrix = np.column_stack([predictions[k] for k in K_VALUES])
    return score_matrix[np.arange(len(selected_indices)), selected_indices]


def selected_interactions(histories, selected_indices):
    lengths = np.asarray([len(history) for history in histories], dtype=np.int64)
    used = np.zeros(len(lengths), dtype=np.int64)
    for index, k in enumerate(K_VALUES):
        mask = selected_indices == index
        if k == 0:
            used[mask] = 0
        elif k == "Full":
            used[mask] = lengths[mask]
        else:
            used[mask] = np.minimum(lengths[mask], int(k))
    return used


def evaluate_policy(y_true, predictions, histories, selected_indices, name):
    scores = selected_scores(predictions, selected_indices)
    used = selected_interactions(histories, selected_indices)
    full_total = int(sum(len(history) for history in histories))
    selected_total = int(used.sum())
    return {
        "policy": name,
        "auc": metric_auc(y_true, scores),
        "logloss": metric_logloss(y_true, scores),
        "mean_history_interactions": float(used.mean()),
        "total_history_interactions": selected_total,
        "history_reduction_vs_full": (
            1.0 - selected_total / max(full_total, 1)
        ),
    }, scores, used


def fixed_k_metrics(y_true, predictions, histories, k):
    index = K_VALUES.index(k)
    selected = np.full(len(y_true), index, dtype=np.int64)
    metrics, _, _ = evaluate_policy(y_true, predictions, histories, selected, f"K={k}")
    return metrics


def calibrate_penalty(
    y_true, predictions, predicted_relative_losses, histories,
    budget_weight, lambda_max, lambda_steps, cost_scale,
):
    """Minimize validation logloss + weighted normalized history cost."""
    cost = history_cost_fraction(histories, cost_scale)
    full_auc = metric_auc(y_true, predictions["Full"])
    candidates = []
    for penalty in np.linspace(0.0, lambda_max, lambda_steps):
        selected = choose_k(predicted_relative_losses, cost, penalty)
        metrics, _, _ = evaluate_policy(
            y_true, predictions, histories, selected, f"lambda={penalty:.6f}"
        )
        cost_fraction = float(np.mean(cost[np.arange(len(selected)), selected]))
        candidates.append({
            "lambda": float(penalty), "auc": metrics["auc"],
            "logloss": metrics["logloss"],
            "mean_history_cost_fraction": cost_fraction,
            "history_reduction_vs_full": metrics["history_reduction_vs_full"],
            "mean_history_interactions": metrics["mean_history_interactions"],
            "budget_weight": float(budget_weight),
            "calibration_utility": float(metrics["logloss"] + budget_weight * cost_fraction),
        })
    table = pd.DataFrame(candidates)
    best = table.sort_values(
        ["calibration_utility", "logloss", "mean_history_interactions"]
    ).iloc[0]
    return float(best["lambda"]), table, full_auc, "quality_budget_utility_minimum"


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.trees <= 0 or args.lambda_steps < 2:
        raise ValueError("batch-size/trees must be positive and lambda-steps >= 2")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must be between 0 and 1")
    if args.budget_weight < 0 or args.lambda_max < 0:
        raise ValueError("budget-weight and lambda-max must be non-negative")

    print("=" * 80)
    print("ADAPTIVE HISTORY SELECTOR V2 — QUALITY PREDICTION + COST POLICY")
    print("=" * 80)

    device = baseline.get_device()
    print(f"Device: {device}")
    vocab, vocab_map = baseline.load_video_vocab()
    vocab_size = len(vocab)

    print("\nLoading validation requests...")
    validation = baseline.KuaiRandDataset("validation", "Full", vocab_map)
    validation.split_name = "validation"
    val_y = np.asarray(validation.labels, dtype=np.int64)
    val_features = history_features(validation.history_ids)

    print("\nGenerating frozen-model validation predictions...")
    val_predictions = collect_predictions(
        validation, vocab_size, device, args.batch_size
    )
    val_loss_matrix = make_loss_matrix(val_y, val_predictions)

    indices = np.arange(len(val_y))
    fit_indices, calibration_indices = train_test_split(
        indices,
        test_size=args.calibration_fraction,
        random_state=args.random_state,
        stratify=val_y,
    )

    print(
        f"\nValidation split: predictor-fit={len(fit_indices)}, "
        f"policy-calibration={len(calibration_indices)}"
    )
    predictor, target_lower, target_upper, predictor_name, model_selection_table = (
        fit_quality_predictor(
            val_features.iloc[fit_indices].reset_index(drop=True),
            val_loss_matrix[fit_indices], val_y[fit_indices], args,
        )
    )
    print(f"Selected quality predictor: {predictor_name}")
    print(model_selection_table.to_string(index=False))
    calibration_loss_predictions = predict_relative_losses(
        predictor, val_features.iloc[calibration_indices], target_lower, target_upper
    )
    calibration_predictions = {
        k: val_predictions[k][calibration_indices] for k in K_VALUES
    }
    calibration_histories = [
        validation.history_ids[i] for i in calibration_indices
    ]
    calibration_cost_scale = max(
        float(np.mean([len(h) for h in calibration_histories])), 1.0
    )
    calibration_actual_losses = val_loss_matrix[calibration_indices]
    penalty, calibration_table, calibration_full_auc, calibration_status = (
        calibrate_penalty(
            val_y[calibration_indices],
            calibration_predictions,
            calibration_loss_predictions,
            calibration_histories,
            args.budget_weight,
            args.lambda_max,
            args.lambda_steps,
            calibration_cost_scale,
        )
    )
    print(f"Calibrated lambda: {penalty:.6f} ({calibration_status})")
    print(f"Calibration Full AUC (reference only): {calibration_full_auc:.6f}")

    print("\nLoading held-out test requests...")
    test = baseline.KuaiRandDataset("test", "Full", vocab_map)
    test.split_name = "test"
    test_y = np.asarray(test.labels, dtype=np.int64)
    test_features = history_features(test.history_ids)

    print("\nGenerating frozen-model test predictions...")
    test_predictions = collect_predictions(
        test, vocab_size, device, args.batch_size
    )
    test_predicted_losses = predict_relative_losses(
        predictor, test_features, target_lower, target_upper
    )
    test_actual_losses = make_loss_matrix(test_y, test_predictions)
    test_cost = history_cost_fraction(test.history_ids, calibration_cost_scale)
    test_selected = choose_k(test_predicted_losses, test_cost, penalty)

    rows = [
        fixed_k_metrics(test_y, test_predictions, test.history_ids, k)
        for k in K_VALUES
    ]
    adaptive_metrics, test_scores, test_used = evaluate_policy(
        test_y, test_predictions, test.history_ids, test_selected, "Adaptive-V2"
    )
    adaptive_metrics["selected_lambda"] = penalty
    adaptive_metrics["calibration_status"] = calibration_status
    adaptive_metrics["quality_predictor"] = predictor_name
    adaptive_metrics["budget_weight"] = args.budget_weight

    # Oracle is strictly post-hoc: it uses test labels only for diagnostics and
    # is never used to select the deployed/test policy.
    test_row_indices = np.arange(len(test_y))
    oracle_indices = np.argmin(test_actual_losses, axis=1)
    selected_actual_losses = test_actual_losses[test_row_indices, test_selected]
    oracle_actual_losses = test_actual_losses[test_row_indices, oracle_indices]
    actual_loss_regret = selected_actual_losses - oracle_actual_losses
    rows.append(adaptive_metrics)

    output_dir = ROOT / "data" / "processed" / "phase9_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "adaptive_v2_metrics.csv"
    decisions_path = output_dir / "adaptive_v2_test_decisions.parquet"
    calibration_path = output_dir / "adaptive_v2_lambda_calibration.csv"
    importance_path = output_dir / "adaptive_v2_feature_importance.csv"
    quality_diagnostics_path = output_dir / "adaptive_v2_quality_diagnostics.csv"
    model_selection_path = output_dir / "adaptive_v2_model_selection.csv"
    model_path = model_dir / "adaptive_history_selector_v2.joblib"

    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    calibration_table.to_csv(calibration_path, index=False)

    decisions = pd.DataFrame({
        "evaluation_id": test.evaluation_ids,
        "user_id": test.user_ids,
        "selected_K": [str(K_VALUES[i]) for i in test_selected],
        "available_history_length": [len(h) for h in test.history_ids],
        "selected_history_length": test_used,
        "long_view": test_y,
        "predicted_long_view_probability": test_scores,
        "selected_actual_logloss": selected_actual_losses,
        "oracle_best_K_posthoc": [str(K_VALUES[i]) for i in oracle_indices],
        "oracle_best_actual_logloss_posthoc": oracle_actual_losses,
        "actual_logloss_regret_posthoc": actual_loss_regret,
        "selected_is_actual_best_posthoc": test_selected == oracle_indices,
        "predicted_delta_logloss_K0_vs_Full": test_predicted_losses[:, 0],
        "predicted_delta_logloss_K1_vs_Full": test_predicted_losses[:, 1],
        "predicted_delta_logloss_K5_vs_Full": test_predicted_losses[:, 2],
        "predicted_delta_logloss_K10_vs_Full": test_predicted_losses[:, 3],
        "predicted_delta_logloss_K20_vs_Full": test_predicted_losses[:, 4],
        "predicted_delta_logloss_K50_vs_Full": test_predicted_losses[:, 5],
        "predicted_delta_logloss_K100_vs_Full": test_predicted_losses[:, 6],
        "predicted_delta_logloss_KFull_vs_Full": test_predicted_losses[:, 7],
        "probability_K0": test_predictions[0],
        "probability_K1": test_predictions[1],
        "probability_K5": test_predictions[5],
        "probability_K10": test_predictions[10],
        "probability_K20": test_predictions[20],
        "probability_K50": test_predictions[50],
        "probability_K100": test_predictions[100],
        "probability_KFull": test_predictions["Full"],
        "actual_logloss_K0": test_actual_losses[:, 0],
        "actual_logloss_K1": test_actual_losses[:, 1],
        "actual_logloss_K5": test_actual_losses[:, 2],
        "actual_logloss_K10": test_actual_losses[:, 3],
        "actual_logloss_K20": test_actual_losses[:, 4],
        "actual_logloss_K50": test_actual_losses[:, 5],
        "actual_logloss_K100": test_actual_losses[:, 6],
        "actual_logloss_KFull": test_actual_losses[:, 7],
    })
    for column in FEATURE_NAMES:
        decisions[column] = test_features[column].to_numpy()
    decisions.to_parquet(decisions_path, index=False)

    quality_diagnostics = (
        quality_prediction_diagnostics(
            calibration_actual_losses,
            calibration_loss_predictions,
            "validation_calibration",
        )
        + quality_prediction_diagnostics(
            test_actual_losses,
            test_predicted_losses,
            "test_posthoc",
        )
    )
    pd.DataFrame(quality_diagnostics).to_csv(
        quality_diagnostics_path, index=False
    )
    model_selection_table.to_csv(model_selection_path, index=False)

    calibration_relative_targets = make_relative_loss_targets(
        calibration_actual_losses
    )
    permutation = permutation_importance(
        predictor,
        val_features.iloc[calibration_indices][FEATURE_NAMES],
        calibration_relative_targets,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=args.random_state,
        n_jobs=-1,
    )
    pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": permutation.importances_mean,
        "importance_std": permutation.importances_std,
    }).sort_values("importance", ascending=False).to_csv(
        importance_path, index=False
    )

    joblib.dump({
        "predictor": predictor,
        "predictor_name": predictor_name,
        "feature_names": FEATURE_NAMES,
        "k_values": K_VALUES,
        "lambda": penalty,
        "budget_weight": args.budget_weight,
        "target_clip_lower": target_lower,
        "target_clip_upper": target_upper,
        "cost_normalization_scale": calibration_cost_scale,
        "calibration_status": calibration_status,
        "calibration_full_auc_reference": calibration_full_auc,
        "cost_definition": "selected interactions normalized by fixed mean Full-history length from validation calibration",
        "target_definition": "per-request log-loss delta for each truncated K relative to Full; Full is exact zero reference",
    }, model_path)

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSelected K distribution:")
    print(decisions["selected_K"].value_counts().sort_index().to_string())
    print("\nSaved artifacts:")
    print(f"  Model: {model_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Decisions: {decisions_path}")
    print(f"  Lambda calibration: {calibration_path}")
    print(f"  Feature importance: {importance_path}")
    print(f"  Quality diagnostics: {quality_diagnostics_path}")
    print(f"  Model selection: {model_selection_path}")
    print(
        "\nProtocol: predictor-fit and lambda calibration use disjoint "
        "validation subsets; test is held out for final evaluation."
    )
    print(
        "Post-hoc diagnostic caveat: oracle-best K and actual per-request "
        "log-loss use test labels only after policy selection; they are not "
        "available to the deployed selector."
    )
    print(
        "Calibration minimizes validation logloss plus budget-weight times "
        "normalized mean history cost; AUC is reported, not a hard gate."
    )
    print(
        "Cost caveat: history-interaction reduction is a budget proxy, not "
        "measured end-to-end latency, monetary cost, energy, or serverless cost."
    )


if __name__ == "__main__":
    main()
