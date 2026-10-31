"""Adaptive History Selector V2: predict per-K quality, then allocate budget.

Protocol
--------
1. Frozen K-specific recommenders produce validation/test probabilities.
2. Validation per-request log-loss is the supervised target for a multi-output
   quality predictor.
3. Validation is split into predictor-fit and policy-calibration subsets.
4. A scalar cost penalty is calibrated on the held-out calibration subset:
       argmin_k predicted_logloss[k] + lambda * history_fraction[k]
   Choose the lambda with the lowest history budget while keeping calibration
   AUC within --max-auc-drop of the Full model.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import log_loss, roc_auc_score
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
        "--max-auc-drop",
        type=float,
        default=0.01,
        help="Maximum allowed calibration AUC drop from the Full model.",
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


def history_cost_fraction(histories):
    lengths = np.asarray([len(history) for history in histories], dtype=np.float64)
    costs = []
    for k in K_VALUES:
        if k == 0:
            used = np.zeros_like(lengths)
        elif k == "Full":
            used = lengths.copy()
        else:
            used = np.minimum(lengths, int(k))
        costs.append(np.divide(
            used, lengths, out=np.zeros_like(used), where=lengths > 0
        ))
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
    y_true,
    predictions,
    predicted_losses,
    histories,
    max_auc_drop,
    lambda_max,
    lambda_steps,
):
    cost = history_cost_fraction(histories)
    full_auc = metric_auc(y_true, predictions["Full"])
    candidates = []

    for penalty in np.linspace(0.0, lambda_max, lambda_steps):
        selected = choose_k(predicted_losses, cost, penalty)
        metrics, _, _ = evaluate_policy(
            y_true, predictions, histories, selected, f"lambda={penalty:.6f}"
        )
        feasible = metrics["auc"] >= full_auc - max_auc_drop
        candidates.append({
            "lambda": float(penalty),
            "auc": metrics["auc"],
            "logloss": metrics["logloss"],
            "history_reduction_vs_full": metrics["history_reduction_vs_full"],
            "mean_history_interactions": metrics["mean_history_interactions"],
            "feasible": bool(feasible),
        })

    table = pd.DataFrame(candidates)
    feasible_table = table[table["feasible"]]
    if not feasible_table.empty:
        best = feasible_table.sort_values(
            ["mean_history_interactions", "auc", "lambda"],
            ascending=[True, False, True],
        ).iloc[0]
        status = "auc_constraint_satisfied"
    else:
        best = table.sort_values(
            ["auc", "mean_history_interactions"],
            ascending=[False, True],
        ).iloc[0]
        status = "no_lambda_met_auc_constraint"

    return float(best["lambda"]), table, full_auc, status


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.trees <= 0 or args.lambda_steps < 2:
        raise ValueError("batch-size/trees must be positive and lambda-steps >= 2")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must be between 0 and 1")
    if args.max_auc_drop < 0 or args.lambda_max < 0:
        raise ValueError("max-auc-drop and lambda-max must be non-negative")

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
    predictor = ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=0.9,
        random_state=args.random_state,
        n_jobs=-1,
    )
    predictor.fit(
        val_features.iloc[fit_indices][FEATURE_NAMES],
        val_loss_matrix[fit_indices],
    )

    calibration_loss_predictions = predictor.predict(
        val_features.iloc[calibration_indices][FEATURE_NAMES]
    )
    calibration_predictions = {
        k: val_predictions[k][calibration_indices] for k in K_VALUES
    }
    calibration_histories = [
        validation.history_ids[i] for i in calibration_indices
    ]
    penalty, calibration_table, calibration_full_auc, calibration_status = (
        calibrate_penalty(
            val_y[calibration_indices],
            calibration_predictions,
            calibration_loss_predictions,
            calibration_histories,
            args.max_auc_drop,
            args.lambda_max,
            args.lambda_steps,
        )
    )
    print(f"Calibrated lambda: {penalty:.6f} ({calibration_status})")
    print(f"Calibration Full AUC: {calibration_full_auc:.6f}")

    print("\nLoading held-out test requests...")
    test = baseline.KuaiRandDataset("test", "Full", vocab_map)
    test.split_name = "test"
    test_y = np.asarray(test.labels, dtype=np.int64)
    test_features = history_features(test.history_ids)

    print("\nGenerating frozen-model test predictions...")
    test_predictions = collect_predictions(
        test, vocab_size, device, args.batch_size
    )
    test_predicted_losses = predictor.predict(test_features[FEATURE_NAMES])
    test_cost = history_cost_fraction(test.history_ids)
    test_selected = choose_k(test_predicted_losses, test_cost, penalty)

    rows = [
        fixed_k_metrics(test_y, test_predictions, test.history_ids, k)
        for k in K_VALUES
    ]
    adaptive_metrics, test_scores, test_used = evaluate_policy(
        test_y, test_predictions, test.history_ids, test_selected, "Adaptive-V2"
    )
    rows.append(adaptive_metrics)

    output_dir = ROOT / "data" / "processed" / "phase9_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "adaptive_v2_metrics.csv"
    decisions_path = output_dir / "adaptive_v2_test_decisions.parquet"
    calibration_path = output_dir / "adaptive_v2_lambda_calibration.csv"
    importance_path = output_dir / "adaptive_v2_feature_importance.csv"
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
        "predicted_loss_K0": test_predicted_losses[:, 0],
        "predicted_loss_K1": test_predicted_losses[:, 1],
        "predicted_loss_K5": test_predicted_losses[:, 2],
        "predicted_loss_K10": test_predicted_losses[:, 3],
        "predicted_loss_K20": test_predicted_losses[:, 4],
        "predicted_loss_K50": test_predicted_losses[:, 5],
        "predicted_loss_K100": test_predicted_losses[:, 6],
        "predicted_loss_KFull": test_predicted_losses[:, 7],
    })
    for column in FEATURE_NAMES:
        decisions[column] = test_features[column].to_numpy()
    decisions.to_parquet(decisions_path, index=False)

    pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": predictor.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        importance_path, index=False
    )

    joblib.dump({
        "predictor": predictor,
        "feature_names": FEATURE_NAMES,
        "k_values": K_VALUES,
        "lambda": penalty,
        "max_auc_drop": args.max_auc_drop,
        "calibration_status": calibration_status,
        "calibration_full_auc": calibration_full_auc,
        "cost_definition": "selected history interactions / available history length",
        "target_definition": "per-request binary log-loss for each frozen K model",
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
    print(
        "\nProtocol: predictor-fit and lambda calibration use disjoint "
        "validation subsets; test is held out for final evaluation."
    )
    print(
        "Cost caveat: history-interaction reduction is a budget proxy, not "
        "measured end-to-end latency, monetary cost, energy, or serverless cost."
    )


if __name__ == "__main__":
    main()
