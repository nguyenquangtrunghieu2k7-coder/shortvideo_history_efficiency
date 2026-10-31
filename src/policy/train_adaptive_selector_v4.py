"""Train and evaluate Adaptive History Selector V4.

V4 selects the minimum-history candidate among fixed-K and learned-lambda
policies that satisfy the validation AUC-drop constraint. Higher AUC breaks
ties in history cost.

Request-level AUC-contribution uplift is a supervised proxy. The policy is selected on a validation tuning split and evaluated once on the held-out test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "policy"))

import train_baseline as baseline  # noqa: E402
from adaptive_v4_features import FEATURE_NAMES, history_features  # noqa: E402
from adaptive_v4_policy import (  # noqa: E402
    K_VALUES,
    TRUNCATED_K_VALUES,
    apply_selected_policy,
    auc_contribution_uplift_targets,
    calibrate_balanced_policy,
    evaluate_actions,
    fit_auc_uplift_predictor,
    predict_auc_uplift,
    score_matrix,
    stratified_group_split,
    stratified_paired_auc_difference_ci,
)


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive History Selector V4")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--min-samples-leaf", type=int, default=25)
    p.add_argument("--max-auc-drop", type=float, default=0.03,
                   help="Maximum allowed global AUC drop vs Full.")
    p.add_argument("--lambda-steps", type=int, default=41)
    p.add_argument("--quality-weight", type=float, default=0.6,
                   help="Weight for normalized quality retention in balanced utility.")
    p.add_argument("--history-cost-weight", type=float, default=0.4,
                   help="Weight for history saving in balanced utility.")
    p.add_argument("--bootstrap-repeats", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument(
        "--audit-splits-only", action="store_true",
        help="Audit user overlap in the group-disjoint validation splits and exit before model inference.",
    )
    return p.parse_args()


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
    path = baseline.MODEL_DIR / baseline.model_filename("full", k)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen recommender checkpoint: {path}. "
            "Train/freeze Phase 8 models first."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = baseline.build_model(vocab_size, mode="full").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_for_k(dataset, k, vocab_size, device, batch_size):
    model = load_frozen_model(k, vocab_size, device)
    output = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            candidate, history, mask, _, _, _ = make_batch(dataset, start, end, k)
            scores = model(candidate.to(device), history.to(device), mask.to(device))
            output.extend(scores.detach().cpu().numpy().tolist())
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(output, dtype=np.float64)


def collect_predictions(dataset, vocab_size, device, batch_size):
    predictions = {}
    for k in K_VALUES:
        print(f"  Predicting {getattr(dataset, 'split_name', 'split')} K={k} ...", flush=True)
        predictions[k] = predict_for_k(dataset, k, vocab_size, device, batch_size)
    return predictions


def report_split_overlap(name, left_groups, right_groups, left_y, right_y):
    """Report same-user overlap across the two sides of a validation split."""
    left_groups = np.asarray(left_groups)
    right_groups = np.asarray(right_groups)
    overlap = np.intersect1d(np.unique(left_groups), np.unique(right_groups))
    print(
        f"{name}: left_rows={len(left_groups)}, right_rows={len(right_groups)}, "
        f"left_users={len(np.unique(left_groups))}, right_users={len(np.unique(right_groups))}, "
        f"overlap_users={len(overlap)}"
    )
    print(
        f"  positive_rate: left={np.mean(left_y):.4f}, "
        f"right={np.mean(right_y):.4f}"
    )


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.trees <= 0 or args.lambda_steps < 2:
        raise ValueError("batch-size/trees must be positive and lambda-steps >= 2")
    if args.max_auc_drop < 0 or args.bootstrap_repeats < 20:
        raise ValueError("max-auc-drop must be non-negative and bootstrap-repeats >= 20")
    if args.quality_weight <= 0 or args.history_cost_weight <= 0 or not np.isclose(args.quality_weight + args.history_cost_weight, 1.0):
        raise ValueError("quality-weight and history-cost-weight must be positive and sum to 1.")

    print("=" * 80)
    print("ADAPTIVE HISTORY SELECTOR V4 — AUC-CONSTRAINED HISTORY MINIMIZATION")
    print("=" * 80)
    device = baseline.get_device()
    print(f"Device: {device}")
    vocab, vocab_map = baseline.load_video_vocab()
    vocab_size = len(vocab)

    print("\nLoading validation requests...")
    validation = baseline.KuaiRandDataset("validation", "Full", vocab_map)
    validation.split_name = "validation"
    val_y = np.asarray(validation.labels, dtype=np.int8)
    val_features = history_features(validation.history_ids)

    # Keep users disjoint between predictor development and policy tuning.
    development_idx, tuning_idx = stratified_group_split(
        val_y, validation.user_ids, n_splits=2,
        random_state=args.random_state,
    )
    development_groups = np.asarray(validation.user_ids)[development_idx]
    tuning_groups = np.asarray(validation.user_ids)[tuning_idx]
    if args.audit_splits_only:
        print("\n=== USER-OVERLAP AUDIT (NO MODEL INFERENCE) ===")
        report_split_overlap(
            "Development vs policy-tuning",
            development_groups, tuning_groups,
            val_y[development_idx], val_y[tuning_idx],
        )
        inner_fit_idx, inner_select_idx = stratified_group_split(
            val_y[development_idx], development_groups,
            n_splits=5, random_state=args.random_state + 11,
        )
        report_split_overlap(
            "Uplift-model fit vs model-selection",
            development_groups[inner_fit_idx],
            development_groups[inner_select_idx],
            val_y[development_idx][inner_fit_idx],
            val_y[development_idx][inner_select_idx],
        )
        print("Audit only: no recommender inference, predictor fitting, or test evaluation was run.")
        return

    print("\nGenerating validation predictions...")
    val_predictions = collect_predictions(
        validation, vocab_size, device, args.batch_size
    )

    # Use user-disjoint validation groups for predictor development and
    # policy tuning. The held-out test split is evaluated once after freezing.
    development_features = val_features.iloc[development_idx].reset_index(drop=True)
    development_predictions = {k: val_predictions[k][development_idx] for k in K_VALUES}
    development_y = val_y[development_idx]

    print(
        f"\nValidation roles: development={len(development_idx)}, "
        f"policy-tuning={len(tuning_idx)}"
    )
    model, model_name, model_table = fit_auc_uplift_predictor(
        development_features, development_y, development_predictions,
        development_groups, args, FEATURE_NAMES
    )
    print(f"Selected uplift model: {model_name}")
    print(model_table.to_string(index=False))

    tuning_features = val_features.iloc[tuning_idx].reset_index(drop=True)
    tuning_predictions = {k: val_predictions[k][tuning_idx] for k in K_VALUES}
    tuning_histories = [validation.history_ids[i] for i in tuning_idx]
    tuning_uplift = predict_auc_uplift(model, tuning_features, FEATURE_NAMES)
    selected, policy_table, cost_scale, tuning_full_auc = calibrate_balanced_policy(
        val_y[tuning_idx], tuning_predictions, tuning_histories, tuning_uplift,
        max_auc_drop=args.max_auc_drop,
        lambda_steps=args.lambda_steps,
        quality_weight=args.quality_weight,
        history_cost_weight=args.history_cost_weight,
    )
    selected["validation_full_auc"] = tuning_full_auc
    selected["validation_auc_drop_tolerance"] = args.max_auc_drop
    print("\nSelected policy from validation tuning:")
    print(json.dumps(selected, indent=2, default=str))

    print("\nLoading held-out test requests...")
    test = baseline.KuaiRandDataset("test", "Full", vocab_map)
    test.split_name = "test"
    test_y = np.asarray(test.labels, dtype=np.int8)
    test_features = history_features(test.history_ids)

    print("\nGenerating held-out test predictions...")
    test_predictions = collect_predictions(test, vocab_size, device, args.batch_size)
    test_actions = apply_selected_policy(
        selected, model, test_features, test.history_ids,
        cost_scale, FEATURE_NAMES,
    )

    result_rows = []
    for k in K_VALUES:
        actions = np.full(len(test_y), K_VALUES.index(k), dtype=int)
        metrics, _ = evaluate_actions(test_y, test_predictions, test.history_ids, actions, f"Fixed-K-{k}")
        result_rows.append(metrics)
    adaptive_metrics, test_scores = evaluate_actions(
        test_y, test_predictions, test.history_ids, test_actions, "Adaptive-V4"
    )
    adaptive_metrics.update({
        "deployment_mode": selected["deployment_mode"],
        "selected_policy": selected["policy"],
        "max_auc_drop_validation": args.max_auc_drop,
        "quality_predictor": model_name,
        "validation_full_auc": tuning_full_auc,
        "validation_auc_drop_tolerance": args.max_auc_drop,
        "selection_objective": selected["selection_objective"],
        "quality_weight": args.quality_weight,
        "history_cost_weight": args.history_cost_weight,
        "validation_quality_retention": selected["quality_retention"],
        "validation_history_saving": selected["history_saving"],
        "validation_balanced_utility": selected["balanced_utility"],
        "validation_selected_policy_auc": selected["auc"],
        "validation_selected_policy_history_cost_ratio": selected["history_cost_ratio"],
    })
    result_rows.append(adaptive_metrics)

    # Test bootstrap is post-hoc inference only; it never changes the selected policy.
    test_ci_low, test_ci_high = stratified_paired_auc_difference_ci(
        test_y, test_scores, test_predictions["Full"],
        repeats=args.bootstrap_repeats,
        random_state=args.bootstrap_seed + 9999,
    )
    adaptive_metrics["test_auc_delta_ci_low_90"] = test_ci_low
    adaptive_metrics["test_auc_delta_ci_high_90"] = test_ci_high

    out = ROOT / "data" / "processed" / "phase9_v4"
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    metrics_path = out / "adaptive_v4_metrics.csv"
    policy_path = out / "adaptive_v4_validation_policy_frontier.csv"
    decisions_path = out / "adaptive_v4_test_decisions.parquet"
    model_path = ROOT / "models" / "adaptive_history_selector_v4.joblib"

    pd.DataFrame(result_rows).to_csv(metrics_path, index=False)
    policy_table.to_csv(policy_path, index=False)
    decisions = pd.DataFrame({
        "evaluation_id": test.evaluation_ids,
        "user_id": test.user_ids,
        "selected_K": [str(K_VALUES[i]) for i in test_actions],
        "selected_policy": selected["policy"],
        "deployment_mode": selected["deployment_mode"],
        "available_history_length": [len(h) for h in test.history_ids],
        "selected_history_length": [
            0 if K_VALUES[i] == 0 else
            (len(test.history_ids[j]) if K_VALUES[i] == "Full" else min(len(test.history_ids[j]), int(K_VALUES[i])))
            for j, i in enumerate(test_actions)
        ],
        "long_view": test_y,
        "predicted_long_view_probability": test_scores,
    })
    for k in K_VALUES:
        decisions[f"probability_K{k}"] = test_predictions[k]
    for feature in FEATURE_NAMES:
        decisions[feature] = test_features[feature].to_numpy()
    decisions.to_parquet(decisions_path, index=False)

    joblib.dump({
        "model": model,
        "model_name": model_name,
        "feature_names": FEATURE_NAMES,
        "k_values": K_VALUES,
        "selected_validation_policy": selected,
        "policy_protocol": "filter_by_validation_auc_drop_constraint_then_maximize_weighted_geometric_quality_history_utility; evaluate_once_on_held_out_test",
        "selection_objective": "maximize(Q**quality_weight * S**history_cost_weight) among validation-feasible candidates",
        "quality_weight": args.quality_weight,
        "history_cost_weight": args.history_cost_weight,
        "policy_frozen_after_validation_tuning": True,
        "cost_normalization_scale": cost_scale,
        "max_auc_drop": args.max_auc_drop,
        "target_definition": "per-request AUC-contribution uplift vs Full reference scores; proxy target only",
        "policy_constraint": "point-estimate validation AUC must be within max_auc_drop of Full; among feasible candidates maximize weighted geometric utility; Full remains an eligible fallback",
    }, model_path)

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(pd.DataFrame(result_rows).to_string(index=False))
    print(f"\nTest paired-bootstrap 90% CI for AUC delta vs Full: [{test_ci_low:.6f}, {test_ci_high:.6f}]")
    print("\nSelected K distribution:")
    print(decisions["selected_K"].value_counts().sort_index().to_string())
    print("\nSaved artifacts:")
    print(f"  Model: {model_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Validation frontier: {policy_path}")
    print(f"  Test decisions: {decisions_path}")
    print(f"\nBalanced objective: quality_weight={args.quality_weight:.2f}, history_cost_weight={args.history_cost_weight:.2f}")
    print("Interpretation: policy is selected on validation only. Test bootstrap is post-hoc inference.")
    print("History interactions are a computational-work proxy, not measured cloud cost, latency, or energy.")
    print("This remains an offline frozen-checkpoint experiment, not online Bloom-FaaS serving.")


if __name__ == "__main__":
    main()
