"""Train, calibrate, and evaluate Adaptive History Selector V3.

V3 targets the final objective directly:
  minimize history interactions subject to a validation-calibrated global AUC
  non-inferiority constraint against the Full-history reference.

Request-level AUC-contribution uplift is only a supervised proxy used to learn
which actions may help. The actual policy is selected using global validation
AUC and a paired, class-stratified bootstrap lower confidence bound.
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
from adaptive_v3_features import FEATURE_NAMES, history_features  # noqa: E402
from adaptive_v3_policy import (  # noqa: E402
    K_VALUES,
    TRUNCATED_K_VALUES,
    apply_selected_policy,
    auc_contribution_uplift_targets,
    calibrate_policy,
    evaluate_actions,
    fit_auc_uplift_predictor,
    predict_auc_uplift,
    score_matrix,
    stratified_paired_auc_difference_ci,
)


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive History Selector V3")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--min-samples-leaf", type=int, default=25)
    p.add_argument("--max-auc-drop", type=float, default=0.01,
                   help="Maximum allowed global AUC drop vs Full.")
    p.add_argument("--lambda-steps", type=int, default=41)
    p.add_argument("--bootstrap-repeats", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
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


def make_simple_heuristic_candidates(features, histories):
    """Transparent, request-time baselines calibrated on validation only."""
    from adaptive_v3_policy import actions_for_fixed_k, interaction_matrix
    n = len(features)
    candidates = []
    lengths = np.asarray([len(h) for h in histories])
    # History-length heuristic: shorter available histories are capped less.
    for cutoff in [20, 50, 100, 200, 400]:
        for low_k, high_k in [(5, 20), (10, 50), (20, 100)]:
            actions = np.where(
                lengths <= cutoff,
                K_VALUES.index(low_k),
                K_VALUES.index(high_k),
            ).astype(int)
            candidates.append((f"heuristic-length-{cutoff}-{low_k}-{high_k}", actions))
    # Repetition/diversity heuristic: choose more history for diverse histories.
    if "recent_unique_ratio_100" in features:
        values = features["recent_unique_ratio_100"].to_numpy(dtype=float)
        thresholds = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
        for threshold in thresholds:
            for low_k, high_k in [(10, 50), (20, 100), (50, "Full")]:
                actions = np.where(
                    values <= threshold,
                    K_VALUES.index(low_k),
                    K_VALUES.index(high_k),
                ).astype(int)
                candidates.append((
                    f"heuristic-diversity-{threshold:.5f}-{low_k}-{high_k}",
                    actions,
                ))
    return candidates


def calibrate_with_heuristics(y, predictions, histories, uplift, features, args):
    """Include transparent heuristics in the same validation feasibility test."""
    from adaptive_v3_policy import (
        actions_for_fixed_k,
        evaluate_actions,
        stratified_paired_auc_difference_ci,
    )
    selected, table, cost_scale, full_auc = calibrate_policy(
        y, predictions, histories, uplift,
        max_auc_drop=args.max_auc_drop,
        lambda_steps=args.lambda_steps,
        bootstrap_repeats=args.bootstrap_repeats,
        random_state=args.bootstrap_seed,
    )
    heuristic_rows = []
    full_scores = np.asarray(predictions["Full"], dtype=float)
    for name, actions in make_simple_heuristic_candidates(features, histories):
        metrics, scores = evaluate_actions(y, predictions, histories, actions, name)
        low, high = stratified_paired_auc_difference_ci(
            y, scores, full_scores,
            repeats=args.bootstrap_repeats,
            random_state=args.bootstrap_seed + len(heuristic_rows) + 1000,
        )
        heuristic_rows.append({
            "policy": name,
            "policy_type": "heuristic",
            "penalty": np.nan,
            "fixed_K": np.nan,
            "auc": metrics["auc"],
            "auc_delta_vs_full": metrics["auc"] - full_auc,
            "auc_delta_ci_low_90": low,
            "auc_delta_ci_high_90": high,
            "history_cost_ratio": metrics["history_cost_ratio"],
            "history_reduction_vs_full": metrics["history_reduction_vs_full"],
            "mean_history_interactions": metrics["mean_history_interactions"],
            # Candidate screening uses point-estimate feasibility. The
            # independent certification split is the sole CI-based gate.
            "feasible": bool((metrics["auc"] - full_auc) >= -args.max_auc_drop),
            "ci_feasible_on_tuning": bool(low >= -args.max_auc_drop),
            "selected_for_test": False,
        })
    if heuristic_rows:
        heuristic_table = pd.DataFrame(heuristic_rows)
        table = pd.concat([table, heuristic_table], ignore_index=True)
        feasible_heuristics = heuristic_table[heuristic_table["feasible"]]
        if not feasible_heuristics.empty:
            best_heuristic = feasible_heuristics.sort_values(
                ["history_cost_ratio", "auc"], ascending=[True, False]
            ).iloc[0]
            if float(best_heuristic["history_cost_ratio"]) < float(selected["history_cost_ratio"]) - 1e-9:
                selected = best_heuristic.to_dict()
                selected["deployment_mode"] = "heuristic"
                selected["heuristic_name"] = selected["policy"]
    table["selected_for_test"] = table["policy"].eq(selected["policy"])
    return selected, table, cost_scale, full_auc


def apply_heuristic(name, features, histories):
    from adaptive_v3_policy import K_VALUES
    parts = name.split("-")
    if parts[1] == "length":
        cutoff, low_k, high_k = int(parts[2]), int(parts[3]), parts[4]
        high_k = "Full" if high_k == "Full" else int(high_k)
        values = np.asarray([len(h) for h in histories])
    else:
        threshold, low_k, high_k = float(parts[2]), int(parts[3]), parts[4]
        high_k = "Full" if high_k == "Full" else int(high_k)
        values = features["recent_unique_ratio_100"].to_numpy(dtype=float)
        cutoff = threshold
    return np.where(
        values <= cutoff,
        K_VALUES.index(low_k),
        K_VALUES.index(high_k),
    ).astype(int)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.trees <= 0 or args.lambda_steps < 2:
        raise ValueError("batch-size/trees must be positive and lambda-steps >= 2")
    if args.max_auc_drop < 0 or args.bootstrap_repeats < 20:
        raise ValueError("max-auc-drop must be non-negative and bootstrap-repeats >= 20")

    print("=" * 80)
    print("ADAPTIVE HISTORY SELECTOR V3 — AUC-CONSTRAINED HISTORY MINIMIZATION")
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

    print("\nGenerating validation predictions...")
    val_predictions = collect_predictions(
        validation, vocab_size, device, args.batch_size
    )

    # Three disjoint validation roles: development, policy tuning, and
    # independent certification. This avoids claiming non-inferiority on the
    # same rows used to choose among many candidate policies.
    from sklearn.model_selection import train_test_split
    development_idx, remainder_idx = train_test_split(
        np.arange(len(val_y)), test_size=0.50,
        random_state=args.random_state, stratify=val_y,
    )
    tuning_idx, certification_idx = train_test_split(
        remainder_idx, test_size=0.50,
        random_state=args.random_state + 1, stratify=val_y[remainder_idx],
    )
    development_features = val_features.iloc[development_idx].reset_index(drop=True)
    development_predictions = {k: val_predictions[k][development_idx] for k in K_VALUES}
    development_y = val_y[development_idx]

    print(
        f"\nValidation roles: development={len(development_idx)}, "
        f"policy-tuning={len(tuning_idx)}, independent-certification={len(certification_idx)}"
    )
    model, model_name, model_table = fit_auc_uplift_predictor(
        development_features, development_y, development_predictions, args, FEATURE_NAMES
    )
    print(f"Selected uplift model: {model_name}")
    print(model_table.to_string(index=False))

    tuning_features = val_features.iloc[tuning_idx].reset_index(drop=True)
    tuning_predictions = {k: val_predictions[k][tuning_idx] for k in K_VALUES}
    tuning_histories = [validation.history_ids[i] for i in tuning_idx]
    tuning_uplift = predict_auc_uplift(model, tuning_features, FEATURE_NAMES)
    selected, policy_table, cost_scale, tuning_full_auc = calibrate_with_heuristics(
        val_y[tuning_idx], tuning_predictions, tuning_histories,
        tuning_uplift, tuning_features, args,
    )

    # Certify the selected policy exactly once on untouched validation rows.
    cert_features = val_features.iloc[certification_idx].reset_index(drop=True)
    cert_predictions = {k: val_predictions[k][certification_idx] for k in K_VALUES}
    cert_histories = [validation.history_ids[i] for i in certification_idx]
    if selected["deployment_mode"] == "heuristic":
        cert_actions = apply_heuristic(selected["heuristic_name"], cert_features, cert_histories)
    elif selected["deployment_mode"] == "fixed_k_fallback":
        fixed = selected["fixed_K"]
        fixed = "Full" if str(fixed) == "Full" else int(fixed)
        cert_actions = np.full(len(certification_idx), K_VALUES.index(fixed), dtype=int)
    else:
        cert_actions = apply_selected_policy(
            selected, model, cert_features, cert_histories, cost_scale, FEATURE_NAMES
        )
    cert_metrics, cert_scores = evaluate_actions(
        val_y[certification_idx], cert_predictions, cert_histories,
        cert_actions, "V3-certification",
    )
    cert_low, cert_high = stratified_paired_auc_difference_ci(
        val_y[certification_idx], cert_scores, cert_predictions["Full"],
        repeats=args.bootstrap_repeats, random_state=args.bootstrap_seed + 5000,
    )
    candidate_policy_name = selected["policy"]

    # Persist the pre-fallback candidate decisions on certification rows.
    # This is an exploratory diagnostic artifact: certification has already
    # been used for the V3 gate and must not be reused as an untouched holdout.
    diagnostic_dir = ROOT / "data" / "processed" / "phase9_v3"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    candidate_decisions = pd.DataFrame({
        "evaluation_id": [validation.evaluation_ids[i] for i in certification_idx],
        "user_id": [validation.user_ids[i] for i in certification_idx],
        "candidate_policy": candidate_policy_name,
        "candidate_selected_K": [str(K_VALUES[i]) for i in cert_actions],
        "available_history_length": [len(h) for h in cert_histories],
        "candidate_selected_history_length": [
            0 if K_VALUES[action] == 0 else
            (len(cert_histories[row]) if K_VALUES[action] == "Full" else
             min(len(cert_histories[row]), int(K_VALUES[action])))
            for row, action in enumerate(cert_actions)
        ],
        "long_view": val_y[certification_idx],
        "candidate_probability": cert_scores,
        "full_probability": cert_predictions["Full"],
    })
    for k in K_VALUES:
        candidate_decisions[f"probability_K{k}"] = cert_predictions[k]
    for feature in FEATURE_NAMES:
        candidate_decisions[feature] = cert_features[feature].to_numpy()
    candidate_decisions_path = diagnostic_dir / "adaptive_v3_certification_candidate_decisions.parquet"
    candidate_decisions.to_parquet(candidate_decisions_path, index=False)
    print(f"Saved pre-fallback candidate diagnostics: {candidate_decisions_path}")

    candidate_certification_passed = bool(cert_low >= -args.max_auc_drop)
    certification_fallback = not candidate_certification_passed
    if certification_fallback:
        selected = {
            "policy": "Fixed-K-Full",
            "deployment_mode": "fixed_k_fallback",
            "fixed_K": "Full",
            "history_cost_ratio": 1.0,
            "certification_fallback": True,
        }
    selected["candidate_policy_before_certification"] = candidate_policy_name
    selected["candidate_certification_passed"] = candidate_certification_passed
    selected["certification_fallback_to_full"] = certification_fallback
    selected["candidate_certification_auc"] = cert_metrics["auc"]
    selected["candidate_certification_auc_delta_ci_low_90"] = cert_low
    selected["candidate_certification_auc_delta_ci_high_90"] = cert_high
    policy_table["candidate_selected_before_certification"] = policy_table["policy"].eq(candidate_policy_name)
    policy_table["selected_for_test"] = policy_table["policy"].eq(selected["policy"])
    print("\nSelected policy after independent certification:")
    print(json.dumps(selected, indent=2, default=str))
    print(
        f"Certification 90% CI for AUC delta: [{cert_low:.6f}, {cert_high:.6f}] "
        f"— {'PASS' if candidate_certification_passed else 'FAIL; Full fallback'}"
    )

    print("\nLoading held-out test requests...")
    test = baseline.KuaiRandDataset("test", "Full", vocab_map)
    test.split_name = "test"
    test_y = np.asarray(test.labels, dtype=np.int8)
    test_features = history_features(test.history_ids)

    print("\nGenerating held-out test predictions...")
    test_predictions = collect_predictions(test, vocab_size, device, args.batch_size)
    if selected["deployment_mode"] == "heuristic":
        test_actions = apply_heuristic(
            selected["heuristic_name"], test_features, test.history_ids
        )
    elif selected["deployment_mode"] == "fixed_k_fallback":
        fixed = selected["fixed_K"]
        fixed = "Full" if str(fixed) == "Full" else int(fixed)
        test_actions = np.full(len(test_y), K_VALUES.index(fixed), dtype=int)
    else:
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
        test_y, test_predictions, test.history_ids, test_actions, "Adaptive-V3"
    )
    adaptive_metrics.update({
        "deployment_mode": selected["deployment_mode"],
        "selected_policy": selected["policy"],
        "max_auc_drop_validation": args.max_auc_drop,
        "quality_predictor": model_name,
        "candidate_policy_before_certification": candidate_policy_name,
        "candidate_certification_passed": candidate_certification_passed,
        "certification_fallback_to_full": certification_fallback,
        "candidate_certification_auc": cert_metrics["auc"],
        "candidate_certification_auc_delta_ci_low_90": cert_low,
        "candidate_certification_auc_delta_ci_high_90": cert_high,
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

    out = ROOT / "data" / "processed" / "phase9_v3"
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    metrics_path = out / "adaptive_v3_metrics.csv"
    policy_path = out / "adaptive_v3_validation_policy_frontier.csv"
    decisions_path = out / "adaptive_v3_test_decisions.parquet"
    model_path = ROOT / "models" / "adaptive_history_selector_v3.joblib"

    pd.DataFrame(result_rows).to_csv(metrics_path, index=False)
    policy_table.to_csv(policy_path, index=False)
    decisions = pd.DataFrame({
        "evaluation_id": test.evaluation_ids,
        "user_id": test.user_ids,
        "selected_K": [str(K_VALUES[i]) for i in test_actions],
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
        "cost_normalization_scale": cost_scale,
        "max_auc_drop": args.max_auc_drop,
        "target_definition": "per-request AUC-contribution uplift vs Full reference scores; proxy target only",
        "policy_constraint": "minimum history interactions among validation policies whose paired-bootstrap 90% lower AUC-difference bound >= -max_auc_drop",
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
    print("\nInterpretation: policy is selected on validation only. Test bootstrap is post-hoc inference.")
    print("History interactions are a computational-work proxy, not measured cloud cost, latency, or energy.")
    print("This remains an offline frozen-checkpoint experiment, not online Bloom-FaaS serving.")


if __name__ == "__main__":
    main()
