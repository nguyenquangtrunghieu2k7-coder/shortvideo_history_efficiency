"""Evaluate a pre-selected V3 lambda policy once on the held-out test set.

This script reuses the saved V3 predictor and per-K test predictions. It does
not retrain the recommender or selector and does not change lambda based on test
results. The default non-inferiority margin is delta=0.03.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "policy"))

from adaptive_v3_policy import (  # noqa: E402
    K_VALUES,
    predict_auc_uplift,
    stratified_paired_auc_difference_ci,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-shot held-out test evaluation for a fixed V3 lambda."
    )
    parser.add_argument("--lambda-penalty", type=float, default=0.006235507341)
    parser.add_argument("--max-auc-drop", type=float, default=0.03)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "adaptive_history_selector_v3.joblib",
    )
    parser.add_argument(
        "--test-decisions",
        type=Path,
        default=ROOT / "data" / "processed" / "phase9_v3" / "adaptive_v3_test_decisions.parquet",
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=ROOT / "data" / "processed" / "phase9_v3" / "adaptive_v3_validation_policy_frontier.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase9_v3",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_auc_drop <= 0:
        raise ValueError("--max-auc-drop must be positive.")
    for path in (args.model, args.test_decisions, args.frontier):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    policy_name = f"V3-lambda-{args.lambda_penalty:.8g}"
    frontier = pd.read_csv(args.frontier)
    matching = frontier.loc[frontier["policy"].eq(policy_name)]
    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one validation-frontier row for {policy_name}; "
            f"found {len(matching)}. Check --lambda-penalty and the frontier file."
        )
    validation_row = matching.iloc[0]
    validation_ci_low = float(validation_row["auc_delta_ci_low_90"])
    validation_delta = float(validation_row["auc_delta_vs_full"])
    if validation_delta < -args.max_auc_drop:
        raise ValueError(
            f"{policy_name} violates the delta={args.max_auc_drop:.3f} "
            f"point-estimate constraint on tuning (delta={validation_delta:.6f})."
        )
    if validation_ci_low < -args.max_auc_drop:
        raise ValueError(
            f"{policy_name} does not meet the delta={args.max_auc_drop:.3f} "
            f"tuning CI screen (lower bound={validation_ci_low:.6f})."
        )

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    cost_scale = float(bundle["cost_normalization_scale"])
    frame = pd.read_parquet(args.test_decisions)

    required = {"long_view", "available_history_length"}
    required.update(feature_names)
    required.update({f"probability_K{k}" for k in K_VALUES})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Test artifact is missing required columns: {missing}")

    y = frame["long_view"].to_numpy(dtype=np.int8)
    lengths = frame["available_history_length"].to_numpy(dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Held-out test labels must contain both classes.")
    if np.any(lengths < 0):
        raise ValueError("History lengths cannot be negative.")

    predictions = {
        k: frame[f"probability_K{k}"].to_numpy(dtype=np.float64)
        for k in K_VALUES
    }
    uplift = predict_auc_uplift(model, frame, feature_names)
    # Equivalent to choose_actions(), without materializing Python history lists.
    costs = np.column_stack([
        np.zeros_like(lengths),
        *[np.minimum(lengths, int(k)) for k in K_VALUES[1:-1]],
        lengths,
    ]).astype(np.float64) / max(cost_scale, 1.0)
    utility = np.column_stack([uplift, np.zeros(len(frame))]) - args.lambda_penalty * costs
    actions = np.argmax(utility, axis=1)
    action_labels = np.asarray([str(k) for k in K_VALUES], dtype=object)
    selected_k = action_labels[actions]
    score_matrix = np.column_stack([predictions[k] for k in K_VALUES])
    policy_scores = score_matrix[np.arange(len(y)), actions]
    full_scores = predictions["Full"]

    policy_auc = float(roc_auc_score(y, policy_scores))
    full_auc = float(roc_auc_score(y, full_scores))
    policy_logloss = float(log_loss(y, np.clip(policy_scores, 1e-7, 1 - 1e-7), labels=[0, 1]))
    full_logloss = float(log_loss(y, np.clip(full_scores, 1e-7, 1 - 1e-7), labels=[0, 1]))
    used = np.where(
        actions == 0, 0,
        np.where(actions == len(K_VALUES) - 1, lengths,
                 np.minimum(lengths, np.asarray([0, 1, 5, 10, 20, 50, 100, 0])[actions]))
    )
    full_total = int(lengths.sum())
    used_total = int(used.sum())
    ci_low, ci_high = stratified_paired_auc_difference_ci(
        y, policy_scores, full_scores,
        repeats=args.bootstrap_repeats,
        random_state=args.bootstrap_seed + 9999,
    )

    result = {
        "policy": policy_name,
        "delta_constraint": args.max_auc_drop,
        "validation_tuning_auc_delta": validation_delta,
        "validation_tuning_ci_low_90": validation_ci_low,
        "test_auc": policy_auc,
        "test_full_auc": full_auc,
        "test_auc_delta_vs_full": policy_auc - full_auc,
        "test_auc_delta_ci_low_90": ci_low,
        "test_auc_delta_ci_high_90": ci_high,
        "test_logloss": policy_logloss,
        "test_full_logloss": full_logloss,
        "test_mean_history_interactions": float(used.mean()),
        "test_total_history_interactions": used_total,
        "test_full_history_interactions": full_total,
        "test_history_reduction_vs_full": 1.0 - used_total / max(full_total, 1),
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "note": "One-shot test evaluation; do not tune lambda using these results.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions = frame[["evaluation_id", "user_id", "long_view", "available_history_length"]].copy()
    decisions["selected_K"] = selected_k
    decisions["selected_probability"] = policy_scores
    decisions["full_probability"] = full_scores
    decisions_path = args.output_dir / "adaptive_v3_lambda_006235_test_decisions.parquet"
    metrics_path = args.output_dir / "adaptive_v3_lambda_006235_test_metrics.csv"
    decisions.to_parquet(decisions_path, index=False)
    pd.DataFrame([result]).to_csv(metrics_path, index=False)

    print("=== FIXED POLICY / HELD-OUT TEST ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("\nSelected K distribution:")
    print(pd.Series(selected_k).value_counts().sort_index().to_string())
    print(f"\nSaved decisions: {decisions_path}")
    print(f"Saved metrics:   {metrics_path}")
    print(
        "\nInterpretation: the test result is a final evaluation, not a new "
        "calibration step. Do not change lambda after inspecting it."
    )


if __name__ == "__main__":
    main()
