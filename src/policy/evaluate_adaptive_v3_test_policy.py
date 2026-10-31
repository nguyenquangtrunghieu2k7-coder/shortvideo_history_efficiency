"""Re-evaluate the test artifact using the exact V3 policy frozen by training.

This script does not accept a lambda override and never chooses a new policy.
It loads selected_validation_policy from the model bundle, verifies that the
policy passed the certification/fallback protocol, and applies the shared
policy implementation from adaptive_v3_policy.py.
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
    apply_selected_policy_from_lengths,
    interaction_matrix_from_lengths,
    selected_scores,
    stratified_paired_auc_difference_ci,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the certified, frozen Adaptive History Selector V3 policy."
    )
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
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase9_v3",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_repeats < 20:
        raise ValueError("--bootstrap-repeats must be at least 20.")
    for path in (args.model, args.test_decisions):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    bundle = joblib.load(args.model)
    if bundle.get("policy_frozen_after_certification") is not True:
        raise ValueError(
            "Model bundle does not declare a frozen post-certification policy. "
            "Run the updated train_adaptive_selector_v3.py first."
        )
    selected = bundle.get("selected_validation_policy")
    if not isinstance(selected, dict):
        raise ValueError("Model bundle is missing selected_validation_policy.")
    if "candidate_certification_passed" not in selected:
        raise ValueError("Frozen policy is missing its certification outcome.")
    if not selected["candidate_certification_passed"]:
        if (
            selected.get("deployment_mode") != "fixed_k_fallback"
            or str(selected.get("fixed_K")) != "Full"
            or not selected.get("certification_fallback_to_full")
        ):
            raise ValueError(
                "Certification failed but the frozen policy is not a Full fallback."
            )
    elif selected.get("certification_fallback_to_full"):
        raise ValueError("Policy metadata contradicts the certification outcome.")

    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    cost_scale = float(bundle["cost_normalization_scale"])
    frame = pd.read_parquet(args.test_decisions)

    required = {
        "evaluation_id", "user_id", "long_view", "available_history_length",
        "selected_policy", "deployment_mode",
    }
    required.update(feature_names)
    required.update({f"probability_K{k}" for k in K_VALUES})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Test artifact is missing required columns: {missing}")

    if not frame["selected_policy"].eq(selected["policy"]).all():
        raise ValueError(
            "Test decisions were generated for a different policy than the frozen model bundle. "
            "Re-run train_adaptive_selector_v3.py before evaluating."
        )
    if not frame["deployment_mode"].eq(selected["deployment_mode"]).all():
        raise ValueError(
            "Test decisions deployment_mode does not match the frozen model bundle."
        )

    y = frame["long_view"].to_numpy(dtype=np.int8)
    lengths = frame["available_history_length"].to_numpy(dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Test labels must contain both binary classes.")
    if np.any(lengths < 0):
        raise ValueError("History lengths cannot be negative.")

    predictions = {
        k: frame[f"probability_K{k}"].to_numpy(dtype=np.float64)
        for k in K_VALUES
    }
    actions = apply_selected_policy_from_lengths(
        selected, model, frame, lengths, cost_scale, feature_names
    )
    policy_scores = selected_scores(predictions, actions)
    full_scores = predictions["Full"]
    policy_auc = float(roc_auc_score(y, policy_scores))
    full_auc = float(roc_auc_score(y, full_scores))
    policy_logloss = float(
        log_loss(y, np.clip(policy_scores, 1e-7, 1 - 1e-7), labels=[0, 1])
    )
    full_logloss = float(
        log_loss(y, np.clip(full_scores, 1e-7, 1 - 1e-7), labels=[0, 1])
    )

    interaction_counts = interaction_matrix_from_lengths(lengths)
    used = interaction_counts[np.arange(len(y)), actions]
    full_total = int(lengths.sum())
    used_total = int(used.sum())
    ci_low, ci_high = stratified_paired_auc_difference_ci(
        y, policy_scores, full_scores,
        repeats=args.bootstrap_repeats,
        random_state=args.bootstrap_seed + 9999,
    )

    result = {
        "policy": selected["policy"],
        "deployment_mode": selected["deployment_mode"],
        "candidate_policy_before_certification": selected.get(
            "candidate_policy_before_certification"
        ),
        "candidate_certification_passed": selected["candidate_certification_passed"],
        "certification_fallback_to_full": selected.get(
            "certification_fallback_to_full", False
        ),
        "delta_constraint": bundle.get("max_auc_drop"),
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
        "note": (
            "Re-evaluation of the frozen post-certification policy; "
            "no lambda or policy selection is performed in this script."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions = frame[
        ["evaluation_id", "user_id", "long_view", "available_history_length"]
    ].copy()
    decisions["selected_K"] = [str(K_VALUES[i]) for i in actions]
    decisions["selected_probability"] = policy_scores
    decisions["full_probability"] = full_scores
    decisions_path = args.output_dir / "adaptive_v3_frozen_policy_test_decisions.parquet"
    metrics_path = args.output_dir / "adaptive_v3_frozen_policy_test_metrics.csv"
    decisions.to_parquet(decisions_path, index=False)
    pd.DataFrame([result]).to_csv(metrics_path, index=False)

    print("=== FROZEN V3 POLICY / HELD-OUT TEST ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("\\nSelected K distribution:")
    print(decisions["selected_K"].value_counts().sort_index().to_string())
    print(f"\\nSaved decisions: {decisions_path}")
    print(f"Saved metrics:   {metrics_path}")
    print(
        "\\nThis script only re-applies the policy frozen by training. "
        "It cannot override lambda or bypass certification."
    )


if __name__ == "__main__":
    main()
