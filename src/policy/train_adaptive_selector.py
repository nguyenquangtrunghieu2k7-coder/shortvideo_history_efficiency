"""Train and evaluate a per-request adaptive history-budget selector.

The selector is a small supervised controller. For each evaluation request it
uses only pre-exposure features from the available history and candidate ID to
choose one of the existing K-specific recommender checkpoints.

Training targets are generated on the validation split:
  1. Run each frozen K-specific recommender.
  2. Compute per-request binary log loss.
  3. Choose the smallest K whose loss is no more than
     FULL_LOGLOSS + max_logloss_regret.

The test split is used only for final evaluation. It never creates selector
training labels.

This is a local adaptive-history prototype, not a serverless benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.dummy import DummyClassifier


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))

import train_baseline as baseline  # noqa: E402


K_VALUES = [0, 1, 5, 10, 20, 50, 100, "Full"]
FEATURE_NAMES = [
    "history_length",
    "unique_history_count",
    "unique_ratio",
    "repeat_ratio",
    "unique_last_5",
    "unique_last_20",
    "unique_last_50",
    "candidate_seen",
    "candidate_frequency",
    "candidate_recency_distance",
]
EPS = 1e-7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a per-request adaptive history selector."
    )
    parser.add_argument(
        "--max-logloss-regret",
        type=float,
        default=0.05,
        help=(
            "Per-request loss tolerance relative to the Full model. "
            "This is a log-loss threshold, not the earlier AUC epsilon."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    return parser.parse_args()


def history_features(histories, candidates) -> pd.DataFrame:
    """Build request-time features; never reads labels or future outcomes."""
    rows = []

    for history, candidate in zip(histories, candidates):
        values = [int(x) for x in history]
        n = len(values)
        unique = set(values)
        unique_n = len(unique)
        candidate = int(candidate)

        positions = [
            i for i, value in enumerate(values)
            if value == candidate
        ]

        rows.append({
            "history_length": n,
            "unique_history_count": unique_n,
            "unique_ratio": unique_n / n if n else 0.0,
            "repeat_ratio": 1.0 - unique_n / n if n else 0.0,
            "unique_last_5": len(set(values[-5:])) / min(n, 5)
            if n else 0.0,
            "unique_last_20": len(set(values[-20:])) / min(n, 20)
            if n else 0.0,
            "unique_last_50": len(set(values[-50:])) / min(n, 50)
            if n else 0.0,
            "candidate_seen": int(bool(positions)),
            "candidate_frequency": len(positions),
            # 0 means the candidate appeared in the most recent history row.
            # n means it has never appeared.
            "candidate_recency_distance": (
                n - 1 - positions[-1] if positions else n
            ),
        })

    return pd.DataFrame(rows, columns=FEATURE_NAMES)


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
    checkpoint_path = (
        baseline.MODEL_DIR / baseline.model_filename("full", k)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing frozen recommender checkpoint: {checkpoint_path}\n"
            "Train the Phase 8 full-mode K-specific models first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
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
    predictions = {}
    for k in K_VALUES:
        print(f"  Predicting {dataset_name(dataset)} with K={k} ...", flush=True)
        predictions[k] = predict_for_k(
            dataset, k, vocab_size, device, batch_size
        )
    return predictions


def dataset_name(dataset):
    return getattr(dataset, "split_name", "split")


def clipped_logloss_per_row(y_true, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y_true, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def build_oracle_labels(y_true, predictions, max_logloss_regret):
    """Smallest K meeting a per-request loss-regret constraint vs Full."""
    full_loss = clipped_logloss_per_row(y_true, predictions["Full"])
    loss_matrix = np.column_stack([
        clipped_logloss_per_row(y_true, predictions[k])
        for k in K_VALUES
    ])

    # K_VALUES is ordered from least history to most history.
    eligible = loss_matrix <= (full_loss[:, None] + max_logloss_regret)
    # Full is its own reference and must always be eligible.
    eligible[:, -1] = True

    first_eligible = eligible.argmax(axis=1)
    return np.asarray([str(K_VALUES[i]) for i in first_eligible], dtype=str), loss_matrix


def metric_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def metric_logloss(y_true, scores):
    return float(log_loss(y_true, np.clip(scores, EPS, 1.0 - EPS), labels=[0, 1]))


def fixed_k_metrics(y_true, predictions, dataset, k):
    scores = predictions[k]
    history_lengths = [len(h) for h in dataset.history_ids]
    used = np.asarray([
        0 if k == 0 else (n if k == "Full" else min(int(k), n))
        for n in history_lengths
    ])
    return {
        "policy": f"K={k}",
        "auc": metric_auc(y_true, scores),
        "logloss": metric_logloss(y_true, scores),
        "mean_history_interactions": float(used.mean()),
        "total_history_interactions": int(used.sum()),
        "history_reduction_vs_full": float(
            1.0 - used.sum() / max(sum(history_lengths), 1)
        ),
    }


def evaluate_adaptive(
    split_name,
    dataset,
    y_true,
    predictions,
    selector,
    features,
):
    selected = selector.predict(features[FEATURE_NAMES])
    selected_scores = np.empty(len(selected), dtype=np.float64)
    selected_lengths = np.empty(len(selected), dtype=np.int64)

    full_lengths = np.asarray([len(h) for h in dataset.history_ids])
    for k in K_VALUES:
        mask = selected == str(k)
        if not np.any(mask):
            continue
        selected_scores[mask] = predictions[k][mask]
        if k == 0:
            selected_lengths[mask] = 0
        elif k == "Full":
            selected_lengths[mask] = full_lengths[mask]
        else:
            selected_lengths[mask] = np.minimum(full_lengths[mask], int(k))

    full_total = int(full_lengths.sum())
    selected_total = int(selected_lengths.sum())

    metrics = {
        "policy": "Adaptive",
        "auc": metric_auc(y_true, selected_scores),
        "logloss": metric_logloss(y_true, selected_scores),
        "mean_history_interactions": float(selected_lengths.mean()),
        "total_history_interactions": selected_total,
        "history_reduction_vs_full": (
            1.0 - selected_total / max(full_total, 1)
        ),
    }

    decisions = pd.DataFrame({
        "evaluation_id": dataset.evaluation_ids,
        "user_id": dataset.user_ids,
        "selected_K": selected,
        "available_history_length": full_lengths,
        "selected_history_length": selected_lengths,
        "long_view": y_true,
        "predicted_long_view_probability": selected_scores,
    })
    return metrics, decisions


def main():
    args = parse_args()
    if args.max_logloss_regret < 0:
        raise ValueError("--max-logloss-regret must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    print("=" * 78)
    print("PER-REQUEST ADAPTIVE HISTORY SELECTOR")
    print("=" * 78)

    device = baseline.get_device()
    print(f"Device: {device}")

    vocab, vocab_map = baseline.load_video_vocab()
    vocab_size = len(vocab)

    print("\nLoading validation requests using Full histories...")
    validation = baseline.KuaiRandDataset("validation", "Full", vocab_map)
    validation.split_name = "validation"
    val_y = np.asarray(validation.labels, dtype=np.int64)

    print("\nBuilding validation request-time features...")
    val_features = history_features(
        validation.history_ids, validation.candidate_ids
    )

    print("\nGenerating validation predictions from frozen K models...")
    val_predictions = collect_predictions(
        validation, vocab_size, device, args.batch_size
    )
    # Release the large Python history lists before loading the test split.
    del validation

    print("\nLoading held-out test requests using Full histories...")
    test = baseline.KuaiRandDataset("test", "Full", vocab_map)
    test.split_name = "test"
    test_y = np.asarray(test.labels, dtype=np.int64)

    print("\nBuilding test request-time features...")
    test_features = history_features(
        test.history_ids, test.candidate_ids
    )

    print("\nGenerating test predictions from frozen K models...")
    test_predictions = collect_predictions(
        test, vocab_size, device, args.batch_size
    )

    print("\nCreating validation-only oracle labels...")
    oracle_k, val_loss_matrix = build_oracle_labels(
        val_y, val_predictions, args.max_logloss_regret
    )
    print("Oracle K distribution:")
    print(pd.Series(oracle_k).value_counts().sort_index().to_string())

    # Train only on validation-derived request features and oracle labels.
    # The test split is not used to fit or tune the selector.
    unique_labels = pd.unique(oracle_k)
    if len(unique_labels) == 1:
        selector = DummyClassifier(strategy="most_frequent")
    else:
        selector = RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.random_state,
            n_jobs=-1,
        )
    selector.fit(val_features[FEATURE_NAMES], oracle_k)

    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    selector_path = MODEL_DIR / "adaptive_history_selector.joblib"
    joblib.dump({
        "selector": selector,
        "feature_names": FEATURE_NAMES,
        "k_values": K_VALUES,
        "max_logloss_regret": args.max_logloss_regret,
        "target_definition": (
            "smallest K with per-request logloss <= "
            "Full per-request logloss + max_logloss_regret"
        ),
    }, selector_path)

    print("\nEvaluating fixed-K baselines and adaptive policy on held-out test...")
    result_rows = [
        fixed_k_metrics(test_y, test_predictions, test, k)
        for k in K_VALUES
    ]
    adaptive_metrics, decisions = evaluate_adaptive(
        "test",
        test,
        test_y,
        test_predictions,
        selector,
        test_features,
    )
    result_rows.append(adaptive_metrics)

    output_dir = ROOT / "data" / "processed" / "phase9"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "adaptive_history_metrics.csv"
    decisions_path = output_dir / "adaptive_history_test_decisions.parquet"
    feature_importance_path = output_dir / "adaptive_selector_feature_importance.csv"

    pd.DataFrame(result_rows).to_csv(metrics_path, index=False)
    decisions.to_parquet(decisions_path, index=False)

    if hasattr(selector, "feature_importances_"):
        pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": selector.feature_importances_,
        }).sort_values("importance", ascending=False).to_csv(
            feature_importance_path, index=False
        )

    print("\n=== TEST RESULTS ===")
    print(pd.DataFrame(result_rows).to_string(index=False))
    print("\nSelected K distribution:")
    print(decisions["selected_K"].value_counts().sort_index().to_string())
    print(f"\nSaved selector: {selector_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved per-request decisions: {decisions_path}")
    if feature_importance_path.exists():
        print(f"Saved feature importance: {feature_importance_path}")
    print("\nProtocol: validation trains the selector; test is held out.")
    print(
        "Runtime note: this experiment precomputes each K model's scores "
        "for evaluation. The reported history-interaction reduction is "
        "the adaptive request budget, not yet an end-to-end serving speedup."
    )


if __name__ == "__main__":
    main()
