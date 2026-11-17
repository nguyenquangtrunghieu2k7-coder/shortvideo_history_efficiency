"""Offline V4 + Bloom-assisted history retrieval + frozen recommender runner.

This is an IN-MEMORY sharded retrieval benchmark, not production remote I/O.
The Bloom filter only avoids exact shard lookups for definite non-members.
No changes to existing policy/recommender/Bloom modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

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
from adaptive_v4_policy import K_VALUES, apply_selected_policy_from_lengths  # noqa: E402
from src.bloom_pipeline.distributed_build import distributed_build  # noqa: E402
from src.bloom_pipeline.probe import probe_filter  # noqa: E402


def build_shards(evaluation_ids, histories, n_shards, bits_per_key, hashes, workers):
    """Deterministic disjoint request-history store, one Bloom per shard.

    Keys are evaluation_id, not video_id: they identify *retrievable histories*.
    Each Bloom is constructed with the existing distributed_build/OR-merge code.
    """
    if n_shards < 1 or bits_per_key < 1 or hashes < 1 or workers < 1:
        raise ValueError("shards/bits-per-key/hashes/workers must be positive")
    stores = [dict() for _ in range(n_shards)]
    for eid, history in zip(evaluation_ids, histories):
        key = str(int(eid))
        shard_id = int(eid) % n_shards
        if key in stores[shard_id]:
            raise ValueError(f"Duplicate evaluation_id: {key}")
        stores[shard_id][key] = list(history)
    filters = []
    started = perf_counter()
    for store in stores:
        m = max(8, bits_per_key * max(len(store), 1))
        _, merged = distributed_build(list(store), num_workers=workers, m=m, k=hashes)
        filters.append(merged)
    return stores, filters, perf_counter() - started


def retrieve_histories(evaluation_ids, stores, filters, use_bloom):
    """Fan-out lookup, Bloom MAYBE -> exact dict verification; fail on missing keys."""
    result = []
    stats = Counter()
    started = perf_counter()
    for eid in evaluation_ids:
        key = str(int(eid))
        found = []
        for store, bloom in zip(stores, filters):
            stats["shard_candidates"] += 1
            if use_bloom:
                maybe, _ = probe_filter([{"evaluation_id": key}], bloom, "evaluation_id")
                if not maybe:
                    stats["bloom_rejections"] += 1
                    if key in store:
                        raise AssertionError("Bloom false negative: unsafe index")
                    continue
                stats["bloom_maybes"] += 1
            stats["exact_lookups"] += 1
            value = store.get(key)
            if value is not None:
                found.append(value)
                stats["exact_hits"] += 1
            elif use_bloom:
                stats["bloom_false_positives"] += 1
        if len(found) != 1:
            raise RuntimeError(f"Expected one history for evaluation_id={key}; got {len(found)}")
        result.append(found[0])
    return result, dict(stats), perf_counter() - started


def select_histories(histories, actions):
    out = []
    for history, index in zip(histories, actions):
        k = K_VALUES[int(index)]
        out.append(list(history) if k == "Full" else (list(history[-int(k):]) if k else []))
    return out


def predict_by_action(dataset, histories, actions, vocab_size, device, batch_size):
    """Use existing K-specific frozen checkpoints and collate_batch."""
    scores = np.empty(len(actions), dtype=np.float64)
    inference_seconds = 0.0
    for action_idx, k in enumerate(K_VALUES):
        indices = np.flatnonzero(actions == action_idx)
        if len(indices) == 0:
            continue
        checkpoint_path = baseline.MODEL_DIR / baseline.model_filename("full", k)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing frozen K={k} checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = baseline.build_model(vocab_size, mode="full").to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch = []
                for i in batch_indices:
                    item = dict(dataset[int(i)])
                    item["history_ids"] = histories[int(i)]
                    batch.append(item)
                candidate, history, mask, _, _, _ = baseline.collate_batch(batch)
                t0 = perf_counter()
                pred = model(candidate.to(device), history.to(device), mask.to(device))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                inference_seconds += perf_counter() - t0
                scores[batch_indices] = pred.detach().cpu().numpy()
        del model
    if not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite recommender predictions")
    return scores, inference_seconds


def run(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.max_requests is not None and args.max_requests < 2:
        raise ValueError("max-requests must be >= 2")
    t_total = perf_counter()
    device = baseline.get_device()
    vocab, vocab_map = baseline.load_video_vocab()
    dataset = baseline.KuaiRandDataset("test", "Full", vocab_map)
    n = min(len(dataset), args.max_requests) if args.max_requests else len(dataset)
    ids = np.asarray(dataset.evaluation_ids[:n])
    labels = np.asarray(dataset.labels[:n], dtype=np.int8)
    originals = dataset.history_ids[:n]
    if len(set(ids.tolist())) != n:
        raise ValueError("Non-unique evaluation IDs")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Selected subset must contain both label classes")

    t0 = perf_counter()
    features = history_features(originals)
    feature_seconds = perf_counter() - t0
    if args.mode == "v4":
        path = args.policy_path.expanduser().resolve() if args.policy_path else ROOT / "models" / "adaptive_history_selector_v4.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen policy: {path}. Supply --policy-path /absolute/path/to/adaptive_history_selector_v4.joblib; do not retrain on test.")
        artifact = joblib.load(path)
        if not artifact.get("policy_frozen_after_validation_tuning"):
            raise ValueError("V4 policy artifact is not marked frozen")
        if artifact["k_values"] != K_VALUES or artifact["feature_names"] != FEATURE_NAMES:
            raise ValueError("V4 feature/action schema mismatch")
        t0 = perf_counter()
        actions = apply_selected_policy_from_lengths(
            artifact["selected_validation_policy"], artifact["model"], features,
            [len(h) for h in originals], artifact["cost_normalization_scale"], FEATURE_NAMES
        )
        policy_seconds = perf_counter() - t0
    else:
        actions = np.full(n, K_VALUES.index("Full"), dtype=np.int64)
        policy_seconds = 0.0

    t0 = perf_counter()
    stores, filters, build_seconds = build_shards(
        ids, originals, args.shards, args.bits_per_key, args.hashes, args.workers
    )
    exact_histories, exact_stats, exact_seconds = retrieve_histories(ids, stores, filters, False)
    bloom_histories, bloom_stats, bloom_seconds = retrieve_histories(ids, stores, filters, True)
    if exact_histories != originals or bloom_histories != originals:
        raise AssertionError("Bloom/exact retrieval mismatch with source history")
    selected = select_histories(bloom_histories if args.bloom else exact_histories, actions)
    selection_seconds = perf_counter() - t0

    scores, inference_seconds = predict_by_action(
        dataset, selected, actions, len(vocab), device, args.batch_size
    )
    full_count = sum(map(len, originals))
    selected_count = sum(map(len, selected))
    result = {
        "status": "PASS",
        "protocol": "offline_in_memory_sharded_evaluation_id_lookup",
        "mode": args.mode, "bloom_enabled": args.bloom,
        "n_requests": n, "auc": float(roc_auc_score(labels, scores)),
        "history_saving": 1 - selected_count / max(full_count, 1),
        "history_full_interactions": full_count,
        "history_selected_interactions": selected_count,
        "action_counts": dict(Counter(str(K_VALUES[int(i)]) for i in actions)),
        "retrieval_equivalence": True,
        "bloom_false_negatives": 0,
        "bloom_stats": bloom_stats, "exact_stats": exact_stats,
        "timings_seconds": {
            "feature": feature_seconds, "policy": policy_seconds,
            "bloom_index_build_offline": build_seconds,
            "exact_retrieval_comparison": exact_seconds,
            "bloom_retrieval_comparison": bloom_seconds,
            "selected_retrieval_and_slice": selection_seconds,
            "recommender_inference": inference_seconds,
            "total_including_both_retrieval_comparisons": perf_counter() - t_total,
        },
        "limitations": [
            "In-memory fan-out benchmark, not network/disk I/O or production latency.",
            "Offline index build excluded from serving comparison; include rebuild cost separately.",
            "Preloaded Full histories used to calculate policy features; not yet lazy retrieval.",
            "Selected-K inference uses independently trained frozen K-specific checkpoints.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["full", "v4"], default="v4")
    p.add_argument("--policy-path", type=Path, help="Path to existing frozen V4 joblib artifact")
    p.add_argument("--bloom", action="store_true")
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--bits-per-key", type=int, default=12)
    p.add_argument("--hashes", type=int, default=5)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-requests", type=int)
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/e2e_m1_report.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
