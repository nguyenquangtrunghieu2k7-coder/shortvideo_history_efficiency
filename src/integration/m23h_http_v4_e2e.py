"""M2.3H: local HTTP retrieval -> frozen V4 -> frozen recommender parity gate.

No retraining, no modifications to V4 or model checkpoints. Local loopback only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import http.client
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from src.integration import e2e_v4_bloom as e2e
from src.policy.adaptive_v4_policy import K_VALUES, apply_selected_policy_from_lengths
from src.policy.adaptive_v4_features import FEATURE_NAMES, history_features


def fetch_selected(ids, actions, *, port, token, expected_snapshot):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    histories = []
    elapsed = perf_counter()
    try:
        conn.request("GET", "/ready", headers={"Authorization": "Bearer " + token})
        response = conn.getresponse()
        ready = json.loads(response.read())
        if response.status != 200 or ready.get("snapshot_id") != expected_snapshot:
            raise RuntimeError("API readiness/snapshot mismatch")
        for eid, action in zip(ids, actions):
            k = K_VALUES[int(action)]
            conn.request("GET", "/history?" + urlencode({"key": str(int(eid)), "k": k}),
                         headers={"Authorization": "Bearer " + token})
            response = conn.getresponse()
            payload = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError("HTTP history retrieval failed (status %s)" % response.status)
            if payload.get("key") != str(int(eid)) or payload.get("k") != str(k):
                raise RuntimeError("HTTP response contract mismatch")
            history = payload.get("history")
            if not isinstance(history, list) or not all(type(x) is int for x in history):
                raise RuntimeError("HTTP history payload invalid")
            histories.append(history)
    finally:
        conn.close()
    return histories, perf_counter() - elapsed


def run(args):
    if args.max_requests is not None and args.max_requests < 2:
        raise ValueError("max_requests must be >= 2")
    token = os.environ.get("M23_API_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("M23_API_TOKEN must be set (never paste it in reports)")
    start = perf_counter()
    device = e2e.baseline.get_device()
    vocab, vocab_map = e2e.baseline.load_video_vocab()
    dataset = e2e.baseline.KuaiRandDataset("test", "Full", vocab_map)
    n = min(len(dataset), args.max_requests) if args.max_requests else len(dataset)
    ids = np.asarray(dataset.evaluation_ids[:n])
    labels = np.asarray(dataset.labels[:n], dtype=np.int8)
    originals = dataset.history_ids[:n]
    if len(set(map(int, ids))) != n or set(map(int, labels)) != {0, 1}:
        raise RuntimeError("Invalid evaluation IDs or label distribution")

    path = args.policy_path or (e2e.ROOT / "models" / "adaptive_history_selector_v4.joblib")
    artifact = joblib.load(path)
    if not artifact.get("policy_frozen_after_validation_tuning"):
        raise RuntimeError("Policy is not marked frozen")
    if artifact["k_values"] != K_VALUES or artifact["feature_names"] != FEATURE_NAMES:
        raise RuntimeError("Frozen V4 schema mismatch")
    features = history_features(originals)
    actions = apply_selected_policy_from_lengths(
        artifact["selected_validation_policy"], artifact["model"], features,
        [len(h) for h in originals], artifact["cost_normalization_scale"], FEATURE_NAMES,
    )
    expected = e2e.select_histories(originals, actions)
    http_histories, retrieval_seconds = fetch_selected(
        ids, actions, port=args.port, token=token, expected_snapshot=args.snapshot_id
    )
    mismatches = sum(a != b for a, b in zip(expected, http_histories))
    if mismatches:
        raise AssertionError("HTTP selected-history parity failed: %s mismatches" % mismatches)

    # Independent frozen-model inference on both paths, without modifying checkpoints.
    exact_scores, exact_inference = e2e.predict_by_action(
        dataset, expected, actions, len(vocab), device, args.batch_size
    )
    http_scores, http_inference = e2e.predict_by_action(
        dataset, http_histories, actions, len(vocab), device, args.batch_size
    )
    delta = np.abs(exact_scores - http_scores)
    max_delta = float(delta.max())
    prediction_equal = bool(np.allclose(exact_scores, http_scores, rtol=0, atol=args.atol))
    auc_exact = float(roc_auc_score(labels, exact_scores))
    auc_http = float(roc_auc_score(labels, http_scores))
    full_count = sum(map(len, originals))
    selected_count = sum(map(len, http_histories))
    report = {
        "status": "PASS" if prediction_equal and mismatches == 0 else "FAIL",
        "gate": "M2.3H_HTTP_TO_FROZEN_V4_LOCAL_E2E",
        "scope": "local_loopback",
        "snapshot_id": args.snapshot_id,
        "n_requests": n,
        "http_history_requests": n,
        "history_mismatches": mismatches,
        "prediction_max_abs_delta": max_delta,
        "prediction_tolerance_atol": args.atol,
        "prediction_parity": prediction_equal,
        "auc_exact": auc_exact,
        "auc_http": auc_http,
        "history_saving": 1 - selected_count / max(full_count, 1),
        "history_full_interactions": full_count,
        "history_selected_interactions": selected_count,
        "action_counts": dict(Counter(str(K_VALUES[int(i)]) for i in actions)),
        "timings_seconds": {
            "http_retrieval": retrieval_seconds,
            "exact_recommender_inference": exact_inference,
            "http_recommender_inference": http_inference,
            "total": perf_counter() - start,
        },
        "limitations": [
            "Local loopback, not cloud or production serving.",
            "Full history is preloaded for frozen policy feature computation.",
            "Bloom assisted in-memory E2E is validated by the separate offline gate; this gate tests HTTP exact retrieval and frozen-model parity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--policy-path", type=Path)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-requests", type=int)
    p.add_argument("--atol", type=float, default=1e-7)
    p.add_argument("--output", type=Path, default=Path("reports/m23h_http_v4_e2e.json"))
    args = p.parse_args()
    if args.batch_size < 1 or args.atol < 0:
        p.error("batch-size must be positive and atol nonnegative")
    run(args)


if __name__ == "__main__":
    main()
