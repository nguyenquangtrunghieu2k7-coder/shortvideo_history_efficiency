"""M2.3F real-data API correctness gate: all records, misses and concurrency.

Does not log keys, payloads, API tokens or HMAC secrets.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import http.client
import json
import os
import random
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlencode

from src.integration.m23_activation import open_active


def _fetch(port, token, key, k):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("GET", "/history?" + urlencode({"key": key, "k": k}),
                     headers={"Authorization": "Bearer " + token})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def run(root, secret, token, *, port=8765, workers=8, sample=0):
    if workers < 1 or sample < 0:
        raise ValueError("workers must be positive and sample nonnegative")
    # Independent SQLite ground truth; do not derive expected values from API/store retrieval.
    with open_active(root, secret) as store:
        snapshot_id = store.manifest["snapshot_id"]
        expected = {}
        for i in range(store.shards):
            db = Path(root) / snapshot_id / f"shard_{i}.sqlite"
            with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
                for key, payload in conn.execute("SELECT key,payload FROM histories"):
                    if key in expected:
                        raise ValueError("Duplicate source key")
                    expected[key] = json.loads(payload)
        if len(expected) != store.manifest["records"]:
            raise ValueError("Source record count mismatch")

    keys = sorted(expected)
    if sample and sample < len(keys):
        keys = random.Random(20261117).sample(keys, sample)
    jobs = []
    for key in keys:
        history = expected[key]
        for k in ("Full", 0, 1, 5, len(history) + 3):
            target = history if k == "Full" else (history[-k:] if k else [])
            jobs.append((key, k, 200, target))
    for i in range(max(100, len(keys) // 10)):
        missing = f"__m23f_absent_{i:08d}__"
        if missing not in expected:
            jobs.append((missing, 5, 404, None))
    random.Random(917).shuffle(jobs)

    def check(job):
        key, k, status, target = job
        started = time.perf_counter()
        try:
            actual_status, data = _fetch(port, token, key, k)
            if actual_status != status:
                reason = "http_status_" + str(actual_status) + "_expected_" + str(status)
            elif status == 200 and data.get("history") != target:
                reason = "payload_mismatch"
            elif status == 200 and data.get("key") != key:
                reason = "key_mismatch"
            elif status == 404 and data.get("error") != "history_not_found":
                reason = "missing_error_contract"
            else:
                reason = "ok"
        except OSError as exc:
            # errno is safe metadata; messages may expose request URLs or keys.
            code = str(exc.errno) if exc.errno is not None else "none"
            reason = "exception_" + type(exc).__name__ + "_errno_" + code
        except Exception as exc:
            # Exception class only, never the message (could contain URLs/keys).
            reason = "exception_" + type(exc).__name__
        return reason, (time.perf_counter() - started) * 1000, str(k) if k == "Full" else ("zero" if k == 0 else "positive"), status

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(check, jobs))
    elapsed = time.perf_counter() - start
    latencies = sorted(ms for _, ms, _, _ in outcomes)
    failure_types = Counter(reason for reason, _, _, _ in outcomes if reason != 'ok')
    failure_by_k = Counter(kind for reason, _, kind, _ in outcomes if reason != 'ok')
    failure_by_expected_status = Counter(str(status) for reason, _, _, status in outcomes if reason != 'ok')
    failures = sum(failure_types.values())
    n = len(latencies)
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "snapshot_id": snapshot_id,
        "source_records": len(expected),
        "tested_records": len(keys),
        "requests": n,
        "failures": failures,
        "failure_types": dict(sorted(failure_types.items())),
        "failure_by_k": dict(sorted(failure_by_k.items())),
        "failure_by_expected_status": dict(sorted(failure_by_expected_status.items())),
        "workers": workers,
        "wall_seconds": round(elapsed, 4),
        "throughput_rps": round(n / elapsed, 2) if elapsed else None,
        "p50_ms": round(latencies[int((n - 1) * .50)], 3),
        "p95_ms": round(latencies[int((n - 1) * .95)], 3),
        "p99_ms": round(latencies[int((n - 1) * .99)], 3),
        "benchmark_type": "loopback_http_pinned_verified_snapshot",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/processed/m23_snapshots"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sample", type=int, default=0, help="0 tests all keys")
    p.add_argument("--output", type=Path, default=Path("reports/m23f_api_gate.json"))
    args = p.parse_args()
    try:
        secret = bytes.fromhex(os.environ.get("M23_HMAC_KEY_HEX", ""))
    except ValueError as exc:
        raise SystemExit("Invalid M23_HMAC_KEY_HEX") from exc
    token = os.environ.get("M23_API_TOKEN", "")
    if len(secret) < 32 or len(token) < 32:
        raise SystemExit("Set M23_HMAC_KEY_HEX and M23_API_TOKEN in this terminal")
    report = run(args.root, secret, token, port=args.port, workers=args.workers, sample=args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
