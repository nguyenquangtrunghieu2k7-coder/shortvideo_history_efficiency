"""M2.3C local serving benchmark: immutable SQLite snapshots, read-only concurrent workers.

No network endpoint or third-party infrastructure. Metrics are observed, not fabricated.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.integration.m23_persistent_store import HistoryStore


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def benchmark(root, snapshot_id, *, workers=8, repeats=5, max_keys=0):
    if workers < 1 or repeats < 1 or max_keys < 0:
        raise ValueError("workers/repeats must be positive; max_keys nonnegative")
    root = Path(root)
    # Read-only snapshot is validated once per worker; connections are not shared.
    started = time.perf_counter()
    with HistoryStore(root, snapshot_id, verify=True) as store:
        keys = []
        for conn in store.connections:
            keys.extend(row[0] for row in conn.execute("SELECT key FROM histories ORDER BY key"))
    validation_ms = (time.perf_counter() - started) * 1000
    keys.sort()
    if max_keys:
        keys = keys[:max_keys]
    if not keys:
        raise ValueError("Snapshot has no histories to benchmark")

    # Partition work into bounded batches. Each task owns and closes its SQLite
    # connections in the SAME thread; never close worker connections in main.
    def run_batch(batch):
        durations = []
        with HistoryStore(root, snapshot_id, verify=True) as store:
            for key, mode in batch:
                start_ns = time.perf_counter_ns()
                result = store.get(key, mode=mode)
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                if result is None:
                    raise AssertionError(f"Known key not found: {key}")
                durations.append(elapsed_ms)
        return durations

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for mode in ("indexed_exact", "sharded_exact", "sharded_bloom"):
            queries = [(key, mode) for _ in range(repeats) for key in keys]
            # Exactly one batch per worker (including workers > request count).
            batch_size = max(1, (len(queries) + workers - 1) // workers)
            batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
            start = time.perf_counter()
            durations = [ms for group in pool.map(run_batch, batches) for ms in group]
            wall = time.perf_counter() - start
            results[mode] = {
                "requests": len(queries),
                "p50_ms": percentile(durations, 50),
                "p95_ms": percentile(durations, 95),
                "p99_ms": percentile(durations, 99),
                "mean_ms": statistics.mean(durations),
                "throughput_rps": len(queries) / wall,
                "wall_seconds": wall,
            }

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS ru_maxrss is bytes; Linux is KiB.
    rss_bytes = rss if os.uname().sysname == "Darwin" else rss * 1024
    return {
        "status": "PASS",
        "snapshot_id": snapshot_id,
        "keys": len(keys),
        "workers": workers,
        "repeats": repeats,
        "snapshot_validation_ms": validation_ms,
        "peak_process_rss_mib": rss_bytes / (1024 * 1024),
        "benchmark_type": "local_warm_cache_read_only_threads",
        "limitations": [
            "No network/API transport measured",
            "Connection startup and validation excluded from per-request latency, included in throughput wall time",
            "Only known keys; missing-key workload not included",
            "OS cache is not cleared; no cold-cache guarantee",
            "RSS is process high-water mark, not per-request allocation",
        ],
        "modes": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/m23_snapshots"))
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-keys", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.root, args.snapshot_id, workers=args.workers,
                       repeats=args.repeats, max_keys=args.max_keys)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
