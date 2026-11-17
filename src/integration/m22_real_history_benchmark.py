"""M2.2: real KuaiRand test histories, SQLite retrieval and Bloom benchmark.

Offline experiment. Does not claim production deployment or cold-cache latency.
"""
from __future__ import annotations
import argparse
import json
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

from src.bloom_pipeline.bloom_filter import BloomFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))


def pct(values, fraction):
    a = sorted(values)
    pos = (len(a) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def run(args):
    if min(args.shards, args.repeats, args.bits_per_key, args.hashes) < 1:
        raise ValueError("shards/repeats/bits-per-key/hashes must be positive")
    import train_baseline as baseline
    _, vocab_map = baseline.load_video_vocab()
    dataset = baseline.KuaiRandDataset("test", "Full", vocab_map)
    keys = [str(x) for x in dataset.evaluation_ids]
    histories = [[int(v) for v in h] for h in dataset.history_ids]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate evaluation_id")
    if args.max_requests:
        keys, histories = keys[:args.max_requests], histories[:args.max_requests]
    n = len(keys)
    if n == 0:
        raise ValueError("Empty workload")
    rng = random.Random(args.seed)
    assignment = list(range(n))
    rng.shuffle(assignment)
    routing = {key: assignment[i] % args.shards for i, key in enumerate(keys)}
    # A known routing map is an optimistic indexed baseline, not a free operation.
    with tempfile.TemporaryDirectory(prefix="m22_") as folder:
        conns, filters = [], []
        build_start = time.perf_counter()
        for shard in range(args.shards):
            db = sqlite3.connect(str(Path(folder) / f"shard_{shard}.sqlite"))
            db.execute("CREATE TABLE histories (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            rows = [(key, json.dumps(history, separators=(",", ":")))
                    for key, history in zip(keys, histories) if routing[key] == shard]
            db.executemany("INSERT INTO histories VALUES (?, ?)", rows)
            db.commit()
            bloom = BloomFilter(max(64, len(rows) * args.bits_per_key), args.hashes)
            for key, _ in rows:
                bloom.add(key)
            conns.append(db)
            filters.append(bloom)
        build_seconds = time.perf_counter() - build_start
        bloom_bytes = sum(len(b._bits) for b in filters)
        # Interleave real hit requests with explicitly absent IDs.
        workload = list(keys)
        workload.extend(f"__m22_missing_{i}" for i in range(round(n * args.miss_ratio / max(1e-9, 1 - args.miss_ratio))))
        rng.shuffle(workload)
        truth = {key: json.dumps(h, separators=(",", ":")) for key, h in zip(keys, histories)}
        methods = ["indexed_exact", "sharded_exact", "sharded_bloom"]
        samples = {m: [] for m in methods}
        lookups = {m: 0 for m in methods}
        fps = {m: 0 for m in methods}
        for rep in range(args.repeats):
            # Rotate order to reduce fixed warm-cache bias.
            order = methods[rep % 3:] + methods[:rep % 3]
            for method in order:
                durations = []
                for key in workload:
                    start = time.perf_counter_ns()
                    found = None
                    shards = ([routing[key]] if key in routing else [0]) if method == "indexed_exact" else range(args.shards)
                    for shard in shards:
                        if method == "sharded_bloom" and not filters[shard].might_contain(key):
                            continue
                        lookups[method] += 1
                        row = conns[shard].execute("SELECT payload FROM histories WHERE key=?", (key,)).fetchone()
                        if row is not None:
                            if found is not None:
                                raise AssertionError("Duplicate key across shards")
                            found = row[0]
                        elif method == "sharded_bloom":
                            fps[method] += 1
                    durations.append((time.perf_counter_ns() - start) / 1e6)
                    if found != truth.get(key):
                        raise AssertionError(f"Retrieval mismatch: {method}, key={key}")
                samples[method].extend(durations)
        for db in conns:
            db.close()
    result = {
        "status": "PASS", "dataset": "KuaiRand frozen test Full histories",
        "requests": n, "query_count_per_repeat": len(workload),
        "repeats": args.repeats, "shards": args.shards,
        "miss_ratio_requested": args.miss_ratio,
        "sqlite_and_bloom_build_seconds": build_seconds,
        "bloom_bit_array_bytes": bloom_bytes,
        "routing_map_entries": len(routing),
        "equivalence": True,
        "results": {m: {"p50_ms": pct(samples[m], .5), "p95_ms": pct(samples[m], .95),
                        "p99_ms": pct(samples[m], .99), "mean_ms": statistics.mean(samples[m]),
                        "exact_lookups": lookups[m], "false_positives": fps[m]}
                    for m in methods},
        "limitations": [
            "Frozen test history payloads, but synthetic randomized shard placement and misses.",
            "SQLite local filesystem and OS cache not controlled; repeated warm-cache timings.",
            "Indexed exact uses an in-memory routing map and one SQLite probe for misses.",
            "Sequential shard fan-out; not remote/network retrieval.",
            "Full history is loaded to construct offline store; no lazy policy feature extraction.",
            "Build timing includes SQLite materialization; excludes baseline dataset loading.",
            "Bloom memory reports bit arrays only, excludes Python and SQLite overhead."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description="M2.2 KuaiRand SQLite history benchmark")
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--bits-per-key", type=int, default=10)
    p.add_argument("--hashes", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-requests", type=int, default=0)
    p.add_argument("--miss-ratio", type=float, default=.2)
    p.add_argument("--output", type=Path, default=Path("data/processed/m22_real_history.json"))
    args = p.parse_args()
    if not 0 <= args.miss_ratio < 1:
        p.error("--miss-ratio must be in [0, 1)")
    run(args)


if __name__ == "__main__":
    main()
