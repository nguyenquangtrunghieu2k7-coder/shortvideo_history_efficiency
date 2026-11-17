"""M2 SQLite retrieval benchmark: synthetic workload, real SQLite reads.

This benchmark does not claim production speedup or use KuaiRand data.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

from src.bloom_pipeline.bloom_filter import BloomFilter


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * p
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] * (high - pos) + values[high] * (pos - low)


def bench(args):
    if args.shards < 2 or args.keys < 1 or args.queries < 1:
        raise ValueError("shards >= 2 and keys/queries >= 1 required")
    with tempfile.TemporaryDirectory(prefix="m2_sqlite_") as folder:
        base = Path(folder)
        connections = []
        filters = []
        # Interleaved keys simulate disjoint history shards, without known routing.
        for shard in range(args.shards):
            db = sqlite3.connect(base / f"shard_{shard}.sqlite")
            db.execute("CREATE TABLE histories (key TEXT PRIMARY KEY, history TEXT NOT NULL)")
            keys = [f"req_{i:09d}" for i in range(shard, args.keys, args.shards)]
            db.executemany("INSERT INTO histories VALUES (?, ?)", ((key, "1,2,3,4,5") for key in keys))
            db.commit()
            bf = BloomFilter(m=max(64, len(keys) * args.bits_per_key), k=args.hashes)
            for key in keys:
                bf.add(key)
            connections.append(db)
            filters.append(bf)

        # 80% hits, 20% misses; deterministic workload.
        import random
        rng = random.Random(args.seed)
        queries = [f"req_{rng.randrange(args.keys):09d}" if i % 5 else f"absent_{i:09d}"
                   for i in range(args.queries)]
        results = {}
        for name in ("indexed_exact", "sharded_exact", "sharded_bloom"):
            latencies = []
            lookups = 0
            false_positives = 0
            outputs = []
            for key in queries:
                t0 = time.perf_counter_ns()
                found = None
                if name == "indexed_exact":
                    # An oracle routing baseline: modulo-shard mapping for known keys;
                    # missing keys are routed to one shard. Not a general index.
                    try:
                        shard_ids = [int(key.split("_")[1]) % args.shards]
                    except (ValueError, IndexError):
                        shard_ids = [0]
                else:
                    shard_ids = range(args.shards)
                for shard in shard_ids:
                    if name == "sharded_bloom" and not filters[shard].might_contain(key):
                        continue
                    lookups += 1
                    row = connections[shard].execute(
                        "SELECT history FROM histories WHERE key = ?", (key,)
                    ).fetchone()
                    if row is not None:
                        if found is not None:
                            raise AssertionError("duplicate key across shards")
                        found = row[0]
                    elif name == "sharded_bloom":
                        false_positives += 1
                latencies.append((time.perf_counter_ns() - t0) / 1e6)
                outputs.append(found)
            results[name] = {
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "mean_ms": statistics.mean(latencies),
                "exact_lookups": lookups,
                "false_positives": false_positives,
            }
            if name == "indexed_exact":
                expected = outputs
            elif outputs != expected:
                raise AssertionError(f"{name} differs from indexed_exact")
        for db in connections:
            db.close()
    return {
        "status": "PASS",
        "dataset": "synthetic",
        "backend": "SQLite local disk; OS cache not controlled",
        "queries": args.queries,
        "keys": args.keys,
        "shards": args.shards,
        "equivalence": True,
        "results": results,
        "limitations": [
            "Synthetic keys and short histories, not real KuaiRand distribution.",
            "Indexed exact uses known deterministic routing; an oracle baseline.",
            "SQLite and OS page caches are not controlled; timings are warm-cache influenced.",
            "Sequential shard probes; not remote/distributed I/O.",
            "Bloom build time and memory are not included in retrieval timing.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keys", type=int, default=10000)
    p.add_argument("--queries", type=int, default=2000)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--bits-per-key", type=int, default=10)
    p.add_argument("--hashes", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("data/processed/m2_sqlite_synthetic.json"))
    args = p.parse_args()
    result = bench(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
