from pathlib import Path
from time import perf_counter
import statistics

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "data/processed/phase7"
RESULTS_DIR = ROOT / "data/processed/phase8"

K_VALUES = ["100", "Full"]
REPEATS = 4
BATCH_SIZE = 256


def benchmark(k, repeat, position):
    path = HISTORY_DIR / f"history_K{k}.parquet"
    parquet = pq.ParquetFile(path)

    rows = 0
    interactions = 0
    start = perf_counter()

    for batch in parquet.iter_batches(
        columns=["history"],
        batch_size=BATCH_SIZE,
    ):
        values = batch.column(0).to_pylist()
        rows += len(values)
        interactions += sum(len(x) for x in values if x is not None)
        del values

    elapsed = perf_counter() - start

    return {
        "K": k,
        "repeat": repeat,
        "position": position,
        "requests": rows,
        "history_interactions": interactions,
        "file_size_MB": path.stat().st_size / (1024 ** 2),
        "read_and_iteration_seconds": elapsed,
    }


def main():
    results = []

    for repeat in range(1, REPEATS + 1):
        # Alternate order to balance first-run/cache effects.
        order = K_VALUES if repeat % 2 == 1 else list(reversed(K_VALUES))
        print(f"\n--- Repeat {repeat}/{REPEATS} ---", flush=True)

        for position, k in enumerate(order, start=1):
            result = benchmark(k, repeat, position)
            results.append(result)
            print(
                f"K={k:>4} | {result['requests']:,} requests | "
                f"{result['history_interactions']:,} interactions | "
                f"{result['read_and_iteration_seconds']:.3f}s",
                flush=True,
            )

    raw = pd.DataFrame(results)
    raw_path = RESULTS_DIR / "runtime_benchmark_repeated.csv"
    raw.to_csv(raw_path, index=False, float_format="%.6f")

    summary = (
        raw.groupby("K")
        .agg(
            runs=("read_and_iteration_seconds", "count"),
            requests=("requests", "first"),
            history_interactions=("history_interactions", "first"),
            median_seconds=("read_and_iteration_seconds", "median"),
            min_seconds=("read_and_iteration_seconds", "min"),
            max_seconds=("read_and_iteration_seconds", "max"),
        )
        .reset_index()
    )

    summary["range_seconds"] = summary["max_seconds"] - summary["min_seconds"]
    summary_path = RESULTS_DIR / "runtime_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    print("\n=== REPEATED BENCHMARK SUMMARY ===")
    print(summary.to_string(
        index=False,
        formatters={
            "median_seconds": lambda x: f"{x:.3f}",
            "min_seconds": lambda x: f"{x:.3f}",
            "max_seconds": lambda x: f"{x:.3f}",
            "range_seconds": lambda x: f"{x:.3f}",
        },
    ))

    medians = summary.set_index("K")["median_seconds"]
    if {"100", "Full"}.issubset(medians.index):
        k100 = medians["100"]
        full = medians["Full"]
        print(f"\nMedian runtime reduction: {(1 - k100 / full):.2%}")

    print(f"\nSaved: {raw_path.relative_to(ROOT)}")
    print(f"Saved: {summary_path.relative_to(ROOT)}")
    print(
        "Caution: repeated local streaming benchmark; results may benefit "
        "from OS caching and do not directly measure serverless cost."
    )


if __name__ == "__main__":
    main()
