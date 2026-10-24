from pathlib import Path
import argparse
import time

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "data/processed/phase7"
RESULTS_FILE = ROOT / "data/processed/phase8/baseline_results.csv"
OUTPUT_FILE = ROOT / "data/processed/phase8/runtime_benchmark.csv"


def benchmark(k: str, batch_size: int = 256) -> dict:
    path = HISTORY_DIR / f"history_K{k}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    file_bytes = path.stat().st_size
    parquet = pq.ParquetFile(path)

    rows = 0
    interactions = 0
    start = time.perf_counter()

    # Stream only the history column; never load the whole table at once.
    for batch in parquet.iter_batches(
        columns=["history"],
        batch_size=batch_size,
    ):
        values = batch.column(0).to_pylist()
        rows += len(values)
        interactions += sum(len(x) for x in values if x is not None)
        del values

    elapsed = time.perf_counter() - start

    return {
        "K": k,
        "requests": rows,
        "history_interactions": interactions,
        "mean_interactions_per_request": interactions / rows if rows else 0,
        "file_size_MB": file_bytes / (1024 ** 2),
        "batch_size": batch_size,
        "read_and_iteration_seconds": elapsed,
        "interactions_per_second": interactions / elapsed if elapsed else None,
        "measurement": "one streaming pass; local machine",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        nargs="+",
        default=["100", "Full"],
        help="History configurations to benchmark, e.g. --k 100 Full",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    rows = []
    for k in args.k:
        print(f"\nBenchmarking K={k} ...", flush=True)
        result = benchmark(k, args.batch_size)
        rows.append(result)
        print(
            f"Requests: {result['requests']:,} | "
            f"Interactions: {result['history_interactions']:,} | "
            f"Time: {result['read_and_iteration_seconds']:.3f}s",
            flush=True,
        )

    out = pd.DataFrame(rows)

    # Attach existing baseline AUC; no model inference or retraining.
    if RESULTS_FILE.exists():
        baseline = pd.read_csv(RESULTS_FILE)
        baseline["K"] = baseline["K"].astype(str).str.strip()
        out["K"] = out["K"].astype(str)
        out = out.merge(
            baseline[["K", "best_val_auc", "test_auc"]],
            on="K",
            how="left",
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    if len(out) == 2:
        indexed = out.set_index("K")
        if {"100", "Full"}.issubset(indexed.index):
            full_n = indexed.loc["Full", "history_interactions"]
            k100_n = indexed.loc["100", "history_interactions"]
            full_t = indexed.loc["Full", "read_and_iteration_seconds"]
            k100_t = indexed.loc["100", "read_and_iteration_seconds"]

            print("\n=== COMPARISON ===")
            print(f"Interaction reduction: {1 - k100_n / full_n:.2%}")
            print(f"K=100 runtime: {k100_t:.3f}s")
            print(f"Full runtime:   {full_t:.3f}s")
            print(
                "Runtime difference: "
                f"{(1 - k100_t / full_t):.2%} "
                "(positive means K=100 was faster in this run)"
            )

    print(f"\nSaved: {OUTPUT_FILE.relative_to(ROOT)}")
    print("Note: one local streaming pass, not a serverless cost measurement.")


if __name__ == "__main__":
    main()
