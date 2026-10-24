from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data/processed/phase8"

BASELINE_FILE = RESULTS_DIR / "baseline_results.csv"
BENCHMARK_FILE = RESULTS_DIR / "runtime_benchmark.csv"
OUTPUT_FILE = RESULTS_DIR / "tradeoff_summary.csv"


def main():
    baseline = pd.read_csv(BASELINE_FILE)
    benchmark = pd.read_csv(BENCHMARK_FILE)

    baseline["K"] = baseline["K"].astype(str).str.strip()
    benchmark["K"] = benchmark["K"].astype(str).str.strip()

    # Keep only the columns needed for the trade-off comparison.
    baseline_cols = [
        c for c in ["K", "best_val_auc", "test_auc"]
        if c in baseline.columns
    ]
    benchmark_cols = [
        c for c in [
            "K",
            "requests",
            "history_interactions",
            "mean_interactions_per_request",
            "file_size_MB",
            "read_and_iteration_seconds",
            "interactions_per_second",
        ]
        if c in benchmark.columns
    ]

    table = baseline[baseline_cols].merge(
        benchmark[benchmark_cols],
        on="K",
        how="inner",
        validate="one_to_one",
    )

    full = table[table["K"].str.lower() == "full"]
    if full.empty:
        raise ValueError("Không tìm thấy cấu hình Full trong bảng kết quả.")

    full = full.iloc[0]
    table["interaction_reduction_pct"] = (
        1 - table["history_interactions"] / full["history_interactions"]
    ) * 100

    table["runtime_reduction_pct"] = (
        1 - table["read_and_iteration_seconds"]
        / full["read_and_iteration_seconds"]
    ) * 100

    table["test_auc_gap_vs_full"] = full["test_auc"] - table["test_auc"]

    # Put K=Full last; sort other configurations by increasing K.
    table["_sort"] = table["K"].map(
        lambda k: float("inf") if k.lower() == "full" else float(k)
    )
    table = table.sort_values("_sort").drop(columns="_sort")

    # Round only for readability; keep the underlying values interpretable.
    table.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")

    display_cols = [
        "K",
        "best_val_auc",
        "test_auc",
        "history_interactions",
        "mean_interactions_per_request",
        "read_and_iteration_seconds",
        "interaction_reduction_pct",
        "runtime_reduction_pct",
        "test_auc_gap_vs_full",
    ]
    display_cols = [c for c in display_cols if c in table.columns]

    print("\n=== HISTORY / QUALITY / RUNTIME TRADE-OFF ===")
    print(table[display_cols].to_string(
        index=False,
        formatters={
            "best_val_auc": lambda x: f"{x:.6f}",
            "test_auc": lambda x: f"{x:.6f}",
            "mean_interactions_per_request": lambda x: f"{x:.2f}",
            "read_and_iteration_seconds": lambda x: f"{x:.3f}",
            "interaction_reduction_pct": lambda x: f"{x:.2f}%",
            "runtime_reduction_pct": lambda x: f"{x:.2f}%",
            "test_auc_gap_vs_full": lambda x: f"{x:.6f}",
        },
    ))

    print(f"\nSaved: {OUTPUT_FILE.relative_to(ROOT)}")
    print(
        "Caution: runtime is from a single local streaming benchmark pass; "
        "it is not a direct serverless latency, energy, or cost measurement."
    )


if __name__ == "__main__":
    main()
