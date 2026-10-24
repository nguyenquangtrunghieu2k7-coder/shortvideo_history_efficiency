
"""Synthetic benchmark for Bloom Filter configurations.

This is a local, sequential simulation—not a real distributed or
serverless benchmark.
"""

from time import perf_counter

from src.bloom_pipeline.distributed_build import distributed_build
from src.bloom_pipeline.probe import probe_filter
from src.bloom_pipeline.calibration import theoretical_fpr


def run_benchmark(
    n: int = 10_000,
    num_workers: int = 4,
) -> list[dict]:
    # Ground-truth keys that exist in the build-side set.
    build_keys = [f"video_{i}" for i in range(n)]
    build_key_set = set(build_keys)

    # Half of the probe records match; half definitely do not.
    probe_records = (
        [{"video_id": key} for key in build_keys]
        + [
            {"video_id": f"absent_{i}"}
            for i in range(n)
        ]
    )

    configurations = [
        (bits_per_key, k)
        for bits_per_key in (8, 10, 12, 16)
        for k in (3, 5, 7)
    ]

    results = []

    for bits_per_key, k in configurations:
        m = bits_per_key * n

        start = perf_counter()
        _, global_filter = distributed_build(
            items=build_keys,
            num_workers=num_workers,
            m=m,
            k=k,
        )
        build_seconds = perf_counter() - start

        start = perf_counter()
        candidates, rejected = probe_filter(
            records=probe_records,
            bloom=global_filter,
            key_field="video_id",
        )
        probe_seconds = perf_counter() - start

        false_positives = sum(
            1
            for record in candidates
            if record["video_id"] not in build_key_set
        )

        false_negatives = sum(
            1
            for record in rejected
            if record["video_id"] in build_key_set
        )

        empirical_fpr = false_positives / n
        rejection_ratio = len(rejected) / len(probe_records)

        results.append({
            "bits_per_key": bits_per_key,
            "m": m,
            "k": k,
            "fill_ratio": global_filter.fill_ratio,
            "theoretical_fpr": theoretical_fpr(m, n, k),
            "empirical_fpr": empirical_fpr,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "rejection_ratio": rejection_ratio,
            "build_seconds": build_seconds,
            "probe_seconds": probe_seconds,
        })

    return results


def main() -> None:
    results = run_benchmark()

    print(
        f"{'bits/key':>8} {'k':>3} {'fill':>9} "
        f"{'theory FPR':>11} {'actual FPR':>11} "
        f"{'FP':>6} {'FN':>4} {'rejected':>10} "
        f"{'build ms':>10} {'probe ms':>10}"
    )

    for row in results:
        print(
            f"{row['bits_per_key']:>8} "
            f"{row['k']:>3} "
            f"{row['fill_ratio']:>9.4f} "
            f"{row['theoretical_fpr']:>11.4%} "
            f"{row['empirical_fpr']:>11.4%} "
            f"{row['false_positives']:>6} "
            f"{row['false_negatives']:>4} "
            f"{row['rejection_ratio']:>9.2%} "
            f"{row['build_seconds'] * 1000:>10.2f} "
            f"{row['probe_seconds'] * 1000:>10.2f}"
        )


if __name__ == "__main__":
    main()
