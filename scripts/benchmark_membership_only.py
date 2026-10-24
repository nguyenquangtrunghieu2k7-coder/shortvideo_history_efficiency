
"""Compare exact set membership with the shared Bloom Filter implementation.

This is a local, sequential microbenchmark, not a distributed/serverless test.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bloom_pipeline.bloom_filter import BloomFilter
from bloom_pipeline.calibration import (
    optimal_bit_count,
    optimal_hash_count,
    theoretical_fpr,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"


def normalize_id(value) -> str:
    """Normalize IDs consistently across the build set and history."""
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    return str(value)


def iter_probe_ids(history_frame, limit=None):
    """Yield history video IDs without flattening the entire dataset in memory."""
    count = 0

    for history in history_frame["history_full"]:
        if history is None:
            continue

        try:
            iterator = iter(history)
        except TypeError:
            continue

        for video_id in iterator:
            yield normalize_id(video_id)
            count += 1

            if limit is not None and count >= limit:
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-probes",
        type=int,
        default=None,
        help="Maximum number of history interactions to probe; default: all",
    )
    args = parser.parse_args()

    if args.max_probes is not None and args.max_probes <= 0:
        parser.error("--max-probes must be positive")

    events_path = DATA_DIR / "evaluation_events.parquet"
    history_path = DATA_DIR / "evaluation_history.parquet"

    events = pd.read_parquet(events_path)
    history = pd.read_parquet(history_path)

    build_ids = {
        normalize_id(video_id)
        for video_id in events.loc[
            events["long_view"] == 1, "video_id"
        ].dropna()
    }

    n = len(build_ids)
    if n == 0:
        raise ValueError("The build-side set is empty.")

    target_fpr = 0.01
    m = optimal_bit_count(n, target_fpr)
    k = optimal_hash_count(m, n)

    # Build the shared Bloom Filter implementation.
    bloom = BloomFilter(m=m, k=k)

    build_start = perf_counter()
    for video_id in build_ids:
        bloom.add(video_id)
    bloom_build_seconds = perf_counter() - build_start

    # Pass 1: exact membership lookup.
    exact_matches = 0
    probe_count = 0

    exact_start = perf_counter()
    for video_id in iter_probe_ids(history, args.max_probes):
        probe_count += 1
        if video_id in build_ids:
            exact_matches += 1
    exact_seconds = perf_counter() - exact_start

    # Pass 2: Bloom membership lookup only.
    # Keep candidate IDs so exact verification happens outside the timed loop.
    candidates = []

    bloom_start = perf_counter()
    for video_id in iter_probe_ids(history, args.max_probes):
        if bloom.might_contain(video_id):
            candidates.append(video_id)
    bloom_probe_seconds = perf_counter() - bloom_start

    # Verify candidates separately using the exact set.
    verification_start = perf_counter()
    verified_matches = sum(
        1 for video_id in candidates if video_id in build_ids
    )
    verification_seconds = perf_counter() - verification_start

    false_positives = len(candidates) - verified_matches
    false_negatives = exact_matches - verified_matches
    exact_nonmatches = probe_count - exact_matches
    rejected = probe_count - len(candidates)

    empirical_fpr = (
        false_positives / exact_nonmatches
        if exact_nonmatches
        else 0.0
    )
    rejection_ratio = rejected / probe_count if probe_count else 0.0
    bloom_pipeline_seconds = bloom_probe_seconds + verification_seconds

    result = {
        "probe_limit": args.max_probes if args.max_probes is not None else "all",
        "probe_records": probe_count,
        "build_unique_ids": n,
        "target_fpr": target_fpr,
        "m_bits": m,
        "k_hashes": k,
        "bit_array_bytes": len(bloom.bit_array),
        "fill_ratio": bloom.fill_ratio,
        "theoretical_fpr": theoretical_fpr(m, n, k),
        "exact_matches": exact_matches,
        "bloom_candidates": len(candidates),
        "verified_matches": verified_matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "empirical_fpr": empirical_fpr,
        "rejected_records": rejected,
        "rejection_ratio": rejection_ratio,
        "bloom_build_seconds": bloom_build_seconds,
        "exact_lookup_seconds": exact_seconds,
        "bloom_lookup_seconds": bloom_probe_seconds,
        "candidate_verification_seconds": verification_seconds,
        "bloom_pipeline_seconds": bloom_pipeline_seconds,
        "exact_over_bloom_lookup_speed_ratio": (
            exact_seconds / bloom_probe_seconds
            if bloom_probe_seconds
            else 0.0
        ),
    }

    print("\n=== Bloom membership-only benchmark ===")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "bloom_membership_benchmark.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
