
import math
import time
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EVENTS_PATH = Path("data/processed/evaluation_events.parquet")
HISTORY_PATH = Path("data/processed/evaluation_history.parquet")
OUTPUT_PATH = Path("reports/bloom_join_benchmark.csv")

TARGET_FPR = 0.01
MASK64 = (1 << 64) - 1


# --------------------------------------------------
# Bloom Filter for integer video IDs
# Uses double hashing to generate k bit positions.
# --------------------------------------------------

def mix64(value: int) -> int:
    """Deterministic 64-bit integer mixing."""
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


class BloomFilter:
    def __init__(self, n: int, target_fpr: float):
        if n <= 0:
            raise ValueError("n must be positive")
        if not 0 < target_fpr < 1:
            raise ValueError("target_fpr must be between 0 and 1")

        self.n = n
        self.target_fpr = target_fpr

        self.m = math.ceil(
            -n * math.log(target_fpr) / (math.log(2) ** 2)
        )
        self.k = max(1, round((self.m / n) * math.log(2)))

        self.bits = bytearray((self.m + 7) // 8)

    def _positions(self, value: int):
        value = int(value)
        h1 = mix64(value)
        h2 = mix64(value ^ 0xD6E8FEB86659FD93) | 1

        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, value: int):
        for position in self._positions(value):
            self.bits[position >> 3] |= 1 << (position & 7)

    def might_contain(self, value: int) -> bool:
        for position in self._positions(value):
            if not (
                self.bits[position >> 3]
                & (1 << (position & 7))
            ):
                return False

        return True


# --------------------------------------------------
# Load data and construct the build-side set
# --------------------------------------------------

def main():
    print("Loading data...")

    events = pd.read_parquet(
        EVENTS_PATH,
        columns=["video_id", "long_view"],
    )
    history = pd.read_parquet(
        HISTORY_PATH,
        columns=["history_full"],
    )

    build_ids = set(
        int(value)
        for value in events.loc[
            events["long_view"] == 1, "video_id"
        ].dropna()
    )

    print(f"Build-side unique IDs: {len(build_ids):,}")
    print(f"Evaluation requests: {len(history):,}")

    # --------------------------------------------------
    # 1. Exact baseline
    # --------------------------------------------------

    print("\n[1/3] Running exact baseline...")

    exact_matches = 0
    probe_records = 0

    start = time.perf_counter()

    for items in history["history_full"]:
        for value in items:
            video_id = int(value)
            probe_records += 1

            if video_id in build_ids:
                exact_matches += 1

    exact_baseline_s = time.perf_counter() - start
    exact_non_matches = probe_records - exact_matches

    print(f"Probe records: {probe_records:,}")
    print(f"Exact matches: {exact_matches:,}")
    print(f"Exact baseline: {exact_baseline_s:.3f}s")

    # --------------------------------------------------
    # 2. Build Bloom Filter
    # --------------------------------------------------

    print("\n[2/3] Building Bloom Filter...")

    bloom = BloomFilter(len(build_ids), TARGET_FPR)

    start = time.perf_counter()

    for video_id in build_ids:
        bloom.add(video_id)

    bloom_build_s = time.perf_counter() - start

    print(f"Target FPR: {TARGET_FPR:.2%}")
    print(f"m (bits): {bloom.m:,}")
    print(f"k (hashes): {bloom.k}")
    print(f"Bit array: {len(bloom.bits):,} bytes")
    print(f"Build time: {bloom_build_s:.3f}s")

    # --------------------------------------------------
    # 3. Bloom probe and exact verification
    # --------------------------------------------------

    print("\n[3/3] Probing history records...")

    candidates = []

    start = time.perf_counter()

    for items in history["history_full"]:
        for value in items:
            video_id = int(value)

            if bloom.might_contain(video_id):
                candidates.append(video_id)

    bloom_probe_s = time.perf_counter() - start

    # Bloom positives must be checked exactly.
    start = time.perf_counter()

    verified_matches = sum(
        video_id in build_ids
        for video_id in candidates
    )

    exact_verification_s = time.perf_counter() - start

    false_positives = len(candidates) - verified_matches
    false_negatives = exact_matches - verified_matches
    records_rejected = probe_records - len(candidates)

    empirical_fpr = (
        false_positives / exact_non_matches
        if exact_non_matches
        else 0.0
    )

    rejection_ratio = records_rejected / probe_records

    bloom_total_s = (
        bloom_build_s + bloom_probe_s + exact_verification_s
    )

    speedup = (
        exact_baseline_s / bloom_total_s
        if bloom_total_s > 0
        else 0.0
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n========== RESULTS ==========")
    print(f"Probe records:          {probe_records:,}")
    print(f"Exact matches:          {exact_matches:,}")
    print(f"Bloom candidates:       {len(candidates):,}")
    print(f"Verified matches:       {verified_matches:,}")
    print(f"False positives:        {false_positives:,}")
    print(f"False negatives:        {false_negatives:,}")
    print(f"Empirical FPR:          {empirical_fpr:.6%}")
    print(f"Records rejected:       {records_rejected:,}")
    print(f"Rejection ratio:        {rejection_ratio:.4%}")
    print(f"Exact baseline:         {exact_baseline_s:.3f}s")
    print(f"Bloom build:            {bloom_build_s:.3f}s")
    print(f"Bloom probe:            {bloom_probe_s:.3f}s")
    print(f"Exact verification:     {exact_verification_s:.3f}s")
    print(f"Bloom pipeline total:   {bloom_total_s:.3f}s")
    print(f"Baseline / Bloom:       {speedup:.3f}x")

    # Correctness checks
    assert false_negatives == 0, (
        "Correctness failure: Bloom Filter produced false negatives"
    )
    assert verified_matches == exact_matches, (
        "Correctness failure: final join results differ from baseline"
    )

    results = {
        "build_unique_ids": len(build_ids),
        "probe_records": probe_records,
        "exact_matches": exact_matches,
        "exact_non_matches": exact_non_matches,
        "match_ratio": exact_matches / probe_records,
        "target_fpr": TARGET_FPR,
        "m_bits": bloom.m,
        "k_hashes": bloom.k,
        "filter_size_bytes": len(bloom.bits),
        "bloom_candidates": len(candidates),
        "verified_matches": verified_matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "empirical_fpr": empirical_fpr,
        "records_rejected": records_rejected,
        "rejection_ratio": rejection_ratio,
        "exact_baseline_s": exact_baseline_s,
        "bloom_build_s": bloom_build_s,
        "bloom_probe_s": bloom_probe_s,
        "exact_verification_s": exact_verification_s,
        "bloom_total_s": bloom_total_s,
        "speedup": speedup,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
