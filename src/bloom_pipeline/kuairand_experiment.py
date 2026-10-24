
"""Bloom membership-filter experiment on processed KuaiRand data.

This is a local prototype, not a reproduction of BLOOM-FaaS.
It tests whether event video IDs occur in the union of history video IDs.
"""

from time import perf_counter

import pandas as pd

from src.bloom_pipeline.bloom_filter import BloomFilter
from src.bloom_pipeline.calibration import theoretical_fpr


EVENTS_PATH = "data/processed/evaluation_events.parquet"
HISTORY_PATH = "data/processed/evaluation_history.parquet"


def run_experiment(
    bits_per_key: int = 12,
    k: int = 5,
) -> dict:
    events = pd.read_parquet(EVENTS_PATH, columns=["video_id"])
    history = pd.read_parquet(HISTORY_PATH, columns=["history_full"])

    # Build the exact set of distinct video IDs in all histories.
    history_video_ids = set()
    for sequence in history["history_full"]:
        history_video_ids.update(int(video_id) for video_id in sequence)

    if not history_video_ids:
        raise ValueError("No video IDs found in history data")

    # Use string keys consistently with the current Bloom Filter API.
    build_keys = [str(video_id) for video_id in history_video_ids]
    exact_set = set(build_keys)

    m = bits_per_key * len(exact_set)
    bloom = BloomFilter(m=m, k=k)

    build_start = perf_counter()
    for key in build_keys:
        bloom.add(key)
    build_seconds = perf_counter() - build_start

    probe_keys = [str(int(video_id)) for video_id in events["video_id"]]

    probe_start = perf_counter()
    predicted = [bloom.might_contain(key) for key in probe_keys]
    probe_seconds = perf_counter() - probe_start

    actual = [key in exact_set for key in probe_keys]

    true_positives = sum(p and a for p, a in zip(predicted, actual))
    false_positives = sum(p and not a for p, a in zip(predicted, actual))
    true_negatives = sum(not p and not a for p, a in zip(predicted, actual))
    false_negatives = sum(not p and a for p, a in zip(predicted, actual))

    actual_absent = sum(not value for value in actual)
    empirical_fpr = (
        false_positives / actual_absent
        if actual_absent
        else 0.0
    )

    return {
        "num_events": len(events),
        "distinct_history_video_ids": len(exact_set),
        "bits_per_key": bits_per_key,
        "m_bits": m,
        "k": k,
        "fill_ratio": bloom.fill_ratio,
        "theoretical_fpr": theoretical_fpr(m, len(exact_set), k),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "actual_absent_events": actual_absent,
        "empirical_fpr": empirical_fpr,
        "rejected_events": true_negatives,
        "rejection_ratio": true_negatives / len(events),
        "build_seconds": build_seconds,
        "probe_seconds": probe_seconds,
    }


def main() -> None:
    result = run_experiment()

    print("=== KuaiRand Bloom Membership Experiment ===")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
