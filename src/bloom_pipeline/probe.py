
"""Probe-side filtering before a join or retrieval step."""

from collections.abc import Iterable

from src.bloom_pipeline.bloom_filter import BloomFilter


def probe_filter(
    records: Iterable[dict],
    bloom: BloomFilter,
    key_field: str,
) -> tuple[list[dict], list[dict]]:
    """Split records into possible matches and definite non-matches.

    Returns:
        candidates: Records whose keys may exist in the Bloom Filter.
        rejected: Records whose keys are definitely absent.
    """
    candidates = []
    rejected = []

    for record in records:
        if key_field not in record:
            raise KeyError(f"Missing key field: {key_field}")

        key = str(record[key_field])

        if bloom.might_contain(key):
            candidates.append(record)
        else:
            rejected.append(record)

    return candidates, rejected
