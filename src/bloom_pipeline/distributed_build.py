
"""Distributed-build simulation for the Bloom pipeline.

Each worker builds a local Bloom Filter from one partition.
The local filters are then merged into a global Bloom Filter.
"""

from collections.abc import Sequence

from src.bloom_pipeline.bloom_filter import BloomFilter
from src.bloom_pipeline.merge import merge_filters


def partition_items(
    items: Sequence[str],
    num_partitions: int,
) -> list[list[str]]:
    """Split items into balanced partitions while preserving order."""
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")

    partitions = [[] for _ in range(num_partitions)]

    for index, item in enumerate(items):
        partitions[index % num_partitions].append(item)

    return partitions


def build_local_filter(
    items: Sequence[str],
    m: int,
    k: int,
) -> BloomFilter:
    """Simulate one worker building a local Bloom Filter."""
    bloom = BloomFilter(m=m, k=k)

    for item in items:
        bloom.add(item)

    return bloom


def distributed_build(
    items: Sequence[str],
    num_workers: int,
    m: int,
    k: int,
) -> tuple[list[BloomFilter], BloomFilter]:
    """Build one filter per partition and merge them into a global filter."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    partitions = partition_items(items, num_workers)

    local_filters = [
        build_local_filter(partition, m=m, k=k)
        for partition in partitions
    ]

    global_filter = merge_filters(local_filters)

    return local_filters, global_filter
