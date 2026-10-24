
import pytest

from src.bloom_pipeline.bloom_filter import BloomFilter
from src.bloom_pipeline.merge import merge_filters


def test_merged_filter_contains_items_from_all_workers():
    partitions = [
        ["video_1", "video_2"],
        ["video_3", "video_4"],
        ["video_5", "video_6"],
    ]

    local_filters = []

    for partition in partitions:
        bloom = BloomFilter(m=2048, k=4)

        for item in partition:
            bloom.add(item)

        local_filters.append(bloom)

    merged = merge_filters(local_filters)

    for partition in partitions:
        for item in partition:
            assert merged.might_contain(item)


def test_merge_requires_compatible_parameters():
    first = BloomFilter(m=1024, k=3)
    second = BloomFilter(m=2048, k=3)

    with pytest.raises(ValueError):
        merge_filters([first, second])


def test_merge_rejects_empty_input():
    with pytest.raises(ValueError):
        merge_filters([])
