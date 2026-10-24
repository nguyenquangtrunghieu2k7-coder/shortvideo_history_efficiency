
import pytest

from src.bloom_pipeline.distributed_build import (
    distributed_build,
    partition_items,
)


def test_partition_items_distributes_all_items():
    items = [f"video_{i}" for i in range(10)]

    partitions = partition_items(items, num_partitions=3)

    flattened = [item for partition in partitions for item in partition]

    assert len(partitions) == 3
    assert sorted(flattened) == sorted(items)
    assert len(flattened) == len(set(flattened))
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1


def test_distributed_build_global_filter_contains_all_items():
    items = [f"video_{i}" for i in range(100)]

    local_filters, global_filter = distributed_build(
        items=items,
        num_workers=4,
        m=4096,
        k=5,
    )

    assert len(local_filters) == 4

    for item in items:
        assert global_filter.might_contain(item)


def test_distributed_build_rejects_invalid_worker_count():
    with pytest.raises(ValueError):
        distributed_build(
            items=["video_1"],
            num_workers=0,
            m=1024,
            k=3,
        )
