
from src.bloom_pipeline.bloom_filter import BloomFilter


def test_inserted_items_are_found():
    bloom = BloomFilter(m=1024, k=4)

    items = ["video_101", "video_202", "video_303"]
    for item in items:
        bloom.add(item)

    assert all(bloom.might_contain(item) for item in items)


def test_empty_filter_rejects_items():
    bloom = BloomFilter(m=1024, k=4)

    assert not bloom.might_contain("video_101")


def test_fill_ratio_is_valid():
    bloom = BloomFilter(m=1024, k=4)

    for i in range(100):
        bloom.add(f"video_{i}")

    assert 0 < bloom.fill_ratio <= 1
    assert bloom.estimated_cardinality > 0
