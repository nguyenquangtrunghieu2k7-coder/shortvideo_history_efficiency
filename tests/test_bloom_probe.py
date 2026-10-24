
import pytest

from src.bloom_pipeline.bloom_filter import BloomFilter
from src.bloom_pipeline.probe import probe_filter


def test_probe_rejects_definitely_absent_keys():
    bloom = BloomFilter(m=8192, k=4)

    metadata_keys = ["video_101", "video_202", "video_303"]
    for key in metadata_keys:
        bloom.add(key)

    records = [
        {"video_id": "video_101", "user_id": "user_A"},
        {"video_id": "definitely_absent_1", "user_id": "user_B"},
        {"video_id": "video_202", "user_id": "user_C"},
        {"video_id": "definitely_absent_2", "user_id": "user_D"},
    ]

    candidates, rejected = probe_filter(
        records,
        bloom,
        key_field="video_id",
    )

    assert {r["video_id"] for r in candidates} >= {
        "video_101",
        "video_202",
    }
    assert {r["video_id"] for r in rejected} == {
        "definitely_absent_1",
        "definitely_absent_2",
    }


def test_probe_raises_for_missing_key_field():
    bloom = BloomFilter(m=1024, k=3)

    with pytest.raises(KeyError):
        probe_filter(
            [{"user_id": "user_A"}],
            bloom,
            key_field="video_id",
        )
