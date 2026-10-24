
"""End-to-end Bloom Filter pipeline demo."""
from src.bloom_pipeline.calibration import calibrate_from_observation
from collections.abc import Sequence
from time import perf_counter

from src.bloom_pipeline.distributed_build import distributed_build
from src.bloom_pipeline.probe import probe_filter


def run_pipeline(
    build_keys: Sequence[str],
    probe_records: Sequence[dict],
    key_field: str = "video_id",
    num_workers: int = 3,
    m: int = 8192,
    k: int = 4,
) -> dict:
    """Build distributed local filters, merge them, and filter probe records."""
    start = perf_counter()

    local_filters, global_filter = distributed_build(
        items=build_keys,
        num_workers=num_workers,
        m=m,
        k=k,
    )

    build_finished = perf_counter()

    candidates, rejected = probe_filter(
        records=probe_records,
        bloom=global_filter,
        key_field=key_field,
    )

    finished = perf_counter()
    calibration = calibrate_from_observation(
    m=m,
    k=k,
    fill_ratio=global_filter.fill_ratio,
    target_fpr=0.01,
    )

    return {
        "num_workers": num_workers,
        "num_build_keys": len(build_keys),
        "num_probe_records": len(probe_records),
        "num_candidates": len(candidates),
        "num_rejected": len(rejected),
        "candidate_records": candidates,
        "rejected_records": rejected,
        "fill_ratio": global_filter.fill_ratio,
        "estimated_cardinality": global_filter.estimated_cardinality,
        "build_seconds": build_finished - start,
        "probe_seconds": finished - build_finished,
        "total_seconds": finished - start,
        "local_filter_fill_ratios": [
            bloom.fill_ratio for bloom in local_filters
        ],
        "calibration": calibration,
    }


def demo() -> None:
    """Run a small reproducible example."""
    build_keys = [
        "video_101",
        "video_202",
        "video_303",
        "video_404",
    ]

    probe_records = [
        {"video_id": "video_101", "user_id": "user_A"},
        {"video_id": "video_999", "user_id": "user_B"},
        {"video_id": "video_202", "user_id": "user_C"},
        {"video_id": "video_888", "user_id": "user_D"},
        {"video_id": "video_404", "user_id": "user_E"},
    ]

    result = run_pipeline(
        build_keys=build_keys,
        probe_records=probe_records,
        key_field="video_id",
        num_workers=3,
        m=8192,
        k=4,
    )

    print("=== Bloom Pipeline Demo ===")
    for key, value in result.items():
        if key not in {"candidate_records", "rejected_records"}:
            print(f"{key}: {value}")

    print("candidate_records:", result["candidate_records"])
    print("rejected_records:", result["rejected_records"])


if __name__ == "__main__":
    demo()
