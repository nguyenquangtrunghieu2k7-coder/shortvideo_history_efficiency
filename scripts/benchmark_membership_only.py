
import random
import statistics
import time

from benchmark_bloom_join import BloomFilter


SAMPLE_SIZES = [10_000, 100_000, 500_000]
REPEATS = 5
BUILD_SIZE = 10_000
TARGET_FPR = 0.01
SEED = 42


def timed_membership(items, contains):
    """Đo thời gian kiểm tra membership trên cùng một danh sách."""
    start = time.perf_counter()
    results = [contains(item) for item in items]
    elapsed = time.perf_counter() - start
    return results, elapsed


def main():
    rng = random.Random(SEED)

    # Build keys: 10.000 ID duy nhất.
    build_ids = set(range(1_000_000, 1_000_000 + BUILD_SIZE))

    bloom = BloomFilter(
        n=BUILD_SIZE,
        false_positive_rate=TARGET_FPR,
    )

    for key in build_ids:
        bloom.add(key)

    print(f"Build keys: {len(build_ids):,}")
    print(f"Bloom bits: {bloom.m:,}")
    print(f"Bloom hashes: {bloom.k}")
    print(f"Bloom fill ratio: {bloom.fill_ratio():.4%}")
    print()

    for sample_size in SAMPLE_SIZES:
        # 50% present, 50% absent.
        present_count = sample_size // 2
        absent_count = sample_size - present_count

        present = rng.sample(list(build_ids), present_count)
        absent = [
            2_000_000 + i
            for i in range(absent_count)
        ]

        probes = present + absent
        rng.shuffle(probes)

        # Warm-up, không tính vào kết quả.
        for key in probes[:min(1_000, len(probes))]:
            key in build_ids
            bloom.might_contain(key)

        set_times = []
        bloom_times = []
        set_results = None
        bloom_results = None

        for _ in range(REPEATS):
            # Cùng probes, chỉ đổi cách kiểm tra.
            current_set_results, set_elapsed = timed_membership(
                probes,
                lambda key: key in build_ids,
            )

            current_bloom_results, bloom_elapsed = timed_membership(
                probes,
                bloom.might_contain,
            )

            set_times.append(set_elapsed)
            bloom_times.append(bloom_elapsed)

            set_results = current_set_results
            bloom_results = current_bloom_results

        exact_matches = sum(set_results)
        bloom_candidates = sum(bloom_results)

        false_positives = sum(
            bloom_result and not exact_result
            for bloom_result, exact_result
            in zip(bloom_results, set_results)
        )

        false_negatives = sum(
            not bloom_result and exact_result
            for bloom_result, exact_result
            in zip(bloom_results, set_results)
        )

        absent_count_actual = sample_size - exact_matches
        empirical_fpr = (
            false_positives / absent_count_actual
            if absent_count_actual else 0.0
        )

        median_set = statistics.median(set_times)
        median_bloom = statistics.median(bloom_times)

        print(f"=== Sample: {sample_size:,} probes ===")
        print(f"Exact matches:       {exact_matches:,}")
        print(f"Bloom candidates:    {bloom_candidates:,}")
        print(f"False positives:     {false_positives:,}")
        print(f"False negatives:     {false_negatives:,}")
        print(f"Empirical FPR:       {empirical_fpr:.4%}")
        print(f"Set median:          {median_set:.6f} s")
        print(f"Bloom median:        {median_bloom:.6f} s")

        if median_bloom > 0:
            print(
                "Set / Bloom speed:   "
                f"{median_bloom / median_set:.2f}x"
            )

        print()


if __name__ == "__main__":
    main()
