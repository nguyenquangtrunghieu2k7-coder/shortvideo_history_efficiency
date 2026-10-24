
"""Fan-in and bitwise OR merge for compatible Bloom Filters."""

from src.bloom_pipeline.bloom_filter import BloomFilter


def merge_filters(filters: list[BloomFilter]) -> BloomFilter:
    """Merge filters that share the same m and k parameters."""
    if not filters:
        raise ValueError("At least one Bloom Filter is required")

    first = filters[0]

    for bloom in filters[1:]:
        if bloom.m != first.m or bloom.k != first.k:
            raise ValueError(
                "All filters must have identical m and k values"
            )

    merged = BloomFilter(m=first.m, k=first.k)

    # OR the underlying bytes. This is equivalent to bitwise OR
    # over the logical bit arrays when m and k are identical.
    merged._bits = bytearray(first.bit_array)

    for bloom in filters[1:]:
        for i, value in enumerate(bloom.bit_array):
            merged._bits[i] |= value

    # Recount set bits because multiple filters may share positions.
    merged._set_bits = sum(byte.bit_count() for byte in merged._bits)
    merged._insertions = sum(bloom._insertions for bloom in filters)

    return merged
