
"""Basic Bloom Filter implementation for the Project 1 prototype."""

from __future__ import annotations

import hashlib
import math


class BloomFilter:
    """A Bloom Filter with a fixed bit-array size and number of hash functions."""

    def __init__(self, m: int, k: int) -> None:
        if m <= 0:
            raise ValueError("m must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        self.m = m
        self.k = k
        self._bits = bytearray((m + 7) // 8)
        self._set_bits = 0
        self._insertions = 0

    def _indices(self, item: str) -> list[int]:
        """Generate k deterministic indices using double hashing."""
        encoded = item.encode("utf-8")

        digest = hashlib.sha256(encoded).digest()
        h1 = int.from_bytes(digest[:16], "big")
        h2 = int.from_bytes(digest[16:], "big")

        # Ensure the second hash can advance through the bit array.
        h2 |= 1

        return [
            (h1 + i * h2) % self.m
            for i in range(self.k)
        ]

    def add(self, item: str) -> None:
        """Insert an item into the filter."""
        self._insertions += 1

        for index in self._indices(item):
            byte_index = index // 8
            bit_mask = 1 << (index % 8)

            if not self._bits[byte_index] & bit_mask:
                self._bits[byte_index] |= bit_mask
                self._set_bits += 1

    def might_contain(self, item: str) -> bool:
        """Return False if definitely absent; True if possibly present."""
        for index in self._indices(item):
            byte_index = index // 8
            bit_mask = 1 << (index % 8)

            if not self._bits[byte_index] & bit_mask:
                return False

        return True

    @property
    def fill_ratio(self) -> float:
        """Fraction of the m logical bits that are set."""
        return self._set_bits / self.m

    @property
    def estimated_cardinality(self) -> float:
        """Estimate distinct inserted items from the fill ratio."""
        if self._set_bits == self.m:
            return math.inf
        if self._set_bits == 0:
            return 0.0

        return -(self.m / self.k) * math.log(1 - self.fill_ratio)

    @property
    def bit_array(self) -> bytes:
        """Return an immutable snapshot of the underlying bit array."""
        return bytes(self._bits)
