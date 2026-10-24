
"""Bloom Filter sizing and calibration utilities."""

from __future__ import annotations

import math


def optimal_bit_count(n: int, target_fpr: float) -> int:
    """Calculate the theoretical bit count for n items and target FPR."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")

    m = -n * math.log(target_fpr) / (math.log(2) ** 2)
    return math.ceil(m)


def optimal_hash_count(m: int, n: int) -> int:
    """Calculate the approximate optimal number of hash functions."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive")

    return max(1, round((m / n) * math.log(2)))


def theoretical_fpr(m: int, n: int, k: int) -> float:
    """Estimate the false-positive probability under the standard model."""
    if m <= 0 or n < 0 or k <= 0:
        raise ValueError("Require m > 0, n >= 0, and k > 0")

    if n == 0:
        return 0.0

    return (1 - math.exp(-k * n / m)) ** k


def estimate_cardinality(m: int, k: int, fill_ratio: float) -> float:
    """Estimate inserted cardinality from the observed fill ratio."""
    if m <= 0 or k <= 0:
        raise ValueError("m and k must be positive")
    if not 0 <= fill_ratio <= 1:
        raise ValueError("fill_ratio must be between 0 and 1")

    if fill_ratio == 0:
        return 0.0
    if fill_ratio == 1:
        return math.inf

    return -(m / k) * math.log1p(-fill_ratio)


def calibrate_from_observation(
    m: int,
    k: int,
    fill_ratio: float,
    target_fpr: float = 0.01,
) -> dict:
    """Recommend settings for a future filter based on an observation."""
    estimated_n = estimate_cardinality(m, k, fill_ratio)

    if not math.isfinite(estimated_n) or estimated_n <= 0:
        raise ValueError(
            "Cannot calibrate from an empty or saturated filter"
        )

    # Cardinality must be an integer for the sizing formula.
    n_for_sizing = max(1, math.ceil(estimated_n))

    recommended_m = optimal_bit_count(n_for_sizing, target_fpr)
    recommended_k = optimal_hash_count(recommended_m, n_for_sizing)

    return {
        "observed_m": m,
        "observed_k": k,
        "observed_fill_ratio": fill_ratio,
        "estimated_cardinality": estimated_n,
        "target_fpr": target_fpr,
        "recommended_m": recommended_m,
        "recommended_k": recommended_k,
        "predicted_fpr": theoretical_fpr(
            recommended_m,
            n_for_sizing,
            recommended_k,
        ),
    }
