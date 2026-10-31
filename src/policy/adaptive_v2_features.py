"""Leakage-safe request-time features for Adaptive History Selector V2.

History order is assumed to be oldest -> newest, matching the Phase 7
temporal-history contract. Features use only the request's available history.
"""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    # Global composition
    "history_length",
    "unique_count",
    "unique_ratio",
    "repeat_ratio",
    "normalized_entropy",
    "effective_unique_count",
    "max_item_share",
    "top5_item_share",
    "singleton_item_ratio",
    "repeated_item_ratio",
    # Local composition at the recent end
    "unique_last_5",
    "unique_last_20",
    "unique_last_50",
    "unique_last_100",
    "recent_unique_rate_20",
    "recent_unique_rate_50",
    "recent_unique_rate_100",
    # Temporal change / novelty profile
    "unique_first_100",
    "head_tail_overlap_100",
    "new_unique_in_last_100",
    "novelty_rate_last_100",
    "unique_growth_20_to_100",
    "unique_growth_50_to_100",
    "adjacent_repeat_ratio",
    "max_adjacent_repeat_run",
]


def _unique_count(values: list[int]) -> int:
    return len(set(values))


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _max_adjacent_repeat_run(values: list[int]) -> int:
    if not values:
        return 0
    longest = 1
    current = 1
    for previous, current_value in zip(values, values[1:]):
        if current_value == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _features_for_one(history) -> dict[str, float]:
    values = [int(x) for x in history]
    n = len(values)
    counts = Counter(values)
    unique_n = len(counts)
    if n:
        probabilities = np.asarray(list(counts.values()), dtype=np.float64) / n
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        normalized_entropy = entropy / math.log(unique_n) if unique_n > 1 else 0.0
        effective_unique = float(math.exp(entropy))
        max_share = max(counts.values()) / n
        top5_share = sum(sorted(counts.values(), reverse=True)[:5]) / n
        singleton_ratio = sum(v == 1 for v in counts.values()) / unique_n
        repeated_item_ratio = sum(v > 1 for v in counts.values()) / unique_n
        adjacent_repeat_ratio = sum(
            left == right for left, right in zip(values, values[1:])
        ) / max(n - 1, 1)
    else:
        normalized_entropy = 0.0
        effective_unique = 0.0
        max_share = 0.0
        top5_share = 0.0
        singleton_ratio = 0.0
        repeated_item_ratio = 0.0
        adjacent_repeat_ratio = 0.0

    last5 = values[-5:]
    last20 = values[-20:]
    last50 = values[-50:]
    last100 = values[-100:]
    first100 = values[:100]

    u5 = _unique_count(last5)
    u20 = _unique_count(last20)
    u50 = _unique_count(last50)
    u100 = _unique_count(last100)
    first_u100 = _unique_count(first100)

    # Novelty in the most recent 100 interactions relative to older history.
    older = values[:-100] if n > 100 else []
    older_set = set(older)
    new_in_last100 = sum(1 for item in set(last100) if item not in older_set)

    head_set = set(first100)
    tail_set = set(last100)
    union_size = len(head_set | tail_set)
    overlap = _safe_rate(len(head_set & tail_set), union_size)

    return {
        "history_length": n,
        "unique_count": unique_n,
        "unique_ratio": _safe_rate(unique_n, n),
        "repeat_ratio": _safe_rate(n - unique_n, n),
        "normalized_entropy": normalized_entropy,
        "effective_unique_count": effective_unique,
        "max_item_share": max_share,
        "top5_item_share": top5_share,
        "singleton_item_ratio": singleton_ratio,
        "repeated_item_ratio": repeated_item_ratio,
        "unique_last_5": _safe_rate(u5, min(n, 5)),
        "unique_last_20": _safe_rate(u20, min(n, 20)),
        "unique_last_50": _safe_rate(u50, min(n, 50)),
        "unique_last_100": _safe_rate(u100, min(n, 100)),
        "recent_unique_rate_20": _safe_rate(u20, min(n, 20)),
        "recent_unique_rate_50": _safe_rate(u50, min(n, 50)),
        "recent_unique_rate_100": _safe_rate(u100, min(n, 100)),
        "unique_first_100": first_u100,
        "head_tail_overlap_100": overlap,
        "new_unique_in_last_100": new_in_last100,
        "novelty_rate_last_100": _safe_rate(new_in_last100, u100),
        "unique_growth_20_to_100": max(u100 - u20, 0),
        "unique_growth_50_to_100": max(u100 - u50, 0),
        "adjacent_repeat_ratio": adjacent_repeat_ratio,
        "max_adjacent_repeat_run": _max_adjacent_repeat_run(values),
    }


def history_features(histories) -> pd.DataFrame:
    """Build one numeric feature row per request, without labels or outcomes."""
    rows = [_features_for_one(history) for history in histories]
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
