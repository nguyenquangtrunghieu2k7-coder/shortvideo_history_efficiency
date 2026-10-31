"""Bounded-cost request-time features for Adaptive History Selector V4.

Only the history length metadata and the most recent 100 interactions are
consulted. In a serving system, history_length should come from stored metadata
and the suffix should be fetched as a bounded window; do not materialize or
scan the full history just to compute selector features.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "history_length",
    "log_history_length",
    "recent_unique_count_100",
    "recent_unique_ratio_100",
    "recent_normalized_entropy_100",
    "recent_effective_unique_count_100",
    "recent_max_item_share_100",
    "recent_top5_item_share_100",
    "recent_singleton_item_ratio_100",
    "recent_repeated_item_ratio_100",
    "unique_last_5",
    "unique_last_20",
    "unique_last_50",
    "recent_adjacent_repeat_ratio_100",
    "recent_max_adjacent_repeat_run_100",
    "unique_growth_20_to_100",
    "unique_growth_50_to_100",
]


def _max_adjacent_repeat_run(values):
    if not values:
        return 0
    best = current = 1
    for left, right in zip(values, values[1:]):
        current = current + 1 if left == right else 1
        best = max(best, current)
    return best


def _features_for_one(history):
    # len(history) is metadata; only the suffix window is materialized.
    history_length = len(history)
    recent = [int(x) for x in history[-100:]]
    n = len(recent)
    counts = Counter(recent)
    unique_n = len(counts)

    if n:
        probs = np.asarray(list(counts.values()), dtype=np.float64) / n
        entropy = float(-(probs * np.log(probs)).sum())
        normalized_entropy = entropy / math.log(unique_n) if unique_n > 1 else 0.0
        effective_unique = float(math.exp(entropy))
        max_share = max(counts.values()) / n
        top5_share = sum(sorted(counts.values(), reverse=True)[:5]) / n
        singleton_ratio = sum(v == 1 for v in counts.values()) / unique_n
        repeated_ratio = sum(v > 1 for v in counts.values()) / unique_n
        adjacent_repeat = sum(a == b for a, b in zip(recent, recent[1:])) / max(n - 1, 1)
    else:
        normalized_entropy = effective_unique = 0.0
        max_share = top5_share = singleton_ratio = repeated_ratio = adjacent_repeat = 0.0

    def unique_ratio(window):
        values = recent[-window:]
        return len(set(values)) / len(values) if values else 0.0

    u20 = len(set(recent[-20:]))
    u50 = len(set(recent[-50:]))
    u100 = unique_n

    return {
        "history_length": history_length,
        "log_history_length": math.log1p(history_length),
        "recent_unique_count_100": unique_n,
        "recent_unique_ratio_100": unique_ratio(100),
        "recent_normalized_entropy_100": normalized_entropy,
        "recent_effective_unique_count_100": effective_unique,
        "recent_max_item_share_100": max_share,
        "recent_top5_item_share_100": top5_share,
        "recent_singleton_item_ratio_100": singleton_ratio,
        "recent_repeated_item_ratio_100": repeated_ratio,
        "unique_last_5": unique_ratio(5),
        "unique_last_20": unique_ratio(20),
        "unique_last_50": unique_ratio(50),
        "recent_adjacent_repeat_ratio_100": adjacent_repeat,
        "recent_max_adjacent_repeat_run_100": _max_adjacent_repeat_run(recent),
        "unique_growth_20_to_100": max(u100 - u20, 0),
        "unique_growth_50_to_100": max(u100 - u50, 0),
    }


def history_features(histories):
    """Build numeric request-time features from at most the latest 100 IDs."""
    frame = pd.DataFrame(
        [_features_for_one(history) for history in histories],
        columns=FEATURE_NAMES,
    )
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
