"""Scheduled recommendation refresh ablation: 24h disjoint vs overlapping windows.

Uses real standard interaction timestamps, fixed held-out evaluation events,
and identical event-time oracle histories. No model retraining, no data overwrite.
All costs are workload *proxies*, not measured CPU/latency/energy.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data/processed/splits/KFull/test.parquet"
STANDARD = [
    ROOT / "data/raw/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv",
    ROOT / "data/raw/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv",
]
DAY = 86_400_000
HOUR = 3_600_000
STRIDES = (24, 12, 6, 3)


def last_refresh(t, stride_hours, local_hour=2, utc_offset_hours=7):
    """Latest fixed-cadence refresh <= event time; epoch ms timestamps."""
    stride = stride_hours * HOUR
    anchor = (local_hour - utc_offset_hours) * HOUR
    return anchor + (int(t) - anchor) // stride * stride


def scheduled_window(times, refresh, window_ms=DAY):
    """Strictly prior standard interactions in [refresh-window, refresh)."""
    lo = bisect_left(times, refresh - window_ms)
    hi = bisect_left(times, refresh)
    return lo, hi


def load_standard_for_users(paths, user_ids, min_time, max_time, chunk_size=250_000):
    """Chunk-filter to evaluation users/time; never load all source logs at once."""
    by_user = defaultdict(list)
    for path in paths:
        for chunk in pd.read_csv(path, usecols=["user_id", "time_ms"], chunksize=chunk_size):
            chunk = chunk.loc[
                chunk["user_id"].isin(user_ids) &
                (chunk["time_ms"] >= min_time) & (chunk["time_ms"] < max_time)
            ]
            for user, group in chunk.groupby("user_id", sort=False):
                by_user[user].extend(map(int, group["time_ms"]))
    for times in by_user.values():
        times.sort()
    return by_user


def evaluate_schedule(events, by_user, *, strides=STRIDES, local_hour=2,
                      utc_offset_hours=7, window_ms=DAY):
    """Return one report per stride on the SAME events; no duplicated labels."""
    if not len(events):
        raise ValueError("Empty evaluation set")
    if not events["evaluation_id"].is_unique:
        raise ValueError("Duplicate evaluation_id")
    if not (0 <= local_hour <= 23 and -12 <= utc_offset_hours <= 14):
        raise ValueError("Invalid local hour / UTC offset")
    for stride in strides:
        if stride <= 0 or 24 % stride:
            raise ValueError("Stride must be a positive divisor of 24h")
    results = []
    # Each scenario reuses the same held-out event IDs and labels.
    for stride in strides:
        total_selected = total_oracle = total_new = 0
        total_batch_interactions = 0
        total_repeated = 0
        total_staleness = 0
        n_nonempty = 0
        distinct_refreshes = set()
        per_user_refreshes = defaultdict(set)
        for row in events.itertuples(index=False):
            t = int(row.time_ms)
            refresh = last_refresh(t, stride, local_hour, utc_offset_hours)
            times = by_user.get(row.user_id, [])
            lo, hi = scheduled_window(times, refresh, window_ms)
            oracle_lo, oracle_hi = scheduled_window(times, t, window_ms)
            selected = hi - lo
            oracle = oracle_hi - oracle_lo
            new = bisect_left(times, t) - hi
            if selected and times[hi - 1] >= t:
                raise AssertionError("Future interaction leakage")
            total_selected += selected
            total_oracle += oracle
            total_new += new
            total_staleness += t - refresh
            n_nonempty += bool(selected)
            distinct_refreshes.add(refresh)
            per_user_refreshes[row.user_id].add(refresh)
        # Batch refresh workload: each user refreshed once per schedule tick
        # between first and last held-out event, not once per event. Users
        # without a request at a tick are not charged (conservative proxy).
        for user, refreshes in per_user_refreshes.items():
            times = by_user.get(user, [])
            previous_hi = None
            for refresh in sorted(refreshes):
                lo, hi = scheduled_window(times, refresh, window_ms)
                total_batch_interactions += hi - lo
                if previous_hi is not None:
                    total_repeated += max(0, min(hi, previous_hi) - lo)
                previous_hi = hi
        results.append({
            "scenario": f"S{STRIDES.index(stride)}" if stride in STRIDES else f"stride_{stride}h",
            "window_hours": window_ms / HOUR,
            "stride_hours": stride,
            "nominal_overlap_fraction": 1 - stride * HOUR / window_ms,
            "n_evaluation_events": len(events),
            "n_distinct_refresh_timestamps": len(distinct_refreshes),
            "n_user_refresh_jobs_observed": sum(map(len, per_user_refreshes.values())),
            "history_interactions_available_at_refresh_sum": total_selected,
            "oracle_event_time_history_interactions_sum": total_oracle,
            "new_interactions_since_refresh_sum": total_new,
            "history_coverage_ratio_vs_oracle": total_selected / total_oracle if total_oracle else None,
            "mean_refresh_age_hours": total_staleness / len(events) / HOUR,
            "nonempty_refresh_history_fraction": n_nonempty / len(events),
            "batch_history_interactions_processed_proxy": total_batch_interactions,
            "repeated_interactions_across_refresh_jobs_proxy": total_repeated,
        })
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=Path, default=EVAL)
    p.add_argument("--standard-files", type=Path, nargs="+", default=STANDARD)
    p.add_argument("--strides", type=int, nargs="+", default=list(STRIDES))
    p.add_argument("--refresh-local-hour", type=int, default=2)
    p.add_argument("--utc-offset-hours", type=int, default=7)
    p.add_argument("--max-events", type=int)
    p.add_argument("--output", type=Path,
                   default=ROOT / "reports/experiments/scheduled_window_ablation.json")
    args = p.parse_args()
    start = perf_counter()
    events = pd.read_parquet(args.events, columns=[
        "evaluation_id", "user_id", "time_ms", "long_view"])
    events = events.sort_values(["time_ms", "evaluation_id"]).reset_index(drop=True)
    if args.max_events:
        if args.max_events < 1:
            p.error("--max-events must be positive")
        events = events.iloc[:args.max_events].copy()
    if not len(events):
        p.error("Empty evaluation data")
    earliest = min(last_refresh(int(events.time_ms.min()), s,
                                args.refresh_local_hour, args.utc_offset_hours)
                   for s in args.strides) - DAY
    by_user = load_standard_for_users(
        args.standard_files, set(events.user_id), earliest, int(events.time_ms.max()))
    rows = evaluate_schedule(events, by_user, strides=args.strides,
                             local_hour=args.refresh_local_hour,
                             utc_offset_hours=args.utc_offset_hours)
    report = {
        "status": "PASS",
        "protocol": "scheduled_24h_history_refresh_ablation",
        "evaluation_file": str(args.events),
        "n_evaluation_events": len(events),
        "refresh_local_hour": args.refresh_local_hour,
        "utc_offset_hours": args.utc_offset_hours,
        "scenarios": rows,
        "wall_seconds": perf_counter() - start,
        "limitations": [
            "Refresh workload proxy counts user-refresh jobs observed among evaluation requests; not every possible production batch job.",
            "Raw interaction counts are not frozen-V4-selected K or measured resource cost.",
            "History coverage ratio compares aggregate counts; scheduled windows can include interactions outside event-time oracle's 24h interval.",
            "No AUC or model inference is measured by this runner.",
            "The dataset does not identify sleeping users or prove a particular commercial scheduling policy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
