# Scheduled 24-hour History Refresh: Disjoint vs Overlapping Sliding Windows

**Experimental branch:** `experiment/sliding-window-history-vs-disjoint-1day`  
**Status:** scenario simulator and unit tests committed; real-data execution pending.

## Research question and business motivation

Batch or background precomputation during low-activity periods is a valid recommendation-system design pattern, often combined with online ranking and other freshness mechanisms. We study whether a **once-daily, 24-hour, non-overlapping refresh** is a good quality–freshness–workload compromise compared with more frequent overlapping refreshes. We do **not** assume that a once-daily batch is globally optimal.

**Important correction:** The original per-event history builder already uses `[t-24h,t)` sliding history; the earlier rolling train/validation/test fold generator on this branch addresses a *different question*. The simulator below explicitly models **scheduled refresh times**, not data-split folds.

## Four scenarios

| Scenario | History lookback | Refresh stride | Nominal overlap between consecutive windows |
|---|---:|---:|---:|
| S0 — Daily disjoint | 24h | 24h | 0% |
| S1 — Sliding 12h | 24h | 12h | 50% |
| S2 — Sliding 6h | 24h | 6h | 75% |
| S3 — Sliding 3h | 24h | 3h | 87.5% |

All schedules are anchored to **02:00 local time** (default UTC+07:00, configurable). A refresh at time `r` uses interactions in **`[r-24h, r)`**. An evaluation event at time `t` uses the most recent refresh `r <= t`. No future events are admitted. The event-time oracle uses `[t-24h,t)`. The same held-out evaluation IDs are used across scenarios.

## What the first experiment measures

- Number of distinct refresh ticks and observed user-refresh jobs
- Mean time between last refresh and evaluation request (staleness)
- Count of new user interactions since the latest refresh
- Interactions in the scheduled refresh window and the event-time oracle
- Aggregate interaction coverage ratio (descriptive only; scheduled and oracle windows can have different lower bounds)
- Approximate total interactions processed across observed refresh jobs
- Approximate repeated interactions across consecutive observed refresh jobs

**These are data/workload proxies, not measured compute, AUC, user satisfaction or actual production costs.** A higher refresh rate may reduce staleness while increasing repeated work. The existing frozen V4 E2E AUC **cannot be reused as the AUC of these scenarios**. Quality evaluation requires mapping each scheduled history into the frozen model's video vocabulary, running V4 on that history, and performing inference on the same held-out labels. That is a separate acceptance stage.

## Run the experiment

```bash
git fetch origin
git switch experiment/sliding-window-history-vs-disjoint-1day
PYTHONPATH=. python -m unittest tests.test_scheduled_window_ablation -v

PYTHONPATH=. python -m src.integration.scheduled_window_ablation \
  --strides 24 12 6 3 \
  --refresh-local-hour 2 \
  --utc-offset-hours 7 \
  --output reports/experiments/scheduled_window_ablation.json
```

The runner reads the existing held-out `data/processed/splits/KFull/test.parquet` and original standard log timestamps; it does not overwrite frozen policy, model, snapshots, splits or baseline reports. If source timestamps are not UTC epoch milliseconds, adjust the anchor after checking the dataset's time convention.

**Interpretation guardrails:** Cross-scenario comparisons use the same held-out events, but the daily schedule is a hypothetical batch-serving policy, not an observed fact about KuaiRand users. There is no sleep/activity classifier in this simulator. Do not call a scenario optimal without testing quality and real processing cost. When presenting results, distinguish *scheduled refresh frequency* from *V4 selected history K*, since this first-stage simulator does not yet execute V4.
