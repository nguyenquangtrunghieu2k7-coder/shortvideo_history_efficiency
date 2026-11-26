"""Leakage-aware rolling temporal evaluation splits (experimental, non-destructive).

Uses existing evaluation events; does NOT rebuild frozen history or retrain V4.
Rolling folds may overlap across folds but train/val/test are disjoint *within*
each fold. An optional embargo separates split boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/processed/phase7/history_KFull.parquet"
DEFAULT_OUTPUT = ROOT / "data/processed/experiments/sliding_window_1day"


def rolling_splits(df, *, train_days=7, validation_days=1, test_days=1,
                   step_days=1, embargo_hours=0):
    if min(train_days, validation_days, test_days, step_days) <= 0 or embargo_hours < 0:
        raise ValueError("Window sizes and step must be positive; embargo nonnegative")
    required = {"evaluation_id", "time_ms", "user_id"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required columns: " + repr(sorted(required - set(df.columns))))
    if df["evaluation_id"].duplicated().any() or df["time_ms"].isna().any():
        raise ValueError("Duplicate evaluation IDs or null timestamps")
    ordered = df.sort_values(["time_ms", "evaluation_id"]).reset_index(drop=True)
    day = 86_400_000
    hour = 3_600_000
    first = (int(ordered["time_ms"].min()) // day) * day
    last = int(ordered["time_ms"].max())
    fold = 0
    while True:
        train_start = first + fold * step_days * day
        train_end = train_start + train_days * day
        val_start = train_end + embargo_hours * hour
        val_end = val_start + validation_days * day
        test_start = val_end + embargo_hours * hour
        test_end = test_start + test_days * day
        if test_end > last + 1:
            break
        ts = ordered["time_ms"]
        splits = {
            "train": ordered.loc[(ts >= train_start) & (ts < train_end)].copy(),
            "validation": ordered.loc[(ts >= val_start) & (ts < val_end)].copy(),
            "test": ordered.loc[(ts >= test_start) & (ts < test_end)].copy(),
        }
        if all(len(s) for s in splits.values()):
            ids = [set(s["evaluation_id"]) for s in splits.values()]
            if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
                raise AssertionError("Within-fold evaluation ID leakage")
            if not (splits["train"]["time_ms"].max() < splits["validation"]["time_ms"].min()
                    < splits["test"]["time_ms"].min()):
                raise AssertionError("Temporal ordering violation")
            yield fold, splits, {
                "fold": fold,
                "bounds_ms": {"train": [train_start, train_end],
                              "validation": [val_start, val_end],
                              "test": [test_start, test_end]},
                "counts": {k: len(v) for k, v in splits.items()},
                "embargo_hours": embargo_hours,
            }
        fold += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--train-days", type=int, default=7)
    p.add_argument("--validation-days", type=int, default=1)
    p.add_argument("--test-days", type=int, default=1)
    p.add_argument("--step-days", type=int, default=1)
    p.add_argument("--embargo-hours", type=int, default=0)
    args = p.parse_args()
    if args.output.resolve() == args.source.resolve():
        p.error("Output must not overwrite source")
    df = pd.read_parquet(args.source)
    rows = []
    for i, splits, manifest in rolling_splits(
        df, train_days=args.train_days, validation_days=args.validation_days,
        test_days=args.test_days, step_days=args.step_days,
        embargo_hours=args.embargo_hours,
    ):
        target = args.output / f"fold_{i:03d}"
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in splits.items():
            frame.to_parquet(target / f"{name}.parquet", index=False)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        rows.append(manifest)
    if not rows:
        raise SystemExit("No complete nonempty folds. Adjust window lengths for dataset time span.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fold_manifest.json").write_text(json.dumps({
        "protocol": "rolling_temporal_evaluation_split",
        "source": str(args.source),
        "folds": rows,
        "notes": [
            "Within-fold train/validation/test are chronologically disjoint.",
            "Cross-fold overlap is expected; do not pool folds as independent examples.",
            "Existing frozen V4 checkpoints were trained on baseline splits: retraining on rolling folds is a separate experiment.",
            "This does not change the existing [t-1day,t) history construction.",
        ],
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "folds": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
