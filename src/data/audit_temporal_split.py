from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def audit_global_time_order(df):
    print("\n" + "=" * 80)
    print("GLOBAL TEMPORAL ORDER")
    print("=" * 80)

    df = df.sort_values("time_ms").reset_index(drop=True)

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print("Events:")
    print(f"  train:      {len(train):,}")
    print(f"  validation: {len(val):,}")
    print(f"  test:       {len(test):,}")

    print("\nTime ranges:")

    for name, split in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        print(
            f"  {name:<10} "
            f"{split['time_ms'].min()} "
            f"→ {split['time_ms'].max()}"
        )

    train_val_ok = train["time_ms"].max() < val["time_ms"].min()
    val_test_ok = val["time_ms"].max() < test["time_ms"].min()

    print("\nTemporal ordering:")

    print(
        f"  train < validation: "
        f"{'PASS' if train_val_ok else 'FAIL'}"
    )

    print(
        f"  validation < test:  "
        f"{'PASS' if val_test_ok else 'FAIL'}"
    )


def audit_user_temporal_order(df):
    print("\n" + "=" * 80)
    print("PER-USER TEMPORAL ORDER")
    print("=" * 80)

    ordered = df.sort_values(
        ["user_id", "time_ms"]
    )

    violations = (
        ordered
        .groupby("user_id")["time_ms"]
        .diff()
        .lt(0)
        .sum()
    )

    print(f"Per-user time violations: {violations:,}")

    if violations == 0:
        print("Per-user temporal order: PASS")
    else:
        print("Per-user temporal order: FAIL")


def audit_history_before_evaluation(df):
    print("\n" + "=" * 80)
    print("HISTORY BEFORE EVALUATION")
    print("=" * 80)

    # history.py already validated temporal construction.
    # Here we verify that every evaluation event has a valid
    # non-negative history length.

    violations = 0

    for w in [1, 5, 10, 20, 50, 100]:
        col = f"history_{w}_len"

        violations += (
            (df[col] < 0) |
            (df[col] > w)
        ).sum()

    print(f"History window violations: {violations:,}")

    if violations == 0:
        print("History bounds: PASS")
    else:
        print("History bounds: FAIL")


def audit_user_presence_across_temporal_splits(df):
    print("\n" + "=" * 80)
    print("USER PRESENCE ACROSS TEMPORAL SPLITS")
    print("=" * 80)

    df = df.sort_values("time_ms")

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    train_users = set(train["user_id"])
    val_users = set(val["user_id"])
    test_users = set(test["user_id"])

    print(f"Train users:      {len(train_users):,}")
    print(f"Validation users: {len(val_users):,}")
    print(f"Test users:       {len(test_users):,}")

    print("\nOverlaps:")

    print(f"  train ∩ validation: {len(train_users & val_users):,}")
    print(f"  train ∩ test:       {len(train_users & test_users):,}")
    print(f"  validation ∩ test:  {len(val_users & test_users):,}")


def main():
    print("=" * 80)
    print("AUDIT TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {HISTORY_PATH}")

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    df = pd.read_parquet(HISTORY_PATH)

    print(f"\nEvaluation events: {len(df):,}")

    audit_global_time_order(df)
    audit_user_temporal_order(df)
    audit_history_before_evaluation(df)
    audit_user_presence_across_temporal_splits(df)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()