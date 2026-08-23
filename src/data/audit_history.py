from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"

W_VALUES = [1, 5, 10, 20, 50, 100]


def main():
    print("=" * 80)
    print("AUDIT KUAIRAND EVALUATION HISTORIES")
    print("=" * 80)

    print(f"Loading: {HISTORY_PATH}")
    df = pd.read_parquet(HISTORY_PATH)

    print("\nSchema:")
    print(df.dtypes)

    print("\nShape:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  None")
    else:
        for col, n in missing.items():
            print(f"  {col}: {n:,}")

    print("\nDuplicate rows:")
    print(f"  {df.duplicated().sum():,}")

    # ------------------------------------------------------------------
    # History columns
    # ------------------------------------------------------------------
    print("\nHistory coverage:")

    for w in W_VALUES:
        candidates = [
            f"history_{w}",
            f"history_w{w}",
        ]

        col = next((c for c in candidates if c in df.columns), None)

        if col is None:
            print(f"  W={w:<3} column not found")
            continue

        lengths = df[col].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) else 0
        )

        full = (lengths >= w).sum()

        print(
            f"  W={w:<3} "
            f"full: {full:>7,} "
            f"({full / len(df):.2%}) "
            f"max: {lengths.max()}"
        )

    # ------------------------------------------------------------------
    # User distribution
    # ------------------------------------------------------------------
    print("\nUser distribution:")

    user_col = next(
        (c for c in ["user_id", "user"] if c in df.columns),
        None,
    )

    if user_col:
        users = df[user_col].nunique()
        events_per_user = df.groupby(user_col).size()

        print(f"  Unique users: {users:,}")
        print(f"  Mean events/user: {events_per_user.mean():.2f}")
        print(f"  Median events/user: {events_per_user.median():.2f}")
        print(f"  Max events/user: {events_per_user.max():,}")
    else:
        print("  user_id column not found")

    # ------------------------------------------------------------------
    # Temporal audit
    # ------------------------------------------------------------------
    print("\nTemporal audit:")

    time_col = next(
        (
            c
            for c in ["timestamp", "time", "ts", "event_time"]
            if c in df.columns
        ),
        None,
    )

    if time_col:
        print(f"  Time column: {time_col}")
        print(f"  Min: {df[time_col].min()}")
        print(f"  Max: {df[time_col].max()}")
    else:
        print("  Timestamp column not found")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()