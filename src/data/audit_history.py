from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"

W_VALUES = [1, 5, 10, 20, 50, 100]


def get_array_len(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return len(x)
    return 0


def audit_basic(df):
    print("\n" + "=" * 80)
    print("BASIC DATA AUDIT")
    print("=" * 80)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("  PASS - None")
    else:
        for col, n in missing.items():
            print(f"  FAIL - {col}: {n:,}")

    key_cols = ["user_id", "video_id", "time_ms"]

    print("\nDuplicate evaluation events:")
    duplicates = df.duplicated(subset=key_cols).sum()

    if duplicates == 0:
        print("  PASS - 0")
    else:
        print(f"  FAIL - {duplicates:,}")

    print("\neval_idx uniqueness:")
    unique_eval_idx = df["eval_idx"].is_unique

    if unique_eval_idx:
        print("  PASS")
    else:
        print("  FAIL")


def audit_users(df):
    print("\n" + "=" * 80)
    print("USER DISTRIBUTION")
    print("=" * 80)

    events_per_user = df.groupby("user_id").size()

    print(f"Unique users:       {df['user_id'].nunique():,}")
    print(f"Mean events/user:   {events_per_user.mean():.2f}")
    print(f"Median events/user: {events_per_user.median():.2f}")
    print(f"Min events/user:    {events_per_user.min():,}")
    print(f"Max events/user:    {events_per_user.max():,}")


def audit_history_lengths(df):
    print("\n" + "=" * 80)
    print("HISTORY LENGTH AUDIT")
    print("=" * 80)

    all_pass = True

    for w in W_VALUES:
        len_col = f"history_{w}_len"
        hist_col = f"history_{w}"

        lengths = df[len_col]

        expected_max = w
        actual_max = lengths.max()

        full_count = (lengths == w).sum()
        partial_count = ((lengths > 0) & (lengths < w)).sum()
        empty_count = (lengths == 0).sum()

        invalid_count = ((lengths < 0) | (lengths > w)).sum()

        array_mismatch = (
            df[hist_col].apply(get_array_len) != lengths
        ).sum()

        status = "PASS"

        if actual_max > expected_max:
            status = "FAIL"
            all_pass = False

        if invalid_count > 0:
            status = "FAIL"
            all_pass = False

        if array_mismatch > 0:
            status = "FAIL"
            all_pass = False

        print(
            f"W={w:<3} "
            f"[{status}] "
            f"full={full_count:>7,} "
            f"partial={partial_count:>7,} "
            f"empty={empty_count:>7,} "
            f"mean={lengths.mean():>6.2f} "
            f"max={actual_max:>3}"
        )

        if array_mismatch > 0:
            print(
                f"       array/length mismatches: "
                f"{array_mismatch:,}"
            )

    print("\nHistory length consistency:", "PASS" if all_pass else "FAIL")


def audit_full_history(df):
    print("\n" + "=" * 80)
    print("FULL HISTORY AUDIT")
    print("=" * 80)

    length_mismatch = (
        df["history_full"].apply(get_array_len)
        != df["history_full_len"]
    ).sum()

    declared_mismatch = (
        df["history_len_full"] != df["history_full_len"]
    ).sum()

    print(f"Array vs history_full_len mismatches: {length_mismatch:,}")
    print(f"history_len_full mismatches:          {declared_mismatch:,}")

    if length_mismatch == 0 and declared_mismatch == 0:
        print("Full history consistency: PASS")
    else:
        print("Full history consistency: FAIL")


def audit_temporal(df):
    print("\n" + "=" * 80)
    print("TEMPORAL AUDIT")
    print("=" * 80)

    print(f"Timestamp column: time_ms")
    print(f"Min timestamp:    {df['time_ms'].min()}")
    print(f"Max timestamp:    {df['time_ms'].max()}")

    sorted_df = df.sort_values(["user_id", "time_ms"])

    user_time_diff = (
        sorted_df
        .groupby("user_id")["time_ms"]
        .diff()
    )

    negative_diffs = (user_time_diff < 0).sum()

    print(f"Per-user ordering violations: {negative_diffs:,}")

    if negative_diffs == 0:
        print("Evaluation temporal ordering: PASS")
    else:
        print("Evaluation temporal ordering: FAIL")


def audit_history_temporal_consistency(df):
    print("\n" + "=" * 80)
    print("HISTORY TEMPORAL CONSISTENCY")
    print("=" * 80)

    # History construction stores video IDs, so this audit verifies
    # structural availability rather than reconstructing raw timestamps.
    # Raw-event temporal leakage was already validated by history.py.

    invalid_lengths = 0

    for w in W_VALUES:
        col = f"history_{w}_len"

        invalid_lengths += (
            (df[col] < 0) | (df[col] > w)
        ).sum()

    if invalid_lengths == 0:
        print("History window bounds: PASS")
    else:
        print(f"History window bounds: FAIL ({invalid_lengths:,})")


def main():
    print("=" * 80)
    print("AUDIT KUAIRAND EVALUATION HISTORIES")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {HISTORY_PATH}")

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"History file not found: {HISTORY_PATH}"
        )

    df = pd.read_parquet(HISTORY_PATH)

    print("\nSchema:")
    print(df.dtypes)

    audit_basic(df)
    audit_users(df)
    audit_history_lengths(df)
    audit_full_history(df)
    audit_temporal(df)
    audit_history_temporal_consistency(df)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()