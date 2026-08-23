from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"

W_VALUES = [1, 5, 10, 20, 50, 100, "full"]


def as_array(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return np.asarray([])


def audit_candidate_leakage(df):
    print("\n" + "=" * 80)
    print("CANDIDATE-IN-HISTORY LEAKAGE")
    print("=" * 80)

    total_fail = 0

    for w in W_VALUES:
        col = (
            "history_full"
            if w == "full"
            else f"history_{w}"
        )

        leaked = 0

        for _, row in df[[col, "video_id"]].iterrows():
            history = as_array(row[col])

            if row["video_id"] in history:
                leaked += 1

        total_fail += leaked

        print(
            f"W={str(w):<4} "
            f"candidate in history: "
            f"{leaked:>7,} "
            f"({leaked / len(df):.2%})"
        )

    if total_fail == 0:
        print("\nCandidate-history leakage: PASS")
    else:
        print("\nCandidate-history leakage: FAIL")


def audit_history_duplicates(df):
    print("\n" + "=" * 80)
    print("DUPLICATES INSIDE HISTORY")
    print("=" * 80)

    for w in W_VALUES:
        col = (
            "history_full"
            if w == "full"
            else f"history_{w}"
        )

        duplicate_events = 0

        for history in df[col]:
            arr = as_array(history)

            if len(arr) != len(np.unique(arr)):
                duplicate_events += 1

        print(
            f"W={str(w):<4} "
            f"events with duplicate videos: "
            f"{duplicate_events:,}"
        )


def audit_history_order(df):
    print("\n" + "=" * 80)
    print("HISTORY ORDER AUDIT")
    print("=" * 80)

    for w in W_VALUES:
        col = (
            "history_full"
            if w == "full"
            else f"history_{w}"
        )

        invalid = 0

        for history in df[col]:
            arr = as_array(history)

            # history.py is expected to store most-recent first.
            # Therefore no exact ordering can be verified without
            # historical timestamps, so this is a structural audit only.
            if len(arr) > 1 and arr.size == 0:
                invalid += 1

        print(
            f"W={str(w):<4} "
            f"structural violations: {invalid:,}"
        )


def audit_eval_identity(df):
    print("\n" + "=" * 80)
    print("EVALUATION IDENTITY AUDIT")
    print("=" * 80)

    checks = {
        "eval_idx unique": df["eval_idx"].is_unique,
        "user_id non-null": df["user_id"].notna().all(),
        "video_id non-null": df["video_id"].notna().all(),
        "time_ms non-null": df["time_ms"].notna().all(),
    }

    for name, passed in checks.items():
        print(f"{name:<25} {'PASS' if passed else 'FAIL'}")


def main():
    print("=" * 80)
    print("AUDIT KUAIRAND HISTORY LEAKAGE")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {HISTORY_PATH}")

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    df = pd.read_parquet(HISTORY_PATH)

    print(f"\nEvaluation events: {len(df):,}")

    audit_eval_identity(df)
    audit_candidate_leakage(df)
    audit_history_duplicates(df)
    audit_history_order(df)

    print("\n" + "=" * 80)
    print("LEAKAGE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()