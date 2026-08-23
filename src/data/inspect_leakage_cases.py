from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"

W_VALUES = [1, 5, 10, 20, 50, 100]


def as_array(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return np.asarray([])


def main():
    print("=" * 80)
    print("INSPECT FULL-HISTORY LEAKAGE CASES")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {HISTORY_PATH}")

    df = pd.read_parquet(HISTORY_PATH)

    cases = []

    for _, row in df.iterrows():
        history = as_array(row["history_full"])

        positions = np.where(history == row["video_id"])[0]

        if len(positions) > 0:
            cases.append(
                {
                    "eval_idx": row["eval_idx"],
                    "user_id": row["user_id"],
                    "video_id": row["video_id"],
                    "time_ms": row["time_ms"],
                    "history_full_len": row["history_full_len"],
                    "candidate_occurrences": len(positions),
                    "nearest_position": int(positions[-1]) + 1,
                    "farthest_position": int(positions[0]) + 1,
                }
            )

    result = pd.DataFrame(cases)

    print(f"\nLeakage cases: {len(result):,}")

    if result.empty:
        print("No full-history leakage cases found.")
        return

    print("\nCases:")
    print(result.to_string(index=False))

    print("\n" + "=" * 80)
    print("POSITION SUMMARY")
    print("=" * 80)

    print(
        f"Nearest position:"
        f" min={result['nearest_position'].min()},"
        f" median={result['nearest_position'].median()},"
        f" max={result['nearest_position'].max()}"
    )

    print(
        f"Farthest position:"
        f" min={result['farthest_position'].min()},"
        f" median={result['farthest_position'].median()},"
        f" max={result['farthest_position'].max()}"
    )

    print("\nCases within each window:")

    for w in W_VALUES:
        count = (
            result["nearest_position"] <= w
        ).sum()

        print(
            f"  W={w:<3}: "
            f"{count:>3} cases"
        )

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()