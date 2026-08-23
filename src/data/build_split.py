from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"
OUTPUT_DIR = ROOT / "data" / "processed"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def main():
    print("=" * 80)
    print("BUILD TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {INPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_parquet(INPUT_PATH)

    # Strict chronological ordering.
    df = df.sort_values("time_ms").reset_index(drop=True)

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    splits = {
        "train": train,
        "validation": val,
        "test": test,
    }

    print("\nSplit sizes:")

    for name, split in splits.items():
        print(
            f"  {name:<10}: "
            f"{len(split):>7,} events | "
            f"{split['time_ms'].min()} → "
            f"{split['time_ms'].max()}"
        )

    # Validate temporal boundaries before writing.
    assert train["time_ms"].max() < val["time_ms"].min()
    assert val["time_ms"].max() < test["time_ms"].min()

    # Validate no events were lost.
    assert len(train) + len(val) + len(test) == len(df)

    # Validate eval_idx uniqueness.
    assert train["eval_idx"].is_unique
    assert val["eval_idx"].is_unique
    assert test["eval_idx"].is_unique

    # Validate split disjointness by evaluation event.
    train_idx = set(train["eval_idx"])
    val_idx = set(val["eval_idx"])
    test_idx = set(test["eval_idx"])

    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": OUTPUT_DIR / "train.parquet",
        "validation": OUTPUT_DIR / "validation.parquet",
        "test": OUTPUT_DIR / "test.parquet",
    }

    for name, split in splits.items():
        split.to_parquet(paths[name], index=False)
        print(f"\nSaved {name}:")
        print(f"  {paths[name]}")

    print("\n" + "=" * 80)
    print("SPLIT VALIDATION")
    print("=" * 80)

    print("Temporal ordering: PASS")
    print("Event coverage:    PASS")
    print("eval_idx unique:   PASS")
    print("Split disjointness: PASS")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()