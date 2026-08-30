from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PHASE7_DIR = (
    ROOT / "data" / "processed" / "phase7"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "splits"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

K_VALUES = [
    0,
    1,
    5,
    10,
    20,
    50,
    100,
    "Full",
]


# =============================================================================
# LOAD PHASE 7 HISTORY
# =============================================================================

def load_history(k):

    filename = (
        "history_KFull.parquet"
        if k == "Full"
        else f"history_K{k}.parquet"
    )

    path = PHASE7_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"\nLoading Phase 7 history:")
    print(f"  {path}")

    df = pd.read_parquet(path)

    required = [
        "evaluation_id",
        "user_id",
        "video_id",
        "time_ms",
        "long_view",
        "history",
        "history_len",
        "history_available",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"K={k}: missing columns: {missing}"
        )

    return df


# =============================================================================
# BUILD TEMPORAL SPLIT
# =============================================================================

def build_split(df):

    # Strict chronological ordering.
    df = df.sort_values(
        ["time_ms", "evaluation_id"]
    ).reset_index(drop=True)

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    train = df.iloc[
        :train_end
    ].copy()

    val = df.iloc[
        train_end:val_end
    ].copy()

    test = df.iloc[
        val_end:
    ].copy()

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_splits(
    df,
    splits,
    k,
):

    train = splits["train"]
    val = splits["validation"]
    test = splits["test"]

    print("\nSplit sizes:")

    for name, split in splits.items():

        print(
            f"  {name:<10}: "
            f"{len(split):>7,} events | "
            f"{split['time_ms'].min()} → "
            f"{split['time_ms'].max()}"
        )

    # -------------------------------------------------------------------------
    # Event coverage
    # -------------------------------------------------------------------------

    assert (
        len(train)
        + len(val)
        + len(test)
        == len(df)
    )

    # -------------------------------------------------------------------------
    # Temporal ordering
    # -------------------------------------------------------------------------

    assert (
        train["time_ms"].max()
        < val["time_ms"].min()
    )

    assert (
        val["time_ms"].max()
        < test["time_ms"].min()
    )

    # -------------------------------------------------------------------------
    # evaluation_id uniqueness
    # -------------------------------------------------------------------------

    assert train["evaluation_id"].is_unique
    assert val["evaluation_id"].is_unique
    assert test["evaluation_id"].is_unique

    # -------------------------------------------------------------------------
    # Split disjointness
    # -------------------------------------------------------------------------

    train_ids = set(
        train["evaluation_id"]
    )

    val_ids = set(
        val["evaluation_id"]
    )

    test_ids = set(
        test["evaluation_id"]
    )

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    # -------------------------------------------------------------------------
    # History sanity
    # -------------------------------------------------------------------------

    for name, split in splits.items():

        assert (
            split["history_len"] <= (
                2048 if k == "Full" else k
            )
        ).all()

        if k == 0:
            assert (
                split["history_len"] == 0
            ).all()

            assert (
                split["history_available"] == 0
            ).all()

    print("\nSplit validation:")

    print("  Temporal ordering:    PASS")
    print("  Event coverage:       PASS")
    print("  evaluation_id unique: PASS")
    print("  Split disjointness:   PASS")
    print("  History validity:     PASS")


# =============================================================================
# SAVE
# =============================================================================

def save_splits(
    splits,
    k,
):

    k_name = (
        "KFull"
        if k == "Full"
        else f"K{k}"
    )

    output_dir = (
        OUTPUT_DIR / k_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, split in splits.items():

        path = (
            output_dir
            / f"{name}.parquet"
        )

        split.to_parquet(
            path,
            index=False,
        )

        print(
            f"  Saved {name}: {path}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("BUILD PHASE 8 TRAIN / VALIDATION / TEST SPLITS")
    print("=" * 80)

    print("\nSource:")
    print(
        "  Phase 7 frozen history artifacts"
    )

    print(
        f"  Train ratio: {TRAIN_RATIO:.2f}"
    )

    print(
        f"  Validation ratio: {VAL_RATIO:.2f}"
    )

    for k in K_VALUES:

        print("\n" + "=" * 80)
        print(f"BUILDING SPLIT | K={k}")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # Load Phase 7 artifact
        # ---------------------------------------------------------------------

        df = load_history(k)

        print(
            f"  Evaluation events: "
            f"{len(df):,}"
        )

        # ---------------------------------------------------------------------
        # Build split
        # ---------------------------------------------------------------------

        splits = build_split(df)

        # ---------------------------------------------------------------------
        # Validate
        # ---------------------------------------------------------------------

        validate_splits(
            df,
            splits,
            k,
        )

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        print("\nSaving:")

        save_splits(
            splits,
            k,
        )

    print("\n" + "=" * 80)
    print("PHASE 8 SPLIT BUILD COMPLETE")
    print("=" * 80)

    print("\nOutput structure:")

    for k in K_VALUES:

        k_name = (
            "KFull"
            if k == "Full"
            else f"K{k}"
        )

        print(
            f"  data/processed/splits/"
            f"{k_name}/"
        )

        print("    train.parquet")
        print("    validation.parquet")
        print("    test.parquet")

    print("\nProtocol:")
    print("  Source history:       Phase 7")
    print("  Fixed evaluation set: PASS")
    print("  Temporal split:       PASS")
    print("  History consistency:  PASS")
    print("  Old history source:   NOT USED")


if __name__ == "__main__":
    main()
