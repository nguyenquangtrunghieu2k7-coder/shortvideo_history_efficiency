from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
PHASE7 = PROCESSED / "phase7"

TRAIN_FILE = PROCESSED / "train.parquet"
VAL_FILE = PROCESSED / "validation.parquet"
TEST_FILE = PROCESSED / "test.parquet"

EVAL_FILE = PROCESSED / "evaluation_events.parquet"

HISTORY_FILES = {
    "K0": PHASE7 / "history_K0.parquet",
    "K1": PHASE7 / "history_K1.parquet",
    "K5": PHASE7 / "history_K5.parquet",
    "K10": PHASE7 / "history_K10.parquet",
    "K20": PHASE7 / "history_K20.parquet",
    "K50": PHASE7 / "history_K50.parquet",
    "K100": PHASE7 / "history_K100.parquet",
    "KFull": PHASE7 / "history_KFull.parquet",
}

REQUIRED_SPLIT_COLS = {
    "eval_idx",
    "user_id",
    "video_id",
    "time_ms",
}

REQUIRED_EVAL_COLS = {
    "eval_idx",
    "user_id",
    "video_id",
    "time_ms",
    "long_view",
}

REQUIRED_HISTORY_COLS = {
    "eval_idx",
    "user_id",
    "video_id",
    "time_ms",
}


def check_columns(path, required):
    schema = pd.read_parquet(path, engine="pyarrow").columns.tolist()
    missing = required - set(schema)

    if missing:
        print(f"  FAIL — missing columns: {sorted(missing)}")
        return False

    print("  PASS")
    return True


def load_keys(path):
    return pd.read_parquet(
        path,
        columns=["eval_idx", "user_id", "video_id", "time_ms"],
    )


def main():
    print("=" * 80)
    print("PHASE 8 — SCHEMA & SPLIT MAPPING AUDIT")
    print("=" * 80)

    failed = False

    # ------------------------------------------------------------------
    # 1. Evaluation artifact
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. EVALUATION EVENT ARTIFACT")
    print("=" * 80)

    print(f"\nLoading:\n  {EVAL_FILE}")

    eval_df = pd.read_parquet(
        EVAL_FILE,
        columns=[
            "eval_idx",
            "user_id",
            "video_id",
            "time_ms",
            "long_view",
        ],
    )

    print(f"  Evaluation events: {len(eval_df):,}")

    print("\nRequired evaluation schema:")
    ok = REQUIRED_EVAL_COLS.issubset(eval_df.columns)

    if ok:
        print("  PASS")
    else:
        print(
            "  FAIL — missing:",
            sorted(REQUIRED_EVAL_COLS - set(eval_df.columns)),
        )
        failed = True

    if eval_df["eval_idx"].is_unique:
        print("  eval_idx uniqueness: PASS")
    else:
        print("  eval_idx uniqueness: FAIL")
        failed = True

    if eval_df["long_view"].isin([0, 1]).all():
        print("  long_view validity: PASS")
    else:
        print("  long_view validity: FAIL")
        failed = True

    # ------------------------------------------------------------------
    # 2. Train / validation / test schemas
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. TRAIN / VALIDATION / TEST SCHEMAS")
    print("=" * 80)

    splits = {
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
        "test": TEST_FILE,
    }

    split_keys = {}

    for name, path in splits.items():
        print(f"\n{name}:")
        print(f"  {path}")

        if not path.exists():
            print("  FAIL — file not found")
            failed = True
            continue

        ok = check_columns(path, REQUIRED_SPLIT_COLS)

        if not ok:
            failed = True
            continue

        split_keys[name] = load_keys(path)

        print(
            f"  Rows: {len(split_keys[name]):,}"
        )

        if split_keys[name]["eval_idx"].is_unique:
            print("  eval_idx uniqueness: PASS")
        else:
            print("  eval_idx uniqueness: FAIL")
            failed = True

    # ------------------------------------------------------------------
    # 3. Split mapping to evaluation events
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. SPLIT → EVALUATION MAPPING")
    print("=" * 80)

    eval_ids = set(eval_df["eval_idx"])

    for name, df in split_keys.items():
        ids = set(df["eval_idx"])

        missing = ids - eval_ids

        if not missing:
            print(f"  {name}: all eval_idx exist in evaluation_events: PASS")
        else:
            print(
                f"  {name}: FAIL — "
                f"{len(missing):,} eval_idx not found"
            )
            failed = True

    # Check disjointness
    names = list(split_keys.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            overlap = (
                set(split_keys[a]["eval_idx"])
                & set(split_keys[b]["eval_idx"])
            )

            if not overlap:
                print(
                    f"  {a} ∩ {b}: PASS"
                )
            else:
                print(
                    f"  {a} ∩ {b}: FAIL — "
                    f"{len(overlap):,} overlapping events"
                )
                failed = True

    # ------------------------------------------------------------------
    # 4. Label availability through evaluation_events
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. LABEL MAPPING")
    print("=" * 80)

    label_map = eval_df.set_index("eval_idx")["long_view"]

    for name, df in split_keys.items():
        labels = df["eval_idx"].map(label_map)

        if labels.notna().all():
            print(f"  {name}: label mapping: PASS")
        else:
            print(
                f"  {name}: label mapping: FAIL — "
                f"{labels.isna().sum():,} missing labels"
            )
            failed = True

    # ------------------------------------------------------------------
    # 5. Phase 7 history schemas
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. PHASE 7 HISTORY ARTIFACTS")
    print("=" * 80)

    history_keys = {}

    for k, path in HISTORY_FILES.items():
        print(f"\n{k}:")
        print(f"  {path}")

        if not path.exists():
            print("  FAIL — file not found")
            failed = True
            continue

        cols = set(
            pd.read_parquet(
                path,
                engine="pyarrow",
            ).columns
        )

        if REQUIRED_HISTORY_COLS.issubset(cols):
            print("  Required schema: PASS")
        else:
            print(
                "  Required schema: FAIL — missing:",
                sorted(REQUIRED_HISTORY_COLS - cols),
            )
            failed = True
            continue

        h = pd.read_parquet(
            path,
            columns=["eval_idx"],
        )

        history_keys[k] = h

        print(f"  Events: {len(h):,}")

        if h["eval_idx"].is_unique:
            print("  eval_idx uniqueness: PASS")
        else:
            print("  eval_idx uniqueness: FAIL")
            failed = True

    # ------------------------------------------------------------------
    # 6. Phase 7 → evaluation mapping
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. PHASE 7 → EVALUATION MAPPING")
    print("=" * 80)

    for k, df in history_keys.items():
        ids = set(df["eval_idx"])

        missing = ids - eval_ids

        if not missing:
            print(f"  {k}: evaluation identity: PASS")
        else:
            print(
                f"  {k}: FAIL — "
                f"{len(missing):,} unknown eval_idx"
            )
            failed = True

    # ------------------------------------------------------------------
    # 7. Fixed evaluation population across K
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. FIXED EVALUATION POPULATION ACROSS K")
    print("=" * 80)

    reference_ids = None

    for k, df in history_keys.items():
        ids = set(df["eval_idx"])

        if reference_ids is None:
            reference_ids = ids
            print(f"  {k}: reference population established")
            continue

        if ids == reference_ids:
            print(f"  {k}: same evaluation population: PASS")
        else:
            print(f"  {k}: same evaluation population: FAIL")
            failed = True

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 8 SCHEMA AUDIT SUMMARY")
    print("=" * 80)

    if failed:
        print("\nRESULT: FAIL")
        print("Do NOT start model training yet.")
        sys.exit(1)

    print("\nRESULT: PASS")
    print("\nSchema contract:")
    print("  evaluation_events.parquet → labels")
    print("  train/validation/test.parquet → split membership via eval_idx")
    print("  phase7/history_K*.parquet → histories via eval_idx")
    print("  evaluation set is fixed across K")
    print("  train/validation/test are disjoint")
    print("\nReady for Phase 8 dataset/model implementation.")


if __name__ == "__main__":
    main()