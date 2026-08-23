from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "KuaiRand-1K"
HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"

KEY_COLS = ["user_id", "video_id", "time_ms"]


def find_csv_files():
    files = sorted(RAW_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    return files


def inspect_file(path):
    print("\n" + "-" * 80)
    print(f"FILE: {path.name}")
    print("-" * 80)

    df = pd.read_csv(path, nrows=5)

    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nSample:")
    print(df.head())


def find_evaluation_candidates(files):
    keywords = [
        "random",
        "evaluation",
        "eval",
    ]

    candidates = [
        p for p in files
        if any(k in p.name.lower() for k in keywords)
    ]

    return candidates


def audit_history_keys():
    print("\n" + "=" * 80)
    print("EVALUATION HISTORY KEY AUDIT")
    print("=" * 80)

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    df = pd.read_parquet(HISTORY_PATH)

    missing = [c for c in KEY_COLS if c not in df.columns]

    if missing:
        print(f"FAIL - Missing key columns: {missing}")
        return

    print(f"Rows: {len(df):,}")

    for col in KEY_COLS:
        print(
            f"{col:<10} "
            f"unique={df[col].nunique():,} "
            f"missing={df[col].isna().sum():,}"
        )

    duplicates = df.duplicated(subset=KEY_COLS).sum()

    print(f"\nDuplicate evaluation keys: {duplicates:,}")

    if duplicates == 0:
        print("Evaluation key uniqueness: PASS")
    else:
        print("Evaluation key uniqueness: FAIL")


def main():
    print("=" * 80)
    print("AUDIT KUAIRAND EVALUATION EVENTS")
    print("=" * 80)

    print(f"\nRaw directory:")
    print(f"  {RAW_DIR}")

    files = find_csv_files()

    print(f"\nTotal CSV files found: {len(files)}")

    print("\nCandidate evaluation files:")

    candidates = find_evaluation_candidates(files)

    if not candidates:
        print("  None found")
    else:
        for path in candidates:
            print(f"  - {path.relative_to(ROOT)}")

    print("\n" + "=" * 80)
    print("RAW FILE SCHEMA INSPECTION")
    print("=" * 80)

    for path in candidates:
        inspect_file(path)

    audit_history_keys()

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()