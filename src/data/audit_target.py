from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"
RANDOM_LOG = RAW_DIR / "log_random_4_22_to_5_08_1k.csv"

TARGET_COLS = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
]


def audit_binary_target(df, col):
    s = df[col]

    print(f"\n{col}:")
    print(f"  dtype: {s.dtype}")
    print(f"  missing: {s.isna().sum():,}")
    print(f"  unique: {sorted(s.dropna().unique().tolist())}")

    if not s.empty:
        counts = s.value_counts(dropna=False).sort_index()

        for value, count in counts.items():
            print(
                f"  value={value}: "
                f"{count:,} "
                f"({count / len(s):.2%})"
            )


def audit_continuous_target(df, col):
    s = df[col]

    print(f"\n{col}:")
    print(f"  dtype: {s.dtype}")
    print(f"  missing: {s.isna().sum():,}")
    print(f"  min: {s.min():,.2f}")
    print(f"  median: {s.median():,.2f}")
    print(f"  mean: {s.mean():,.2f}")
    print(f"  max: {s.max():,.2f}")

    print(
        f"  zero: {(s == 0).sum():,} "
        f"({(s == 0).mean():.2%})"
    )


def audit_random_exposure(df):
    print("\n" + "=" * 80)
    print("RANDOM EXPOSURE AUDIT")
    print("=" * 80)

    if "is_rand" not in df.columns:
        print("FAIL - is_rand column not found")
        return

    counts = df["is_rand"].value_counts(dropna=False).sort_index()

    print("is_rand distribution:")

    for value, count in counts.items():
        print(
            f"  value={value}: "
            f"{count:,} "
            f"({count / len(df):.2%})"
        )

    if set(df["is_rand"].dropna().unique()).issubset({0, 1}):
        print("Random exposure encoding: PASS")
    else:
        print("Random exposure encoding: REVIEW")


def audit_target_relationships(df):
    print("\n" + "=" * 80)
    print("TARGET RELATIONSHIPS")
    print("=" * 80)

    binary_cols = [
        c for c in TARGET_COLS
        if c in df.columns
        and set(df[c].dropna().unique()).issubset({0, 1})
    ]

    if not binary_cols:
        print("No binary target columns found")
        return

    print("Binary target correlations:")

    corr = df[binary_cols].corr()

    print(corr.round(3).to_string())


def audit_target_missingness(df):
    print("\n" + "=" * 80)
    print("TARGET MISSINGNESS")
    print("=" * 80)

    for col in TARGET_COLS:
        if col not in df.columns:
            print(f"{col:<22} NOT FOUND")
            continue

        missing = df[col].isna().sum()

        print(
            f"{col:<22} "
            f"missing={missing:>8,} "
            f"({missing / len(df):.2%})"
        )


def main():
    print("=" * 80)
    print("AUDIT KUAIRAND TARGET SEMANTICS")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {RANDOM_LOG}")

    if not RANDOM_LOG.exists():
        raise FileNotFoundError(RANDOM_LOG)

    df = pd.read_csv(RANDOM_LOG)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    audit_random_exposure(df)

    print("\n" + "=" * 80)
    print("BINARY TARGET AUDIT")
    print("=" * 80)

    binary_candidates = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
    ]

    for col in binary_candidates:
        if col in df.columns:
            audit_binary_target(df, col)

    print("\n" + "=" * 80)
    print("CONTINUOUS BEHAVIOR AUDIT")
    print("=" * 80)

    continuous_candidates = [
        "play_time_ms",
        "duration_ms",
        "profile_stay_time",
        "comment_stay_time",
    ]

    for col in continuous_candidates:
        if col in df.columns:
            audit_continuous_target(df, col)

    audit_target_missingness(df)
    audit_target_relationships(df)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()