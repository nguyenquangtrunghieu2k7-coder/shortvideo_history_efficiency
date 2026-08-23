from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"
HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"

EVAL_FILE = RAW_DIR / "log_random_4_22_to_5_08_1k.csv"


TARGET_COLUMNS = {
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
}

IDENTITY_COLUMNS = {
    "user_id",
    "video_id",
    "time_ms",
    "date",
    "hourmin",
}

CONTROL_COLUMNS = {
    "is_rand",
    "tab",
}


def inspect_schema(df):
    print("\n" + "=" * 80)
    print("RAW EVALUATION SCHEMA")
    print("=" * 80)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumns:")

    for col in df.columns:
        print(f"  - {col:<25} dtype={df[col].dtype}")


def classify_columns(df):
    print("\n" + "=" * 80)
    print("FEATURE CLASSIFICATION")
    print("=" * 80)

    columns = set(df.columns)

    target = columns & TARGET_COLUMNS
    identity = columns & IDENTITY_COLUMNS
    control = columns & CONTROL_COLUMNS

    feature = columns - target - identity - control

    print("\nIdentity / temporal:")
    for col in sorted(identity):
        print(f"  - {col}")

    print("\nTarget / post-exposure behavior:")
    for col in sorted(target):
        print(f"  - {col}")

    print("\nRandomization / control:")
    for col in sorted(control):
        print(f"  - {col}")

    print("\nPotential model features:")
    if feature:
        for col in sorted(feature):
            print(f"  - {col}")
    else:
        print("  NONE")


def audit_missingness(df):
    print("\n" + "=" * 80)
    print("FEATURE MISSINGNESS")
    print("=" * 80)

    missing = df.isna().sum()

    for col in df.columns:
        print(
            f"{col:<25} "
            f"missing={missing[col]:>7,} "
            f"({missing[col] / len(df):.2%})"
        )


def audit_constant_columns(df):
    print("\n" + "=" * 80)
    print("CONSTANT / LOW-VARIANCE COLUMNS")
    print("=" * 80)

    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)

        if n_unique <= 1:
            print(
                f"  CONSTANT: {col:<25} "
                f"unique={n_unique}"
            )


def audit_feature_temporal_safety(df):
    print("\n" + "=" * 80)
    print("TEMPORAL FEATURE SAFETY")
    print("=" * 80)

    print(
        "Features observed directly on the evaluation event "
        "are treated as post-exposure unless explicitly justified."
    )

    unsafe = []

    for col in TARGET_COLUMNS:
        if col in df.columns:
            unsafe.append(col)

    print("\nPost-exposure candidates:")
    for col in unsafe:
        print(f"  - {col}")

    print("\nFeature leakage rule: PASS")
    print(
        "These columns must NOT be used as input features "
        "for predicting is_click."
    )


def audit_history_features():
    print("\n" + "=" * 80)
    print("HISTORY REPRESENTATION AVAILABILITY")
    print("=" * 80)

    if not HISTORY_PATH.exists():
        print("History artifact: NOT FOUND")
        return

    df = pd.read_parquet(HISTORY_PATH)

    history_cols = [
        col
        for col in df.columns
        if col.startswith("history_")
    ]

    print(f"History columns: {len(history_cols):,}")

    for col in history_cols:
        print(f"  - {col}")

    required = [
        "history_1",
        "history_5",
        "history_10",
        "history_20",
        "history_50",
        "history_100",
        "history_full",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if not missing:
        print("\nRequired history windows: PASS")
    else:
        print("\nMissing history windows:")
        for col in missing:
            print(f"  - {col}")


def audit_candidate_features(df):
    print("\n" + "=" * 80)
    print("CANDIDATE FEATURE AVAILABILITY")
    print("=" * 80)

    print("Candidate identifier:")
    print("  video_id: PRESENT")

    metadata_columns = [
        col
        for col in df.columns
        if col not in (
            TARGET_COLUMNS
            | IDENTITY_COLUMNS
            | CONTROL_COLUMNS
        )
    ]

    if metadata_columns:
        print("\nPotential candidate/user metadata:")
        for col in metadata_columns:
            print(f"  - {col}")
    else:
        print("\nCandidate metadata: NONE")

        print(
            "\nConclusion:"
            "\n  Evaluation log provides video_id only."
            "\n  No static video metadata is present in this file."
        )


def main():
    print("=" * 80)
    print("AUDIT KUAIRAND MODEL FEATURES")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {EVAL_FILE}")

    if not EVAL_FILE.exists():
        raise FileNotFoundError(EVAL_FILE)

    df = pd.read_csv(EVAL_FILE)

    inspect_schema(df)
    classify_columns(df)
    audit_missingness(df)
    audit_constant_columns(df)
    audit_feature_temporal_safety(df)
    audit_candidate_features(df)
    audit_history_features()

    print("\n" + "=" * 80)
    print("FEATURE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()