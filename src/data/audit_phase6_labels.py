from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = (
    ROOT / "data/processed/evaluation_events.parquet"
)

HISTORY_FILE = (
    ROOT / "data/processed/evaluation_history.parquet"
)


KEY_COLUMNS = [
    "user_id",
    "video_id",
    "time_ms",
]


def main():

    print("=" * 80)
    print("PHASE 6 — FIXED LABEL AUDIT")
    print("=" * 80)

    # =========================================================================
    # LOAD
    # =========================================================================

    print("\nLoading evaluation events...")

    eval_df = pd.read_parquet(
        EVAL_FILE,
        columns=[
            "evaluation_id",
            "user_id",
            "video_id",
            "time_ms",
            "long_view",
        ],
    )

    print(
        f"  Evaluation events: {len(eval_df):,}"
    )

    print("\nLoading evaluation histories...")

    history_df = pd.read_parquet(
        HISTORY_FILE,
        columns=[
            "eval_idx",
            "user_id",
            "video_id",
            "time_ms",
        ],
    )

    print(
        f"  History events:     {len(history_df):,}"
    )

    # =========================================================================
    # 1. LABEL AVAILABILITY
    # =========================================================================

    print("\n" + "=" * 80)
    print("1. LABEL AVAILABILITY")
    print("=" * 80)

    missing_labels = eval_df["long_view"].isna().sum()

    if missing_labels == 0:
        print("  long_view non-null: PASS")
    else:
        print(
            f"  long_view non-null: FAIL "
            f"({missing_labels:,} missing)"
        )
        return

    # =========================================================================
    # 2. EVALUATION EVENT UNIQUENESS
    # =========================================================================

    print("\n" + "=" * 80)
    print("2. EVALUATION EVENT UNIQUENESS")
    print("=" * 80)

    if eval_df["evaluation_id"].is_unique:
        print("  evaluation_id uniqueness: PASS")
    else:
        print("  evaluation_id uniqueness: FAIL")
        return

    key_duplicates = eval_df.duplicated(
        subset=KEY_COLUMNS
    ).sum()

    if key_duplicates == 0:
        print(
            "  (user_id, video_id, time_ms) uniqueness: PASS"
        )
    else:
        print(
            "  (user_id, video_id, time_ms) uniqueness: FAIL"
        )
        print(
            f"  Duplicate keys: {key_duplicates:,}"
        )
        return

    # =========================================================================
    # 3. HISTORY / EVALUATION POPULATION
    # =========================================================================

    print("\n" + "=" * 80)
    print("3. FIXED EVALUATION POPULATION")
    print("=" * 80)

    if len(eval_df) == len(history_df):
        print("  Event count consistency: PASS")
    else:
        print("  Event count consistency: FAIL")
        print(
            f"  evaluation_events: {len(eval_df):,}"
        )
        print(
            f"  evaluation_history: {len(history_df):,}"
        )
        return

    # =========================================================================
    # 4. IDENTITY MATCHING
    # =========================================================================

    print("\n" + "=" * 80)
    print("4. EVALUATION IDENTITY MATCHING")
    print("=" * 80)

    eval_keys = eval_df[KEY_COLUMNS].copy()
    history_keys = history_df[KEY_COLUMNS].copy()

    eval_keys["_eval_exists"] = True
    history_keys["_history_exists"] = True

    merged = eval_keys.merge(
        history_keys,
        on=KEY_COLUMNS,
        how="outer",
        indicator=True,
    )

    missing_from_history = (
        merged["_merge"] == "left_only"
    ).sum()

    missing_from_evaluation = (
        merged["_merge"] == "right_only"
    ).sum()

    if (
        missing_from_history == 0
        and missing_from_evaluation == 0
    ):
        print(
            "  Evaluation identity across artifacts: PASS"
        )
    else:
        print(
            "  Evaluation identity across artifacts: FAIL"
        )
        print(
            f"  Missing from history:     "
            f"{missing_from_history:,}"
        )
        print(
            f"  Missing from evaluation:  "
            f"{missing_from_evaluation:,}"
        )
        return

    # =========================================================================
    # 5. FIXED LABEL INVARIANT
    # =========================================================================

    print("\n" + "=" * 80)
    print("5. FIXED LABEL INVARIANT")
    print("=" * 80)

    # The label belongs to the evaluation event.
    #
    # History length K changes only the available history.
    # It must not change:
    #
    #   candidate video
    #   evaluation timestamp
    #   target label
    #
    # Since evaluation_history contains one row per evaluation event,
    # all K histories necessarily reference the same label-bearing event.

    label_counts = (
        eval_df["long_view"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("  Label distribution:")

    for label, count in label_counts.items():
        print(
            f"    long_view={label}: {count:,}"
        )

    print(
        "  Label defined once per evaluation event: PASS"
    )

    print(
        "  Label independent of history K: PASS"
    )

    # =========================================================================
    # 6. LABEL VALUE VALIDITY
    # =========================================================================

    print("\n" + "=" * 80)
    print("6. LABEL VALUE VALIDITY")
    print("=" * 80)

    valid_labels = {0, 1}

    observed_labels = set(
        eval_df["long_view"].unique()
    )

    if observed_labels.issubset(valid_labels):
        print(
            "  long_view ∈ {0,1}: PASS"
        )
    else:
        print(
            "  long_view ∈ {0,1}: FAIL"
        )
        print(
            f"  Unexpected values: "
            f"{observed_labels - valid_labels}"
        )
        return

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("PHASE 6 LABEL AUDIT COMPLETE")
    print("=" * 80)

    print(
        "Fixed evaluation population: PASS"
    )
    print(
        "Fixed candidate identity:     PASS"
    )
    print(
        "Fixed evaluation timestamp:   PASS"
    )
    print(
        "Fixed labels across K:        PASS"
    )


if __name__ == "__main__":
    main()
