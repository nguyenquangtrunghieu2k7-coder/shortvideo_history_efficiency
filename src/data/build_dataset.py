from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data/raw/KuaiRand-1K/data"
PROCESSED_DIR = ROOT / "data/processed"

EVAL_FILE = RAW_DIR / "log_random_4_22_to_5_08_1k.csv"
VIDEO_FILE = RAW_DIR / "video_features_basic_1k.csv"

SPLITS = {
    "train": PROCESSED_DIR / "train.parquet",
    "validation": PROCESSED_DIR / "validation.parquet",
    "test": PROCESSED_DIR / "test.parquet",
}

TARGET = "is_click"

WINDOWS = [1, 5, 10, 20, 50, 100]

IDENTITY = [
    "eval_idx",
    "user_id",
    "video_id",
    "time_ms",
]

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

METADATA_COLS = [
    "video_id",
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "video_duration",
    "server_width",
    "server_height",
    "music_id",
    "music_type",
    "tag",
]


def banner(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_history(path):
    print(f"Loading:\n  {path}")

    df = pd.read_parquet(path)

    required = IDENTITY + [
        "history_len_full",
        "history_full",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"History missing columns: {missing}")

    if df["eval_idx"].duplicated().any():
        raise ValueError("Duplicate eval_idx in history.")

    return df


def load_targets():
    banner("LOADING EVALUATION TARGETS")

    print(f"Loading:\n  {EVAL_FILE}")

    df = pd.read_csv(EVAL_FILE)

    required = ["user_id", "video_id", "time_ms"] + TARGET_COLS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Evaluation file missing columns: {missing}")

    print(f"Raw evaluation rows: {len(df):,}")

    # ------------------------------------------------------------
    # Raw evaluation contains one duplicate key.
    # Do NOT blindly reject it.
    # ------------------------------------------------------------
    key = ["user_id", "video_id", "time_ms"]

    dup_mask = df.duplicated(key, keep=False)
    dup_rows = df.loc[dup_mask].copy()

    if len(dup_rows):
        print()
        print("Duplicate evaluation keys detected:")
        print(f"  Rows involved: {len(dup_rows):,}")
        print(f"  Duplicate keys: {dup_rows[key].drop_duplicates().shape[0]:,}")

        # Check whether duplicated records have identical targets.
        grouped = dup_rows.groupby(key, sort=False)

        bad_keys = []

        for k, g in grouped:
            for col in TARGET_COLS:
                if g[col].nunique(dropna=False) > 1:
                    bad_keys.append((k, col))

        if bad_keys:
            print()
            print("ERROR: duplicate evaluation keys have conflicting targets.")

            for k, col in bad_keys[:10]:
                print(f"  key={k}, column={col}")

            raise ValueError(
                "Cannot safely deduplicate evaluation targets because "
                "duplicate rows have conflicting target values."
            )

        # Safe: duplicate rows are semantically identical.
        before = len(df)

        df = df.drop_duplicates(
            subset=key,
            keep="first",
        ).copy()

        print(
            f"  Safe deduplication: {before:,} -> {len(df):,} rows"
        )

    else:
        print("Duplicate evaluation keys: 0")

    print(f"Final evaluation targets: {len(df):,}")

    return df


def align_targets(history, targets):
    banner("ALIGN TARGETS WITH HISTORY")

    key = ["user_id", "video_id", "time_ms"]

    # History is authoritative for evaluation-event identity.
    merged = history[key].merge(
        targets[key + TARGET_COLS],
        on=key,
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(history):
        raise ValueError(
            "Target alignment changed row count."
        )

    missing = merged[TARGET].isna().sum()

    print(f"History events: {len(history):,}")
    print(f"Target rows:     {len(targets):,}")
    print(f"Aligned rows:    {len(merged):,}")
    print(f"Missing targets: {missing:,}")

    if missing:
        raise ValueError(
            f"{missing:,} history events could not be aligned with targets."
        )

    # Binary target must remain binary.
    values = set(merged[TARGET].unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            f"Invalid {TARGET} values after alignment: {values}"
        )

    print("Target alignment: PASS")

    return merged


def load_metadata():
    banner("LOADING VIDEO METADATA")

    print(f"Loading:\n  {VIDEO_FILE}")

    cols = METADATA_COLS

    df = pd.read_csv(
        VIDEO_FILE,
        usecols=cols,
    )

    if df["video_id"].duplicated().any():
        raise ValueError(
            "video_features_basic_1k.csv contains duplicate video_id."
        )

    print(f"Metadata rows: {len(df):,}")
    print(f"Unique videos: {df['video_id'].nunique():,}")

    return df


def add_history_features(df):
    # Keep only the representations required by the experiment.
    keep = [
        "history_len_full",
        "history_full",
    ]

    for w in WINDOWS:
        keep += [
            f"history_{w}",
            f"history_{w}_len",
        ]

    missing = [c for c in keep if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing history columns: {missing}"
        )

    return df


def build_split(
    name,
    split_path,
    history,
    targets,
    metadata,
):
    banner(f"BUILD DATASET: {name.upper()}")

    print(f"Loading split:\n  {split_path}")

    split = pd.read_parquet(split_path)

    if "eval_idx" not in split.columns:
        raise ValueError(
            f"{name}: missing eval_idx."
        )

    if split["eval_idx"].duplicated().any():
        raise ValueError(
            f"{name}: duplicate eval_idx."
        )

    print(f"Split rows: {len(split):,}")

    # ------------------------------------------------------------
    # Select history rows by eval_idx.
    # This preserves the exact temporal split.
    # ------------------------------------------------------------
    base = history[
        history["eval_idx"].isin(split["eval_idx"])
    ].copy()

    if len(base) != len(split):
        raise ValueError(
            f"{name}: history coverage mismatch. "
            f"split={len(split):,}, history={len(base):,}"
        )

    base = split[["eval_idx"]].merge(
        base,
        on="eval_idx",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------
    # Attach targets through the already validated key mapping.
    # ------------------------------------------------------------
    target_cols = ["eval_idx"] + TARGET_COLS

    aligned_targets = history[
        ["eval_idx", "user_id", "video_id", "time_ms"]
    ].merge(
        targets,
        on=["user_id", "video_id", "time_ms"],
        how="left",
        validate="one_to_one",
    )

    target_part = aligned_targets[
        target_cols
    ]

    base = base.merge(
        target_part,
        on="eval_idx",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------
    # Attach static video metadata.
    # ------------------------------------------------------------
    before = len(base)

    base = base.merge(
        metadata,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    if len(base) != before:
        raise ValueError(
            f"{name}: metadata merge changed row count."
        )

    missing_metadata = base["author_id"].isna().sum()

    print(
        f"Missing metadata: "
        f"{missing_metadata:,} "
        f"({missing_metadata / len(base) * 100:.2f}%)"
    )

    if missing_metadata:
        raise ValueError(
            f"{name}: candidate videos missing metadata."
        )

    # ------------------------------------------------------------
    # Final feature selection.
    # ------------------------------------------------------------
    feature_cols = [
        "eval_idx",
        "user_id",
        "video_id",
        "time_ms",
    ]

    feature_cols += [
        "history_len_full",
        "history_full",
    ]

    for w in WINDOWS:
        feature_cols += [
            f"history_{w}",
            f"history_{w}_len",
        ]

    feature_cols += [
        "author_id",
        "video_type",
        "upload_dt",
        "upload_type",
        "visible_status",
        "video_duration",
        "server_width",
        "server_height",
        "music_id",
        "music_type",
        "tag",
    ]

    feature_cols += TARGET_COLS

    dataset = base[feature_cols].copy()

    # ------------------------------------------------------------
    # Final validation.
    # ------------------------------------------------------------
    if len(dataset) != len(split):
        raise ValueError(
            f"{name}: final row count mismatch."
        )

    if dataset["eval_idx"].duplicated().any():
        raise ValueError(
            f"{name}: duplicate eval_idx."
        )

    if dataset[TARGET].isna().any():
        raise ValueError(
            f"{name}: missing target."
        )

    # No post-exposure behavior as features.
    forbidden = [
        c for c in TARGET_COLS
        if c != TARGET and c in dataset.columns
    ]

    # These are retained as labels/analysis targets, but NOT model features.
    # The final model code must select input features explicitly.

    out = PROCESSED_DIR / f"{name}_dataset.parquet"

    dataset.to_parquet(
        out,
        index=False,
    )

    print()
    print(f"Final rows:    {len(dataset):,}")
    print(f"Final columns: {len(dataset.columns):,}")

    print()
    print("Target distribution:")
    print(
        dataset[TARGET]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(f"Saved:\n  {out}")

    return dataset


def validate_cross_split(datasets):
    banner("CROSS-SPLIT VALIDATION")

    ids = {
        name: set(df["eval_idx"])
        for name, df in datasets.items()
    }

    names = list(ids)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            overlap = ids[a] & ids[b]

            print(
                f"{a} ∩ {b}: {len(overlap):,}"
            )

            if overlap:
                raise ValueError(
                    f"Split overlap detected: {a} / {b}"
                )

    print("Split disjointness: PASS")


def main():
    banner("BUILD KUAIRAND MODEL DATASETS")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. History is the authoritative evaluation index.
    history = pd.read_parquet(
        PROCESSED_DIR / "evaluation_history.parquet"
    )

    print(
        f"Evaluation history events: {len(history):,}"
    )

    # 2. Load and safely deduplicate raw evaluation targets.
    raw_targets = load_targets()

    # 3. Align targets to evaluation history.
    targets = align_targets(
        history,
        raw_targets,
    )

    # 4. Load static video metadata.
    metadata = load_metadata()

    # 5. Build each temporal split.
    datasets = {}

    for name, path in SPLITS.items():
        datasets[name] = build_split(
            name=name,
            split_path=path,
            history=history,
            targets=targets,
            metadata=metadata,
        )

    # 6. Cross-split integrity.
    validate_cross_split(datasets)

    banner("DATASET BUILD COMPLETE")

    for name, df in datasets.items():
        print(
            f"{name:10s}: "
            f"{len(df):,} rows | "
            f"{len(df.columns):,} columns"
        )

    print()
    print("All dataset validation checks: PASS")


if __name__ == "__main__":
    main()