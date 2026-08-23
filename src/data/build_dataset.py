from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"
PROCESSED_DIR = ROOT / "data" / "processed"

EVAL_PATH = RAW_DIR / "log_random_4_22_to_5_08_1k.csv"
VIDEO_PATH = RAW_DIR / "video_features_basic_1k.csv"

SPLITS = {
    "train": PROCESSED_DIR / "train.parquet",
    "validation": PROCESSED_DIR / "validation.parquet",
    "test": PROCESSED_DIR / "test.parquet",
}

OUTPUT_DIR = PROCESSED_DIR / "dataset"

TARGET = "is_click"

EVAL_KEY = [
    "user_id",
    "video_id",
    "time_ms",
]

WINDOWS = [1, 5, 10, 20, 50, 100, "full"]

CAT_FEATURES = [
    "author_id",
    "video_type",
    "upload_type",
    "visible_status",
    "music_id",
    "music_type",
    "tag",
]

NUM_FEATURES = [
    "video_duration",
    "server_width",
    "server_height",
]


def load_video_metadata():
    df = pd.read_csv(VIDEO_PATH)

    if df["video_id"].duplicated().any():
        raise ValueError(
            "Duplicate video_id in video metadata."
        )

    cols = (
        ["video_id"]
        + CAT_FEATURES
        + NUM_FEATURES
        + ["upload_dt"]
    )

    df = df[cols].copy()

    # Categorical features
    for col in CAT_FEATURES:
        df[col] = (
            df[col]
            .fillna("UNK")
            .astype(str)
        )

    # Numerical features
    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        df[f"{col}_missing"] = (
            df[col]
            .isna()
            .astype(np.int8)
        )

        median = df[col].median()

        df[col] = df[col].fillna(median)

    # Upload time
    df["upload_dt"] = pd.to_datetime(
        df["upload_dt"],
        errors="coerce",
    )

    return df


def load_targets():
    df = pd.read_csv(
        EVAL_PATH,
        usecols=EVAL_KEY + [TARGET],
    )

    if df.duplicated(EVAL_KEY).any():
        raise ValueError(
            "Duplicate evaluation keys in raw evaluation file."
        )

    df[TARGET] = df[TARGET].astype(np.int8)

    return df


def attach_targets(df, targets):
    before = len(df)

    df = df.merge(
        targets,
        on=EVAL_KEY,
        how="left",
        validate="one_to_one",
    )

    if len(df) != before:
        raise ValueError(
            "Row count changed after target merge."
        )

    missing = df[TARGET].isna().sum()

    if missing:
        raise ValueError(
            f"Missing targets after merge: {missing:,}"
        )

    df[TARGET] = df[TARGET].astype(np.int8)

    return df


def attach_video_features(df, video_df):
    before = len(df)

    df = df.merge(
        video_df,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    if len(df) != before:
        raise ValueError(
            "Row count changed after video metadata merge."
        )

    # Every evaluation candidate must have metadata.
    metadata_missing = df[
        CAT_FEATURES + NUM_FEATURES
    ].isna().all(axis=1)

    if metadata_missing.any():
        raise ValueError(
            "Missing video metadata for "
            f"{metadata_missing.sum():,} events."
        )

    # Convert upload timestamp to age at exposure.
    exposure_time = pd.to_datetime(
        df["time_ms"],
        unit="ms",
    )

    age_days = (
        exposure_time - df["upload_dt"]
    ).dt.total_seconds() / 86400.0

    df["upload_dt_missing"] = (
        df["upload_dt"]
        .isna()
        .astype(np.int8)
    )

    df["video_age_days"] = (
        age_days
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # upload_dt itself is not exposed to the model.
    df.drop(
        columns=["upload_dt"],
        inplace=True,
    )

    return df


def validate_history(df, history_col, expected):
    lengths = df[history_col].map(len)

    if expected == "full":
        expected_lengths = df["history_full_len"]

        if not lengths.equals(expected_lengths):
            raise ValueError(
                f"{history_col}: history length mismatch."
            )

    else:
        if (lengths > expected).any():
            raise ValueError(
                f"{history_col}: length exceeds W={expected}."
            )


def get_feature_columns():
    return (
        CAT_FEATURES
        + NUM_FEATURES
        + [
            f"{col}_missing"
            for col in NUM_FEATURES
        ]
        + [
            "upload_dt_missing",
            "video_age_days",
        ]
    )


def build_window(
    df,
    name,
    history_col,
    history_len_col,
    output_dir,
    feature_cols,
):
    validate_history(
        df,
        history_col,
        (
            "full"
            if history_col == "history_full"
            else int(
                history_col.replace(
                    "history_",
                    "",
                )
            )
        ),
    )

    cols = [
        "eval_idx",
        "user_id",
        "video_id",
        "time_ms",
        TARGET,
    ]

    cols += feature_cols
    cols += [
        history_col,
        history_len_col,
    ]

    out = df[cols].copy()

    path = output_dir / f"W_{name}.parquet"

    out.to_parquet(
        path,
        index=False,
    )

    return out, path


def build_split(
    name,
    split_path,
    targets,
    video_df,
    feature_cols,
):
    print("\n" + "=" * 80)
    print(f"BUILD DATASET: {name.upper()}")
    print("=" * 80)

    print("\nLoading:")
    print(f"  {split_path}")

    df = pd.read_parquet(split_path)

    print(
        f"Rows before enrichment: {len(df):,}"
    )

    # 1. Attach target from raw randomized evaluation.
    df = attach_targets(
        df,
        targets,
    )

    # 2. Attach candidate metadata.
    df = attach_video_features(
        df,
        video_df,
    )

    print(
        f"Rows after enrichment:  {len(df):,}"
    )

    output_dir = OUTPUT_DIR / name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nBuilding windows:")

    for w in WINDOWS:
        if w == "full":
            history_col = "history_full"
            history_len_col = "history_full_len"
            label = "full"
        else:
            history_col = f"history_{w}"
            history_len_col = f"history_{w}_len"
            label = str(w)

        out, path = build_window(
            df=df,
            name=label,
            history_col=history_col,
            history_len_col=history_len_col,
            output_dir=output_dir,
            feature_cols=feature_cols,
        )

        print(
            f"  W={label:<4} "
            f"rows={len(out):,} "
            f"-> {path}"
        )


def main():
    print("=" * 80)
    print("BUILD KUAIRAND MODEL DATASETS")
    print("=" * 80)

    for path in [
        EVAL_PATH,
        VIDEO_PATH,
        *SPLITS.values(),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("\nLoading video metadata:")
    print(f"  {VIDEO_PATH}")

    video_df = load_video_metadata()

    print(
        f"Unique video metadata: "
        f"{len(video_df):,}"
    )

    print("\nLoading evaluation targets:")
    print(f"  {EVAL_PATH}")

    targets = load_targets()

    print(
        f"Evaluation events: "
        f"{len(targets):,}"
    )

    feature_cols = get_feature_columns()

    print("\nModel feature columns:")

    for col in feature_cols:
        print(f"  - {col}")

    for name, split_path in SPLITS.items():
        build_split(
            name=name,
            split_path=split_path,
            targets=targets,
            video_df=video_df,
            feature_cols=feature_cols,
        )

    print("\n" + "=" * 80)
    print("DATASET BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()