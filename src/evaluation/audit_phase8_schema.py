from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

EVALUATION_EVENTS_PATH = PROCESSED_DIR / "evaluation_events.parquet"
EVALUATION_HISTORY_PATH = PROCESSED_DIR / "evaluation_history.parquet"

TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VALIDATION_PATH = PROCESSED_DIR / "validation.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"


EXPECTED_N_EVAL = 43_027

EXPECTED_SPLIT_ROWS = {
    "train": 30_118,
    "validation": 6_454,
    "test": 6_455,
}


EVALUATION_EVENT_REQUIRED = {
    "evaluation_id",
    "user_id",
    "video_id",
    "time_ms",
    "long_view",
}


EVALUATION_HISTORY_REQUIRED = {
    "eval_idx",
    "user_id",
    "video_id",
    "time_ms",
    "history_len_full",
    "history_1",
    "history_1_len",
    "history_5",
    "history_5_len",
    "history_10",
    "history_10_len",
    "history_20",
    "history_20_len",
    "history_50",
    "history_50_len",
    "history_100",
    "history_100_len",
    "history_full",
    "history_full_len",
}


SPLIT_REQUIRED = {
    "eval_idx",
    "user_id",
    "video_id",
    "time_ms",
}


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        raise AssertionError(message)


def check_file(path: Path, label: str) -> None:
    check(path.exists(), f"File exists: {path}")


def check_required_columns(
    df: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(df.columns))

    check(
        not missing,
        f"{label} contains required columns"
        + (f" | missing={missing}" if missing else ""),
    )


def audit_evaluation_events() -> pd.DataFrame:
    print()
    print("=" * 80)
    print("1. EVALUATION EVENT ARTIFACT")
    print("=" * 80)

    check_file(
        EVALUATION_EVENTS_PATH,
        "evaluation_events",
    )

    df = pd.read_parquet(EVALUATION_EVENTS_PATH)

    print(f"  rows: {len(df):,}")
    print(f"  columns: {len(df.columns)}")

    check(
        len(df) == EXPECTED_N_EVAL,
        f"evaluation_events rows = {EXPECTED_N_EVAL:,}",
    )

    check_required_columns(
        df,
        EVALUATION_EVENT_REQUIRED,
        "evaluation_events",
    )

    check(
        "evaluation_id" in df.columns,
        "evaluation_events uses evaluation_id",
    )

    check(
        "eval_idx" not in df.columns,
        "evaluation_events does NOT incorrectly require eval_idx",
    )

    check(
        df["evaluation_id"].is_unique,
        "evaluation_id is unique",
    )

    expected_ids = np.arange(len(df), dtype=np.int64)

    check(
        np.array_equal(
            np.sort(df["evaluation_id"].to_numpy()),
            expected_ids,
        ),
        "evaluation_id covers 0..N-1",
    )

    check(
        set(df["long_view"].dropna().unique()).issubset({0, 1}),
        "long_view is binary",
    )

    return df


def audit_evaluation_history(
    eval_events: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 80)
    print("2. EVALUATION HISTORY ARTIFACT")
    print("=" * 80)

    check_file(
        EVALUATION_HISTORY_PATH,
        "evaluation_history",
    )

    df = pd.read_parquet(EVALUATION_HISTORY_PATH)

    print(f"  rows: {len(df):,}")
    print(f"  columns: {len(df.columns)}")

    check(
        len(df) == EXPECTED_N_EVAL,
        f"evaluation_history rows = {EXPECTED_N_EVAL:,}",
    )

    check_required_columns(
        df,
        EVALUATION_HISTORY_REQUIRED,
        "evaluation_history",
    )

    check(
        "eval_idx" in df.columns,
        "evaluation_history uses eval_idx",
    )

    check(
        "evaluation_id" not in df.columns,
        "evaluation_history does NOT incorrectly require evaluation_id",
    )

    check(
        df["eval_idx"].is_unique,
        "eval_idx is unique in evaluation_history",
    )

    event_ids = set(
        eval_events["evaluation_id"].astype(np.int64)
    )

    history_ids = set(
        df["eval_idx"].astype(np.int64)
    )

    check(
        history_ids == event_ids,
        "evaluation_history.eval_idx maps exactly to evaluation_events.evaluation_id",
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # All mappings below are performed by ID.
    #
    # DO NOT compare evaluation_history rows against
    # evaluation_events rows positionally because the row orders differ.
    # ------------------------------------------------------------------

    events_by_id = eval_events.set_index("evaluation_id")

    mapped_users = events_by_id["user_id"].reindex(df["eval_idx"])

    check(
        mapped_users.notna().all(),
        "every history row maps to an evaluation event",
    )

    check(
        (
            df["user_id"].to_numpy()
            == mapped_users.to_numpy()
        ).all(),
        "user_id mapping is consistent",
    )

    mapped_videos = events_by_id["video_id"].reindex(df["eval_idx"])

    check(
        (
            df["video_id"].to_numpy()
            == mapped_videos.to_numpy()
        ).all(),
        "video_id mapping is consistent",
    )

    mapped_times = events_by_id["time_ms"].reindex(df["eval_idx"])

    check(
        (
            df["time_ms"].to_numpy()
            == mapped_times.to_numpy()
        ).all(),
        "time_ms mapping is consistent",
    )

    # ------------------------------------------------------------------
    # History length consistency
    # ------------------------------------------------------------------

    history_columns = {
        1: ("history_1", "history_1_len"),
        5: ("history_5", "history_5_len"),
        10: ("history_10", "history_10_len"),
        20: ("history_20", "history_20_len"),
        50: ("history_50", "history_50_len"),
        100: ("history_100", "history_100_len"),
        "full": ("history_full", "history_full_len"),
    }

    for k, (history_col, len_col) in history_columns.items():
        lengths = df[history_col].apply(len)

        check(
            (
                lengths.to_numpy()
                == df[len_col].to_numpy()
            ).all(),
            f"{history_col} matches {len_col}",
        )

    # ------------------------------------------------------------------
    # Window nesting
    #
    # history_1 is the most recent 1 interaction,
    # history_5 is the most recent 5, etc.
    #
    # Therefore, the tail of a larger window should equal
    # the smaller window.
    # ------------------------------------------------------------------

    nesting_pairs = [
        (1, 5),
        (5, 10),
        (10, 20),
        (20, 50),
        (50, 100),
    ]

    for small_k, large_k in nesting_pairs:
        small_col = f"history_{small_k}"
        large_col = f"history_{large_k}"

        ok = True

        for small, large in zip(
            df[small_col],
            df[large_col],
        ):
            small = np.asarray(small)
            large = np.asarray(large)

            if len(small) > len(large):
                ok = False
                break

            if len(small) > 0:
                if not np.array_equal(
                    small,
                    large[-len(small):],
                ):
                    ok = False
                    break

        check(
            ok,
            f"{small_col} is the suffix of {large_col}",
        )

    return df


def audit_split(
    name: str,
    path: Path,
    eval_events: pd.DataFrame,
    eval_history: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 80)
    print(f"3.{['train', 'validation', 'test'].index(name) + 1} {name.upper()} SPLIT")
    print("=" * 80)

    check_file(path, name)

    df = pd.read_parquet(path)

    print(f"  rows: {len(df):,}")
    print(f"  columns: {len(df.columns)}")

    check(
        len(df) == EXPECTED_SPLIT_ROWS[name],
        f"{name} rows = {EXPECTED_SPLIT_ROWS[name]:,}",
    )

    check_required_columns(
        df,
        SPLIT_REQUIRED,
        name,
    )

    check(
        df["eval_idx"].is_unique,
        f"{name}.eval_idx is unique",
    )

    eval_ids = set(
        eval_events["evaluation_id"].astype(np.int64)
    )

    split_ids = set(
        df["eval_idx"].astype(np.int64)
    )

    check(
        split_ids.issubset(eval_ids),
        f"all {name} eval_idx values map to evaluation events",
    )

    history_ids = set(
        eval_history["eval_idx"].astype(np.int64)
    )

    check(
        split_ids.issubset(history_ids),
        f"all {name} eval_idx values map to evaluation history",
    )

    # Validate metadata against evaluation_events by ID.
    events_by_id = eval_events.set_index("evaluation_id")

    mapped_users = events_by_id["user_id"].reindex(df["eval_idx"])
    mapped_videos = events_by_id["video_id"].reindex(df["eval_idx"])
    mapped_times = events_by_id["time_ms"].reindex(df["eval_idx"])

    check(
        (
            df["user_id"].to_numpy()
            == mapped_users.to_numpy()
        ).all(),
        f"{name} user_id mapping is consistent",
    )

    check(
        (
            df["video_id"].to_numpy()
            == mapped_videos.to_numpy()
        ).all(),
        f"{name} video_id mapping is consistent",
    )

    check(
        (
            df["time_ms"].to_numpy()
            == mapped_times.to_numpy()
        ).all(),
        f"{name} time_ms mapping is consistent",
    )

    # Validate that split rows correspond exactly to
    # the same history rows.
    history_by_id = eval_history.set_index("eval_idx")

    mapped_history_users = history_by_id["user_id"].reindex(
        df["eval_idx"]
    )

    mapped_history_videos = history_by_id["video_id"].reindex(
        df["eval_idx"]
    )

    mapped_history_times = history_by_id["time_ms"].reindex(
        df["eval_idx"]
    )

    check(
        (
            df["user_id"].to_numpy()
            == mapped_history_users.to_numpy()
        ).all(),
        f"{name} user_id mapping to history is consistent",
    )

    check(
        (
            df["video_id"].to_numpy()
            == mapped_history_videos.to_numpy()
        ).all(),
        f"{name} video_id mapping to history is consistent",
    )

    check(
        (
            df["time_ms"].to_numpy()
            == mapped_history_times.to_numpy()
        ).all(),
        f"{name} time_ms mapping to history is consistent",
    )

    return df


def audit_split_disjointness(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print("4. SPLIT DISJOINTNESS")
    print("=" * 80)

    train_ids = set(train["eval_idx"])
    val_ids = set(validation["eval_idx"])
    test_ids = set(test["eval_idx"])

    check(
        train_ids.isdisjoint(val_ids),
        "train and validation eval_idx are disjoint",
    )

    check(
        train_ids.isdisjoint(test_ids),
        "train and test eval_idx are disjoint",
    )

    check(
        val_ids.isdisjoint(test_ids),
        "validation and test eval_idx are disjoint",
    )

    combined = train_ids | val_ids | test_ids

    check(
        len(combined) == EXPECTED_N_EVAL,
        "train + validation + test cover all evaluation events",
    )

    check(
        len(train_ids)
        + len(val_ids)
        + len(test_ids)
        == EXPECTED_N_EVAL,
        "split row counts sum to total evaluation events",
    )


def audit_user_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print("5. USER-LEVEL SPLIT")
    print("=" * 80)

    train_users = set(train["user_id"])
    val_users = set(validation["user_id"])
    test_users = set(test["user_id"])

    check(
        train_users.isdisjoint(val_users),
        "train and validation users are disjoint",
    )

    check(
        train_users.isdisjoint(test_users),
        "train and test users are disjoint",
    )

    check(
        val_users.isdisjoint(test_users),
        "validation and test users are disjoint",
    )

    print(f"  train users: {len(train_users):,}")
    print(f"  validation users: {len(val_users):,}")
    print(f"  test users: {len(test_users):,}")


def audit_temporal_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print("6. TEMPORAL SPLIT")
    print("=" * 80)

    train_max = train["time_ms"].max()
    val_min = validation["time_ms"].min()
    val_max = validation["time_ms"].max()
    test_min = test["time_ms"].min()

    check(
        train_max < val_min,
        "train occurs before validation",
    )

    check(
        val_max < test_min,
        "validation occurs before test",
    )

    print(f"  train max time: {train_max}")
    print(f"  validation range: {val_min} -> {val_max}")
    print(f"  test min time: {test_min}")


def main() -> None:
    print("=" * 80)
    print("PHASE 8 — SCHEMA & SPLIT MAPPING AUDIT")
    print("=" * 80)

    eval_events = audit_evaluation_events()

    eval_history = audit_evaluation_history(
        eval_events
    )

    train = audit_split(
        "train",
        TRAIN_PATH,
        eval_events,
        eval_history,
    )

    validation = audit_split(
        "validation",
        VALIDATION_PATH,
        eval_events,
        eval_history,
    )

    test = audit_split(
        "test",
        TEST_PATH,
        eval_events,
        eval_history,
    )

    audit_split_disjointness(
        train,
        validation,
        test,
    )

    audit_user_split(
        train,
        validation,
        test,
    )

    audit_temporal_order(
        train,
        validation,
        test,
    )

    print()
    print("=" * 80)
    print("PHASE 8 SCHEMA AUDIT: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()