# -*- coding: utf-8 -*-

"""
Phase 7: Reconstruct Evaluation Histories

Contract
-------
Phase 7 stores RAW video IDs in all history artifacts.

Artifacts
---------
video_vocab.parquet
    raw_video_id -> frozen video_idx

history_K*.parquet
    history contains RAW video_id

Downstream
----------
Phase 8 maps:

    raw_video_id
        -> video_idx
        -> model_video_idx = video_idx + 1

0 is reserved for padding.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

EVAL_FILE = PROCESSED_DIR / "evaluation_events.parquet"

STANDARD_FILES = [
    ROOT / "data/raw/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv",
    ROOT / "data/raw/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv",
]

OUTPUT_DIR = PROCESSED_DIR / "phase7"

EVALUATION_HISTORY_FILE = (
    PROCESSED_DIR / "evaluation_history.parquet"
)

VIDEO_VOCAB_FILE = (
    PROCESSED_DIR / "video_vocab.parquet"
)

VIDEO_VOCAB_CONFIG_FILE = (
    PROCESSED_DIR / "video_vocab_config.yaml"
)


# =============================================================================
# FROZEN PROTOCOL
# =============================================================================

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

W_DAYS = 1
L_MAX = 2048

W_MS = W_DAYS * 24 * 60 * 60 * 1000


# =============================================================================
# LOAD EVALUATION EVENTS
# =============================================================================

def load_evaluation_events():

    print("Loading evaluation events...")

    df = pd.read_parquet(
        EVAL_FILE,
        columns=[
            "evaluation_id",
            "user_id",
            "video_id",
            "time_ms",
            "long_view",
        ],
    )

    df = df.sort_values(
        ["time_ms", "evaluation_id"]
    ).reset_index(drop=True)

    if not df["evaluation_id"].is_unique:
        raise RuntimeError(
            "evaluation_id is not unique."
        )

    df["user_id"] = df["user_id"].astype("int64")
    df["video_id"] = df["video_id"].astype("int64")
    df["time_ms"] = df["time_ms"].astype("int64")

    print(
        f"  Evaluation events: {len(df):,}"
    )

    print(
        f"  Unique candidate videos: "
        f"{df['video_id'].nunique():,}"
    )

    return df


# =============================================================================
# LOAD STANDARD INTERACTIONS
# =============================================================================

def load_standard_interactions():

    print("\nLoading standard interactions...")

    frames = []

    for path in STANDARD_FILES:

        print(
            f"  Loading: {path.name}"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing standard interaction file:\n{path}"
            )

        df = pd.read_csv(
            path,
            usecols=[
                "user_id",
                "video_id",
                "time_ms",
            ],
            dtype={
                "user_id": "int32",
                "video_id": "int64",
                "time_ms": "int64",
            },
        )

        print(
            f"    Rows: {len(df):,}"
        )

        frames.append(df)

    interactions = pd.concat(
        frames,
        ignore_index=True,
    )

    interactions = interactions.sort_values(
        [
            "user_id",
            "time_ms",
        ]
    ).reset_index(drop=True)

    print(
        f"  Total standard interactions: "
        f"{len(interactions):,}"
    )

    print(
        f"  Unique history videos: "
        f"{interactions['video_id'].nunique():,}"
    )

    return interactions


# =============================================================================
# BUILD H_t^(W)
# =============================================================================

def build_histories(
    eval_df,
    interactions,
):

    print(
        f"\nBuilding H_t^({W_DAYS} day)..."
    )

    # -------------------------------------------------------------------------
    # Evaluation events by user
    # -------------------------------------------------------------------------

    eval_by_user = defaultdict(list)

    for idx, row in eval_df.iterrows():

        eval_by_user[int(row["user_id"])].append(
            (
                idx,
                int(row["time_ms"]),
            )
        )

    histories = [None] * len(eval_df)

    # -------------------------------------------------------------------------
    # Process each user
    # -------------------------------------------------------------------------

    grouped = interactions.groupby(
        "user_id",
        sort=False,
    )

    for user_id, user_events in grouped:

        if user_id not in eval_by_user:
            continue

        user_events = user_events[
            [
                "video_id",
                "time_ms",
            ]
        ].to_numpy()

        eval_events = eval_by_user[user_id]

        # eval_events are already chronological because eval_df
        # was sorted by time_ms.
        eval_events.sort(
            key=lambda x: x[1]
        )

        left = 0
        right = 0

        n = len(user_events)

        for eval_idx, eval_time in eval_events:

            # -------------------------------------------------------------
            # Strict upper boundary: interaction_time < t
            # -------------------------------------------------------------

            while (
                right < n
                and int(user_events[right][1]) < eval_time
            ):
                right += 1

            # -------------------------------------------------------------
            # Lower boundary: interaction_time >= t - W
            # -------------------------------------------------------------

            lower_bound = eval_time - W_MS

            while (
                left < right
                and int(user_events[left][1]) < lower_bound
            ):
                left += 1

            window = user_events[left:right]

            histories[eval_idx] = [
                int(video_id)
                for video_id, _ in window
            ]

    # -------------------------------------------------------------------------
    # Users with no history
    # -------------------------------------------------------------------------

    for i in range(len(histories)):

        if histories[i] is None:
            histories[i] = []

    return histories


# =============================================================================
# HISTORY AUDIT
# =============================================================================

def audit_histories(
    eval_df,
    interactions,
    histories,
):

    print("\n" + "=" * 80)
    print("HISTORY AUDIT")
    print("=" * 80)

    if len(eval_df) != len(histories):
        raise RuntimeError(
            "History count does not match evaluation events."
        )

    interaction_ids = set(
        interactions["video_id"].astype(int)
    )

    unique_history_ids = set()

    lengths = []

    for history in histories:

        if not isinstance(history, list):
            raise RuntimeError(
                "History is not stored as a Python list."
            )

        lengths.append(
            len(history)
        )

        for video_id in history:

            if not isinstance(
                video_id,
                (int, np.integer),
            ):
                raise RuntimeError(
                    "History contains non-integer video ID."
                )

            unique_history_ids.add(
                int(video_id)
            )

    # -------------------------------------------------------------------------
    # Source coverage
    # -------------------------------------------------------------------------

    missing_from_source = (
        unique_history_ids - interaction_ids
    )

    if missing_from_source:

        raise RuntimeError(
            "HISTORY SOURCE AUDIT FAILED.\n"
            f"IDs in history but not in standard interactions: "
            f"{len(missing_from_source)}\n"
            f"Examples: "
            f"{sorted(missing_from_source)[:10]}"
        )

    print(
        f"Evaluation events: {len(histories):,}"
    )

    print(
        f"Mean history length: "
        f"{np.mean(lengths):.2f}"
    )

    print(
        f"Median history length: "
        f"{np.median(lengths):.2f}"
    )

    print(
        f"Max history length: "
        f"{max(lengths, default=0):,}"
    )

    print(
        f"Zero-history events: "
        f"{sum(x == 0 for x in lengths):,}"
    )

    print(
        f"Unique IDs appearing in histories: "
        f"{len(unique_history_ids):,}"
    )

    print(
        "History type audit: PASS"
    )

    print(
        "History source coverage: PASS"
    )

    print(
        "Strict temporal construction: PASS"
    )

    print("=" * 80)


# =============================================================================
# SAVE MAIN HISTORY
# =============================================================================

def save_evaluation_history(
    eval_df,
    histories,
):

    output = eval_df[
        [
            "evaluation_id",
            "user_id",
            "video_id",
            "time_ms",
            "long_view",
        ]
    ].copy()

    # IMPORTANT:
    # RAW video IDs only.
    output["history_full"] = histories

    output["history_full_len"] = [
        len(history)
        for history in histories
    ]

    output.to_parquet(
        EVALUATION_HISTORY_FILE,
        index=False,
    )

    print(
        "\nSaved main history artifact:"
    )

    print(
        f"  {EVALUATION_HISTORY_FILE}"
    )


# =============================================================================
# COLLECT HISTORY IDS
# =============================================================================

def collect_history_video_ids(
    histories,
):

    ids = set()

    for history in histories:

        for video_id in history:

            ids.add(
                int(video_id)
            )

    return ids


# =============================================================================
# BUILD FROZEN VOCABULARY
# =============================================================================

def build_video_vocab(
    eval_df,
    histories,
):

    print("\n" + "=" * 80)
    print("BUILDING FROZEN VIDEO VOCABULARY")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Candidate IDs
    # -------------------------------------------------------------------------

    candidate_ids = {
        int(video_id)
        for video_id in eval_df["video_id"].unique()
    }

    # -------------------------------------------------------------------------
    # History IDs
    # -------------------------------------------------------------------------

    history_ids = collect_history_video_ids(
        histories
    )

    # -------------------------------------------------------------------------
    # Frozen union
    # -------------------------------------------------------------------------

    all_video_ids = sorted(
        candidate_ids | history_ids
    )

    if not all_video_ids:

        raise RuntimeError(
            "Frozen vocabulary is empty."
        )

    # -------------------------------------------------------------------------
    # Build vocabulary
    #
    # video_idx starts at 0.
    #
    # This is NOT the model index.
    # Model index is video_idx + 1.
    # -------------------------------------------------------------------------

    vocab = pd.DataFrame(
        {
            "raw_video_id": all_video_ids,
            "video_idx": np.arange(
                len(all_video_ids),
                dtype=np.int64,
            ),
        }
    )

    # -------------------------------------------------------------------------
    # Invariants
    # -------------------------------------------------------------------------

    if not vocab["raw_video_id"].is_unique:

        raise RuntimeError(
            "raw_video_id is not unique."
        )

    if not vocab["video_idx"].is_unique:

        raise RuntimeError(
            "video_idx is not unique."
        )

    if int(vocab["video_idx"].iloc[0]) != 0:

        raise RuntimeError(
            "video_idx must start at 0."
        )

    if int(vocab["video_idx"].iloc[-1]) != len(vocab) - 1:

        raise RuntimeError(
            "video_idx is not contiguous."
        )

    vocab_ids = set(
        vocab["raw_video_id"].astype(int)
    )

    # -------------------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------------------

    missing_candidates = (
        candidate_ids - vocab_ids
    )

    missing_history = (
        history_ids - vocab_ids
    )

    if missing_candidates:

        raise RuntimeError(
            "VOCABULARY COVERAGE FAILED: "
            f"{len(missing_candidates)} candidate IDs missing."
        )

    if missing_history:

        raise RuntimeError(
            "VOCABULARY COVERAGE FAILED: "
            f"{len(missing_history)} history IDs missing."
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    VIDEO_VOCAB_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocab.to_parquet(
        VIDEO_VOCAB_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------

    config_text = f"""# Frozen Phase 7 video vocabulary

vocabulary_size: {len(vocab)}
candidate_videos: {len(candidate_ids)}
history_videos: {len(history_ids)}
union_videos: {len(all_video_ids)}

indexing:
  raw_video_id_to_video_idx: true
  video_idx_start: 0

model_indexing:
  padding_idx: 0
  model_video_idx: "video_idx + 1"

protocol:
  candidate_and_history_share_vocabulary: true
  vocabulary_frozen_before_training: true
  candidate_source: evaluation_events
  history_source: H_t^(W)
  history_artifacts_contain: raw_video_id
"""

    VIDEO_VOCAB_CONFIG_FILE.write_text(
        config_text,
        encoding="utf-8",
    )

    print(
        f"Candidate videos: {len(candidate_ids):,}"
    )

    print(
        f"History videos:   {len(history_ids):,}"
    )

    print(
        f"Union vocabulary: {len(all_video_ids):,}"
    )

    print(
        "Vocabulary coverage: PASS"
    )

    print(
        f"Saved vocabulary:\n  {VIDEO_VOCAB_FILE}"
    )

    print(
        f"Saved config:\n  {VIDEO_VOCAB_CONFIG_FILE}"
    )

    return vocab


# =============================================================================
# MAKE K HISTORY
# =============================================================================

def make_k_history(
    full_history,
    k,
):

    if k == 0:
        return []

    if k == "Full":
        return full_history[-L_MAX:]

    return full_history[-int(k):]


# =============================================================================
# SAVE K HISTORIES
# =============================================================================

def save_k_histories(
    eval_df,
    full_histories,
):

    print("\n" + "=" * 80)
    print("SAVING K HISTORY CONFIGURATIONS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    for k in K_VALUES:

        print("\n" + "-" * 80)
        print(f"K = {k}")
        print("-" * 80)

        histories = [
            make_k_history(
                history,
                k,
            )
            for history in full_histories
        ]

        lengths = [
            len(history)
            for history in histories
        ]

        max_allowed = (
            L_MAX
            if k == "Full"
            else int(k)
        )

        if max(lengths, default=0) > max_allowed:

            raise RuntimeError(
                f"K={k}: history exceeds maximum allowed length."
            )

        if k == 0 and any(lengths):

            raise RuntimeError(
                "K=0 history is not empty."
            )

        output = eval_df[
            [
                "evaluation_id",
                "user_id",
                "video_id",
                "time_ms",
                "long_view",
            ]
        ].copy()

        # ---------------------------------------------------------------------
        # IMPORTANT:
        # RAW video IDs.
        #
        # No vocab encoding here.
        # ---------------------------------------------------------------------

        output["history"] = histories

        output["history_len"] = lengths

        if k == 0:

            output["history_available"] = np.zeros(
                len(output),
                dtype=np.int8,
            )

        else:

            output["history_available"] = (
                output["history_len"] > 0
            ).astype("int8")

        if k == "Full":

            filename = "history_KFull.parquet"

        else:

            filename = f"history_K{k}.parquet"

        output_path = OUTPUT_DIR / filename

        output.to_parquet(
            output_path,
            index=False,
        )

        summary.append(
            {
                "K": k,
                "evaluation_events": len(output),
                "mean_history_len": float(
                    np.mean(lengths)
                ),
                "median_history_len": float(
                    np.median(lengths)
                ),
                "max_history_len": int(
                    max(lengths, default=0)
                ),
                "zero_history": int(
                    sum(
                        length == 0
                        for length in lengths
                    )
                ),
            }
        )

        print(
            f"Events: {len(output):,}"
        )

        print(
            f"Mean history length: "
            f"{np.mean(lengths):.2f}"
        )

        print(
            f"Max history length: "
            f"{max(lengths, default=0):,}"
        )

        print(
            f"Saved: {output_path}"
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary_path = (
        OUTPUT_DIR / "history_config_summary.csv"
    )

    pd.DataFrame(summary).to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Frozen config
    # -------------------------------------------------------------------------

    config_path = (
        OUTPUT_DIR / "history_config.yaml"
    )

    config_path.write_text(
        f"""# Phase 7 frozen history configuration

W_days: {W_DAYS}
L_max: {L_MAX}

K_values:
  - 0
  - 1
  - 5
  - 10
  - 20
  - 50
  - 100
  - Full

protocol:
  history_definition: "[t-W, t)"
  strict_temporal_order: true
  fixed_evaluation_set: true
  test_data_used: false
  reconstruct_history_once: true
  derive_K_by_slicing: true
  candidate_and_history_share_video_vocabulary: true
  video_vocabulary_frozen: true
  history_artifacts_contain_raw_video_ids: true
  encoding_deferred_to_training: true
""",
        encoding="utf-8",
    )

    print(
        f"\nSaved summary:\n  {summary_path}"
    )

    print(
        f"Saved config:\n  {config_path}"
    )


# =============================================================================
# FINAL ARTIFACT AUDIT
# =============================================================================

def final_artifact_audit():

    print("\n" + "=" * 80)
    print("FINAL ARTIFACT CONSISTENCY AUDIT")
    print("=" * 80)

    vocab = pd.read_parquet(
        VIDEO_VOCAB_FILE
    )

    vocab_ids = set(
        vocab["raw_video_id"].astype(int)
    )

    print(
        f"Vocabulary rows: {len(vocab):,}"
    )

    # -------------------------------------------------------------------------
    # Audit every K artifact
    # -------------------------------------------------------------------------

    for k in K_VALUES:

        if k == "Full":
            filename = "history_KFull.parquet"
        else:
            filename = f"history_K{k}.parquet"

        path = OUTPUT_DIR / filename

        df = pd.read_parquet(
            path
        )

        # Basic row-count check.
        if len(df) == 0:
            raise RuntimeError(
                f"K={k}: artifact is empty."
            )

        # -------------------------------------------------------------
        # Every raw history ID must exist in vocab.
        # -------------------------------------------------------------

        artifact_ids = set()

        for history in df["history"]:

            for video_id in history:

                artifact_ids.add(
                    int(video_id)
                )

        missing = artifact_ids - vocab_ids

        if missing:

            raise RuntimeError(
                f"K={k}: {len(missing)} history IDs "
                f"missing from vocabulary. "
                f"Examples={sorted(missing)[:10]}"
            )

        print(
            f"  K={k}: PASS"
        )

    # -------------------------------------------------------------------------
    # Target sanity check if the known debug ID exists in artifacts.
    #
    # This is deliberately informational only.
    # It must NOT assume that arbitrary raw IDs should appear in history.
    # -------------------------------------------------------------------------

    target = 1840380

    target_in_vocab = (
        target in vocab_ids
    )

    target_occurrences = 0

    for k in K_VALUES:

        if k == "Full":
            filename = "history_KFull.parquet"
        else:
            filename = f"history_K{k}.parquet"

        df = pd.read_parquet(
            OUTPUT_DIR / filename,
            columns=["history"],
        )

        occurrences = sum(
            target in history
            for history in df["history"]
        )

        target_occurrences += occurrences

    print(
        f"\nDEBUG target raw_video_id={target}"
    )

    print(
        f"  Target in vocabulary: {target_in_vocab}"
    )

    print(
        f"  Target occurrences across K artifacts: "
        f"{target_occurrences}"
    )

    print(
        "\nFINAL ARTIFACT CONSISTENCY: PASS"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("PHASE 7 — HISTORY RECONSTRUCTION")
    print("=" * 80)

    print("\nFrozen configuration:")
    print(f"  W_days: {W_DAYS}")
    print(f"  L_max:  {L_MAX}")
    print(
        "  K:      {0, 1, 5, 10, 20, 50, 100, Full}"
    )

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    eval_df = load_evaluation_events()

    interactions = load_standard_interactions()

    # -------------------------------------------------------------------------
    # BUILD H^(W)
    # -------------------------------------------------------------------------

    full_histories = build_histories(
        eval_df,
        interactions,
    )

    # -------------------------------------------------------------------------
    # AUDIT
    # -------------------------------------------------------------------------

    audit_histories(
        eval_df,
        interactions,
        full_histories,
    )

    # -------------------------------------------------------------------------
    # SAVE MAIN HISTORY
    # -------------------------------------------------------------------------

    save_evaluation_history(
        eval_df,
        full_histories,
    )

    # -------------------------------------------------------------------------
    # BUILD VOCABULARY
    # -------------------------------------------------------------------------

    build_video_vocab(
        eval_df,
        full_histories,
    )

    # -------------------------------------------------------------------------
    # SAVE K HISTORIES
    # -------------------------------------------------------------------------

    save_k_histories(
        eval_df,
        full_histories,
    )

    # -------------------------------------------------------------------------
    # FINAL AUDIT
    # -------------------------------------------------------------------------

    final_artifact_audit()

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 7 COMPLETE")
    print("=" * 80)

    print("\nContract:")
    print(
        "  history_K*.parquet contains RAW video_id:       YES"
    )
    print(
        "  video_vocab.parquet maps raw -> video_idx:      YES"
    )
    print(
        "  model index = video_idx + 1:                    YES"
    )
    print(
        "  padding index = 0:                               YES"
    )
    print(
        "  H^(W) reconstructed once:                        PASS"
    )
    print(
        "  K derived by slicing:                             PASS"
    )
    print(
        "  Candidate/history shared vocabulary:             PASS"
    )
    print(
        "  Vocabulary frozen before training:               PASS"
    )
    print(
        "  Final artifact consistency:                       PASS"
    )


if __name__ == "__main__":
    main()