# -*- coding: utf-8 -*-

"""
Phase 8 — Baseline Training & Evaluation

Contract:
    Phase 7 history_K*.parquet:
        history contains RAW video_id.

    Phase 7 video_vocab.parquet:
        raw_video_id -> video_idx

    Model:
        model_video_idx = video_idx + 1
        0 = padding

Main conditions:
    K = {0, 1, 5, 10, 20, 50, 100, Full}

Each K is trained independently using the same:
    - architecture
    - embedding dimensions
    - hidden dimensions
    - optimizer
    - learning rate
    - batch size
    - loss
    - early stopping
    - training procedure

Primary metric:
    Validation ROC-AUC for model selection.

Final test evaluation:
    Performed only after the best validation checkpoint is frozen.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

from baseline_models import KuaiRandRecommender


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
PHASE7_DIR = PROCESSED_DIR / "phase7"

MODEL_DIR = ROOT / "models"
RESULT_DIR = PROCESSED_DIR / "phase8"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN EXPERIMENT CONFIGURATION
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

EMBEDDING_DIM = 64
HIDDEN_DIMS = [128, 64]

BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 3

SEED = 42

NUM_WORKERS = 0

PADDING_IDX = 0
MODEL_ID_SHIFT = 1


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DEVICE
# =============================================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    if torch.backends.mps.is_available():

        return torch.device("mps")

    return torch.device("cpu")


# =============================================================================
# FILE HELPERS
# =============================================================================

def history_filename(k):

    if k == "Full":

        return "history_KFull.parquet"

    return f"history_K{k}.parquet"


def model_filename(mode, k):

    if k == "Full":

        k_name = "KFull"

    else:

        k_name = f"K{k}"

    return f"baseline_{mode}_{k_name}.pth"


def metrics_filename(mode, k):

    if k == "Full":

        k_name = "KFull"

    else:

        k_name = f"K{k}"

    return f"metrics_{mode}_{k_name}.csv"


# =============================================================================
# LOAD FROZEN VIDEO VOCABULARY
# =============================================================================

def load_video_vocab():

    print("=" * 80)
    print("LOADING FROZEN VIDEO VOCABULARY")
    print("=" * 80)

    if not VIDEO_VOCAB_FILE.exists():

        raise FileNotFoundError(
            f"Frozen vocabulary not found:\n"
            f"  {VIDEO_VOCAB_FILE}\n\n"
            "Run Phase 7 first."
        )

    vocab = pd.read_parquet(
        VIDEO_VOCAB_FILE
    )

    required_columns = {
        "raw_video_id",
        "video_idx",
    }

    missing_columns = (
        required_columns
        - set(vocab.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Vocabulary missing required columns: "
            f"{sorted(missing_columns)}"
        )

    vocab["raw_video_id"] = (
        vocab["raw_video_id"]
        .astype("int64")
    )

    vocab["video_idx"] = (
        vocab["video_idx"]
        .astype("int64")
    )

    # -------------------------------------------------------------------------
    # Vocabulary invariants
    # -------------------------------------------------------------------------

    if not vocab["raw_video_id"].is_unique:

        raise RuntimeError(
            "raw_video_id is not unique in frozen vocabulary."
        )

    if not vocab["video_idx"].is_unique:

        raise RuntimeError(
            "video_idx is not unique in frozen vocabulary."
        )

    if vocab["video_idx"].min() != 0:

        raise RuntimeError(
            "Frozen vocabulary must start at video_idx=0."
        )

    expected_indices = np.arange(
        len(vocab),
        dtype=np.int64,
    )

    actual_indices = (
        vocab["video_idx"]
        .sort_values()
        .to_numpy()
    )

    if not np.array_equal(
        actual_indices,
        expected_indices,
    ):

        raise RuntimeError(
            "video_idx is not contiguous 0..N-1."
        )

    vocab_map = dict(
        zip(
            vocab["raw_video_id"],
            vocab["video_idx"],
        )
    )

    vocab_size = len(vocab)

    print(
        f"Raw video ID column: raw_video_id"
    )

    print(
        f"Vocabulary index column: video_idx"
    )

    print(
        f"Vocabulary index range: "
        f"0 -> {vocab_size - 1}"
    )

    print(
        f"Unique vocabulary IDs: "
        f"{vocab_size:,}"
    )

    print(
        f"Max raw video ID: "
        f"{vocab['raw_video_id'].max():,}"
    )

    print(
        f"Embedding vocabulary size: "
        f"{vocab_size + MODEL_ID_SHIFT:,}"
    )

    print(
        "Index 0 reserved for padding."
    )

    return vocab, vocab_map


# =============================================================================
# LOAD EVALUATION LABELS
# =============================================================================

def load_evaluation_labels():

    events_path = (
        PROCESSED_DIR
        / "evaluation_events.parquet"
    )

    if not events_path.exists():

        raise FileNotFoundError(
            f"Evaluation events not found:\n"
            f"  {events_path}"
        )

    events = pd.read_parquet(
        events_path,
        columns=[
            "evaluation_id",
            "user_id",
            "video_id",
            "time_ms",
            "long_view",
        ],
    )

    events["evaluation_id"] = (
        events["evaluation_id"]
        .astype("int64")
    )

    events["user_id"] = (
        events["user_id"]
        .astype("int64")
    )

    events["video_id"] = (
        events["video_id"]
        .astype("int64")
    )

    events["long_view"] = (
        events["long_view"]
        .astype("int64")
    )

    if not events["evaluation_id"].is_unique:

        raise RuntimeError(
            "evaluation_id is not unique."
        )

    if not set(
        events["long_view"].unique()
    ).issubset({0, 1}):

        raise RuntimeError(
            "long_view must contain only 0/1."
        )

    return events


# =============================================================================
# MAP RAW VIDEO ID -> MODEL ID
# =============================================================================

def map_raw_ids_to_model_ids(
    raw_ids,
    vocab_map,
    context,
):

    mapped = []

    missing = set()

    for raw_id in raw_ids:

        raw_id = int(raw_id)

        video_idx = vocab_map.get(
            raw_id
        )

        if video_idx is None:

            missing.add(raw_id)

        else:

            # 0 is reserved for padding.
            model_idx = (
                int(video_idx)
                + MODEL_ID_SHIFT
            )

            mapped.append(
                model_idx
            )

    if missing:

        examples = sorted(
            missing
        )[:10]

        raise RuntimeError(
            f"{context}: "
            f"{len(missing):,} raw video IDs are "
            "absent from frozen vocabulary.\n"
            f"Examples: {examples}"
        )

    return mapped


# =============================================================================
# DATASET
# =============================================================================

class KuaiRandDataset(Dataset):

    def __init__(
        self,
        split_name,
        k,
        vocab_map,
    ):

        super().__init__()

        print(
            f"Loading {split_name} split "
            f"(K={k})..."
        )

        split_path = (
            PROCESSED_DIR
            / f"{split_name}.parquet"
        )

        history_path = (
            PHASE7_DIR
            / history_filename(k)
        )

        if not split_path.exists():

            raise FileNotFoundError(
                f"Split not found:\n"
                f"  {split_path}"
            )

        if not history_path.exists():

            raise FileNotFoundError(
                f"Phase 7 history not found:\n"
                f"  {history_path}"
            )

        # ---------------------------------------------------------------------
        # Load split
        #
        # split.parquet contains eval_idx.
        # Phase 7 history files contain evaluation_id.
        # ---------------------------------------------------------------------

        split_df = pd.read_parquet(
            split_path,
            columns=[
                "eval_idx",
            ],
        )

        split_df["eval_idx"] = (
            split_df["eval_idx"]
            .astype("int64")
        )

        if not split_df["eval_idx"].is_unique:

            raise RuntimeError(
                f"{split_name}: eval_idx is not unique."
            )

        # ---------------------------------------------------------------------
        # Load frozen Phase 7 K-history
        # ---------------------------------------------------------------------

        history_df = pd.read_parquet(
            history_path,
            columns=[
                "evaluation_id",
                "user_id",
                "video_id",
                "time_ms",
                "long_view",
                "history",
                "history_len",
                "history_available",
            ],
        )

        history_df["evaluation_id"] = (
            history_df["evaluation_id"]
            .astype("int64")
        )

        history_df["user_id"] = (
            history_df["user_id"]
            .astype("int64")
        )

        history_df["video_id"] = (
            history_df["video_id"]
            .astype("int64")
        )

        history_df["long_view"] = (
            history_df["long_view"]
            .astype("int64")
        )

        # ---------------------------------------------------------------------
        # Join split -> Phase 7 history
        # ---------------------------------------------------------------------

        merged = split_df.merge(
            history_df,
            left_on="eval_idx",
            right_on="evaluation_id",
            how="left",
            validate="one_to_one",
        )

        if len(merged) != len(split_df):

            raise RuntimeError(
                f"{split_name} K={k}: "
                "join changed number of rows."
            )

        if merged["evaluation_id"].isna().any():

            missing_count = int(
                merged["evaluation_id"]
                .isna()
                .sum()
            )

            raise RuntimeError(
                f"{split_name} K={k}: "
                f"{missing_count} split events have no "
                "matching Phase 7 history artifact."
            )

        # ---------------------------------------------------------------------
        # Candidate consistency audit
        # ---------------------------------------------------------------------

        event_df = load_evaluation_labels()

        event_lookup = event_df[
            [
                "evaluation_id",
                "user_id",
                "video_id",
                "time_ms",
                "long_view",
            ]
        ]

        merged = merged.merge(
            event_lookup,
            on="evaluation_id",
            how="left",
            suffixes=(
                "_phase7",
                "_event",
            ),
            validate="one_to_one",
        )

        if len(merged) != len(split_df):

            raise RuntimeError(
                f"{split_name} K={k}: "
                "evaluation event join changed row count."
            )

        # Phase 7 artifact and evaluation_events must agree.
        checks = [
            "user_id",
            "video_id",
            "time_ms",
            "long_view",
        ]

        for col in checks:

            left_col = f"{col}_phase7"
            right_col = f"{col}_event"

            mismatch = (
                merged[left_col]
                != merged[right_col]
            )

            if mismatch.any():

                count = int(
                    mismatch.sum()
                )

                examples = merged.loc[
                    mismatch,
                    [
                        "evaluation_id",
                        left_col,
                        right_col,
                    ],
                ].head(5)

                raise RuntimeError(
                    f"{split_name} K={k}: "
                    f"{col} mismatch in {count} rows.\n"
                    f"{examples}"
                )

        # ---------------------------------------------------------------------
        # Candidate encoding
        # ---------------------------------------------------------------------

        raw_candidates = (
            merged["video_id_event"]
            .astype("int64")
            .tolist()
        )

        candidate_ids = (
            map_raw_ids_to_model_ids(
                raw_candidates,
                vocab_map,
                f"{split_name} K={k} candidate",
            )
        )

        # ---------------------------------------------------------------------
        # History encoding
        #
        # Phase 7 history contains RAW video IDs.
        # Convert them through the SAME frozen vocabulary.
        # ---------------------------------------------------------------------

        raw_histories = (
            merged["history"]
            .tolist()
        )

        history_ids = []

        history_lengths = []

        for row_idx, history in enumerate(
            raw_histories
        ):

            if history is None:

                history = []

            history = list(history)

            mapped_history = (
                map_raw_ids_to_model_ids(
                    history,
                    vocab_map,
                    (
                        f"{split_name} K={k} "
                        f"history row {row_idx}"
                    ),
                )
            )

            history_ids.append(
                mapped_history
            )

            history_lengths.append(
                len(mapped_history)
            )

        # ---------------------------------------------------------------------
        # K invariants
        # ---------------------------------------------------------------------

        if k == 0:

            if any(
                len(history) != 0
                for history in history_ids
            ):

                raise RuntimeError(
                    f"{split_name} K=0: "
                    "history is not empty."
                )

        elif k != "Full":

            if any(
                len(history) > k
                for history in history_ids
            ):

                raise RuntimeError(
                    f"{split_name} K={k}: "
                    "history exceeds K."
                )

        else:

            if any(
                len(history) > 2048
                for history in history_ids
            ):

                raise RuntimeError(
                    f"{split_name} K=Full: "
                    "history exceeds L_max=2048."
                )

        # ---------------------------------------------------------------------
        # Store
        # ---------------------------------------------------------------------

        self.candidate_ids = candidate_ids

        self.history_ids = history_ids

        self.history_lengths = history_lengths

        self.labels = (
            merged["long_view_event"]
            .astype("float32")
            .tolist()
        )

        self.user_ids = (
            merged["user_id_event"]
            .astype("int64")
            .tolist()
        )

        self.evaluation_ids = (
            merged["evaluation_id"]
            .astype("int64")
            .tolist()
        )

        self.k = k

        print(
            f"  Samples: {len(self):,}"
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return {
            "candidate_id": self.candidate_ids[idx],
            "history_ids": self.history_ids[idx],
            "label": self.labels[idx],
            "user_id": self.user_ids[idx],
            "evaluation_id": self.evaluation_ids[idx],
        }


# =============================================================================
# COLLATE
# =============================================================================

def collate_batch(batch):

    batch_size = len(batch)

    candidate_ids = torch.tensor(
        [
            item["candidate_id"]
            for item in batch
        ],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [
            item["label"]
            for item in batch
        ],
        dtype=torch.float32,
    )

    max_len = max(
        (
            len(item["history_ids"])
            for item in batch
        ),
        default=0,
    )

    # K=0 still needs a sequence dimension.
    max_len = max(
        max_len,
        1,
    )

    history_ids = torch.zeros(
        (
            batch_size,
            max_len,
        ),
        dtype=torch.long,
    )

    history_mask = torch.zeros(
        (
            batch_size,
            max_len,
        ),
        dtype=torch.float32,
    )

    for row_idx, item in enumerate(batch):

        history = item["history_ids"]

        length = len(history)

        if length == 0:

            continue

        history_ids[
            row_idx,
            :length,
        ] = torch.tensor(
            history,
            dtype=torch.long,
        )

        history_mask[
            row_idx,
            :length,
        ] = 1.0

    user_ids = torch.tensor(
        [
            item["user_id"]
            for item in batch
        ],
        dtype=torch.long,
    )

    evaluation_ids = torch.tensor(
        [
            item["evaluation_id"]
            for item in batch
        ],
        dtype=torch.long,
    )

    return (
        candidate_ids,
        history_ids,
        history_mask,
        labels,
        user_ids,
        evaluation_ids,
    )


# =============================================================================
# PRE-FLIGHT AUDIT
# =============================================================================

def preflight_audit(
    vocab_map,
):

    print("\n" + "=" * 80)
    print("PHASE 8 PRE-FLIGHT AUDIT")
    print("=" * 80)

    for k in K_VALUES:

        print(f"\nChecking K{k}...")

        train_dataset = KuaiRandDataset(
            "train",
            k,
            vocab_map,
        )

        val_dataset = KuaiRandDataset(
            "validation",
            k,
            vocab_map,
        )

        test_dataset = KuaiRandDataset(
            "test",
            k,
            vocab_map,
        )

        if len(train_dataset) == 0:

            raise RuntimeError(
                f"K={k}: empty train dataset."
            )

        if len(val_dataset) == 0:

            raise RuntimeError(
                f"K={k}: empty validation dataset."
            )

        if len(test_dataset) == 0:

            raise RuntimeError(
                f"K={k}: empty test dataset."
            )

        print(
            f"  K={k}: PASS"
        )

    print("\n" + "=" * 80)
    print("PRE-FLIGHT AUDIT: PASS")
    print("=" * 80)


# =============================================================================
# BUILD MODEL
# =============================================================================

def build_model(
    vocab_size,
    mode="full",
):

    model = KuaiRandRecommender(
        num_videos=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        mode=mode,
    )

    return model


# =============================================================================
# EVALUATE
# =============================================================================

def evaluate_model(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []

    labels = []

    users = []

    evaluation_ids = []

    with torch.no_grad():

        for (
            candidate_ids,
            history_ids,
            history_mask,
            batch_labels,
            batch_users,
            batch_evaluation_ids,
        ) in loader:

            candidate_ids = (
                candidate_ids.to(device)
            )

            history_ids = (
                history_ids.to(device)
            )

            history_mask = (
                history_mask.to(device)
            )

            preds = model(
                candidate_ids,
                history_ids,
                history_mask,
            )

            predictions.extend(
                preds.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            labels.extend(
                batch_labels.numpy()
                .tolist()
            )

            users.extend(
                batch_users.numpy()
                .tolist()
            )

            evaluation_ids.extend(
                batch_evaluation_ids.numpy()
                .tolist()
            )

    labels_np = np.asarray(
        labels,
        dtype=np.int64,
    )

    predictions_np = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if len(np.unique(labels_np)) < 2:

        auc = float("nan")

    else:

        auc = roc_auc_score(
            labels_np,
            predictions_np,
        )

    return {
        "auc": auc,
        "predictions": predictions_np,
        "labels": labels_np,
        "user_ids": np.asarray(
            users,
            dtype=np.int64,
        ),
        "evaluation_ids": np.asarray(
            evaluation_ids,
            dtype=np.int64,
        ),
    }


# =============================================================================
# TRAIN ONE K
# =============================================================================

def train_one_k(
    k,
    vocab_size,
    vocab_map,
    device,
    mode="full",
):

    print("\n" + "=" * 80)
    print(
        f"TRAINING MODEL — "
        f"Mode={mode} | K={k}"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    train_dataset = KuaiRandDataset(
        "train",
        k,
        vocab_map,
    )

    val_dataset = KuaiRandDataset(
        "validation",
        k,
        vocab_map,
    )

    test_dataset = KuaiRandDataset(
        "test",
        k,
        vocab_map,
    )

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=False,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = build_model(
        vocab_size,
        mode=mode,
    ).to(device)

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    criterion = nn.BCELoss()

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR
        / model_filename(mode, k)
    )

    metrics_path = (
        RESULT_DIR
        / metrics_filename(mode, k)
    )

    # -------------------------------------------------------------------------
    # Training state
    # -------------------------------------------------------------------------

    best_val_auc = -np.inf

    best_epoch = 0

    patience_counter = 0

    metrics = []

    # -------------------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        total_samples = 0

        for (
            candidate_ids,
            history_ids,
            history_mask,
            labels,
            _,
            _,
        ) in train_loader:

            candidate_ids = (
                candidate_ids.to(device)
            )

            history_ids = (
                history_ids.to(device)
            )

            history_mask = (
                history_mask.to(device)
            )

            labels = (
                labels.to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                candidate_ids,
                history_ids,
                history_mask,
            )

            loss = criterion(
                predictions,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                candidate_ids.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_loss
            / total_samples
        )

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        val_result = evaluate_model(
            model,
            val_loader,
            device,
        )

        val_auc = val_result["auc"]

        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_auc,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val ROC-AUC: {val_auc:.5f}"
        )

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        if np.isfinite(val_auc) and (
            val_auc > best_val_auc
        ):

            best_val_auc = val_auc

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "k": k,

                    "mode": mode,

                    "vocab_size":
                        vocab_size,

                    "embedding_dim":
                        EMBEDDING_DIM,

                    "hidden_dims":
                        HIDDEN_DIMS,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "epoch":
                        epoch,

                    "best_val_auc":
                        best_val_auc,

                    "seed":
                        SEED,

                    "model_id_shift":
                        MODEL_ID_SHIFT,

                    "padding_idx":
                        PADDING_IDX,
                },
                checkpoint_path,
            )

            print(
                f"  -> Best model saved: "
                f"{checkpoint_path.name}"
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print(
                    "\n[EARLY STOPPING] "
                    f"Best Val AUC={best_val_auc:.5f} "
                    f"at epoch {best_epoch}"
                )

                break

    # -------------------------------------------------------------------------
    # Save training metrics
    # -------------------------------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Load frozen best checkpoint
    # -------------------------------------------------------------------------

    if not checkpoint_path.exists():

        raise RuntimeError(
            f"No best checkpoint produced for K={k}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -------------------------------------------------------------------------
    # Final validation with frozen checkpoint
    # -------------------------------------------------------------------------

    frozen_val_result = evaluate_model(
        model,
        val_loader,
        device,
    )

    frozen_val_auc = (
        frozen_val_result["auc"]
    )

    # -------------------------------------------------------------------------
    # Final test with frozen checkpoint
    #
    # IMPORTANT:
    # No parameter update occurs here.
    # -------------------------------------------------------------------------

    frozen_test_result = evaluate_model(
        model,
        test_loader,
        device,
    )

    test_auc = (
        frozen_test_result["auc"]
    )

    print("\n" + "-" * 80)
    print(
        f"FROZEN CHECKPOINT — K={k}"
    )
    print("-" * 80)

    print(
        f"Best epoch:       {best_epoch}"
    )

    print(
        f"Best Val ROC-AUC: {best_val_auc:.5f}"
    )

    print(
        f"Frozen Val AUC:   {frozen_val_auc:.5f}"
    )

    print(
        f"Frozen Test AUC:  {test_auc:.5f}"
    )

    print(
        f"Checkpoint:       {checkpoint_path}"
    )

    # -------------------------------------------------------------------------
    # Save final summary
    # -------------------------------------------------------------------------

    summary = {
        "mode": mode,
        "K": k,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dims": str(HIDDEN_DIMS),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs_max": EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
    }

    return summary


# =============================================================================
# SAVE RESULT TABLE
# =============================================================================

def save_result_table(
    results,
    filename="baseline_results.csv",
):

    path = RESULT_DIR / filename

    df = pd.DataFrame(results)

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved results:"
        f"\n  {path}"
    )

    return df


# =============================================================================
# FINAL ARTIFACT AUDIT
# =============================================================================

def final_artifact_audit(
    vocab_size,
    results,
):

    print("\n" + "=" * 80)
    print("PHASE 8 FINAL ARTIFACT AUDIT")
    print("=" * 80)

    expected_count = len(K_VALUES)

    if len(results) != expected_count:

        raise RuntimeError(
            "Unexpected number of trained conditions: "
            f"{len(results)} != {expected_count}"
        )

    for result in results:

        k = result["K"]

        checkpoint_path = (
            MODEL_DIR
            / model_filename(
                "full",
                k,
            )
        )

        metrics_path = (
            RESULT_DIR
            / metrics_filename(
                "full",
                k,
            )
        )

        if not checkpoint_path.exists():

            raise RuntimeError(
                f"K={k}: checkpoint missing:\n"
                f"  {checkpoint_path}"
            )

        if not metrics_path.exists():

            raise RuntimeError(
                f"K={k}: metrics missing:\n"
                f"  {metrics_path}"
            )

        if result["vocab_size"] != vocab_size:

            raise RuntimeError(
                f"K={k}: vocabulary size mismatch."
            )

        print(
            f"  K={k}: PASS"
        )

    result_df = pd.DataFrame(
        results
    )

    print("\nResults:")

    print(
        result_df[
            [
                "K",
                "best_epoch",
                "best_val_auc",
                "test_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPHASE 8 FINAL ARTIFACT AUDIT: PASS"
    )


# =============================================================================
# PATH CONSTANT
# =============================================================================

VIDEO_VOCAB_FILE = (
    PROCESSED_DIR
    / "video_vocab.parquet"
)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("PHASE 8 — BASELINE TRAINING & EVALUATION")
    print("=" * 80)

    set_seed()

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    # -------------------------------------------------------------------------
    # Load frozen vocabulary
    # -------------------------------------------------------------------------

    vocab, vocab_map = (
        load_video_vocab()
    )

    vocab_size = len(vocab)

    # -------------------------------------------------------------------------
    # Pre-flight
    # -------------------------------------------------------------------------

    preflight_audit(
        vocab_map
    )

    # -------------------------------------------------------------------------
    # Train one independent model per K
    # -------------------------------------------------------------------------

    results = []

    for k in K_VALUES:

        # Reset seed before each independent condition.
        set_seed()

        result = train_one_k(
            k=k,
            vocab_size=vocab_size,
            vocab_map=vocab_map,
            device=device,
            mode="full",
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    save_result_table(
        results,
        "baseline_results.csv",
    )

    # -------------------------------------------------------------------------
    # Final audit
    # -------------------------------------------------------------------------

    final_artifact_audit(
        vocab_size,
        results,
    )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)

    print(
        "\nConditions trained:"
    )

    for k in K_VALUES:

        print(
            f"  K={k}: trained + frozen"
        )

    print(
        "\nProtocol:"
    )

    print(
        "  Frozen video vocabulary:       PASS"
    )

    print(
        "  Raw ID -> vocab -> model ID:   PASS"
    )

    print(
        "  Padding index = 0:             PASS"
    )

    print(
        "  Same architecture across K:    PASS"
    )

    print(
        "  Independent training per K:    PASS"
    )

    print(
        "  Validation model selection:    PASS"
    )

    print(
        "  Test after checkpoint freeze:  PASS"
    )

    print(
        "\nPrimary metric:"
    )

    print(
        "  ROC-AUC"
    )

    print(
        "\nResult file:"
    )

    print(
        f"  {RESULT_DIR / 'baseline_results.csv'}"
    )


if __name__ == "__main__":

    main()
