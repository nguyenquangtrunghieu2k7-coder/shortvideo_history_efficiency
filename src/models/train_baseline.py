# -*- coding: utf-8 -*-
"""
Phase 8 — Baseline Training & Evaluation

Baselines:
    1. video_only  | K=0
    2. history_only | K=Full

Data contracts:
    - Fixed train / validation / test splits
    - Frozen video vocabulary from Phase 7
    - Shared vocabulary for candidate + history videos
    - Padding index = 0
    - Best checkpoint selected by validation ROC-AUC
    - Test evaluated once using best checkpoint
"""

from pathlib import Path

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
MODEL_DIR = ROOT / "models"
METRIC_DIR = MODEL_DIR / "phase8_metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 3

EMBEDDING_DIM = 64
HIDDEN_DIMS = [128, 64]

L_MAX = 2048

K_VIDEO_ONLY = 0
K_HISTORY_ONLY = "full"

VOCAB_FILE = PROCESSED_DIR / "video_vocab.parquet"

SPLITS = ["train", "validation", "test"]


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
# LOAD FROZEN VIDEO VOCABULARY
# =============================================================================

def load_video_vocab():
    print("=" * 80)
    print("LOADING FROZEN VIDEO VOCABULARY")
    print("=" * 80)

    if not VOCAB_FILE.exists():
        raise FileNotFoundError(
            f"Missing vocabulary artifact:\n{VOCAB_FILE}"
        )

    vocab = pd.read_parquet(VOCAB_FILE)

    print(f"Vocabulary rows: {len(vocab):,}")
    print(f"Columns: {list(vocab.columns)}")

    if len(vocab) == 0:
        raise ValueError("Video vocabulary is empty.")

    return vocab


def detect_vocab_columns(vocab):
    """
    Detect raw video-id and contiguous vocabulary-index columns.

    The Phase 7 vocabulary is expected to contain a raw video ID
    and its corresponding frozen vocabulary index.
    """

    raw_candidates = [
        "video_id",
        "raw_video_id",
    ]

    index_candidates = [
        "vocab_id",
        "video_idx",
        "video_index",
        "index",
        "id",
    ]

    raw_col = next(
        (c for c in raw_candidates if c in vocab.columns),
        None,
    )

    idx_col = next(
        (c for c in index_candidates if c in vocab.columns),
        None,
    )

    if raw_col is None:
        raise ValueError(
            "Cannot find raw video-id column in video_vocab.parquet. "
            f"Available columns: {list(vocab.columns)}"
        )

    if idx_col is None:
        raise ValueError(
            "Cannot find vocabulary-index column in video_vocab.parquet. "
            f"Available columns: {list(vocab.columns)}"
        )

    return raw_col, idx_col


def build_vocab_mapping(vocab):
    raw_col, idx_col = detect_vocab_columns(vocab)

    print(f"Raw video ID column: {raw_col}")
    print(f"Vocabulary index column: {idx_col}")

    if vocab[raw_col].duplicated().any():
        raise ValueError(
            "Frozen vocabulary contains duplicate raw video IDs."
        )

    if vocab[idx_col].duplicated().any():
        raise ValueError(
            "Frozen vocabulary contains duplicate vocabulary indices."
        )

    vocab = vocab[[raw_col, idx_col]].copy()

    vocab[raw_col] = vocab[raw_col].astype(np.int64)
    vocab[idx_col] = vocab[idx_col].astype(np.int64)

    mapping = dict(
        zip(
            vocab[raw_col].tolist(),
            vocab[idx_col].tolist(),
        )
    )

    max_idx = vocab[idx_col].max()
    min_idx = vocab[idx_col].min()

    print(f"Vocabulary index range: {min_idx} -> {max_idx}")
    print(f"Unique vocabulary IDs: {len(mapping):,}")

    return mapping, max_idx


# =============================================================================
# ID MAPPING
# =============================================================================

def map_video_ids(values, mapping, name):
    """
    Convert raw video IDs to frozen vocabulary IDs.

    Padding is NOT handled here.
    Padding index 0 is added later by the Dataset.
    """

    mapped = []

    missing = set()

    for value in values:
        value = int(value)

        if value not in mapping:
            missing.add(value)
        else:
            mapped.append(mapping[value])

    if missing:
        examples = sorted(missing)[:10]

        raise ValueError(
            f"{name}: {len(missing):,} raw video IDs are absent "
            f"from frozen video vocabulary. Examples: {examples}"
        )

    return mapped


# =============================================================================
# DATASET
# =============================================================================

class KuaiRandDataset(Dataset):

    def __init__(
        self,
        split_name,
        k_budget,
        vocab_mapping,
        num_vocab,
    ):
        super().__init__()

        self.split_name = split_name
        self.k_budget = k_budget

        print(
            f"Loading {split_name} split "
            f"(K={k_budget})..."
        )

        split_path = PROCESSED_DIR / f"{split_name}.parquet"
        eval_path = PROCESSED_DIR / "evaluation_events.parquet"

        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing split file:\n{split_path}"
            )

        # =====================================================================
        # 1. LOAD SPLIT
        # =====================================================================

        split_df = pd.read_parquet(split_path)

        required_split = [
            "user_id",
            "video_id",
            "time_ms",
        ]

        for col in required_split:
            if col not in split_df.columns:
                raise ValueError(
                    f"{split_name}: missing required column '{col}'"
                )

        # =====================================================================
        # 2. LOAD LABELS FROM FIXED EVALUATION SET
        # =====================================================================

        events_df = pd.read_parquet(
            eval_path,
            columns=[
                "evaluation_id",
                "user_id",
                "video_id",
                "time_ms",
                "long_view",
            ],
        )

        # =====================================================================
        # 3. ONE-TO-ONE LABEL JOIN
        # =====================================================================

        merged = pd.merge(
            split_df,
            events_df[
                [
                    "evaluation_id",
                    "user_id",
                    "video_id",
                    "time_ms",
                    "long_view",
                ]
            ],
            on=[
                "user_id",
                "video_id",
                "time_ms",
            ],
            how="left",
            validate="one_to_one",
        )

        # ---------------------------------------------------------------------
        # Data Contract
        # ---------------------------------------------------------------------

        assert len(merged) == len(split_df), (
            f"{split_name}: merge changed row count. "
            f"Before={len(split_df)}, After={len(merged)}"
        )

        assert not merged["long_view"].isna().any(), (
            f"{split_name}: some samples could not be mapped "
            f"to evaluation labels."
        )

        # =====================================================================
        # 4. CANDIDATE VIDEO
        # =====================================================================

        candidate_raw = (
            merged["video_id"]
            .astype(np.int64)
            .tolist()
        )

        candidate_vocab = map_video_ids(
            candidate_raw,
            vocab_mapping,
            f"{split_name} candidate",
        )

        # +1 because embedding index 0 = padding.
        self.candidate_ids = torch.tensor(
            [x + 1 for x in candidate_vocab],
            dtype=torch.long,
        )

        # =====================================================================
        # 5. LABEL
        # =====================================================================

        self.labels = torch.tensor(
            merged["long_view"]
            .astype(np.float32)
            .values,
            dtype=torch.float32,
        )

        # =====================================================================
        # 6. HISTORY
        # =====================================================================

        if k_budget == 0:

            self.history_ids = torch.zeros(
                (len(merged), 1),
                dtype=torch.long,
            )

            self.history_mask = torch.zeros(
                (len(merged), 1),
                dtype=torch.float32,
            )

        else:

            hist_col = (
                "history_full"
                if k_budget == "full"
                else f"history_{k_budget}"
            )

            if hist_col not in merged.columns:
                raise ValueError(
                    f"{split_name}: missing history column "
                    f"'{hist_col}'"
                )

            histories = merged[hist_col].tolist()

            processed_histories = []

            for hist in histories:

                if hist is None:
                    seq = []

                elif isinstance(hist, np.ndarray):
                    seq = hist.tolist()

                else:
                    seq = list(hist)

                # Defensive L_max truncation.
                if len(seq) > L_MAX:
                    seq = seq[-L_MAX:]

                processed_histories.append(seq)

            # =================================================================
            # 7. MAP HISTORY IDs THROUGH FROZEN VOCABULARY
            # =================================================================

            mapped_histories = []
            missing = set()

            for hist in processed_histories:

                mapped = []

                for raw_id in hist:

                    raw_id = int(raw_id)

                    if raw_id not in vocab_mapping:
                        missing.add(raw_id)
                    else:
                        mapped.append(
                            vocab_mapping[raw_id] + 1
                        )

                mapped_histories.append(mapped)

            if missing:
                examples = sorted(missing)[:10]

                raise ValueError(
                    f"{split_name} history: "
                    f"{len(missing):,} raw video IDs are absent "
                    f"from frozen vocabulary. "
                    f"Examples: {examples}"
                )

            # =================================================================
            # 8. PAD HISTORY
            # =================================================================

            max_len = max(
                (len(x) for x in mapped_histories),
                default=0,
            )

            max_len = max(max_len, 1)

            self.history_ids = torch.zeros(
                (len(merged), max_len),
                dtype=torch.long,
            )

            self.history_mask = torch.zeros(
                (len(merged), max_len),
                dtype=torch.float32,
            )

            for i, seq in enumerate(mapped_histories):

                if not seq:
                    continue

                seq_len = len(seq)

                self.history_ids[
                    i,
                    :seq_len
                ] = torch.tensor(
                    seq,
                    dtype=torch.long,
                )

                self.history_mask[
                    i,
                    :seq_len
                ] = 1.0

        # =====================================================================
        # 9. FINAL CONTRACT
        # =====================================================================

        assert len(self.candidate_ids) == len(merged)
        assert len(self.labels) == len(merged)
        assert self.history_ids.shape[0] == len(merged)
        assert self.history_mask.shape[0] == len(merged)

        assert self.candidate_ids.min().item() >= 1
        assert self.candidate_ids.max().item() <= num_vocab

        if self.history_ids.numel() > 0:
            assert self.history_ids.min().item() >= 0
            assert self.history_ids.max().item() <= num_vocab

        print(
            f"  Samples: {len(self.labels):,}"
        )
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return (
            self.candidate_ids[idx],
            self.history_ids[idx],
            self.history_mask[idx],
            self.labels[idx],
        )


# =============================================================================
# DATA LOADERS
# =============================================================================

def build_dataloaders(
    k_budget,
    vocab_mapping,
    num_vocab,
):

    train_dataset = KuaiRandDataset(
        "train",
        k_budget,
        vocab_mapping,
        num_vocab,
    )

    val_dataset = KuaiRandDataset(
        "validation",
        k_budget,
        vocab_mapping,
        num_vocab,
    )

    test_dataset = KuaiRandDataset(
        "test",
        k_budget,
        vocab_mapping,
        num_vocab,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def predict(model, loader, device):

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for (
            candidate_ids,
            history_ids,
            history_mask,
            batch_labels,
        ) in loader:

            candidate_ids = candidate_ids.to(device)
            history_ids = history_ids.to(device)
            history_mask = history_mask.to(device)

            preds = model(
                candidate_ids,
                history_ids,
                history_mask,
            )

            predictions.extend(
                preds.detach().cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return np.asarray(predictions), np.asarray(labels)


def evaluate_auc(model, loader, device):

    predictions, labels = predict(
        model,
        loader,
        device,
    )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "ROC-AUC cannot be calculated: "
            "labels contain only one class."
        )

    return roc_auc_score(
        labels,
        predictions,
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_and_evaluate(
    mode,
    k_budget,
    vocab_mapping,
    num_vocab,
    device,
):

    print("\n" + "=" * 80)
    print(
        f"PHASE 8 | MODE={mode} | K={k_budget}"
    )
    print("=" * 80)

    print(f"Device: {device}")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        k_budget,
        vocab_mapping,
        num_vocab,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = KuaiRandRecommender(
        num_videos=num_vocab,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        mode=mode,
    ).to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR
        / f"baseline_{mode}_K{k_budget}.pth"
    )

    metrics_path = (
        METRIC_DIR
        / f"metrics_{mode}_K{k_budget}.csv"
    )

    # -------------------------------------------------------------------------
    # Training state
    # -------------------------------------------------------------------------

    best_val_auc = -np.inf
    best_epoch = 0
    patience_counter = 0

    metrics_history = []

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0

        for (
            candidate_ids,
            history_ids,
            history_mask,
            labels,
        ) in train_loader:

            candidate_ids = candidate_ids.to(device)
            history_ids = history_ids.to(device)
            history_mask = history_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

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

            total_loss += (
                loss.item()
                * candidate_ids.size(0)
            )

        train_loss = (
            total_loss
            / len(train_dataset)
        )

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        val_auc = evaluate_auc(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f}"
        )

        metrics_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_auc,
            }
        )

        # ---------------------------------------------------------------------
        # Best checkpoint
        # ---------------------------------------------------------------------

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": mode,
                    "k_budget": k_budget,
                    "num_vocab": num_vocab,
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dims": HIDDEN_DIMS,
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )

            print(
                f"  -> Best checkpoint saved: "
                f"{checkpoint_path.name}"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"\n[EARLY STOPPING] "
                    f"Epoch {epoch}"
                )

                print(
                    f"Best Validation AUC: "
                    f"{best_val_auc:.4f}"
                )

                break

    # -------------------------------------------------------------------------
    # Save training metrics
    # -------------------------------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics_history
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Restore BEST checkpoint
    # -------------------------------------------------------------------------

    print(
        "\n[P3] Loading best validation checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -------------------------------------------------------------------------
    # Final TEST
    # -------------------------------------------------------------------------

    test_auc = evaluate_auc(
        model,
        test_loader,
        device,
    )

    print("\n" + "=" * 80)
    print(
        f"FINAL TEST ROC-AUC | "
        f"{mode} | K={k_budget}: "
        f"{test_auc:.4f}"
    )
    print("=" * 80)

    return {
        "Mode": mode,
        "K": k_budget,
        "Best_Epoch": best_epoch,
        "Best_Val_AUC": best_val_auc,
        "Test_AUC": test_auc,
        "Checkpoint": str(checkpoint_path),
        "Metrics": str(metrics_path),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("PHASE 8 — BASELINE TRAINING & EVALUATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load frozen vocabulary ONCE
    # -------------------------------------------------------------------------

    vocab = pd.read_parquet(
        VOCAB_FILE
    )

    vocab_mapping, max_vocab_idx = build_vocab_mapping(
        vocab
    )

    # Vocabulary indices are expected to start at 0.
    if max_vocab_idx < 0:
        raise ValueError(
            "Invalid vocabulary index."
        )

    # +1 is for padding index 0.
    num_vocab = max_vocab_idx + 1

    print(
        f"\nEmbedding vocabulary size: "
        f"{num_vocab + 1:,}"
    )

    print(
        "Index 0 reserved for padding."
    )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = get_device()

    # -------------------------------------------------------------------------
    # Run baselines
    # -------------------------------------------------------------------------

    results = []

    # Baseline: learned video-only model
    results.append(
        train_and_evaluate(
            mode="video_only",
            k_budget=K_VIDEO_ONLY,
            vocab_mapping=vocab_mapping,
            num_vocab=num_vocab,
            device=device,
        )
    )

    # Diagnostic: history-only model
    results.append(
        train_and_evaluate(
            mode="history_only",
            k_budget=K_HISTORY_ONLY,
            vocab_mapping=vocab_mapping,
            num_vocab=num_vocab,
            device=device,
        )
    )

    # -------------------------------------------------------------------------
    # Save final summary
    # -------------------------------------------------------------------------

    summary_path = (
        MODEL_DIR
        / "phase8_test_results.csv"
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved final results:\n"
        f"  {summary_path}"
    )

    print("\nResults:")

    print(
        results_df[
            [
                "Mode",
                "K",
                "Best_Epoch",
                "Best_Val_AUC",
                "Test_AUC",
            ]
        ].to_string(index=False)
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()