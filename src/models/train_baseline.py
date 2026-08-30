# -*- coding: utf-8 -*-
"""
Phase 8: Baseline Models Training & Evaluation

Baselines:
    1. Video-only (K=0)
    2. History-only Diagnostic (K=Full)

Protocol:
    - Fixed train / validation / test splits
    - Fixed evaluation events
    - History is read from frozen split files
    - Validation AUC is used for early stopping
    - Test set is evaluated only with the best validation checkpoint
"""

from pathlib import Path

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

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FROZEN / EXPERIMENT CONFIGURATION
# =============================================================================

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 3

NUM_VIDEOS = 7388
L_MAX = 2048

# Phase 8 baselines
BASELINES = [
    ("video_only", 0),
    ("history_only", "full"),
]


# =============================================================================
# DATASET
# =============================================================================

class KuaiRandDataset(Dataset):
    """
    Dataset built only from frozen split files.

    Expected history columns:
        history_1
        history_5
        history_10
        history_20
        history_50
        history_100
        history_full
    """

    def __init__(self, split_name, k_budget):

        super().__init__()

        print(
            f"Loading {split_name} split "
            f"(K={k_budget})..."
        )

        # ---------------------------------------------------------------------
        # 1. LOAD FROZEN SPLIT
        # ---------------------------------------------------------------------

        split_path = (
            PROCESSED_DIR / f"{split_name}.parquet"
        )

        split_df = pd.read_parquet(split_path)

        # ---------------------------------------------------------------------
        # 2. LOAD FIXED EVALUATION LABELS
        # ---------------------------------------------------------------------

        events_path = (
            PROCESSED_DIR / "evaluation_events.parquet"
        )

        events_df = pd.read_parquet(
            events_path,
            columns=[
                "user_id",
                "video_id",
                "time_ms",
                "long_view",
            ],
        )

        # ---------------------------------------------------------------------
        # 3. MAP LABELS USING COMPOSITE KEY
        # ---------------------------------------------------------------------

        merged = pd.merge(
            split_df,
            events_df,
            on=[
                "user_id",
                "video_id",
                "time_ms",
            ],
            how="left",
            validate="one_to_one",
        )

        # ---------------------------------------------------------------------
        # 4. DATA CONTRACT ASSERTIONS
        # ---------------------------------------------------------------------

        assert len(merged) == len(split_df), (
            f"Row count changed after merge: "
            f"{len(split_df)} -> {len(merged)}"
        )

        assert not merged["long_view"].isna().any(), (
            f"{split_name}: unmapped long_view labels detected."
        )

        labels = merged["long_view"].to_numpy()

        assert set(pd.unique(labels)).issubset({0, 1}), (
            f"{split_name}: long_view must be binary."
        )

        # ---------------------------------------------------------------------
        # 5. CANDIDATE ID
        # ---------------------------------------------------------------------

        candidate_ids = merged["video_id"].to_numpy()

        assert candidate_ids.min() >= 0, (
            f"{split_name}: negative video_id detected."
        )

        assert candidate_ids.max() < NUM_VIDEOS, (
            f"{split_name}: video_id exceeds NUM_VIDEOS="
            f"{NUM_VIDEOS}."
        )

        # Shift by +1 because embedding index 0 is padding.
        self.candidate_ids = torch.tensor(
            candidate_ids + 1,
            dtype=torch.long,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32,
        )

        # ---------------------------------------------------------------------
        # 6. HISTORY
        # ---------------------------------------------------------------------

        if k_budget == 0:

            # K=0 has no usable history.
            self.history_ids = torch.zeros(
                (len(merged), 1),
                dtype=torch.long,
            )

            self.history_mask = torch.zeros(
                (len(merged), 1),
                dtype=torch.float32,
            )

        else:

            if k_budget == "full":
                hist_col = "history_full"
            else:
                hist_col = f"history_{k_budget}"

            assert hist_col in merged.columns, (
                f"{split_name}: missing required column "
                f"'{hist_col}'."
            )

            hist_lists = merged[hist_col].tolist()

            # ---------------------------------------------------------------
            # Truncate Full history to protocol limit.
            # ---------------------------------------------------------------

            truncated_hists = []

            for hist in hist_lists:

                if hist is None or len(hist) == 0:
                    truncated_hists.append([])

                elif len(hist) > L_MAX:
                    truncated_hists.append(
                        hist[-L_MAX:]
                    )

                else:
                    truncated_hists.append(hist)

            max_len = max(
                (len(hist) for hist in truncated_hists),
                default=0,
            )

            # At least one position is required for tensor construction.
            max_len = max(max_len, 1)

            num_samples = len(merged)

            self.history_ids = torch.zeros(
                (num_samples, max_len),
                dtype=torch.long,
            )

            self.history_mask = torch.zeros(
                (num_samples, max_len),
                dtype=torch.float32,
            )

            # ---------------------------------------------------------------
            # Convert history video IDs -> shifted embedding IDs.
            # ---------------------------------------------------------------

            for i, hist in enumerate(truncated_hists):

                if not hist:
                    continue

                hist_ids = [int(video_id) for video_id in hist]

                assert min(hist_ids) >= 0, (
                    f"{split_name}: negative history video_id detected."
                )

                assert max(hist_ids) < NUM_VIDEOS, (
                    f"{split_name}: history video_id exceeds "
                    f"NUM_VIDEOS={NUM_VIDEOS}."
                )

                seq_len = len(hist_ids)

                self.history_ids[i, :seq_len] = torch.tensor(
                    [video_id + 1 for video_id in hist_ids],
                    dtype=torch.long,
                )

                self.history_mask[i, :seq_len] = 1.0

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
# DEVICE
# =============================================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, loader, device):

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

    auc = roc_auc_score(
        labels,
        predictions,
    )

    return auc


# =============================================================================
# TRAIN ONE BASELINE
# =============================================================================

def train_and_evaluate(
    mode="video_only",
    k_budget=0,
):

    print("\n" + "=" * 80)
    print(
        f"PHASE 8 | MODE={mode} | K={k_budget}"
    )
    print("=" * 80)

    device = get_device()

    print(
        f"Device: {device}"
    )

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    train_dataset = KuaiRandDataset(
        "train",
        k_budget,
    )

    val_dataset = KuaiRandDataset(
        "validation",
        k_budget,
    )

    test_dataset = KuaiRandDataset(
        "test",
        k_budget,
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

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    model = KuaiRandRecommender(
        num_videos=NUM_VIDEOS,
        mode=mode,
    ).to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------------------------------
    # OUTPUT PATHS
    # -------------------------------------------------------------------------

    save_path = (
        MODEL_DIR
        / f"baseline_{mode}_K{k_budget}.pth"
    )

    metrics_path = (
        MODEL_DIR
        / f"metrics_{mode}_K{k_budget}.csv"
    )

    # -------------------------------------------------------------------------
    # TRAINING STATE
    # -------------------------------------------------------------------------

    best_val_auc = -float("inf")
    patience_counter = 0

    metrics_history = []

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------

    for epoch in range(EPOCHS):

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
        # VALIDATION
        # ---------------------------------------------------------------------

        val_auc = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f}"
        )

        metrics_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_auc": val_auc,
            }
        )

        # ---------------------------------------------------------------------
        # BEST CHECKPOINT
        # ---------------------------------------------------------------------

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                save_path,
            )

            print(
                f"  -> Best checkpoint saved: "
                f"{save_path.name}"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "\n[EARLY STOPPING] "
                    f"Epoch {epoch + 1}"
                )

                print(
                    f"Best Validation AUC: "
                    f"{best_val_auc:.4f}"
                )

                break

    # -------------------------------------------------------------------------
    # SAVE TRAINING METRICS
    # -------------------------------------------------------------------------

    pd.DataFrame(
        metrics_history
    ).to_csv(
        metrics_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # RESTORE BEST CHECKPOINT
    # -------------------------------------------------------------------------

    print(
        "\n[P3] Loading best validation checkpoint..."
    )

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
        )
    )

    # -------------------------------------------------------------------------
    # FINAL TEST
    # -------------------------------------------------------------------------

    test_auc = evaluate(
        model,
        test_loader,
        device,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"FINAL TEST ROC-AUC | "
        f"{mode} | K={k_budget}: "
        f"{test_auc:.4f}"
    )

    print("=" * 80)

    # -------------------------------------------------------------------------
    # RETURN RESULT
    # -------------------------------------------------------------------------

    return {
        "Mode": mode,
        "K": k_budget,
        "Best_Val_AUC": best_val_auc,
        "Test_AUC": test_auc,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("PHASE 8 — BASELINE TRAINING & EVALUATION")
    print("=" * 80)

    summary_path = (
        MODEL_DIR
        / "phase8_test_results.csv"
    )

    # Start a fresh Phase 8 result file.
    if summary_path.exists():
        summary_path.unlink()

    results = []

    # -------------------------------------------------------------------------
    # BASELINE 1 — VIDEO ONLY
    # -------------------------------------------------------------------------

    results.append(
        train_and_evaluate(
            mode="video_only",
            k_budget=0,
        )
    )

    # -------------------------------------------------------------------------
    # BASELINE 2 — HISTORY ONLY DIAGNOSTIC
    # -------------------------------------------------------------------------

    results.append(
        train_and_evaluate(
            mode="history_only",
            k_budget="full",
        )
    )

    # -------------------------------------------------------------------------
    # SAVE SUMMARY
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(results)

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved summary:\n"
        f"  {summary_path}"
    )

    print("\nResults:")

    print(
        summary_df.to_string(
            index=False
        )
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
