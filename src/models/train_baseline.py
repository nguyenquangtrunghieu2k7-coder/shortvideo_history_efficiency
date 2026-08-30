# -*- coding: utf-8 -*-

"""
PHASE 8 — BASELINE TRAINING & EVALUATION

Baselines:
    1. video_only   : K=0
    2. history_only : Diagnostic, K=Full
    3. full         : Candidate + history

Data source:
    data/processed/splits/K*/
    data/processed/video_vocab.parquet

Frozen vocabulary:
    raw_video_id -> video_idx
    model index = video_idx + 1
    index 0      = padding

IMPORTANT:
    This script does NOT use:
        data/processed/train.parquet
        old history artifacts

It only uses the Phase 7 -> Phase 8 split pipeline.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from models import KuaiRandRecommender


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
VOCAB_FILE = PROCESSED_DIR / "video_vocab.parquet"

OUTPUT_DIR = PROCESSED_DIR / "phase8"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULT_DIR = OUTPUT_DIR / "results"


# =============================================================================
# FROZEN EXPERIMENT CONFIG
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

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EMBEDDING_DIM = 64
HIDDEN_DIMS = [128, 64]

PATIENCE = 3

NUM_WORKERS = 0

SEED = 42

# If True, an existing completed result is skipped.
SKIP_EXISTING = False


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
# K HELPERS
# =============================================================================

def k_name(k):

    if k == "Full":
        return "KFull"

    return f"K{k}"


def mode_name(mode, k):

    if mode == "video_only":
        return "video_only_K0"

    if mode == "history_only":
        return "history_only_Kfull"

    return f"full_K{k}"


# =============================================================================
# LOAD FROZEN VIDEO VOCABULARY
# =============================================================================

def load_video_vocabulary():

    print("=" * 80)
    print("LOADING FROZEN VIDEO VOCABULARY")
    print("=" * 80)

    if not VOCAB_FILE.exists():
        raise FileNotFoundError(
            f"Frozen vocabulary not found:\n{VOCAB_FILE}"
        )

    vocab = pd.read_parquet(VOCAB_FILE)

    required = {
        "raw_video_id",
        "video_idx",
    }

    missing = required - set(vocab.columns)

    if missing:
        raise ValueError(
            f"Vocabulary missing columns: {sorted(missing)}"
        )

    vocab = vocab[
        ["raw_video_id", "video_idx"]
    ].copy()

    if vocab["raw_video_id"].isna().any():
        raise ValueError(
            "Vocabulary contains null raw_video_id."
        )

    if vocab["video_idx"].isna().any():
        raise ValueError(
            "Vocabulary contains null video_idx."
        )

    if not vocab["raw_video_id"].is_unique:
        raise ValueError(
            "raw_video_id is not unique in frozen vocabulary."
        )

    if not vocab["video_idx"].is_unique:
        raise ValueError(
            "video_idx is not unique in frozen vocabulary."
        )

    min_idx = int(vocab["video_idx"].min())
    max_idx = int(vocab["video_idx"].max())

    if min_idx != 0:
        raise ValueError(
            f"Expected vocabulary video_idx to start at 0, "
            f"got {min_idx}."
        )

    expected = np.arange(
        len(vocab),
        dtype=np.int64,
    )

    actual = np.sort(
        vocab["video_idx"].to_numpy(dtype=np.int64)
    )

    if not np.array_equal(actual, expected):
        raise ValueError(
            "video_idx is not a contiguous range 0..N-1."
        )

    # -------------------------------------------------------------------------
    # Dense lookup array
    #
    # raw_video_id can be much larger than the number of videos.
    # A dense int32 array is substantially more memory-efficient than
    # a Python dictionary.
    # -------------------------------------------------------------------------

    raw_ids = vocab["raw_video_id"].to_numpy(
        dtype=np.int64
    )

    video_indices = vocab["video_idx"].to_numpy(
        dtype=np.int64
    )

    max_raw_id = int(raw_ids.max())

    # -1 means "not present in frozen vocabulary".
    lookup = np.full(
        max_raw_id + 1,
        -1,
        dtype=np.int32,
    )

    lookup[raw_ids] = (
        video_indices + 1
    ).astype(np.int32)

    num_vocab = len(vocab) + 1

    print(
        f"Raw video ID column: raw_video_id"
    )
    print(
        f"Vocabulary index column: video_idx"
    )
    print(
        f"Vocabulary index range: "
        f"{min_idx} -> {max_idx}"
    )
    print(
        f"Unique vocabulary IDs: {len(vocab):,}"
    )
    print(
        f"Max raw video ID: {max_raw_id:,}"
    )
    print(
        f"Embedding vocabulary size: {num_vocab:,}"
    )
    print(
        "Index 0 reserved for padding."
    )

    print()

    return lookup, num_vocab


# =============================================================================
# RAW ID -> MODEL INDEX
# =============================================================================

def map_raw_ids(
    raw_ids,
    lookup,
    context,
):

    if raw_ids is None:
        return []

    if isinstance(raw_ids, np.ndarray):
        raw_ids = raw_ids.tolist()

    if not isinstance(raw_ids, (list, tuple)):
        raise ValueError(
            f"{context}: history must be list/tuple/ndarray, "
            f"got {type(raw_ids)}"
        )

    if len(raw_ids) == 0:
        return []

    raw = np.asarray(
        raw_ids,
        dtype=np.int64,
    )

    if raw.min() < 0:
        raise ValueError(
            f"{context}: negative raw video ID detected."
        )

    if raw.max() >= len(lookup):
        bad = raw[
            raw >= len(lookup)
        ][:10].tolist()

        raise ValueError(
            f"{context}: raw video IDs exceed vocabulary lookup. "
            f"Examples: {bad}"
        )

    mapped = lookup[raw]

    missing_mask = mapped < 0

    if missing_mask.any():

        missing = raw[
            missing_mask
        ][:10].tolist()

        raise ValueError(
            f"{context}: raw video IDs are absent from "
            f"frozen vocabulary. Examples: {missing}"
        )

    return mapped.astype(
        np.int64
    ).tolist()


# =============================================================================
# DATASET
# =============================================================================

class KuaiRandDataset(Dataset):

    def __init__(
        self,
        split_name,
        k_budget,
        lookup,
    ):

        self.split_name = split_name
        self.k_budget = k_budget

        path = (
            SPLIT_DIR
            / k_name(k_budget)
            / f"{split_name}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Split not found:\n{path}"
            )

        print(
            f"Loading {split_name} split "
            f"(K={k_budget})..."
        )

        df = pd.read_parquet(path)

        required = {
            "evaluation_id",
            "user_id",
            "video_id",
            "time_ms",
            "long_view",
            "history",
            "history_len",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{split_name}: missing required columns "
                f"{sorted(missing)}"
            )

        # ---------------------------------------------------------------------
        # Labels
        # ---------------------------------------------------------------------

        labels = df["long_view"]

        if labels.isna().any():
            raise ValueError(
                f"{split_name}: long_view contains null values."
            )

        unique_labels = set(
            labels.astype(int).unique().tolist()
        )

        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"{split_name}: long_view must be binary, "
                f"got {sorted(unique_labels)}"
            )

        self.labels = labels.to_numpy(
            dtype=np.float32
        )

        # ---------------------------------------------------------------------
        # Candidate IDs
        # ---------------------------------------------------------------------

        raw_candidates = df["video_id"].to_numpy(
            dtype=np.int64
        )

        self.candidate_ids = []

        for i, raw_id in enumerate(raw_candidates):

            mapped = map_raw_ids(
                [raw_id],
                lookup,
                f"{split_name} candidate row {i}",
            )[0]

            self.candidate_ids.append(
                mapped
            )

        self.candidate_ids = np.asarray(
            self.candidate_ids,
            dtype=np.int64,
        )

        # ---------------------------------------------------------------------
        # Histories
        # ---------------------------------------------------------------------

        self.histories = []

        for i, hist in enumerate(
            df["history"]
        ):

            mapped = map_raw_ids(
                hist,
                lookup,
                f"{split_name} history row {i}",
            )

            declared_len = int(
                df.iloc[i]["history_len"]
            )

            if declared_len != len(mapped):
                raise ValueError(
                    f"{split_name} row {i}: "
                    f"history_len={declared_len}, "
                    f"actual={len(mapped)}"
                )

            self.histories.append(mapped)

        # ---------------------------------------------------------------------
        # Evaluation IDs
        # ---------------------------------------------------------------------

        self.evaluation_ids = df[
            "evaluation_id"
        ].to_numpy(
            dtype=np.int64
        )

        if len(
            np.unique(self.evaluation_ids)
        ) != len(self.evaluation_ids):

            raise ValueError(
                f"{split_name}: evaluation_id is not unique."
            )

        # ---------------------------------------------------------------------
        # K audit
        # ---------------------------------------------------------------------

        if k_budget == 0:

            if any(
                len(hist) != 0
                for hist in self.histories
            ):
                raise ValueError(
                    f"{split_name}: K=0 contains non-empty history."
                )

        elif k_budget != "Full":

            max_len = max(
                (len(hist) for hist in self.histories),
                default=0,
            )

            if max_len > int(k_budget):
                raise ValueError(
                    f"{split_name}: K={k_budget}, "
                    f"history length exceeds K. "
                    f"Max={max_len}"
                )

        print(
            f"  Samples: {len(self):,}"
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        candidate_id = int(
            self.candidate_ids[idx]
        )

        history = self.histories[idx]

        label = float(
            self.labels[idx]
        )

        return (
            candidate_id,
            history,
            label,
        )


# =============================================================================
# COLLATE
# =============================================================================

def make_collate_fn():

    def collate(batch):

        candidate_ids = [
            item[0]
            for item in batch
        ]

        histories = [
            item[1]
            for item in batch
        ]

        labels = [
            item[2]
            for item in batch
        ]

        max_len = max(
            (len(hist) for hist in histories),
            default=0,
        )

        # At least one column is needed for tensor shape.
        seq_len = max(
            1,
            max_len,
        )

        history_ids = np.zeros(
            (
                len(batch),
                seq_len,
            ),
            dtype=np.int64,
        )

        history_mask = np.zeros(
            (
                len(batch),
                seq_len,
            ),
            dtype=np.float32,
        )

        for i, hist in enumerate(histories):

            if len(hist) == 0:
                continue

            history_ids[
                i,
                :len(hist),
            ] = hist

            history_mask[
                i,
                :len(hist),
            ] = 1.0

        return {
            "candidate_ids": torch.tensor(
                candidate_ids,
                dtype=torch.long,
            ),
            "history_ids": torch.tensor(
                history_ids,
                dtype=torch.long,
            ),
            "history_mask": torch.tensor(
                history_mask,
                dtype=torch.float32,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.float32,
            ),
        }

    return collate


# =============================================================================
# DATALOADERS
# =============================================================================

def build_dataloaders(
    k_budget,
    lookup,
):

    collate_fn = make_collate_fn()

    train_dataset = KuaiRandDataset(
        "train",
        k_budget,
        lookup,
    )

    val_dataset = KuaiRandDataset(
        "validation",
        k_budget,
        lookup,
    )

    test_dataset = KuaiRandDataset(
        "test",
        k_budget,
        lookup,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# =============================================================================
# MODEL
# =============================================================================

def build_model(
    mode,
    num_vocab,
    device,
):

    model = KuaiRandRecommender(
        num_videos=num_vocab,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        mode=mode,
    )

    model = model.to(device)

    return model


# =============================================================================
# FORWARD
# =============================================================================

def forward_batch(
    model,
    batch,
    device,
):

    candidate_ids = batch[
        "candidate_ids"
    ].to(device)

    history_ids = batch[
        "history_ids"
    ].to(device)

    history_mask = batch[
        "history_mask"
    ].to(device)

    labels = batch[
        "labels"
    ].to(device)

    predictions = model(
        candidate_ids,
        history_ids,
        history_mask,
    )

    return predictions, labels


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion,
):

    model.eval()

    losses = []
    all_predictions = []
    all_labels = []

    for batch in loader:

        predictions, labels = forward_batch(
            model,
            batch,
            device,
        )

        loss = criterion(
            predictions,
            labels,
        )

        losses.append(
            loss.item()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    if len(
        np.unique(all_labels)
    ) < 2:

        raise ValueError(
            "ROC-AUC cannot be computed: "
            "only one label class is present."
        )

    auc = roc_auc_score(
        all_labels,
        all_predictions,
    )

    mean_loss = float(
        np.mean(losses)
    )

    return mean_loss, float(auc)


# =============================================================================
# PRE-FLIGHT AUDIT
# =============================================================================

def preflight_audit(
    lookup,
):

    print("=" * 80)
    print("PHASE 8 PRE-FLIGHT AUDIT")
    print("=" * 80)

    for k in K_VALUES:

        print()
        print(
            f"Checking {k_name(k)}..."
        )

        for split in [
            "train",
            "validation",
            "test",
        ]:

            # Dataset constructor performs all validations.
            dataset = KuaiRandDataset(
                split,
                k,
                lookup,
            )

            if len(dataset) == 0:
                raise ValueError(
                    f"{split} / K={k} is empty."
                )

        print(
            f"  K={k}: PASS"
        )

    print()
    print("=" * 80)
    print("PRE-FLIGHT AUDIT: PASS")
    print("All K configurations are safe to train.")
    print("=" * 80)
    print()


# =============================================================================
# CHECKPOINT
# =============================================================================

def checkpoint_path(
    mode,
    k,
):

    return (
        CHECKPOINT_DIR
        / f"baseline_{mode_name(mode, k)}.pth"
    )


def result_path(
    mode,
    k,
):

    return (
        RESULT_DIR
        / f"baseline_{mode_name(mode, k)}.json"
    )


# =============================================================================
# TRAIN ONE MODEL
# =============================================================================

def train_and_evaluate(
    mode,
    k_budget,
    lookup,
    num_vocab,
    device,
):

    experiment_name = mode_name(
        mode,
        k_budget,
    )

    print("=" * 80)
    print(
        f"PHASE 8 | MODE={mode} | K={k_budget}"
    )
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    final_result = result_path(
        mode,
        k_budget,
    )

    if (
        SKIP_EXISTING
        and final_result.exists()
    ):

        print(
            f"Existing result found:\n"
            f"  {final_result}"
        )

        print(
            "Skipping."
        )

        with open(
            final_result,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        k_budget,
        lookup,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = build_model(
        mode,
        num_vocab,
        device,
    )

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------------------------
    # Training state
    # -------------------------------------------------------------------------

    best_val_auc = -np.inf
    best_epoch = 0
    patience_counter = 0

    history = []

    ckpt = checkpoint_path(
        mode,
        k_budget,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        train_losses = []

        for batch in train_loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions, labels = forward_batch(
                model,
                batch,
                device,
            )

            loss = criterion(
                predictions,
                labels,
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(train_losses)
        )

        val_loss, val_auc = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_auc,
        }

        history.append(
            epoch_record
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f}"
        )

        # ---------------------------------------------------------------------
        # Best checkpoint
        # ---------------------------------------------------------------------

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            # Save state_dict only.
            # This avoids PyTorch 2.6 pickle/weights_only problems.
            torch.save(
                model.state_dict(),
                ckpt,
            )

            print(
                f"  -> Best checkpoint saved: "
                f"{ckpt.name}"
            )

        else:

            patience_counter += 1

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        if patience_counter >= PATIENCE:

            print()
            print(
                f"[EARLY STOPPING] Epoch {epoch}"
            )

            break

    print()
    print(
        f"Best Validation AUC: "
        f"{best_val_auc:.4f}"
    )

    # -------------------------------------------------------------------------
    # Load best validation checkpoint
    # -------------------------------------------------------------------------

    print()
    print(
        "[P3] Loading best validation checkpoint..."
    )

    state_dict = torch.load(
        ckpt,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    # -------------------------------------------------------------------------
    # Final test
    # -------------------------------------------------------------------------

    test_loss, test_auc = evaluate(
        model,
        test_loader,
        device,
        criterion,
    )

    print()
    print("=" * 80)
    print(
        f"FINAL TEST ROC-AUC | "
        f"{mode} | K={k_budget}: "
        f"{test_auc:.4f}"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Save result
    # -------------------------------------------------------------------------

    result = {
        "experiment": experiment_name,
        "mode": mode,
        "K": k_budget,

        "device": str(device),

        "num_vocab": int(num_vocab),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dims": HIDDEN_DIMS,

        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,

        "best_epoch": best_epoch,
        "best_validation_roc_auc": best_val_auc,

        "test_loss": test_loss,
        "test_roc_auc": test_auc,

        "checkpoint": str(ckpt),

        "history": history,
    }

    with open(
        final_result,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved result:"
    )
    print(
        f"  {final_result}"
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print("=" * 80)
    print("PHASE 8 — BASELINE TRAINING & EVALUATION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Vocabulary
    # -------------------------------------------------------------------------

    lookup, num_vocab = load_video_vocabulary()

    # -------------------------------------------------------------------------
    # PRE-FLIGHT
    #
    # IMPORTANT:
    # This happens BEFORE any model training.
    # -------------------------------------------------------------------------

    preflight_audit(
        lookup
    )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = get_device()

    print(
        f"Training device: {device}"
    )

    print()

    # -------------------------------------------------------------------------
    # Experiment definitions
    # -------------------------------------------------------------------------

    experiments = [
        (
            "video_only",
            0,
        ),

        (
            "history_only",
            "Full",
        ),
    ]

    # -------------------------------------------------------------------------
    # Run baselines
    # -------------------------------------------------------------------------

    results = []

    for mode, k in experiments:

        result = train_and_evaluate(
            mode=mode,
            k_budget=k,
            lookup=lookup,
            num_vocab=num_vocab,
            device=device,
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary_rows = []

    for result in results:

        summary_rows.append(
            {
                "experiment": result[
                    "experiment"
                ],

                "mode": result[
                    "mode"
                ],

                "K": result[
                    "K"
                ],

                "best_epoch": result[
                    "best_epoch"
                ],

                "validation_roc_auc": result[
                    "best_validation_roc_auc"
                ],

                "test_roc_auc": result[
                    "test_roc_auc"
                ],
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULT_DIR
        / "baseline_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)

    print()
    print("Baseline results:")

    for row in summary_rows:

        print(
            f"  {row['experiment']:<25} "
            f"Val AUC: "
            f"{row['validation_roc_auc']:.4f} | "
            f"Test AUC: "
            f"{row['test_roc_auc']:.4f}"
        )

    print()
    print(
        f"Summary saved:"
    )
    print(
        f"  {summary_path}"
    )

    print()
    print("Artifacts:")

    for path in sorted(
        OUTPUT_DIR.rglob("*")
    ):

        if path.is_file():

            print(
                f"  {path}"
            )


if __name__ == "__main__":
    main()
