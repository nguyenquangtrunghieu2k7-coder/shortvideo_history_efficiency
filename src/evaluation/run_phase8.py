from pathlib import Path

import pandas as pd
import torch

from .dataset import load_phase7_dataset
from .evaluate import evaluate_history_model
from ..models.recommender import HistoryVideoRecommender
from ..models.train import train_model


ROOT = Path(__file__).resolve().parents[2]

PHASE7_DIR = ROOT / "data/processed/phase7"
OUT_DIR = ROOT / "data/processed/phase8"

K_VALUES = [0, 1, 5, 10, 20, 50, 100, "Full"]

TRAIN_USERS = None


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_train_frames():

    frames = {}

    for k in K_VALUES:

        path = (
            PHASE7_DIR
            / f"history_K{k}.parquet"
        )

        df = pd.read_parquet(path)

        # Train split is identified by evaluation_id.
        # The Phase 7 artifact contains the complete
        # fixed evaluation population, so we use the
        # Phase 4/5 split artifacts to identify train rows.

        train_path = (
            ROOT
            / "data/processed/train.parquet"
        )

        train_ids = pd.read_parquet(
            train_path,
            columns=["eval_idx"],
        )["eval_idx"]

        if "eval_idx" in df.columns:
            df = df[
                df["eval_idx"].isin(train_ids)
            ].copy()

        elif "evaluation_id" in df.columns:
            ids = set(train_ids.tolist())

            df = df[
                df["evaluation_id"].isin(ids)
            ].copy()

        frames[k] = df

    return frames


def get_num_videos():

    path = (
        ROOT
        / "data/processed/phase5"
        / "development_events.parquet"
    )

    df = pd.read_parquet(
        path,
        columns=["video_id"],
    )

    return int(df["video_id"].max()) + 1


def main():

    print("=" * 80)
    print("PHASE 8 — BASELINES")
    print("=" * 80)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"\nDevice: {device}")

    # --------------------------------------------------
    # 1. LOAD TRAINING DATA
    # --------------------------------------------------

    print("\nLoading training histories...")

    train_frames = load_train_frames()

    for k, df in train_frames.items():
        print(
            f"  K={k}: {len(df):,} events"
        )

    # --------------------------------------------------
    # 2. TRAIN ONCE
    # --------------------------------------------------

    print("\nTraining shared History+Video model once...")

    num_videos = get_num_videos()

    model = train_model(
        train_frames=train_frames,
        num_videos=num_videos,
        epochs=5,
        batch_size=512,
        lr=1e-3,
    )

    # Freeze inference parameters.
    for p in model.parameters():
        p.requires_grad = False

    model.eval()

    # --------------------------------------------------
    # 3. VALIDATION
    # --------------------------------------------------

    print("\nEvaluating validation conditions...")

    validation_path = (
        ROOT
        / "data/processed/validation.parquet"
    )

    validation_ids = set(
        pd.read_parquet(
            validation_path,
            columns=["eval_idx"],
        )["eval_idx"]
    )

    results = []

    for k in K_VALUES:

        print(f"  Evaluating K={k}...")

        df = load_phase7_dataset(k)

        if "eval_idx" in df.columns:
            df = df[
                df["eval_idx"].isin(validation_ids)
            ].copy()

        result = evaluate_history_model(
            model,
            df,
            device,
        )

        result["split"] = "validation"
        result["K"] = str(k)

        results.append(result)

        print(
            f"    AUC = {result['roc_auc']:.6f}"
        )

    results_df = pd.DataFrame(results)

    results_file = (
        OUT_DIR
        / "phase8_validation_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    print(
        f"\nSaved:\n  {results_file}"
    )

    # --------------------------------------------------
    # 4. SAVE EXPERIMENT MANIFEST
    # --------------------------------------------------

    manifest = pd.DataFrame(
        [
            {
                "model": "HistoryVideoRecommender",
                "training_runs": 1,
                "K_values": ",".join(
                    map(str, K_VALUES)
                ),
                "test_used": False,
                "device": str(device),
                "checkpoint":
                    "history_video_model.pt",
            }
        ]
    )

    manifest_file = (
        OUT_DIR
        / "phase8_manifest.csv"
    )

    manifest.to_csv(
        manifest_file,
        index=False,
    )

    print(
        f"  {manifest_file}"
    )

    print("\n" + "=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()