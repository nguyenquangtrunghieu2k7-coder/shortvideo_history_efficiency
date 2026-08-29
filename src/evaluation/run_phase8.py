from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.evaluation.dataset import (
    Phase8Dataset,
    build_vocab,
    load_split_with_labels,
    phase8_collate,
)
from src.evaluation.evaluate import evaluate_model, save_predictions
from src.evaluation.popularity import (
    compute_popularity_from_training_interactions,
    evaluate_popularity,
    save_popularity,
)
from src.models.history_only import HistoryOnlyModel
from src.models.recommender import HistoryVideoModel, VideoOnlyModel
from src.models.train import get_device, seed_everything, train_model


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "phase8"
EVENTS = PROCESSED / "evaluation_events.parquet"

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
EMB_DIM = 64
HIDDEN = 128
SEED = 42


def load_all():
    train = load_split_with_labels(PROCESSED / "train.parquet", EVENTS)
    val = load_split_with_labels(PROCESSED / "validation.parquet", EVENTS)
    test = load_split_with_labels(PROCESSED / "test.parquet", EVENTS)
    return train, val, test


def make_loader(df, vocab, condition, shuffle):
    dset = Phase8Dataset(df, vocab, condition)
    return DataLoader(
        dset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=phase8_collate,
        pin_memory=False,
    )


def attach_history(df, k):
    out = df.copy()

    if k == 0:
        out["history"] = [None] * len(out)
    else:
        col = "history_full" if k == "Full" else f"history_{k}"
        out["history"] = out[col]

    return out


def run_learned(
    name,
    condition,
    k,
    train,
    val,
    test,
    vocab,
    device,
):
    print("\n" + "=" * 80)
    print(f"{name} | K={k}")
    print("=" * 80)

    tr = attach_history(train, k)
    va = attach_history(val, k)
    te = attach_history(test, k)

    train_loader = make_loader(tr, vocab, condition, True)
    val_loader = make_loader(va, vocab, condition, False)
    test_loader = make_loader(te, vocab, condition, False)

    n_videos = len(vocab)

    if condition == "video_only":
        model = VideoOnlyModel(n_videos, EMB_DIM, HIDDEN)
    elif condition == "history_only":
        model = HistoryOnlyModel(n_videos, EMB_DIM, HIDDEN)
    else:
        model = HistoryVideoModel(n_videos, EMB_DIM, HIDDEN)

    safe_name = name.lower().replace("+", "_").replace(" ", "_")
    run_dir = OUT / "models" / safe_name / str(k)

    model, history, best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        epochs=EPOCHS,
        lr=LR,
        resume=True,
    )

    test_metrics = evaluate_model(model, test_loader, device)

    save_predictions(
        model,
        test_loader,
        device,
        te,
        run_dir / "test_predictions.parquet",
    )

    result = {
        "condition": name,
        "K": k,
        "best_validation_auc": best_auc,
        "test_auc": test_metrics["auc"],
        "test_n": test_metrics["n"],
        "test_positive_rate": test_metrics["positive_rate"],
        "device": str(device),
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    seed_everything(SEED)

    device = get_device("auto")
    print(f"Device: {device}")

    OUT.mkdir(parents=True, exist_ok=True)

    train, val, test = load_all()

    # Vocabulary is learned from TRAIN only.
    # Unknown candidate/history IDs in validation/test map to 0.
    vocab = build_vocab(train)

    with open(OUT / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in vocab.items()}, f)

    results = []

    # ---------------------------------------------------------------
    # Baseline 1 — Popularity
    # ---------------------------------------------------------------
    popularity = compute_popularity_from_training_interactions(train)
    save_popularity(popularity, OUT / "popularity.parquet")

    pop_result = evaluate_popularity(test, popularity)
    pop_result["condition"] = "Popularity"
    pop_result["K"] = None
    results.append(pop_result)

    with open(OUT / "popularity_result.json", "w", encoding="utf-8") as f:
        json.dump(pop_result, f, indent=2)

    # ---------------------------------------------------------------
    # Baseline 2 — Video-only learned model = K=0
    # ---------------------------------------------------------------
    results.append(
        run_learned(
            "Video-only",
            "video_only",
            0,
            train,
            val,
            test,
            vocab,
            device,
        )
    )

    # ---------------------------------------------------------------
    # Diagnostic — History-only
    # ---------------------------------------------------------------
    for k in [1, 5, 10, 20, 50, 100, "Full"]:
        results.append(
            run_learned(
                "History-only",
                "history_only",
                k,
                train,
                val,
                test,
                vocab,
                device,
            )
        )

    # ---------------------------------------------------------------
    # Main condition — History + Video
    # ---------------------------------------------------------------
    for k in [1, 5, 10, 20, 50, 100, "Full"]:
        results.append(
            run_learned(
                "History+Video",
                "history_video",
                k,
                train,
                val,
                test,
                vocab,
                device,
            )
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "phase8_results.csv", index=False)

    print("\n" + "=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
