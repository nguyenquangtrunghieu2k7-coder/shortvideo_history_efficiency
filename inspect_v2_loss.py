
import pandas as pd

df = pd.read_parquet(
    "data/processed/phase9_v2/adaptive_v2_test_decisions.parquet"
)

k_values = [0, 1, 5, 10, 20, 50, 100, "Full"]

rows = []

for selected_k, group in df.groupby("selected_K"):
    losses = {}

    for k in k_values:
        col = f"predicted_loss_K{k}"
        losses[str(k)] = group[col].mean()

    row = {
        "selected_K": str(selected_k),
        "requests": len(group),
    }

    row.update(losses)
    rows.append(row)

loss_df = pd.DataFrame(rows).set_index("selected_K")

print("\n=== MEAN PREDICTED LOSS BY SELECTED K ===")
print(loss_df.round(4).to_string())

print("\n=== BEST PREDICTED K BY GROUP ===")

for selected_k, group in df.groupby("selected_K"):
    mean_losses = {
        str(k): group[f"predicted_loss_K{k}"].mean()
        for k in k_values
    }

    best_k = min(mean_losses, key=mean_losses.get)

    print(
        f"Selected={selected_k:>4} | "
        f"Best predicted K={best_k:>4} | "
        f"Requests={len(group):>4}"
    )
