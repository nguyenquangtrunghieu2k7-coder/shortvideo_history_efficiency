from pathlib import Path
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/phase8/baseline_results.csv"
OUTPUT = ROOT / "data/processed/phase8/adaptive_policy_summary.csv"
POLICY_OUTPUT = ROOT / "data/processed/phase8/adaptive_policy.txt"

# Prototype setting: maximum allowed validation-AUC drop vs Full
EPSILON = 0.05

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Cannot find baseline results: {INPUT}")

    df = pd.read_csv(INPUT)
    required = {"K", "best_val_auc", "test_auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Normalize K values; keep "Full" as a string.
    df["K_label"] = df["K"].astype(str).str.strip()
    full_rows = df[df["K_label"].str.lower() == "full"]

    if len(full_rows) != 1:
        raise ValueError("Expected exactly one Full baseline row.")

    full = full_rows.iloc[0]
    full_val_auc = float(full["best_val_auc"])
    threshold = full_val_auc - EPSILON

    # Only numeric K values are candidates for the adaptive policy.
    candidates = df[df["K_label"].str.fullmatch(r"\d+")].copy()
    candidates["K_numeric"] = candidates["K_label"].astype(int)
    candidates = candidates.sort_values("K_numeric")

    eligible = candidates[candidates["best_val_auc"] >= threshold]

    if eligible.empty:
        selected = None
        print("No numeric K meets the validation-AUC threshold.")
    else:
        selected = eligible.iloc[0]
        print("\n=== ADAPTIVE POLICY ===")
        print(f"Full validation AUC : {full_val_auc:.6f}")
        print(f"Epsilon             : {EPSILON:.3f}")
        print(f"Required validation : {threshold:.6f}")
        print(f"Selected K          : {int(selected['K_numeric'])}")
        print(f"Selected val AUC    : {selected['best_val_auc']:.6f}")
        print(f"Selected test AUC   : {selected['test_auc']:.6f}")
        print(f"Full test AUC       : {float(full['test_auc']):.6f}")

    # Save an auditable comparison table.
    df["validation_threshold"] = threshold
    df["epsilon"] = EPSILON
    df["eligible_by_validation"] = (
        df["K_label"].str.fullmatch(r"\d+").fillna(False)
        & (df["best_val_auc"] >= threshold)
    )
    df["selected_by_policy"] = False

    if selected is not None:
        df.loc[df["K_label"] == selected["K_label"], "selected_by_policy"] = True

    df.to_csv(OUTPUT, index=False)

    if selected is not None:
        POLICY_OUTPUT.write_text(
            "policy_name=global_minimum_history\n"
            f"epsilon={EPSILON}\n"
            f"reference=Full\n"
            f"full_validation_auc={full_val_auc:.6f}\n"
            f"validation_threshold={threshold:.6f}\n"
            f"selected_K={int(selected['K_numeric'])}\n"
            "selection_metric=best_val_auc\n"
            "test_set_used_for_selection=false\n",
            encoding="utf-8",
        )

    print(f"\nComparison saved to: {OUTPUT.relative_to(ROOT)}")
    if selected is not None:
        print(f"Policy saved to:     {POLICY_OUTPUT.relative_to(ROOT)}")
        print("\n=== VALIDATION COMPARISON ===")
        display_cols = ["K_label", "best_val_auc", "test_auc",
                        "eligible_by_validation", "selected_by_policy"]
        print(df[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
