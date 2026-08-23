from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


def build_user_split(df):
    users = df["user_id"].drop_duplicates().sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    )

    n = len(users)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_users = set(users.iloc[:n_train])
    val_users = set(users.iloc[n_train:n_train + n_val])
    test_users = set(users.iloc[n_train + n_val:])

    return train_users, val_users, test_users


def audit_split(df, train_users, val_users, test_users):
    print("\n" + "=" * 80)
    print("USER-LEVEL SPLIT AUDIT")
    print("=" * 80)

    sets = {
        "train": train_users,
        "validation": val_users,
        "test": test_users,
    }

    print("Users:")
    for name, users in sets.items():
        print(f"  {name:<10}: {len(users):,}")

    print("\nUser overlap:")

    failed = False

    names = list(sets.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            overlap = sets[a] & sets[b]

            print(f"  {a} ∩ {b}: {len(overlap):,}")

            if overlap:
                failed = True

    print("\nEvaluation events:")

    for name, users in sets.items():
        subset = df[df["user_id"].isin(users)]

        print(
            f"  {name:<10}: "
            f"{len(subset):>7,} events "
            f"({len(subset) / len(df):.2%})"
        )

    if failed:
        print("\nUser-disjoint split: FAIL")
    else:
        print("\nUser-disjoint split: PASS")


def audit_event_distribution(df, train_users, val_users, test_users):
    print("\n" + "=" * 80)
    print("SPLIT DISTRIBUTION AUDIT")
    print("=" * 80)

    splits = {
        "train": train_users,
        "validation": val_users,
        "test": test_users,
    }

    for name, users in splits.items():
        subset = df[df["user_id"].isin(users)]

        click_rate = subset["is_click"].mean() \
            if "is_click" in subset.columns else None

        print(f"\n{name}:")
        print(f"  users:  {subset['user_id'].nunique():,}")
        print(f"  events: {len(subset):,}")

        if click_rate is not None:
            print(f"  click rate: {click_rate:.2%}")


def audit_user_history_leakage(df, train_users, val_users, test_users):
    print("\n" + "=" * 80)
    print("USER HISTORY LEAKAGE AUDIT")
    print("=" * 80)

    all_users = set(df["user_id"])

    assigned = train_users | val_users | test_users

    missing = all_users - assigned
    extra = assigned - all_users

    print(f"Users missing from split: {len(missing):,}")
    print(f"Unknown users in split:   {len(extra):,}")

    if not missing and not extra:
        print("User assignment coverage: PASS")
    else:
        print("User assignment coverage: FAIL")


def main():
    print("=" * 80)
    print("AUDIT TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {HISTORY_PATH}")

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    df = pd.read_parquet(HISTORY_PATH)

    print(f"\nEvaluation events: {len(df):,}")
    print(f"Unique users:      {df['user_id'].nunique():,}")

    train_users, val_users, test_users = build_user_split(df)

    audit_split(
        df,
        train_users,
        val_users,
        test_users,
    )

    audit_user_history_leakage(
        df,
        train_users,
        val_users,
        test_users,
    )

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()