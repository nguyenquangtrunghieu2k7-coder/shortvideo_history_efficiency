from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "evaluation_history.parquet"


WINDOWS = [1, 5, 10, 20, 50, 100, "full"]


def to_array(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return np.asarray([])


def collect_history_videos(df, col):
    videos = set()

    for x in df[col]:
        arr = to_array(x)

        if len(arr):
            videos.update(arr.tolist())

    return videos


def audit_video_vocabulary(df):
    print("\n" + "=" * 80)
    print("VIDEO VOCABULARY AUDIT")
    print("=" * 80)

    candidate_videos = set(df["video_id"].unique())

    print(f"Unique candidate videos: {len(candidate_videos):,}")

    for w in WINDOWS:
        col = (
            "history_full"
            if w == "full"
            else f"history_{w}"
        )

        history_videos = collect_history_videos(df, col)

        overlap = candidate_videos & history_videos
        candidate_oov = candidate_videos - history_videos

        print(f"\nW={str(w):<4}")
        print(f"  unique history videos:   {len(history_videos):,}")
        print(f"  candidate/history overlap: {len(overlap):,}")
        print(f"  candidate OOV:            {len(candidate_oov):,}")

        if candidate_videos:
            print(
                f"  candidate OOV rate:       "
                f"{len(candidate_oov) / len(candidate_videos):.2%}"
            )


def audit_event_level_oov(df):
    print("\n" + "=" * 80)
    print("EVENT-LEVEL CANDIDATE OOV AUDIT")
    print("=" * 80)

    for w in WINDOWS:
        col = (
            "history_full"
            if w == "full"
            else f"history_{w}"
        )

        history_videos = collect_history_videos(df, col)

        oov = ~df["video_id"].isin(history_videos)

        print(
            f"W={str(w):<4} "
            f"OOV events: {oov.sum():>7,} "
            f"({oov.mean():.2%})"
        )


def audit_history_ids(df):
    print("\n" + "=" * 80)
    print("HISTORY ID VALIDITY")
    print("=" * 80)

    for w in WINDOWS:
        col = (
            "history_full"
            if w == "full"
            else f"history_{w}"
        )

        invalid = 0
        negative = 0

        for x in df[col]:
            arr = to_array(x)

            if len(arr) == 0:
                continue

            negative += int((arr < 0).sum())

        print(
            f"W={str(w):<4} "
            f"negative IDs: {negative:,}"
        )

        invalid += negative

        if invalid == 0:
            print("       ID validity: PASS")
        else:
            print("       ID validity: FAIL")


def audit_user_vocabulary(df):
    print("\n" + "=" * 80)
    print("USER VOCABULARY AUDIT")
    print("=" * 80)

    users = df["user_id"]

    print(f"Unique users: {users.nunique():,}")
    print(f"Min user_id:  {users.min()}")
    print(f"Max user_id:  {users.max()}")

    unique = np.sort(users.unique())

    expected = np.arange(
        unique.min(),
        unique.max() + 1,
    )

    missing = np.setdiff1d(expected, unique)

    print(f"Missing IDs inside range: {len(missing):,}")

    if len(missing) == 0:
        print("User ID continuity: PASS")
    else:
        print("User ID continuity: INFO")


def audit_history_length(df):
    print("\n" + "=" * 80)
    print("HISTORY LENGTH / COMPUTATION AUDIT")
    print("=" * 80)

    for w in WINDOWS:
        col = (
            "history_full"
            if w == "full"
            else f"history_{w}"
        )

        lengths = df[col].map(
            lambda x: len(to_array(x))
        )

        print(
            f"W={str(w):<4} "
            f"mean={lengths.mean():>8.2f} "
            f"median={lengths.median():>8.2f} "
            f"max={lengths.max():>8}"
        )


def main():
    print("=" * 80)
    print("AUDIT KUAIRAND VOCABULARY")
    print("=" * 80)

    print(f"\nLoading:")
    print(f"  {HISTORY_PATH}")

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    df = pd.read_parquet(HISTORY_PATH)

    print(f"\nEvaluation events: {len(df):,}")

    audit_video_vocabulary(df)
    audit_event_level_oov(df)
    audit_history_ids(df)
    audit_user_vocabulary(df)
    audit_history_length(df)

    print("\n" + "=" * 80)
    print("VOCABULARY AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()