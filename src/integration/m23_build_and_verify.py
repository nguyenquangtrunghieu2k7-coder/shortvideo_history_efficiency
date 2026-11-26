"""Build and validate an M2.3A persistent snapshot from frozen KuaiRand test histories."""
import argparse
import json
import sys
from pathlib import Path

from src.integration.m23_persistent_store import HistoryStore, build_snapshot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/processed/m23_snapshots"))
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--max-requests", type=int, default=0)
    args = p.parse_args()
    import train_baseline as baseline
    _, vocab = baseline.load_video_vocab()
    data = baseline.KuaiRandDataset("test", "Full", vocab)
    records = [(str(key), [int(x) for x in history])
               for key, history in zip(data.evaluation_ids, data.history_ids)]
    if args.max_requests:
        records = records[:args.max_requests]
    sid = build_snapshot(args.root, records, shards=args.shards)
    with HistoryStore(args.root, sid) as store:
        for key, history in records:
            for mode in ("indexed_exact", "sharded_exact", "sharded_bloom"):
                if store.get(key, mode=mode) != history:
                    raise AssertionError(f"Mismatch for key={key}, mode={mode}")
        if store.get("__missing__") is not None:
            raise AssertionError("Missing key must return None")
    print(json.dumps({"status": "PASS", "snapshot_id": sid,
                      "snapshot_path": str(args.root / sid),
                      "records": len(records), "shards": args.shards,
                      "retrieval_modes_verified": ["indexed_exact", "sharded_exact", "sharded_bloom"]}, indent=2))


if __name__ == "__main__":
    main()
