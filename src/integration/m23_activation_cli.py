"""Local M2.3D activation CLI. Secret is read from environment, never arguments."""
import argparse
import json
import os
from pathlib import Path

from src.integration.m23_activation import activate, open_active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("activate", "verify"))
    parser.add_argument("--root", type=Path, default=Path("data/processed/m23_snapshots"))
    parser.add_argument("--snapshot-id")
    args = parser.parse_args()
    raw = os.environ.get("M23_HMAC_KEY_HEX", "")
    try:
        key = bytes.fromhex(raw)
    except ValueError as exc:
        raise SystemExit("M23_HMAC_KEY_HEX must be valid hex") from exc
    if len(key) < 32:
        raise SystemExit("Set M23_HMAC_KEY_HEX to at least 32 random bytes (hex)")
    if args.action == "activate":
        if not args.snapshot_id:
            parser.error("--snapshot-id required for activate")
        sid = activate(args.root, args.snapshot_id, key)
        print(json.dumps({"status": "ACTIVATED", "snapshot_id": sid}))
    else:
        with open_active(args.root, key) as store:
            print(json.dumps({"status": "VERIFIED", "snapshot_id": store.manifest["snapshot_id"],
                              "records": store.manifest["records"], "shards": store.shards}))


if __name__ == "__main__":
    main()
