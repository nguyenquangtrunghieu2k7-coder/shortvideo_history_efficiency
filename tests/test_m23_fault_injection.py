"""M2.3B fault-injection gates. Uses only generated synthetic histories."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.integration.m23_persistent_store import HistoryStore, build_snapshot


class FaultInjectionTests(unittest.TestCase):
    def test_stale_bloom_with_recomputed_checksum_falls_back(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [(str(i), [i]) for i in range(100)], shards=2)
            path = Path(root) / sid
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for i in range(2):
                bloom_path = path / f"shard_{i}.bloom"
                bloom_path.write_bytes(bytes(bloom_path.stat().st_size))
                manifest["bloom"][str(i)]["sha256"] = hashlib.sha256(bloom_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with HistoryStore(root, sid) as store:
                self.assertTrue(all(b is None for b in store.blooms))
                for i in range(100):
                    self.assertEqual(store.get(str(i), mode="sharded_bloom"), [i])

    def test_manifest_missing_shard_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [("a", [1])], shards=2)
            path = Path(root) / sid / "manifest.json"
            manifest = json.loads(path.read_text())
            del manifest["files"]["shard_1.sqlite"]
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                HistoryStore(root, sid)

    def test_manifest_wrong_total_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [("a", [1])], shards=2)
            path = Path(root) / sid / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["records"] += 1
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                HistoryStore(root, sid)

    def test_failed_build_leaves_no_published_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                build_snapshot(root, [("a", [1]), ("a", [2])])
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_existing_snapshot_survives_next_build(self):
        with tempfile.TemporaryDirectory() as root:
            first = build_snapshot(root, [("a", [1])])
            second = build_snapshot(root, [("a", [2])])
            with HistoryStore(root, first) as old, HistoryStore(root, second) as new:
                self.assertEqual(old.get("a"), [1])
                self.assertEqual(new.get("a"), [2])

    def test_concurrent_readers_use_separate_connections(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [(str(i), [i, i + 1]) for i in range(200)])
            def worker(i):
                with HistoryStore(root, sid) as store:
                    return store.get(str(i), mode="sharded_bloom")
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(worker, range(200)))
            self.assertEqual(results, [[i, i + 1] for i in range(200)])

    def test_corrupt_sqlite_fails_before_serving(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [("a", [1])])
            db = Path(root) / sid / "shard_0.sqlite"
            db.write_bytes(b"not a sqlite database")
            with self.assertRaises(ValueError):
                HistoryStore(root, sid)


if __name__ == "__main__":
    unittest.main()
