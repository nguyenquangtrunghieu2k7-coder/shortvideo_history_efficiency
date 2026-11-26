import tempfile
import unittest
from pathlib import Path

from src.integration.m23_persistent_store import HistoryStore, build_snapshot


class PersistentStoreTests(unittest.TestCase):
    def test_restart_modes_and_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            sid = build_snapshot(tmp, [("a", [1, 2, 3]), ("b", []), ("c", [7])], shards=3)
            for _ in range(2):
                with HistoryStore(tmp, sid) as store:
                    for mode in ("indexed_exact", "sharded_exact", "sharded_bloom"):
                        self.assertEqual(store.get("a", mode=mode), [1, 2, 3])
                        self.assertEqual(store.get("b", mode=mode), [])
                        self.assertIsNone(store.get("missing", mode=mode))
                    self.assertEqual(store.get_last_k("a", 2), [2, 3])
                    self.assertEqual(store.get_last_k("a", 0), [])
                    self.assertEqual(store.get_last_k("a", "Full"), [1, 2, 3])

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_snapshot(tmp, [("a", [1]), ("a", [2])])

    def test_bloom_corruption_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            sid = build_snapshot(tmp, [("a", [1, 2])])
            (Path(tmp) / sid / "shard_0.bloom").write_bytes(b"corrupt")
            with HistoryStore(tmp, sid) as store:
                self.assertEqual(store.get("a", mode="sharded_bloom"), [1, 2])

    def test_db_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sid = build_snapshot(tmp, [("a", [1])])
            (Path(tmp) / sid / "shard_0.sqlite").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                HistoryStore(tmp, sid)


if __name__ == "__main__":
    unittest.main()
