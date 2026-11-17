import json
import secrets
import tempfile
import unittest
from pathlib import Path

from src.integration.m23_activation import activate, open_active
from src.integration.m23_persistent_store import build_snapshot


class ActivationTests(unittest.TestCase):
    def test_activate_restart_and_roll_forward(self):
        with tempfile.TemporaryDirectory() as root:
            key = secrets.token_bytes(32)
            first = build_snapshot(root, [("a", [1])])
            activate(root, first, key)
            with open_active(root, key) as store:
                self.assertEqual(store.get("a"), [1])
            second = build_snapshot(root, [("a", [2])])
            activate(root, second, key)
            with open_active(root, key) as store:
                self.assertEqual(store.get("a"), [2])
            # Existing pinned readers can still open old snapshot.
            from src.integration.m23_persistent_store import HistoryStore
            with HistoryStore(root, first) as store:
                self.assertEqual(store.get("a"), [1])

    def test_wrong_secret_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [("a", [1])])
            activate(root, sid, b"a" * 32)
            with self.assertRaises(ValueError):
                open_active(root, b"b" * 32)

    def test_pointer_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [("a", [1])])
            activate(root, sid, b"a" * 32)
            path = Path(root) / "ACTIVE.json"
            payload = json.loads(path.read_text())
            payload["manifest_sha256"] = "0" * 64
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                open_active(root, b"a" * 32)

    def test_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [("a", [1])])
            activate(root, sid, b"a" * 32)
            manifest = Path(root) / sid / "manifest.json"
            manifest.write_text(manifest.read_text() + " ")
            with self.assertRaises(ValueError):
                open_active(root, b"a" * 32)

    def test_failed_activation_preserves_previous_pointer(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [("a", [1])])
            activate(root, sid, b"a" * 32)
            pointer = (Path(root) / "ACTIVE.json").read_bytes()
            with self.assertRaises(FileNotFoundError):
                activate(root, "f" * 32, b"a" * 32)
            self.assertEqual((Path(root) / "ACTIVE.json").read_bytes(), pointer)


if __name__ == "__main__":
    unittest.main()
