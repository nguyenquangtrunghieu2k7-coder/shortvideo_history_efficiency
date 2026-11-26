"""M2.3G regression tests: pinned activation, fail-closed startup, concurrency.

All data and credentials are synthetic; no real histories or secrets are logged.
"""
import concurrent.futures
import http.client
import json
import secrets
import tempfile
import threading
import unittest
from pathlib import Path

from src.integration.m23_activation import activate
from src.integration.m23_local_api import create_server
from src.integration.m23_persistent_store import build_snapshot


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.secret = secrets.token_bytes(32)
        self.token = secrets.token_hex(32)
        self.old_id = build_snapshot(self.root, [("known", [1, 2, 3])], shards=2)
        activate(self.root, self.old_id, self.secret)
        self.server = None
        self.thread = None

    def start_server(self):
        self.server = create_server(self.root, self.secret, self.token, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            conn.request("GET", path, headers={"Authorization": "Bearer " + self.token})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read())
        finally:
            conn.close()

    def test_pinned_snapshot_survives_pointer_activation(self):
        self.start_server()
        self.assertEqual(self.get("/history?key=known")[1]["history"], [1, 2, 3])
        new_id = build_snapshot(self.root, [("known", [7, 8]), ("new", [9])], shards=2)
        self.assertNotEqual(new_id, self.old_id)
        activate(self.root, new_id, self.secret)
        # A running server must not silently change its data source.
        self.assertEqual(self.get("/ready")[1]["snapshot_id"], self.old_id)
        self.assertEqual(self.get("/history?key=known")[1]["history"], [1, 2, 3])
        self.assertEqual(self.get("/history?key=new")[0], 404)

    def test_restart_picks_up_new_activation(self):
        new_id = build_snapshot(self.root, [("known", [7, 8])], shards=2)
        activate(self.root, new_id, self.secret)
        self.start_server()
        self.assertEqual(self.get("/ready")[1]["snapshot_id"], new_id)
        self.assertEqual(self.get("/history?key=known")[1]["history"], [7, 8])

    def test_invalid_signature_fails_closed_at_startup(self):
        pointer = self.root / "ACTIVE.json"
        record = json.loads(pointer.read_text())
        record["signature"] = "0" * 64
        pointer.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            create_server(self.root, self.secret, self.token, port=0)

    def test_modified_manifest_fails_closed_at_startup(self):
        manifest = self.root / self.old_id / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            create_server(self.root, self.secret, self.token, port=0)

    def test_eight_worker_burst(self):
        self.start_server()
        def request(i):
            if i % 3 == 0:
                status, body = self.get("/history?key=missing")
                return status == 404 and body.get("error") == "history_not_found"
            if i % 3 == 1:
                status, body = self.get("/history?key=known&k=0")
                return status == 200 and body.get("history") == []
            status, body = self.get("/history?key=known&k=Full")
            return status == 200 and body.get("history") == [1, 2, 3]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(request, range(240)))
        self.assertTrue(all(results), "Concurrent API burst failed")


if __name__ == "__main__":
    unittest.main()
