import http.client
import json
import secrets
import tempfile
import threading
import unittest
from urllib.parse import quote

from src.integration.m23_activation import activate
from src.integration.m23_local_api import create_server
from src.integration.m23_persistent_store import build_snapshot


class LocalAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.secret = secrets.token_bytes(32)
        self.token = secrets.token_hex(32)
        sid = build_snapshot(self.tmp.name, [("known", [1, 2, 3]), ("empty", [])], shards=2)
        activate(self.tmp.name, sid, self.secret)
        self.server = create_server(self.tmp.name, self.secret, self.token, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def request(self, path, token=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        try:
            conn.request("GET", path, headers=headers)
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read())
        finally:
            conn.close()

    def test_auth_and_health(self):
        self.assertEqual(self.request("/health")[0], 401)
        self.assertEqual(self.request("/health", self.token)[0], 200)
        self.assertEqual(self.request("/ready", self.token)[0], 200)

    def test_history_full_and_k(self):
        self.assertEqual(self.request("/history?key=known", self.token)[1]["history"], [1, 2, 3])
        self.assertEqual(self.request("/history?key=known&k=2", self.token)[1]["history"], [2, 3])
        self.assertEqual(self.request("/history?key=known&k=0", self.token)[1]["history"], [])
        self.assertEqual(self.request("/history?key=empty", self.token)[1]["history"], [])

    def test_missing_and_invalid(self):
        self.assertEqual(self.request("/history?key=missing", self.token)[0], 404)
        self.assertEqual(self.request("/history?key=known&k=-1", self.token)[0], 400)
        self.assertEqual(self.request("/history?key=known&key=empty", self.token)[0], 400)

    def test_loopback_only(self):
        with self.assertRaises(ValueError):
            create_server(self.tmp.name, self.secret, self.token, host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
