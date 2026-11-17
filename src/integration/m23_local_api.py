"""M2.3E loopback-only authenticated HTTP history API (standard library).

Deliberately local-only: not a public network production deployment.
Each request opens a verified snapshot; correctness over throughput.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from src.integration.m23_activation import open_active

MAX_URL = 4096
MAX_KEY = 256


def make_handler(root, hmac_key, api_token):
    if len(api_token) < 32:
        raise ValueError("API token must have at least 32 characters")

    class Handler(BaseHTTPRequestHandler):
        server_version = "HistoryAPI/0.1"
        sys_version = ""

        def _reply(self, status, data):
            body = json.dumps(data, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if len(self.path) > MAX_URL:
                return self._reply(414, {"error": "uri_too_long"})
            token = self.headers.get("Authorization", "")
            if not hmac.compare_digest(token, "Bearer " + api_token):
                return self._reply(401, {"error": "unauthorized"})
            parsed = urlsplit(self.path)
            if parsed.path == "/health":
                return self._reply(200, {"status": "alive"})
            if parsed.path == "/ready":
                try:
                    with open_active(root, hmac_key) as store:
                        return self._reply(200, {"status": "ready", "snapshot_id": store.manifest["snapshot_id"]})
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    return self._reply(503, {"status": "unavailable"})
            if parsed.path != "/history":
                return self._reply(404, {"error": "not_found"})
            params = parse_qs(parsed.query, keep_blank_values=True)
            if set(params) - {"key", "k"} or len(params.get("key", [])) != 1 or len(params.get("k", [])) > 1:
                return self._reply(400, {"error": "invalid_query"})
            key = params["key"][0]
            if not key or len(key) > MAX_KEY:
                return self._reply(400, {"error": "invalid_key"})
            raw_k = params.get("k", ["Full"])[0]
            if raw_k == "Full":
                k = "Full"
            elif raw_k.isascii() and raw_k.isdecimal() and len(raw_k) <= 9:
                k = int(raw_k)
            else:
                return self._reply(400, {"error": "invalid_k"})
            try:
                with open_active(root, hmac_key) as store:
                    history = store.get_last_k(key, k, mode="indexed_exact")
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                return self._reply(503, {"error": "snapshot_unavailable"})
            if history is None:
                return self._reply(404, {"error": "history_not_found"})
            return self._reply(200, {"key": key, "k": raw_k, "history": history})

        def do_POST(self):
            return self._reply(405, {"error": "method_not_allowed"})

        def log_message(self, fmt, *args):
            # Avoid logging request URLs (history keys) or Authorization headers.
            pass

    return Handler


def create_server(root, hmac_key, api_token, host="127.0.0.1", port=8765):
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("M2.3E preview only binds to loopback")
    handler = make_handler(Path(root), hmac_key, api_token)
    return ThreadingHTTPServer((host, port), handler)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/processed/m23_snapshots"))
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    try:
        secret = bytes.fromhex(os.environ.get("M23_HMAC_KEY_HEX", ""))
    except ValueError:
        raise SystemExit("Invalid M23_HMAC_KEY_HEX")
    if len(secret) < 32:
        raise SystemExit("M23_HMAC_KEY_HEX must contain at least 32 bytes")
    token = os.environ.get("M23_API_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("M23_API_TOKEN must contain at least 32 characters")
    server = create_server(args.root, secret, token, port=args.port)
    print(f"Local history API listening on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
