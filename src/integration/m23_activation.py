"""M2.3D: authenticated atomic activation of validated immutable snapshots.

Standard-library HMAC. Keys are supplied by the caller, never stored in the repo.
Threat model: detects accidental/unauthorized on-disk edits without the secret;
does not defend against an attacker who controls the runtime or secret.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from pathlib import Path

from src.integration.m23_persistent_store import HistoryStore

_ID = re.compile(r"^[0-9a-f]{32}$")
_POINTER = "ACTIVE.json"


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _signature(secret, payload):
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("HMAC secret must be at least 32 bytes")
    return hmac.new(secret, _canonical(payload), hashlib.sha256).hexdigest()


def _manifest_digest(path):
    # Digest raw bytes: even harmless manifest edits require re-activation.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_id(snapshot_id):
    if not isinstance(snapshot_id, str) or not _ID.fullmatch(snapshot_id):
        raise ValueError("Invalid snapshot ID")


def activate(root, snapshot_id, secret):
    """Verify full snapshot before atomic pointer replacement.

    The active pointer contains a signed manifest digest, not the secret.
    All activation writes are local filesystem operations.
    """
    _validate_id(snapshot_id)
    root = Path(root)
    with HistoryStore(root, snapshot_id, verify=True):
        digest = _manifest_digest(root / snapshot_id / "manifest.json")
    payload = {"format": 1, "snapshot_id": snapshot_id, "manifest_sha256": digest}
    record = dict(payload, signature=_signature(secret, payload))
    root.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".active-", suffix=".tmp",
                                         dir=root, delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(_canonical(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, root / _POINTER)
        # Best-effort directory durability on POSIX.
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return snapshot_id


def open_active(root, secret):
    """Open a pinned active snapshot; reject tampering before serving requests."""
    root = Path(root)
    record = json.loads((root / _POINTER).read_text(encoding="utf-8"))
    if set(record) != {"format", "snapshot_id", "manifest_sha256", "signature"}:
        raise ValueError("Invalid active pointer")
    if record["format"] != 1:
        raise ValueError("Unsupported active pointer format")
    snapshot_id = record["snapshot_id"]
    _validate_id(snapshot_id)
    payload = {k: record[k] for k in ("format", "snapshot_id", "manifest_sha256")}
    expected = _signature(secret, payload)
    if not hmac.compare_digest(record["signature"], expected):
        raise ValueError("Active pointer authentication failed")
    digest = _manifest_digest(root / snapshot_id / "manifest.json")
    if not hmac.compare_digest(record["manifest_sha256"], digest):
        raise ValueError("Active snapshot manifest changed")
    return HistoryStore(root, snapshot_id, verify=True)
