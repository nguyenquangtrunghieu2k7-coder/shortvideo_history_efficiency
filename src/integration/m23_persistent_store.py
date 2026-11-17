"""M2.3A: independently implemented persistent, integrity-checked history store.

Immutable snapshots, SQLite exact retrieval, optional Bloom shard pruning.
No third-party BLOOM-FaaS code, infrastructure settings, or credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from pathlib import Path

from src.bloom_pipeline.bloom_filter import BloomFilter

SCHEMA_VERSION = 1


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_snapshot(root, records, *, shards=8, bits_per_key=10, hashes=7):
    """Build a new immutable snapshot. records is iterable of (str key, list[int]).

    Snapshot is published only after DBs and manifest are finalized.
    Returns snapshot ID. Existing snapshots are never modified.
    """
    if min(shards, bits_per_key, hashes) < 1:
        raise ValueError("shards/bits_per_key/hashes must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_id = uuid.uuid4().hex
    work = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
    conns = []
    try:
        for i in range(shards):
            conn = sqlite3.connect(work / f"shard_{i}.sqlite")
            conn.execute("CREATE TABLE histories (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conns.append(conn)
        seen = set()
        buckets = [[] for _ in range(shards)]
        for key, history in records:
            if not isinstance(key, str) or not key:
                raise ValueError("History key must be a nonempty string")
            if key in seen:
                raise ValueError(f"Duplicate history key: {key}")
            if not isinstance(history, (list, tuple)) or not all(isinstance(x, int) for x in history):
                raise ValueError(f"History must be a list of integer model IDs: {key}")
            seen.add(key)
            # Hash routing is deterministic and is the primary retrieval path.
            shard = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % shards
            buckets[shard].append((key, json.dumps(list(history), separators=(",", ":"))))
        for i, conn in enumerate(conns):
            conn.executemany("INSERT INTO histories VALUES (?, ?)", buckets[i])
            conn.commit()
            conn.close()
        conns = []
        manifest = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id,
                    "shards": shards, "records": len(seen), "bits_per_key": bits_per_key,
                    "hashes": hashes, "files": {}, "bloom": {}}
        for i in range(shards):
            db_path = work / f"shard_{i}.sqlite"
            manifest["files"][db_path.name] = _sha(db_path)
            bloom = BloomFilter(max(64, len(buckets[i]) * bits_per_key), hashes)
            for key, _ in buckets[i]:
                bloom.add(key)
            bloom_name = f"shard_{i}.bloom"
            (work / bloom_name).write_bytes(bloom._bits)
            manifest["bloom"][str(i)] = {"file": bloom_name, "sha256": _sha(work / bloom_name),
                                          "m": bloom.m, "k": bloom.k, "keys": len(buckets[i])}
        _atomic_json(work / "manifest.json", manifest)
        final = root / snapshot_id
        os.replace(work, final)
        return snapshot_id
    except Exception:
        for conn in conns:
            conn.close()
        shutil.rmtree(work, ignore_errors=True)
        raise


class HistoryStore:
    """Read-only snapshot. Corrupt SQLite fails closed; invalid Bloom falls back to exact."""

    def __init__(self, root, snapshot_id, *, verify=True):
        self.path = Path(root) / snapshot_id
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if m["schema_version"] != SCHEMA_VERSION or m["snapshot_id"] != snapshot_id:
            raise ValueError("Snapshot contract mismatch")
        self.shards = m["shards"]
        if not isinstance(self.shards, int) or self.shards < 1:
            raise ValueError("Invalid shard count")
        if set(m["files"]) != {f"shard_{i}.sqlite" for i in range(self.shards)}:
            raise ValueError("Manifest shard inventory mismatch")
        if set(m["bloom"]) != {str(i) for i in range(self.shards)}:
            raise ValueError("Manifest Bloom inventory mismatch")
        if not isinstance(m["records"], int) or m["records"] < 0:
            raise ValueError("Invalid record count")
        if not isinstance(m["bits_per_key"], int) or m["bits_per_key"] < 1:
            raise ValueError("Invalid bits_per_key")
        if not isinstance(m["hashes"], int) or m["hashes"] < 1:
            raise ValueError("Invalid hashes")
        if verify:
            for name, expected in m["files"].items():
                if _sha(self.path / name) != expected:
                    raise ValueError(f"Corrupt database shard: {name}")
        self.connections = []
        self.blooms = []
        try:
            for i in range(self.shards):
                db = self.path / f"shard_{i}.sqlite"
                # SQLite URI mode=ro prevents accidental creation or modification.
                conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
                self.connections.append(conn)
                meta = m["bloom"][str(i)]
                if (meta["file"] != f"shard_{i}.bloom" or
                        meta["k"] != m["hashes"] or
                        not isinstance(meta["keys"], int) or meta["keys"] < 0 or
                        meta["m"] != max(64, meta["keys"] * m["bits_per_key"])):
                    raise ValueError(f"Invalid Bloom metadata in shard {i}")
                if verify:
                    count = 0
                    for (stored_key,) in conn.execute("SELECT key FROM histories"):
                        count += 1
                        route = int.from_bytes(hashlib.sha256(stored_key.encode()).digest()[:8], "big") % self.shards
                        if route != i:
                            raise ValueError(f"Wrong shard routing: {stored_key}")
                    if count != meta["keys"]:
                        raise ValueError(f"Shard key count mismatch: {i}")
                bloom_path = self.path / meta["file"]
                if not bloom_path.is_file() or _sha(bloom_path) != meta["sha256"]:
                    self.blooms.append(None)
                    continue
                bloom = BloomFilter(meta["m"], meta["k"])
                bits = bloom_path.read_bytes()
                if len(bits) != len(bloom._bits):
                    self.blooms.append(None)
                    continue
                bloom._bits[:] = bits
                # Check every SQLite key: a stale Bloom with a valid checksum
                # must never suppress a real history lookup.
                if verify and any(
                    not bloom.might_contain(stored_key)
                    for (stored_key,) in conn.execute("SELECT key FROM histories")
                ):
                    self.blooms.append(None)
                else:
                    self.blooms.append(bloom)
            if verify and sum(m["bloom"][str(i)]["keys"] for i in range(self.shards)) != m["records"]:
                raise ValueError("Snapshot total key count mismatch")
        except Exception:
            self.close()
            raise

    def close(self):
        for conn in self.connections:
            conn.close()
        self.connections = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get(self, key, *, mode="indexed_exact"):
        if mode not in ("indexed_exact", "sharded_exact", "sharded_bloom"):
            raise ValueError(f"Unknown retrieval mode: {mode}")
        shard = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % self.shards
        candidates = [shard] if mode == "indexed_exact" else range(self.shards)
        found = None
        for i in candidates:
            if mode == "sharded_bloom" and self.blooms[i] is not None and not self.blooms[i].might_contain(key):
                continue
            row = self.connections[i].execute(
                "SELECT payload FROM histories WHERE key=?", (key,)
            ).fetchone()
            if row is not None:
                if found is not None:
                    raise ValueError(f"Duplicate key across shards: {key}")
                found = json.loads(row[0])
        return found

    def get_last_k(self, key, k, *, mode="indexed_exact"):
        history = self.get(key, mode=mode)
        if history is None:
            return None
        if k == "Full":
            return history
        if not isinstance(k, int) or k < 0:
            raise ValueError("K must be nonnegative integer or 'Full'")
        return history[-k:] if k else []
