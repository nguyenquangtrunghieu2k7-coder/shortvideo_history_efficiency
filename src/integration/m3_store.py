"""M3 bounded history store. Builds a new immutable signed index; never mutates M2."""
from contextlib import closing
from pathlib import Path
import hashlib
import hmac
import json
import os
import sqlite3
import tempfile
import shutil
import uuid
import numpy as np
from src.bloom_pipeline.bloom_filter import BloomFilter
from src.integration.m23_persistent_store import _sha

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":")).encode()

def route(key,n):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")%n

def quantize_centroid(v):
    v=np.asarray(v,dtype=np.float32)
    return np.rint(127*v/max(float(np.linalg.norm(v)),1e-8)).clip(-127,127).astype(np.int8)

def build(root,records,embedding,secret,shards=8,block=32,metadata=None,compact=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    sid=uuid.uuid4().hex
    work=Path(tempfile.mkdtemp(prefix=".build-",dir=root))
    conns=[];keys=[[] for _ in range(shards)]
    try:
        for i in range(shards):
            c=sqlite3.connect(work/f"shard_{i}.sqlite");conns.append(c)
            c.executescript("""CREATE TABLE meta(key TEXT PRIMARY KEY,n INTEGER NOT NULL);
            CREATE TABLE tokens(key TEXT NOT NULL,pos INTEGER NOT NULL,item INTEGER NOT NULL,PRIMARY KEY(key,pos)) WITHOUT ROWID;
            CREATE TABLE blocks(key TEXT NOT NULL,b INTEGER NOT NULL,lo INTEGER NOT NULL,hi INTEGER NOT NULL,centroid BLOB NOT NULL,PRIMARY KEY(key,b)) WITHOUT ROWID;""")
        for key,history in records:
            key=str(key);h=list(map(int,history))
            if len(h)>2048 or any(v<=0 or v>=len(embedding) for v in h):
                raise ValueError("Invalid history")
            i=route(key,shards);c=conns[i];keys[i].append(key)
            c.execute("INSERT INTO meta VALUES (?,?)",(key,len(h)))
            c.executemany("INSERT INTO tokens VALUES (?,?,?)",((key,j,v) for j,v in enumerate(h)))
            for j in range(0,len(h),block):
                hi=min(j+block,len(h))
                cent=np.asarray(embedding[h[j:hi]].mean(0),dtype="<f4")
                c.execute("INSERT INTO blocks VALUES (?,?,?,?,?)",(key,j//block,j,hi,(quantize_centroid(cent) if compact else cent).tobytes()))
        for c in conns:c.commit();c.close()
        conns=[]
        manifest={"id":sid,"shards":shards,"block":block,"records":sum(map(len,keys)),
                  "dim":int(embedding.shape[1]),"centroid_encoding":"int8_unit" if compact else "float32","files":{},"bloom":[],"metadata":metadata or {}}
        for i,k in enumerate(keys):
            bloom=BloomFilter(max(64,len(k)*12),7)
            for v in k:bloom.add(v)
            f=work/f"shard_{i}.bloom";f.write_bytes(bloom.bit_array)
            manifest["bloom"].append({"m":bloom.m,"k":bloom.k,"sha256":_sha(f)})
            name=f"shard_{i}.sqlite";manifest["files"][name]=_sha(work/name)
        payload={"manifest":manifest,"signature":hmac.new(secret,canonical(manifest),hashlib.sha256).hexdigest()}
        (work/"manifest.json").write_bytes(canonical(payload))
        os.replace(work,root/sid)
        return sid
    finally:
        for c in conns:c.close()
        shutil.rmtree(work,ignore_errors=True)

class Store:
    def __init__(self,root,sid,secret):
        self.path=Path(root)/sid
        p=json.loads((self.path/"manifest.json").read_text())
        m=p["manifest"]
        if not hmac.compare_digest(p["signature"],hmac.new(secret,canonical(m),hashlib.sha256).hexdigest()) or m["id"]!=sid:
            raise ValueError("Snapshot signature mismatch")
        self.manifest=m;self.blooms=[]
        for i in range(m["shards"]):
            db=self.path/f"shard_{i}.sqlite"
            if _sha(db)!=m["files"][db.name]:raise ValueError("Corrupt database")
            f=self.path/f"shard_{i}.bloom";b=m["bloom"][i]
            bloom=None
            if f.is_file() and _sha(f)==b["sha256"]:
                bloom=BloomFilter(b["m"],b["k"]);bits=f.read_bytes()
                if len(bits)==len(bloom._bits):
                    bloom._bits[:]=bits
                    with self.connect(i) as c:
                        if any(not bloom.might_contain(k) for k, in c.execute("SELECT key FROM meta")):bloom=None
                else:bloom=None
            self.blooms.append(bloom)
    def connect(self,i):
        return closing(sqlite3.connect((self.path/f"shard_{i}.sqlite").resolve().as_uri()+"?mode=ro",uri=True))
    def lookup(self,key,op,lo=0,hi=0,mode="indexed_bloom",ranges=None):
        if mode not in ("indexed_exact","indexed_bloom","sharded_exact","sharded_bloom"):raise ValueError("Bad mode")
        candidates=[route(key,len(self.blooms))] if mode.startswith("indexed") else range(len(self.blooms))
        stats=dict(bloom_probes=0,shards_skipped=0,sqlite_lookups=0,false_positives=0,bloom_fallbacks=0,token_rows=0,summary_rows=0,logical_payload_bytes=0)
        result=None
        for i in candidates:
            b=self.blooms[i]
            if mode.endswith("bloom"):
                if b is None:stats["bloom_fallbacks"]+=1
                else:
                    stats["bloom_probes"]+=1
                    if not b.might_contain(key):stats["shards_skipped"]+=1;continue
            stats["sqlite_lookups"]+=1
            with self.connect(i) as c:
                row=c.execute("SELECT n FROM meta WHERE key=?",(key,)).fetchone()
                if row is None:
                    if mode.endswith("bloom") and b is not None:stats["false_positives"]+=1
                    continue
                if result is not None:raise ValueError("Duplicate key")
                n=row[0]
                if op=="meta":result={"n":n}
                elif op=="summaries":
                    rows=c.execute("SELECT lo,hi,centroid FROM blocks WHERE key=? ORDER BY b",(key,)).fetchall()
                    result={"n":n,"blocks":[[a,b,np.frombuffer(v,dtype=np.int8 if self.manifest.get("centroid_encoding")=="int8_unit" else "<f4").astype(np.float32).tolist()] for a,b,v in rows]}
                    stats["summary_rows"]+=len(rows);stats["logical_payload_bytes"]+=sum(16+len(v) for a,b,v in rows)
                elif op=="tokens":
                    ranges=ranges if ranges is not None else [(lo,hi)]
                    if len(ranges)>64 or any(not 0<=a<=b<=n for a,b in ranges):raise ValueError("Invalid bounds")
                    rows=[]
                    for a,b in ranges:
                        rows.extend(c.execute("SELECT pos,item FROM tokens WHERE key=? AND pos>=? AND pos<? ORDER BY pos",(key,a,b)).fetchall())
                    if len(rows)!=len({p for p,v in rows}):raise ValueError("Overlapping ranges")
                    result={"n":n,"tokens":sorted(rows)}
                    stats["token_rows"]+=len(rows);stats["logical_payload_bytes"]+=16*len(rows)
                else:raise ValueError("Unknown operation")
        return result,stats
