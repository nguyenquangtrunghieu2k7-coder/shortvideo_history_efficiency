"""Bloom utility study on the actual verified persistent store.
Identical queries, controlled hit ratios, exact equivalence; network cost is
an explicit serial-lookup sensitivity model, not a cloud measurement."""
from contextlib import closing
import random
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.integration.m3_store import Store

def run(root,sid,secret,output,seed=42,queries=400):
    store=Store(root,sid,secret);keys=[]
    for i in range(store.manifest["shards"]):
        with store.connect(i) as c:keys.extend(k for k, in c.execute("SELECT key FROM meta ORDER BY key"))
    if not keys:raise ValueError("Empty store")
    rng=random.Random(seed);existing=set(keys);rows=[]
    modes=["indexed_exact","indexed_bloom","sharded_exact","sharded_bloom"]
    for miss in (0.,.1,.5,.9,1.):
        n=int(queries*miss)
        work=[(rng.choice(keys),True) for _ in range(queries-n)]
        work.extend((f"__missing_{seed}_{i}",False) for i in range(n))
        if any(k in existing for k,hit in work if not hit):raise ValueError("Missing-key collision")
        rng.shuffle(work)
        ref=[store.lookup(k,"meta",mode="indexed_exact")[0] for k,_ in work]
        for mode in modes:
            stats={};times=[];lookups=[]
            for (key,hit),expected in zip(work,ref):
                t=time.perf_counter()
                value,s=store.lookup(key,"meta",mode=mode)
                times.append((time.perf_counter()-t)*1000);lookups.append(s["sqlite_lookups"])
                if value!=expected or (value is not None)!=hit:raise AssertionError("Bloom parity failed")
                for k,v in s.items():stats[k]=stats.get(k,0)+v
            row=dict(missing_fraction=miss,mode=mode,requests=queries,errors=0,
                mean_local_ms=float(np.mean(times)),p95_local_ms=float(np.quantile(times,.95)),
                exact_lookups=stats["sqlite_lookups"],mean_exact_lookups=float(np.mean(lookups)),
                bloom_probes=stats["bloom_probes"],shards_skipped=stats["shards_skipped"],
                false_positives=stats["false_positives"],fallbacks=stats["bloom_fallbacks"],
                filter_bytes=sum(len(b._bits) for b in store.blooms if b is not None))
            for rtt in (.5,2.,10.):
                row[f"serial_rtt_{rtt:g}ms_mean_estimate"]=float(np.mean(np.asarray(times)+rtt*np.asarray(lookups)))
            rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(output,index=False)
    return rows
