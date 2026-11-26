"""M3: paired real-log refresh workloads through the same recommendation HTTP API.

No retraining or selection on held-out labels. Refresh records are deduplicated by
(user, refresh timestamp); rolling construction advances each user's event cursor.
Immutable signed snapshots are built offline, not mutated while serving.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import random
import tempfile
import threading
from time import perf_counter
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from src.integration.m3_product import (
    Engine, credentials, events_only, gauc, load_model, make_server, ROOT, baseline)
from src.integration.m3_store import build
from src.integration.m23_persistent_store import _sha
from src.integration.scheduled_window_ablation import STANDARD, DAY, HOUR, last_refresh
from src.integration.scheduled_window_controlled import load_raw, paired_user_cluster_ci

CADENCES=(0,24,12,6,3)
POLICIES=("full","fixed50","semantic50","adaptive","adaptive_exact")

def refresh_time(t,hours,anchor,offset):
    return int(t) if hours==0 else last_refresh(int(t),hours,anchor,offset)

def make_jobs(events,hours,anchor,offset):
    jobs=defaultdict(set);keys=[]
    for row in events.itertuples(index=False):
        u=int(row.user_id);t=int(row.time_ms);r=refresh_time(t,hours,anchor,offset)
        if r>t:raise AssertionError("Future refresh")
        jobs[u].add(r);keys.append(f"{u}:{r}")
    return jobs,keys

def common_cohort(events,raw,vocab,anchor,offset):
    keep=[];excluded=Counter()
    for row in events.itertuples(index=False):
        times,ids=raw.get(int(row.user_id),([],[]))
        valid=True
        for h in CADENCES:
            r=refresh_time(row.time_ms,h,anchor,offset)
            lo=bisect_left(times,r-DAY);hi=bisect_left(times,r)
            if any(v not in vocab for v in ids[max(lo,hi-2048):hi]):
                excluded[str(h)]+=1;valid=False
        keep.append(valid)
    return np.asarray(keep),dict(excluded)

def rolling_records(jobs,raw,vocab,stats):
    """Advance event cursor and expire old records; verify against independent bisect."""
    for user,refreshes in sorted(jobs.items()):
        times,ids=raw.get(user,([],[]));q=deque();cursor=0
        previous=None
        for refresh in sorted(refreshes):
            if previous is None:cursor=bisect_left(times,refresh-DAY)
            while cursor<len(times) and times[cursor]<refresh:
                if times[cursor]>=refresh-DAY:
                    q.append((times[cursor],ids[cursor]));stats["inserted_events"]+=1
                else:stats["skipped_expired_events"]+=1
                cursor+=1
            while q and q[0][0]<refresh-DAY:q.popleft();stats["expired_events"]+=1
            window=[v for _,v in q][-2048:]
            lo=bisect_left(times,refresh-DAY);hi=bisect_left(times,refresh)
            if window!=ids[max(lo,hi-2048):hi]:raise AssertionError("Rolling window mismatch")
            stats["full_rebuild_token_equivalent"]+=len(window)
            stats["refresh_jobs"]+=1
            if previous is not None:stats["incremental_refresh_jobs"]+=1
            previous=refresh
            yield f"{user}:{refresh}",[int(vocab[v])+1 for v in window]

def one_trial(server,token,events,keys,policy,workers):
    jobs=[(i,key,int(c),int(u)) for i,(key,c,u) in enumerate(zip(keys,events.candidate,events.user_id))]
    def bucket(items):
        conn=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=60)
        results=[]
        try:
            for i,key,c,u in items:
                data=json.dumps(dict(history_key=key,candidate_id=c,user_id=u,policy=policy))
                start=perf_counter()
                conn.request("POST","/recommend",data,{"Authorization":"Bearer "+token,"Content-Type":"application/json"})
                response=conn.getresponse();body=response.read();elapsed=(perf_counter()-start)*1000
                if response.status!=200:raise RuntimeError("Recommendation HTTP status "+str(response.status))
                d=json.loads(body)
                if not np.isfinite(d["score"]):raise RuntimeError("Nonfinite score")
                if policy!="full" and d["read_tokens"]>512:raise AssertionError("Bound exceeded")
                results.append((i,d,elapsed,len(body)))
        finally:conn.close()
        return results
    t=perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result=[v for a in pool.map(bucket,[jobs[i::workers] for i in range(workers)]) for v in a]
    wall=perf_counter()-t
    result.sort(key=lambda x:x[0])
    if [v[0] for v in result]!=list(range(len(events))):raise AssertionError("Missing requests")
    out=[v[1] for v in result];lat=np.asarray([v[2] for v in result]);stats=Counter()
    for d in out:stats.update(d["storage"])
    return np.asarray([d["score"] for d in out]),dict(
        workers=workers,requests=len(out),throughput_rps=len(out)/wall,
        p50_ms=float(np.quantile(lat,.5)),p95_ms=float(np.quantile(lat,.95)),p99_ms=float(np.quantile(lat,.99)),
        mean_read_tokens=float(np.mean([d["read_tokens"] for d in out])),
        mean_model_tokens=float(np.mean([d["model_tokens"] for d in out])),
        mean_linear_macs=float(np.mean([d["linear_macs"] for d in out])),
        mean_full_tokens=float(np.mean([d["history_length"] for d in out])),
        logical_payload_bytes=stats["logical_payload_bytes"],
        response_body_bytes=sum(v[3] for v in result),storage_totals=dict(stats))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=ROOT/"models/m3_shared.pt")
    p.add_argument("--bundle",type=Path,default=ROOT/"models/m3_bundle.joblib")
    p.add_argument("--root",type=Path,default=ROOT/"data/processed/m3_refresh_snapshots")
    p.add_argument("--output",type=Path,default=ROOT/"reports/m3_refresh_workloads.json")
    p.add_argument("--sample",type=int,default=0)
    p.add_argument("--workers",type=int,nargs="+",default=[1,8])
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--anchor-hour",type=int,default=2)
    p.add_argument("--utc-offset",type=int,default=7)
    p.add_argument("--bootstrap",type=int,default=200)
    a=p.parse_args()
    if a.sample<0 or min(a.workers)<1 or a.repeats<1 or a.bootstrap<20 or not 0<=a.anchor_hour<=23 or not -12<=a.utc_offset<=14:p.error("Invalid args")
    secret,token=credentials();events=events_only()
    _,vocab=baseline.load_video_vocab()
    lower=min(refresh_time(events.time_ms.min(),h,a.anchor_hour,a.utc_offset) for h in CADENCES)-DAY
    raw=load_raw(STANDARD,set(map(int,events.user_id)),lower,int(events.time_ms.max()))
    mask,excluded=common_cohort(events,raw,vocab,a.anchor_hour,a.utc_offset)
    total=len(events);events=events.loc[mask].reset_index(drop=True);eligible=len(events)
    if a.sample:events=events.sample(min(a.sample,eligible),random_state=42).reset_index(drop=True)
    y=events.long_view.to_numpy(dtype=int);u=events.user_id.to_numpy(dtype=int)
    if len(np.unique(y))!=2:raise ValueError("Common cohort must contain both classes")
    bundle=joblib.load(a.bundle);sha=_sha(a.checkpoint)
    if bundle["checkpoint_sha256"]!=sha:raise ValueError("Bundle/checkpoint mismatch")
    model=load_model(a.checkpoint,baseline.get_device());emb=model.item.weight.detach().cpu().numpy()
    del model
    rows=[];refreshes=[];scores={}
    a.root.mkdir(parents=True,exist_ok=True);a.output.parent.mkdir(parents=True,exist_ok=True)
    for hours in CADENCES:
        jobs,keys=make_jobs(events,hours,a.anchor_hour,a.utc_offset)
        stats=Counter();t=perf_counter()
        sid=build(a.root,rolling_records(jobs,raw,vocab,stats),emb,secret,
                  metadata={"checkpoint_sha256":sha,"vocab_sha256":_sha(baseline.VIDEO_VOCAB_FILE),"refresh_hours":hours})
        refreshes.append(dict(refresh_hours=hours,snapshot_id=sid,build_wall_seconds=perf_counter()-t,
            snapshot_bytes=sum(f.stat().st_size for f in (a.root/sid).iterdir() if f.is_file()),
            mean_refresh_age_hours=float(np.mean([(int(t)-refresh_time(t,hours,a.anchor_hour,a.utc_offset))/HOUR for t in events.time_ms])),
            distinct_history_keys=len(set(keys)),**stats))
        with tempfile.TemporaryDirectory() as tmp:
            bp=Path(tmp)/"bundle.joblib";joblib.dump(dict(bundle,snapshot_id=sid),bp)
            engine=Engine(SimpleNamespace(bundle=bp,checkpoint=a.checkpoint,root=a.root))
            server=make_server(engine,token,0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for workers in sorted(set(a.workers)):
                    for rep in range(a.repeats):
                        order=list(POLICIES);random.Random(42+rep).shuffle(order)
                        for policy in order:
                            pred,metrics=one_trial(server,token,events,keys,policy,workers)
                            name=f"h{hours}_{policy}"
                            if name in scores and not np.allclose(scores[name],pred,rtol=0,atol=1e-6):raise RuntimeError("Prediction drift across workload")
                            scores[name]=pred
                            row=dict(refresh_hours=hours,policy=policy,repeat=rep+1,auc=float(roc_auc_score(y,pred)),gauc=gauc(y,u,pred),logloss=float(log_loss(y,pred,labels=[0,1])),**metrics)
                            rows.append(row);print(json.dumps(row),flush=True)
                if not np.allclose(scores[f"h{hours}_adaptive"],scores[f"h{hours}_adaptive_exact"],rtol=0,atol=1e-7):raise RuntimeError("Bloom parity failed")
            finally:
                server.shutdown();server.server_close();thread.join()
            del engine
    ci=[]
    for hours in CADENCES:
        for policy in POLICIES:
            name=f"h{hours}_{policy}";ref=f"h{hours}_full"
            ci.append(dict(refresh_hours=hours,policy=policy,
                delta_vs_same_refresh_full=float(roc_auc_score(y,scores[name])-roc_auc_score(y,scores[ref])),
                ci_vs_same_refresh_full=paired_user_cluster_ci(y,u,scores[name],scores[ref],a.bootstrap),
                ci_vs_event_time_full=paired_user_cluster_ci(y,u,scores[name],scores["h0_full"],a.bootstrap)))
    report=dict(status="PASS",scope="real_log_24h_rolling_history_immutable_refresh_snapshot_loopback_http",
        checkpoint_sha256=sha,source_requests=total,common_cohort_requests=eligible,tested_requests=len(events),
        excluded_by_cadence=excluded,anchor_hour=a.anchor_hour,utc_offset=a.utc_offset,
        refreshes=refreshes,rows=rows,paired_ci=ci,
        notes=["All cadences use a common vocabulary-safe cohort and frozen selector/checkpoint.",
               "Incremental event cursor construction is real; persistent index publication is a full immutable rebuild.",
               "Concurrent closed-loop workers, not open-loop burst arrivals. Model inference serialized by engine lock.",
               "Logical result bytes and linear MACs are not physical I/O or total FLOPs."])
    a.output.write_text(json.dumps(report,indent=2)+"\n");pd.DataFrame(rows).to_csv(a.output.with_suffix(".csv"),index=False)
    np.savez_compressed(a.output.with_suffix(".npz"),labels=y,users=u,evaluation_ids=events.evaluation_id.to_numpy(),**scores)
    print(json.dumps({"status":"PASS","requests":len(events),"refreshes":refreshes,"output":str(a.output)},indent=2))

if __name__=="__main__":main()
