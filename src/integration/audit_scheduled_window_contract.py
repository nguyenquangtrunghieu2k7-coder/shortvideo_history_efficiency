"""Read-only audit: Phase7 reconstruction parity and OOV by scheduled window.

No vocabulary mutation, no dataset rewriting, no inference.
"""
from collections import defaultdict, Counter
from bisect import bisect_left
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.integration.scheduled_window_ablation import STANDARD, DAY, last_refresh
from src.integration import e2e_v4_bloom as e2e

def main():
    vocab,vocab_map=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset("test","Full",vocab_map)
    events=pd.read_parquet(e2e.ROOT/"data/processed/evaluation_events.parquet",
        columns=["evaluation_id","user_id","time_ms"])
    ev=events.set_index("evaluation_id").loc[ds.evaluation_ids].reset_index()
    assert ev.user_id.astype("int64").tolist()==list(map(int,ds.user_ids))
    lower=min(last_refresh(int(ev.time_ms.min()),s,2,7) for s in (24,12,6,3))-DAY
    upper=int(ev.time_ms.max())
    users=set(map(int,ev.user_id))
    records=defaultdict(list)
    order=0
    total_raw=0
    for path in STANDARD:
        for chunk in pd.read_csv(path,usecols=["user_id","video_id","time_ms"],chunksize=250000):
            sub=chunk.loc[chunk.user_id.isin(users)&(chunk.time_ms>=lower)&(chunk.time_ms<upper)]
            for user,raw,t in sub[["user_id","video_id","time_ms"]].itertuples(index=False,name=None):
                records[int(user)].append((int(t),order,int(raw)))
                total_raw+=1
                order+=1
    indexed={}
    for user,values in records.items():
        values.sort(key=lambda x:(x[0],x[1]))
        indexed[user]=([v[0] for v in values],[v[2] for v in values])
    parity=Counter()
    mismatch_examples=[]
    for row,frozen in zip(ev.itertuples(index=False),ds.history_ids):
        times,raws=indexed.get(int(row.user_id),([],[]))
        lo,hi=bisect_left(times,int(row.time_ms)-DAY),bisect_left(times,int(row.time_ms))
        reconstructed=raws[max(lo,hi-2048):hi]
        oov=[v for v in reconstructed if v not in vocab_map]
        parity["event_time_oov"]+=len(oov)
        mapped=[vocab_map[v]+1 for v in reconstructed if v in vocab_map]
        parity["matching_requests"]+=int(mapped==list(frozen))
        parity["mismatching_requests"]+=int(mapped!=list(frozen))
        if mapped!=list(frozen) and len(mismatch_examples)<5:
            mismatch_examples.append({"evaluation_id":int(row.evaluation_id),
                "reconstructed_len":len(mapped),"frozen_len":len(frozen),
                "reconstructed_head":mapped[:5],"frozen_head":list(frozen)[:5]})
    scenarios=[]
    for stride in (24,12,6,3):
        jobs=set()
        window_rows=window_oov=window_oov_capped=0
        request_oov=0
        for row in ev.itertuples(index=False):
            user,t=int(row.user_id),int(row.time_ms)
            refresh=last_refresh(t,stride,2,7)
            times,raws=indexed.get(user,([],[]))
            lo,hi=bisect_left(times,refresh-DAY),bisect_left(times,refresh)
            window=raws[lo:hi]
            capped=window[-2048:]
            request_oov+=sum(v not in vocab_map for v in capped)
            key=(user,refresh)
            if key not in jobs:
                jobs.add(key)
                window_rows+=len(window)
                window_oov+=sum(v not in vocab_map for v in window)
                window_oov_capped+=sum(v not in vocab_map for v in capped)
        scenarios.append({"stride_hours":stride,"observed_jobs":len(jobs),
            "job_window_raw_rows":window_rows,"job_window_oov_rows":window_oov,
            "job_window_oov_rate":window_oov/max(window_rows,1),
            "job_window_oov_capped_rows":window_oov_capped,
            "request_window_oov_capped_rows":request_oov})
    report={"status":"PASS" if parity["mismatching_requests"]==0 and parity["event_time_oov"]==0 else "FAIL",
        "raw_rows_scanned":total_raw,"frozen_vocab_size":len(vocab_map),
        "event_time_reconstruction_parity":dict(parity),
        "mismatch_examples":mismatch_examples,"scheduled_windows":scenarios,
        "warning":"Audit does not change frozen artifacts; original Stage2 OOV removal changes window contents and must not be considered baseline-equivalent."}
    dest=e2e.ROOT/"reports/experiments/scheduled_window_contract_audit.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if report["status"]!="PASS":raise SystemExit("STRICT AUDIT FAILED: do not accept Stage2 AUC")
if __name__=="__main__":main()
