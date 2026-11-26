"""Fast-path verification using saved M5 predictions, no recommender retraining
or full action scoring. Local HTTP only. Compare same requests and decisions."""
import argparse
import json
from pathlib import Path
from collections import Counter
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
from src.integration import m3_product as p
from src.integration import m5_selector as s
from src.integration.m5_selector_experiment import Engine,http_check

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m4-dir",type=Path,default=Path("reports/m4_research_v3"))
    ap.add_argument("--m5-dir",type=Path,default=Path("reports/m5_release_v1"))
    ap.add_argument("--output",type=Path,default=Path("reports/m5_fast_v1"))
    ap.add_argument("--http-sample",type=int,default=256)
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    m4=a.m4_dir;m5=a.m5_dir
    ck=m4/(json.loads((m4/"summary.json").read_text())["selected_on_validation"]+".pt")
    engine=Engine(SimpleNamespace(checkpoint=ck,root=m4/"snapshots",bundle=m4/"int8.joblib",selector=m5/"selector.joblib"))
    _,vmap=p.baseline.load_video_vocab()
    ds=p.baseline.KuaiRandDataset("test","Full",vmap)
    with np.load(m5/"predictions.npz",allow_pickle=False) as data:
        if not np.array_equal(data["evaluation_ids"],np.asarray(ds.evaluation_ids)):raise ValueError("Saved prediction order mismatch")
        P=data["action_predictions"].copy()
        saved={k:data[k].copy() for k in data.files if k.startswith("q") or k.startswith("extended_q")}
    if P.shape!=(len(ds),len(s.ACTIONS)):raise ValueError("Action grid mismatch")
    # Features only: reuse the frozen embedding and signed index. No recommender inference here.
    X=[];N=[]
    for key,c,u in zip(ds.evaluation_ids,ds.candidate_ids,ds.user_ids):
        meta,_=engine.store.lookup(str(key),"summaries",mode="indexed_exact")
        if meta is None:raise KeyError(key)
        X.append(s.feature_vector(meta["n"],meta["blocks"],engine.emb[c],engine.ue[u%len(engine.ue)]));N.append(meta["n"])
    X=np.asarray(X);res=s.resource_table(N)
    old=engine.policy_model["regressor"];fast=engine.fast_predictor
    reference=old.predict(X);pred=fast.predict(X)
    diff=float(np.max(np.abs(reference-pred)))
    if diff>1e-8:raise RuntimeError(f"Regret parity failed: {diff}")
    plan=json.loads((m5/"frozen_plan.json").read_text())["profiles"]
    decisions={};prefs={}
    for row in plan:
        name=row["name"];pref=row["preferences"]
        ix=s.choose(reference,res,pref);fx=s.choose(pred,res,pref)
        if not np.array_equal(ix,fx):raise RuntimeError("Action parity failed: "+name)
        if name in saved and not np.allclose(P[np.arange(len(ds)),ix],saved[name],atol=1e-8,rtol=0):raise RuntimeError("Saved prediction mismatch: "+name)
        if name in ("q0","q0.5","q0.9","q0.99","q1"):
            decisions[name]=ix;prefs[name]=pref
    # Warm all prediction routes, then compare per-request timing.
    sample=X[np.random.default_rng(42).choice(len(X),min(a.http_sample,len(X)),replace=False)]
    profiles=[]
    original_jobs=getattr(old,"n_jobs",None)
    predictors=[("original",old)]
    if hasattr(old,"n_jobs"):
        predictors.append(("single_job",old))
    predictors.append(("native_fast",fast))
    for name,model in predictors:
        if name=="single_job":old.n_jobs=1
        elif name=="original" and hasattr(old,"n_jobs"):old.n_jobs=original_jobs
        model.predict(sample[:1]);times=[]
        for x in sample:
            t=perf_counter();model.predict(x[None,:]);times.append((perf_counter()-t)*1000)
        profiles.append(dict(implementation=name,p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),mean_ms=float(np.mean(times))))
    if hasattr(old,"n_jobs"):old.n_jobs=original_jobs
    rows=[]
    for name in ("recent_0","recent_5","recent_50","recent_2048"):
        j=s.ACTION_NAMES.index(name);decisions[name]=np.full(len(ds),j);prefs[name]=None
    for name,enabled in (("original",False),("optimized",True)):
        engine.fast_enabled=enabled;engine.fast_path_enabled=enabled
        result=http_check(a,engine,ds,P,decisions,prefs)
        rows.extend(dict(implementation=name,**v) for v in result)
    pd.DataFrame(profiles).to_csv(a.output/"selector_latency.csv",index=False)
    pd.DataFrame(rows).to_csv(a.output/"http_comparison.csv",index=False)
    # Cloud sensitivity: serial compute + bytes transfer proxy, not measured cloud.
    df=pd.DataFrame(rows);opt=df[df.implementation=="optimized"].set_index("scenario")
    sensitivity=[]
    for name in ("q0.5","q0.9","q0.99"):
        for base in ("recent_50","recent_2048"):
            v=opt.loc[name];b=opt.loc[base]
            # time = local p50 + extra per-token remote handling + payload transfer.
            # These are illustrative scenarios; p50 addition is not an end-to-end quantile.
            for per_token_us in (0,1,5,10,50,100):
                for mbps in (10,100,1000):
                    def estimate(z):
                        return z.p50_ms+z.mean_read_tokens*per_token_us/1000+z.mean_logical_bytes*8/(mbps*1000)
                    sensitivity.append(dict(profile=name,baseline=base,extra_token_us=per_token_us,link_mbps=mbps,
                        adaptive_estimate_ms=float(estimate(v)),baseline_estimate_ms=float(estimate(b)),
                        adaptive_faster=bool(estimate(v)<estimate(b))))
    pd.DataFrame(sensitivity).to_csv(a.output/"cloud_sensitivity.csv",index=False)
    report=dict(status="EXECUTION_PASS",regret_max_abs_delta=diff,action_parity=True,profiles=profiles,http=rows,
        scope="local HTTP; saved offline predictions; cloud sensitivity is hypothetical, not deployment evidence")
    (a.output/"summary.json").write_text(json.dumps(report,indent=2))
    print(pd.DataFrame(profiles).to_string(index=False))
    print(df[["implementation","scenario","p95_ms","mean_read_tokens","mean_logical_bytes"]].to_string(index=False))
    print("REPORT:",a.output,flush=True)
if __name__=="__main__":main()
