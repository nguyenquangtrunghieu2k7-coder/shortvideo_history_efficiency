"""One-command cache-only calibration + sampled real HTTP release gate.
Does not train or score all actions again. Missing caches fail closed.
Creates a portable runtime directory after successful native gates."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib,http.client,json,platform,random,shutil,socket,subprocess,threading
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from src.integration import m3_product as p,m5_selector as s
from src.integration.global_policy import KS,compile_policy,select,publish
from src.integration.global_serving import Engine
from src.integration.m23_persistent_store import _sha

def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(path,obj):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m4-dir",type=Path,default=Path("reports/m4_research_v3"))
    ap.add_argument("--m5-dir",type=Path,default=Path("reports/m5_release_v1"))
    ap.add_argument("--cache",type=Path,default=Path("reports/m6_cache_v1"))
    ap.add_argument("--output",type=Path,default=Path("reports/global_release_v1"))
    ap.add_argument("--sample",type=int,default=128)
    ap.add_argument("--workers",type=int,nargs="+",default=[1,4,8])
    ap.add_argument("--repeats",type=int,default=2)
    a=ap.parse_args()
    if min(a.sample,a.repeats,*a.workers)<1:ap.error("Positive counts required")
    d=a.output.resolve();d.mkdir(parents=True,exist_ok=True)
    m4=a.m4_dir;m5=a.m5_dir;cache=a.cache
    if (d/"runtime").exists():raise ValueError("Runtime already exists; use a new output directory")
    ck=m4/(json.loads((m4/"summary.json").read_text())["selected_on_validation"]+".pt")
    prior=json.loads((m5/"summary.json").read_text());contract=json.loads((cache/"contract.json").read_text())
    if prior["checkpoint_sha256"]!=_sha(ck) or contract["checkpoint"]!=_sha(ck) or contract["snapshot"]!=prior["snapshot_id"] or contract["test_predictions_sha256"]!=_sha(m5/"predictions.npz"):raise ValueError("Cache provenance mismatch")
    # Verify inputs against the original M6 contract before reuse.
    for path,digest in contract["input_hashes"].items():
        if _sha(Path(path))!=digest:raise ValueError("Input changed: "+path)
    V=read(cache/"validation_features.npz");T=read(cache/"test_features.npz");Z=read(m5/"predictions.npz")
    if not np.array_equal(T["ids"],Z["evaluation_ids"]):raise ValueError("Test IDs mismatch")
    names=[f"recent_{k}" for k in KS];columns=[s.ACTION_NAMES.index(n) for n in names]
    if tuple(prior["actions"])!=s.ACTION_NAMES:raise ValueError("Action contract mismatch")
    P=Z["action_predictions"][:,columns];vp=[]
    for n in names:
        c=read(cache/f"val_{n}.npz")
        if not np.array_equal(c["ids"],V["ids"]):raise ValueError("Validation IDs mismatch")
        vp.append(c["pred"])
    VP=np.column_stack(vp);groups=np.unique(V["users"]);np.random.default_rng(42).shuffle(groups)
    fit=np.isin(V["users"],groups[:max(1,int(.65*len(groups)))])
    if not fit.any() or not (~fit).any() or not np.isfinite(VP).all() or not np.isfinite(P).all():raise ValueError("Invalid cache")
    loss=s.loss_matrix(V["labels"][fit],VP[fit]);N=V["N"][fit]
    rows=[]
    for i,k in enumerate(KS):
        eff=np.minimum(N,k);rows.append(dict(k=k,mean_tokens=float(eff.mean()),mean_bce=float(loss[:,i].mean()),
            mean_macs=float((s.HEAD_MACS+s.TOKEN_MACS*np.maximum(eff,1)).mean())))
    table=dict(schema=1,checkpoint_sha256=_sha(ck),snapshot_id=prior["snapshot_id"],vocab_sha256=_sha(p.baseline.VIDEO_VOCAB_FILE),fit_requests=int(fit.sum()),rows=rows)
    write(d/"calibration.json",table);print("CALIBRATION CACHE HIT",flush=True)
    configs=[(f"q{w:g}",dict(quality_weight=w),None) for w in s.PROFILES]
    configs += [(f"budget_{b}",dict(quality_weight=1.),b) for b in (5,10,20,30,50,100)]
    configs += [("cap50",dict(quality_weight=.99,max_read_tokens=50,max_logical_bytes=800,max_linear_macs=s.HEAD_MACS+50*s.TOKEN_MACS),None)]
    plans={};docs={};quality=[];actions={}
    y=Z["labels"];u=Z["users"];ids=T["ids"];j=np.arange(len(ids))
    for idx,(name,pref,budget) in enumerate(configs,1):
        doc=dict(revision=idx,preferences=pref,average_tokens=budget);plan=compile_policy(table,doc)
        plans[name]=plan;docs[name]=doc
        ks=np.array([select(plan,str(int(key))) for key in ids]);ix=np.array([KS.index(k) for k in ks]);actions[name]=ix
        eff=np.minimum(T["N"],ks);score=P[j,ix]
        quality.append(dict(profile=name,k_a=plan["k_a"],k_b=plan["k_b"],prob_b=plan["prob_b"],expected_validation_tokens=plan["expected_tokens"],
            test_mean_tokens=float(eff.mean()),mean_bytes=float(16*eff.mean()),mean_macs=float((s.HEAD_MACS+s.TOKEN_MACS*np.maximum(eff,1)).mean()),
            auc=float(roc_auc_score(y,score)),gauc=p.gauc(y,u,score),logloss=float(log_loss(y,score,labels=[0,1]))))
    for k in KS:
        name=f"fixed_{k}";ix=np.full(len(ids),KS.index(k));actions[name]=ix;eff=np.minimum(T["N"],k);score=P[j,ix]
        quality.append(dict(profile=name,k_a=k,k_b=k,prob_b=0.,expected_validation_tokens=rows[KS.index(k)]["mean_tokens"],test_mean_tokens=float(eff.mean()),mean_bytes=float(16*eff.mean()),mean_macs=float((s.HEAD_MACS+s.TOKEN_MACS*np.maximum(eff,1)).mean()),auc=float(roc_auc_score(y,score)),gauc=p.gauc(y,u,score),logloss=float(log_loss(y,score,labels=[0,1]))))
    q=pd.DataFrame(quality);q.to_csv(d/"quality_cost.csv",index=False);write(d/"compiled_plans.json",plans)
    ev=p.events_only().set_index("evaluation_id").loc[ids]
    if not np.array_equal(ev.long_view.to_numpy(),y) or not np.array_equal(ev.user_id.to_numpy(),u):raise ValueError("Event alignment mismatch")
    subset=np.sort(np.random.default_rng(42).choice(len(ids),min(a.sample,len(ids)),replace=False))
    req=[dict(history_key=str(int(ids[i])),candidate_id=int(ev.candidate.iloc[i]),user_id=int(u[i]),policy="preference") for i in subset]
    _,token=p.credentials()
    policyfile=d/"benchmark_policy.json"
    if policyfile.exists():policyfile.unlink()
    doc=dict(revision=1,preferences=dict(quality_weight=0.),average_tokens=None);publish(policyfile,table,doc)
    engine=Engine(ck,m4/"snapshots",table,policyfile)
    server=p.make_server(engine,token,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def connection():
        c=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=90);c.connect();c.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);return c
    def send(c,body,auth=token):
        t=perf_counter();c.request("POST","/recommend",json.dumps(body),{"Authorization":"Bearer "+auth,"Content-Type":"application/json"})
        res=c.getresponse();out=json.loads(res.read());return res.status,out,(perf_counter()-t)*1000
    trials=[];revision=1
    try:
        c=connection()
        try:
            status,out,_=send(c,req[0]);assert status==200 and out["policy_revision"]==1 and out["budget"]==0
            revision+=1;publish(policyfile,table,dict(revision=revision,preferences=dict(quality_weight=1.,max_read_tokens=50),average_tokens=None))
            status,out,_=send(c,req[0]);assert status==200 and out["policy_revision"]==revision and out["budget"]<=50
            policyfile.write_text("{");status,out,_=send(c,req[0]);assert status==200 and out["policy_revision"]==revision and out["policy_reload_error"]
            assert send(c,dict(req[0],preferences={"quality_weight":0.}))[0]==400
            assert send(c,req[0],auth="bad")[0]==401
            assert send(c,dict(req[0],history_key="missing-release-key"))[0]==404
            revision+=1;policyfile.unlink();publish(policyfile,table,dict(revision=revision,preferences=dict(quality_weight=0.),average_tokens=None))
        finally:c.close()
        # Compare all policies on identical request IDs, same single-lookup path.
        scenarios=["q0","q0.5","q0.9","q0.99","q1","budget_30","cap50","fixed_5","fixed_50","fixed_2048"]
        for workers in a.workers:
            for rep in range(a.repeats):
                order=scenarios.copy();random.Random(42+rep).shuffle(order)
                for name in order:
                    revision+=1
                    if name.startswith("fixed_"):
                        k=int(name.split("_")[1]);engine.policy_store=None;engine.fixed_k=k
                    else:
                        engine.fixed_k=None
                        from src.integration.global_policy import PolicyStore
                        publish(policyfile,table,dict(docs[name],revision=revision))
                        if engine.policy_store is None:engine.policy_store=PolicyStore(policyfile,table)
                    expected=actions[name][subset]
                    def task(chunk):
                        con=connection();out=[]
                        try:
                            for pos in chunk:
                                status,res,ms=send(con,req[pos]);i=subset[pos];col=expected[pos]
                                if status!=200 or res["budget"]!=KS[col] or abs(res["score"]-P[i,col])>2e-5 or res["storage"]["sqlite_lookups"]!=1 or res["storage"]["summary_rows"]!=0:raise RuntimeError("HTTP parity/accounting failed: "+name+str(res))
                                out.append((ms,res))
                        finally:con.close()
                        return out
                    task([0]);start=perf_counter()
                    with ThreadPoolExecutor(max_workers=workers) as pool:results=[x for z in pool.map(task,[list(range(i,len(req),workers)) for i in range(workers)]) for x in z]
                    wall=perf_counter()-start;lat=np.array([x[0] for x in results]);out=[x[1] for x in results]
                    row=dict(profile=name,workers=workers,repeat=rep+1,requests=len(out),p50_ms=float(np.median(lat)),p95_ms=float(np.quantile(lat,.95)),p99_ms=float(np.quantile(lat,.99)),rps=len(out)/wall,mean_tokens=float(np.mean([v["read_tokens"] for v in out])),mean_selector_ms=float(np.mean([v["timing"]["selector_ms"] for v in out])),mean_storage_ms=float(np.mean([v["timing"]["storage_ms"] for v in out])),mean_model_ms=float(np.mean([v["timing"]["model_ms"] for v in out])))
                    trials.append(row);print("HTTP",row,flush=True)
                    pd.DataFrame(trials).to_csv(d/"http_trials.csv",index=False)
    finally:server.shutdown();server.server_close();thread.join()
    frame=pd.DataFrame(trials);agg=frame.groupby(["profile","workers"],as_index=False)[["p50_ms","p95_ms","p99_ms","rps","mean_tokens","mean_selector_ms","mean_storage_ms","mean_model_ms"]].median();agg.to_csv(d/"http_comparison.csv",index=False)
    # Portable package: no training data, predictions, secrets or joblib.
    runtime=d/"runtime";staging=d/"runtime.tmp"
    if staging.exists():raise ValueError("Incomplete package exists; remove runtime.tmp after inspection")
    staging.mkdir();(staging/"data").mkdir();(staging/"snapshots").mkdir();(staging/"policy").mkdir()
    shutil.copy2(ck,staging/"ranker.pt");shutil.copy2(p.baseline.VIDEO_VOCAB_FILE,staging/"data"/"video_vocab.parquet")
    shutil.copytree(m4/"snapshots"/table["snapshot_id"],staging/"snapshots"/table["snapshot_id"])
    shutil.copy2(d/"calibration.json",staging/"calibration.json")
    write(staging/"policy"/"business.json",dict(revision=1,preferences=dict(quality_weight=.9),average_tokens=None))
    smoke_i=int(subset[0]);write(staging/"smoke.json",dict(request=req[0],expected_score=float(P[smoke_i,actions["q0.9"][smoke_i]]),expected_k=KS[actions["q0.9"][smoke_i]]))
    manifest={str(f.relative_to(staging)):_sha(f) for f in staging.rglob("*") if f.is_file()};write(staging/"manifest.json",manifest);staging.replace(runtime)
    def md(df):
        return "| "+" | ".join(df.columns)+" |\n| "+" | ".join(["---"]*len(df.columns))+" |\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in df.itertuples(index=False,name=None))
    report="# Global policy release — native gate\n\n"+md(q.round(6))+"\n\n## HTTP trials (median per-trial metrics)\n\n"+md(agg.round(6))+"\n\nNative HTTP parity, hard caps, hot reload, invalid policy fallback, unauthorized and override rejection passed. Docker verification pending. Reused test is exploratory; average budget is expectation on calibration, not a hard fleet cap.\n"
    (d/"REPORT.md").write_text(report)
    write(d/"summary.json",dict(status="NATIVE_PASS",docker_verified=False,checkpoint_sha256=table["checkpoint_sha256"],snapshot_id=table["snapshot_id"],platform=platform.platform(),device=str(engine.device),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),test_requests=len(ids),http_requests=int(frame.requests.sum()),gates=dict(parity=True,single_lookup=True,policy_reload=True,invalid_policy_fallback=True,authorization=True,override_rejection=True,hard_caps=True),scope="Local native HTTP; cached test quality; no retraining; not production certification"))
    print(report,flush=True);print("RELEASE:",d)
if __name__=="__main__":main()
