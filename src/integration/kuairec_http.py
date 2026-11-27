"""KuaiRec or VK-LSVD native HTTP benchmark of the frozen raw Global Policy.
Reuses the shared M4 checkpoint, offline predictions, and original policy compiler.
Builds one signed snapshot for the full warm-item test cohort; not a cloud benchmark.
No training or whole-cohort model inference, no contextual serving claim."""
import argparse,http.client,json,platform,random,socket,subprocess,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from src.integration import m3_product as p,m5_selector as s
from src.integration.global_policy import KS,compile_policy,select,publish
from src.integration.global_serving import Engine
from src.integration.m3_store import build
from src.integration.m23_persistent_store import _sha
from src.integration.kuairec_experiment import read,write,NAMES

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",choices=["kuairec","vklsvd","wechat"],default="kuairec")
    ap.add_argument("--source",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--sample",type=int,default=128)
    ap.add_argument("--workers",type=int,nargs="+",default=[1,4,8])
    ap.add_argument("--repeats",type=int,default=2)
    a=ap.parse_args()
    if min(a.sample,a.repeats,*a.workers)<1:ap.error("Positive counts required")
    d=(a.output or Path(f"reports/{a.dataset}_http_v1")).resolve();src=(a.source or Path(f"reports/{a.dataset}_v1")).resolve();cache=src/"cache";data=p.baseline.PROCESSED_DIR
    if d.name!=f"{a.dataset}_http_v1" or src.name!=f"{a.dataset}_v1" or d==src:raise ValueError("Use isolated dataset HTTP report directory")
    if data.name!="processed" or data.parent.name!=a.dataset:raise ValueError(f"Set SHORTVIDEO_PROCESSED_DIR to data/{a.dataset}/processed")
    d.mkdir(parents=True,exist_ok=True)
    if (d/"summary.json").exists():raise ValueError("Benchmark already complete; preserve the results")
    ck=src/"behavior.pt";summary=json.loads((src/"summary.json").read_text());cc=json.loads((cache/"contract.json").read_text())
    mf=data/"adapter_manifest.json";manifest=json.loads(mf.read_text())
    for path,digest in manifest["artifacts"].items():
        if _sha(data/path)!=digest:raise ValueError("Data changed: "+path)
    digest=_sha(ck)
    expected={"kuairec":"KuaiRec 2.0 big matrix","vklsvd":"VK-LSVD up0.001_ip0.001 weeks 20-25","wechat":"WeChat D2Co days 1-14"}[a.dataset]
    if summary["dataset"]!=expected:raise ValueError("Dataset mismatch")
    if summary["status"]!="OFFLINE_PASS" or digest!=summary["checkpoint_sha256"] or digest!=cc["checkpoint_sha256"] or _sha(mf)!=cc["training"]["adapter_sha256"] or list(NAMES)!=cc["actions"]:raise ValueError("Offline contract mismatch")
    if _sha(Path(__file__).with_name("m5_selector.py"))!=cc["selector_sha256"] or _sha(Path(__file__).with_name("m4_ranker.py"))!=cc["training"]["ranker_sha256"]:raise ValueError("Frozen code mismatch")
    table=json.loads((src/"calibration.json").read_text())
    if table["checkpoint_sha256"]!=digest or table["vocab_sha256"]!=_sha(p.baseline.VIDEO_VOCAB_FILE):raise ValueError("Calibration provenance mismatch")
    frozen=json.loads((src/"frozen_plans.json").read_text())["global_plans"]
    T=read(cache/"test_features.npz");ids=T["ids"];y=T["y"];u=T["u"];P=[]
    for k in KS:
        z=read(cache/f"test_recent_{k}.npz")
        if not np.array_equal(z["ids"],ids):raise ValueError("Cache ID mismatch")
        P.append(z["pred"])
    P=np.column_stack(P)
    if P.shape!=(len(ids),len(KS)) or not np.isfinite(P).all() or len(ids)!=summary["test_requests"]:raise ValueError("Invalid cache")
    _,vmap=p.baseline.load_video_vocab()
    test=p.baseline.KuaiRandDataset("test","Full",vmap)
    if not all(np.array_equal(x,z) for x,z in ((test.evaluation_ids,ids),(test.labels,y),(test.user_ids,u),(test.candidate_ids,T["candidate"]),(list(map(len,test.history_ids)),T["N"]))):raise ValueError("Test cohort alignment mismatch")
    snapshot_contract=dict(checkpoint_sha256=digest,vocab_sha256=table["vocab_sha256"],adapter_sha256=_sha(mf),test_features_sha256=_sha(cache/"test_features.npz"),scope="full warm-item test cohort only")
    sc=d/"snapshot_contract.json"
    if sc.exists():
        saved=json.loads(sc.read_text())
        if saved["contract"]!=snapshot_contract:raise ValueError("Snapshot contract mismatch")
        sid=saved["snapshot_id"];print("SNAPSHOT CACHE HIT",sid,flush=True)
    else:
        print("BUILDING TEST SNAPSHOT",len(ids),flush=True)
        model=p.load_model(ck,p.baseline.get_device());emb=p.semantic_embeddings(model);del model
        secret,_=p.credentials()
        sid=build(d/"snapshots",((str(int(i)),h) for i,h in zip(ids,test.history_ids)),emb,secret,compact=True,metadata={"checkpoint_sha256":digest,"vocab_sha256":table["vocab_sha256"]})
        del emb
        write(sc,dict(contract=snapshot_contract,snapshot_id=sid))
        print("SNAPSHOT SAVED",sid,flush=True)
    del test
    table["snapshot_id"]=sid;write(d/"calibration.json",table)
    configs=[(f"q{w:g}",dict(quality_weight=w),None) for w in s.PROFILES]
    configs += [(f"budget_{b}",dict(quality_weight=1.),b) for b in (5,10,20,30,50,100)]
    configs += [("cap50",dict(quality_weight=.99,max_read_tokens=50,max_logical_bytes=800,max_linear_macs=s.HEAD_MACS+50*s.TOKEN_MACS),None)]
    plans={};docs={};actions={}
    raw=read(src/"selected_actions.npz")
    if not np.array_equal(raw["ids"],ids):raise ValueError("Decision IDs mismatch")
    for rev,(name,pref,budget) in enumerate(configs,1):
        doc=dict(revision=rev,preferences=pref,average_tokens=budget);plan=compile_policy(table,doc)
        if plan!=frozen[name]:raise ValueError("Frozen policy mismatch: "+name)
        plans[name]=plan;docs[name]=doc
        ks=np.array([select(plan,str(int(key))) for key in ids]);ix=np.array([KS.index(int(k)) for k in ks]);actions[name]=ix
        if not np.array_equal(raw["global__"+name],np.array([NAMES.index(f"recent_{k}") for k in ks])):raise ValueError("Frozen action mismatch: "+name)
    for k in KS:actions[f"fixed_{k}"]=np.full(len(ids),KS.index(k))
    subset=np.sort(np.random.default_rng(42).choice(len(ids),min(a.sample,len(ids)),replace=False))
    req=[dict(history_key=str(int(ids[i])),candidate_id=int(T["candidate"][i]),user_id=int(u[i]),policy="preference") for i in subset]
    _,token=p.credentials()
    policyfile=d/"benchmark_policy.json"
    if policyfile.exists():policyfile.unlink()
    doc=dict(revision=1,preferences=dict(quality_weight=0.),average_tokens=None);publish(policyfile,table,doc)
    engine=Engine(ck,d/"snapshots",table,policyfile)
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
    write(d/"summary.json",dict(status="NATIVE_PASS",dataset={"kuairec":"KuaiRec","vklsvd":"VK-LSVD","wechat":"WeChat"}[a.dataset],checkpoint_sha256=digest,snapshot_id=sid,snapshot_scope=snapshot_contract["scope"],platform=platform.platform(),device=str(engine.device),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),test_requests=len(ids),http_requests=int(frame.requests.sum()),gates=dict(parity=True,single_lookup=True,policy_reload=True,invalid_policy_fallback=True,authorization=True,override_rejection=True,hard_caps=True),scope="Local raw Global Policy only; not contextual serving, calibrated serving, Docker, or cloud; snapshot contains test histories only"))
    print(agg.to_string(index=False),flush=True)
    print(a.dataset.upper()+"_NATIVE_PASS",flush=True)
if __name__=="__main__":main()
