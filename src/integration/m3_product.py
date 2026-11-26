"""Unified M3 experiment: offline calibration, bounded online scoring, HTTP API.

No evaluation history arrays are loaded by the online service or benchmark client.
Only the offline build/calibration stage reads frozen full history artifacts.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
from time import perf_counter, process_time
import resource
import platform

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from src.integration.e2e_v4_bloom import baseline, ROOT
from src.integration.m3_ranker import Ranker
from src.integration.m3_store import build, Store
from src.integration.m23_persistent_store import _sha

BUDGETS=(16,32,50,64,128,256,512)
BLOCK=32
POLICIES=("full","fixed50","random50","semantic50","adaptive","prune","merge","adaptive_exact")

def credentials():
    secret=bytes.fromhex(os.environ.get("M23_HMAC_KEY_HEX",""))
    token=os.environ.get("M23_API_TOKEN","")
    if len(secret)<32 or len(token)<32:raise RuntimeError("Source m23.env first")
    return secret,token

def load_model(path,device):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    if ck["vocab_sha256"]!=_sha(baseline.VIDEO_VOCAB_FILE):raise ValueError("Vocabulary changed")
    if ck.get("architecture")=="m4":
        from src.integration.m4_ranker import Ranker as Model
    else:Model=Ranker
    m=Model(ck["vocab_size"],ck["dim"]);m.load_state_dict(ck["state"]);m.eval()
    return m.to(device)

def semantic_embeddings(model):
    base=model.item.weight.detach().cpu().numpy()
    if not hasattr(model.item,"residual"):return base
    # Export the identical item representation used by training retrieval.
    residual=model.item.residual.weight.detach().cpu().numpy()
    out=base.copy()
    for lo in range(1,len(out),65536):
        hi=min(len(out),lo+65536)
        ids=np.arange(lo,hi,dtype=np.int64)
        out[lo:hi]+=residual[1+(ids-1)%(len(residual)-1)]
    return out

def sync(device):
    if device.type=="cuda":torch.cuda.synchronize()
    elif device.type=="mps":torch.mps.synchronize()

def summaries(h,emb,compact=False):
    from src.integration.m3_store import quantize_centroid
    if compact:
        return [[j,min(j+BLOCK,len(h)),quantize_centroid(emb[h[j:j+BLOCK]].mean(0)).tolist()] for j in range(0,len(h),BLOCK)]
    return [[j,min(j+BLOCK,len(h)),emb[h[j:j+BLOCK]].mean(0).tolist()]
            for j in range(0,len(h),BLOCK)]

def similarities(blocks,q):
    if not blocks:return np.empty(0)
    x=np.asarray([b[2] for b in blocks],dtype=np.float32)
    return (x@q)/(np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(q),1e-8))

def features(n,blocks,q,u):
    s=similarities(blocks,q)
    return np.r_[np.log1p(n)/np.log(2049),
                 [s.max(),s.mean(),s.std(),s[-1]] if len(s) else [0]*4,q,u]

def positions_to_ranges(pos):
    out=[]
    for p in sorted(set(pos)):
        if out and p==out[-1][1]:out[-1][1]+=1
        else:out.append([p,p+1])
    return out

def semantic_ranges(blocks,q,k):
    left=k;pos=[]
    scores=similarities(blocks,q)
    for j in np.argsort(-scores,kind="stable"):
        a,b,_=blocks[int(j)]
        take=min(left,b-a);pos.extend(range(b-take,b));left-=take
        if not left:break
    return positions_to_ranges(pos)

def predict_batch(model,ds,histories,device,mode="recent",budget=50,batch=64):
    scores=[]
    with torch.inference_mode():
        for i in range(0,len(histories),batch):
            items=[dict(ds[j],history_ids=histories[j]) for j in range(i,min(i+batch,len(histories)))]
            c,h,m,_,u,_=baseline.collate_batch(items)
            z,_,_=model(c.to(device),u.to(device),h.to(device),m.to(device),mode,budget)
            scores.extend(torch.sigmoid(z).cpu().tolist())
    return np.asarray(scores)

def calibrate(model,ds,device,tolerance=.01,compact=False):
    emb=semantic_embeddings(model)
    ue=model.user.weight.detach().cpu().numpy()
    histories=ds.history_ids
    blocks=[summaries(h,emb,compact) for h in histories]
    X=np.asarray([features(len(h),b,emb[c],ue[u%len(ue)]) for h,b,c,u in zip(histories,blocks,ds.candidate_ids,ds.user_ids)])
    pred=[]
    for k in BUDGETS:
        chosen=[[h[j] for a,b in semantic_ranges(s,emb[c],k) for j in range(a,b)]
                for h,s,c in zip(histories,blocks,ds.candidate_ids)]
        pred.append(predict_batch(model,ds,chosen,device))
    pred=np.column_stack(pred);full=predict_batch(model,ds,histories,device)
    y=np.asarray(ds.labels);users=np.asarray(ds.user_ids);n=np.asarray(list(map(len,histories)))
    if len(np.unique(y))!=2:raise ValueError("Need both classes")
    groups=np.unique(users);np.random.default_rng(42).shuffle(groups)
    fit=np.isin(users,groups[:max(1,int(.65*len(groups)))]);cal=~fit
    if not fit.any() or len(np.unique(y[cal]))!=2:raise ValueError("Insufficient validation calibration users")
    p=np.clip(pred,1e-6,1-1e-6)
    loss=-(y[:,None]*np.log(p)+(1-y[:,None])*np.log1p(-p))
    f=np.clip(full,1e-6,1-1e-6)
    ref=-(y*np.log(f)+(1-y)*np.log1p(-f))
    reg=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    reg.fit(X[fit],loss[fit]-ref[fit,None])
    expected=reg.predict(X[cal])
    costs=np.minimum(n[cal,None],np.asarray(BUDGETS)[None,:])/50.
    full_auc=float(roc_auc_score(y[cal],full[cal]))
    candidates=[]
    for lam in (0.,.0003,.001,.003,.01,.03,.1,.3,1.):
        a=np.argmin(expected+lam*costs,axis=1);ii=np.arange(len(a))
        candidates.append(dict(penalty=lam,auc=float(roc_auc_score(y[cal],pred[cal][ii,a])),
                               mean_k=float((50*costs[ii,a]).mean())))
    feasible=[v for v in candidates if v["auc"]>=full_auc-tolerance]
    chosen=min(feasible,key=lambda v:(v["mean_k"],-v["auc"])) if feasible else max(candidates,key=lambda v:(v["auc"],-v["mean_k"]))
    return {"regressor":reg,"penalty":chosen["penalty"],"budgets":BUDGETS,
            "calibration":{"full_auc":full_auc,"tolerance":tolerance,"feasible":bool(feasible),
                           "chosen":chosen,"candidates":candidates,"fit_requests":int(fit.sum()),
                           "calibration_requests":int(cal.sum())}}

def prepare(args):
    secret,_=credentials();device=baseline.get_device();model=load_model(args.checkpoint,device)
    vocab,vmap=baseline.load_video_vocab()
    val=baseline.KuaiRandDataset("validation","Full",vmap)
    test=baseline.KuaiRandDataset("test","Full",vmap)
    if set(val.evaluation_ids)&set(test.evaluation_ids):raise ValueError("Split overlap")
    compact=getattr(args,"compact",False)
    selector=calibrate(model,val,device,compact=compact)
    sha=_sha(args.checkpoint)
    emb=semantic_embeddings(model)
    sid=build(args.root,((str(e),h) for ds in (val,test) for e,h in zip(ds.evaluation_ids,ds.history_ids)),
              emb,secret,block=BLOCK,compact=compact,metadata={"checkpoint_sha256":sha,"vocab_sha256":_sha(baseline.VIDEO_VOCAB_FILE)})
    selector.update(snapshot_id=sid,checkpoint_sha256=sha)
    args.bundle.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.bundle.with_suffix(".tmp");joblib.dump(selector,tmp);tmp.replace(args.bundle)
    print(json.dumps({"snapshot_id":sid,"calibration":selector["calibration"]},indent=2),flush=True)

class Engine:
    def __init__(self,args):
        secret,_=credentials();self.device=baseline.get_device()
        self.bundle=joblib.load(args.bundle)
        sha=_sha(args.checkpoint)
        if self.bundle["checkpoint_sha256"]!=sha:raise ValueError("Bundle/checkpoint mismatch")
        self.store=Store(args.root,self.bundle["snapshot_id"],secret)
        if self.store.manifest["metadata"]["checkpoint_sha256"]!=sha:raise ValueError("Index/checkpoint mismatch")
        self.model=load_model(args.checkpoint,self.device)
        self.emb=semantic_embeddings(self.model)
        self.ue=self.model.user.weight.detach().cpu().numpy()
        self.lock=threading.Lock()
        self.is_m4=hasattr(self.model.item,"residual")
        reg=self.bundle["regressor"]
        scaler, ridge=reg.steps[0][1],reg.steps[1][1]
        self.selector_w=ridge.coef_/scaler.scale_[None,:]
        self.selector_b=ridge.intercept_-self.selector_w@scaler.mean_
        self.macs=0
        for layer in self.model.modules():
            if isinstance(layer,torch.nn.Linear):
                layer.register_forward_hook(self.count_linear)
    def count_linear(self,layer,inputs,out):
        self.macs+=out.numel()*layer.in_features
    def score(self,key,c,u,policy):
        start=perf_counter();stats=Counter();timing=Counter()
        if policy not in (POLICIES+("hybrid","adaptive_eager","adaptive_pipeline") if self.is_m4 else POLICIES) or not 0<c<len(self.emb) or u<0:raise ValueError("Invalid request")
        mode="indexed_exact" if policy=="adaptive_exact" else "indexed_bloom"
        def lookup(op,**kwargs):
            t=perf_counter();r,s=self.store.lookup(key,op,mode=mode,**kwargs)
            timing["storage_ms"]+=(perf_counter()-t)*1000;stats.update(s)
            if r is None:raise KeyError("History not found")
            return r
        semantic=policy in ("semantic50","adaptive","adaptive_exact","prune","merge","hybrid","adaptive_eager","adaptive_pipeline")
        meta=lookup("summaries" if semantic else "meta");n=meta["n"]
        t=perf_counter();q=self.emb[c];usr=self.ue[u%len(self.ue)]
        if policy=="full":k=n
        elif policy in ("fixed50","random50","semantic50"):k=min(n,50)
        else:
            f=features(n,meta["blocks"],q,usr)[None,:]
            regret=(self.selector_w@f[0]+self.selector_b) if self.is_m4 and policy!="adaptive_pipeline" else self.bundle["regressor"].predict(f)[0]
            costs=np.minimum(n,np.asarray(BUDGETS))/50.
            k=min(n,BUDGETS[int(np.argmin(regret+self.bundle["penalty"]*costs))])
        read_budget=k
        if policy in ("prune","merge","hybrid"):read_budget=min(n,max(k,2*k),512)
        if semantic:ranges=semantic_ranges(meta["blocks"],q,read_budget)
        elif policy=="random50":
            ranges=positions_to_ranges(random.Random(key).sample(range(n),k))
        else:ranges=[[n-k,n]]
        timing["selector_ms"]=(perf_counter()-t)*1000
        eager=policy=="adaptive_eager"
        if eager:
            all_tokens=lookup("tokens",ranges=[[0,n]])["tokens"] if n else []
            tokens=[all_tokens[j] for a,b in ranges for j in range(a,b)]
            if stats["token_rows"]!=n:raise RuntimeError("Eager read accounting failed")
        else:tokens=lookup("tokens",ranges=ranges)["tokens"] if read_budget else []
        if len(tokens)!=read_budget or (not eager and stats["token_rows"]!=read_budget):raise RuntimeError("Bounded retrieval invariant failed")
        if policy not in ("full","adaptive_eager") and stats["token_rows"]>512:raise RuntimeError("Online token budget exceeded")
        h=[v for _,v in tokens];d=self.device
        t=perf_counter()
        with self.lock,torch.inference_mode():
            self.macs=0
            C=torch.tensor([c],device=d);U=torch.tensor([u],device=d)
            H=torch.tensor([h or [0]],device=d);M=H.ne(0)
            op=policy if policy in ("prune","merge","hybrid") and k else "recent"
            z,_,count=self.model(C,U,H,M,op,max(k,1))
            prob=float(torch.sigmoid(z).cpu().item());used=int(count.cpu().item());macs=self.macs
            sync(d)
        timing["model_ms"]=(perf_counter()-t)*1000
        return dict(score=prob,history_length=n,budget=k,read_tokens=stats["token_rows"],model_tokens=used,
                    linear_macs=macs,storage=dict(stats),timing=dict(timing),server_ms=(perf_counter()-start)*1000)

class RecommendationHTTPServer(ThreadingHTTPServer):
    """Bounded local HTTP service with sufficient concurrent accept backlog."""
    request_queue_size=128
    daemon_threads=True
    block_on_close=False


def make_server(engine,token,port):
    class Handler(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"
        def log_message(self,*args):pass
        def reply(self,status,payload):
            b=json.dumps(payload,separators=(",",":"),allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization",""),"Bearer "+token)
        def do_GET(self):
            if not self.authorized():return self.reply(401,{"error":"unauthorized"})
            if self.path=="/ready":return self.reply(200,{"snapshot_id":engine.bundle["snapshot_id"],"status":"ready"})
            return self.reply(404,{"error":"not_found"})
        def do_POST(self):
            if not self.authorized():return self.reply(401,{"error":"unauthorized"})
            if self.path!="/recommend":return self.reply(404,{"error":"not_found"})
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=8192:raise ValueError()
                p=json.loads(self.rfile.read(size))
                if set(p)!={"history_key","candidate_id","user_id","policy"}:raise ValueError()
                if type(p["candidate_id"]) is not int or type(p["user_id"]) is not int or not isinstance(p["history_key"],str) or len(p["history_key"])>128:raise ValueError()
                out=engine.score(p["history_key"],p["candidate_id"],p["user_id"],p["policy"])
            except KeyError:return self.reply(404,{"error":"history_not_found"})
            except (ValueError,TypeError):return self.reply(400,{"error":"invalid_request"})
            except Exception:return self.reply(503,{"error":"scoring_unavailable"})
            return self.reply(200,out)
    return RecommendationHTTPServer(("127.0.0.1",port),Handler)

def events_only():
    _,vmap=baseline.load_video_vocab()
    split=pd.read_parquet(baseline.PROCESSED_DIR/"test.parquet",columns=["eval_idx"])
    ev=split.merge(baseline.load_evaluation_labels(),left_on="eval_idx",right_on="evaluation_id",validate="one_to_one",how="left")
    if ev.isna().any().any():raise ValueError("Missing evaluation event")
    ev["candidate"]=baseline.map_raw_ids_to_model_ids(ev.video_id,vmap,"test candidate")
    return ev

def gauc(y,u,p):
    total=weight=0
    for user in np.unique(u):
        ix=u==user
        if len(np.unique(y[ix]))==2:
            w=int(ix.sum());total+=w*roc_auc_score(y[ix],p[ix]);weight+=w
    return float(total/weight) if weight else None

def evaluate(args,engine):
    _,token=credentials();ev=events_only()
    if args.sample:ev=ev.sample(min(args.sample,len(ev)),random_state=42).reset_index(drop=True)
    y=ev.long_view.to_numpy(dtype=int);u=ev.user_id.to_numpy(dtype=int)
    if len(np.unique(y))!=2:raise ValueError("Sample needs both classes")
    server=make_server(engine,token,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=60)
    rows=[];scores={};trials=defaultdict(list)
    policies=list(POLICIES)+(["hybrid","adaptive_eager","adaptive_pipeline"] if engine.is_m4 else [])
    def request(row,policy):
        payload=json.dumps(dict(history_key=str(int(row.evaluation_id)),candidate_id=int(row.candidate),user_id=int(row.user_id),policy=policy))
        start=perf_counter()
        conn.request("POST","/recommend",payload,{"Authorization":"Bearer "+token,"Content-Type":"application/json"})
        resp=conn.getresponse();body=resp.read();elapsed=(perf_counter()-start)*1000
        if resp.status!=200:raise RuntimeError("Recommendation HTTP status "+str(resp.status))
        d=json.loads(body);d["client_ms"]=elapsed;d["response_body_bytes"]=len(body);return d
    try:
        for p in policies:request(next(ev.itertuples(index=False)),p)
        for rep in range(args.repeats):
            order=policies.copy();random.Random(42+rep).shuffle(order)
            for policy in order:
                t=perf_counter();cpu_start=process_time()
                out=[request(row,policy) for row in ev.itertuples(index=False)]
                wall=perf_counter()-t;cpu_seconds=process_time()-cpu_start
                pred=np.asarray([d["score"] for d in out])
                if not np.isfinite(pred).all():raise RuntimeError("Nonfinite score")
                if policy in scores and not np.allclose(scores[policy],pred,atol=1e-6,rtol=0):raise RuntimeError("Repeated prediction drift")
                scores[policy]=pred
                stats=Counter()
                for d in out:stats.update(d["storage"])
                metrics=dict(policy=policy,auc=float(roc_auc_score(y,pred)),gauc=gauc(y,u,pred),
                    logloss=float(log_loss(y,pred,labels=[0,1])),mean_k=float(np.mean([d["model_tokens"] for d in out])),
                    mean_read_tokens=float(np.mean([d["read_tokens"] for d in out])),
                    full_tokens=int(sum(d["history_length"] for d in out)),
                    mean_linear_macs=float(np.mean([d["linear_macs"] for d in out])),
                    p50_ms=float(np.quantile([d["client_ms"] for d in out],.5)),
                    p95_ms=float(np.quantile([d["client_ms"] for d in out],.95)),
                    p99_ms=float(np.quantile([d["client_ms"] for d in out],.99)),
                    mean_server_ms=float(np.mean([d["server_ms"] for d in out])),
                    mean_storage_ms=float(np.mean([d["timing"]["storage_ms"] for d in out])),
                    mean_selector_ms=float(np.mean([d["timing"]["selector_ms"] for d in out])),
                    mean_model_ms=float(np.mean([d["timing"]["model_ms"] for d in out])),
                    throughput_rps=len(out)/wall,cpu_seconds=cpu_seconds,process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024)),storage_totals=dict(stats),repeat=rep+1)
                trials[policy].append(metrics);print(json.dumps(metrics),flush=True)
        if not np.allclose(scores["adaptive"],scores["adaptive_exact"],atol=1e-7,rtol=0):raise RuntimeError("Bloom prediction parity failed")
        if engine.is_m4:
            for alias in ("adaptive_eager","adaptive_pipeline"):
                if not np.allclose(scores["adaptive"],scores[alias],atol=1e-7,rtol=0):raise RuntimeError("Optimization prediction parity failed: "+alias)
        from src.integration.evaluate_shared_variable_k import paired_ci
        for p in policies:
            a=trials[p];row=dict(a[0]);row.pop("repeat")
            for key in ("p50_ms","p95_ms","p99_ms","throughput_rps","mean_server_ms","mean_storage_ms","mean_selector_ms","mean_model_ms","cpu_seconds","process_peak_rss_bytes"):
                row[key]=float(np.median([v[key] for v in a]))
            row["delta_auc_vs_full"]=row["auc"]-a[0]["auc"] if p=="full" else row["auc"]-trials["full"][0]["auc"]
            row["delta_auc_ci95"]=paired_ci(y,u,scores[p],scores["full"],200,42)
            rows.append(row)
        for row in rows:
            row["pareto_quality_latency_read_tokens"]=not any(
                v["auc"]>=row["auc"] and v["p95_ms"]<=row["p95_ms"] and v["mean_read_tokens"]<=row["mean_read_tokens"] and
                (v["auc"]>row["auc"] or v["p95_ms"]<row["p95_ms"] or v["mean_read_tokens"]<row["mean_read_tokens"]) for v in rows)
        report=dict(status="PASS",scope="single_request_loopback_http_frozen_event_history",snapshot_id=engine.bundle["snapshot_id"],
                    requests=len(ev),repeats=args.repeats,calibration=engine.bundle["calibration"],rows=rows,trials=dict(trials),
                    accounting="linear MACs measured by layer hooks; logical SQLite result bytes, not physical disk I/O",
                    bloom_prediction_parity=True)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
        pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"),index=False)
        np.savez_compressed(args.output.with_suffix(".npz"),labels=y,users=u,evaluation_ids=ev.evaluation_id.to_numpy(),**scores)
        print(pd.DataFrame(rows)[["policy","auc","mean_k","mean_read_tokens","p95_ms","mean_linear_macs"]].to_string(index=False))
    finally:
        conn.close();server.shutdown();server.server_close();thread.join()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["all","prepare","evaluate","serve"])
    p.add_argument("--checkpoint",type=Path,default=ROOT/"models/m3_shared.pt")
    p.add_argument("--root",type=Path,default=ROOT/"data/processed/m3_snapshots")
    p.add_argument("--bundle",type=Path,default=ROOT/"models/m3_bundle.joblib")
    p.add_argument("--output",type=Path,default=ROOT/"reports/m3_product.json")
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--sample",type=int,default=0)
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--port",type=int,default=8777)
    p.add_argument("--reuse-checkpoint",action="store_true")
    args=p.parse_args();credentials()
    if args.repeats<1 or args.sample<0:raise ValueError("Invalid arguments")
    if args.command=="all":
        if not args.reuse_checkpoint:
            subprocess.run([sys.executable,"-m","src.integration.m3_train","--epochs",str(args.epochs),"--batch-size",str(args.batch_size),"--output",str(args.checkpoint)],check=True)
        prepare(args);evaluate(args,Engine(args))
    elif args.command=="prepare":prepare(args)
    elif args.command=="evaluate":evaluate(args,Engine(args))
    else:
        _,token=credentials();server=make_server(Engine(args),token,args.port)
        print("M3 recommendation API listening on 127.0.0.1:"+str(args.port),flush=True)
        try:server.serve_forever()
        finally:server.server_close()
if __name__=="__main__":main()
