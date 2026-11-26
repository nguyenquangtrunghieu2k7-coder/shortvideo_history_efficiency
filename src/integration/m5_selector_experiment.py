"""Train on validation only and evaluate preference policies on held-out requests.
Reuse frozen M4 checkpoint and signed INT8 index. No recommender retraining.
The same decision function serves HTTP requests and offline counterfactual replay."""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import hashlib
import http.client
import json
import threading
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,log_loss
from src.integration import m3_product as p
from src.integration import m5_selector as s
from src.integration.m23_persistent_store import _sha
from src.integration.evaluate_shared_variable_k import paired_ci

def arrays(model,ds,device,emb,ue):
    blocks=[p.summaries(h,emb,True) for h in ds.history_ids]
    X=np.asarray([s.feature_vector(len(h),b,emb[c],ue[u%len(ue)]) for h,b,c,u in zip(ds.history_ids,blocks,ds.candidate_ids,ds.user_ids)])
    probs=[]
    for mode,k in s.ACTIONS:
        if mode=="recent":hist=[h[-k:] if k else [] for h in ds.history_ids]
        else:hist=[[h[j] for a,b in p.semantic_ranges(bl,emb[c],k) for j in range(a,b)] for h,bl,c in zip(ds.history_ids,blocks,ds.candidate_ids)]
        probs.append(p.predict_batch(model,ds,hist,device))
        print("OFFLINE ACTION",mode,k,flush=True)
    return X,np.column_stack(probs),np.asarray(list(map(len,ds.history_ids)))

def pick_fixed(probs,y,res,target,pref):
    costs=s.cost_table(res,pref).mean(0)
    caps=(res[:,:,0]<=pref["max_read_tokens"]).all(0)&s.grid_mask(pref["grid"])
    if pref["max_logical_bytes"] is not None:caps &= (res[:,:,1]<=pref["max_logical_bytes"]).all(0)
    if pref["max_linear_macs"] is not None:caps &= (res[:,:,2]<=pref["max_linear_macs"]).all(0)
    valid=np.flatnonzero(caps)
    if not len(valid):raise ValueError("No fixed action fits cap")
    # Validation-selected best fixed action at or below adaptive mean resource cost.
    eligible=[j for j in valid if costs[j]<=target+1e-10]
    if not eligible:eligible=[min(valid,key=lambda j:costs[j])]
    return int(max(eligible,key=lambda j:(roc_auc_score(y,probs[:,j]),-costs[j],-j)))

def metric(name,ix,prob,y,users,n,res,pref,ci_reps,reference):
    j=np.arange(len(ix));z=prob[j,ix];r=res[j,ix]
    counts=Counter(s.ACTION_NAMES[i] for i in ix)
    return dict(scenario=name,auc=float(roc_auc_score(y,z)),gauc=p.gauc(y,users,z),logloss=float(log_loss(y,z,labels=[0,1])),
        mean_read_tokens=float(r[:,0].mean()),mean_logical_bytes=float(r[:,1].mean()),mean_linear_macs=float(r[:,2].mean()),
        mean_normalized_cost=float(s.cost_table(res,pref)[j,ix].mean()),action_histogram=dict(counts),
        effective_k_histogram={str(k):int(v) for k,v in Counter(r[:,0]).items()},
        delta_auc_vs_full=float(roc_auc_score(y,z)-roc_auc_score(y,reference)),
        ci95_vs_full=paired_ci(y,users,z,reference,ci_reps,42))

class Engine(p.Engine):
    def __init__(self,args):
        super().__init__(args)
        self.policy_model=joblib.load(args.selector)
        from src.integration.m5_fast_predictor import FastPredictor
        self.fast_predictor=FastPredictor(self.policy_model["regressor"])
        self.fast_enabled=True
        self.fast_path_enabled=True
        if self.store.manifest.get("centroid_encoding")!="int8_unit" or self.policy_model["actions"]!=s.ACTIONS or self.policy_model["checkpoint_sha256"]!=_sha(args.checkpoint) or self.policy_model["snapshot_id"]!=self.bundle["snapshot_id"]:
            raise ValueError("Selector binding mismatch")
    def score(self,key,c,u,policy):
        if getattr(self,"business_policy",None) is not None:
            if policy!="preference":raise ValueError("Managed service accepts preference policy only")
            doc,error=self.business_policy.get()
            out=self._score(key,c,u,policy,doc["preferences"])
            out.update(policy_revision=doc["revision"],policy_reload_error=error is not None)
            return out
        return self.score_with_preference(key,c,u,policy,{})
    def score_with_preference(self,key,c,u,policy,options):
        if getattr(self,"business_policy",None) is not None:raise ValueError("Caller overrides disabled in managed service")
        return self._score(key,c,u,policy,options)
    def _score(self,key,c,u,policy,options):
        if policy!="preference" and policy not in s.ACTION_NAMES:raise ValueError("Unknown M5 policy")
        if not 0<c<len(self.emb) or u<0:raise ValueError("Invalid request")
        if policy!="preference" and options:raise ValueError("Preferences only apply to adaptive policy")
        pref=s.preference(**options)
        t=perf_counter();stats=Counter();timing=Counter()
        def read(op,**kw):
            start=perf_counter();result,st=self.store.lookup(key,op,mode="indexed_exact",**kw)
            stats.update(st);timing["storage_ms"]+=(perf_counter()-start)*1000
            if result is None:raise KeyError("History not found")
            return result
        adaptive=policy=="preference"
        fixed_i=None if adaptive else s.ACTION_NAMES.index(policy)
        shortcut=adaptive and self.fast_path_enabled and (pref["quality_weight"]==0 or pref["max_read_tokens"]==0)
        if adaptive and self.fast_path_enabled and not shortcut and pref["max_logical_bytes"] is not None:
            n=read("meta")["n"]
            if ((n+31)//32)*(16+self.model.d)>pref["max_logical_bytes"]:
                raise ValueError("Summary retrieval exceeds logical-byte cap")
        needs_summary=(adaptive and not shortcut) or (not adaptive and s.ACTIONS[fixed_i][0]=="semantic")
        meta=read("summaries" if needs_summary else "meta")
        n=meta["n"];blocks=meta.get("blocks",[]);q=self.emb[c]
        start=perf_counter()
        resources=s.resource_table([n],self.model.d,True,adaptive and not shortcut)
        if shortcut:
            ix=0;predicted_regret=None
            if pref["max_linear_macs"] is not None and resources[0,0,2]>pref["max_linear_macs"]:raise ValueError("No feasible zero-history action")
        elif adaptive:
            x=s.feature_vector(n,blocks,q,self.ue[u%len(self.ue)])[None,:]
            regret=(self.fast_predictor if self.fast_enabled else self.policy_model["regressor"]).predict(x)
            ix=int(s.choose(regret,resources,pref)[0])
            predicted_regret=float(regret[0,ix])
        else:ix=fixed_i;predicted_regret=None
        mode,k=s.ACTIONS[ix];k=min(k,n)
        if adaptive and pref["max_logical_bytes"] is not None and resources[0,ix,1]>pref["max_logical_bytes"]:raise ValueError("Logical-byte cap exceeded")
        ranges=p.semantic_ranges(blocks,q,k) if mode=="semantic" else [[n-k,n]]
        timing["selector_ms"]=(perf_counter()-start)*1000
        tokens=read("tokens",ranges=ranges)["tokens"] if k else []
        if len(tokens)!=k or stats["token_rows"]!=k:raise RuntimeError("Bounded retrieval failed")
        start=perf_counter()
        with self.lock,torch.inference_mode():
            self.macs=0;d=self.device
            h=torch.tensor([[v for _,v in tokens] or [0]],device=d)
            z,_,count=self.model(torch.tensor([c],device=d),torch.tensor([u],device=d),h,h.ne(0),"recent",max(k,1))
            prob=float(torch.sigmoid(z).cpu().item());used=int(count.cpu().item());macs=self.macs;p.sync(d)
        timing["model_ms"]=(perf_counter()-start)*1000
        expected=resources[0,ix]
        if (k,stats["logical_payload_bytes"],macs)!=tuple(expected):raise RuntimeError("Resource accounting mismatch")
        return dict(score=prob,action=s.ACTION_NAMES[ix],read_tokens=k,model_tokens=used,linear_macs=macs,
            estimated_regret=predicted_regret,preferences=pref,storage=dict(stats),timing=dict(timing),server_ms=(perf_counter()-t)*1000)

def http_check(args,engine,ds,probs,decisions,preferences):
    _,token=p.credentials();server=p.make_server(engine,token,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=60)
    ix=np.sort(np.random.default_rng(42).choice(len(ds),min(args.http_sample,len(ds)),replace=False))
    rows=[]
    try:
        for name,actions in decisions.items():
            pref=preferences[name];times=[];reads=[];errors=[];macs=[];bytes_read=[]
            for i in ix:
                req=dict(history_key=str(ds.evaluation_ids[i]),candidate_id=int(ds.candidate_ids[i]),user_id=int(ds.user_ids[i]),
                         policy="preference" if pref is not None else name)
                if pref is not None:req["preferences"]=pref
                t=perf_counter();conn.request("POST","/recommend",json.dumps(req),{"Authorization":"Bearer "+token,"Content-Type":"application/json"})
                response=conn.getresponse();body=response.read();times.append((perf_counter()-t)*1000)
                if response.status!=200:raise RuntimeError(f"HTTP {response.status}: {body[:200]}")
                out=json.loads(body);j=int(actions[i]);err=abs(out["score"]-probs[i,j])
                if out["action"]!=s.ACTION_NAMES[j] or err>2e-5:raise RuntimeError(f"Offline/HTTP parity failed {name} {out['action']} {s.ACTION_NAMES[j]} {err}")
                errors.append(err);reads.append(out["read_tokens"]);macs.append(out["linear_macs"]);bytes_read.append(out["storage"]["logical_payload_bytes"])
            rows.append(dict(scenario=name,requests=len(ix),p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
                p99_ms=float(np.quantile(times,.99)),mean_read_tokens=float(np.mean(reads)),mean_linear_macs=float(np.mean(macs)),
                mean_logical_bytes=float(np.mean(bytes_read)),max_prediction_delta=float(max(errors))))
            print("HTTP",rows[-1],flush=True)
    finally:conn.close();server.shutdown();server.server_close();thread.join()
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-dir",type=Path,default=p.ROOT/"reports/m4_research_v3")
    parser.add_argument("--output",type=Path,default=p.ROOT/"reports/m5_selector_v1")
    parser.add_argument("--http-sample",type=int,default=256)
    parser.add_argument("--bootstrap",type=int,default=1000)
    parser.add_argument("--serve",action="store_true")
    parser.add_argument("--port",type=int,default=8778)
    parser.add_argument("--policy-file",type=Path)
    parser.add_argument("--bind",default="127.0.0.1")
    args=parser.parse_args()
    if min(args.http_sample,args.bootstrap)<1:parser.error("Invalid sample/bootstrap")
    d=args.output.resolve();d.mkdir(parents=True,exist_ok=True)
    m4=args.m4_dir.resolve();summary=json.loads((m4/"summary.json").read_text())
    ck=m4/(summary["selected_on_validation"]+".pt")
    args.checkpoint=ck;args.root=m4/"snapshots";args.bundle=m4/"int8.joblib";args.selector=d/"selector.joblib"
    if args.serve:
        from src.integration.m5_policy import PolicyStore
        _,token=p.credentials();engine=Engine(args)
        if args.policy_file is None:parser.error("--serve requires --policy-file")
        engine.business_policy=PolicyStore(args.policy_file)
        server=p.make_server(engine,token,args.port,host=args.bind)
        print(f"M5 preference-aware service listening on {args.bind}:{args.port}",flush=True)
        try:server.serve_forever()
        finally:server.server_close()
        return
    device=p.baseline.get_device();model=p.load_model(ck,device)
    if not hasattr(model.item,"residual") or model.d!=32:raise ValueError("Expected M4 ranker")
    emb=p.semantic_embeddings(model);ue=model.user.weight.detach().cpu().numpy()
    _,vmap=p.baseline.load_video_vocab()
    val=p.baseline.KuaiRandDataset("validation","Full",vmap);test=p.baseline.KuaiRandDataset("test","Full",vmap)
    if set(val.evaluation_ids)&set(test.evaluation_ids):raise ValueError("Split overlap")
    X,P,N=arrays(model,val,device,emb,ue);y=np.asarray(val.labels);users=np.asarray(val.user_ids)
    # Disjoint validation users for fitting and selection, no test use.
    groups=np.unique(users);np.random.default_rng(42).shuffle(groups)
    fit=np.isin(users,groups[:max(1,int(.65*len(groups)))]);cal=~fit
    if not fit.any() or len(np.unique(y[cal]))!=2:raise ValueError("Insufficient validation users")
    loss=s.loss_matrix(y,P);target=loss-loss[:,-1,None]
    resources=s.resource_table(N)
    models={"ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.)),
            "extra_trees":ExtraTreesRegressor(n_estimators=96,min_samples_leaf=40,max_features=.7,random_state=42,n_jobs=4)}
    validation=[];cal_pred={}
    # Select estimator by realized scalarized utility on calibration across predeclared weights.
    for name,reg in models.items():
        reg.fit(X[fit],target[fit]);pred=reg.predict(X[cal]);cal_pred[name]=pred
        realized=[]
        for w in s.PROFILES:
            pref=s.preference(w);ix=s.choose(pred,resources[cal],pref);j=np.arange(cal.sum())
            utility=w*target[cal][j,ix]/s.QUALITY_SCALE+(1-w)*s.cost_table(resources[cal],pref)[j,ix]
            realized.append(float(utility.mean()))
        validation.append(dict(model=name,mean_realized_utility=float(np.mean(realized)),mean_regret_rmse=float(np.sqrt(np.mean((pred-target[cal])**2)))))
    winner=min(validation,key=lambda v:v["mean_realized_utility"])["model"]
    reg=models[winner];predcal=cal_pred[winner]
    mean_regret=target[fit].mean(0)
    bundle=p.joblib.load(args.bundle)
    joblib.dump(dict(regressor=reg,checkpoint_sha256=_sha(ck),snapshot_id=bundle["snapshot_id"],actions=s.ACTIONS,
        quality_scale=s.QUALITY_SCALE,selected_model=winner,fit_requests=int(fit.sum()),calibration_requests=int(cal.sum())),args.selector)
    pd.DataFrame(validation).to_csv(d/"selector_model_ablation.csv",index=False)
    # Lock estimator and all fixed comparison actions using validation before test predictions.
    plans=[]
    for w in s.PROFILES:
        pref=s.preference(w);a=s.choose(predcal,resources[cal],pref);j=np.arange(cal.sum())
        target_cost=float(s.cost_table(resources[cal],pref)[j,a].mean())
        fixed=pick_fixed(P[cal],y[cal],s.resource_table(N[cal],selector=False),target_cost,pref)
        plans.append((f"q{w:g}",pref,fixed))
    for w,cap in ((.9,50),(.99,128),(1.,50)):
        pref=s.preference(w,max_read_tokens=cap);a=s.choose(predcal,resources[cal],pref);j=np.arange(cal.sum())
        target_cost=float(s.cost_table(resources[cal],pref)[j,a].mean())
        fixed=pick_fixed(P[cal],y[cal],s.resource_table(N[cal],selector=False),target_cost,pref)
        plans.append((f"q{w:g}_cap{cap}",pref,fixed))
    for name,pref in [
        ("q0.9_io",s.preference(.9,cost_weights=(0,1,0))),
        ("q0.9_compute",s.preference(.9,cost_weights=(0,0,1))),
        ("q0.99_mac_cap50",s.preference(.99,max_linear_macs=s.HEAD_MACS+s.TOKEN_MACS*50))]:
        a=s.choose(predcal,resources[cal],pref);j=np.arange(cal.sum())
        target_cost=float(s.cost_table(resources[cal],pref)[j,a].mean())
        fixed=pick_fixed(P[cal],y[cal],s.resource_table(N[cal],selector=False),target_cost,pref)
        plans.append((name,pref,fixed))
    for w in s.PROFILES:
        pref=s.preference(w,grid="extended");a=s.choose(predcal,resources[cal],pref);j=np.arange(cal.sum())
        target_cost=float(s.cost_table(resources[cal],pref)[j,a].mean())
        fixed=pick_fixed(P[cal],y[cal],s.resource_table(N[cal],selector=False),target_cost,pref)
        plans.append((f"extended_q{w:g}",pref,fixed))
    (d/"frozen_plan.json").write_text(json.dumps(dict(selected_model=winner,profiles=[dict(name=name,preferences=pref,fixed_comparator=s.ACTION_NAMES[fixed]) for name,pref,fixed in plans]),indent=2))
    Xt,Pt,Nt=arrays(model,test,device,emb,ue)
    yt=np.asarray(test.labels);ut=np.asarray(test.user_ids);rt=s.resource_table(Nt);rf=s.resource_table(Nt,selector=False)
    estimate=reg.predict(Xt);reference=Pt[:,-1];rows=[];hist=[];http_actions={};http_prefs={}
    default=s.preference(.9)
    for j,name in enumerate(s.ACTION_NAMES):
        ix=np.full(len(yt),j)
        rows.append(metric(name,ix,Pt,yt,ut,Nt,rf,default,args.bootstrap,reference))
        if name in ("recent_0","recent_5","recent_16","semantic_16","recent_50","recent_2048"):
            http_actions[name]=ix;http_prefs[name]=None
    comparisons=[];scores={}
    for name,pref,fixed in plans:
        ix=s.choose(estimate,rt,pref);neutral=s.choose(np.broadcast_to(mean_regret,estimate.shape),rt,pref);fi=np.full(len(yt),fixed)
        ar=metric(name,ix,Pt,yt,ut,Nt,rt,pref,args.bootstrap,reference)
        nr=metric(name+"_context_free",neutral,Pt,yt,ut,Nt,rt,pref,args.bootstrap,reference)
        fr=metric(name+"_fixed",fi,Pt,yt,ut,Nt,rf,pref,args.bootstrap,reference)
        for row in (ar,nr,fr):row["quality_weight"]=pref["quality_weight"];row["cost_weight"]=pref["cost_weight"];row["max_read_tokens"]=pref["max_read_tokens"];row["grid"]=pref["grid"]
        rows.extend((ar,nr,fr));j=np.arange(len(yt))
        scores[name]=Pt[j,ix]
        comparisons.append(dict(profile=name,fixed_action=s.ACTION_NAMES[fixed],adaptive_auc=ar["auc"],fixed_auc=fr["auc"],
            context_free_auc=nr["auc"],delta_auc_vs_fixed=ar["auc"]-fr["auc"],ci95_vs_fixed=paired_ci(yt,ut,Pt[j,ix],Pt[j,fi],args.bootstrap,42),
            ci95_vs_context_free=paired_ci(yt,ut,Pt[j,ix],Pt[j,neutral],args.bootstrap,42),
            adaptive_cost=ar["mean_normalized_cost"],fixed_cost=fr["mean_normalized_cost"],context_free_cost=nr["mean_normalized_cost"],
            adaptive_mean_k=ar["mean_read_tokens"],fixed_mean_k=fr["mean_read_tokens"],context_free_mean_k=nr["mean_read_tokens"]))
        for act,count in ar["action_histogram"].items():hist.append(dict(profile=name,action=act,requests=count))
        if name in ("q0","q0.5","q0.9","q0.99","q1","q0.9_cap50","q0.9_io","q0.9_compute","q0.99_mac_cap50","extended_q0.9","extended_q0.99"):
            http_actions[name]=ix;http_prefs[name]=pref
    pd.DataFrame(rows).to_csv(d/"quality_cost_frontier.csv",index=False)
    pd.DataFrame(comparisons).to_csv(d/"matched_fixed_comparison.csv",index=False)
    pd.DataFrame(hist).to_csv(d/"action_distribution.csv",index=False)
    np.savez_compressed(d/"predictions.npz",evaluation_ids=np.asarray(test.evaluation_ids),labels=yt,users=ut,action_predictions=Pt,**scores)
    # Per-request monotonic weighted cost as business quality weight decreases.
    chosen=[s.choose(estimate,rt,s.preference(w)) for w in s.PROFILES]
    costs=np.column_stack([s.cost_table(rt,s.preference(w))[np.arange(len(yt)),a] for w,a in zip(s.PROFILES,chosen)])
    if np.any(np.diff(costs,axis=1)<-1e-8):raise RuntimeError("Preference monotonicity failed")
    for grid in s.GRIDS:
        chosen=[s.choose(estimate,rt,s.preference(w,grid=grid)) for w in s.PROFILES]
        costs=np.column_stack([s.cost_table(rt,s.preference(w,grid=grid))[np.arange(len(yt)),a] for w,a in zip(s.PROFILES,chosen)])
        if np.any(np.diff(costs,axis=1)<-1e-8):raise RuntimeError("Grid preference monotonicity failed: "+grid)
    del model,emb,ue
    engine=Engine(args);http=http_check(args,engine,test,Pt,http_actions,http_prefs)
    pd.DataFrame(http).to_csv(d/"http_validation.csv",index=False)
    report=dict(original_budgets=s.ORIGINAL_BUDGETS,extended_budgets=tuple(sorted(set(s.ORIGINAL_BUDGETS+s.EXPLORATORY_BUDGETS))),status="EXECUTION_PASS",selected_model=winner,requests=len(yt),fit_requests=int(fit.sum()),calibration_requests=int(cal.sum()),
        checkpoint_sha256=_sha(ck),snapshot_id=bundle["snapshot_id"],quality_scale=s.QUALITY_SCALE,actions=s.ACTION_NAMES,
        validation_models=validation,profiles=comparisons,monotonic_cost=True,http_prediction_parity=True,
        caveats=["Test previously inspected: exploratory only.","Expected BCE regret is a surrogate, not per-request AUC.",
            "Resource cost includes read tokens, logical bytes and model linear MACs; not physical IO, complete FLOPs or cloud billing.",
            "Fixed comparison is validation-selected at or below adaptive weighted cost, not exactly cost-matched on test.",
            "HTTP subset is local single-worker; quality on full test is offline action replay.",
            "Selector fit and model checkpoint selection reuse validation; independent holdout required."])
    (d/"summary.json").write_text(json.dumps(report,indent=2))
    def md(df):
        return "| "+" | ".join(df.columns)+" |\n| "+" | ".join(["---"]*len(df.columns))+" |\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in df.itertuples(index=False,name=None))
    cols=["profile","fixed_action","adaptive_auc","fixed_auc","context_free_auc","adaptive_mean_k","fixed_mean_k","adaptive_cost","fixed_cost"]
    text="# Preference-conditioned adaptive selector\n\nSelected on validation: **"+winner+"**.\n\n"+md(pd.DataFrame(comparisons)[cols].round(6))+"\n\n## HTTP parity and resource measurements\n\n"+md(pd.DataFrame(http).round(6))+"\n\n## Limitations\n\n"+"\n".join("- "+v for v in report["caveats"])+"\n"
    (d/"REPORT.md").write_text(text)
    print(text,flush=True)

if __name__=="__main__":main()
