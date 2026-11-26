"""Compare minimal business-policy controllers against full context.
Fit/calibrate on disjoint validation users; reuse M5 cached test predictions.
Only original budgets. No recommender training. Reports exploratory."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,log_loss
from src.integration import m3_product as p,m5_selector as s,m6_cache as cache
from src.integration.m5_selector_experiment import Engine
from src.integration.m23_persistent_store import _sha

def resources(N,summary):
    return s.resource_table(N,selector=summary)[:,cache.COLS,:]

def decide(pred,res,pref,recent):
    costs=s.cost_table(res,pref)
    utility=pref["quality_weight"]*pred/s.QUALITY_SCALE+pref["cost_weight"]*costs
    feasible=res[:,:,0]<=pref["max_read_tokens"]
    if pref["max_linear_macs"] is not None:feasible &= res[:,:,2]<=pref["max_linear_macs"]
    if pref["max_logical_bytes"] is not None:feasible &= res[:,:,1]<=pref["max_logical_bytes"]
    if recent:feasible &= np.array([m=="recent" for m,k in cache.ACTIONS])[None,:]
    if not feasible.any(1).all():raise ValueError("No feasible action")
    utility=np.where(feasible,utility,np.inf)
    ties=utility<=utility.min(1,keepdims=True)+1e-12
    return np.argmin(np.where(ties,costs,np.inf),axis=1)

def features(X,name):
    if name=="length":return X[:,:1]
    if name=="embedding":return X[:,np.r_[0,10:X.shape[1]]]
    return X

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m4-dir",type=Path,default=Path("reports/m4_research_v3"))
    ap.add_argument("--m5-dir",type=Path,default=Path("reports/m5_release_v1"))
    ap.add_argument("--cache",type=Path,default=Path("reports/m6_cache_v1"))
    ap.add_argument("--output",type=Path,default=Path("reports/m6_selector_v1"))
    args=ap.parse_args();d=args.output;d.mkdir(parents=True,exist_ok=True)
    m4=args.m4_dir;m5=args.m5_dir
    ck=m4/(json.loads((m4/"summary.json").read_text())["selected_on_validation"]+".pt")
    engine=Engine(SimpleNamespace(checkpoint=ck,root=m4/"snapshots",bundle=m4/"int8.joblib",selector=m5/"selector.joblib"))
    V,T=cache.load_or_build(args.cache,engine,ck,m5)
    groups=np.unique(V["users"]);np.random.default_rng(42).shuffle(groups)
    fit=np.isin(V["users"],groups[:max(1,int(.65*len(groups)))]);cal=~fit
    if not fit.any() or len(np.unique(V["labels"][cal]))!=2:raise ValueError("Bad validation split")
    loss=s.loss_matrix(V["labels"],V["P"]);target=loss-loss[:,-1,None]
    methods=[("global",True,False),("length",True,False),("embedding",True,False),("full_recent",True,True),("full_joint",False,True)]
    plans=[(f"q{w:g}",s.preference(w)) for w in s.PROFILES]+[("q0.9_cap50",s.preference(.9,max_read_tokens=50)),("q0.9_io",s.preference(.9,cost_weights=(0,1,0))),("q0.9_compute",s.preference(.9,cost_weights=(0,0,1)))]
    models={};validation=[];test=[];frozen={}
    def evaluate(data,idx,ix,res,pref):
        j=np.arange(len(ix));P=data["P"][idx];y=data["labels"][idx];u=data["users"][idx];z=P[j,ix];r=res[j,ix]
        cost=s.cost_table(res,pref)[j,ix];loss=s.loss_matrix(y,P)
        regret=loss[j,ix]-loss[:,-1]
        return dict(auc=float(roc_auc_score(y,z)),gauc=p.gauc(y,u,z),logloss=float(log_loss(y,z,labels=[0,1])),
            mean_k=float(r[:,0].mean()),mean_bytes=float(r[:,1].mean()),mean_macs=float(r[:,2].mean()),
            mean_cost=float(cost.mean()),mean_utility_loss=float(np.mean(pref["quality_weight"]*regret/s.QUALITY_SCALE+pref["cost_weight"]*cost)),
            actions=json.dumps({cache.NAMES[i]:int(n) for i,n in zip(*np.unique(ix,return_counts=True))},sort_keys=True))
    # Model/feature ablation with identical Ridge family. Existing ExtraTrees is a separate reference.
    for name,recent,summary in methods:
        rv=resources(V["N"][cal],summary);rt=resources(T["N"],summary)
        if name=="global":
            mean=target[fit].mean(0);pv=np.broadcast_to(mean,rv.shape[:2]);pt=np.broadcast_to(mean,rt.shape[:2]);models[name]=mean
        else:
            reg=make_pipeline(StandardScaler(),Ridge(alpha=10.))
            reg.fit(features(V["X"][fit],name),target[fit]);models[name]=reg
            pv=reg.predict(features(V["X"][cal],name));pt=reg.predict(features(T["X"],name))
        for profile,pref in plans:
            # Global selects one nominal action at policy update using aggregate validation resource costs.
            if name=="global":
                a=decide(mean[None,:],rv.max(0,keepdims=True),pref,True)[0] if any(pref[k] is not None for k in ("max_logical_bytes","max_linear_macs")) else decide(mean[None,:],rv.mean(0,keepdims=True),pref,True)[0]
                # Enforce nominal token cap, not just mean cost.
                if cache.ACTIONS[a][1]>pref["max_read_tokens"]:
                    capres=rv.mean(0,keepdims=True).copy();capres[0,:,0]=[k for m,k in cache.ACTIONS]
                    a=decide(mean[None,:],capres,pref,True)[0]
                iv=np.full(len(pv),a);it=np.full(len(pt),a)
            else:iv=decide(pv,rv,pref,recent);it=decide(pt,rt,pref,recent)
            validation.append(dict(method=name,profile=profile,**evaluate(V,cal,iv,rv,pref)))
            frozen[(name,profile)]=(it,rt,pref)
    # Existing M5 model is fixed, not re-selected on test.
    name="m5_extra_trees";rv=resources(V["N"][cal],True);rt=resources(T["N"],True)
    pv=engine.fast_predictor.predict(V["X"][cal])[:,cache.COLS]
    pt=engine.fast_predictor.predict(T["X"])[:,cache.COLS]
    for profile,pref in plans:
        iv=decide(pv,rv,pref,False);it=decide(pt,rt,pref,False)
        validation.append(dict(method=name,profile=profile,**evaluate(V,cal,iv,rv,pref)))
        frozen[(name,profile)]=(it,rt,pref)
    vf=pd.DataFrame(validation);vf.to_csv(d/"validation_comparison.csv",index=False)
    # Choose by mean validation utility over profiles, never by test.
    ranking=vf.groupby("method").mean_utility_loss.mean().sort_values()
    winner=ranking.index[0]
    (d/"frozen_selection.json").write_text(json.dumps(dict(winner=winner,validation_utility=ranking.to_dict(),profiles=[n for n,_ in plans]),indent=2))
    for (name,profile),(ix,res,pref) in frozen.items():
        test.append(dict(method=name,profile=profile,**evaluate(T,slice(None),ix,res,pref)))
    tf=pd.DataFrame(test);tf.to_csv(d/"test_comparison.csv",index=False)
    joblib.dump(dict(models=models,actions=cache.ACTIONS,checkpoint_sha256=_sha(ck),snapshot_id=engine.bundle["snapshot_id"],selected_on_validation=winner),d/"selector_alternatives.joblib")
    np.savez_compressed(d/"selected_actions.npz",evaluation_ids=T["ids"],**{name+"__"+profile:ix for (name,profile),(ix,_,_) in frozen.items()})
    report=dict(status="EXECUTION_PASS",selected_on_validation=winner,validation_utility=ranking.to_dict(),test_requests=len(T["ids"]),
        caveats=["Exploratory reused test.","Resource proxies include summary payloads but exclude selector CPU and RPC overhead.","Same Ridge family across contextual feature ablations; existing M5 tree is separate reference.","Global uses cohort mean utility and nominal cap; no per-request learned selection.","No serving implementation change or new latency benchmark in this experiment."])
    (d/"summary.json").write_text(json.dumps(report,indent=2))
    cols=["method","profile","auc","mean_k","mean_bytes","mean_macs","mean_cost","mean_utility_loss"]
    print("VALIDATION RANKING",ranking.to_string(),flush=True)
    print(tf[cols].round(6).to_string(index=False));print("REPORT:",d)
if __name__=="__main__":main()
