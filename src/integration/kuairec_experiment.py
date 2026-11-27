"""Frozen KuaiRand protocol replay on KuaiRec. One shared M4 training run.
All action predictions persist individually, so evaluation restarts are cache-only.
Fit on validation users, freeze decisions before loading test labels.
Do not compare raw AUC across datasets as if tasks were identical."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.integration import m3_product as p,m5_selector as s
from src.integration.global_policy import KS,compile_policy,select
from src.integration.m6_matched_audit import mixture
from src.integration.m23_persistent_store import _sha

ACTIONS=tuple(a for a in s.ACTIONS if a[1] in KS)
NAMES=tuple(f"{m}_{k}" for m,k in ACTIONS)
COLS=np.array([s.ACTIONS.index(a) for a in ACTIONS])
RECENT=np.array([i for i,(m,k) in enumerate(ACTIONS) if m=="recent"])

def write(path,obj):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def save(path,**arrays):
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def cached_split(d,split,model,ds,device,batch,emb,ue):
    ids=np.asarray(ds.evaluation_ids);y=np.asarray(ds.labels);u=np.asarray(ds.user_ids)
    N=np.asarray(list(map(len,ds.history_ids)));c=np.asarray(ds.candidate_ids)
    def checked(path):
        z=read(path)
        if not np.array_equal(z["ids"],ids):raise ValueError("Cache ID mismatch: "+str(path))
        return z
    fp=d/f"{split}_features.npz";blocks=None
    def get_blocks():
        nonlocal blocks
        if blocks is None:blocks=[p.summaries(h,emb,True) for h in ds.history_ids]
        return blocks
    if fp.exists():
        z=checked(fp)
        if not all(np.array_equal(z[k],v) for k,v in (("N",N),("y",y),("u",u),("candidate",c))):raise ValueError("Feature alignment mismatch")
        X=z["X"];print("CACHE HIT",split,"features",flush=True)
    else:
        X=np.asarray([s.feature_vector(len(h),b,emb[q],ue[int(user)%len(ue)]) for h,b,q,user in zip(ds.history_ids,get_blocks(),c,u)])
        save(fp,ids=ids,X=X,N=N,y=y,u=u,candidate=c);print("CACHE SAVED",split,"features",flush=True)
    probs=[]
    for mode,k in ACTIONS:
        path=d/f"{split}_{mode}_{k}.npz"
        if path.exists():
            pred=checked(path)["pred"];print("CACHE HIT",split,mode,k,flush=True)
        else:
            if mode=="recent":h=[v[-k:] if k else [] for v in ds.history_ids]
            else:h=[[v[j] for a,b in p.semantic_ranges(bl,emb[q],k) for j in range(a,b)] for v,bl,q in zip(ds.history_ids,get_blocks(),c)]
            pred=p.predict_batch(model,ds,h,device,batch=batch)
            save(path,ids=ids,pred=pred);print("CACHE SAVED",split,mode,k,flush=True)
        if pred.shape!=(len(ids),) or not np.isfinite(pred).all() or np.any((pred<0)|(pred>1)):raise ValueError("Invalid predictions")
        probs.append(pred)
    return dict(ids=ids,y=y,u=u,N=N,X=X,P=np.column_stack(probs),candidate=c)

def resources(N,summary):
    return s.resource_table(N,selector=summary)[:,COLS,:]
def decision(regret,res,pref,recent=False):
    cost=s.cost_table(res,pref);obj=pref["quality_weight"]*regret/s.QUALITY_SCALE+pref["cost_weight"]*cost
    feasible=res[:,:,0]<=pref["max_read_tokens"]
    if pref["max_linear_macs"] is not None:feasible &= res[:,:,2]<=pref["max_linear_macs"]
    if pref["max_logical_bytes"] is not None:feasible &= res[:,:,1]<=pref["max_logical_bytes"]
    if recent:feasible &= np.array([m=="recent" for m,k in ACTIONS])[None,:]
    if not feasible.any(1).all():raise ValueError("No feasible action")
    obj=np.where(feasible,obj,np.inf);ties=obj<=obj.min(1,keepdims=True)+1e-12
    return np.argmin(np.where(ties,cost,np.inf),axis=1)
def choose_global(plan,ids):
    return np.array([NAMES.index(f"recent_{select(plan,str(int(x)))}") for x in ids])
def metrics(data,ix,res,pref):
    j=np.arange(len(ix));y=data["y"];z=data["P"][j,ix];r=res[j,ix]
    loss=s.loss_matrix(y,data["P"]);cost=s.cost_table(res,pref)[j,ix]
    return dict(auc=float(roc_auc_score(y,z)),auprc=float(average_precision_score(y,z)),gauc=p.gauc(y,data["u"],z),
        bce=float(log_loss(y,z,labels=[0,1])),mean_k=float(r[:,0].mean()),mean_bytes=float(r[:,1].mean()),mean_macs=float(r[:,2].mean()),
        mean_cost=float(cost.mean()),utility_loss=float(np.mean(pref["quality_weight"]*(loss[j,ix]-loss[:,-1])/s.QUALITY_SCALE+pref["cost_weight"]*cost)),
        actions=json.dumps({NAMES[i]:int(n) for i,n in zip(*np.unique(ix,return_counts=True))}))
def subset(data,mask):
    return {k:v[mask] for k,v in data.items()}
def features(X,name):
    if name=="length":return X[:,:1]
    if name=="embedding":return X[:,np.r_[0,10:X.shape[1]]]
    return X
def paired_auc_ci(y,u,a,b,repeats=500):
    groups,inv=np.unique(u,return_inverse=True);rng=np.random.default_rng(42);values=[]
    for _ in range(repeats):
        weights=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))[inv]
        if weights[y==0].sum()==0 or weights[y==1].sum()==0:continue
        values.append(roc_auc_score(y,a,sample_weight=weights)-roc_auc_score(y,b,sample_weight=weights))
    return [float(x) for x in np.quantile(values,[.025,.975])] if values else None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=Path("reports/kuairec_v1"))
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--inference-batch",type=int,default=32)
    a=ap.parse_args()
    if min(a.epochs,a.batch_size,a.inference_batch)<1:ap.error("positive arguments required")
    d=a.output.resolve();d.mkdir(parents=True,exist_ok=True);data=p.baseline.PROCESSED_DIR
    if data.name!="processed" or data.parent.name!="kuairec":raise ValueError("Set SHORTVIDEO_PROCESSED_DIR to data/kuairec/processed")
    if d.name!="kuairec_v1":raise ValueError("Use isolated reports/kuairec_v1")
    mf=data/"adapter_manifest.json";manifest=json.loads(mf.read_text())
    for name,h in manifest["artifacts"].items():
        if _sha(data/name)!=h:raise ValueError("Data artifact changed: "+name)
    contract=dict(adapter_sha256=_sha(mf),trainer_sha256=_sha(Path(__file__).with_name("m4_train.py")),
        ranker_sha256=_sha(Path(__file__).with_name("m4_ranker.py")),behavior_sha256=_sha(Path(__file__).with_name("m4_behavior.py")),
        epochs=a.epochs,batch_size=a.batch_size,seed=42,behavior=True,distill=False)
    cf=d/"training_contract.json"
    if cf.exists() and json.loads(cf.read_text())!=contract:raise ValueError("Training contract mismatch")
    if not cf.exists():write(cf,contract)
    ck=d/"behavior.pt";log=d/"behavior.training.json"
    if ck.exists() and log.exists():print("TRAINING CACHE HIT",flush=True)
    elif ck.exists() or log.exists():raise RuntimeError("Incomplete training artifacts; inspect before restarting. Do not delete caches automatically.")
    else:
        subprocess.run([sys.executable,"-m","src.integration.m4_train","--output",str(ck),"--behavior","--epochs",str(a.epochs),"--batch-size",str(a.batch_size),"--patience","3","--seed","42"],check=True)
    cache=d/"cache";cache.mkdir(exist_ok=True)
    cc=dict(training=contract,checkpoint_sha256=_sha(ck),actions=list(NAMES),feature_schema=1,selector_sha256=_sha(Path(__file__).with_name("m5_selector.py")),inference_batch=a.inference_batch)
    if (cache/"contract.json").exists() and json.loads((cache/"contract.json").read_text())!=cc:raise ValueError("Prediction cache contract mismatch")
    write(cache/"contract.json",cc)
    device=p.baseline.get_device();model=p.load_model(ck,device);emb=p.semantic_embeddings(model);ue=model.user.weight.detach().cpu().numpy()
    _,vmap=p.baseline.load_video_vocab()
    val=p.baseline.KuaiRandDataset("validation","Full",vmap)
    V=cached_split(cache,"validation",model,val,device,a.inference_batch,emb,ue)
    del val
    groups=np.unique(V["u"]);np.random.default_rng(42).shuffle(groups)
    mask=np.isin(V["u"],groups[:max(1,int(.65*len(groups)))]);fit=subset(V,mask);cal=subset(V,~mask)
    if len(np.unique(cal["y"]))!=2 or len(np.unique(fit["y"]))!=2:raise ValueError("Insufficient validation class coverage")
    loss=s.loss_matrix(fit["y"],fit["P"]);target=loss-loss[:,-1,None]
    rows=[]
    for i in RECENT:
        k=ACTIONS[i][1];n=np.minimum(fit["N"],k)
        rows.append(dict(k=k,mean_tokens=float(n.mean()),mean_bce=float(loss[:,i].mean()),mean_macs=float((s.HEAD_MACS+s.TOKEN_MACS*np.maximum(n,1)).mean())))
    table=dict(schema=1,checkpoint_sha256=_sha(ck),vocab_sha256=_sha(p.baseline.VIDEO_VOCAB_FILE),fit_requests=len(fit["ids"]),rows=rows)
    write(d/"calibration.json",table)
    models={}
    for name in ("length","embedding","full_recent","full_joint","extra_trees"):
        reg=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=40,max_features=.7,random_state=42,n_jobs=4) if name=="extra_trees" else make_pipeline(StandardScaler(),Ridge(alpha=10.))
        reg.fit(features(fit["X"],name),target);models[name]=reg
    # Freeze every global and validation-cost-matched comparator before test load.
    profiles=[(f"q{w:g}",dict(quality_weight=w),None) for w in s.PROFILES]+[(f"budget_{b}",dict(quality_weight=1.),b) for b in (5,10,20,30,50,100)]
    profiles += [("cap50",dict(quality_weight=.99,max_read_tokens=50,max_logical_bytes=800,max_linear_macs=s.HEAD_MACS+50*s.TOKEN_MACS),None)]
    plans={};matches={};valrows=[]
    lc=s.loss_matrix(cal["y"],cal["P"]);r0=resources(cal["N"],False);r1=resources(cal["N"],True)
    for rev,(name,opts,budget) in enumerate(profiles,1):
        plan=compile_policy(table,dict(revision=rev,preferences=opts,average_tokens=budget));plans[name]=plan;pref=plan["preferences"]
        ix=choose_global(plan,cal["ids"]);valrows.append(dict(method="global",profile=name,**metrics(cal,ix,r0,pref)))
        if not name.startswith("q"):continue
        for method,reg in models.items():
            full=method in ("full_recent","full_joint","extra_trees")
            # Exact q=0 shortcut used by deployed M5; do not charge unread summaries.
            rr=r1 if full and pref["quality_weight"]>0 else r0
            ix=decision(reg.predict(features(cal["X"],method)),rr,pref,method in ("length","embedding","full_recent"))
            m=metrics(cal,ix,rr,pref);valrows.append(dict(method=method,profile=name,**m))
            i,j,w=mixture(s.cost_table(r0,pref).mean(0),lc.mean(0),m["mean_cost"],RECENT)
            matches[method+"__"+name]=dict(i=int(i),j=int(j),prob_b=w,validation_cost=m["mean_cost"])
    write(d/"frozen_plans.json",dict(global_plans=plans,matched_mixtures=matches))
    pd.DataFrame(valrows).to_csv(d/"validation_comparison.csv",index=False)
    import joblib
    joblib.dump(dict(models=models,actions=ACTIONS,checkpoint_sha256=_sha(ck)),d/"selector_references.joblib")
    print("VALIDATION PLANS FROZEN",flush=True)
    test=p.baseline.KuaiRandDataset("test","Full",vmap)
    if set(V["ids"])&set(test.evaluation_ids):raise ValueError("Split overlap")
    T=cached_split(cache,"test",model,test,device,a.inference_batch,emb,ue)
    del model,test,emb,ue
    R0=resources(T["N"],False);R1=resources(T["N"],True);y=T["y"];j=np.arange(len(y))
    score=T["P"];testrows=[];decisions={};matched=[]
    for i,k in zip(RECENT,KS):
        ix=np.full(len(y),i);testrows.append(dict(method="fixed",profile=f"K{k}",**metrics(T,ix,R0,s.preference(.9))))
    for name,plan in plans.items():
        pref=plan["preferences"];ix=choose_global(plan,T["ids"]);decisions["global__"+name]=ix
        testrows.append(dict(method="global",profile=name,**metrics(T,ix,R0,pref)))
        if not name.startswith("q"):continue
        for method,reg in models.items():
            full=method in ("full_recent","full_joint","extra_trees")
            rr=R1 if full and pref["quality_weight"]>0 else R0
            ix=decision(reg.predict(features(T["X"],method)),rr,pref,method in ("length","embedding","full_recent"));key=method+"__"+name;decisions[key]=ix
            m=metrics(T,ix,rr,pref);testrows.append(dict(method=method,profile=name,**m))
            mix=matches[key];i,b,w=mix["i"],mix["j"],mix["prob_b"]
            # 20 independent fixed assignments; expected BCE and cost exact.
            aucs=[roc_auc_score(y,score[j,np.where(np.random.default_rng(2027+seed).random(len(y))<w,b,i)]) for seed in range(20)]
            lm=s.loss_matrix(y,score);ct=s.cost_table(R0,pref)
            ml=float(((1-w)*lm[:,i]+w*lm[:,b]).mean());mc=float(((1-w)*ct[:,i]+w*ct[:,b]).mean())
            matched.append(dict(method=method,profile=name,validation_budget=mix["validation_cost"],action_a=NAMES[i],action_b=NAMES[b],prob_b=w,
                adaptive_auc=m["auc"],mixture_auc_mean=float(np.mean(aucs)),mixture_auc_seed_sd=float(np.std(aucs)),adaptive_cost=m["mean_cost"],mixture_expected_cost=mc,adaptive_bce=m["bce"],mixture_expected_bce=ml,delta_auc=m["auc"]-float(np.mean(aucs)),delta_utility=pref["quality_weight"]*(m["bce"]-ml)/s.QUALITY_SCALE+pref["cost_weight"]*(m["mean_cost"]-mc)))
    tf=pd.DataFrame(testrows);tf.to_csv(d/"quality_cost.csv",index=False);pd.DataFrame(matched).to_csv(d/"matched_comparison.csv",index=False)
    save(d/"selected_actions.npz",ids=T["ids"],**decisions)
    comparisons={}
    for name,ix in (("fixed50_vs_full",np.full(len(y),NAMES.index("recent_50"))),("global_q90_vs_full",decisions["global__q0.9"]),("extra_trees_q90_vs_global",decisions["extra_trees__q0.9"])):
        ref=score[j,decisions["global__q0.9"]] if name=="extra_trees_q90_vs_global" else score[:,-1]
        pred=score[j,ix];comparisons[name]=dict(delta_auc=float(roc_auc_score(y,pred)-roc_auc_score(y,ref)),user_cluster_ci95=paired_auc_ci(y,T["u"],pred,ref))
    summary=dict(status="OFFLINE_PASS",dataset="KuaiRec 2.0 big matrix",checkpoint_sha256=_sha(ck),train_requests=manifest["splits"]["train"]["sampled"],validation_requests=len(V["ids"]),test_requests=len(y),positive_rate=float(y.mean()),fit_requests=len(fit["ids"]),calibration_requests=len(cal["ids"]),comparisons=comparisons,
        limitations=manifest["limitations"]+["Single trained checkpoint; no new specialist-model comparison.","M4 training protocol reused including variable history operators.","Original action grid only; no new model search.","Average precision is reported as AUPRC.","Expected cost excludes selector CPU and RPC overhead.","Mixtures matched on validation, not necessarily test; seed variation is not a confidence interval.","Checkpoint selection and calibration reuse validation; KuaiRec test not used for policy selection.","No serving measurements in this offline report."])
    write(d/"summary.json",summary)
    def md(df):
        return "| "+" | ".join(df.columns)+" |\n| "+" | ".join(["---"]*len(df.columns))+" |\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in df.itertuples(index=False,name=None))
    cols=["method","profile","auc","auprc","bce","mean_k","mean_bytes","mean_macs","mean_cost"]
    report="# KuaiRec cross-dataset validation\n\n"+md(tf[cols].round(6))+"\n\n## Matched-cost comparisons\n\n"+md(pd.DataFrame(matched)[["method","profile","adaptive_auc","mixture_auc_mean","adaptive_cost","mixture_expected_cost","delta_utility"]].round(6))+"\n\n## Limitations\n\n"+"\n".join("- "+v for v in summary["limitations"])+"\n"
    (d/"REPORT.md").write_text(report)
    print(tf[cols].round(6).to_string(index=False));print("OFFLINE_PASS",json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
