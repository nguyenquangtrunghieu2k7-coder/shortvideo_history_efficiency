"""Post-hoc probability calibration sensitivity, cache-only.
Calibration sensitivity was motivated by the previously inspected KuaiRec results.
Treat this as exploratory, not independent confirmation.
One positive-slope logistic mapping, fitted across all validation-fit actions,
keeps ranker scores ordered under a common monotone transform. No ranker
inference, no checkpoint modifications, no test-based fitting or selection.
Primary raw results remain unchanged."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import log_loss,roc_auc_score
from src.integration import m5_selector as s
from src.integration.kuairec_experiment import ACTIONS,NAMES,RECENT,read,write,resources,metrics,subset,decision,choose_global,paired_auc_ci
from src.integration.global_policy import KS,compile_policy
from src.integration.m6_matched_audit import mixture
from src.integration.m23_persistent_store import _sha

def load(cache,split):
    D=read(cache/f"{split}_features.npz");pred=[]
    for name in NAMES:
        z=read(cache/f"{split}_{name}.npz")
        if not np.array_equal(z["ids"],D["ids"]):raise ValueError("Cache ID mismatch")
        pred.append(z["pred"])
    D["P"]=np.column_stack(pred)
    if not np.isfinite(D["P"]).all() or np.any((D["P"]<0)|(D["P"]>1)):raise ValueError("Bad probability cache")
    return D

def transform(P,a,b):return expit(a*logit(np.clip(P,1e-7,1-1e-7))+b)
def fit_map(P,y):
    x=logit(np.clip(P,1e-7,1-1e-7)).ravel();y=np.repeat(y,P.shape[1])
    def fun(v):
        z=v[0]*x+v[1];err=expit(z)-y
        return float(np.mean(np.logaddexp(0,z)-y*z)),np.array([np.mean(err*x),np.mean(err)])
    sol=minimize(fun,[1.,0.],jac=True,method="L-BFGS-B",bounds=[(1e-4,None),(None,None)])
    if not sol.success or not np.isfinite(sol.x).all():raise RuntimeError("Calibration failed: "+str(sol.message))
    return map(float,sol.x)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",choices=["kuairec","vklsvd","wechat"],default="kuairec")
    ap.add_argument("--source",type=Path)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args();d=(a.output or Path(f"reports/{a.dataset}_calibration_v1")).resolve();src=(a.source or Path(f"reports/{a.dataset}_v1")).resolve();cache=src/"cache"
    if d==src or src.name!=f"{a.dataset}_v1" or d.name!=f"{a.dataset}_calibration_v1":raise ValueError("Use separate dataset-specific calibration output")
    d.mkdir(parents=True,exist_ok=True)
    cc=json.loads((cache/"contract.json").read_text());summary=json.loads((src/"summary.json").read_text())
    expected={"kuairec":"KuaiRec 2.0 big matrix","vklsvd":"VK-LSVD up0.001_ip0.001 weeks 20-25","wechat":"WeChat D2Co days 1-14"}[a.dataset]
    if summary["dataset"]!=expected or summary["status"]!="OFFLINE_PASS":raise ValueError("Dataset mismatch")
    if cc["actions"]!=list(NAMES) or cc["checkpoint_sha256"]!=summary["checkpoint_sha256"] or cc["checkpoint_sha256"]!=_sha(src/"behavior.pt"):raise ValueError("Cache provenance mismatch")
    if cc["selector_sha256"]!=_sha(Path(__file__).with_name("m5_selector.py")):raise ValueError("Selector changed")
    V=load(cache,"validation")
    groups=np.unique(V["u"]);np.random.default_rng(42).shuffle(groups)
    mask=np.isin(V["u"],groups[:max(1,int(.65*len(groups)))])
    fit=subset(V,mask);cal=subset(V,~mask)
    if len(fit["ids"])!=summary["fit_requests"] or len(cal["ids"])!=summary["calibration_requests"]:raise ValueError("Validation split mismatch")
    slope,intercept=fit_map(fit["P"],fit["y"])
    F=dict(fit,P=transform(fit["P"],slope,intercept));C=dict(cal,P=transform(cal["P"],slope,intercept))
    loss=s.loss_matrix(F["y"],F["P"])
    table=dict(schema=1,checkpoint_sha256=cc["checkpoint_sha256"],rows=[])
    for i,k in zip(RECENT,KS):
        n=np.minimum(F["N"],k)
        table["rows"].append(dict(k=k,mean_tokens=float(n.mean()),mean_bce=float(loss[:,i].mean()),mean_macs=float((s.HEAD_MACS+s.TOKEN_MACS*np.maximum(n,1)).mean())))
    reg=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=40,max_features=.7,random_state=42,n_jobs=4)
    reg.fit(F["X"],loss-loss[:,-1,None])
    plans={};matches={};validation=[]
    r0=resources(C["N"],False);r1=resources(C["N"],True);lc=s.loss_matrix(C["y"],C["P"])
    for rev,w in enumerate(s.PROFILES,1):
        name=f"q{w:g}";plan=compile_policy(table,dict(revision=rev,preferences=dict(quality_weight=w),average_tokens=None));plans[name]=plan;pref=plan["preferences"]
        ix=choose_global(plan,C["ids"]);validation.append(dict(method="global",profile=name,**metrics(C,ix,r0,pref)))
        rr=r1 if w>0 else r0;ix=decision(reg.predict(C["X"]),rr,pref)
        m=metrics(C,ix,rr,pref);validation.append(dict(method="extra_trees",profile=name,**m))
        i,j,z=mixture(s.cost_table(r0,pref).mean(0),lc.mean(0),m["mean_cost"],RECENT)
        matches[name]=dict(i=int(i),j=int(j),prob_b=z,validation_cost=m["mean_cost"])
    write(d/"frozen_calibration.json",dict(slope=slope,intercept=intercept,fit_requests=len(fit["ids"]),mapping="shared sigmoid(a*logit(p)+b), a>0",table=table,plans=plans,matched=matches,checkpoint_sha256=cc["checkpoint_sha256"],exploratory=True))
    pd.DataFrame(validation).to_csv(d/"validation_comparison.csv",index=False)
    print("CALIBRATION FROZEN",slope,intercept,flush=True)
    T=load(cache,"test")
    if set(V["ids"])&set(T["ids"]):raise ValueError("Split overlap")
    U=dict(T,P=transform(T["P"],slope,intercept))
    audit=[]
    for split,raw,corr in (("validation_cal",cal,C),("test",T,U)):
        for i,name in enumerate(NAMES):
            y=raw["y"];p0=raw["P"][:,i];p1=corr["P"][:,i]
            audit.append(dict(split=split,action=name,positive_rate=float(y.mean()),raw_mean=float(p0.mean()),calibrated_mean=float(p1.mean()),raw_bce=float(log_loss(y,p0)),calibrated_bce=float(log_loss(y,p1)),raw_auc=float(roc_auc_score(y,p0)),calibrated_auc=float(roc_auc_score(y,p1)),raw_brier=float(np.mean((p0-y)**2)),calibrated_brier=float(np.mean((p1-y)**2))))
    ad=pd.DataFrame(audit);ad.to_csv(d/"calibration_audit.csv",index=False)
    rows=[];matched=[];y=U["y"];P=U["P"];j=np.arange(len(y));R0=resources(U["N"],False);R1=resources(U["N"],True)
    raw_actions=read(src/"selected_actions.npz")
    if not np.array_equal(raw_actions["ids"],U["ids"]):raise ValueError("Raw decision IDs mismatch")
    for name,plan in plans.items():
        pref=plan["preferences"];w=pref["quality_weight"];rr=R1 if w>0 else R0
        gi=choose_global(plan,U["ids"]);ei=decision(reg.predict(U["X"]),rr,pref)
        for method,ix,res in (("global",gi,R0),("extra_trees",ei,rr)):
            rows.append(dict(method=method,profile=name,variant="recompiled",**metrics(U,ix,res,pref)))
            rows.append(dict(method=method,profile=name,variant="original_actions_calibrated_scores",**metrics(U,raw_actions[method+"__"+name],res,pref)))
        mx=matches[name];i,b,z=mx["i"],mx["j"],mx["prob_b"];m=metrics(U,ei,rr,pref)
        aucs=[roc_auc_score(y,P[j,np.where(np.random.default_rng(2027+seed).random(len(y))<z,b,i)]) for seed in range(20)]
        lm=s.loss_matrix(y,P);ct=s.cost_table(R0,pref)
        ml=float(((1-z)*lm[:,i]+z*lm[:,b]).mean());mc=float(((1-z)*ct[:,i]+z*ct[:,b]).mean())
        matched.append(dict(profile=name,adaptive_auc=m["auc"],mixture_auc_mean=float(np.mean(aucs)),adaptive_cost=m["mean_cost"],mixture_expected_cost=mc,adaptive_bce=m["bce"],mixture_expected_bce=ml,delta_auc=m["auc"]-float(np.mean(aucs)),delta_utility=w*(m["bce"]-ml)/s.QUALITY_SCALE+pref["cost_weight"]*(m["mean_cost"]-mc)))
    df=pd.DataFrame(rows);df.to_csv(d/"quality_cost.csv",index=False);md=pd.DataFrame(matched);md.to_csv(d/"matched_comparison.csv",index=False)
    write(d/"summary.json",dict(status="CALIBRATION_AUDIT_PASS",dataset=expected,exploratory=True,slope=slope,intercept=intercept,checkpoint_sha256=cc["checkpoint_sha256"],limitations=["Sensitivity motivated by prior KuaiRec test inspection; not independent confirmation.","One shared monotone mapping fitted on validation-fit users; no ranker retraining.","Recompiled policies use calibrated BCE with same quality scale; utility across raw and calibrated objectives is not directly comparable.","Serving remains original raw protocol."]))
    print(ad[(ad.split=="test")&(ad.action=="recent_2048")].to_string(index=False))
    print(md.to_string(index=False));print(df[df.variant=="recompiled"][["method","profile","auc","bce","mean_k","mean_cost"]].to_string(index=False))
    print("CALIBRATION_AUDIT_PASS",flush=True)
if __name__=="__main__":main()
