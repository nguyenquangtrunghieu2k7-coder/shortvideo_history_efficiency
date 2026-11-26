"""Cache-only matched-budget audit. No ranker, selector training, or HTTP.
Validation chooses mixtures of recent fixed actions; test evaluates frozen
mixtures with label-independent assignments. Expected BCE/cost are exact
expectations over mixture draws; AUC is empirical across fixed seeds.
This is exploratory: the test cohort has been inspected previously."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.integration import m5_selector as s

ACTIONS=tuple(a for a in s.ACTIONS if a[1] in s.ORIGINAL_BUDGETS)
NAMES=tuple(f"{m}_{k}" for m,k in ACTIONS)
RECENT=np.array([i for i,(m,k) in enumerate(ACTIONS) if m=="recent"])
COLS=np.array([s.ACTIONS.index(a) for a in ACTIONS])

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def mixture(cost,loss,budget,eligible):
    """Linear objective over probability simplex and one expected cost cap.
    A vertex optimum has at most two nonzero action probabilities."""
    candidates=[]
    for i in eligible:
        if cost[i]<=budget+1e-10:candidates.append((loss[i],cost[i],i,i,0.))
    for i in eligible:
        for j in eligible:
            if cost[i]<budget<cost[j]:
                w=(budget-cost[i])/(cost[j]-cost[i])
                candidates.append(((1-w)*loss[i]+w*loss[j],budget,i,j,w))
    if not candidates:raise ValueError("No feasible mixture")
    _,_,i,j,w=min(candidates)
    return i,j,float(w)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache",type=Path,default=Path("reports/m6_cache_v1"))
    ap.add_argument("--m5-dir",type=Path,default=Path("reports/m5_release_v1"))
    ap.add_argument("--m6-dir",type=Path,default=Path("reports/m6_selector_v1"))
    ap.add_argument("--output",type=Path,default=Path("reports/m6_matched_audit_v1"))
    ap.add_argument("--seeds",type=int,default=20)
    a=ap.parse_args()
    if a.seeds<1:ap.error("positive seeds required")
    # Deliberately no fallback that scores a missing cache.
    import hashlib
    def sha(path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
        return h.hexdigest()
    contract=json.loads((a.cache/"contract.json").read_text())
    prior=json.loads((a.m5_dir/"summary.json").read_text())
    if tuple(contract["actions"])!=NAMES or contract["checkpoint"]!=prior["checkpoint_sha256"] or contract["snapshot"]!=prior["snapshot_id"] or contract["test_predictions_sha256"]!=sha(a.m5_dir/"predictions.npz"):
        raise ValueError("Cache contract mismatch")
    v=read_npz(a.cache/"validation_features.npz");t=read_npz(a.cache/"test_features.npz")
    pp=[]
    for name in NAMES:
        z=read_npz(a.cache/f"val_{name}.npz")
        if not np.array_equal(z["ids"],v["ids"]):raise ValueError("Validation order mismatch")
        pp.append(z["pred"])
    V=np.column_stack(pp)
    z=read_npz(a.m5_dir/"predictions.npz")
    chosen=read_npz(a.m6_dir/"selected_actions.npz")
    if not np.array_equal(t["ids"],z["evaluation_ids"]) or not np.array_equal(t["ids"],chosen["evaluation_ids"]):raise ValueError("Test order mismatch")
    P=z["action_predictions"][:,COLS];y=z["labels"]
    if not np.isfinite(V).all() or not np.isfinite(P).all():raise ValueError("Nonfinite cache")
    groups=np.unique(v["users"]);np.random.default_rng(42).shuffle(groups)
    cal=~np.isin(v["users"],groups[:max(1,int(.65*len(groups)))])
    lv=s.loss_matrix(v["labels"][cal],V[cal]);lt=s.loss_matrix(y,P)
    vf=pd.read_csv(a.m6_dir/"validation_comparison.csv")
    methods=("length","embedding","full_recent","full_joint","m5_extra_trees")
    rows=[];frozen=[]
    # Focus on the pre-existing unconstrained profiles; no test-based choice.
    for w in (.25,.5,.75,.9,.95,.99,1.):
        profile=f"q{w:g}";pref=s.preference(w)
        rv=s.resource_table(v["N"][cal],selector=False)[:,COLS,:]
        rt=s.resource_table(t["N"],selector=False)[:,COLS,:]
        rs=s.resource_table(t["N"],selector=True)[:,COLS,:]
        cv=s.cost_table(rv,pref).mean(0);ct=s.cost_table(rt,pref)
        j=np.arange(len(y))
        for method in methods:
            budget=float(vf.loc[(vf.method==method)&(vf.profile==profile),"mean_cost"].iloc[0])
            i,k,p=mixture(cv,lv.mean(0),budget,RECENT)
            frozen.append(dict(method=method,profile=profile,validation_budget=budget,action_a=NAMES[i],action_b=NAMES[k],prob_b=p))
            ix=chosen[method+"__"+profile]
            if ix.shape!=(len(y),) or (ix<0).any() or (ix>=len(ACTIONS)).any():raise ValueError("Invalid saved decisions")
            ar=rs if method in ("full_recent","full_joint","m5_extra_trees") else rt
            ac=s.cost_table(ar,pref)[j,ix].mean()
            au=float(roc_auc_score(y,P[j,ix]));al=float(lt[j,ix].mean())
            mc=float(((1-p)*ct[:,i]+p*ct[:,k]).mean())
            ml=float(((1-p)*lt[:,i]+p*lt[:,k]).mean())
            auc=[]
            for seed in range(a.seeds):
                pick=np.where(np.random.default_rng(2027+seed).random(len(y))<p,k,i)
                auc.append(float(roc_auc_score(y,P[j,pick])))
            rows.append(dict(method=method,profile=profile,validation_budget=budget,action_a=NAMES[i],action_b=NAMES[k],prob_b=p,
                adaptive_auc=au,mixture_auc_mean=float(np.mean(auc)),mixture_auc_seed_sd=float(np.std(auc)),
                adaptive_cost=float(ac),mixture_expected_cost=mc,adaptive_bce=al,mixture_expected_bce=ml,
                delta_auc_vs_mixture=au-float(np.mean(auc)),
                delta_utility_loss_vs_mixture=w*(al-ml)/s.QUALITY_SCALE+(1-w)*(float(ac)-mc)))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"frozen_mixtures.json").write_text(json.dumps(frozen,indent=2))
    df=pd.DataFrame(rows);df.to_csv(a.output/"comparison.csv",index=False)
    (a.output/"summary.json").write_text(json.dumps(dict(status="EXECUTION_PASS",test_requests=len(y),seeds=a.seeds,comparisons=len(rows),
        scope="Cache-only exploratory comparison; no scoring or training",
        caveats=["Mixture matches validation expected budget, not necessarily test budget.","Mixture optimized for BCE, not AUC.","Seed SD describes assignment randomness, not population confidence interval.","Resource proxies exclude selector compute and RPC overhead."]),indent=2))
    print(df[df.method.isin(["m5_extra_trees","length","embedding"])].round(6).to_string(index=False));print("REPORT:",a.output)
if __name__=="__main__":main()
