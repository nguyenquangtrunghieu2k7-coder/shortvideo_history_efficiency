"""Fixed-50-first adaptive gating experiment, validation only.

Uses a user-disjoint fit/calibration split and the frozen shared model's
existing validation paired predictions. This is exploratory because the
shared checkpoint itself was selected on this validation split.
No test artifacts are read, and no test evaluation is triggered.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
from src.policy.adaptive_v4_features import history_features,FEATURE_NAMES

ROOT=e2e.ROOT
ACTIONS=[0,10,20,50,100,"Full"]
BASE=ACTIONS.index(50)
PENALTIES=[0.,.0005,.001,.002,.005,.01,.02,.05,.1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",type=Path,default=ROOT/"reports/experiments/shared_variable_k_validation_evaluation.npz")
    p.add_argument("--output",type=Path,default=ROOT/"models/experimental_adaptive_v2_gate.joblib")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--fit-fraction",type=float,default=.65)
    p.add_argument("--alpha",type=float,default=10.)
    p.add_argument("--auc-tolerance",type=float,default=.005)
    a=p.parse_args()
    if not 0<a.fit_fraction<1 or a.alpha<=0 or a.auc_tolerance<0:
        raise ValueError("Invalid experiment configuration")
    if "test" in a.predictions.name.lower():
        raise ValueError("V2 development forbids test predictions")
    z=np.load(a.predictions,allow_pickle=False)
    required={"labels","users","evaluation_ids"}|{f"Fixed_{k}" for k in ACTIONS}
    if not required.issubset(z.files):raise RuntimeError(f"Missing fields: {required-set(z.files)}")
    vocab,vmap=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset("validation","Full",vmap)
    y=np.asarray(z["labels"],dtype=np.int8)
    users=np.asarray(z["users"],dtype=np.int64)
    ids=np.asarray(z["evaluation_ids"],dtype=np.int64)
    if not np.array_equal(ids,np.asarray(ds.evaluation_ids)) or not np.array_equal(users,np.asarray(ds.user_ids)) or not np.array_equal(y,np.asarray(ds.labels,dtype=np.int8)):
        raise RuntimeError("Validation alignment mismatch")
    if len(np.unique(ids))!=len(ids):raise RuntimeError("Duplicate evaluation IDs")
    preds=np.column_stack([np.asarray(z[f"Fixed_{k}"],dtype=float) for k in ACTIONS])
    if not np.isfinite(preds).all():raise RuntimeError("Nonfinite predictions")
    lengths=np.asarray([len(h) for h in ds.history_ids],dtype=float)
    used=np.column_stack([np.minimum(lengths,int(k)) if k!="Full" else lengths for k in ACTIONS])
    features=history_features(ds.history_ids)
    X=features[FEATURE_NAMES].to_numpy(dtype=float)
    groups=np.unique(users);rng=np.random.default_rng(a.seed);rng.shuffle(groups)
    nfit=int(len(groups)*a.fit_fraction)
    fit=np.isin(users,groups[:nfit]);cal=~fit
    if min(fit.sum(),cal.sum())<100 or len(np.unique(y[cal]))!=2:raise RuntimeError("Invalid split")
    # Supervise a bounded-cost gate using per-request log-loss advantage over K50.
    # This is NOT an exact ROC-AUC improvement target.
    clipped=np.clip(preds,1e-6,1-1e-6)
    loss=-(y[:,None]*np.log(clipped)+(1-y[:,None])*np.log1p(-clipped))
    advantage=loss[:,BASE,None]-loss
    model=make_pipeline(StandardScaler(),Ridge(alpha=a.alpha))
    model.fit(X[fit],advantage[fit])
    predicted=np.asarray(model.predict(X[cal]),dtype=float)
    predicted[:,BASE]=0.
    # Penalize incremental history relative to K50; positive penalty
    # can favor cheaper actions, including 0/10/20.
    incremental=(used[cal]-used[cal,BASE,None])/max(float(used[cal,BASE].mean()),1.)
    base_auc=float(roc_auc_score(y[cal],preds[cal,BASE]))
    candidates=[]
    for penalty in PENALTIES:
        decisions=np.argmax(predicted-penalty*incremental,axis=1)
        chosen=preds[cal][np.arange(cal.sum()),decisions]
        auc=float(roc_auc_score(y[cal],chosen))
        mean_k=float(used[cal][np.arange(cal.sum()),decisions].mean())
        candidates.append({"penalty":penalty,"auc":auc,"delta_auc_vs_fixed50":auc-base_auc,
            "mean_k":mean_k,"mean_k_ratio_vs_fixed50":mean_k/max(float(used[cal,BASE].mean()),1e-12),
            "action_counts":{str(k):int((decisions==i).sum()) for i,k in enumerate(ACTIONS)}})
    feasible=[x for x in candidates if x["auc"]>=base_auc-a.auc_tolerance]
    selected=min(feasible,key=lambda x:(x["mean_k"],-x["auc"])) if feasible else None
    # A lower mean K is only a proxy. The gate's feature/prediction cost
    # must be measured separately before claiming E2E efficiency.
    beats_proxy=bool(selected is not None and selected["mean_k"]<float(used[cal,BASE].mean()))
    status="PROXY_GATE_CANDIDATE" if beats_proxy else "NO_GAIN_OVER_FIXED50_PROXY"
    result={"status":status,"split":"validation_user_disjoint_diagnostic",
        "n_fit":int(fit.sum()),"n_calibration":int(cal.sum()),
        "n_fit_users":int(nfit),"n_calibration_users":int(len(groups)-nfit),
        "fixed50_calibration_auc":base_auc,
        "fixed50_mean_k":float(used[cal,BASE].mean()),
        "auc_tolerance_vs_fixed50":a.auc_tolerance,
        "selected":selected,"candidates":candidates,
        "limitations":["Shared checkpoint was selected on validation: not independent confirmation.",
            "Gate uses log-loss regret surrogate, not AUC gradients.",
            "Penalty selected on calibration: reported calibration AUC is selection-biased.",
            "Mean K is a proxy; no measured feature, retrieval, policy or model E2E costs.",
            "No test data used. Do not tune on previously viewed test."]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"features":FEATURE_NAMES,"actions":ACTIONS,
        "base_action":50,"penalty":None if selected is None else selected["penalty"],
        "seed":a.seed,"alpha":a.alpha,"fit_fraction":a.fit_fraction,
        "auc_tolerance":a.auc_tolerance,"status":status,
        "prediction_sha256":hashlib.sha256(a.predictions.read_bytes()).hexdigest()},a.output)
    report=a.output.with_suffix(".json")
    report.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":status,"fixed50_calibration_auc":base_auc,
        "fixed50_mean_k":result["fixed50_mean_k"],"selected":selected,
        "output":str(a.output),"report":str(report)},indent=2))
if __name__=="__main__":main()
