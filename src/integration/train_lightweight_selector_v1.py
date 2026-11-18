"""Lightweight selector research prototype, fitted ONLY on validation predictions.

CAUTION: validation was already used to select the shared checkpoint. The
user-disjoint holdout here is diagnostic, NOT independent final evaluation.
No test labels, predictions or hyperparameter selection are used.
Measured cost table is optional; without it, use a clearly labelled proxy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
from src.policy.adaptive_v4_features import history_features,FEATURE_NAMES
from src.policy.adaptive_v4_policy import K_VALUES

ROOT=e2e.ROOT
NAMES=[f"Fixed-{k}" for k in K_VALUES]
def auc(y,p):
    return float(roc_auc_score(y,p))
def actions_to_scores(predictions,actions):
    return predictions[np.arange(len(actions)),actions]
def build_cost(lengths,mode,measured=None):
    sizes=np.column_stack([np.minimum(lengths,int(k)) if k!="Full" else lengths for k in K_VALUES]).astype(float)
    if mode=="interaction_proxy":
        return sizes / max(float(np.mean(lengths)),1.)
    if measured is None:raise ValueError("Measured cost CSV required")
    df=pd.read_csv(measured)
    required={"policy","loop_wall_s_diagnostic"}
    if not required.issubset(df.columns):
        raise ValueError(f"Cost table must contain {sorted(required)}")
    if df.policy.duplicated().any():raise ValueError("Duplicate policy cost")
    values=[]
    for name in NAMES:
        row=df.loc[df.policy==name]
        if len(row)!=1:raise ValueError(f"Missing cost for {name}")
        values.append(float(row.iloc[0].loop_wall_s_diagnostic))
    if min(values)<0 or not np.isfinite(values).all():raise ValueError("Invalid measured costs")
    # These are aggregate per-policy loop costs, NOT per-request variable-K costs.
    # Normalize as diagnostic estimates; do not claim actual end-to-end savings.
    return np.tile(np.asarray(values)/max(max(values),1e-12),(len(lengths),1))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",type=Path,default=ROOT/"reports/experiments/shared_variable_k_validation_evaluation.npz")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--fit-fraction",type=float,default=.65)
    p.add_argument("--alpha",type=float,default=10.)
    p.add_argument("--cost-mode",choices=["interaction_proxy","measured_policy_loop_proxy"],default="interaction_proxy")
    p.add_argument("--cost-csv",type=Path)
    p.add_argument("--output",type=Path,default=ROOT/"models/experimental_lightweight_selector_v1.joblib")
    args=p.parse_args()
    if not 0<args.fit_fraction<1 or args.alpha<=0:raise ValueError("Invalid parameters")
    z=np.load(args.predictions,allow_pickle=False)
    required={"labels","users","evaluation_ids"}|{name.replace("-","_") for name in NAMES}
    if not required.issubset(z.files):raise RuntimeError(f"Missing predictions: {sorted(required-set(z.files))}")
    labels=np.asarray(z["labels"],dtype=np.int8)
    users=np.asarray(z["users"],dtype=np.int64)
    ids=np.asarray(z["evaluation_ids"],dtype=np.int64)
    if len(np.unique(ids))!=len(ids):raise RuntimeError("Duplicate evaluation IDs")
    if len(set(labels.tolist()))!=2:raise RuntimeError("Both classes required")
    preds=np.column_stack([z[name.replace("-","_")] for name in NAMES])
    if preds.shape!=(len(labels),len(K_VALUES)) or not np.isfinite(preds).all():
        raise RuntimeError("Prediction matrix invalid")
    vocab,vmap=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset("validation","Full",vmap)
    if not np.array_equal(ids,np.asarray(ds.evaluation_ids)) or not np.array_equal(users,np.asarray(ds.user_ids)) or not np.array_equal(labels,np.asarray(ds.labels,dtype=np.int8)):
        raise RuntimeError("Prediction artifact does not align with frozen validation dataset")
    features=history_features(ds.history_ids)
    X=features[FEATURE_NAMES].to_numpy(dtype=np.float64)
    lengths=np.asarray([len(h) for h in ds.history_ids],dtype=float)
    costs=build_cost(lengths,args.cost_mode,args.cost_csv)
    rng=np.random.default_rng(args.seed)
    unique=np.unique(users);rng.shuffle(unique)
    nfit=int(len(unique)*args.fit_fraction)
    fit=np.isin(users,unique[:nfit]);cal=~fit
    if min(fit.sum(),cal.sum())<100 or len(np.unique(labels[cal]))<2:
        raise RuntimeError("Insufficient fit/calibration split")
    # Label-based per-request log-loss regret vs Full. This is a SURROGATE,
    # not an exact per-request decomposition of global ROC-AUC.
    clipped=np.clip(preds,1e-6,1-1e-6)
    losses=-(labels[:,None]*np.log(clipped)+(1-labels[:,None])*np.log1p(-clipped))
    target=losses[:,-1,None]-losses[:,:-1]
    model=make_pipeline(StandardScaler(),Ridge(alpha=args.alpha))
    model.fit(X[fit],target[fit])
    t=time.perf_counter();uplift=np.column_stack([model.predict(X[cal]),np.zeros(cal.sum())])
    policy_predict_s=time.perf_counter()-t
    full_auc=auc(labels[cal],preds[cal,-1])
    # Calibration chooses ONLY on held-out user-disjoint validation subset.
    # Constrain AUC relative to Full; among feasible choose min mean cost.
    candidates=[]
    for penalty in [0.,.001,.003,.01,.03,.1,.3,1.,3.,10.]:
        actions=np.argmax(uplift-penalty*costs[cal],axis=1)
        score=actions_to_scores(preds[cal],actions)
        candidates.append({"penalty":penalty,"auc":auc(labels[cal],score),
            "mean_cost_proxy":float(np.mean(costs[cal][np.arange(len(actions)),actions])),
            "mean_k":float(np.mean(np.column_stack([np.minimum(lengths[cal],int(k)) if k!="Full" else lengths[cal] for k in K_VALUES])[np.arange(len(actions)),actions])),
            "actions":np.bincount(actions,minlength=len(K_VALUES)).tolist()})
    # Quality constraint is a user-chosen design parameter, not optimized from test.
    tolerance=.01
    feasible=[v for v in candidates if v["auc"]>=full_auc-tolerance]
    chosen=min(feasible,key=lambda v:(v["mean_cost_proxy"],-v["auc"])) if feasible else None
    status="CALIBRATION_FEASIBLE" if chosen else "NO_FEASIBLE_ADAPTIVE_CANDIDATE"
    report={"status":status,"n_fit":int(fit.sum()),"n_calibration":int(cal.sum()),
        "fit_users":int(len(unique[:nfit])),"calibration_users":int(len(unique[nfit:])),
        "full_calibration_auc":full_auc,"auc_tolerance":tolerance,
        "cost_mode":args.cost_mode,"policy_predict_calibration_wall_s":policy_predict_s,
        "candidates":candidates,"selected":chosen,
        "limitations":["Checkpoint itself was selected on validation; calibration is NOT an independent unbiased estimate.",
            "Per-request log-loss regret is not global AUC uplift.",
            "Cost modes are proxies, not measured end-to-end per-request costs.",
            "Ridge and alpha are fixed before seeing calibration results; penalty selected only on calibration.",
            "Do not tune on held-out test; require fresh independent evaluation for confirmatory claims."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"features":FEATURE_NAMES,"k_values":K_VALUES,
        "penalty":chosen["penalty"] if chosen else None,
        "cost_mode":args.cost_mode,"status":status,
        "cost_csv":str(args.cost_csv) if args.cost_csv else None,
        "validation_prediction_sha256":hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "auc_tolerance":tolerance},args.output)
    report_path=args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":status,"full_calibration_auc":full_auc,
        "selected":chosen,"output":str(args.output),"report":str(report_path)},indent=2))
if __name__=="__main__":main()
