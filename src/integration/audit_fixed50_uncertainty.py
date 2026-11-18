"""Paired user-cluster uncertainty audit for Fixed-50 versus Full and V1.

Uses existing validation predictions only. No policy tuning, no test access.
"""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
ROOT=e2e.ROOT
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",type=Path,default=ROOT/"reports/experiments/shared_variable_k_validation_evaluation.npz")
    p.add_argument("--selector-report",type=Path,default=ROOT/"models/experimental_lightweight_selector_v1.json")
    p.add_argument("--repeats",type=int,default=1000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path,default=ROOT/"reports/experiments/fixed50_validation_uncertainty.json")
    a=p.parse_args()
    if a.repeats<100:raise ValueError("Use at least 100 bootstrap repeats")
    z=np.load(a.predictions,allow_pickle=False)
    report=json.loads(a.selector_report.read_text())
    y=np.asarray(z["labels"],dtype=int)
    users=np.asarray(z["users"],dtype=np.int64)
    unique=np.unique(users)
    rng=np.random.default_rng(a.seed)
    rng.shuffle(unique)
    nfit=int(len(unique)*.65)
    cal=~np.isin(users,unique[:nfit])
    if int(cal.sum())!=report["n_calibration"]:
        raise RuntimeError("Calibration cohort mismatch")
    y=y[cal];u=users[cal]
    p50=np.asarray(z["Fixed_50"])[cal]
    p100=np.asarray(z["Fixed_100"])[cal]
    pfull=np.asarray(z["Fixed_Full"])[cal]
    by_user={int(g):np.flatnonzero(u==g) for g in np.unique(u)}
    gids=np.asarray(list(by_user))
    rng=np.random.default_rng(a.seed+1)
    def compare(a,b):
        delta=float(roc_auc_score(y,a)-roc_auc_score(y,b))
        draws=[]
        for _ in range(a_args.repeats):
            sampled=rng.choice(gids,len(gids),replace=True)
            idx=np.concatenate([by_user[int(g)] for g in sampled])
            if len(np.unique(y[idx]))<2:continue
            draws.append(float(roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx])))
        return {"delta_auc":delta,"user_cluster_bootstrap_95_ci":
                [float(v) for v in np.quantile(draws,[.025,.975])],
                "valid_draws":len(draws)}
    a_args=a
    result={"status":"VALIDATION_CALIBRATION_DIAGNOSTIC_NOT_INDEPENDENT",
        "n_requests":len(y),"n_users":len(gids),
        "fixed50_minus_full":compare(p50,pfull),
        "fixed50_minus_fixed100":compare(p50,p100),
        "warning":"Checkpoint and selector were selected using validation. No new policy is selected here; do not claim independent test evidence."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
