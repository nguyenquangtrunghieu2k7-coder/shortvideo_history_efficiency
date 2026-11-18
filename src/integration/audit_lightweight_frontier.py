"""Audit lightweight selector feasibility against ALL fixed-K baselines on the
same user-disjoint calibration cohort. Diagnostic only: selected penalty was
already tuned on this cohort, so this is NOT independent validation.
"""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
from src.policy.adaptive_v4_policy import K_VALUES

ROOT=e2e.ROOT
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,default=ROOT/"models/experimental_lightweight_selector_v1.json")
    p.add_argument("--predictions",type=Path,default=ROOT/"reports/experiments/shared_variable_k_validation_evaluation.npz")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--fit-fraction",type=float,default=.65)
    p.add_argument("--output",type=Path,default=ROOT/"reports/experiments/lightweight_selector_calibration_frontier.json")
    args=p.parse_args()
    report=json.loads(args.report.read_text())
    z=np.load(args.predictions,allow_pickle=False)
    y=np.asarray(z["labels"],dtype=int);users=np.asarray(z["users"],dtype=np.int64)
    unique=np.unique(users);np.random.default_rng(args.seed).shuffle(unique)
    nfit=int(len(unique)*args.fit_fraction)
    mask=~np.isin(users,unique[:nfit])
    if mask.sum()!=report["n_calibration"]:
        raise RuntimeError("Calibration cohort mismatch; check seed and fit-fraction")
    vocab,vmap=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset("validation","Full",vmap)
    if not np.array_equal(np.asarray(ds.evaluation_ids),z["evaluation_ids"]):
        raise RuntimeError("Evaluation IDs mismatch")
    lengths=np.asarray([len(h) for h in ds.history_ids],dtype=int)[mask]
    full_mean=float(lengths.mean())
    rows=[]
    for k in K_VALUES:
        key=f"Fixed_{k}"
        scores=np.asarray(z[key])[mask]
        used=lengths if k=="Full" else np.minimum(lengths,int(k))
        rows.append({"policy":f"Fixed-{k}","auc":float(roc_auc_score(y[mask],scores)),
                     "mean_k":float(used.mean()),
                     "history_saving_vs_full":float(1-used.mean()/max(full_mean,1e-12))})
    selected=report.get("selected")
    if selected is None:
        raise RuntimeError("No calibrated policy selected")
    rows.append({"policy":"Lightweight-V1","auc":selected["auc"],
                 "mean_k":selected["mean_k"],
                 "history_saving_vs_full":float(1-selected["mean_k"]/max(full_mean,1e-12))})
    target=rows[-1]["auc"]
    dominating=[x["policy"] for x in rows[:-1]
                if x["auc"]>=target and x["mean_k"]<=rows[-1]["mean_k"]
                and (x["auc"]>target or x["mean_k"]<rows[-1]["mean_k"])]
    feasible=[x for x in rows if x["auc"]>=rows[-2]["auc"]-report["auc_tolerance"]]
    result={"status":"DIAGNOSTIC_CALIBRATION_FRONTIER_NOT_INDEPENDENT",
            "n_calibration":int(mask.sum()),"mean_full_history":full_mean,
            "auc_tolerance":report["auc_tolerance"],
            "fixed_policies_dominating_lightweight":dominating,
            "best_feasible_by_mean_k":min(feasible,key=lambda x:x["mean_k"])["policy"],
            "rows":rows,
            "limitations":["Lightweight penalty selected on same calibration cohort.",
                           "Mean K is a history-volume proxy, not total measured cost.",
                           "Shared checkpoint already selected on validation.",
                           "Do not select new hyperparameters using test results."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
