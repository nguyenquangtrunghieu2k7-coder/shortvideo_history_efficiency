"""Strict paired scheduled-refresh sensitivity experiment.

No frozen artifact changes. Primary: complete-case test requests with zero OOV
in *all four* scheduled capped windows (OOV counted BEFORE any filtering).
Secondary: full-cohort OOV-exclusion results, explicitly marked confounded.
Produces paired user-cluster bootstrap CIs and raw score arrays for reproducibility.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
from src.integration.scheduled_window_ablation import DAY, STANDARD, last_refresh
from src.integration.scheduled_window_v4_e2e import read_logs, build_histories
from src.policy.adaptive_v4_features import FEATURE_NAMES, history_features
from src.policy.adaptive_v4_policy import K_VALUES, apply_selected_policy_from_lengths

STRIDES=(24,12,6,3)

def complete_case_mask(events,raw_records,vocab_map):
    """Exclude a request if ANY scenario's capped raw window has an unseen video."""
    good=np.ones(len(events),dtype=bool)
    oov_counts=np.zeros((len(events),len(STRIDES)),dtype=np.int32)
    for i,(user,t) in enumerate(events[["user_id","time_ms"]].itertuples(index=False,name=None)):
        times,raws=raw_records.get(int(user),([],[]))
        for j,stride in enumerate(STRIDES):
            r=last_refresh(int(t),stride,2,7)
            lo,hi=bisect_left(times,r-DAY),bisect_left(times,r)
            capped=raws[max(lo,hi-2048):hi]
            oov_counts[i,j]=sum(int(raw) not in vocab_map for raw in capped)
        good[i]=not np.any(oov_counts[i])
    return good,oov_counts

def paired_user_cluster_ci(labels,users,scores_a,scores_b,repeats=400,seed=42):
    """Resample users (all their requests together); preserve pairing."""
    rng=np.random.default_rng(seed)
    groups=defaultdict(list)
    for i,u in enumerate(users):groups[int(u)].append(i)
    ids=np.array(sorted(groups))
    arr=[np.asarray(groups[u],dtype=int) for u in ids]
    values=[]
    for _ in range(repeats):
        chosen=rng.integers(0,len(arr),len(arr))
        idx=np.concatenate([arr[k] for k in chosen])
        if len(np.unique(labels[idx]))<2:continue
        values.append(roc_auc_score(labels[idx],scores_a[idx])-
                      roc_auc_score(labels[idx],scores_b[idx]))
    if len(values)<max(20,repeats//2):
        raise RuntimeError("Insufficient valid user-cluster bootstrap replicates")
    return {"lower_95":float(np.quantile(values,.025)),
            "upper_95":float(np.quantile(values,.975)),
            "valid_replicates":len(values)}

def load_raw(paths,users,lower,upper):
    rec=defaultdict(list)
    seq=0
    for path in paths:
        for chunk in pd.read_csv(path,usecols=["user_id","video_id","time_ms"],chunksize=250000):
            sub=chunk.loc[chunk.user_id.isin(users)&(chunk.time_ms>=lower)&(chunk.time_ms<upper)]
            for u,v,t in sub[["user_id","video_id","time_ms"]].itertuples(index=False,name=None):
                rec[int(u)].append((int(t),seq,int(v)))
                seq+=1
    out={}
    for u,rows in rec.items():
        rows.sort()
        out[u]=([r[0] for r in rows],[r[2] for r in rows])
    return out

def run(args):
    if args.bootstrap<20:raise ValueError("bootstrap must be >=20")
    vocab,vocab_map=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset("test","Full",vocab_map)
    events=pd.read_parquet(e2e.ROOT/"data/processed/evaluation_events.parquet",
            columns=["evaluation_id","user_id","time_ms","long_view"])
    events=events.set_index("evaluation_id").loc[ds.evaluation_ids].reset_index()
    labels=np.asarray(ds.labels,dtype=np.int8)
    users=np.asarray(ds.user_ids,dtype=np.int64)
    if list(map(int,events.user_id))!=users.tolist() or list(map(int,events.long_view))!=labels.tolist():
        raise RuntimeError("Frozen event/label alignment mismatch")
    policy=joblib.load(e2e.ROOT/"models/adaptive_history_selector_v4.joblib")
    if not policy.get("policy_frozen_after_validation_tuning") or policy["k_values"]!=K_VALUES or policy["feature_names"]!=FEATURE_NAMES:
        raise RuntimeError("Frozen policy contract mismatch")
    lower=min(last_refresh(int(events.time_ms.min()),s,2,7) for s in STRIDES)-DAY
    upper=int(events.time_ms.max())
    raw=load_raw(STANDARD,set(users.tolist()),lower,upper)
    eligible,oov=complete_case_mask(events,raw,vocab_map)
    if not eligible.any() or len(np.unique(labels[eligible]))!=2:
        raise RuntimeError("Complete-case cohort cannot support AUC")
    # Compare the same 6455 requests; only secondary inference uses OOV filtering.
    mapped,scan=read_logs(STANDARD,set(users.tolist()),lower,upper,vocab_map)
    scores={}
    metrics=[]
    for i,stride in enumerate(STRIDES):
        histories,work=build_histories(events,mapped,stride,2,7)
        features=history_features(histories)
        actions=apply_selected_policy_from_lengths(
            policy["selected_validation_policy"],policy["model"],features,
            [len(h) for h in histories],policy["cost_normalization_scale"],FEATURE_NAMES)
        selected=e2e.select_histories(histories,actions)
        pred,inference=e2e.predict_by_action(ds,selected,actions,len(vocab),
                                              e2e.baseline.get_device(),args.batch_size)
        scores[f"S{i}"]=pred
        metrics.append({"scenario":f"S{i}","stride_hours":stride,
            "full_cohort_auc_oov_excluded":float(roc_auc_score(labels,pred)),
            "complete_case_auc":float(roc_auc_score(labels[eligible],pred[eligible])),
            "complete_case_selected_interactions":int(sum(len(selected[k]) for k in np.flatnonzero(eligible))),
            "complete_case_full_interactions":int(sum(len(histories[k]) for k in np.flatnonzero(eligible))),
            "complete_case_mean_refresh_age_hours":float(np.mean(
                (events.time_ms.to_numpy()[eligible]-np.asarray([
                    last_refresh(int(t),stride,2,7) for t in events.time_ms.to_numpy()[eligible]]))/3600000)),
            "batch_workload_capped_interactions_all_observed_jobs":work["batch_workload_capped_interactions"],
            "model_inference_seconds":float(inference)})
        print(f"S{i}: complete-case AUC={metrics[-1]['complete_case_auc']:.6f}",flush=True)
    intervals={}
    for i in range(1,4):
        intervals[f"S{i}_minus_S0"]=paired_user_cluster_ci(
            labels[eligible],users[eligible],scores[f"S{i}"][eligible],scores["S0"][eligible],
            args.bootstrap,args.seed)
    report={"status":"DIAGNOSTIC_COMPLETE_CASE","n_all":len(labels),
            "n_complete_case":int(eligible.sum()),
            "complete_case_fraction":float(eligible.mean()),
            "n_complete_case_users":int(len(set(users[eligible].tolist()))),
            "oov_requests_by_scenario":{f"S{i}":int((oov[:,i]>0).sum()) for i in range(4)},
            "oov_raw_capped_occurrences_by_scenario":{f"S{i}":int(oov[:,i].sum()) for i in range(4)},
            "metrics":metrics,"paired_user_cluster_bootstrap_95_ci":intervals,
            "limitations":["Complete-case restriction selects a different evaluation population; not a replacement for the 6455-request primary cohort.",
              "Full-cohort scores use OOV exclusion; may confound refresh effect with vocabulary coverage.",
              "Observed refresh-job workload is not a per-request complete-case compute cost.",
              "No new model training or policy tuning on held-out test."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(args.output.with_suffix(".npz"),labels=labels,users=users,
                        evaluation_ids=np.asarray(ds.evaluation_ids),complete_case=eligible,
                        oov_counts=oov,**scores)
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--bootstrap",type=int,default=400)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path,default=Path("reports/experiments/scheduled_window_controlled.json"))
    run(p.parse_args())
if __name__=="__main__":main()
