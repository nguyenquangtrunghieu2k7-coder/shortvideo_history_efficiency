"""Policy x refresh matrix on identical frozen-vocabulary-safe test requests.

8 fixed-K policies + frozen Adaptive V4 x 4 refresh schedules.
Raw histories are capped at 2048 BEFORE mapping. No frozen data/model changes.
Outputs JSON/CSV and a compressed paired-score archive for future bootstrap.
"""
import argparse
from pathlib import Path
from collections import Counter
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
from src.integration.scheduled_window_ablation import DAY,last_refresh
from src.integration.scheduled_window_controlled import (
    STRIDES,load_raw,complete_case_mask,paired_user_cluster_ci)
from src.policy.adaptive_v4_features import FEATURE_NAMES,history_features
from src.policy.adaptive_v4_policy import K_VALUES,apply_selected_policy_from_lengths
from bisect import bisect_left

def raw_capped_histories(events,raw,stride,vocab,eligible):
    result=[]
    for user,t in events.loc[eligible,["user_id","time_ms"]].itertuples(index=False,name=None):
        r=last_refresh(int(t),stride,2,7)
        times,ids=raw.get(int(user),([],[]))
        lo,hi=bisect_left(times,r-DAY),bisect_left(times,r)
        capped=ids[max(lo,hi-2048):hi]
        if any(int(v) not in vocab for v in capped):
            raise RuntimeError("OOV in complete-case cohort: strict fail")
        result.append([int(vocab[int(v)])+1 for v in capped])
    return result

def subset_dataset(ds,idx):
    import copy
    sub=copy.copy(ds)
    for field in ("candidate_ids","history_ids","history_lengths","labels","user_ids","evaluation_ids"):
        old=getattr(ds,field)
        setattr(sub,field,[old[int(i)] for i in idx])
    return sub

def run(args):
    if args.bootstrap<20:raise ValueError("bootstrap must be >=20")
    vocab,vocab_map=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset("test","Full",vocab_map)
    all_events=pd.read_parquet(e2e.ROOT/"data/processed/evaluation_events.parquet",
        columns=["evaluation_id","user_id","time_ms","long_view"])
    if not all_events.evaluation_id.is_unique:raise RuntimeError("Duplicate evaluation_id")
    events=all_events.set_index("evaluation_id").loc[ds.evaluation_ids].reset_index()
    labels=np.asarray(ds.labels,dtype=np.int8)
    users=np.asarray(ds.user_ids,dtype=np.int64)
    if list(map(int,events.user_id))!=users.tolist() or list(map(int,events.long_view))!=labels.tolist():
        raise RuntimeError("Frozen evaluation alignment failed")
    policy=joblib.load(e2e.ROOT/"models/adaptive_history_selector_v4.joblib")
    if not policy.get("policy_frozen_after_validation_tuning") or policy["k_values"]!=K_VALUES or policy["feature_names"]!=FEATURE_NAMES:
        raise RuntimeError("Frozen V4 contract failed")
    lower=min(last_refresh(int(events.time_ms.min()),s,2,7) for s in STRIDES)-DAY
    raw=load_raw(e2e.ROOT/"data/raw/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv",[],0,0) if False else None
    from src.integration.scheduled_window_ablation import STANDARD
    raw=load_raw(STANDARD,set(users.tolist()),lower,int(events.time_ms.max()))
    eligible,oov=complete_case_mask(events,raw,vocab_map)
    idx=np.flatnonzero(eligible)
    if len(idx)<2 or len(set(labels[idx].tolist()))!=2:raise RuntimeError("No valid common cohort")
    sub=subset_dataset(ds,idx)
    y=labels[idx];u=users[idx]
    ev=events.loc[eligible].reset_index(drop=True)
    rows=[];score_store={}
    device=e2e.baseline.get_device()
    for i,stride in enumerate(STRIDES):
        histories=raw_capped_histories(events,raw,stride,vocab_map,eligible)
        features=history_features(histories)
        adaptive=apply_selected_policy_from_lengths(
            policy["selected_validation_policy"],policy["model"],features,
            [len(h) for h in histories],policy["cost_normalization_scale"],FEATURE_NAMES)
        policies={f"Fixed-{k}":np.full(len(idx),j,dtype=np.int64)
                  for j,k in enumerate(K_VALUES)}
        policies["Adaptive-V4"]=np.asarray(adaptive,dtype=np.int64)
        full=sum(map(len,histories))
        for name,actions in policies.items():
            selected=e2e.select_histories(histories,actions)
            pred,seconds=e2e.predict_by_action(sub,selected,actions,len(vocab),device,args.batch_size)
            used=sum(map(len,selected))
            key=f"S{i}__{name}"
            score_store[key]=pred
            rows.append({"scenario":f"S{i}","stride_hours":stride,"policy":name,
                "auc":float(roc_auc_score(y,pred)),
                "selected_history_interactions":int(used),
                "mean_selected_history_length":float(used/len(idx)),
                "history_saving_vs_full":float(1-used/max(full,1)),
                "full_history_interactions":int(full),
                "model_inference_seconds_diagnostic":float(seconds),
                "action_counts":dict(Counter(str(K_VALUES[int(a)]) for a in actions))})
            print(f"{key}: AUC={rows[-1]['auc']:.6f}, mean_K={rows[-1]['mean_selected_history_length']:.2f}",flush=True)
    # Three-objective Pareto: quality up, selected history down, refresh workload down.
    # Refresh workload from original full-cohort audit is a proxy and cohort mismatch.
    stage_path=e2e.ROOT/"reports/experiments/scheduled_window_v4_e2e.json"
    workload={}
    if stage_path.exists():
        stage=json.loads(stage_path.read_text())
        workload={x["scenario"]:int(x["batch_workload_capped_interactions"]) for x in stage["scenarios"]}
    for row in rows:
        row["observed_refresh_workload_all_jobs_proxy"]=workload.get(row["scenario"])
    for row in rows:
        row["pareto_quality_history_only"]=not any(
            q["auc"]>=row["auc"] and
            q["selected_history_interactions"]<=row["selected_history_interactions"] and
            (q["auc"]>row["auc"] or q["selected_history_interactions"]<row["selected_history_interactions"])
            for q in rows if q is not row)
        row["pareto_quality_history_refresh_proxy"]=None if not workload else not any(
            q["auc"]>=row["auc"] and
            q["selected_history_interactions"]<=row["selected_history_interactions"] and
            q["observed_refresh_workload_all_jobs_proxy"]<=row["observed_refresh_workload_all_jobs_proxy"] and
            (q["auc"]>row["auc"] or q["selected_history_interactions"]<row["selected_history_interactions"] or
             q["observed_refresh_workload_all_jobs_proxy"]<row["observed_refresh_workload_all_jobs_proxy"])
            for q in rows if q is not row)
    comparisons={}
    for i in range(4):
        key=f"S{i}__Adaptive-V4"
        comparisons[f"S{i}_adaptive_minus_fixed_full"]=paired_user_cluster_ci(
            y,u,score_store[key],score_store[f"S{i}__Fixed-Full"],args.bootstrap,args.seed)
    report={"status":"EXPLORATORY_POLICY_REFRESH_MATRIX","n_events_total":len(labels),
        "n_complete_case":len(idx),"n_users_complete_case":len(set(u.tolist())),
        "n_policies":9,"n_scenarios":4,"rows":rows,
        "paired_user_cluster_bootstrap_95_ci":comparisons,
        "caveats":["Complete-case selected cohort; no claims for excluded test requests.",
                   "Frozen checkpoints trained on event-time histories; scheduled histories shift distribution.",
                   "Refresh workload proxy covers all observed jobs, not the complete-case subset.",
                   "Inference timings are not production serving benchmarks.",
                   "Post-hoc Pareto selection on test is exploratory, not independent validation."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    pd.DataFrame([{k:v for k,v in row.items() if k!="action_counts"} for row in rows]).to_csv(args.output.with_suffix(".csv"),index=False)
    np.savez_compressed(args.output.with_suffix(".npz"),labels=y,users=u,evaluation_ids=np.asarray(ds.evaluation_ids)[idx],**score_store)
    print(json.dumps({"status":report["status"],"n_complete_case":len(idx),
        "policy_scenario_rows":len(rows),
        "pareto_quality_history_only":[f"{z['scenario']}:{z['policy']}" for z in rows if z["pareto_quality_history_only"]],
        "adaptive_vs_fixed_full_ci":comparisons},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--bootstrap",type=int,default=400)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path,default=Path("reports/experiments/policy_refresh_matrix.json"))
    run(p.parse_args())
if __name__=="__main__":main()
