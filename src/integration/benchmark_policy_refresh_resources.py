"""M2.3-compatible measured cost matrix for 36 policy/schedule configurations.

Run on the SAME machine as policy_refresh_matrix.py. Reconstruct strict OOV-free
scheduled histories, benchmark in-memory Bloom/exact retrieval and policy/feature
costs, and join inference timing from the 36-cell runner (diagnostic only).
No fake GPU, network, disk, energy or production resource claims.
"""
import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from src.integration import e2e_v4_bloom as e2e
from src.integration.scheduled_window_ablation import STANDARD,last_refresh,DAY
from src.integration.scheduled_window_controlled import STRIDES,load_raw,complete_case_mask
from src.integration.policy_refresh_matrix import raw_capped_histories
from src.policy.adaptive_v4_features import FEATURE_NAMES,history_features
from src.policy.adaptive_v4_policy import K_VALUES,apply_selected_policy_from_lengths

ROOT=e2e.ROOT
BASE=ROOT/"reports/experiments"
def measured(fn):
    tracemalloc.start()
    cpu=time.process_time()
    wall=time.perf_counter()
    value=fn()
    dt=time.perf_counter()-wall
    dc=time.process_time()-cpu
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value,{"wall_s":dt,"process_cpu_s":dc,"python_alloc_peak_mib":peak/(1024**2)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shards",type=int,default=8)
    p.add_argument("--bits-per-key",type=int,default=12)
    p.add_argument("--hashes",type=int,default=5)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--output",type=Path,default=BASE/"policy_refresh_resource_matrix.csv")
    args=p.parse_args()
    if args.repeats<1:raise ValueError("repeats must be positive")
    source=pd.read_csv(BASE/"policy_refresh_matrix.csv")
    if len(source)!=36 or source[["scenario","policy"]].duplicated().any():
        raise RuntimeError("Need exact 36-row policy matrix")
    stage=json.loads((BASE/"scheduled_window_v4_e2e.json").read_text())
    work={x["scenario"]:x for x in stage["scenarios"]}
    vocab,vmap=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset("test","Full",vmap)
    events=pd.read_parquet(ROOT/"data/processed/evaluation_events.parquet",
        columns=["evaluation_id","user_id","time_ms","long_view"])
    events=events.set_index("evaluation_id").loc[ds.evaluation_ids].reset_index()
    labels=np.asarray(ds.labels,dtype=np.int8)
    users=np.asarray(ds.user_ids,dtype=np.int64)
    if events.user_id.astype(int).tolist()!=users.tolist() or events.long_view.astype(int).tolist()!=labels.tolist():
        raise RuntimeError("Frozen alignment failed")
    lower=min(last_refresh(int(events.time_ms.min()),s,2,7) for s in STRIDES)-DAY
    raw=load_raw(STANDARD,set(users.tolist()),lower,int(events.time_ms.max()))
    eligible,_=complete_case_mask(events,raw,vmap)
    idx=np.flatnonzero(eligible)
    if len(idx)!=5254:raise RuntimeError(f"Cohort drift: {len(idx)} != 5254")
    ids=np.asarray(ds.evaluation_ids)[idx]
    policy=joblib.load(ROOT/"models/adaptive_history_selector_v4.joblib")
    if not policy.get("policy_frozen_after_validation_tuning") or policy["k_values"]!=K_VALUES or policy["feature_names"]!=FEATURE_NAMES:
        raise RuntimeError("Policy frozen contract failed")
    rows=[]
    for i,stride in enumerate(STRIDES):
        scenario=f"S{i}"
        histories=raw_capped_histories(events,raw,stride,vmap,eligible)
        (features,ft)=measured(lambda:history_features(histories))
        (adaptive,pt)=measured(lambda:apply_selected_policy_from_lengths(
            policy["selected_validation_policy"],policy["model"],features,
            [len(h) for h in histories],policy["cost_normalization_scale"],FEATURE_NAMES))
        (built,bt)=measured(lambda:e2e.build_shards(
            ids,histories,args.shards,args.bits_per_key,args.hashes,args.workers))
        stores,filters,internal_build_s=built
        if len(stores)!=args.shards:raise RuntimeError("Shard count mismatch")
        exact_runs=[];bloom_runs=[]
        for _ in range(args.repeats):
            (ex,et)=measured(lambda:e2e.retrieve_histories(ids,stores,filters,False))
            (bl,lt)=measured(lambda:e2e.retrieve_histories(ids,stores,filters,True))
            eh,es,_=ex;bh,bs,_=bl
            if eh!=histories or bh!=histories:raise RuntimeError("Retrieval parity failed")
            exact_runs.append(et);bloom_runs.append(lt)
        exact_wall=float(np.median([z["wall_s"] for z in exact_runs]))
        bloom_wall=float(np.median([z["wall_s"] for z in bloom_runs]))
        for j,k in enumerate(K_VALUES+["Adaptive-V4"]):
            name=f"Fixed-{k}" if j<len(K_VALUES) else "Adaptive-V4"
            actions=np.full(len(idx),j,dtype=np.int64) if j<len(K_VALUES) else np.asarray(adaptive,dtype=np.int64)
            (selected,st)=measured(lambda:e2e.select_histories(histories,actions))
            used=sum(map(len,selected))
            row=source.loc[(source.scenario==scenario)&(source.policy==name)]
            if len(row)!=1:raise RuntimeError("Missing policy matrix row")
            old=row.iloc[0]
            if abs(used/len(idx)-float(old.mean_selected_history_length))>1e-6:
                raise RuntimeError(f"History mismatch {scenario} {name}")
            # Same full-history retrieval for all policies; it is NOT selected-K lazy retrieval.
            # Timed inference comes from the prior matrix run, not a fresh comparable repeat.
            rows.append({"scenario":scenario,"stride_hours":stride,"policy":name,
                "auc":float(old.auc),"mean_selected_history_length":used/len(idx),
                "selected_history_interactions":used,
                "history_saving_vs_full":1-used/max(sum(map(len,histories)),1),
                "refresh_observed_jobs":work[scenario]["observed_user_refresh_jobs"],
                "refresh_workload_interactions_all_jobs":work[scenario]["batch_workload_capped_interactions"],
                "refresh_workload_ratio_vs_S0":work[scenario]["batch_workload_ratio_vs_S0"],
                "feature_wall_s":ft["wall_s"] if name=="Adaptive-V4" else 0.,
                "policy_wall_s":pt["wall_s"] if name=="Adaptive-V4" else 0.,
                "index_build_wall_s":bt["wall_s"],
                "index_build_process_cpu_s":bt["process_cpu_s"],
                "index_build_python_alloc_peak_mib":bt["python_alloc_peak_mib"],
                "exact_retrieval_wall_s_median":exact_wall,
                "bloom_retrieval_wall_s_median":bloom_wall,
                "bloom_retrieval_process_cpu_s_median":float(np.median([z["process_cpu_s"] for z in bloom_runs])),
                "bloom_retrieval_python_alloc_peak_mib_median":float(np.median([z["python_alloc_peak_mib"] for z in bloom_runs])),
                "bloom_exact_lookups":bs["exact_lookups"],
                "bloom_rejections":bs["bloom_rejections"],
                "selection_wall_s":st["wall_s"],
                "selection_process_cpu_s":st["process_cpu_s"],
                "selection_python_alloc_peak_mib":st["python_alloc_peak_mib"],
                "inference_wall_s_from_prior_run_diagnostic":float(old.model_inference_seconds_diagnostic),
                "diagnostic_serving_sum_wall_s":(
                    (ft["wall_s"]+pt["wall_s"] if name=="Adaptive-V4" else 0)
                    +bloom_wall+st["wall_s"]+float(old.model_inference_seconds_diagnostic)),
                "retrieval_protocol":"M2.3 in-memory full-history fan-out; not lazy selected-K retrieval",
                "timing_protocol":"mixed: measured repeated retrieval; prior-run inference; no production latency"})
        print(scenario,"completed",len(histories),"requests",flush=True)
    out=args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(out,index=False)
    manifest={"status":"MEASURED_OFFLINE_DIAGNOSTIC_NOT_PRODUCTION","n_requests":len(idx),
        "rows":len(df),"repeats_retrieval":args.repeats,
        "shards":args.shards,"bits_per_key":args.bits_per_key,"hashes":args.hashes,
        "workers":args.workers,"output":str(out),
        "limitations":["Refresh workload is a count proxy; refresh CPU/memory is NOT measured here.",
            "Bloom/index/feature measurements are Python wall/process CPU and tracemalloc allocations, not process peak RSS.",
            "Inference timing is imported from previous 36-cell run and is not a repeated end-to-end benchmark.",
            "M2.3 retrieval loads Full history before policy slicing; no true lazy K-aware retrieval.",
            "Bloom and exact retrieval are alternative paths; do not sum both.",
            "Index build is offline; do not add to per-request serving latency without amortization.",
            "Timing measurements are not synchronized across all components or measured on production hardware."]}
    out.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
