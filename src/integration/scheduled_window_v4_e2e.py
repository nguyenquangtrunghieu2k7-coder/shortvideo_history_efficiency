"""Scheduled S0-S3 frozen V4 -> frozen recommender experiment."""
import argparse
from bisect import bisect_left
from collections import defaultdict, Counter
from pathlib import Path
from time import perf_counter
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
from src.integration.scheduled_window_ablation import DAY, HOUR, STANDARD, last_refresh
from src.policy.adaptive_v4_features import FEATURE_NAMES, history_features
from src.policy.adaptive_v4_policy import K_VALUES, apply_selected_policy_from_lengths

def read_logs(paths, users, lower, upper, vocab):
    out = defaultdict(list)
    seq = 0
    for path in paths:
        for chunk in pd.read_csv(path, usecols=["user_id","video_id","time_ms"], chunksize=250000):
            chunk = chunk.loc[chunk.user_id.isin(users) & (chunk.time_ms >= lower) & (chunk.time_ms < upper)]
            for user, raw, t in chunk[["user_id","video_id","time_ms"]].itertuples(index=False,name=None):
                idx = vocab.get(int(raw))
                if idx is None:
                    raise RuntimeError(f"Unknown video in frozen vocabulary: {raw}")
                out[int(user)].append((int(t),seq,int(idx)+1))
                seq += 1
    return {u:([v[0] for v in sorted(vals)],[v[2] for v in sorted(vals)]) for u,vals in out.items()}

def build_histories(events, records, stride, hour, offset):
    cache = {}
    windows = {}
    result = []
    refreshes = []
    for user,t in events[["user_id","time_ms"]].itertuples(index=False,name=None):
        user,t = int(user),int(t)
        r = last_refresh(t,stride,hour,offset)
        key = (user,r)
        if key not in cache:
            times,vids = records.get(user,([],[]))
            lo,hi = bisect_left(times,r-DAY),bisect_left(times,r)
            windows[key] = (max(lo,hi-2048),hi)
            cache[key] = list(vids[max(lo,hi-2048):hi])
        result.append(cache[key])
        refreshes.append(r)
    repeated = 0
    jobs = defaultdict(list)
    for (user,r),bounds in windows.items():
        jobs[user].append((r,bounds))
    for user,items in jobs.items():
        previous = None
        for _,bounds in sorted(items):
            if previous is not None:
                repeated += max(0,min(previous[1],bounds[1])-max(previous[0],bounds[0]))
            previous = bounds
    oracle_count = overlap_count = new_count = 0
    for (user,t),r in zip(events[["user_id","time_ms"]].itertuples(index=False,name=None),refreshes):
        times = records.get(int(user),([],[]))[0]
        ol,oh = bisect_left(times,int(t)-DAY),bisect_left(times,int(t))
        sl,sh = bisect_left(times,r-DAY),bisect_left(times,r)
        oracle_count += oh-ol
        overlap_count += max(0,min(oh,sh)-max(ol,sl))
        new_count += oh-sh
    return result, {
        "observed_user_refresh_jobs":len(cache),
        "batch_workload_capped_interactions":sum(map(len,cache.values())),
        "repeated_workload_capped_interactions":repeated,
        "mean_refresh_age_hours":float(np.mean((events.time_ms.to_numpy()-np.asarray(refreshes))/HOUR)),
        "exact_raw_interaction_overlap_vs_event_time":overlap_count/max(oracle_count,1),
        "new_interactions_since_refresh":new_count,
    }

def run(args):
    start = perf_counter()
    device = e2e.baseline.get_device()
    vocab,vocab_map = e2e.baseline.load_video_vocab()
    dataset = e2e.baseline.KuaiRandDataset("test","Full",vocab_map)
    if args.max_requests is not None:
        if args.max_requests < 2: raise ValueError("max-requests must be >=2")
        for field in ("candidate_ids","history_ids","history_lengths","labels","user_ids","evaluation_ids"):
            setattr(dataset,field,getattr(dataset,field)[:args.max_requests])
    labels = np.asarray(dataset.labels,dtype=np.int8)
    if len(set(labels.tolist()))!=2: raise ValueError("AUC requires both classes")
    all_events = pd.read_parquet(args.events,columns=["evaluation_id","user_id","time_ms","long_view"])
    if not all_events.evaluation_id.is_unique: raise ValueError("Duplicate event IDs")
    events = all_events.set_index("evaluation_id").loc[dataset.evaluation_ids].reset_index()
    if list(map(int,events.user_id))!=list(map(int,dataset.user_ids)) or list(map(int,events.long_view))!=list(map(int,labels)):
        raise AssertionError("Event alignment failed")
    path = args.policy_path or e2e.ROOT/"models/adaptive_history_selector_v4.joblib"
    policy = joblib.load(path)
    if not policy.get("policy_frozen_after_validation_tuning") or policy["k_values"]!=K_VALUES or policy["feature_names"]!=FEATURE_NAMES:
        raise ValueError("Frozen V4 schema mismatch")
    lower = min(last_refresh(int(events.time_ms.min()),s,args.refresh_local_hour,args.utc_offset_hours) for s in (24,12,6,3))-DAY
    records = read_logs(args.standard_files,set(map(int,events.user_id)),lower,int(events.time_ms.max()),vocab_map)
    rows = []
    for i,stride in enumerate((24,12,6,3)):
        tick = perf_counter()
        histories,workload = build_histories(events,records,stride,args.refresh_local_hour,args.utc_offset_hours)
        features = history_features(histories)
        actions = apply_selected_policy_from_lengths(
            policy["selected_validation_policy"],policy["model"],features,
            [len(h) for h in histories],policy["cost_normalization_scale"],FEATURE_NAMES)
        selected = e2e.select_histories(histories,actions)
        scores,inference_seconds = e2e.predict_by_action(dataset,selected,actions,len(vocab),device,args.batch_size)
        total_full = sum(map(len,histories))
        row = {"scenario":f"S{i}","stride_hours":stride,"auc":float(roc_auc_score(labels,scores)),
               "history_saving":1-sum(map(len,selected))/max(total_full,1),
               "history_full_interactions":total_full,
               "history_selected_interactions":sum(map(len,selected)),
               "action_counts":dict(Counter(str(K_VALUES[int(x)]) for x in actions)),
               **workload,"model_inference_seconds":inference_seconds,
               "scenario_wall_seconds":perf_counter()-tick}
        rows.append(row)
        print(f"S{i} AUC={row['auc']:.6f} saving={row['history_saving']:.2%}",flush=True)
    for row in rows:
        row["auc_delta_vs_S0"]=row["auc"]-rows[0]["auc"]
        row["batch_workload_ratio_vs_S0"]=row["batch_workload_capped_interactions"]/max(rows[0]["batch_workload_capped_interactions"],1)
    report={"status":"PASS","gate":"SCHEDULED_REFRESH_FROZEN_V4_E2E","n_events":len(events),
            "device":str(device),"scenarios":rows,"wall_seconds":perf_counter()-start,
            "limitations":["Frozen V4 and K checkpoints trained on event-time histories; distribution shift.",
            "Observed user-refresh jobs only, not full production scheduling.",
            "Offline local inference timing, not HTTP serving latency.",
            "Do not interpret raw interaction overlap as recommendation accuracy.",
            "No retraining; AUC on same held-out requests, not a proof of optimality."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events",type=Path,default=e2e.ROOT/"data/processed/splits/KFull/test.parquet")
    p.add_argument("--standard-files",type=Path,nargs="+",default=STANDARD)
    p.add_argument("--refresh-local-hour",type=int,default=2)
    p.add_argument("--utc-offset-hours",type=int,default=7)
    p.add_argument("--policy-path",type=Path)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--max-requests",type=int)
    p.add_argument("--output",type=Path,default=Path("reports/experiments/scheduled_window_v4_e2e.json"))
    args=p.parse_args()
    if args.batch_size<1:p.error("batch-size must be positive")
    run(args)

if __name__=="__main__":main()
