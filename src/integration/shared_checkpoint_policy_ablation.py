"""Audit single-checkpoint variable-K inference and policy bottlenecks.

Experimental branch only. Uses frozen Full checkpoint for ALL K without retraining.
This is an architectural ablation, NOT a fair substitute for independently
trained K-specific models; compare AUC and wall time before drawing conclusions.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
from src.integration.policy_refresh_matrix import raw_capped_histories,subset_dataset
from src.integration.scheduled_window_ablation import STANDARD,DAY,last_refresh
from src.integration.scheduled_window_controlled import STRIDES,load_raw,complete_case_mask
from src.policy.adaptive_v4_features import FEATURE_NAMES,history_features
from src.policy.adaptive_v4_policy import K_VALUES,choose_actions_from_lengths

ROOT=e2e.ROOT
OUT=ROOT/"reports/experiments"
def sync(device):
    if device.type=="cuda":torch.cuda.synchronize()
    if device.type=="mps":torch.mps.synchronize()

def predict_shared(model,ds,histories,actions,device,batch_size):
    selected=e2e.select_histories(histories,actions)
    scores=np.empty(len(selected),dtype=np.float64)
    forward=0.
    t0=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(selected),batch_size):
            ix=np.arange(start,min(start+batch_size,len(selected)))
            batch=[]
            for i in ix:
                item=dict(ds[int(i)])
                item["history_ids"]=selected[int(i)]
                batch.append(item)
            candidate,history,mask,*_=e2e.baseline.collate_batch(batch)
            sync(device);before=time.perf_counter()
            pred=model(candidate.to(device),history.to(device),mask.to(device))
            sync(device);forward+=time.perf_counter()-before
            scores[ix]=pred.detach().cpu().numpy()
    return scores,forward,time.perf_counter()-t0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scenario",choices=["S0","S1","S2","S3"],default="S2")
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--output",type=Path,default=OUT/"shared_checkpoint_policy_ablation.json")
    args=p.parse_args()
    vocab,vmap=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset("test","Full",vmap)
    events=pd.read_parquet(ROOT/"data/processed/evaluation_events.parquet",
        columns=["evaluation_id","user_id","time_ms","long_view"])
    events=events.set_index("evaluation_id").loc[ds.evaluation_ids].reset_index()
    y=np.asarray(ds.labels,dtype=np.int8)
    u=np.asarray(ds.user_ids,dtype=np.int64)
    if events.user_id.astype(int).tolist()!=u.tolist() or events.long_view.astype(int).tolist()!=y.tolist():
        raise RuntimeError("Evaluation alignment failure")
    lower=min(last_refresh(int(events.time_ms.min()),s,2,7) for s in STRIDES)-DAY
    raw=load_raw(STANDARD,set(u.tolist()),lower,int(events.time_ms.max()))
    eligible,_=complete_case_mask(events,raw,vmap)
    ix=np.flatnonzero(eligible)
    sub=subset_dataset(ds,ix)
    histories=raw_capped_histories(events,raw,STRIDES[int(args.scenario[1])],vmap,eligible)
    artifact=joblib.load(ROOT/"models/adaptive_history_selector_v4.joblib")
    if not artifact.get("policy_frozen_after_validation_tuning") or artifact["k_values"]!=K_VALUES or artifact["feature_names"]!=FEATURE_NAMES:
        raise RuntimeError("Frozen policy contract mismatch")
    selected=artifact["selected_validation_policy"]
    if selected.get("deployment_mode")!="learned_adaptive":
        raise RuntimeError("Frozen policy is not learned adaptive; profile mode separately")
    t=time.perf_counter()
    features=history_features(histories)
    feature_s=time.perf_counter()-t
    t=time.perf_counter()
    uplift=np.asarray(artifact["model"].predict(features[FEATURE_NAMES]),dtype=float)
    model_predict_s=time.perf_counter()-t
    t=time.perf_counter()
    actions=choose_actions_from_lengths(uplift,[len(h) for h in histories],
        selected["penalty"],artifact["cost_normalization_scale"])
    action_s=time.perf_counter()-t
    device=e2e.baseline.get_device()
    checkpoint=e2e.baseline.MODEL_DIR/e2e.baseline.model_filename("full","Full")
    t=time.perf_counter()
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    model=e2e.baseline.build_model(len(vocab),mode="full").to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval();sync(device)
    model_load_s=time.perf_counter()-t
    rows=[]
    for k in K_VALUES+["Adaptive-V4"]:
        a=actions if k=="Adaptive-V4" else np.full(len(ix),K_VALUES.index(k),dtype=np.int64)
        scores,forward,total=predict_shared(model,sub,histories,a,device,args.batch_size)
        used=sum(len(h) if K_VALUES[int(act)]=="Full" else min(len(h),int(K_VALUES[int(act)]))
                 for h,act in zip(histories,a))
        rows.append({"policy":f"Fixed-{k}" if k!="Adaptive-V4" else k,
            "auc_shared_full_checkpoint":float(roc_auc_score(y[ix],scores)),
            "mean_selected_history":used/len(ix),
            "forward_wall_s":forward,"inference_loop_wall_s":total})
        print(rows[-1],flush=True)
    result={"status":"EXPLORATORY_SHARED_CHECKPOINT_NO_RETRAIN",
        "scenario":args.scenario,"n_requests":len(ix),"n_users":len(set(u[ix].tolist())),
        "shared_checkpoint":str(checkpoint),"device":str(device),
        "policy_model_type":type(artifact["model"]).__name__,
        "policy_selected_model":artifact.get("selected_model_name","unknown"),
        "policy_feature_wall_s":feature_s,
        "policy_model_predict_wall_s":model_predict_s,
        "policy_action_selection_wall_s":action_s,
        "checkpoint_load_wall_s_excluded_from_inference":model_load_s,
        "rows":rows,
        "limitations":["Full checkpoint was trained for Full histories, not for every truncated K.",
                       "This is not an apples-to-apples quality comparison with K-specific checkpoints.",
                       "No retraining, no test-time hyperparameter selection.",
                       "Model remains loaded for all policies; per-policy inference loop measured separately.",
                       "Model and policy timing require repeated warmed trials for latency claims."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"),index=False)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()
