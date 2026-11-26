"""Evaluate one shared variable-K checkpoint on frozen event-time histories.

No training or selection. Validation by default; test only with explicit --split test.
Reports fixed-K and transferred frozen V4, paired user-cluster bootstrap, and
diagnostic forward times. All policies use identical shared weights.
"""
from __future__ import annotations
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from src.integration import e2e_v4_bloom as e2e
from src.policy.adaptive_v4_features import FEATURE_NAMES,history_features
from src.policy.adaptive_v4_policy import K_VALUES,apply_selected_policy_from_lengths

ROOT=e2e.ROOT
def sync(device):
    if device.type=="cuda":torch.cuda.synchronize()
    elif device.type=="mps":torch.mps.synchronize()

def predict(model,ds,histories,device,batch_size):
    out=np.empty(len(ds),dtype=np.float64)
    forward=0.;start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(ds),batch_size):
            batch=[]
            for j in range(i,min(i+batch_size,len(ds))):
                item=dict(ds[j]);item["history_ids"]=histories[j]
                batch.append(item)
            candidate,hist,mask,*_=e2e.baseline.collate_batch(batch)
            candidate=candidate.to(device);hist=hist.to(device);mask=mask.to(device)
            sync(device);t=time.perf_counter()
            p=model(candidate,hist,mask)
            sync(device);forward+=time.perf_counter()-t
            out[i:i+len(batch)]=p.detach().cpu().numpy()
    if not np.isfinite(out).all():raise RuntimeError("Nonfinite predictions")
    return out,forward,time.perf_counter()-start

def paired_ci(y,users,a,b,repeats,seed):
    groups=np.unique(users)
    rng=np.random.default_rng(seed)
    differences=[]
    for _ in range(repeats):
        sampled=rng.choice(groups,len(groups),replace=True)
        counts=Counter(sampled.tolist())
        idx=np.concatenate([np.repeat(np.flatnonzero(users==g),n)
                            for g,n in counts.items()])
        if len(np.unique(y[idx]))<2:continue
        differences.append(float(roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx])))
    if len(differences)<repeats//2:raise RuntimeError("Insufficient valid cluster bootstrap draws")
    return [float(v) for v in np.quantile(differences,[.025,.975])]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=["validation","test"],default="validation")
    p.add_argument("--checkpoint",type=Path,default=ROOT/"models/experimental_shared_variable_k.pth")
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--bootstrap",type=int,default=200)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.batch_size<2 or args.bootstrap<20:raise ValueError("Invalid benchmark args")
    device=e2e.baseline.get_device()
    vocab,vmap=e2e.baseline.load_video_vocab()
    ds=e2e.baseline.KuaiRandDataset(args.split,"Full",vmap)
    y=np.asarray(ds.labels,dtype=np.int8)
    users=np.asarray(ds.user_ids,dtype=np.int64)
    if set(np.unique(y))!={0,1}:raise RuntimeError("Need both classes")
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if ckpt.get("training_protocol")!="uniform_random_K_per_training_example_each_epoch" or ckpt.get("k_values")!=K_VALUES or ckpt.get("vocab_size")!=len(vocab):
        raise RuntimeError("Shared checkpoint schema mismatch")
    model=e2e.baseline.build_model(len(vocab),mode="full").to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval();sync(device)
    histories=ds.history_ids
    features=history_features(histories)
    artifact=joblib.load(ROOT/"models/adaptive_history_selector_v4.joblib")
    if not artifact.get("policy_frozen_after_validation_tuning") or artifact["k_values"]!=K_VALUES or artifact["feature_names"]!=FEATURE_NAMES:
        raise RuntimeError("Frozen policy schema mismatch")
    t=time.perf_counter()
    actions=apply_selected_policy_from_lengths(artifact["selected_validation_policy"],
        artifact["model"],features,[len(h) for h in histories],
        artifact["cost_normalization_scale"],FEATURE_NAMES)
    policy_seconds=time.perf_counter()-t
    scores={};rows=[]
    full_count=sum(map(len,histories))
    for k in K_VALUES:
        chosen=[h if k=="Full" else h[-int(k):] if k else [] for h in histories]
        pred,forward,total=predict(model,ds,chosen,device,args.batch_size)
        name=f"Fixed-{k}";scores[name]=pred
        used=sum(map(len,chosen))
        rows.append({"policy":name,"auc":float(roc_auc_score(y,pred)),
            "mean_history":used/len(ds),"history_saving_vs_full":1-used/max(full_count,1),
            "forward_wall_s_diagnostic":forward,"loop_wall_s_diagnostic":total})
        print(rows[-1],flush=True)
    selected=e2e.select_histories(histories,actions)
    pred,forward,total=predict(model,ds,selected,device,args.batch_size)
    scores["Adaptive-V4-transfer"]=pred
    used=sum(map(len,selected))
    rows.append({"policy":"Adaptive-V4-transfer","auc":float(roc_auc_score(y,pred)),
        "mean_history":used/len(ds),"history_saving_vs_full":1-used/max(full_count,1),
        "forward_wall_s_diagnostic":forward,"loop_wall_s_diagnostic":total})
    print(rows[-1],flush=True)
    comparisons={}
    for reference in ["Fixed-100","Fixed-Full"]:
        comparisons["Adaptive-V4-transfer_minus_"+reference]={
            "delta_auc":float(roc_auc_score(y,pred)-roc_auc_score(y,scores[reference])),
            "user_cluster_bootstrap_95_ci":paired_ci(y,users,pred,scores[reference],
                args.bootstrap,args.seed)}
    output=args.output or ROOT/"reports/experiments"/f"shared_variable_k_{args.split}_evaluation.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"),labels=y,users=users,
        evaluation_ids=np.asarray(ds.evaluation_ids),**{k.replace("-","_"):v for k,v in scores.items()})
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"),index=False)
    report={"status":"EXPLORATORY_SHARED_MODEL_EVALUATION",
        "split":args.split,"n_requests":len(ds),"n_users":len(np.unique(users)),
        "checkpoint":str(args.checkpoint),"checkpoint_best_epoch":ckpt["epoch"],
        "device":str(device),"rows":rows,"comparisons":comparisons,
        "frozen_v4_policy_prediction_wall_s":policy_seconds,
        "action_counts":dict(Counter(str(K_VALUES[int(a)]) for a in actions)),
        "limitations":["Event-time frozen histories, not S2 scheduled complete-case histories.",
            "Frozen V4 was calibrated for K-specific checkpoints; transfer is diagnostic.",
            "Bootstrap is user-cluster paired; not a prospective confidence guarantee.",
            "No checkpoint or policy selection using held-out test.",
            "Timing is one non-warmed pass, excludes feature and retrieval costs.",
            "Do not select policies or tune hyperparameters on test outcomes."]}
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(output),"comparisons":comparisons},indent=2))
if __name__=="__main__":main()
