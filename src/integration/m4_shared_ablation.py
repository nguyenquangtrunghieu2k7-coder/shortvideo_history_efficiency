"""Matched shared-weight ablation, no test-based selection.
Train fixed-50 and fixed-Full with the same M4 architecture as the selected
shared model; compare both checkpoints on both history budgets."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from src.integration import m3_product as product
from src.integration.m23_persistent_store import _sha
from src.integration.evaluate_shared_variable_k import paired_ci

def run(d,shared,epochs,batch_size,seed,sample=0):
    d=Path(d);device=product.baseline.get_device()
    ck=torch.load(shared,map_location="cpu",weights_only=True)
    behavior=ck["behavior"];distill=ck["distill"];del ck
    paths={"shared":Path(shared)}
    costs={}
    for k in (50,2048):
        p=d/f"fixed_{k}.pt";m=d/f"fixed_{k}.done.json"
        if not(p.exists() and m.exists() and json.loads(m.read_text()).get("sha256")==_sha(p) and json.loads(m.read_text()).get("shared_sha")==_sha(shared)):
            cmd=[sys.executable,"-m","src.integration.m4_train","--output",str(p),"--epochs",str(epochs),
                "--batch-size",str(batch_size),"--seed",str(seed),"--fixed-k",str(k)]
            if behavior:cmd.extend(["--behavior","--init-behavior",str(shared)])
            if distill:cmd.append("--distill")
            t=time.perf_counter();subprocess.run(cmd,check=True)
            m.write_text(json.dumps(dict(sha256=_sha(p),shared_sha=_sha(shared),training_seconds=time.perf_counter()-t)))
        paths[f"fixed_{k}"]=p;costs[f"fixed_{k}"]=json.loads(m.read_text())["training_seconds"]
    marker=Path(shared).with_suffix(".done.json")
    if marker.exists():costs["shared"]=json.loads(marker.read_text())["training_seconds"]
    _,vmap=product.baseline.load_video_vocab()
    ds=product.baseline.KuaiRandDataset("test","Full",vmap)
    if sample:
        from src.integration.policy_refresh_matrix import subset_dataset
        ix=np.sort(np.random.default_rng(seed).choice(len(ds),min(sample,len(ds)),replace=False));ds=subset_dataset(ds,ix)
    y=np.asarray(ds.labels);u=np.asarray(ds.user_ids)
    scores={};rows=[];sizes={}
    for name,p in paths.items():
        model=product.load_model(p,device)
        sizes[name]=dict(checkpoint_bytes=p.stat().st_size,trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
            total_tensor_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()))
        for k in (50,2048):
            h=[x[-k:] for x in ds.history_ids]
            product.sync(device);t=time.perf_counter()
            pred=product.predict_batch(model,ds,h,device)
            product.sync(device)
            scores[f"{name}_{k}"]=pred
            rows.append(dict(training=name,k=k,auc=float(roc_auc_score(y,pred)),gauc=product.gauc(y,u,pred),
                logloss=float(log_loss(y,pred,labels=[0,1])),offline_batch_scoring_seconds=time.perf_counter()-t,
                **sizes[name]))
        del model
    for row in rows:
        k=row["k"];p=scores[f'{row["training"]}_{k}'];ref=scores[f"fixed_{k}_{k}"]
        row["delta_auc_vs_specialist"]=float(roc_auc_score(y,p)-roc_auc_score(y,ref))
        row["ci95_vs_specialist"]=paired_ci(y,u,p,ref,200,seed)
    deployments=[]
    for name,models in (("independent",["fixed_50","fixed_2048"]),("full_only",["fixed_2048"]),("shared",["shared"])):
        deployments.append(dict(strategy=name,checkpoints=len(models),checkpoint_bytes=sum(sizes[m]["checkpoint_bytes"] for m in models),
            tensor_bytes_if_all_resident=sum(sizes[m]["total_tensor_bytes"] for m in models),
            training_seconds=sum(costs.get(m,0) for m in models)))
    pd.DataFrame(rows).to_csv(d/"shared_weights_quality.csv",index=False)
    pd.DataFrame(deployments).to_csv(d/"shared_weights_cost.csv",index=False)
    np.savez_compressed(d/"shared_weights_scores.npz",labels=y,users=u,evaluation_ids=np.asarray(ds.evaluation_ids),**scores)
    return dict(rows=rows,deployments=deployments,notes=["Matched architecture; fixed models use the same frozen behavioral vectors if selected.",
        "Training times exclude prior shared representation search for specialists; shared time may include pretraining.",
        "Tensor footprint is not process peak RAM; offline scoring time is not HTTP latency."])
