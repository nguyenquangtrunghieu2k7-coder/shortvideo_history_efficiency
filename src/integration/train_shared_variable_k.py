"""Train ONE variable-history recommender with randomized K exposure.

Uses only frozen TRAIN for fitting and VALIDATION for checkpoint selection.
Never touches held-out test. Preserves vocabulary, labels, original checkpoints.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import roc_auc_score
from src.models import train_baseline as base

ROOT=base.ROOT
K_VALUES=[0,1,5,10,20,50,100,"Full"]
class VariableKDataset(Dataset):
    def __init__(self,ds,seed=42):
        self.ds=ds
        self.seed=seed
        self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.ds)
    def __getitem__(self,index):
        item=dict(self.ds[index])
        # Deterministic per-row, per-epoch independent of loader ordering.
        rng=random.Random(self.seed+1000003*self.epoch+9176*int(index))
        k=K_VALUES[rng.randrange(len(K_VALUES))]
        h=item["history_ids"]
        item["history_ids"]=h if k=="Full" else h[-int(k):] if k else []
        return item

def validation_auc_by_k(model,ds,device,batch_size):
    model.eval()
    scores={}
    labels=np.asarray(ds.labels,dtype=np.int8)
    with torch.inference_mode():
        for k in K_VALUES:
            preds=[]
            for start in range(0,len(ds),batch_size):
                batch=[]
                for i in range(start,min(start+batch_size,len(ds))):
                    item=dict(ds[i])
                    h=item["history_ids"]
                    item["history_ids"]=h if k=="Full" else h[-int(k):] if k else []
                    batch.append(item)
                candidate,history,mask,*_=base.collate_batch(batch)
                p=model(candidate.to(device),history.to(device),mask.to(device))
                preds.extend(p.detach().cpu().numpy().tolist())
            scores[str(k)]=float(roc_auc_score(labels,np.asarray(preds)))
    return scores

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--patience",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--lr",type=float,default=0.0005)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--init",choices=["scratch","frozen-full"],default="frozen-full")
    p.add_argument("--selection",choices=["mean_auc","full_auc"],default="mean_auc")
    p.add_argument("--output",type=Path,default=ROOT/"models/experimental_shared_variable_k.pth")
    args=p.parse_args()
    if args.epochs<1 or args.patience<1 or args.batch_size<2:raise ValueError("Invalid training arguments")
    base.set_seed(args.seed)
    vocab,vmap=base.load_video_vocab()
    train=base.KuaiRandDataset("train","Full",vmap)
    val=base.KuaiRandDataset("validation","Full",vmap)
    device=base.get_device()
    model=base.build_model(len(vocab),mode="full").to(device)
    if args.init=="frozen-full":
        source=ROOT/"models"/base.model_filename("full","Full")
        checkpoint=torch.load(source,map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Initialized from frozen Full checkpoint; original unchanged.",flush=True)
    training=VariableKDataset(train,args.seed)
    loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,
        num_workers=0,collate_fn=base.collate_batch)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    loss_fn=torch.nn.BCELoss()
    best=-float("inf");best_epoch=0;history=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        training.set_epoch(epoch)
        model.train()
        total=0.;n=0
        for candidate,hist,mask,labels,*_ in loader:
            optimizer.zero_grad(set_to_none=True)
            pred=model(candidate.to(device),hist.to(device),mask.to(device))
            loss=loss_fn(pred,labels.to(device))
            loss.backward()
            optimizer.step()
            total+=float(loss.item())*len(labels);n+=len(labels)
        auc=validation_auc_by_k(model,val,device,args.batch_size)
        metric=float(np.mean(list(auc.values()))) if args.selection=="mean_auc" else auc["Full"]
        entry={"epoch":epoch,"train_loss":total/max(n,1),"validation_auc_by_k":auc,
            "selection_metric":metric}
        history.append(entry)
        print(json.dumps(entry),flush=True)
        if np.isfinite(metric) and metric>best+1e-9:
            best=metric;best_epoch=epoch
            torch.save({"model_state_dict":model.state_dict(),
                "mode":"full","k":"variable","vocab_size":len(vocab),
                "embedding_dim":base.EMBEDDING_DIM,"hidden_dims":base.HIDDEN_DIMS,
                "epoch":epoch,"selection_metric":metric,
                "training_protocol":"uniform_random_K_per_training_example_each_epoch",
                "k_values":K_VALUES,"seed":args.seed,"init":args.init,
                "selection":args.selection},args.output)
        if epoch-best_epoch>=args.patience:
            print("Early stopping",flush=True);break
    report={"status":"TRAINED_VALIDATION_SELECTED_NOT_TESTED",
        "checkpoint":str(args.output),"best_epoch":best_epoch,
        "best_validation_selection_metric":best,
        "protocol":"TRAIN only, randomized K; VALIDATION only, checkpoint selection",
        "selection":args.selection,"init":args.init,"seed":args.seed,
        "epochs_requested":args.epochs,"history":history,
        "caveats":["Uniform random K does not guarantee Pareto superiority.",
            "Validation mean AUC weights K settings equally, not deployment requests.",
            "Do not tune further using test results.",
            "Checkpoint is experimental; original frozen models untouched."]}
    args.output.with_suffix(".training.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="history"},indent=2))
if __name__=="__main__":main()
