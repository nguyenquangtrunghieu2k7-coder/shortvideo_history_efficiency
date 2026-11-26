"""M3 offline shared training and validation-only budget calibration."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.integration.e2e_v4_bloom import baseline, ROOT
from src.integration.m3_ranker import Ranker, BUDGETS
from src.integration.m23_persistent_store import _sha

def batch_forward(model,b,device,mode,k):
    c,h,m,y,u,_=[x.to(device) for x in b]
    if mode=="recent":
        # Gather most recent k per example, not the padded right edge.
        n=m.sum(1).long()
        ix=(n[:,None]-k+torch.arange(k,device=device)).clamp(min=0,max=h.shape[1]-1)
        valid=(torch.arange(k,device=device)[None,:]>=k-n[:,None])
        h=h.gather(1,ix)*valid;m=valid
    return model(c,u,h,m,mode,k),y

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--dim",type=int,default=32)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path,default=ROOT/"models/m3_shared.pt")
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.dim)<1:raise ValueError("Invalid configuration")
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    vocab,vmap=baseline.load_video_vocab()
    train=baseline.KuaiRandDataset("train","Full",vmap)
    val=baseline.KuaiRandDataset("validation","Full",vmap)
    if set(train.evaluation_ids)&set(val.evaluation_ids):raise ValueError("Split overlap")
    device=baseline.get_device();model=Ranker(len(vocab),args.dim).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=baseline.collate_batch)
    vl=DataLoader(val,batch_size=args.batch_size,collate_fn=baseline.collate_batch)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    best=-float("inf")
    for ep in range(args.epochs):
        model.train();losses=[]
        for b in loader:
            mode=str(rng.choice(["recent","recent","prune","merge"]))
            k=int(rng.choice(BUDGETS))
            (score,aux,_),y=batch_forward(model,b,device,mode,k)
            loss=F.binary_cross_entropy_with_logits(score,y)+aux
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval();scores={}
        with torch.inference_mode():
            for k in (50,2048):
                pred=[]
                for b in vl:
                    (s,_,_),_=batch_forward(model,b,device,"recent",k)
                    pred.extend(torch.sigmoid(s).cpu().tolist())
                scores[str(k)]=float(roc_auc_score(val.labels,pred))
        quality=sum(scores.values())/2
        print(json.dumps({"epoch":ep+1,"loss":float(np.mean(losses)),"val_auc":scores}),flush=True)
        if quality>best:
            best=quality
            tmp=args.output.with_suffix(".tmp")
            torch.save({"state":model.state_dict(),"dim":args.dim,"vocab_size":len(vocab),
                        "vocab_sha256":_sha(baseline.VIDEO_VOCAB_FILE),"val_auc":scores,
                        "epoch":ep+1,"seed":args.seed,"budgets":BUDGETS},tmp)
            tmp.replace(args.output)
if __name__=="__main__":main()
