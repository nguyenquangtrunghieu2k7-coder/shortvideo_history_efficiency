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

def select_semantic(model,c,h,m,k):
    # Exactly the online block-centroid rule: rank blocks by learned cosine,
    # take the most recent positions of the final partial block.
    with torch.no_grad():
        block=32
        n=h.shape[1]
        pad=(-n)%block
        hp=F.pad(h,(0,pad));mp=F.pad(m.bool(),(0,pad))
        x=model.item(hp).reshape(len(h),-1,block,model.d)
        valid=mp.reshape(len(h),-1,block)
        cent=(x*valid[...,None]).sum(2)/valid.sum(2,keepdim=True).clamp_min(1)
        sim=F.cosine_similarity(cent,model.item(c)[:,None,:],dim=-1)
        sim=sim.masked_fill(~valid.any(2),-2.)
        order=sim.argsort(dim=1,descending=True,stable=True)
        rank=torch.zeros_like(order).scatter(1,order,torch.arange(order.shape[1],device=h.device)[None,:].expand_as(order))
        positions=torch.arange(hp.shape[1],device=h.device)
        # Stable ordering by block rank then reverse chronological position.
        priority=rank[:,positions//block]*block+(block-1-positions%block)
        priority=priority.masked_fill(~mp,10**8)
        ix=priority.argsort(dim=1,stable=True)[:,:min(k,n)].sort(1).values
        mask=mp.gather(1,ix)
        return hp.gather(1,ix)*mask,mask

def batch_forward(model,b,device,mode,k):
    c,h,m,y,u,_=[x.to(device) for x in b]
    if mode=="recent":
        k=min(k,h.shape[1])
        n=m.sum(1).long()
        ix=(n[:,None]-k+torch.arange(k,device=device)).clamp(min=0,max=h.shape[1]-1)
        valid=(torch.arange(k,device=device)[None,:]>=k-n[:,None])
        h=h.gather(1,ix)*valid;m=valid
    elif mode in ("semantic","prune","merge"):
        read=k if mode=="semantic" else min(512,2*k)
        h,m=select_semantic(model,c,h,m,read)
    op="recent" if mode=="semantic" else mode
    return model(c,u,h,m,op,k),y

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--dim",type=int,default=32)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--patience",type=int,default=5)
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
    best=-float("inf"); stale=0
    for ep in range(args.epochs):
        model.train();losses=[]
        for b in loader:
            mode=str(rng.choice(["recent","recent","semantic","semantic","prune","merge"]))
            k=int(rng.choice(BUDGETS if mode=="recent" else BUDGETS[:-1]))
            (score,aux,_),y=batch_forward(model,b,device,mode,k)
            loss=F.binary_cross_entropy_with_logits(score,y)+aux
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval();scores={}
        with torch.inference_mode():
            for mode,k in (("recent",50),("recent",2048),("semantic",50),("semantic",128),("prune",50),("merge",50)):
                pred=[]
                for b in vl:
                    (s,_,_),_=batch_forward(model,b,device,mode,k)
                    pred.extend(torch.sigmoid(s).cpu().tolist())
                scores[f"{mode}_{k}"]=float(roc_auc_score(val.labels,pred))
        quality=sum(scores.values())/len(scores)
        print(json.dumps({"epoch":ep+1,"loss":float(np.mean(losses)),"val_auc":scores}),flush=True)
        if quality>best:
            best=quality;stale=0
            tmp=args.output.with_suffix(".tmp")
            torch.save({"state":model.state_dict(),"dim":args.dim,"vocab_size":len(vocab),
                        "vocab_sha256":_sha(baseline.VIDEO_VOCAB_FILE),"val_auc":scores,
                        "epoch":ep+1,"seed":args.seed,"budgets":BUDGETS},tmp)
            tmp.replace(args.output)
        else:
            stale+=1
            if stale>=args.patience:break
if __name__=="__main__":main()
