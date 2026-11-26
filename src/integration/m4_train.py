"""M4 offline training. Teacher is the same shared model, stopped-gradient.
Behavioral pretraining uses train histories only; no content metadata is assumed safe."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.integration.e2e_v4_bloom import baseline, ROOT
from src.integration.m4_behavior import behavioral_vectors
from src.integration.m4_ranker import Ranker
from src.integration.m3_train import select_semantic
from src.integration.m23_persistent_store import _sha

def forward(model,b,device,mode,k):
    c,h,m,y,u,_=[x.to(device) for x in b]
    if mode=="recent":
        k=min(k,h.shape[1]);n=m.sum(1).long()
        ix=(n[:,None]-k+torch.arange(k,device=device)).clamp(0,h.shape[1]-1)
        valid=torch.arange(k,device=device)[None,:]>=k-n[:,None]
        h=h.gather(1,ix)*valid;m=valid
    else:
        h,m=select_semantic(model,c,h,m,k if mode=="semantic" else min(512,2*k))
    return model(c,u,h,m,"recent" if mode=="semantic" else mode,k),y

def train(args):
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    vocab,vmap=baseline.load_video_vocab()
    tr=baseline.KuaiRandDataset("train","Full",vmap)
    va=baseline.KuaiRandDataset("validation","Full",vmap)
    if set(tr.evaluation_ids)&set(va.evaluation_ids):raise ValueError("Split overlap")
    device=baseline.get_device()
    model=Ranker(len(vocab),32)
    if args.behavior:
        vec,stats=behavioral_vectors(tr.history_ids,len(vocab),32,args.seed)
        model.item.weight.copy_(torch.from_numpy(vec));del vec
    else:stats={"active_items":0}
    print(json.dumps({"behavioral_pretraining":stats}),flush=True)
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
    loader=DataLoader(tr,batch_size=args.batch_size,shuffle=True,collate_fn=baseline.collate_batch)
    vl=DataLoader(va,batch_size=args.batch_size,collate_fn=baseline.collate_batch)
    best=-1.;stale=0;history=[]
    conditions=[("recent",50),("recent",2048),("semantic",50),("semantic",128),("prune",50),("merge",50),("hybrid",50)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for ep in range(args.epochs):
        model.train();losses=[]
        for b in loader:
            mode=str(rng.choice(["recent","recent","semantic","semantic","prune","merge","hybrid"]))
            k=int(rng.choice([16,32,50,64,128,256,512,2048] if mode=="recent" else [16,32,50,64,128,256]))
            (z,aux,_),y=forward(model,b,device,mode,k)
            loss=F.binary_cross_entropy_with_logits(z,y)+aux
            if args.distill and k<2048 and ep>0:
                model.eval()
                with torch.no_grad():
                    (teacher,_,_),_=forward(model,b,device,"recent",2048)
                model.train()
                loss=loss+.25*F.binary_cross_entropy_with_logits(z,torch.sigmoid(teacher))
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach().cpu()))
        model.eval();scores={}
        with torch.inference_mode():
            for mode,k in conditions:
                pred=[]
                for b in vl:
                    (z,_,_),_=forward(model,b,device,mode,k)
                    pred.extend(torch.sigmoid(z).cpu().tolist())
                scores[f"{mode}_{k}"]=float(roc_auc_score(va.labels,pred))
        quality=float(np.mean(list(scores.values())))
        row={"epoch":ep+1,"loss":float(np.mean(losses)),"val_auc":scores,"selection_score":quality}
        history.append(row);print(json.dumps(row),flush=True)
        if quality>best:
            best=quality;stale=0
            tmp=args.output.with_suffix(".tmp")
            torch.save(dict(state=model.state_dict(),dim=32,vocab_size=len(vocab),vocab_sha256=_sha(baseline.VIDEO_VOCAB_FILE),
                architecture="m4",epoch=ep+1,seed=args.seed,behavior=args.behavior,distill=args.distill,
                behavioral_stats=stats,val_auc=scores,selection_score=quality),tmp)
            tmp.replace(args.output)
        else:
            stale+=1
            if stale>=args.patience:break
    args.output.with_suffix(".training.json").write_text(json.dumps(history,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--patience",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--behavior",action="store_true")
    p.add_argument("--distill",action="store_true")
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.patience)<1:p.error("Invalid training arguments")
    train(a)
if __name__=="__main__":main()
