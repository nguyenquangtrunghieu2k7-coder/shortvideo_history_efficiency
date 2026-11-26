"""Shared M4 ranker: frozen behavioral vectors + low-capacity categorical residual,
two target-aware interests, differentiable pruning and pairwise rate-distortion."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Item(nn.Module):
    def __init__(self, n, d, buckets=32768):
        super().__init__()
        self.register_buffer("weight",torch.zeros(n+1,d))
        self.residual=nn.Embedding(buckets,d,padding_idx=0)
        nn.init.normal_(self.residual.weight,std=.02)
        with torch.no_grad():self.residual.weight[0].zero_()
    def forward(self,ids):
        bucket=torch.where(ids>0,1+(ids-1)%(self.residual.num_embeddings-1),0)
        return self.weight[ids]+self.residual(bucket)

class Ranker(nn.Module):
    def __init__(self,vocab_size,d=32):
        super().__init__()
        self.d=d;self.item=Item(vocab_size,d);self.user=nn.Embedding(4096,d)
        nn.init.normal_(self.user.weight,std=.05)
        self.gate=nn.Sequential(nn.Linear(4*d+1,32),nn.GELU(),nn.Linear(32,1))
        self.attn=nn.Sequential(nn.Linear(4*d+1,48),nn.GELU(),nn.Linear(48,2))
        self.merge_threshold=nn.Linear(2*d,1)
        self.head=nn.Sequential(nn.Linear(7*d+1,96),nn.LayerNorm(96),nn.GELU(),nn.Dropout(.25),nn.Linear(96,1))
    def forward(self,candidate,user,history,mask,mode="recent",budget=50):
        x=self.item(history);c=self.item(candidate);u=self.user(user%4096)
        mask=mask.bool();weights=mask.float()
        pos=torch.log1p((mask.sum(1,keepdim=True)-mask.long().cumsum(1)).clamp_min(0).float())/math.log(2049)
        aux=x.new_zeros(())
        def feat(x,p):
            cc=c[:,None,:].expand_as(x);uu=u[:,None,:].expand_as(x)
            return torch.cat((x,cc,uu,x*cc,p[...,None]),-1)
        def compact(x,p,w):
            keep=w.detach()>0
            n=max(1,int(keep.sum(1).max().item()))
            ix=(~keep).long().argsort(dim=1,stable=True)[:,:n]
            return x.gather(1,ix[...,None].expand(-1,-1,self.d)),p.gather(1,ix),w.gather(1,ix)
        if mode in ("prune","hybrid"):
            logits=self.gate(feat(x,pos)).squeeze(-1)
            soft=torch.sigmoid(logits)*mask
            k=min(max(1,budget),x.shape[1])
            ix=logits.masked_fill(~mask,-1e9).topk(k,dim=1).indices.sort(1).values
            hard=torch.zeros_like(soft).scatter(1,ix,1.)*mask
            aux=.02*((soft.sum(1)-hard.sum(1))/mask.sum(1).clamp_min(1)).square().mean()
            if self.training and mode=="prune":
                weights=hard+soft-soft.detach()
            else:
                weights=(hard+soft-soft.detach() if self.training else hard).gather(1,ix)
                x=x.gather(1,ix[...,None].expand(-1,-1,self.d));pos=pos.gather(1,ix)
        if mode in ("merge","hybrid"):
            # Merge consecutive *retrieved* tokens, not necessarily consecutive raw events.
            if x.shape[1]%2:
                x=F.pad(x,(0,0,0,1));pos=F.pad(pos,(0,1));weights=F.pad(weights,(0,1))
            a,b=x[:,::2],x[:,1::2];wa,wb=weights[:,::2],weights[:,1::2]
            valid=(wa.detach()>0)&(wb.detach()>0)
            sim=F.cosine_similarity(a,b,dim=-1)
            threshold=torch.tanh(self.merge_threshold(torch.cat((c,u),-1)))
            soft=torch.sigmoid(8*(sim-threshold))*valid
            hard=(sim>=threshold).float()*valid
            z=hard+soft-soft.detach() if self.training else hard
            left=a+z[...,None]*((a+b)*.5-a)
            x=torch.stack((left,b),2).flatten(1,2)
            weights=torch.stack((wa,wb*(1-z)),2).flatten(1,2)
            aux=aux+.01*((1-sim)*soft).sum()/valid.sum().clamp_min(1)+.01*(valid.float()-soft).sum()/valid.sum().clamp_min(1)
        if not self.training and mode in ("prune","merge","hybrid"):
            x,pos,weights=compact(x,pos,weights)
        logits=self.attn(feat(x,pos)).clamp(-20,20)
        w=torch.exp(logits)*weights[...,None]
        pooled=torch.einsum("bnh,bnd->bhd",w,x)/w.sum(1)[...,None].clamp_min(1e-8)
        p1,p2=pooled[:,0],pooled[:,1]
        length=weights.detach().sum(1,keepdim=True)
        z=self.head(torch.cat((c,u,p1,p2,c*p1,c*p2,p1*p2,torch.log1p(length)/math.log(2049)),-1)).squeeze(-1)
        return z,aux,length.squeeze(-1)
