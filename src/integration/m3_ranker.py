"""Shared candidate-aware ranker and compression operators for M3.

IDs are categorical; semantic scores use learned embeddings, never raw ID distance.
All policies share exactly one checkpoint. Training is offline.
"""
import math
import torch
from torch import nn

BUDGETS = (16, 32, 50, 64, 128, 256, 512, 2048)

class Ranker(nn.Module):
    def __init__(self, vocab_size, d=32, users=4096):
        super().__init__()
        self.d = d
        self.item = nn.Embedding(vocab_size + 1, d, padding_idx=0)
        self.user = nn.Embedding(users, d)
        self.gate = nn.Sequential(nn.Linear(4*d+1, 64), nn.GELU(), nn.Linear(64, 1))
        self.attn = nn.Sequential(nn.Linear(4*d+1, 64), nn.GELU(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(4*d+1, 128), nn.LayerNorm(128),
                                  nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1))
        self.merge_threshold = nn.Linear(2*d, 1)

    def forward(self, candidate, user, history, mask, mode="recent", budget=50):
        x = self.item(history)
        c = self.item(candidate)
        u = self.user(user.remainder(self.user.num_embeddings))
        mask = mask.bool()
        pos = (mask.sum(1, keepdim=True)-mask.long().cumsum(1)).clamp(min=0)
        pos = torch.log1p(pos.float()) / math.log(2049)
        def features(x, pos):
            cc = c[:,None,:].expand_as(x)
            uu = u[:,None,:].expand_as(x)
            return torch.cat((x, cc, uu, x*cc, pos[...,None]), -1)
        aux = x.new_zeros(())
        if mode == "prune":
            logits = self.gate(features(x,pos)).squeeze(-1)
            soft = torch.sigmoid(logits) * mask
            count = min(budget, x.shape[1])
            idx = logits.masked_fill(~mask,-1e9).topk(count,dim=1).indices.sort(1).values
            hard = torch.zeros_like(soft).scatter(1,idx,1.) * mask
            if self.training:
                # Straight-through gate; dense training, hard sparse serving.
                weights = hard + soft - soft.detach()
                aux = .02*(soft.sum(1)/mask.sum(1).clamp_min(1)-hard.sum(1)/mask.sum(1).clamp_min(1)).square().mean()
            else:
                x = x.gather(1,idx[...,None].expand(-1,-1,self.d))
                pos = pos.gather(1,idx)
                mask = mask.gather(1,idx)
                weights = mask.float()
        elif mode == "merge":
            # Adjacent pair rate-distortion: semantic distortion versus token rate.
            if x.shape[1] % 2:
                x = torch.cat((x,torch.zeros_like(x[:,:1])),1)
                pos = torch.cat((pos,torch.zeros_like(pos[:,:1])),1)
                mask = torch.cat((mask,torch.zeros_like(mask[:,:1])),1)
            a,b=x[:,::2],x[:,1::2]
            ma,mb=mask[:,::2],mask[:,1::2]
            sim=nn.functional.cosine_similarity(a,b,dim=-1)
            threshold=torch.tanh(self.merge_threshold(torch.cat((c,u),-1)))
            soft=torch.sigmoid(8*(sim-threshold))* (ma&mb)
            hard=(sim>=threshold).float()*(ma&mb)
            z=hard+soft-soft.detach() if self.training else hard
            merged=(a+b)/2
            left=a+z[...,None]*(merged-a)
            x=torch.stack((left,b),2).flatten(1,2)
            weights=torch.stack((ma.float(),mb.float()*(1-z)),2).flatten(1,2)
            aux=.01*((1-sim)*soft).mean()+.01*(1-soft).mean()
            if not self.training and len(x)==1:
                keep=weights[0]>0
                if not keep.any(): keep[0]=True
                x=x[:,keep];pos=pos[:,keep];weights=weights[:,keep]
        else:
            weights=mask.float()
        logits=self.attn(features(x,pos)).squeeze(-1).clamp(-20,20)
        # Stable masked attention, including an empty history.
        w=torch.exp(logits)*weights
        pooled=(w[...,None]*x).sum(1)/w.sum(1,keepdim=True).clamp_min(1e-8)
        length=weights.detach().sum(1,keepdim=True)
        score=self.head(torch.cat((c,u,pooled,c*pooled,torch.log1p(length)/math.log(2049)),-1)).squeeze(-1)
        return score,aux,length.squeeze(-1)
