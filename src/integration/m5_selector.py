"""Preference-conditioned action selector. Pure NumPy decision core.
Quality = predicted excess binary cross-entropy relative to Full, not AUC.
Cost = normalized resource proxies, not money or latency guarantees.
All preferences are runtime inputs; no model retraining on preference changes."""
import math
import numpy as np

BUDGETS=(16,32,50,64,128,256,512)
ACTIONS=tuple((mode,k) for k in BUDGETS for mode in ("recent","semantic"))+(("recent",2048),)
ACTION_NAMES=tuple(f"{m}_{k}" for m,k in ACTIONS)
QUALITY_SCALE=.01
PROFILES=(0.,.25,.5,.75,.9,.95,.99,1.)
DEFAULT_COST_WEIGHTS=(1.,1.,1.)
# M4 ranker: attention 129*48 + 48*2 per input, head 225*96 + 96.
TOKEN_MACS=6288
HEAD_MACS=21696

def preference(quality_weight=.9,cost_weight=None,cost_weights=DEFAULT_COST_WEIGHTS,max_read_tokens=2048,max_linear_macs=None):
    if cost_weight is None:cost_weight=1-float(quality_weight)
    vals=np.asarray([quality_weight,cost_weight],dtype=float)
    cw=np.asarray(cost_weights,dtype=float)
    if vals.shape!=(2,) or cw.shape!=(3,) or not np.isfinite(vals).all() or not np.isfinite(cw).all() or (vals<0).any() or (cw<0).any() or vals.sum()<=0 or cw.sum()<=0:
        raise ValueError("Weights must be finite nonnegative, with positive sum")
    if type(max_read_tokens) is not int or not 0<=max_read_tokens<=2048:raise ValueError("Invalid max_read_tokens")
    if max_linear_macs is not None and (type(max_linear_macs) is not int or max_linear_macs<0):raise ValueError("Invalid max_linear_macs")
    vals/=vals.sum();cw/=cw.sum()
    return dict(quality_weight=float(vals[0]),cost_weight=float(vals[1]),cost_weights=cw.tolist(),
        max_read_tokens=max_read_tokens,max_linear_macs=max_linear_macs)

def resource_table(n,dim=32,compact=True,selector=True):
    n=np.asarray(n,dtype=np.int64).reshape(-1)
    k=np.minimum(n[:,None],np.asarray([k for _,k in ACTIONS])[None,:])
    summaries=((n+31)//32)*(16+dim*(1 if compact else 4))
    # Selector fetches summaries once, independent of the selected action.
    if selector:overhead=summaries[:,None]
    else:overhead=summaries[:,None]*np.asarray([m=="semantic" for m,_ in ACTIONS])[None,:]
    return np.stack((k,16*k+overhead,HEAD_MACS+TOKEN_MACS*np.maximum(k,1)),axis=-1)

def cost_table(resources,pref):
    return (resources/np.asarray([50.,800.,HEAD_MACS+TOKEN_MACS*50]))@np.asarray(pref["cost_weights"])

def choose(regrets,resources,pref):
    regrets=np.asarray(regrets,dtype=float)
    if regrets.shape!=resources.shape[:2] or not np.isfinite(regrets).all():raise ValueError("Invalid quality estimates")
    cost=cost_table(resources,pref)
    feasible=resources[:,:,0]<=pref["max_read_tokens"]
    if pref["max_linear_macs"] is not None:feasible &= resources[:,:,2]<=pref["max_linear_macs"]
    if not feasible.any(1).all():raise ValueError("No feasible history action under requested constraints")
    utility=pref["quality_weight"]*regrets/QUALITY_SCALE+pref["cost_weight"]*cost
    utility=np.where(feasible,utility,np.inf)
    # Stable ties: lower resource cost, then canonical action index.
    minimum=utility.min(1,keepdims=True)
    ties=utility<=minimum+1e-12
    return np.argmin(np.where(ties,cost,np.inf),axis=1)

def loss_matrix(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    y=np.asarray(y,dtype=float)[:,None]
    return -(y*np.log(p)+(1-y)*np.log1p(-p))

def feature_vector(n,blocks,q,u):
    # Only compact summaries, candidate and user representations; no raw history.
    x=np.asarray([v[2] for v in blocks],dtype=np.float32)
    if len(x):
        s=(x@q)/np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(q),1e-8)
        extra=[s.max(),s.mean(),s.std(),s[-1],np.quantile(s,.75),np.quantile(s,.25),
               np.mean(s>0),s.max()-s[-1],np.mean(s[-min(4,len(s)):])]
    else:extra=[0.]*9
    return np.r_[np.log1p(n)/np.log(2049),extra,q,u].astype(np.float32)
