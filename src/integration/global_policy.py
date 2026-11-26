"""Global policy compiler: no per-request learned model or summary reads.
A validation table describes expected resource and BCE for each original K.
Hard caps apply to nominal K (safe for any observed length); average budget
is an expectation under the calibration distribution, not a fleet SLA."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import numpy as np
from src.integration import m5_selector as s

KS=tuple(s.ORIGINAL_BUDGETS)
def compile_policy(table,doc):
    if not isinstance(doc,dict) or set(doc)!={"revision","preferences","average_tokens"} or type(doc["revision"]) is not int or doc["revision"]<1:
        raise ValueError("Invalid policy document")
    pref=s.preference(**doc["preferences"])
    if pref["grid"]!="original":raise ValueError("Global release uses original grid")
    budget=doc["average_tokens"]
    if budget is not None and (isinstance(budget,bool) or not isinstance(budget,(int,float)) or not math.isfinite(budget) or not 0<=budget<=2048):raise ValueError("Invalid average_tokens")
    rows=table["rows"]
    if tuple(row["k"] for row in rows)!=KS:raise ValueError("Invalid calibration grid")
    k=np.asarray(KS);n=np.asarray([row["mean_tokens"] for row in rows]);loss=np.asarray([row["mean_bce"] for row in rows])
    mac=np.asarray([row["mean_macs"] for row in rows])
    cost=(np.column_stack((n,16*n,mac))/np.array([50,800,s.HEAD_MACS+s.TOKEN_MACS*50]))@np.asarray(pref["cost_weights"])
    utility=pref["quality_weight"]*(loss-loss[-1])/s.QUALITY_SCALE+pref["cost_weight"]*cost
    valid=k<=pref["max_read_tokens"]
    if pref["max_logical_bytes"] is not None:valid &= 16*k<=pref["max_logical_bytes"]
    if pref["max_linear_macs"] is not None:valid &= s.HEAD_MACS+s.TOKEN_MACS*np.maximum(k,1)<=pref["max_linear_macs"]
    ix=np.flatnonzero(valid);opts=[]
    for i in ix:
        if budget is None or n[i]<=budget+1e-10:opts.append((float(utility[i]),float(cost[i]),int(i),int(i),0.))
    if budget is not None:
        for i in ix:
            for j in ix:
                if n[i]<budget<n[j]:
                    w=float((budget-n[i])/(n[j]-n[i]))
                    opts.append((float((1-w)*utility[i]+w*utility[j]),float((1-w)*cost[i]+w*cost[j]),int(i),int(j),w))
    if not opts:raise ValueError("No feasible global policy")
    obj,c,i,j,w=min(opts)
    return dict(revision=doc["revision"],preferences=pref,average_tokens=budget,k_a=KS[i],k_b=KS[j],prob_b=w,
        expected_tokens=float((1-w)*n[i]+w*n[j]),expected_bce=float((1-w)*loss[i]+w*loss[j]),expected_cost=c,objective=obj)

def select(plan,key):
    # Stable label-independent assignment across revisions and workers.
    h=int.from_bytes(hashlib.sha256(("global-v1:"+key).encode()).digest()[:8],"big")
    return plan["k_b"] if h < int(plan["prob_b"]*2**64) else plan["k_a"]

class PolicyStore:
    def __init__(self,path,table):
        self.path=Path(path);self.table=table;self.lock=threading.Lock();self.active=None;self.digest=None
        self.get()
    def get(self):
        with self.lock:
            error=False
            try:
                raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                if digest!=self.digest:
                    plan=compile_policy(self.table,json.loads(raw))
                    if self.active is not None and plan["revision"]<=self.active["revision"]:raise ValueError("Revision must increase")
                    self.active=plan;self.digest=digest
            except (OSError,ValueError,TypeError,KeyError):
                if self.active is None:raise
                error=True
            return self.active,error

def publish(path,table,doc):
    compile_policy(table,doc);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and doc["revision"]<=json.loads(path.read_text())["revision"]:raise ValueError("Revision must increase")
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=".policy-")
    try:
        with os.fdopen(fd,"w") as f:json.dump(doc,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
