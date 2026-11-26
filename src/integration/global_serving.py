"""Production-oriented reference engine: one bounded recent lookup per request.
No contextual feature extraction, selector model, or semantic embedding export."""
import argparse,json
from pathlib import Path
import threading
from time import perf_counter
import torch
from src.integration import m3_product as p,m5_selector as s
from src.integration.global_policy import PolicyStore,select,compile_policy
from src.integration.m3_store import Store
from src.integration.m23_persistent_store import _sha

class Engine:
    def __init__(self,checkpoint,root,table,policy_file=None):
        secret,_=p.credentials()
        self.table=table;self.bundle={"snapshot_id":table["snapshot_id"]}
        if table["checkpoint_sha256"]!=_sha(checkpoint):raise ValueError("Calibration/checkpoint mismatch")
        self.store=Store(root,table["snapshot_id"],secret)
        if self.store.manifest["metadata"]["checkpoint_sha256"]!=table["checkpoint_sha256"]:raise ValueError("Snapshot/checkpoint mismatch")
        if self.store.manifest["metadata"]["vocab_sha256"]!=_sha(p.baseline.VIDEO_VOCAB_FILE):raise ValueError("Snapshot/vocab mismatch")
        self.device=p.baseline.get_device();self.model=p.load_model(checkpoint,self.device)
        if not hasattr(self.model.item,"residual") or self.model.d!=32:raise ValueError("Expected M4 ranker")
        self.lock=threading.Lock();self.macs=0
        for layer in self.model.modules():
            if isinstance(layer,torch.nn.Linear):layer.register_forward_hook(self.count)
        self.policy_store=PolicyStore(policy_file,table) if policy_file else None
        self.plan=None;self.fixed_k=None # benchmark only, never client-controlled
    def count(self,layer,inputs,out):
        self.macs+=out.numel()*layer.in_features
    def score(self,key,c,u,policy):
        start=perf_counter()
        if policy!="preference" or not 0<c<len(self.model.item.weight) or u<0:raise ValueError("Invalid request")
        if self.policy_store:
            plan,error=self.policy_store.get();k=select(plan,key)
        elif self.fixed_k is not None:
            plan=None;error=False;k=self.fixed_k
        else:
            plan=self.plan;error=False;k=select(plan,key)
        selector_ms=(perf_counter()-start)*1000
        t=perf_counter();result,stats=self.store.lookup(key,"recent",hi=k,mode="indexed_exact")
        storage_ms=(perf_counter()-t)*1000
        if result is None:raise KeyError("History not found")
        n=result["n"];eff=min(n,k);tokens=result["tokens"]
        if len(tokens)!=eff or stats["token_rows"]!=eff or stats["summary_rows"]!=0 or stats["logical_payload_bytes"]!=16*eff or stats["sqlite_lookups"]!=1:raise RuntimeError("Bounded retrieval invariant")
        d=self.device;t=perf_counter()
        with self.lock,torch.inference_mode():
            self.macs=0
            h=torch.tensor([[v for _,v in tokens] or [0]],device=d)
            z,_,count=self.model(torch.tensor([c],device=d),torch.tensor([u],device=d),h,h.ne(0),"recent",max(eff,1))
            score=float(torch.sigmoid(z).cpu().item());used=int(count.cpu().item());macs=self.macs;p.sync(d)
        model_ms=(perf_counter()-t)*1000
        if used!=eff or macs!=s.HEAD_MACS+s.TOKEN_MACS*max(eff,1):raise RuntimeError("Model accounting mismatch")
        if plan:
            pref=plan["preferences"]
            if eff>pref["max_read_tokens"] or (pref["max_logical_bytes"] is not None and stats["logical_payload_bytes"]>pref["max_logical_bytes"]) or (pref["max_linear_macs"] is not None and macs>pref["max_linear_macs"]):raise RuntimeError("Hard cap violated")
        return dict(score=score,action=f"recent_{k}",budget=k,read_tokens=eff,model_tokens=used,linear_macs=macs,
            history_length=n,policy_revision=plan["revision"] if plan else None,policy_reload_error=error,
            storage=stats,timing=dict(selector_ms=selector_ms,storage_ms=storage_ms,model_ms=model_ms),server_ms=(perf_counter()-start)*1000)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--table",type=Path,required=True);ap.add_argument("--policy-file",type=Path,required=True)
    ap.add_argument("--bind",default="127.0.0.1");ap.add_argument("--port",type=int,default=8778)
    a=ap.parse_args();table=json.loads(a.table.read_text())
    engine=Engine(a.checkpoint,a.root,table,a.policy_file)
    _,token=p.credentials();server=p.make_server(engine,token,a.port,host=a.bind)
    print("READY",a.bind,a.port,engine.bundle["snapshot_id"],flush=True)
    try:server.serve_forever()
    finally:server.server_close()
if __name__=="__main__":main()
