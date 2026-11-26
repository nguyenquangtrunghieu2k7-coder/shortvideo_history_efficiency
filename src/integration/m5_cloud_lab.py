"""Measured local process-separated HTTP + controlled link emulation.
Injected RTT and bandwidth delay per storage RPC; no claims of cloud deployment.
Closed-loop workloads, repeated randomized policy order, paired request IDs.
No recommender or selector training, no full offline inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import http.client
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
import pandas as pd
from src.integration import m3_product as p
from src.integration import m5_selector as s
from src.integration.m5_selector_experiment import Engine

class RemoteStore:
    def __init__(self,store,port,token,scenario):
        self.manifest=store.manifest;self.port=port;self.token=token;self.scenario=scenario
        self.local=threading.local();self.connections=[];self.lock=threading.Lock()
    def lookup(self,key,op,mode="indexed_exact",ranges=None):
        if mode!="indexed_exact":raise ValueError("Only indexed exact RPC supported")
        conn=getattr(self.local,"conn",None)
        if conn is None:
            conn=http.client.HTTPConnection("127.0.0.1",self.port,timeout=60)
            conn.connect();conn.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.local.conn=conn
            with self.lock:self.connections.append(conn)
        payload=json.dumps(dict(key=key,op=op,ranges=ranges),separators=(",",":")).encode()
        start=time.perf_counter()
        conn.request("POST","/lookup",payload,{"Authorization":"Bearer "+self.token,"Content-Type":"application/json"})
        response=conn.getresponse();body=response.read()
        if response.status!=200:raise RuntimeError("Storage HTTP "+str(response.status))
        out=json.loads(body)
        if out["snapshot"]!=self.manifest["id"]:raise RuntimeError("Snapshot mismatch")
        # Actual JSON body bytes, not logical compact index size. Per-flow
        # delay, NOT a shared bottleneck or TCP congestion simulation.
        injected=self.scenario["rtt_ms"]/1000+(len(payload)+len(body))*8/(self.scenario["mbps"]*1e6)
        if injected:time.sleep(injected)
        stats=out["stats"];stats["rpc_calls"]=1
        stats["rpc_body_bytes"]=len(payload)+len(body)
        stats["injected_delay_us"]=round(injected*1e6)
        stats["rpc_elapsed_us"]=round((time.perf_counter()-start)*1e6)
        return out["result"],stats
    def close(self):
        for c in self.connections:c.close()

def run(engine,requests,policy,pref,workers,token,port):
    def task(chunk):
        conn=http.client.HTTPConnection("127.0.0.1",port,timeout=120)
        conn.connect();conn.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        results=[]
        try:
            for req,expected in chunk:
                body=dict(req,policy=policy)
                if pref is not None:body["preferences"]=pref
                start=time.perf_counter()
                conn.request("POST","/recommend",json.dumps(body),{"Authorization":"Bearer "+token,"Content-Type":"application/json"})
                response=conn.getresponse();raw=response.read();elapsed=(time.perf_counter()-start)*1000
                if response.status!=200:raise RuntimeError("Recommendation HTTP "+str(response.status)+": "+str(raw[:150]))
                out=json.loads(raw)
                if abs(out["score"]-expected)>2e-5:raise RuntimeError("Prediction parity failed")
                results.append(dict(latency=elapsed,score=out["score"],action=out["action"],tokens=out["read_tokens"],
                    macs=out["linear_macs"],storage=out["storage"],timing=out["timing"]))
        finally:conn.close()
        return results
    chunks=[requests[i::workers] for i in range(workers)]
    # warm each policy separately outside timed workload
    task(requests[:1])
    start=time.perf_counter();cpu=time.process_time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out=[v for chunk in pool.map(task,chunks) for v in chunk]
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu
    stats=Counter()
    for row in out:stats.update(row["storage"])
    l=[v["latency"] for v in out]
    return dict(requests=len(out),errors=0,p50_ms=float(np.quantile(l,.5)),p95_ms=float(np.quantile(l,.95)),
        p99_ms=float(np.quantile(l,.99)),mean_ms=float(np.mean(l)),rps=len(out)/wall,
        recommendation_process_cpu_s=cpu,mean_tokens=float(np.mean([v["tokens"] for v in out])),
        mean_macs=float(np.mean([v["macs"] for v in out])),mean_logical_bytes=stats["logical_payload_bytes"]/len(out),
        mean_rpc_calls=stats["rpc_calls"]/len(out),mean_rpc_body_bytes=stats["rpc_body_bytes"]/len(out),
        mean_injected_ms=stats["injected_delay_us"]/len(out)/1000,
        mean_storage_ms=float(np.mean([v["timing"]["storage_ms"] for v in out])),
        mean_selector_ms=float(np.mean([v["timing"]["selector_ms"] for v in out])),
        mean_model_ms_including_lock_wait=float(np.mean([v["timing"]["model_ms"] for v in out])))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m4-dir",type=Path,default=Path("reports/m4_research_v3"))
    ap.add_argument("--m5-dir",type=Path,default=Path("reports/m5_release_v1"))
    ap.add_argument("--output",type=Path,default=Path("reports/m5_cloud_lab_v1"))
    ap.add_argument("--sample",type=int,default=128)
    ap.add_argument("--workers",type=int,nargs="+",default=[1,4,8])
    ap.add_argument("--repeats",type=int,default=2)
    a=ap.parse_args()
    if min(a.sample,a.repeats,*a.workers)<1:ap.error("Positive workload required")
    a.output.mkdir(parents=True,exist_ok=True)
    _,token=p.credentials()
    m4=a.m4_dir;m5=a.m5_dir
    ck=m4/(json.loads((m4/"summary.json").read_text())["selected_on_validation"]+".pt")
    engine=Engine(SimpleNamespace(checkpoint=ck,root=m4/"snapshots",bundle=m4/"int8.joblib",selector=m5/"selector.joblib"))
    local_store=engine.store
    ev=p.events_only()
    with np.load(m5/"predictions.npz",allow_pickle=False) as data:
        ids=np.asarray(data["evaluation_ids"]);index={int(v):i for i,v in enumerate(ids)}
        if len(index)!=len(ids) or set(index)!=set(map(int,ev.evaluation_id)):raise ValueError("Prediction IDs mismatch")
        ev=ev.sample(min(a.sample,len(ev)),random_state=42).reset_index(drop=True)
        pos=np.asarray([index[int(i)] for i in ev.evaluation_id])
        req=[dict(history_key=str(int(v.evaluation_id)),candidate_id=int(v.candidate),user_id=int(v.user_id)) for v in ev.itertuples(index=False)]
        plans={}
        for name in ("recent_0","recent_5","recent_50","recent_2048","q0.5","q0.9","q0.99"):
            if name.startswith("q"):
                pref=s.preference(float(name[1:]));scores=data[name][pos];policy="preference"
            else:
                pref=None;scores=data["action_predictions"][pos,s.ACTION_NAMES.index(name)];policy=name
            plans[name]=(policy,pref,list(zip(req,scores.tolist())))
    scenarios=[dict(name="local_direct",rtt_ms=0,mbps=10000),
        dict(name="rpc_loopback",rtt_ms=0,mbps=10000),
        dict(name="same_zone_fast",rtt_ms=.5,mbps=1000),
        dict(name="remote_moderate",rtt_ms=2,mbps=100),
        dict(name="remote_constrained",rtt_ms=10,mbps=10)]
    rows=[]
    with tempfile.TemporaryDirectory() as td:
        ready=Path(td)/"ready.json"
        proc=subprocess.Popen([sys.executable,"-m","src.integration.m5_remote_storage","--root",str(m4/"snapshots"),
            "--snapshot",engine.bundle["snapshot_id"],"--ready-file",str(ready)])
        server=None;thread=None
        try:
            deadline=time.monotonic()+120
            while not ready.exists():
                if proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError("Storage startup failed")
                time.sleep(.1)
            remote_port=json.loads(ready.read_text())["port"]
            server=p.make_server(engine,token,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            for scenario in scenarios:
                remote=None
                if scenario["name"]=="local_direct":engine.store=local_store
                else:
                    remote=RemoteStore(local_store,remote_port,token,scenario);engine.store=remote
                try:
                    for workers in a.workers:
                        for repeat in range(a.repeats):
                            order=list(plans);random.Random(42+repeat).shuffle(order)
                            for name in order:
                                policy,pref,requests=plans[name]
                                row=run(engine,requests,policy,pref,workers,token,server.server_port)
                                row.update(environment=scenario["name"],rtt_ms=scenario["rtt_ms"],mbps=scenario["mbps"],
                                    workers=workers,repeat=repeat+1,policy=name)
                                rows.append(row);print(json.dumps(row),flush=True)
                finally:
                    if remote:remote.close()
        finally:
            if server:server.shutdown();server.server_close()
            if thread:thread.join()
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    frame=pd.DataFrame(rows);frame.to_csv(a.output/"trials.csv",index=False)
    cols=["p50_ms","p95_ms","p99_ms","rps","mean_tokens","mean_macs","mean_logical_bytes","mean_rpc_calls","mean_rpc_body_bytes","mean_injected_ms","mean_storage_ms","mean_selector_ms","mean_model_ms_including_lock_wait","recommendation_process_cpu_s"]
    agg=frame.groupby(["environment","workers","policy"],as_index=False)[cols].median()
    agg.to_csv(a.output/"comparison.csv",index=False)
    comparisons=[]
    for (env,w),group in agg.groupby(["environment","workers"]):
        g=group.set_index("policy")
        for name in ("q0.5","q0.9","q0.99"):
            for base in ("recent_5","recent_50","recent_2048"):
                v,b=g.loc[name],g.loc[base]
                comparisons.append(dict(environment=env,workers=int(w),profile=name,baseline=base,
                    p95_ratio=float(v.p95_ms/b.p95_ms),throughput_ratio=float(v.rps/b.rps),
                    logical_byte_ratio=float(v.mean_logical_bytes/b.mean_logical_bytes),
                    token_ratio=float(v.mean_tokens/b.mean_tokens)))
    pd.DataFrame(comparisons).to_csv(a.output/"relative_tradeoffs.csv",index=False)
    summary=dict(status="EXECUTION_PASS",requests=int(frame.requests.sum()),errors=int(frame.errors.sum()),
        snapshot_id=engine.bundle["snapshot_id"],scenarios=scenarios,workers=a.workers,repeats=a.repeats,
        scope="Measured local HTTP with separate storage process and controlled per-RPC injected delay; NOT a cloud deployment",
        limitations=["Closed-loop workloads, no open-loop arrival SLA.","Per-flow independent bandwidth delay; no shared congestion model.",
            "JSON body bytes exclude HTTP/TCP/TLS framing; logical bytes are not wire bytes.",
            "Recommendation CPU excludes storage subprocess CPU; model lock serializes inference.",
            "Median of trial quantiles, not pooled p95.","No cloud price or measured cloud RTT claims."])
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(agg.to_string(index=False));print("REPORT:",a.output)
if __name__=="__main__":main()
