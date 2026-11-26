"""One-command M4 integration and ablation report. No sliding-window reruns.
Train/validation choose the model, test is assessment only. M3 files untouched."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import hashlib
import os
import platform
import resource
import numpy as np
import pandas as pd
import torch
from src.integration import m3_product as product
from src.integration.m23_persistent_store import _sha

def dump(path,data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,default=product.ROOT/"reports/m4_release_v1")
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--sample",type=int,default=0)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.repeats)<1 or a.sample<0:p.error("Invalid arguments")
    product.credentials();d=a.run_dir.resolve();d.mkdir(parents=True,exist_ok=True)
    config=dict(dataset_root=str(product.baseline.PROCESSED_DIR),device=str(product.baseline.get_device()),epochs=a.epochs,batch_size=a.batch_size,repeats=a.repeats,seed=a.seed,sample=a.sample,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        inputs={str(f):_sha(f) for f in (product.baseline.VIDEO_VOCAB_FILE,
            product.baseline.PROCESSED_DIR/"train.parquet",
            product.baseline.PROCESSED_DIR/"validation.parquet",
            product.baseline.PROCESSED_DIR/"test.parquet",
            product.baseline.PROCESSED_DIR/"evaluation_events.parquet",
            product.baseline.PHASE7_DIR/"history_KFull.parquet")})
    cf=d/"config.json"
    if cf.exists() and json.loads(cf.read_text())!=config:raise RuntimeError("Run configuration changed; choose a new --run-dir")
    dump(cf,config)
    variants=[("residual_only",False,False),("behavior",True,False),("behavior_distill",True,True)]
    model_rows=[]
    for name,behavior,distill in variants:
        ck=d/(name+".pt");marker=d/(name+".done.json")
        if not (marker.exists() and ck.exists() and json.loads(marker.read_text()).get("sha256")==_sha(ck)):
            t=time.perf_counter()
            cmd=[sys.executable,"-m","src.integration.m4_train","--output",str(ck),
                "--epochs",str(a.epochs),"--batch-size",str(a.batch_size),"--seed",str(a.seed)]
            if behavior:cmd.append("--behavior")
            if distill:cmd.append("--distill")
            subprocess.run(cmd,check=True)
            dump(marker,dict(sha256=_sha(ck),training_seconds=time.perf_counter()-t))
        c=torch.load(ck,map_location="cpu",weights_only=True)
        model_rows.append(dict(scenario=name,behavior=behavior,distill=distill,epoch=c["epoch"],
            selection_score=c["selection_score"],**c["val_auc"],**json.loads(marker.read_text())))
        del c
    models=pd.DataFrame(model_rows)
    models.to_csv(d/"training_ablation.csv",index=False)
    winner=max(model_rows,key=lambda x:x["selection_score"])["scenario"]
    print("VALIDATION SELECTED:",winner,flush=True)
    ck=d/(winner+".pt")
    results=[];index_rows=[]
    for name,compact in [("float32",False),("int8",True)]:
        args=SimpleNamespace(checkpoint=ck,root=d/"snapshots",bundle=d/(name+".joblib"),
            compact=compact,output=d/(name+".json"),sample=a.sample,repeats=a.repeats if compact else 1)
        marker=d/(name+"_prepared.json")
        if not (marker.exists() and args.bundle.exists() and json.loads(marker.read_text()).get("bundle_sha256")==_sha(args.bundle)):
            t=time.perf_counter();product.prepare(args)
            dump(marker,dict(bundle_sha256=_sha(args.bundle),prepare_seconds=time.perf_counter()-t))
        engine=product.Engine(args)
        if not args.output.exists():product.evaluate(args,engine)
        report=json.loads(args.output.read_text())
        if report["status"]!="PASS" or report["snapshot_id"]!=engine.bundle["snapshot_id"]:raise RuntimeError("Evaluation gate failed")
        index_rows.append(dict(encoding=name,bytes=sum(f.stat().st_size for f in engine.store.path.iterdir() if f.is_file()),
            **json.loads(marker.read_text())))
        for row in report["rows"]:results.append(dict(encoding=name,**row))
        del engine
    from src.integration.m4_shared_ablation import run as shared_run
    from src.integration.m4_bloom_ablation import run as bloom_run
    shared=shared_run(d,ck,a.epochs,a.batch_size,a.seed,a.sample)
    secret,_=product.credentials()
    int8_bundle=product.joblib.load(d/"int8.joblib")
    bloom=bloom_run(d/"snapshots",int8_bundle["snapshot_id"],secret,d/"bloom_ablation.csv",a.seed)
    dump(d/"shared_weights.json",shared)
    frame=pd.DataFrame(results)
    frame.to_csv(d/"policy_comparison.csv",index=False)
    pd.DataFrame(index_rows).to_csv(d/"index_comparison.csv",index=False)
    compact=frame[frame.encoding=="int8"].set_index("policy")
    full=compact.loc["full"];fixed=compact.loc["fixed50"];ad=compact.loc["adaptive"]
    gates=dict(full_test=a.sample==0,mean_read_tokens_le_50=bool(ad.mean_read_tokens<=50),
        point_auc_margin=bool(ad.auc-full.auc>=-.01),noninferiority_ci=bool(ad.delta_auc_ci95[0]>-.01),
        adaptive_auc_ge_fixed50=bool(ad.auc>=fixed.auc),
        p95_within_20pct_fixed50=bool(ad.p95_ms<=1.2*fixed.p95_ms),
        bloom_prediction_parity=True)
    summary=dict(status="EXECUTION_PASS",quality_targets=gates,selected_on_validation=winner,
        selection_reason="maximum mean validation AUC across seven predeclared policy configurations",
        adaptive_vs_full=dict(delta_auc=float(ad.auc-full.auc),read_token_reduction=float(1-ad.mean_read_tokens/full.mean_read_tokens),
            linear_mac_reduction=float(1-ad.mean_linear_macs/full.mean_linear_macs)),
        adaptive_vs_fixed50=dict(delta_auc=float(ad.auc-fixed.auc),p95_ratio=float(ad.p95_ms/fixed.p95_ms)),
        index_comparison=index_rows,scope="local HTTP; existing frozen disjoint history dataset; test reused for exploratory M4 assessment",
        unfinished=["point-in-time content metadata provenance and model integration","dynamic micro-batching",
            "open-loop burst and cold-start workload","physical disk IO and complete FLOPs accounting"])
    summary["environment"]={"platform":platform.platform(),"device":str(product.baseline.get_device()),"cpu_count":os.cpu_count(),"torch_version":str(torch.__version__),"scope":"local loopback; no cloud claims"}
    summary["shared_weights"]=shared["deployments"]
    summary["lazy_vs_eager"]={k:float(compact.loc["adaptive",k]/max(compact.loc["adaptive_eager",k],1e-12)) for k in ("mean_read_tokens","p95_ms","cpu_seconds")}
    summary["fused_vs_pipeline"]={k:float(compact.loc["adaptive",k]/max(compact.loc["adaptive_pipeline",k],1e-12)) for k in ("mean_selector_ms","p95_ms","cpu_seconds")}
    dump(d/"summary.json",summary)
    cols=["encoding","policy","auc","gauc","mean_read_tokens","mean_linear_macs","p95_ms","throughput_rps"]
    table=frame[cols].round(6)
    # Markdown tables without optional tabulate dependency.
    def md(df):
        headers=list(df.columns)
        return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"\n".join("| "+" | ".join(map(str,row))+" |" for row in df.itertuples(index=False,name=None))
    text="# M4 integrated comparison\n\nSelected on validation: **"+winner+"**.\n\n"
    text+="## Training ablation (validation only)\n\n"+md(models.drop(columns=["sha256"]).round(5))+"\n\n"
    text+="## Index ablation\n\n"+md(pd.DataFrame(index_rows).drop(columns=["bundle_sha256"]).round(5))+"\n\n"
    text+="## HTTP policy comparison (test assessment)\n\n"+md(table)+"\n\n"
    text+="## Why these scenarios\n\nResidual-only isolates the new model capacity; behavior adds co-occurrence vectors; behavior+distill adds a stopped-gradient full-history teacher. FP32 vs INT8 changes summary encoding and recalibrates the selector, so policy quality differences are joint effects. Full/fixed/random/semantic/adaptive/prune/merge/hybrid compare history operators with one checkpoint. Adaptive exact checks Bloom parity.\n\n"
    text+="## Acceptance targets\n\n"+md(pd.DataFrame([{"gate":k,"met":v} for k,v in gates.items()]))+"\n\n"
    text+="Execution PASS does not mean research targets passed. Timing is loopback single-request; linear MACs and logical bytes are partial cost accounting. Test has been previously inspected; confirmatory claims require a new independent holdout.\n"
    text+="\n## Shared weights (matched architecture)\n\n"+md(pd.DataFrame(shared["rows"])[["training","k","auc","gauc","delta_auc_vs_specialist","ci95_vs_specialist"]].round(5))+"\n\n"+md(pd.DataFrame(shared["deployments"]).round(4))+"\n\n"
    text+="Independent models specialize per budget; full-only tests truncation without variable-budget training; shared tests one checkpoint trained across budgets. Tensor footprint is not process RAM.\n\n"
    text+="## Bloom utility (actual local storage)\n\n"+md(pd.DataFrame(bloom)[["missing_fraction","mode","mean_exact_lookups","p95_local_ms","false_positives","filter_bytes","serial_rtt_2ms_mean_estimate"]].round(5))+"\n\n"
    text+="Known-shard routing and fan-out are different workloads. Remote RTT estimates assume serial lookups; they are sensitivity models, not measured cloud latency.\n\n"
    text+="## Lazy retrieval and selector optimization\n\n"+md(compact.reset_index().loc[lambda f:f.policy.isin(["adaptive","adaptive_eager","adaptive_pipeline"]),["policy","auc","mean_read_tokens","mean_k","mean_selector_ms","cpu_seconds","p95_ms"]].round(6))+"\n\n"
    text+="Eager reads all tokens before selecting the same positions; pipeline uses the original sklearn selector instead of fused linear inference. Both must match adaptive predictions. CPU time includes client and server in one process. Peak RSS is process lifetime high-water mark, not per-policy memory.\n"
    (d/"REPORT.md").write_text(text)
    print(table.to_string(index=False));print(json.dumps(summary,indent=2));print("REPORT:",d/"REPORT.md")

if __name__=="__main__":main()
