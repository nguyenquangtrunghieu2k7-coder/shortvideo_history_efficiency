"""VK-LSVD subset replay. Official train weeks 20..24 and official validation week 25.
Split week 25 at its middle row into our validation/test, before any sampling.
The published files are chronologically ordered, but no event timestamp column
is exposed. time_ms in the downstream compatibility schema is an ordinal,
NOT a physical time or strict timestamp-tie guarantee.
The official popular-user/item subset is biased; no full-platform claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.integration.kuairec_adapter import sha

SUBSET="up0.001_ip0.001"
FILES=[f"subsamples/{SUBSET}/train/week_{i:02}.parquet" for i in range(20,25)]+[f"subsamples/{SUBSET}/validation/week_25.parquet"]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",type=Path,default=Path("data/vklsvd/raw"))
    ap.add_argument("--output",type=Path,default=Path("data/vklsvd/processed"))
    ap.add_argument("--download",action="store_true")
    ap.add_argument("--train",type=int,default=30118)
    ap.add_argument("--validation",type=int,default=6454)
    ap.add_argument("--test",type=int,default=6455)
    a=ap.parse_args()
    if min(a.train,a.validation,a.test)<1:ap.error("positive sample sizes required")
    raw=a.raw.resolve();d=a.output.resolve()
    if d.name!="processed" or d.parent.name!="vklsvd":raise ValueError("Use isolated data/vklsvd/processed")
    if a.download:
        from huggingface_hub import HfApi,hf_hub_download
        raw.mkdir(parents=True,exist_ok=True);lock=raw/"source_revision.json"
        if lock.exists():rev=json.loads(lock.read_text())["revision"]
        else:
            rev=HfApi().dataset_info("deepvk/VK-LSVD").sha
            lock.write_text(json.dumps(dict(repo="deepvk/VK-LSVD",revision=rev),indent=2))
        for name in FILES:
            print("DOWNLOAD",name,flush=True)
            hf_hub_download(repo_id="deepvk/VK-LSVD",repo_type="dataset",filename=name,revision=rev,local_dir=raw)
    paths=[raw/name for name in FILES]
    for path in paths:
        if not path.is_file():raise FileNotFoundError(str(path)+"; rerun with --download")
    config=dict(source_files={name:sha(path) for name,path in zip(FILES,paths)},
        source_revision=json.loads((raw/"source_revision.json").read_text()) if (raw/"source_revision.json").exists() else None,
        subset=SUBSET,seed=42,counts=[a.train,a.validation,a.test],label="timespent >= 10 seconds",schema=1,
        temporal_protocol="official train weeks 20-24; official validation week 25 split at middle row")
    manifest=d/"adapter_manifest.json"
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old["config"]!=config:raise ValueError("Adapter configuration mismatch")
        for name,digest in old["artifacts"].items():
            if sha(d/name)!=digest:raise ValueError("Adapter artifact changed: "+name)
        print("ADAPTER CACHE HIT",json.dumps(old["splits"],indent=2));return
    if d.exists() and any(d.iterdir()):raise ValueError("Output directory not empty; inspect before replacing")
    parts=[];offset=0;week_counts={}
    for i,path in enumerate(paths,20):
        f=pd.read_parquet(path,columns=["user_id","item_id","timespent"])
        n=len(f);week_counts[str(i)]=n
        if not n or f.isna().any().any():raise ValueError("Empty file or missing required values: "+str(path))
        if (f.user_id<0).any() or (f.item_id<0).any() or not f.timespent.between(0,255).all():raise ValueError("Invalid input values")
        f=f.rename(columns={"item_id":"video_id"})
        f["user_id"]=f.user_id.astype("int64");f["video_id"]=f.video_id.astype("int64")
        f["long_view"]=(f.pop("timespent")>=10).astype(np.int8)
        f["time_ms"]=np.arange(offset,offset+n,dtype=np.int64)
        f["evaluation_id"]=f["time_ms"]
        if i==25:cut1=offset;cut2=offset+n//2
        offset+=n;parts.append(f);print("READ",i,n,flush=True)
    f=pd.concat(parts,ignore_index=True);del parts
    # Reject duplicates instead of silently guessing the correct order.
    if f.duplicated(["user_id","video_id"]).any():raise ValueError("Unexpected repeated exposures; inspect dataset contract")
    t=f.time_ms.to_numpy();train=t<cut1;val=(t>=cut1)&(t<cut2);test=t>=cut2
    vocab=np.sort(f.loc[train,"video_id"].unique());warm=f.video_id.isin(vocab).to_numpy()
    rng=np.random.default_rng(42);selected=[];stats={}
    for name,mask,count in zip(("train","validation","test"),(train,val,test),config["counts"]):
        eligible=np.flatnonzero(mask&warm)
        if len(eligible)<count:raise ValueError("Insufficient eligible targets: "+name)
        ix=np.sort(rng.choice(eligible,count,replace=False));part=f.iloc[ix].copy()
        if part.long_view.nunique()!=2:raise ValueError("Need both labels: "+name)
        part["split"]=name;selected.append(part)
        stats[name]=dict(raw_events=int(mask.sum()),eligible_events=len(eligible),excluded_cold_items=int((mask&~warm).sum()),sampled=len(part),positive_rate=float(part.long_view.mean()),min_ordinal=int(part.time_ms.min()),max_ordinal=int(part.time_ms.max()))
    targets=pd.concat(selected,ignore_index=True)
    stream=f.loc[warm,["user_id","video_id","time_ms"]].sort_values(["user_id","time_ms"],kind="stable")
    wanted={int(u):g for u,g in targets.groupby("user_id",sort=False)};histories={}
    for user,g in stream.groupby("user_id",sort=False):
        if int(user) not in wanted:continue
        q=wanted[int(user)];times=g.time_ms.to_numpy();items=g.video_id.to_numpy()
        end=np.searchsorted(times,q.time_ms.to_numpy(),side="left")
        for eid,e,qt in zip(q.evaluation_id,end,q.time_ms):
            start=max(0,int(e)-2048)
            if e and times[e-1]>=qt:raise RuntimeError("History order leakage")
            histories[int(eid)]=items[start:e].tolist()
    targets["history"]=[histories[int(i)] for i in targets.evaluation_id]
    targets["history_len"]=targets.history.map(len);targets["history_available"]=targets.history_len>0
    for name,g in targets.groupby("split"):
        stats[name].update(mean_history=float(g.history_len.mean()),empty_history=int((g.history_len==0).sum()),max_history=int(g.history_len.max()))
    if not stats["train"]["max_ordinal"]<stats["validation"]["min_ordinal"]<=stats["validation"]["max_ordinal"]<stats["test"]["min_ordinal"]:raise RuntimeError("Split violation")
    d.mkdir(parents=True,exist_ok=True);(d/"phase7").mkdir()
    fields=["evaluation_id","user_id","video_id","time_ms","long_view"]
    targets[fields].to_parquet(d/"evaluation_events.parquet",index=False)
    targets[fields+["history","history_len","history_available"]].to_parquet(d/"phase7/history_KFull.parquet",index=False)
    pd.DataFrame(dict(raw_video_id=vocab,video_idx=np.arange(len(vocab)))).to_parquet(d/"video_vocab.parquet",index=False)
    for name,g in targets.groupby("split"):
        g[["evaluation_id"]].rename(columns={"evaluation_id":"eval_idx"}).to_parquet(d/f"{name}.parquet",index=False)
    result=dict(config=config,raw_rows=len(f),clean_rows=len(f),vocab_size=len(vocab),week_counts=week_counts,cutoffs_ordinal=[cut1,cut2],splits=stats,
        artifacts={str(p.relative_to(d)):sha(p) for p in d.rglob("*.parquet")},
        limitations=["Popular-user/item official subset; six weeks only; not representative of all VK.","Warm-item targets only; cold-item exclusion reported.","Official validation week divided into two ordered halves, not the official test benchmark.","Row order is documented chronological; exact timestamps and ties unavailable. time_ms is a compatibility ordinal, not a real timestamp.","Prior exposures enter history by row order; no labels in history.","Absolute ten-second watch threshold is a distinct engagement proxy, not KuaiRand or KuaiRec label."])
    tmp=manifest.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2));tmp.replace(manifest)
    print("ADAPTER_PASS",json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
