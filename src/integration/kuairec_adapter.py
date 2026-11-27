"""KuaiRec chronological observed-event replay adapter for the frozen M4 contract.
Big matrix only; timestamp ties excluded from histories; train-only vocabulary.
Targets sampled uniformly within chronological splits after warm-item filtering.
Prior observed exposures may enter histories during validation/test replay;
their labels are not used as model features. Not the original fully observed
small-matrix evaluation protocol."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("data/kuairec/processed"))
    ap.add_argument("--train",type=int,default=30118)
    ap.add_argument("--validation",type=int,default=6454)
    ap.add_argument("--test",type=int,default=6455)
    a=ap.parse_args()
    if min(a.train,a.validation,a.test)<1:ap.error("positive sample sizes required")
    d=a.output.resolve();src=a.csv.resolve()
    if d.name!="processed" or d.parent.name!="kuairec":raise ValueError("Use an isolated data/kuairec/processed directory")
    config=dict(source_sha256=sha(src),seed=42,counts=[a.train,a.validation,a.test],label="watch_ratio > 2.0",schema=1)
    manifest=d/"adapter_manifest.json"
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old["config"]!=config:raise ValueError("Existing adapter configuration differs")
        for name,digest in old["artifacts"].items():
            if sha(d/name)!=digest:raise ValueError("Adapter artifact changed: "+name)
        print("ADAPTER CACHE HIT",json.dumps(old["splits"],indent=2));return
    if d.exists() and any(d.iterdir()):raise ValueError("Output directory is not empty; inspect before replacing")
    print("Reading big matrix",flush=True)
    f=pd.read_csv(src,usecols=["user_id","video_id","timestamp","watch_ratio"],dtype={"user_id":"int32","video_id":"int32","timestamp":"float64","watch_ratio":"float64"})
    raw_n=len(f)
    valid=np.isfinite(f.timestamp)&np.isfinite(f.watch_ratio)&(f.watch_ratio>=0)&(f.timestamp>0)&(f.user_id>=0)&(f.video_id>=0)
    f=f.loc[valid].copy()
    f["time_ms"]=np.rint(f.timestamp.to_numpy()*1000).astype(np.int64)
    f["long_view"]=(f.watch_ratio>2.).astype(np.int8)
    f=f.drop(columns=["timestamp","watch_ratio"])
    # Exact duplicate observed keys with conflicting labels are excluded entirely.
    keys=["user_id","video_id","time_ms"]
    conflict=f.groupby(keys,sort=False).long_view.transform("nunique")>1
    conflicts=int(conflict.sum());f=f.loc[~conflict].drop_duplicates(keys).reset_index(drop=True)
    f["evaluation_id"]=np.arange(len(f),dtype=np.int64)
    t=f.time_ms.to_numpy()
    t1,t2=np.quantile(t,[.70,.85],method="inverted_cdf").astype(np.int64)
    if t1>=t2:raise ValueError("Degenerate timestamp split")
    train=t<t1;val=(t>=t1)&(t<t2);test=t>=t2
    vocab=np.sort(f.loc[train,"video_id"].unique())
    warm=f.video_id.isin(vocab).to_numpy()
    rng=np.random.default_rng(42);selected=[];stats={}
    for name,mask,count in zip(("train","validation","test"),(train,val,test),config["counts"]):
        eligible=np.flatnonzero(mask&warm)
        if len(eligible)<count:raise ValueError("Insufficient eligible targets: "+name)
        ix=np.sort(rng.choice(eligible,count,replace=False));part=f.iloc[ix].copy()
        if part.long_view.nunique()!=2:raise ValueError("Need both labels: "+name)
        part["split"]=name;selected.append(part)
        stats[name]=dict(raw_events=int(mask.sum()),eligible_events=len(eligible),excluded_cold_items=int((mask&~warm).sum()),sampled=len(part),positive_rate=float(part.long_view.mean()),min_time_ms=int(part.time_ms.min()),max_time_ms=int(part.time_ms.max()))
    targets=pd.concat(selected,ignore_index=True)
    # Histories include only known-item events strictly earlier in time.
    # Group by user once; vectorized lower-bound search excludes all ties.
    stream=f.loc[warm,["user_id","video_id","time_ms"]].sort_values(["user_id","time_ms","video_id"],kind="stable")
    wanted={int(u):g for u,g in targets.groupby("user_id",sort=False)}
    histories={}
    for user,g in stream.groupby("user_id",sort=False):
        if int(user) not in wanted:continue
        q=wanted[int(user)];times=g.time_ms.to_numpy();items=g.video_id.to_numpy()
        end=np.searchsorted(times,q.time_ms.to_numpy(),side="left")
        for eid,e,qt in zip(q.evaluation_id,end,q.time_ms):
            start=max(0,int(e)-2048)
            if e and times[e-1]>=qt:raise RuntimeError("History time leakage")
            histories[int(eid)]=items[start:e].tolist()
    targets["history"]=[histories[int(i)] for i in targets.evaluation_id]
    targets["history_len"]=targets.history.map(len)
    targets["history_available"]=targets.history_len>0
    for name,g in targets.groupby("split"):
        stats[name].update(mean_history=float(g.history_len.mean()),empty_history=int((g.history_len==0).sum()),max_history=int(g.history_len.max()))
    if not stats["train"]["max_time_ms"]<stats["validation"]["min_time_ms"]<=stats["validation"]["max_time_ms"]<stats["test"]["min_time_ms"]:raise RuntimeError("Chronological split violation")
    d.mkdir(parents=True,exist_ok=True);(d/"phase7").mkdir()
    fields=["evaluation_id","user_id","video_id","time_ms","long_view"]
    targets[fields].to_parquet(d/"evaluation_events.parquet",index=False)
    targets[fields+["history","history_len","history_available"]].to_parquet(d/"phase7/history_KFull.parquet",index=False)
    pd.DataFrame(dict(raw_video_id=vocab,video_idx=np.arange(len(vocab)))).to_parquet(d/"video_vocab.parquet",index=False)
    for name,g in targets.groupby("split"):
        g[["evaluation_id"]].rename(columns={"evaluation_id":"eval_idx"}).to_parquet(d/f"{name}.parquet",index=False)
    result=dict(config=config,raw_rows=raw_n,clean_rows=len(f),conflict_rows_excluded=conflicts,vocab_size=len(vocab),cutoffs_ms=[int(t1),int(t2)],splits=stats,
        artifacts={str(p.relative_to(d)):sha(p) for p in d.rglob("*.parquet")},
        limitations=["Warm-item targets only; cold-item exclusion reported.","Observed exposure replay, not randomized exposure or small-matrix protocol.","Prior validation/test exposures are available only after their timestamps; no labels in history.","Label is repeat-watch proxy, not KuaiRand long_view."])
    tmp=manifest.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2));tmp.replace(manifest)
    print("ADAPTER_PASS",json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
