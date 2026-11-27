"""WeChat D2Co 14-day observed exposure replay.
Same-day events are excluded from every target history. Within older days,
a deterministic label-independent hashed video order approximates unknown order.
The compatibility time_ms field encodes day index times 86400000, not a timestamp.
No aggregate watch statistics, gain, or other precomputed labels are used."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.integration.kuairec_adapter import sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",type=Path,default=Path("data/wechat/raw/wechat_video_debias.csv"))
    ap.add_argument("--output",type=Path,default=Path("data/wechat/processed"))
    ap.add_argument("--train",type=int,default=30118)
    ap.add_argument("--validation",type=int,default=6454)
    ap.add_argument("--test",type=int,default=6455)
    a=ap.parse_args()
    if min(a.train,a.validation,a.test)<1:ap.error("positive sample sizes required")
    d=a.output.resolve();src=a.csv.resolve()
    if d.name!="processed" or d.parent.name!="wechat":raise ValueError("Use isolated data/wechat/processed")
    config=dict(source_sha256=sha(src),seed=42,counts=[a.train,a.validation,a.test],label="playseconds_truncate / videoplayseconds >= 0.8",schema=1,split_days=[[1,10],[11,12],[13,14]],history_order="past days only; stable video hash within day")
    mf=d/"adapter_manifest.json"
    if mf.exists():
        old=json.loads(mf.read_text())
        if old["config"]!=config:raise ValueError("Adapter configuration mismatch")
        for name,digest in old["artifacts"].items():
            if sha(d/name)!=digest:raise ValueError("Adapter artifact changed: "+name)
        print("ADAPTER CACHE HIT",json.dumps(old["splits"],indent=2));return
    if d.exists() and any(d.iterdir()):raise ValueError("Output directory not empty; inspect before replacing")
    print("Reading WeChat raw CSV",flush=True)
    cols=["userid","feedid","date_","playseconds_truncate","videoplayseconds"]
    f=pd.read_csv(src,usecols=cols,dtype={c:"int32" for c in cols})
    raw_n=len(f)
    if not raw_n or set(f.date_.unique())!=set(range(1,15)):raise ValueError("Expected 14-day dataset with day indices 1..14")
    valid=(f.userid>=0)&(f.feedid>=0)&(f.videoplayseconds>0)&(f.playseconds_truncate>=0)&(f.playseconds_truncate<=f.videoplayseconds)
    invalid=int((~valid).sum())
    f=f.loc[valid].copy()
    # Integer comparison avoids boundary rounding; int64 avoids overflow.
    f["long_view"]=(5*f.playseconds_truncate.astype("int64")>=4*f.videoplayseconds.astype("int64")).astype(np.int8)
    durations=dict(min=int(f.videoplayseconds.min()),max=int(f.videoplayseconds.max()),mean=float(f.videoplayseconds.mean()),distinct=int(f.videoplayseconds.nunique()))
    f=f.drop(columns=["playseconds_truncate","videoplayseconds"]).rename(columns={"userid":"user_id","feedid":"video_id","date_":"day"})
    # Collapse same user/day/item duplicates only if binary labels agree.
    # Conflicting keys are removed rather than guessing which exposure came first.
    keys=["user_id","day","video_id"];dup=f.duplicated(keys,keep=False)
    conflicts=0;dedup=0
    if dup.any():
        g=f.loc[dup].copy()
        bad=g.groupby(keys,sort=False).long_view.transform("nunique")>1
        conflicts=int(bad.sum());clean=g.loc[~bad].drop_duplicates(keys)
        dedup=int((~bad).sum()-len(clean))
        f=pd.concat([f.loc[~dup],clean],ignore_index=True)
    else:f=f.reset_index(drop=True)
    f["evaluation_id"]=np.arange(len(f),dtype=np.int64)
    f["time_ms"]=f.day.astype("int64")*86400000
    days=f.day.to_numpy();train=days<=10;val=(days>=11)&(days<=12);test=days>=13
    vocab=np.sort(f.loc[train,"video_id"].unique());warm=f.video_id.isin(vocab).to_numpy()
    rng=np.random.default_rng(42);selected=[];stats={}
    for name,mask,count in zip(("train","validation","test"),(train,val,test),config["counts"]):
        eligible=np.flatnonzero(mask&warm)
        if len(eligible)<count:raise ValueError("Insufficient eligible targets: "+name)
        ix=np.sort(rng.choice(eligible,count,replace=False));part=f.iloc[ix].copy()
        if part.long_view.nunique()!=2:raise ValueError("Need both classes: "+name)
        part["split"]=name;selected.append(part)
        stats[name]=dict(raw_events=int(mask.sum()),eligible_events=len(eligible),excluded_cold_items=int((mask&~warm).sum()),sampled=len(part),positive_rate=float(part.long_view.mean()),min_day=int(part.day.min()),max_day=int(part.day.max()))
    targets=pd.concat(selected,ignore_index=True)
    # Only histories for sampled users are needed. Hash permutation within day
    # avoids ordering by label or by original duration-sorted source rows.
    stream=f.loc[warm&f.user_id.isin(targets.user_id),["user_id","video_id","day"]].copy()
    stream["tie"]=(stream.video_id.to_numpy(dtype=np.uint64)*np.uint64(11400714819323198485))
    stream=stream.sort_values(["user_id","day","tie","video_id"],kind="stable")
    wanted={int(u):g for u,g in targets.groupby("user_id",sort=False)}
    histories={}
    for user,g in stream.groupby("user_id",sort=False):
        q=wanted[int(user)];times=g.day.to_numpy();items=g.video_id.to_numpy()
        end=np.searchsorted(times,q.day.to_numpy(),side="left")
        for eid,e,qt in zip(q.evaluation_id,end,q.day):
            start=max(0,int(e)-2048)
            if e and times[e-1]>=qt:raise RuntimeError("Same-day or future history leakage")
            histories[int(eid)]=items[start:e].tolist()
    targets["history"]=[histories[int(i)] for i in targets.evaluation_id]
    targets["history_len"]=targets.history.map(len);targets["history_available"]=targets.history_len>0
    for name,g in targets.groupby("split"):
        stats[name].update(mean_history=float(g.history_len.mean()),median_history=float(g.history_len.median()),p95_history=float(g.history_len.quantile(.95)),empty_history=int((g.history_len==0).sum()),max_history=int(g.history_len.max()),unique_users=int(g.user_id.nunique()))
    if not stats["train"]["max_day"]<stats["validation"]["min_day"]<=stats["validation"]["max_day"]<stats["test"]["min_day"]:raise RuntimeError("Split overlap")
    d.mkdir(parents=True,exist_ok=True);(d/"phase7").mkdir()
    fields=["evaluation_id","user_id","video_id","time_ms","long_view"]
    targets[fields].to_parquet(d/"evaluation_events.parquet",index=False)
    targets[fields+["history","history_len","history_available"]].to_parquet(d/"phase7/history_KFull.parquet",index=False)
    pd.DataFrame(dict(raw_video_id=vocab,video_idx=np.arange(len(vocab)))).to_parquet(d/"video_vocab.parquet",index=False)
    for name,g in targets.groupby("split"):
        g[["evaluation_id"]].rename(columns={"evaluation_id":"eval_idx"}).to_parquet(d/f"{name}.parquet",index=False)
    result=dict(config=config,raw_rows=raw_n,clean_rows=len(f),invalid_rows_excluded=invalid,conflict_rows_excluded=conflicts,duplicate_rows_collapsed=dedup,vocab_size=len(vocab),duration_stats=durations,splits=stats,
        artifacts={str(p.relative_to(d)):sha(p) for p in d.rglob("*.parquet")},
        limitations=["D2Co WeChat preprocessed exposure dataset, not original full platform logs.","Day-level ordering only; same-day exposures excluded and prior within-day order approximated by deterministic video hash.","time_ms encodes relative day, not real timestamp.","Warm-item targets only; cold-item exclusion reported.","Prior observed exposures enter histories without labels.","Watch ratio >= 0.8 is a distinct engagement proxy; no aggregate watch statistics used.","Conflicting same-user/day/item binary labels removed; agreeing duplicate keys collapsed."])
    tmp=mf.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2));tmp.replace(mf)
    print("ADAPTER_PASS",json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
