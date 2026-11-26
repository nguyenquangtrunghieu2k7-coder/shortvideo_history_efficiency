"""Read-only KuaiRand-1K / frozen Phase7 data inventory and leakage-risk audit.

Run: PYTHONPATH=. python -m src.integration.audit_kuairand_data
No model imports, training, checkpoint writes, or test metrics.
Requires pandas + pyarrow. Writes JSON report only.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
PATTERNS={
 "logs":("log_standard","log_random"),
 "video_basic":("video_features_basic",),
 "video_statistics":("video_features_statistic",),
 "user_features":("user_features",),
 "content_optional":("caption","category","categories","embedding","feature_visual","video_features_extra"),
}
TIME_HINTS=("time_ms","timestamp","upload_time","date","time")
LEAK_HINTS=("statistic","play_cnt","like_cnt","comment_cnt","share_cnt","view_cnt","watch_cnt","popularity","total_play","total_like")

def as_json(v):
 if isinstance(v,dict):return {str(k):as_json(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [as_json(x) for x in v]
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 if isinstance(v,float) and not np.isfinite(v):return None
 return v

def file_meta(path):
 return {"path":str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
         "bytes":path.stat().st_size}

def discover(roots,max_files):
 found=[];skipped=[]
 for root in roots:
  if not root.exists():skipped.append(str(root));continue
  if root.is_file():found.append(root);continue
  # Exclude git and model artifacts; cap traversal result to avoid huge walks.
  for f in root.rglob("*"):
   if len(found)>=max_files:break
   if f.is_file() and f.suffix.lower() in (".csv",".parquet",".json",".jsonl",".npy",".npz"):
    if not any(part in (".git","models","node_modules") for part in f.parts):found.append(f)
  if len(found)>=max_files:break
 return sorted(set(found)),skipped

def describe_csv(path,chunk_size,scan_rows):
 # Schema always obtained from file header; sample is a bounded prefix, not full coverage.
 try:
  head=pd.read_csv(path,nrows=0)
  cols=list(map(str,head.columns))
  total=0;null=Counter();nunique={};examples={}
  sample=pd.DataFrame()
  if scan_rows:
   chunks=[]
   for ch in pd.read_csv(path,chunksize=chunk_size,low_memory=False):
    remaining=scan_rows-total
    if remaining<=0:break
    ch=ch.iloc[:remaining];total+=len(ch);chunks.append(ch)
    if total>=scan_rows:break
   if chunks:sample=pd.concat(chunks,ignore_index=True)
  if len(sample):
   null={str(k):float(v) for k,v in sample.isna().mean().items()}
   nunique={str(k):int(v) for k,v in sample.nunique(dropna=True).items()}
   examples={str(k):[str(x)[:90] for x in sample[k].dropna().head(2)] for k in cols[:40]}
  return {"columns":cols,"sample_rows":int(total),"sample_only":True,
          "sample_null_fraction":null,"sample_distinct":nunique,
          "sample_examples":examples}
 except Exception as ex:return {"error":f"{type(ex).__name__}: {ex}"}

def summarize_lengths(lengths):
 a=np.asarray(lengths,dtype=np.int64)
 if not len(a):return {"n":0}
 return {"n":int(len(a)),"min":int(a.min()),"mean":float(a.mean()),
         "max":int(a.max()),"fraction_zero":float(np.mean(a==0)),
         "fraction_at_cap_2048":float(np.mean(a==2048)),
         "quantiles":{str(q):float(np.quantile(a,q)) for q in (0,.1,.25,.5,.75,.9,.95,.99,1)}}

def audit_frozen(root,include_test=False):
 proc=root/"data/processed";out={"paths":{},"warnings":[]}
 vp=proc/"video_vocab.parquet"
 vocab_ids=None
 if vp.exists():
  df=pd.read_parquet(vp,columns=["raw_video_id","video_idx"])
  raw=pd.to_numeric(df.raw_video_id,errors="coerce")
  idx=pd.to_numeric(df.video_idx,errors="coerce")
  valid=bool(raw.notna().all() and idx.notna().all())
  out["vocabulary"]={"n":len(df),"unique_raw":bool(df.raw_video_id.is_unique),
      "unique_idx":bool(df.video_idx.is_unique),
      "idx_min":None if not len(df) else int(idx.min()),
      "idx_max":None if not len(df) else int(idx.max()),
      "contiguous_zero_based":bool(valid and df.video_idx.is_unique and idx.min()==0 and idx.max()==len(df)-1),
      "model_id_shift":1,"padding_id":0}
  vocab_ids=set(raw.dropna().astype("int64").tolist())
 else:out["warnings"].append("Frozen video_vocab.parquet not found")
 hp=proc/"phase7/history_KFull.parquet"
 ev=proc/"evaluation_events.parquet"
 if hp.exists():
  h=pd.read_parquet(hp,columns=["evaluation_id","user_id","video_id","time_ms","long_view","history","history_len","history_available"])
  seq=[list(x) if x is not None and not isinstance(x,float) else [] for x in h.history]
  lens=np.asarray([len(x) for x in seq])
  declared=pd.to_numeric(h.history_len,errors="coerce")
  out["history"]={"rows":len(h),"unique_evaluation_ids":bool(h.evaluation_id.is_unique),
      "lengths":summarize_lengths(lens),
      "history_len_mismatch":int(np.sum(declared.to_numpy()!=lens)),
      "over_2048":int(np.sum(lens>2048)),
      "duplicate_item_within_history_mean_fraction":float(np.mean([1-len(set(x))/len(x) for x in seq if x])) if any(lens) else 0.,
      "same_candidate_in_history_requests":int(sum(int(v) in set(s) for v,s in zip(h.video_id,seq))),
      "history_available_values":{str(k):int(v) for k,v in h.history_available.value_counts(dropna=False).items()}}
  if vocab_ids is not None:
   hist_unique=set(int(v) for x in seq for v in x)
   cand_unique=set(h.video_id.astype("int64"))
   out["history"]["history_oov_unique"]=len(hist_unique-vocab_ids)
   out["history"]["candidate_oov_unique"]=len(cand_unique-vocab_ids)
  if ev.exists():
   e=pd.read_parquet(ev,columns=["evaluation_id","user_id","video_id","time_ms","long_view"])
   out["events"]={"rows":len(e),"unique_evaluation_ids":bool(e.evaluation_id.is_unique),
      "n_users":int(e.user_id.nunique()),"n_candidates":int(e.video_id.nunique()),
      "label_counts":{str(k):int(v) for k,v in e.long_view.value_counts(dropna=False).items()},
      "time_ms_min":int(e.time_ms.min()),"time_ms_max":int(e.time_ms.max())}
   if e.evaluation_id.is_unique and h.evaluation_id.is_unique:
    joined=h[["evaluation_id","user_id","video_id","time_ms","long_view"]].merge(e,on="evaluation_id",suffixes=("_history","_events"),validate="one_to_one",how="left",indicator=True)
    out["events"]["history_missing_event"]=int((joined["_merge"]!="both").sum())
    out["events"]["history_event_mismatch"]={c:int((joined[c+"_history"]!=joined[c+"_events"]).sum()) for c in ("user_id","video_id","time_ms","long_view")}
 else:out["warnings"].append("Frozen history_KFull.parquet not found")
 splits={}
 split_sets={}
 for split in ("train","validation","test"):
  f=proc/(split+".parquet")
  if not f.exists():continue
  if split=="test" and not include_test:
   splits[split]={"present":True,"inspected":False,"reason":"test reserved by default"}
   continue
  d=pd.read_parquet(f,columns=["eval_idx"])
  ids=set(d.eval_idx.astype("int64"))
  splits[split]={"n":len(d),"unique_ids":len(ids)==len(d)}
  split_sets[split]=ids
 out["splits"]=splits
 out["split_id_overlap"]={a+"__"+b:len(split_sets[a]&split_sets[b]) for a in split_sets for b in split_sets if a<b}
 return out

def metadata_coverage(paths,required_ids,chunk_size):
 # This checks raw-video-ID joins only; no assumption of pretrained content vectors.
 results=[]
 for path in paths:
  try:
   columns=list(pd.read_csv(path,nrows=0).columns)
   id_col=next((c for c in ("video_id","raw_video_id") if c in columns),None)
   if id_col is None:
    results.append({"path":str(path),"status":"NO_VIDEO_ID_COLUMN","columns":columns});continue
   found=set();unique=set();nrows=0;dupes=0
   for ch in pd.read_csv(path,usecols=[id_col],chunksize=chunk_size):
    s=pd.to_numeric(ch[id_col],errors="coerce").dropna().astype("int64")
    nrows+=len(ch);unique.update(s.tolist());found.update(set(s.tolist())&required_ids)
   results.append({"path":str(path),"status":"OK","id_column":id_col,
     "rows":nrows,"unique_video_ids":len(unique),
     "required_video_ids":len(required_ids),
     "covered_required_ids":len(found),
     "coverage":len(found)/len(required_ids) if required_ids else None})
  except Exception as ex:results.append({"path":str(path),"error":str(ex)})
 return results

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--raw-dir",action="append",type=Path,default=[],help="Repeat for each raw data folder")
 p.add_argument("--max-files",type=int,default=2000)
 p.add_argument("--sample-rows",type=int,default=2000)
 p.add_argument("--chunk-size",type=int,default=100000)
 p.add_argument("--include-test",action="store_true",help="Inspect test split IDs only; no test labels or metrics")
 p.add_argument("--coverage",action="store_true",help="Full-scan metadata IDs for exact coverage; may take time")
 p.add_argument("--output",type=Path,default=ROOT/"reports/experiments/kuairand_data_audit.json")
 a=p.parse_args()
 if a.max_files<1 or a.sample_rows<0 or a.chunk_size<1:raise ValueError("Invalid limits")
 roots=[ROOT/"data/raw",ROOT/"data",ROOT/"public_test/data",ROOT.parent/"KuaiRand",*a.raw_dir]
 files,missing=discover(roots,a.max_files)
 # Deduplicate file paths from overlapping roots.
 categories={}
 for f in files:
  n=f.name.lower()
  for group,patterns in PATTERNS.items():
   if any(x in n for x in patterns):categories.setdefault(group,[]).append(f)
 report={"status":"AUDIT_COMPLETED_WITH_LIMITATIONS","repo":str(ROOT),
  "roots_checked":list(map(str,roots)),"roots_missing":missing,
  "files_discovered":len(files),"max_files":a.max_files,
  "categories":{},"frozen":{},"metadata_coverage":[],
  "limitations":["CSV null/distinct statistics are prefix samples, not population estimates.",
    "File presence and metadata join coverage do not establish semantic validity or temporal availability.",
    "No raw event-time reconstruction is performed; use frozen strict reconstruction audit separately.",
    "No model weights, train data, or test predictions are changed."]}
 for group,paths in categories.items():
  report["categories"][group]=[]
  for f in sorted(set(paths)):
   entry=file_meta(f)
   if f.suffix.lower()==".csv":
    entry.update(describe_csv(f,a.chunk_size,a.sample_rows))
    entry["temporal_leakage_risk_columns"]=[c for c in entry.get("columns",[]) if any(h in c.lower() for h in LEAK_HINTS)]
   report["categories"][group].append(entry)
 try:report["frozen"]=audit_frozen(ROOT,a.include_test)
 except Exception as ex:report["frozen_error"]=f"{type(ex).__name__}: {ex}"
 if a.coverage:
  vp=ROOT/"data/processed/video_vocab.parquet"
  if vp.exists():
   required=set(pd.read_parquet(vp,columns=["raw_video_id"]).raw_video_id.astype("int64"))
   targets=sorted(set(categories.get("video_basic",[])+categories.get("video_statistics",[])+categories.get("content_optional",[])))
   report["metadata_coverage"]=metadata_coverage([f for f in targets if f.suffix.lower()==".csv"],required,a.chunk_size)
  else:report["limitations"].append("Coverage skipped: no frozen vocabulary")
 else:report["limitations"].append("Exact metadata coverage disabled; pass --coverage to enable full ID scans")
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(as_json(report),indent=2,ensure_ascii=False)+"\n")
 print(json.dumps({"status":report["status"],"output":str(a.output),
   "files_discovered":len(files),"categories":{k:len(v) for k,v in categories.items()},
   "frozen_error":report.get("frozen_error"),
   "history":report.get("frozen",{}).get("history",{}).get("lengths"),
   "coverage":report["metadata_coverage"]},indent=2,ensure_ascii=False))
if __name__=="__main__":main()
