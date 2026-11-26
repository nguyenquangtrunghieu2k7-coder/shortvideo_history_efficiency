"""Build conservative request-time metadata masks without modifying frozen artifacts.

Output: one row per evaluation request with candidate and history metadata masks.
Only upload availability is machine-checkable; snapshot immutability requires an
explicit assumption. Defaults to zero usable metadata features (fail closed).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
FEATURES=("author_id","video_type","upload_type","video_duration",
          "server_width","server_height","music_id","music_type")
DENIED=("tag","visible_status")
BAD={"","null","none","nan","na","n/a","unknown","-1"}

def usable(x):
 if x is None:return False
 try:
  if bool(pd.isna(x)):return False
 except (ValueError,TypeError):return False
 return str(x).strip().lower() not in BAD

def upload_bounds(s,timezone):
 if not usable(s):return None
 value=str(s).strip()
 fmt="%Y%m%d" if len(value)==8 and value.isdigit() else None
 ts=pd.to_datetime(value,format=fmt,errors="coerce")
 if pd.isna(ts):return None
 if ts.tzinfo is not None:
  return int(ts.tz_convert("UTC").value//10**6)
 # Treat calendar-day precision as an interval; do not assume midnight upload.
 if fmt or (len(value)==10 and value[4]=="-" and value[7]=="-"):
  start=ts.tz_localize(timezone).tz_convert("UTC")
  return int((start+pd.Timedelta(days=1)).value//10**6)
 # For precise local timestamps, still require a declared timezone assumption.
 return int(ts.tz_localize(timezone).tz_convert("UTC").value//10**6)+1

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--history",type=Path,default=ROOT/"data/processed/phase7/history_KFull.parquet")
 p.add_argument("--metadata",type=Path,default=ROOT/"data/raw/KuaiRand-1K/data/video_features_basic_1k.csv")
 p.add_argument("--output",type=Path,default=ROOT/"data/processed/experimental_metadata/point_in_time_masks.parquet")
 p.add_argument("--report",type=Path,default=ROOT/"reports/experiments/metadata_mask_build.json")
 p.add_argument("--timezone",default="Asia/Shanghai")
 p.add_argument("--chunk-size",type=int,default=100000)
 p.add_argument("--assume-immutable-upload-features",action="store_true",
   help="Explicit experimental assumption, NOT proof of metadata snapshot immutability")
 a=p.parse_args()
 if a.chunk_size<1:p.error("chunk-size must be positive")
 h=pd.read_parquet(a.history,columns=["evaluation_id","video_id","time_ms","history"])
 if not h.evaluation_id.is_unique:p.error("evaluation_id must be unique")
 needed=set(h.video_id.astype("int64"))
 for seq in h.history:
  if seq is not None:needed.update(int(x) for x in seq)
 cols=["video_id","upload_dt",*FEATURES,*DENIED]
 meta={}
 dup=0
 for chunk in pd.read_csv(a.metadata,usecols=cols,chunksize=a.chunk_size,low_memory=False):
  for rec in chunk.loc[chunk.video_id.isin(needed)].to_dict("records"):
   vid=int(rec["video_id"])
   if vid in meta:dup+=1
   else:meta[vid]=rec
 if dup:raise RuntimeError(f"Duplicate metadata video IDs: {dup}")
 ends={v:upload_bounds(rec["upload_dt"],a.timezone) for v,rec in meta.items()}
 # Do not serialize full snapshot feature values into the mask artifact.
 # A separate join at model input must use these per-occurrence masks.
 rows=[];stats={"candidate_available":0,"candidate_masked":0,
   "history_available":0,"history_masked":0,"missing_metadata_ids":len(needed-set(meta))}
 for row in h.itertuples(index=False):
  t=int(row.time_ms)
  def flags(vid):
   end=ends.get(int(vid))
   available=bool(end is not None and end<=t)
   active=bool(available and a.assume_immutable_upload_features)
   rec=meta.get(int(vid))
   mask=[bool(active and rec is not None and usable(rec[f])) for f in FEATURES]
   return available,mask
  cand_avail,cand_mask=flags(row.video_id)
  seq=[] if row.history is None else [int(x) for x in row.history]
  history_avail=[];history_masks=[]
  for vid in seq:
   available,mask=flags(vid)
   history_avail.append(available);history_masks.append(mask)
  stats["candidate_available"]+=int(cand_avail)
  stats["candidate_masked"]+=int(not any(cand_mask))
  stats["history_available"]+=sum(history_avail)
  stats["history_masked"]+=sum(not any(m) for m in history_masks)
  rows.append({"evaluation_id":int(row.evaluation_id),
   "candidate_metadata_mask":cand_mask,
   "history_metadata_mask":history_masks,
   "candidate_upload_before_request":cand_avail,
   "history_upload_before_request":history_avail})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_parquet(a.output,index=False)
 report={"status":"BUILT_CONDITIONAL" if a.assume_immutable_upload_features else "BUILT_FAIL_CLOSED",
  "requests":len(rows),"feature_order":list(FEATURES),
  "denied_features":list(DENIED)+["video_features_statistic_1k.csv","user_features_1k.csv"],
  "assume_immutable_upload_features":a.assume_immutable_upload_features,
  "timezone_assumption":a.timezone,"timezone_verified":False,
  "statistics":stats,"output":str(a.output),
  "notes":["Mask is request-specific; no candidate labels or held-out metrics read.",
   "Upload timestamp alone does not prove immutable feature values.",
   "False means zero feature contribution, not removal of the history event.",
   "This output does not change existing training entrypoints; models must explicitly consume masks.",
   "Training gate remains BLOCK pending source timezone and immutable-feature provenance."]}
 a.report.parent.mkdir(parents=True,exist_ok=True)
 a.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
