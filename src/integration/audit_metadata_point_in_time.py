"""Strict metadata point-in-time audit, read-only, no held-out performance inspection.

Checks all candidate and history occurrences, not only distinct IDs. Upload-date
timezone/precision uncertainty is explicit. No snapshot feature is silently approved.
Run: PYTHONPATH=. python -m src.integration.audit_metadata_point_in_time
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RAW=ROOT/"data/raw/KuaiRand-1K/data/video_features_basic_1k.csv"
DEFAULT_HISTORY=ROOT/"data/processed/phase7/history_KFull.parquet"
DEFAULT_OUTPUT=ROOT/"reports/experiments/metadata_point_in_time_audit.json"
FIELDS=("author_id","video_type","upload_type","visible_status","video_duration",
        "server_width","server_height","music_id","music_type","tag")
CONDITIONAL=("author_id","video_type","upload_type","video_duration",
             "server_width","server_height","music_id","music_type")
BLOCKED=("visible_status","tag")
SENTINELS={"","null","none","nan","na","n/a","unknown","-1"}
def missing(x):
 if pd.isna(x):return True
 return str(x).strip().lower() in SENTINELS

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--metadata",type=Path,default=DEFAULT_RAW)
 p.add_argument("--history",type=Path,default=DEFAULT_HISTORY)
 p.add_argument("--chunk-size",type=int,default=100000)
 p.add_argument("--timezone",default="Asia/Shanghai",
  help="Hypothesis for upload_dt local calendar date; not proof of source timezone")
 p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
 a=p.parse_args()
 if a.chunk_size<1:p.error("chunk-size must be positive")
 for f in (a.metadata,a.history):
  if not f.is_file():p.error(f"missing file: {f}")
 cols=pd.read_csv(a.metadata,nrows=0).columns.tolist()
 required=["video_id","upload_dt",*FIELDS]
 absent=sorted(set(required)-set(cols))
 if absent:p.error(f"metadata columns missing: {absent}")
 h=pd.read_parquet(a.history,columns=["evaluation_id","video_id","time_ms","history"])
 if h.evaluation_id.isna().any() or not h.evaluation_id.is_unique:
  p.error("history evaluation_id null/duplicate")
 if h.time_ms.isna().any():p.error("history request time missing")
 # Per-video earliest request is sufficient for detecting upload-after-first-use;
 # per-occurrence counts are computed separately.
 occurrences=Counter()
 earliest={}
 for row in h.itertuples(index=False):
  t=int(row.time_ms)
  ids=[int(row.video_id)]
  if row.history is not None:ids.extend(int(x) for x in row.history)
  for vid in ids:
   occurrences[vid]+=1
   if vid not in earliest or t<earliest[vid]:earliest[vid]=t
 requested=set(occurrences)
 metadata={};dup=Counter();rows=0
 for ch in pd.read_csv(a.metadata,usecols=required,chunksize=a.chunk_size,low_memory=False):
  rows+=len(ch)
  sub=ch.loc[ch.video_id.isin(requested)]
  for rec in sub.to_dict("records"):
   vid=int(rec["video_id"])
   if vid in metadata:dup[vid]+=1
   else:metadata[vid]=rec
 missing_ids=requested-set(metadata)
 # Date-only values are treated as an interval in an assumed timezone.
 # Exact timestamps are also supported, but must have a documented source timezone.
 date_only=0;precise=0;invalid=0;definitely_future=0;ambiguous=0;definitely_before=0
 sample_bad=[];field_missing=Counter();field_nonmissing=Counter()
 availability={}
 for vid,rec in metadata.items():
  for col in FIELDS:
   (field_missing if missing(rec[col]) else field_nonmissing)[col]+=1
  raw=rec["upload_dt"]
  try:
   if missing(raw):raise ValueError("missing upload_dt")
   s=str(raw).strip()
   # KuaiRand upload_dt can be YYYYMMDD integer or YYYY-MM-DD.
   date_format=("%Y%m%d" if len(s)==8 and s.isdigit() else None)
   stamp=pd.to_datetime(s,format=date_format,errors="raise")
   if pd.isna(stamp):raise ValueError("NaT")
   if stamp.tzinfo is None:
    if (len(s)==8 and s.isdigit()) or (len(s)==10 and s[4:5]=="-" and s[7:8]=="-"):
     start=stamp.tz_localize(a.timezone).tz_convert("UTC")
     end=start+pd.Timedelta(days=1)
     date_only+=1
    else:
     start=stamp.tz_localize(a.timezone).tz_convert("UTC")
     end=start+pd.Timedelta(nanoseconds=1)
     precise+=1
   else:
    start=stamp.tz_convert("UTC");end=start+pd.Timedelta(nanoseconds=1);precise+=1
   request=pd.Timestamp(earliest[vid],unit="ms",tz="UTC")
   if start>request:
    availability[vid]="FUTURE";definitely_future+=1
   elif end>request:
    availability[vid]="AMBIGUOUS";ambiguous+=1
   else:
    availability[vid]="BEFORE";definitely_before+=1
  except Exception as exc:
   availability[vid]="INVALID";invalid+=1
   if len(sample_bad)<12:sample_bad.append({"video_id":vid,"upload_dt":str(raw),"error":str(exc)[:180]})
 occurrence_counts=Counter()
 for vid,n in occurrences.items():
  occurrence_counts[availability.get(vid,"MISSING_METADATA")]+=n
 # Critical: snapshot attributes remain unverified even when video was uploaded earlier.
 statuses={c:{"status":"BLOCKED_SNAPSHOT_UNVERIFIED" if c in BLOCKED else "CONDITIONAL_IMMUTABILITY_UNVERIFIED",
  "missing_or_sentinel_ids":field_missing[c],
  "nonmissing_ids":field_nonmissing[c]} for c in FIELDS}
 report={
  "status":"FAIL" if definitely_future or dup else "BLOCKED_UNVERIFIED",
  "training_gate":"BLOCK",
  "scope":{"requests":len(h),"unique_required_video_ids":len(requested),
    "total_candidate_and_history_occurrences":sum(occurrences.values()),
    "metadata_file_rows":rows,"metadata_matched_unique_ids":len(metadata),
    "metadata_missing_unique_ids":len(missing_ids),
    "metadata_duplicate_matched_ids":len(dup)},
  "upload_availability":{"assumed_timezone":a.timezone,
    "timezone_confirmed":False,
    "date_only_ids":date_only,"timestamp_precision_ids":precise,
    "before_first_request_ids":definitely_before,
    "same_date_or_precision_ambiguous_ids":ambiguous,
    "after_first_request_ids":definitely_future,
    "invalid_or_missing_upload_ids":invalid,
    "missing_metadata_ids":len(missing_ids),
    "occurrences_by_class":dict(occurrence_counts),
    "invalid_examples":sample_bad,
    "future_video_id_examples":[int(v) for v in availability if availability[v]=="FUTURE"][:20],
    "ambiguous_video_id_examples":[int(v) for v in availability if availability[v]=="AMBIGUOUS"][:20]},
  "field_policy":statuses,
  "denied_sources":["video_features_statistic_1k.csv","user_features_1k.csv"],
  "notes":[
   "Upload time checked against earliest request containing each video, including histories.",
   "An earlier upload is necessary, not sufficient, for snapshot attribute as-of correctness.",
   "Assumed timezone and upload_dt precision must be confirmed from source documentation.",
   "A date-only upload on the request day is ambiguous, not automatically safe.",
   "This audit does not alter frozen histories, metadata or test predictions.",
   "No training is approved until missing/ambiguous/future items and immutable feature provenance are resolved."]}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({"status":report["status"],"training_gate":report["training_gate"],
   "scope":report["scope"],"upload_availability":report["upload_availability"],
   "output":str(a.output)},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
