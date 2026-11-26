"""Fail-closed temporal-leakage audit for frozen Phase7 + KuaiRand metadata.

Read-only. No training/test-label inspection; outputs a development audit JSON.
PASS means only checked invariants; UNVERIFIED means evidence is insufficient.
Run: PYTHONPATH=. python -m src.integration.audit_temporal_leakage
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/"data/raw/KuaiRand-1K/data"
PROC=ROOT/"data/processed"
DAY=86_400_000
# Explicit contract: these are never safe as static features for event-time prediction.
DENIED_SOURCES={"video_features_statistic_1k.csv"}
DENIED_CANDIDATE_COLUMNS={"long_view","play_time_ms","is_like","is_follow",
 "is_comment","is_forward","is_hate","is_click","is_profile_enter",
 "profile_stay_time","comment_stay_time"}
BASIC_COLUMNS=("video_id","upload_dt","tag","author_id","music_id",
               "video_duration","video_type","upload_type")

def item(status,details=None):
 return {"status":status,"details":details or {}}

def iso(v):
 return pd.to_datetime(int(v),unit="ms",utc=True).isoformat()

def split_audit(events,include_test_ids=False):
 result={};groups={}
 for name in ("train","validation","test"):
  f=PROC/f"{name}.parquet"
  if not f.exists():
   result[name]=item("FAIL",{"reason":"split missing"});continue
  if name=="test" and not include_test_ids:
   result[name]=item("UNVERIFIED",{"reason":"test split intentionally unopened"});continue
  df=pd.read_parquet(f,columns=["eval_idx"])
  ids=pd.to_numeric(df.eval_idx,errors="coerce")
  if ids.isna().any() or ids.duplicated().any():
   result[name]=item("FAIL",{"reason":"null/duplicate eval_idx"});continue
  sub=events.loc[events.evaluation_id.isin(ids)]
  found=set(sub.evaluation_id.astype("int64"))
  missing=set(ids.astype("int64"))-found
  groups[name]=sub
  result[name]=item("FAIL" if missing else "PASS",{
   "rows":len(df),"matched":len(sub),"missing":len(missing),
   "unique_users":int(sub.user_id.nunique()),
   "min_time_ms":int(sub.time_ms.min()) if len(sub) else None,
   "max_time_ms":int(sub.time_ms.max()) if len(sub) else None,
   "min_time_utc":iso(sub.time_ms.min()) if len(sub) else None,
   "max_time_utc":iso(sub.time_ms.max()) if len(sub) else None})
 pairs={}
 for a in groups:
  for b in groups:
   if a>=b:continue
   x,y=groups[a],groups[b]
   ix=set(x.evaluation_id)&set(y.evaluation_id)
   users=set(x.user_id)&set(y.user_id)
   # User overlap is not inherently leakage for temporal generalization.
   pairs[a+"__"+b]={"evaluation_id_overlap":len(ix),
      "user_overlap":len(users),
      "strict_time_order":bool(x.time_ms.max()<y.time_ms.min()) if len(x) and len(y) else None}
 return result,pairs

def check_basic(events,raw_dir,chunk_size):
 f=raw_dir/"video_features_basic_1k.csv"
 if not f.exists():return item("FAIL",{"reason":"missing video basic file"})
 ids=set(events.video_id.astype("int64"))
 # Only candidate metadata; history metadata requires an additional full mapping.
 matched=set();duplicate_ids=0;seen=set();future=0;same_day=0;missing_dt=0;parse_error=0
 missing_tag=0;missing_author=0;missing_music=0;observed=0
 first_times=events.groupby("video_id",sort=False).time_ms.min().to_dict()
 for ch in pd.read_csv(f,usecols=list(BASIC_COLUMNS),chunksize=chunk_size,low_memory=False):
  ch=ch.loc[ch.video_id.isin(ids)]
  if ch.empty:continue
  observed+=len(ch)
  missing_tag+=int(ch.tag.isna().sum())
  missing_author+=int(ch.author_id.isna().sum())
  missing_music+=int(ch.music_id.isna().sum())
  for row in ch.itertuples(index=False):
   vid=int(row.video_id)
   if vid in seen:duplicate_ids+=1
   seen.add(vid);matched.add(vid)
   try:
    date=pd.to_datetime(row.upload_dt,utc=True,errors="raise")
    if pd.isna(date):missing_dt+=1;continue
    # upload_dt is date precision, not precise time; UTC date comparison is provisional.
    target=pd.to_datetime(int(first_times[vid]),unit="ms",utc=True)
    if date.date()>target.date():future+=1
    elif date.date()==target.date():same_day+=1
   except Exception:parse_error+=1
 return item("FAIL" if future or duplicate_ids or parse_error else "UNVERIFIED",{
   "candidate_unique_ids":len(ids),"covered_candidate_ids":len(matched),
   "missing_candidate_ids":len(ids-matched),"duplicate_metadata_ids":duplicate_ids,
   "upload_after_first_impression_utc_date":future,
   "same_utc_date_upload_impression_ambiguous":same_day,
   "missing_upload_dt":missing_dt,"parse_errors":parse_error,
   "missing_tag_candidate_rows":missing_tag,
   "missing_author_candidate_rows":missing_author,
   "missing_music_candidate_rows":missing_music,
   "date_precision_note":"Upload date alone cannot establish precise availability; UTC date may differ from source timezone.",
   "coverage_scope":"candidate IDs only; historical-item metadata validity remains unverified"})

def history_checks(events,chunk_size):
 path=PROC/"phase7/history_KFull.parquet"
 if not path.exists():return item("FAIL",{"reason":"missing history"})
 cols=["evaluation_id","user_id","video_id","time_ms","long_view","history","history_len","history_available"]
 n=0;bad_len=0;over=0;future_id_same=0;zero=0;bad_event=0
 # Cross-check against events, not proof that each historical ID was observed earlier.
 event_by_id=events.set_index("evaluation_id")
 pf=pd.read_parquet(path,columns=cols)
 for start in range(0,len(pf),chunk_size):
  h=pf.iloc[start:start+chunk_size]
  n+=len(h)
  for row in h.itertuples(index=False):
   seq=list(row.history) if row.history is not None else []
   bad_len+=int(len(seq)!=int(row.history_len))
   over+=int(len(seq)>2048)
   zero+=int(len(seq)==0)
   future_id_same+=int(int(row.video_id) in set(map(int,seq)))
   if row.evaluation_id not in event_by_id.index:bad_event+=1
   else:
    e=event_by_id.loc[row.evaluation_id]
    bad_event+=int(any(int(getattr(row,c))!=int(e[c]) for c in ("user_id","video_id","time_ms","long_view")))
 return item("FAIL" if any((bad_len,over,future_id_same,bad_event)) else "PASS",{
  "rows":n,"history_len_mismatch":bad_len,"over_2048":over,
  "candidate_id_in_history":future_id_same,"evaluation_event_mismatch":bad_event,
  "empty_history":zero,
  "critical_limitation":"History contains video IDs but no per-event timestamps or source event IDs; strict past-only 24h provenance cannot be proved from this file."})

def source_policy():
 return {
  "video_features_basic_1k.csv":{
   "status":"CONDITIONAL","allow":["video_id","tag","author_id","music_id",
       "video_type","video_duration","upload_type"],
   "conditions":["upload_dt must precede request with documented timezone and sufficient precision",
       "feature values must be immutable or historically available as-of request",
       "missing/sentinel values need explicit handling"]},
  "video_features_statistic_1k.csv":{"status":"DENY",
    "reason":"Full-period aggregate counters have no as-of timestamp; reconstruct past-only rolling aggregates from raw logs if needed."},
  "user_features_1k.csv":{"status":"DENY_UNTIL_PROVEN",
    "reason":"Snapshot collection time and historical availability not established; may encode future user behavior."},
  "log_standard_*.csv":{"status":"CONDITIONAL",
    "allow_history_fields":["user_id","video_id","time_ms","long_view",
      "is_like","is_follow","is_comment","is_forward","play_time_ms"],
    "conditions":["Use feedback ONLY from events strictly earlier than candidate time",
      "For same time_ms, exclude all tied events unless deterministic causal order is proven",
      "Preserve source identity and reconstruct history as-of request"]},
  "log_random_*.csv":{"status":"CONDITIONAL",
    "conditions":["Keep exposure protocol separate from standard logs",
       "Do not mix evaluation targets with historical labels without strict event-time joins"]},
  "candidate_impression_feedback":{"status":"DENY",
    "columns":sorted(DENIED_CANDIDATE_COLUMNS)},
  "vocabulary":{"status":"UNVERIFIED",
    "reason":"Frozen global ID vocabulary may reveal future item existence; fit training-only vocab or declare transductive protocol and freeze unknown-item behavior."},
  "learned_item_embeddings":{"status":"DENY_UNTIL_PROVEN",
    "reason":"Train exclusively on training-time eligible interactions/labels; avoid validation/test feedback."},
  "normalizers_and_encoders":{"status":"DENY_UNTIL_PROVEN",
    "reason":"Fit on training-only data, persist fitted state and reuse for validation/test."}
 }

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--raw-dir",type=Path,default=RAW)
 ap.add_argument("--chunk-size",type=int,default=100000)
 ap.add_argument("--include-test-ids",action="store_true",
   help="Inspect test split identifiers only, no test labels/metrics")
 ap.add_argument("--output",type=Path,default=ROOT/"reports/experiments/temporal_leakage_audit.json")
 a=ap.parse_args()
 if a.chunk_size<=0:ap.error("chunk-size must be positive")
 path=PROC/"evaluation_events.parquet"
 if not path.exists():raise FileNotFoundError(path)
 events=pd.read_parquet(path,columns=["evaluation_id","user_id","video_id","time_ms","long_view"])
 errors=[]
 if events.evaluation_id.isna().any() or events.evaluation_id.duplicated().any():errors.append("evaluation_id not unique/non-null")
 if events.time_ms.isna().any():errors.append("missing request timestamps")
 splits,pairs=split_audit(events,a.include_test_ids)
 hist=history_checks(events,a.chunk_size)
 meta=check_basic(events,a.raw_dir,a.chunk_size)
 checks={"events":item("FAIL" if errors else "PASS",{"rows":len(events),"errors":errors}),
   "history_structure":hist,"metadata_availability":meta,
   "split_integrity":item("FAIL" if any(v["status"]=="FAIL" for v in splits.values()) or any(v["evaluation_id_overlap"] for v in pairs.values()) else "PASS",
      {"splits":splits,"pairs":pairs}),
   "history_event_time_provenance":item("UNVERIFIED",{"reason":"Frozen history lacks event timestamps; independent raw-log reconstruction required"}),
   "feature_asof_provenance":item("UNVERIFIED",{"reason":"Metadata snapshot timestamps not established"}),
   "training_pipeline_provenance":item("UNVERIFIED",{"reason":"Model/scaler/embedding training code and checkpoints not fully audited"}),
   "test_holdout":item("UNVERIFIED",{"reason":"Existing test was previously viewed; not pristine for future model selection"})}
 failed=[k for k,v in checks.items() if v["status"]=="FAIL"]
 unverified=[k for k,v in checks.items() if v["status"]=="UNVERIFIED"]
 result={"status":"FAIL" if failed else "BLOCKED_UNVERIFIED" if unverified else "PASS",
  "checks":checks,"failed":failed,"unverified":unverified,
  "source_feature_policy":source_policy(),
  "training_gate":"BLOCK" if failed or unverified else "ALLOW",
  "notes":["No frozen files were changed.",
   "No test labels, test predictions or performance metrics were opened.",
   "Same candidate ID absent from history is not evidence of causal history.",
   "Static item metadata may change over time; 100% ID coverage does not prove as-of validity."]}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps({"status":result["status"],"training_gate":result["training_gate"],
   "failed":failed,"unverified":unverified,"output":str(a.output)},indent=2))
if __name__=="__main__":main()
