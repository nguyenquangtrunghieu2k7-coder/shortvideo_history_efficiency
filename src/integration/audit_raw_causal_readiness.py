"""Read-only source-schema and causal-reconstruction readiness audit.

Does NOT claim to prove history causality. No labels/metrics from held-out split.
Usage: PYTHONPATH=. python -m src.integration.audit_raw_causal_readiness
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/"data/raw/KuaiRand-1K/data"
OUT=ROOT/"reports/experiments/raw_causal_readiness.json"
LOG_PREFIXES=("log_standard","log_random")
TIME_CANDIDATES=("time_ms","timestamp","event_time","time","date")
USER_CANDIDATES=("user_id","user")
VIDEO_CANDIDATES=("video_id","item_id")
EVENT_ID_CANDIDATES=("event_id","request_id","impression_id","log_id")
def find(cols,candidates):
 return [c for c in cols if c.lower() in candidates]
def inspect(path,sample):
 info={"path":str(path),"bytes":path.stat().st_size}
 try:
  header=pd.read_csv(path,nrows=0)
  cols=list(map(str,header.columns))
  info["columns"]=cols
  info["candidate_user_keys"]=find(cols,USER_CANDIDATES)
  info["candidate_video_keys"]=find(cols,VIDEO_CANDIDATES)
  info["candidate_time_keys"]=find(cols,TIME_CANDIDATES)
  info["candidate_event_keys"]=find(cols,EVENT_ID_CANDIDATES)
  frame=pd.read_csv(path,nrows=sample,low_memory=False)
  info["sample_rows"]=len(frame)
  info["sample_null_fraction"]={c:float(frame[c].isna().mean()) for c in cols}
  for c in info["candidate_time_keys"]:
   x=pd.to_numeric(frame[c],errors="coerce")
   info.setdefault("time_numeric_sample",{})[c]={
    "numeric_fraction":float(x.notna().mean()),
    "min":float(x.min()) if x.notna().any() else None,
    "max":float(x.max()) if x.notna().any() else None}
  info["source_kind"]="random" if path.name.startswith("log_random") else "standard"
  info["causal_reconstruction_ready"]=bool(info["candidate_user_keys"] and info["candidate_video_keys"] and info["candidate_time_keys"])
  info["event_identity_verified"]=False
  info["notes"]=["Column presence alone does not prove event identity, timestamp semantics, timezone or source completeness."]
 except Exception as exc:
  info["error"]=f"{type(exc).__name__}: {exc}"
 return info

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--raw-dir",type=Path,default=RAW)
 parser.add_argument("--sample-rows",type=int,default=2000)
 parser.add_argument("--output",type=Path,default=OUT)
 args=parser.parse_args()
 if args.sample_rows<0:parser.error("sample-rows must be >=0")
 if not args.raw_dir.exists():parser.error(f"raw directory not found: {args.raw_dir}")
 paths=sorted(p for p in args.raw_dir.rglob("*.csv") if p.name.startswith(LOG_PREFIXES))
 sources=[inspect(p,args.sample_rows) for p in paths]
 video=args.raw_dir/"video_features_basic_1k.csv"
 meta=inspect(video,args.sample_rows) if video.exists() else {"status":"MISSING"}
 processed=ROOT/"data/processed/evaluation_events.parquet"
 processed_schema=None
 if processed.exists():
  import pyarrow.parquet as pq
  processed_schema=pq.read_schema(processed).names
 missing=[s["path"] for s in sources if not s.get("causal_reconstruction_ready")]
 result={
  "status":"SOURCE_SCHEMA_READY_FOR_RECONSTRUCTION_DESIGN" if sources and not missing else "BLOCKED_SOURCE_SCHEMA",
  "source_logs":sources,
  "video_metadata":meta,
  "evaluation_events_columns":processed_schema,
  "missing_causal_keys":missing,
  "next_requirements":[
   "Establish which log source produced frozen history and its timestamp semantics.",
   "Prove source event identity and deterministic ordering for equal timestamps.",
   "Independently reconstruct past-only 24-hour histories and compare exact ordered IDs.",
   "Verify upload date parsing, timezone and feature snapshot as-of availability.",
   "Audit training-only fit of vocabulary/embeddings/normalizers and prevent candidate-label feature leakage.",
   "Do not use previously viewed test as a pristine final holdout."],
  "training_gate":"BLOCK",
  "reason":"Schema inventory is not a causal proof; no automatic PASS promotion."}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps({"status":result["status"],"source_logs":len(sources),
  "missing_causal_keys":missing,"output":str(args.output),
  "training_gate":result["training_gate"]},indent=2))
if __name__=="__main__":main()
