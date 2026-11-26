"""Operator-only policy publishing tool; run from repository root."""
import argparse
import json
from pathlib import Path
from src.integration.m5_policy import validate,atomic_write

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--file",type=Path,required=True)
    p.add_argument("--from-json",type=Path,required=True)
    a=p.parse_args()
    doc=validate(json.loads(a.from_json.read_text()))
    if a.file.exists():
        old=validate(json.loads(a.file.read_text()))
        if doc["revision"]<=old["revision"]:raise SystemExit("Revision must increase")
    atomic_write(a.file,doc)
    print("Published revision",doc["revision"])
if __name__=="__main__":main()
