"""Verify packaged service through host HTTP. No training or dataset access."""
import argparse,json,os,http.client
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--runtime",type=Path,required=True);p.add_argument("--port",type=int,default=8778);a=p.parse_args()
    smoke=json.loads((a.runtime/"smoke.json").read_text());token=os.environ["M23_API_TOKEN"]
    c=http.client.HTTPConnection("127.0.0.1",a.port,timeout=60)
    try:
        c.request("POST","/recommend",json.dumps(smoke["request"]),{"Authorization":"Bearer "+token,"Content-Type":"application/json"})
        r=c.getresponse();v=json.loads(r.read())
        if r.status!=200 or v["budget"]!=smoke["expected_k"] or abs(v["score"]-smoke["expected_score"])>2e-5 or v["storage"]["sqlite_lookups"]!=1 or v["storage"]["summary_rows"]!=0:raise RuntimeError(f"Smoke failed: {r.status} {v}")
        print("HTTP_SMOKE_PASS",json.dumps(v))
    finally:c.close()
if __name__=="__main__":main()
