import argparse,json
from pathlib import Path
from src.integration.global_policy import publish
def main():
    p=argparse.ArgumentParser();p.add_argument("--runtime",type=Path,required=True);p.add_argument("--from-json",type=Path,required=True);a=p.parse_args()
    publish(a.runtime/"policy"/"business.json",json.loads((a.runtime/"calibration.json").read_text()),json.loads(a.from_json.read_text()))
    print("POLICY_PUBLISHED")
if __name__=="__main__":main()
