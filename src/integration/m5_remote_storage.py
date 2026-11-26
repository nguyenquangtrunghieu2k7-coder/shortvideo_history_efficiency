"""Authenticated loopback storage process for controlled network experiments.
Serves immutable signed snapshot lookup operations, not arbitrary SQL.
No cloud account or external network required."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
from src.integration.m3_store import Store

class StorageServer(ThreadingHTTPServer):
    request_queue_size=128
    daemon_threads=True
    block_on_close=False

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--root",type=Path,required=True)
    a.add_argument("--snapshot",required=True)
    a.add_argument("--ready-file",type=Path,required=True)
    a.add_argument("--port",type=int,default=0)
    args=a.parse_args()
    secret=bytes.fromhex(os.environ.get("M23_HMAC_KEY_HEX",""));token=os.environ.get("M23_API_TOKEN","")
    if len(secret)<32 or len(token)<32:raise ValueError("Missing credentials")
    store=Store(args.root,args.snapshot,secret)
    class Handler(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"
        def setup(self):
            super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        def log_message(self,*args):pass
        def reply(self,status,payload):
            b=json.dumps(payload,separators=(",",":"),allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get("Authorization",""),"Bearer "+token):
                self.close_connection=True;return self.reply(401,{"error":"unauthorized"})
            if self.path!="/lookup":
                self.close_connection=True;return self.reply(404,{"error":"not_found"})
            try:
                n=int(self.headers.get("Content-Length","0"))
                if not 0<n<=16384:raise ValueError("Bad length")
                req=json.loads(self.rfile.read(n))
                if not isinstance(req,dict) or set(req)!={"key","op","ranges"}:raise ValueError("Bad request")
                key,op,ranges=req["key"],req["op"],req["ranges"]
                if not isinstance(key,str) or len(key)>128 or op not in ("meta","summaries","tokens"):raise ValueError("Bad key or operation")
                if op=="tokens":
                    if not isinstance(ranges,list) or len(ranges)>64 or any(not isinstance(v,list) or len(v)!=2 or any(type(x) is not int for x in v) for v in ranges):raise ValueError("Bad ranges")
                elif ranges is not None:raise ValueError("Unexpected ranges")
                result,stats=store.lookup(key,op,mode="indexed_exact",ranges=ranges)
            except (ValueError,TypeError,KeyError):
                self.close_connection=True;return self.reply(400,{"error":"invalid_request"})
            except Exception:
                self.close_connection=True;return self.reply(503,{"error":"storage_unavailable"})
            return self.reply(200,{"result":result,"stats":stats,"snapshot":args.snapshot})
    with StorageServer(("127.0.0.1",args.port),Handler) as server:
        tmp=args.ready_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"port":server.server_port,"snapshot":args.snapshot}))
        tmp.replace(args.ready_file)
        server.serve_forever()

if __name__=="__main__":main()
