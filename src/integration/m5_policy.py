"""Versioned business control plane. An operator atomically updates a JSON policy
file; every request pins one validated policy revision. No caller override.
Single-host reference deployment; use an external transactional policy store
for multi-replica consensus. Never put credentials in the policy file."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from src.integration.m5_selector import preference

def validate(doc):
    if not isinstance(doc,dict) or set(doc)!={"revision","preferences"} or type(doc["revision"]) is not int or doc["revision"]<1 or not isinstance(doc["preferences"],dict):
        raise ValueError("Policy requires positive integer revision and preferences object")
    return {"revision":doc["revision"],"preferences":preference(**doc["preferences"])}

def atomic_write(path,doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    doc=validate(doc)
    fd,tmp=tempfile.mkstemp(prefix=".policy-",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(doc,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

class PolicyStore:
    def __init__(self,path):
        self.path=Path(path);self.lock=threading.Lock();self.active=None
        self.digest=None;self.error=None
        self.get()
    def get(self):
        with self.lock:
            try:
                raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                if digest!=self.digest:
                    doc=validate(json.loads(raw))
                    if self.active is not None and doc["revision"]<=self.active["revision"]:
                        raise ValueError("Policy revision must increase")
                    self.active=doc;self.digest=digest
                self.error=None
            except (OSError,ValueError,TypeError) as exc:
                self.error=str(exc)
                if self.active is None:raise ValueError("No valid startup policy") from exc
            return json.loads(json.dumps(self.active)),self.error
