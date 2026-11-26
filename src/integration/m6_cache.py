"""Resumable original-grid validation cache. Never reruns test scoring.
Each completed action is atomically persisted. Cache binds to checkpoint,
signed snapshot, input parquet hashes and action/feature schema."""
import json
from pathlib import Path
import numpy as np
from src.integration import m3_product as p,m5_selector as s
from src.integration.m23_persistent_store import _sha

ACTIONS=tuple(a for a in s.ACTIONS if a[1] in s.ORIGINAL_BUDGETS)
NAMES=tuple(f"{m}_{k}" for m,k in ACTIONS)
COLS=np.asarray([s.ACTIONS.index(a) for a in ACTIONS])
def save(path,**arrays):
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)

def load_or_build(directory,engine,ck,m5):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    prior=json.loads((m5/"summary.json").read_text())
    if prior["checkpoint_sha256"]!=_sha(ck) or prior["snapshot_id"]!=engine.bundle["snapshot_id"] or tuple(prior["actions"])!=s.ACTION_NAMES:raise ValueError("M5 cache provenance mismatch")
    _,vmap=p.baseline.load_video_vocab()
    val=p.baseline.KuaiRandDataset("validation","Full",vmap)
    ev=p.events_only()
    files=[p.baseline.VIDEO_VOCAB_FILE,p.baseline.PHASE7_DIR/"history_KFull.parquet"]+[p.baseline.PROCESSED_DIR/f for f in ("validation.parquet","test.parquet","evaluation_events.parquet")]
    contract=dict(schema=1,checkpoint=_sha(ck),snapshot=engine.bundle["snapshot_id"],actions=list(NAMES),
        input_hashes={str(f):_sha(f) for f in files},test_predictions_sha256=_sha(m5/"predictions.npz"))
    cf=d/"contract.json"
    if cf.exists() and json.loads(cf.read_text())!=contract:raise ValueError("Cache contract changed; use a new cache directory")
    if not cf.exists():
        tmp=cf.with_suffix(".tmp");tmp.write_text(json.dumps(contract,indent=2));tmp.replace(cf)
    def features(ids,cs,us):
        x=[];n=[]
        for key,c,u in zip(ids,cs,us):
            meta,_=engine.store.lookup(str(int(key)),"summaries",mode="indexed_exact")
            if meta is None:raise KeyError(key)
            n.append(meta["n"]);x.append(s.feature_vector(meta["n"],meta["blocks"],engine.emb[c],engine.ue[u%len(engine.ue)]))
        return np.asarray(x),np.asarray(n)
    vp=d/"validation_features.npz";vid=np.asarray(val.evaluation_ids)
    if not vp.exists():
        X,N=features(vid,val.candidate_ids,val.user_ids)
        if not np.array_equal(N,list(map(len,val.history_ids))):raise ValueError("Validation history mismatch")
        save(vp,ids=vid,X=X,N=N,labels=np.asarray(val.labels),users=np.asarray(val.user_ids))
    with np.load(vp,allow_pickle=False) as z:
        if not np.array_equal(z["ids"],vid):raise ValueError("Validation IDs changed")
        V={k:z[k].copy() for k in z.files}
    cols=[]
    for mode,k in ACTIONS:
        f=d/f"val_{mode}_{k}.npz"
        if f.exists():
            with np.load(f,allow_pickle=False) as z:
                if not np.array_equal(z["ids"],vid):raise ValueError("Cached action ID mismatch")
                pred=z["pred"].copy()
            print("CACHE HIT",mode,k,flush=True)
        else:
            if mode=="recent":h=[v[-k:] if k else [] for v in val.history_ids]
            else:
                h=[]
                for key,c,v in zip(vid,val.candidate_ids,val.history_ids):
                    meta,_=engine.store.lookup(str(int(key)),"summaries",mode="indexed_exact")
                    h.append([v[j] for a,b in p.semantic_ranges(meta["blocks"],engine.emb[c],k) for j in range(a,b)])
            pred=p.predict_batch(engine.model,val,h,engine.device)
            save(f,ids=vid,pred=pred)
            print("CACHE SAVED",mode,k,flush=True)
        if pred.shape!=(len(vid),) or not np.isfinite(pred).all():raise ValueError("Invalid cached action")
        cols.append(pred)
    V["P"]=np.column_stack(cols)
    with np.load(m5/"predictions.npz",allow_pickle=False) as z:
        ids=z["evaluation_ids"];P=z["action_predictions"][:,COLS].copy();y=z["labels"].copy();u=z["users"].copy()
    if len(set(ids))!=len(ids) or P.shape!=(len(ids),len(ACTIONS)) or not np.isfinite(P).all():raise ValueError("Invalid test cache")
    e=ev.set_index("evaluation_id").loc[ids]
    if len(e)!=len(ev) or not np.array_equal(e.long_view.to_numpy(),y) or not np.array_equal(e.user_id.to_numpy(),u):raise ValueError("Test cache label/user alignment mismatch")
    tp=d/"test_features.npz"
    if not tp.exists():
        X,N=features(ids,e.candidate.to_numpy(),u);save(tp,ids=ids,X=X,N=N)
    with np.load(tp,allow_pickle=False) as z:
        if not np.array_equal(z["ids"],ids):raise ValueError("Test feature ID mismatch")
        T=dict(ids=ids,X=z["X"].copy(),N=z["N"].copy(),labels=y,users=u,P=P,candidates=e.candidate.to_numpy())
    print("TEST CACHE HIT: all original actions; no test ranker inference",flush=True)
    return V,T
