"""M4 behavioral representation: train-only context co-occurrence, no test labels
or histories. Sparse PPMI factorization with frozen low-dimensional vectors."""
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

def behavioral_vectors(histories, vocab_size, dim=32, seed=42):
    # Count directed context pairs from training histories only.
    # The context vocabulary is hashed; item IDs are categorical, not distances.
    rows=[];cols=[]
    buckets=2048
    for history in histories:
        h=np.asarray(history,dtype=np.int64)
        if len(h)<2:continue
        for offset in (1,2,4):
            if len(h)<=offset:continue
            a,b=h[:-offset],h[offset:]
            rows.extend((a,b));cols.extend(((b*2654435761)%buckets,(a*2654435761)%buckets))
    if not rows:raise ValueError("No behavioral pairs")
    row=np.concatenate(rows);col=np.concatenate(cols)
    mat=sparse.coo_matrix((np.ones(len(row),dtype=np.float32),(row,col)),shape=(vocab_size+1,buckets)).tocsr()
    del rows,cols,row,col
    active=np.flatnonzero(np.asarray(mat.sum(1)).ravel())
    mat=mat[active].tocoo()
    rs=np.asarray(mat.tocsr().sum(1)).ravel()
    cs=np.asarray(mat.tocsc().sum(0)).ravel()
    mat.data=np.maximum(np.log(mat.data*float(mat.data.sum())/np.maximum(rs[mat.row]*cs[mat.col],1e-8)),0)
    mat=mat.tocsr();mat.eliminate_zeros()
    u,s,_=randomized_svd(mat,n_components=min(dim,min(mat.shape)-1),n_iter=4,random_state=seed)
    vec=u*np.sqrt(s)
    vec/=np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-8)
    out=np.zeros((vocab_size+1,dim),dtype=np.float32)
    out[active,:vec.shape[1]]=vec
    return out,{"active_items":int(len(active)),"context_buckets":buckets,"nonzero_ppmi":int(mat.nnz)}
