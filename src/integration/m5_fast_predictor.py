"""Exact single-request inference for frozen ExtraTrees / Ridge.
Avoid joblib scheduling and repeated estimator validation. No refitting."""
import numpy as np

class FastPredictor:
    def __init__(self,model):
        self.model=model
        if hasattr(model,"estimators_"):
            self.kind="native_tree_loop"
            self.trees=tuple(est.tree_ for est in model.estimators_)
        elif hasattr(model,"steps"):
            self.kind="fused_ridge"
            scaler,ridge=(step[1] for step in model.steps)
            self.w=ridge.coef_/scaler.scale_[None,:]
            self.b=ridge.intercept_-self.w@scaler.mean_
        else:raise TypeError("Unsupported selector")
    def predict(self,X):
        if self.kind=="fused_ridge":
            return np.asarray(X,dtype=np.float64)@self.w.T+self.b
        X=np.ascontiguousarray(X,dtype=np.float32)
        if X.ndim!=2 or X.shape[1]!=self.model.n_features_in_:raise ValueError("Bad selector features")
        # Same C-level traversal and leaf values as sklearn. Deterministic
        # accumulation order; parallel sklearn may differ at roundoff level.
        out=np.zeros((len(X),self.model.n_outputs_),dtype=np.float64)
        for tree in self.trees:
            out+=tree.predict(X).reshape(out.shape)
        return out/len(self.trees)
