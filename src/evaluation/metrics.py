from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_auc(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores)

    if np.unique(y).size < 2:
        return float("nan")

    return float(roc_auc_score(y, s))


def classification_metrics(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores)

    return {
        "auc": binary_auc(y, s),
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
    }
