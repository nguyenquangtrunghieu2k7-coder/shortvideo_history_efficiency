from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y_true, scores, n_bootstrap=2000, seed=42):
    y = np.asarray(y_true)
    s = np.asarray(scores)

    rng = np.random.default_rng(seed)

    base = roc_auc_score(y, s)
    values = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]

        if np.unique(yy).size < 2:
            continue

        values.append(roc_auc_score(yy, s[idx]))

    if not values:
        return {
            "auc": float(base),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    lo, hi = np.percentile(values, [2.5, 97.5])

    return {
        "auc": float(base),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
