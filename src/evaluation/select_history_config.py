from __future__ import annotations

HISTORY_CONFIGS = [0, 1, 5, 10, 20, 50, 100, "Full"]


def history_column(k):
    if k == 0:
        return None
    if k == "Full":
        return "history_full"
    return f"history_{k}"


def history_label(k):
    return "K=Full" if k == "Full" else f"K={k}"
