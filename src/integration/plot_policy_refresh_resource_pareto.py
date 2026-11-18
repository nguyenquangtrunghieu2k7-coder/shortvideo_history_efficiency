"""Plot measured 36-cell resource matrix with within-scenario Pareto curves.

Never treat diagnostic mixed-run wall times as production latency.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"reports/experiments"
OUT=BASE/"figures"
COSTS={
    "mean_selected_history_length":"Mean selected history interactions/request",
    "diagnostic_serving_sum_wall_s":"Diagnostic component-sum time (s; mixed runs)",
    "selection_process_cpu_s":"History selection process CPU (s)",
    "selection_python_alloc_peak_mib":"History selection Python allocation peak (MiB)",
    "refresh_workload_ratio_vs_S0":"Refresh interaction workload / S0",
}
def pareto(g,cost):
    # Equal AUC and equal cost are tied; both kept.
    return [not any((q.auc>=p.auc and getattr(q,cost)<=getattr(p,cost)
                      and (q.auc>p.auc or getattr(q,cost)<getattr(p,cost)))
                    for q in g.itertuples() if q.Index!=p.Index)
            for p in g.itertuples()]

def main():
    df=pd.read_csv(BASE/"policy_refresh_resource_matrix.csv")
    if len(df)!=36 or df[["scenario","policy"]].duplicated().any():raise RuntimeError("Expected 36 unique cells")
    OUT.mkdir(parents=True,exist_ok=True)
    allfront=[]
    for cost,xlabel in COSTS.items():
        fig,axes=plt.subplots(2,2,figsize=(13,10))
        for ax,(s,g) in zip(axes.flat,df.groupby("scenario",sort=True)):
            g=g.copy()
            g["on_frontier"]=pareto(g,cost)
            allfront.extend({"scenario":s,"policy":p.policy,"cost_metric":cost,
                             "pareto_within_scenario":bool(p.on_frontier)}
                            for p in g.itertuples())
            ax.scatter(g[cost],g.auc,s=50,alpha=.7)
            f=g[g.on_frontier].sort_values(cost)
            ax.plot(f[cost],f.auc,"o-",linewidth=1,label="Within-scenario frontier")
            for p in g.itertuples():
                ax.annotate(p.policy.replace("Fixed-","K="),
                            (getattr(p,cost),p.auc),xytext=(3,4),
                            textcoords="offset points",fontsize=7)
            ax.set_title(f"{s}: {len(g)} policies")
            ax.set_xlabel(xlabel);ax.set_ylabel("ROC-AUC")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle("Quality vs cost: policy-specific Pareto within each refresh scenario")
        fig.text(.01,.005,"Descriptive point-estimate frontiers. Full-history in-memory retrieval; not production lazy K-aware serving.",fontsize=8)
        fig.tight_layout(rect=(0,.02,1,.96))
        for ext in ("png","svg"):
            fig.savefig(OUT/f"pareto_by_scenario_{cost}.{ext}",dpi=220,bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(allfront).to_csv(BASE/"policy_refresh_resource_pareto_flags.csv",index=False)
    # 36-row comprehensive matrix is already stored as policy_refresh_resource_matrix.csv
    print("Saved 5 per-scenario Pareto figures (PNG/SVG) and resource Pareto flags.")
if __name__=="__main__":main()
