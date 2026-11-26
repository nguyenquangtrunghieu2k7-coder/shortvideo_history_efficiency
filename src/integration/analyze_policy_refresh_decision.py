"""Decision-grade descriptive Pareto analysis for the 36-cell policy matrix.

Three objectives: maximize AUC, minimize mean selected history/request,
minimize observed refresh-job workload proxy. No arbitrary utility weights.
All conclusions exploratory; no claims of statistical dominance.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"reports/experiments"
OUT=BASE/"figures"

def dominates(a,b):
    return (a.auc>=b.auc and
            a.mean_selected_history_length<=b.mean_selected_history_length and
            a.refresh_workload_ratio<=b.refresh_workload_ratio and
            (a.auc>b.auc or a.mean_selected_history_length<b.mean_selected_history_length or
             a.refresh_workload_ratio<b.refresh_workload_ratio))

def main():
    matrix=pd.read_csv(BASE/"policy_refresh_matrix.csv")
    stage=json.loads((BASE/"scheduled_window_v4_e2e.json").read_text())
    if len(matrix)!=36 or matrix[["scenario","policy"]].duplicated().any():
        raise RuntimeError("Expected 36 unique scenario/policy configurations")
    ratios={x["scenario"]:x["batch_workload_ratio_vs_S0"] for x in stage["scenarios"]}
    matrix["refresh_workload_ratio"]=matrix.scenario.map(ratios)
    if matrix.refresh_workload_ratio.isna().any():raise RuntimeError("Missing refresh workload")
    # Equal outcomes are equivalent operating points, not improvements.
    records=list(matrix.itertuples(index=False))
    matrix["pareto_3objective"]=[not any(dominates(q,p) for q in records) for p in records]
    matrix["pareto_auc_history"]=[not any(
        q.auc>=p.auc and q.mean_selected_history_length<=p.mean_selected_history_length and
        (q.auc>p.auc or q.mean_selected_history_length<p.mean_selected_history_length)
        for q in records) for p in records]
    # Select minimum refresh workload, then minimum selected history,
    # then maximum AUC, subject to an EXPLICIT user-provided AUC floor.
    decisions=[]
    for floor in (0.58,0.59,0.60,0.61,0.62):
        feasible=matrix[matrix.auc>=floor]
        if feasible.empty:continue
        chosen=feasible.sort_values(["refresh_workload_ratio","mean_selected_history_length","auc"],
                                     ascending=[True,True,False]).iloc[0]
        decisions.append({"auc_floor":floor,"selection_rule":"minimize refresh workload, then history, subject to AUC floor",
                          "scenario":chosen.scenario,"policy":chosen.policy,
                          "auc":float(chosen.auc),"mean_history":float(chosen.mean_selected_history_length),
                          "refresh_workload_ratio":float(chosen.refresh_workload_ratio)})
    OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,6))
    for s,part in matrix.groupby("scenario",sort=True):
        ax.scatter(part.mean_selected_history_length,part.auc,label=s,s=48,alpha=.7)
        for p in part.itertuples():
            if p.policy in ("Adaptive-V4","Fixed-Full","Fixed-100","Fixed-0"):
                ax.annotate(f"{s}:{p.policy.replace('Fixed-','')}",
                            (p.mean_selected_history_length,p.auc),
                            xytext=(4,4),textcoords="offset points",fontsize=7)
    frontier=matrix[matrix.pareto_3objective]
    ax.scatter(frontier.mean_selected_history_length,frontier.auc,
               facecolors="none",edgecolors="black",s=170,linewidths=1.3,
               label="3-objective non-dominated")
    ax.set(xlabel="Mean selected history interactions / request (lower is better)",
           ylabel="ROC-AUC (higher is better)",
           title="36 policy x refresh configurations: quality vs history cost")
    ax.grid(alpha=.2);ax.legend()
    fig.text(.01,.01,"Circle outline = non-dominated considering refresh workload too. Exploratory, point estimates only.",fontsize=8)
    for ext in ("png","svg"):fig.savefig(OUT/f"policy_refresh_quality_history_pareto.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)
    matrix.to_csv(BASE/"policy_refresh_decision_matrix.csv",index=False)
    report={"status":"EXPLORATORY_NOT_PRODUCTION_DECISION",
            "objective_1":"maximize held-out AUC",
            "objective_2":"minimize selected history interactions per request",
            "objective_3":"minimize observed refresh batch workload proxy",
            "non_dominated_3objective":[f"{p.scenario}:{p.policy}" for p in matrix[matrix.pareto_3objective].itertuples()],
            "threshold_scenarios_illustrative_not_prespecified":decisions,
            "limitations":["AUC floors were selected for illustration after test inspection; not prospective decision thresholds.",
                "Refresh workload measured on all observed jobs, AUC/history on common OOV-free cohort.",
                "No production CPU, RAM, latency, energy or financial costs measured.",
                "Point-estimate Pareto is not statistical Pareto; confidence intervals needed.",
                "Zero-history Fixed-0 still incurs an unnecessary refresh workload in this schedule-level model; production would avoid refresh entirely."]}
    (BASE/"policy_refresh_decision_analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
