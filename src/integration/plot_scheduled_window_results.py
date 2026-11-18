"""Create reproducible diagnostic figures from local experiment reports.

Run: PYTHONPATH=. python -m src.integration.plot_scheduled_window_results
Input: reports/experiments/scheduled_window_v4_e2e.json
       reports/experiments/scheduled_window_contract_audit.json
       reports/experiments/scheduled_window_controlled.json
Output: reports/experiments/figures/*.png, *.svg and summary.json
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"reports/experiments"
FIG=BASE/"figures"

def read(name):
    p=BASE/name
    if not p.exists():raise FileNotFoundError(f"Required experiment report missing: {p}")
    return json.loads(p.read_text())

def save(fig,name):
    for ext in ("png","svg"):
        fig.savefig(FIG/f"{name}.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)

def main():
    stage=read("scheduled_window_v4_e2e.json")
    audit=read("scheduled_window_contract_audit.json")
    controlled=read("scheduled_window_controlled.json")
    if audit["status"]!="PASS" or audit["event_time_reconstruction_parity"]["matching_requests"]!=6455:
        raise RuntimeError("Frozen history parity audit not passed")
    a=sorted(stage["scenarios"],key=lambda x:x["stride_hours"],reverse=True)
    c=sorted(controlled["metrics"],key=lambda x:x["stride_hours"],reverse=True)
    assert [x["scenario"] for x in a]==[x["scenario"] for x in c]==["S0","S1","S2","S3"]
    assert controlled["n_all"]==stage["n_events"]
    FIG.mkdir(parents=True,exist_ok=True)
    names=[x["scenario"] for x in a]
    x=[z["batch_workload_ratio_vs_S0"] for z in a]
    y=[z["auc"] for z in a]
    non_dom=[]
    for i,z in enumerate(a):
        dominated=any(
            q["auc"]>=z["auc"] and q["batch_workload_ratio_vs_S0"]<=z["batch_workload_ratio_vs_S0"]
            and (q["auc"]>z["auc"] or q["batch_workload_ratio_vs_S0"]<z["batch_workload_ratio_vs_S0"])
            for j,q in enumerate(a) if i!=j)
        non_dom.append(not dominated)
    fig,ax=plt.subplots(figsize=(8,5))
    for i,n in enumerate(names):
        ax.scatter(x[i],y[i],s=110,marker="o" if non_dom[i] else "x")
        ax.annotate(n,(x[i],y[i]),xytext=(6,5),textcoords="offset points")
    pts=sorted((x[i],y[i]) for i in range(4) if non_dom[i])
    ax.plot([p[0] for p in pts],[p[1] for p in pts],"--",label="Non-dominated frontier")
    ax.set(xlabel="Relative observed batch workload (S0 = 1)",ylabel="ROC-AUC",
           title="Diagnostic Pareto: quality vs refresh workload")
    ax.grid(alpha=.25);ax.legend()
    fig.text(.02,.01,"Full cohort: OOV-exclusion confounded; workload is a proxy.",fontsize=8)
    save(fig,"pareto_full_cohort_diagnostic")
    fig,ax=plt.subplots(figsize=(8,5))
    auc=[z["complete_case_auc"] for z in c]
    ci=controlled["paired_user_cluster_bootstrap_95_ci"]
    for i,n in enumerate(names):
        ax.scatter(x[i],auc[i],s=110)
        ax.annotate(n,(x[i],auc[i]),xytext=(6,5),textcoords="offset points")
    ax.set(xlabel="All-job workload proxy / S0 (not subset-specific)",
           ylabel="Complete-case ROC-AUC",
           title=f"OOV-free sensitivity (n={controlled['n_complete_case']})")
    ax.grid(alpha=.25)
    fig.text(.02,.01,"Common OOV-free cohort; workload measured on full observed jobs.",fontsize=8)
    save(fig,"pareto_complete_case_sensitivity")
    fig,ax=plt.subplots(figsize=(8,5))
    for j,n in enumerate(names[1:],start=1):
        d=auc[j]-auc[0]
        bounds=ci[n+"_minus_S0"]
        low,high=bounds["lower_95"],bounds["upper_95"]
        ax.errorbar([d],[j],xerr=[[d-low],[high-d]],fmt="o",capsize=5)
    ax.axvline(0,linestyle="--")
    ax.set(yticks=[1,2,3],yticklabels=names[1:],
           xlabel="Paired delta AUC vs S0 (95% user-cluster bootstrap CI)",
           title="Controlled quality comparison")
    ax.grid(axis="x",alpha=.25)
    save(fig,"paired_auc_ci_complete_case")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    oov={q["stride_hours"]:q["job_window_oov_rate"] for q in audit["scheduled_windows"]}
    stride=[z["stride_hours"] for z in a]
    axes[0].plot(stride,[z["auc"] for z in a],"o-",label="All requests; OOV excluded")
    axes[0].plot(stride,auc,"o-",label="Common OOV-free cohort")
    axes[0].set(xlabel="Refresh stride (hours)",ylabel="ROC-AUC",title="Quality sensitivity")
    axes[0].legend()
    axes[1].plot(stride,[100*oov[s] for s in stride],"o-")
    axes[1].set(xlabel="Refresh stride (hours)",ylabel="OOV in refresh jobs (%)",
                title="Frozen vocabulary coverage")
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();save(fig,"quality_oov_diagnostics")
    fig,ax=plt.subplots(figsize=(8,5))
    for z in a:
        ax.scatter(z["mean_refresh_age_hours"],z["auc"],s=110)
        ax.annotate(z["scenario"],(z["mean_refresh_age_hours"],z["auc"]),
                    xytext=(6,5),textcoords="offset points")
    ax.set(xlabel="Mean refresh age (hours)",ylabel="Full-cohort ROC-AUC",
           title="Freshness-quality tradeoff (diagnostic)")
    ax.grid(alpha=.25);save(fig,"freshness_quality_diagnostic")
    summary={"status":"DIAGNOSTIC_NOT_FINAL",
             "pareto_non_dominated":[names[i] for i,v in enumerate(non_dom) if v],
             "pareto_dominated":[names[i] for i,v in enumerate(non_dom) if not v],
             "complete_case_n":controlled["n_complete_case"],
             "caveats":["Full-cohort OOV handling changes histories",
                        "Complete-case population is selected, not the full test population",
                        "Workload is an observed-job interaction proxy, not measured CPU/GPU cost",
                        "No prospective SLA or production resource measurement"]}
    (FIG/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    print("Wrote PNG/SVG figures to",FIG)
if __name__=="__main__":main()
