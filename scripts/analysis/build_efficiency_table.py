import pandas as pd
from pathlib import Path

summary_path = Path("results/final_tables/priority1_gbdt_relstack_summary.csv")
if not summary_path.exists():
    summary_path = Path("results/final_tables/gbdt_compiled_features_summary.csv")

if not summary_path.exists():
    raise FileNotFoundError("Missing priority1_gbdt_relstack_summary.csv or gbdt_compiled_features_summary.csv")

timing_path = Path("results/final_tables/efficiency_timing_summary_relstack.csv")
if not timing_path.exists():
    raise FileNotFoundError("Missing efficiency_timing_summary_relstack.csv. Run build_timing_summary_from_logs.py first.")

perf = pd.read_csv(summary_path)
timing = pd.read_csv(timing_path)

rows = []

mat = timing[timing["stage"] == "feature_materialization"]
mat_time = mat["elapsed_wall_time_sec"].iloc[0] if len(mat) else None
mat_mem = mat["peak_memory_mb"].iloc[0] if len(mat) else None

for _, r in perf.iterrows():
    if r["dataset"] != "rel-stack" or r["task"] != "user-badge":
        continue

    dec = timing[
        (timing["stage"] == "decoder_eval") &
        (timing["variant"] == r["variant"]) &
        (timing["model"] == r["model"])
    ]

    rows.append({
        "dataset": r["dataset"],
        "task": r["task"],
        "model": r["model"],
        "variant": r["variant"],
        "n_features": r["n_features_mean"],
        "n_runs_for_metrics": r["n_runs"],
        "metric_seeds": r["seeds"],
        "feature_materialization_time_sec_seed41_44_export_audit": mat_time,
        "feature_materialization_peak_memory_mb": mat_mem,
        "decoder_eval_time_sec_seed41_audit": dec["elapsed_wall_time_sec"].iloc[0] if len(dec) else None,
        "decoder_peak_memory_mb_seed41_audit": dec["peak_memory_mb"].iloc[0] if len(dec) else None,
        "roc_auc_mean": r["roc_auc_mean"],
        "average_precision_mean": r["average_precision_mean"],
        "log_loss_mean": r["log_loss_mean"],
        "note": "Timing is an implementation audit; feature export was measured over seeds 41-44, decoder timing over seed 41. Predictive metrics are averaged over seeds 41-44.",
    })

out = pd.DataFrame(rows)
out.to_csv("results/final_tables/efficiency_paper_table_relstack.csv", index=False)
out.to_csv("results/efficiency/efficiency_paper_table_relstack.csv", index=False)

print(out.to_string(index=False))
print("\nSaved: results/final_tables/efficiency_paper_table_relstack.csv")
