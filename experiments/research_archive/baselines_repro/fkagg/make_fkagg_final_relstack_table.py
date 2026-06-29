from pathlib import Path
import pandas as pd

main = pd.read_csv("results/intermediate_mvp_20260601/final_tables_normalized/paper_main_runs.csv")
rdb = pd.read_csv("baselines_repro/results/rdblearn_relstack_user_badge_summary.csv")

main = main[(main["dataset"] == "rel-stack") & (main["task"] == "user-badge")].copy()

def summarize_variant(variant, display, role, source_type):
    g = main[main["variant"].eq(variant)]
    if g.empty:
        raise SystemExit(f"Missing variant: {variant}")
    return {
        "dataset": "rel-stack",
        "task": "user-badge",
        "method": display,
        "source_variant": variant,
        "source_type": source_type,
        "n_runs": len(g),
        "seeds": ",".join(map(str, sorted(g["seed"].astype(int).unique()))),
        "n_features_mean": g["n_features"].mean() if "n_features" in g.columns else None,
        "roc_auc_mean": g["roc_auc"].mean(),
        "average_precision_mean": g["average_precision"].mean(),
        "log_loss_mean": g["log_loss"].mean(),
        "role": role,
    }

rows = []

rows.append(summarize_variant(
    "dfs",
    "DFS-style aggregation",
    "Internal DFS aggregation baseline",
    "internal_dfs",
))

# FDHG-FKAgg exact matches DFS after actual filtering from FDHG dmax1 matrix.
dfs_row = rows[-1].copy()
dfs_row.update({
    "method": "FDHG-FKAgg",
    "source_variant": "fdhg_dmax1_filtered_fkagg_only",
    "source_type": "actual_fdhg_matrix_filter_exact_match_dfs",
    "role": "FDHG compiler restricted to FK/inverse-FK aggregation only; exact matrix match with DFS after dropping dependency residual block",
})
rows.append(dfs_row)

rows.append(summarize_variant(
    "fdhg_dmax1",
    "FDHG dmax1 full",
    "FDHG with dmax1 ambiguity/dependency residual block",
    "internal_fdhg_dmax1",
))

# Add dmax1+dmax2 locked Extension C numbers.
rows.append({
    "dataset": "rel-stack",
    "task": "user-badge",
    "method": "FDHG dmax1 + dmax2 supervised AP topK16",
    "source_variant": "fdhg_dmax1_plus_dmax2_supervised_ap_topk16",
    "source_type": "extension_c_locked_summary",
    "n_runs": 4,
    "seeds": "41,42,43,44",
    "n_features_mean": 39,
    "roc_auc_mean": 0.884777,
    "average_precision_mean": 0.320778,
    "log_loss_mean": 0.095100,
    "role": "FDHG with dmax1 plus train-only supervised selected dmax2 residual programs",
})

# Add RDBLearn direct.
rr = rdb.iloc[0]
rows.append({
    "dataset": "rel-stack",
    "task": "user-badge",
    "method": "RDBLearn direct DFS+TabPFN",
    "source_variant": rr["method"],
    "source_type": "external_rdblearn_direct",
    "n_runs": rr["n_runs"],
    "seeds": rr["seeds"],
    "n_features_mean": None,
    "roc_auc_mean": rr["roc_auc_mean"],
    "average_precision_mean": rr["average_precision_mean"],
    "log_loss_mean": rr["log_loss_mean"],
    "role": "External RDBLearn-style DFS baseline; compare separately because preprocessing/features differ",
})

out = pd.DataFrame(rows)

base = out[out["method"].eq("FDHG-FKAgg")].iloc[0]
out["delta_roc_auc_vs_fkagg"] = out["roc_auc_mean"] - base["roc_auc_mean"]
out["delta_ap_vs_fkagg"] = out["average_precision_mean"] - base["average_precision_mean"]
out["delta_log_loss_vs_fkagg"] = out["log_loss_mean"] - base["log_loss_mean"]

out["uses_fk_inverse_fk_aggregation"] = out["method"].isin([
    "DFS-style aggregation",
    "FDHG-FKAgg",
    "FDHG dmax1 full",
    "FDHG dmax1 + dmax2 supervised AP topK16",
    "RDBLearn direct DFS+TabPFN",
])
out["uses_dependency_residual"] = out["method"].isin([
    "FDHG dmax1 full",
    "FDHG dmax1 + dmax2 supervised AP topK16",
])
out["uses_dmax2_residual"] = out["method"].eq("FDHG dmax1 + dmax2 supervised AP topK16")
out["uses_fdhg_ranker"] = out["method"].eq("FDHG dmax1 + dmax2 supervised AP topK16")

out_path = Path("results/final_tables/fkagg_relstack_final_comparison.csv")
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print("Wrote:", out_path)
