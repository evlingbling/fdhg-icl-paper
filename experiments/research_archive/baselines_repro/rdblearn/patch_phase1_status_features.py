from pathlib import Path
import pandas as pd

p = Path("baselines_repro/results/phase1_relstack_status_table.csv")
df = pd.read_csv(p)

# RDBLearn debug log listed 53 raw DFS columns including timestamp/UserId before preprocessing.
# After TemporalDiffTransformer, RDBLearn additionally generates temporal-difference features internally.
df["n_features_note"] = ""
df["feature_budget_note"] = ""
df["preprocessing_note"] = ""

mask = df["method"].eq("RDBLearn_DFS_TabPFN_no_target_history")
df.loc[mask, "n_features_note"] = (
    "RDBLearn generated DFS feature set; debug log lists 53 augmented columns before internal preprocessing"
)
df.loc[mask, "feature_budget_note"] = (
    "External direct RDBLearn baseline; not identical feature-budget ablation"
)
df.loc[mask, "preprocessing_note"] = (
    "RDBLearn uses its own DFS generation and preprocessing pipeline with cutoff_time_column=timestamp"
)

mask2 = df["method"].isin(["target_only", "naive", "dfs", "fdhg_dmax1"])
df.loc[mask2, "feature_budget_note"] = (
    "Existing intermediate MVP TabPFN result; see paper_main_runs for exact n_features"
)
df.loc[mask2, "preprocessing_note"] = (
    "Internal FDHG-ICL preprocessing/evaluation pipeline"
)

out = Path("results/final_tables/phase1_relstack_status_table_with_notes.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(df.to_string(index=False))
print("Wrote:", out)
