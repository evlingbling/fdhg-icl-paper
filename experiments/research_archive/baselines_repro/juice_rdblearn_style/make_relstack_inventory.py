from pathlib import Path
import pandas as pd

OUT_DIR = Path("baselines_repro/juice_rdblearn_style/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

candidate_tables = [
    Path("results/intermediate_mvp_20260601/final_tables_normalized/clean_main_4task_runs.csv"),
    Path("results/intermediate_mvp_20260601/final_tables_normalized/final_main_runs.csv"),
    Path("results/intermediate_mvp_20260601/final_tables_normalized/final_all_runs.csv"),
    Path("results/intermediate_mvp_20260601/final_tables_normalized/paper_main_runs.csv"),
]

frames = []
for p in candidate_tables:
    if not p.exists():
        continue
    df = pd.read_csv(p)
    df["_source_csv"] = str(p)
    frames.append(df)

if not frames:
    raise SystemExit("No candidate run tables found.")

df = pd.concat(frames, ignore_index=True)

# Normalize columns defensively
for col in ["dataset", "task", "variant"]:
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}")

rel = df[
    (df["dataset"].astype(str).str.contains("rel-stack", case=False, na=False)) &
    (df["task"].astype(str).str.contains("user-badge", case=False, na=False))
].copy()

if rel.empty:
    raise SystemExit("No rel-stack/user-badge rows found.")

cols = [
    c for c in [
        "dataset", "task", "variant", "seed", "model", "split",
        "n_train", "n_val", "n_features", "feature_budget",
        "dmax", "uses_afd", "uses_ambiguity", "uses_dmax2",
        "uses_supervised_ranker", "is_fdhg_fallback",
        "accuracy", "roc_auc", "average_precision", "log_loss",
        "runtime_sec", "feature_path", "result_path", "status",
        "failure_reason", "_source_csv"
    ]
    if c in rel.columns
]

rel_out = rel[cols].drop_duplicates()
rel_out.to_csv(OUT_DIR / "relstack_existing_run_inventory.csv", index=False)

summary_cols = [c for c in ["variant", "model", "seed", "n_features", "roc_auc", "average_precision", "log_loss", "feature_path", "_source_csv"] if c in rel_out.columns]
print(rel_out[summary_cols].to_string(index=False))
print()
print("Wrote:", OUT_DIR / "relstack_existing_run_inventory.csv")
