import pandas as pd
from pathlib import Path

paths = sorted(Path(".").glob("results_budget_curve_relstack_user_badge_feature_budget_curve_rel-stack_user-badge_fdhg_dmax1_small_budget_seed*.csv"))
paths += sorted(Path("results/budget_curve/relstack_user_badge").glob("feature_budget_curve_rel-stack_user-badge_fdhg_dmax1_small_budget_seed*.csv"))

if not paths:
    raise SystemExit("No budget curve csv files found.")

df = pd.concat([pd.read_csv(p).assign(source_file=str(p)) for p in paths], ignore_index=True)

out_dir = Path("results/final_tables")
out_dir.mkdir(parents=True, exist_ok=True)

all_path = out_dir / "feature_budget_curve_relstack_user_badge_all_runs.csv"
summary_path = out_dir / "feature_budget_curve_relstack_user_badge_summary.csv"

df.to_csv(all_path, index=False)

summary = df.groupby(["dataset", "task", "variant", "K"], as_index=False).agg(
    n_runs=("seed", "nunique"),
    seeds=("seed", lambda s: ",".join(map(str, sorted(set(map(int, s)))))),
    n_features_total_mean=("n_features_total", "mean"),
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    roc_auc_mean=("roc_auc", "mean"),
    roc_auc_std=("roc_auc", "std"),
    average_precision_mean=("average_precision", "mean"),
    average_precision_std=("average_precision", "std"),
    log_loss_mean=("log_loss", "mean"),
    log_loss_std=("log_loss", "std"),
)

summary.to_csv(summary_path, index=False)

print("Saved:")
print(all_path)
print(summary_path)
print()
print(summary.to_string(index=False))
