from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("results/final_tables")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

def add_row(
    dataset, task, variant, protocol,
    n_features=None, n_val=None,
    accuracy=None, roc_auc=None, average_precision=None, log_loss=None,
    source="", note=""
):
    rows.append({
        "dataset": dataset,
        "task": task,
        "variant": variant,
        "protocol": protocol,
        "n_features": n_features,
        "n_val": n_val,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "log_loss": log_loss,
        "source": source,
        "note": note,
    })

def load_summary(path):
    p = Path(path)
    if not p.exists():
        print("[MISSING]", p)
        return None
    return pd.read_csv(p).iloc[0]

# -------------------------
# rel-stack/user-badge
# -------------------------

add_row(
    "rel-stack", "user-badge", "dfs",
    "full-val seeds 41-44",
    n_features=17, n_val=247398,
    accuracy=None,
    roc_auc=0.866341,
    average_precision=0.293363,
    log_loss=0.098644,
    source="locked main summary",
    note="DFS aggregation baseline"
)

add_row(
    "rel-stack", "user-badge", "fdhg_fkagg",
    "full-val seeds 41-44",
    n_features=17, n_val=247398,
    accuracy=None,
    roc_auc=0.866341,
    average_precision=0.293363,
    log_loss=0.098644,
    source="FKAgg exact-match audit",
    note="exactly matches DFS after removing FDHG ambiguity residuals"
)

add_row(
    "rel-stack", "user-badge", "fdhg_dmax1",
    "full-val seeds 41-44",
    n_features=25, n_val=247398,
    accuracy=0.971805,
    roc_auc=0.881294,
    average_precision=0.310060,
    log_loss=0.096172,
    source="locked main summary",
    note="improves over DFS/FKAgg"
)

add_row(
    "rel-stack", "user-badge", "fdhg_dmax1_plus_dmax2_supervised_ap_topK16",
    "full-val seeds 41-44",
    n_features=41, n_val=247398,
    accuracy=0.972110,
    roc_auc=0.884777,
    average_precision=0.320778,
    log_loss=0.095100,
    source="Extension C AP-ranker summary",
    note="best FDHG TabPFN rel-stack variant currently recorded"
)

add_row(
    "rel-stack", "user-badge", "rdblearn_direct",
    "full-val seeds 41-44",
    n_features=2, n_val=247398,
    accuracy=None,
    roc_auc=0.875645,
    average_precision=0.323222,
    log_loss=0.325207,
    source="RDBLearn direct note",
    note="higher AP than FDHG dmax1, worse AUROC than FDHG dmax1"
)

s = load_summary("results/final_tables/phase1_relstack_juice_style_matched_summary.csv")
if s is None:
    s = load_summary("results/phase1_juice_style_relstack_eval_fullval/summary.csv")
if s is not None:
    add_row(
        "rel-stack", "user-badge", "juice_style_matched_v0",
        "full-val seeds 41-44",
        n_features=s.get("n_features_mean"),
        n_val=s.get("n_val_used_mean"),
        accuracy=s.get("accuracy_mean"),
        roc_auc=s.get("roc_auc_mean"),
        average_precision=s.get("average_precision_mean"),
        log_loss=s.get("log_loss_mean"),
        source="phase1_relstack_juice_style_matched_summary",
        note="stronger than current FDHG variants; no FDHG-specific columns"
    )

s = load_summary("results/final_tables/phase1_relstack_juice_on_fdhg_sample_summary.csv")
if s is None:
    s = load_summary("results/phase1_juice_on_fdhg_sample_relstack_eval/summary.csv")
if s is not None:
    add_row(
        "rel-stack", "user-badge", "juice_on_fdhg_sample",
        "sampled-val same-row seeds 41-44",
        n_features=s.get("n_features_mean"),
        n_val=s.get("n_val_used_mean"),
        accuracy=s.get("accuracy_mean"),
        roc_auc=s.get("roc_auc_mean"),
        average_precision=s.get("average_precision_mean"),
        log_loss=s.get("log_loss_mean"),
        source="same-sample residual comparison",
        note="JUICE-only baseline on same sampled rows as residual test"
    )

s = load_summary("results/final_tables/phase1_relstack_juice_plus_fdhg_residual_sample_summary.csv")
if s is None:
    s = load_summary("results/phase1_juice_plus_fdhg_residual_relstack_eval_fullval/summary.csv")
if s is not None:
    add_row(
        "rel-stack", "user-badge", "juice_plus_fdhg_residual",
        "sampled-val same-row seeds 41-44",
        n_features=s.get("n_features_mean"),
        n_val=s.get("n_val_used_mean"),
        accuracy=s.get("accuracy_mean"),
        roc_auc=s.get("roc_auc_mean"),
        average_precision=s.get("average_precision_mean"),
        log_loss=s.get("log_loss_mean"),
        source="same-sample residual comparison",
        note="does not improve over JUICE-only on same sampled rows"
    )

# -------------------------
# rel-amazon/item-churn
# -------------------------

add_row(
    "rel-amazon", "item-churn", "dfs",
    "sampled 10k/2k seeds 41-44",
    n_features=None, n_val=2000,
    accuracy=None,
    roc_auc=0.757267,
    average_precision=0.631661,
    log_loss=0.603545,
    source="locked item-churn summary",
    note="DFS sampled protocol"
)

add_row(
    "rel-amazon", "item-churn", "fdhg_fkagg",
    "sampled 10k/2k seeds 41-44",
    n_features=None, n_val=2000,
    accuracy=None,
    roc_auc=0.757267,
    average_precision=0.631661,
    log_loss=0.603545,
    source="FKAgg exact-match audit",
    note="exactly matches DFS after removing brand->category ambiguity residuals"
)

add_row(
    "rel-amazon", "item-churn", "fdhg_dmax1",
    "sampled 10k/2k seeds 41-44",
    n_features=None, n_val=2000,
    accuracy=None,
    roc_auc=0.758332,
    average_precision=0.632755,
    log_loss=0.602888,
    source="locked item-churn summary",
    note="small positive over DFS"
)

add_row(
    "rel-amazon", "item-churn", "fdhg_dmax1_plus_dmax2",
    "sampled 10k/2k seeds 41-44",
    n_features=None, n_val=2000,
    accuracy=None,
    roc_auc=0.757267 + 0.007340,
    average_precision=0.631661 + 0.007282,
    log_loss=0.603545 - 0.012407,
    source="Stage C dmax2 mean deltas plus DFS",
    note="beats DFS and JUICE-style v0 on AUROC/AP/log-loss"
)

add_row(
    "rel-amazon", "item-churn", "rdblearn_direct",
    "sampled/compatible protocol seeds 41-44",
    n_features=2, n_val=2000,
    accuracy=0.731932,
    roc_auc=0.815231,
    average_precision=0.720716,
    log_loss=0.522804,
    source="RDBLearn direct summary",
    note="strongest current item-churn baseline"
)

s = load_summary("results/final_tables/phase1_amazon_item_churn_juice_style_matched_summary.csv")
if s is None:
    s = load_summary("results/phase1_juice_style_amazon_item_churn_eval/summary.csv")
if s is not None:
    add_row(
        "rel-amazon", "item-churn", "juice_style_matched_v0",
        "sampled 10k/2k seeds 41-44",
        n_features=s.get("n_features_mean"),
        n_val=s.get("n_val_used_mean"),
        accuracy=s.get("accuracy_mean"),
        roc_auc=s.get("roc_auc_mean"),
        average_precision=s.get("average_precision_mean"),
        log_loss=s.get("log_loss_mean"),
        source="phase1_amazon_item_churn_juice_style_matched_summary",
        note="competitive with DFS AUROC, below FDHG+dmax2 and RDBLearn direct"
    )

# -------------------------
# rel-amazon/item-ltv
# -------------------------

add_row(
    "rel-amazon", "item-ltv", "dfs_catboost",
    "sampled/compatible CatBoost regression seeds 41-44",
    n_features=9,
    n_val=None,
    accuracy=None,
    roc_auc=None,
    average_precision=None,
    log_loss=None,
    source="item-ltv CatBoost regression summary",
    note="RMSE 1357.621071, MAE 124.563358, R2 -28.548098, RMSE_log1p 1.262821"
)

add_row(
    "rel-amazon", "item-ltv", "fdhg_dmax1_catboost",
    "sampled/compatible CatBoost regression seeds 41-44",
    n_features=17,
    n_val=None,
    accuracy=None,
    roc_auc=None,
    average_precision=None,
    log_loss=None,
    source="item-ltv CatBoost regression summary",
    note="RMSE 700.466833, MAE 106.807111, R2 -6.915054, RMSE_log1p 1.224791"
)

# -------------------------
# Save
# -------------------------

df = pd.DataFrame(rows)

metric_cols = ["accuracy", "roc_auc", "average_precision", "log_loss"]
for c in metric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df.to_csv(OUT / "phase1_baseline_comparison_table.csv", index=False)

# Pairwise deltas for key claims.
delta_rows = []

def add_delta(dataset, task, better, base, metric="roc_auc"):
    sub = df[(df["dataset"] == dataset) & (df["task"] == task)]
    a = sub[sub["variant"] == better]
    b = sub[sub["variant"] == base]
    if a.empty or b.empty:
        return
    ar = a.iloc[0]
    br = b.iloc[0]
    row = {
        "dataset": dataset,
        "task": task,
        "comparison": f"{better} - {base}",
    }
    for m in ["accuracy", "roc_auc", "average_precision", "log_loss"]:
        av = ar[m]
        bv = br[m]
        row[m + "_delta"] = av - bv if pd.notna(av) and pd.notna(bv) else np.nan
    delta_rows.append(row)

add_delta("rel-stack", "user-badge", "fdhg_dmax1", "dfs")
add_delta("rel-stack", "user-badge", "fdhg_dmax1_plus_dmax2_supervised_ap_topK16", "dfs")
add_delta("rel-stack", "user-badge", "juice_style_matched_v0", "fdhg_dmax1_plus_dmax2_supervised_ap_topK16")
add_delta("rel-stack", "user-badge", "juice_plus_fdhg_residual", "juice_on_fdhg_sample")

add_delta("rel-amazon", "item-churn", "fdhg_dmax1", "dfs")
add_delta("rel-amazon", "item-churn", "fdhg_dmax1_plus_dmax2", "dfs")
add_delta("rel-amazon", "item-churn", "fdhg_dmax1_plus_dmax2", "juice_style_matched_v0")
add_delta("rel-amazon", "item-churn", "rdblearn_direct", "fdhg_dmax1_plus_dmax2")

deltas = pd.DataFrame(delta_rows)
deltas.to_csv(OUT / "phase1_baseline_comparison_deltas.csv", index=False)

print("\n=== phase1_baseline_comparison_table ===")
print(df.to_string(index=False))

print("\n=== phase1_baseline_comparison_deltas ===")
print(deltas.to_string(index=False))

print("\nSaved:")
print(OUT / "phase1_baseline_comparison_table.csv")
print(OUT / "phase1_baseline_comparison_deltas.csv")
