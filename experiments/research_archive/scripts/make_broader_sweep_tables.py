#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

METRIC_COLS = ["accuracy", "roc_auc", "average_precision", "log_loss"]

VARIANT_NORMALIZATION = {
    "target": "target_only",
    "target_only": "target_only",
    "base_only": "target_only",

    "naive": "naive",
    "naive_latest_flatten": "naive_latest_flatten",

    "dfs": "dfs",
    "DFS": "dfs",

    "fdhg": "fdhg_dmax1",
    "fdhg_dmax1": "fdhg_dmax1",
    "fdhg_dmax1_full": "fdhg_dmax1",
    "fdhg_full": "fdhg_dmax1",

    "fdhg_fkagg": "fdhg_fkagg",
    "fkagg": "fdhg_fkagg",

    "RDBLearn_DFS_TabPFN_no_target_history": "rdblearn",
    "RDBLearn direct DFS+TabPFN": "rdblearn",
    "rdblearn": "rdblearn",
    "juice_or_rdblearn": "juice_or_rdblearn",

    "last_only": "last_only",
    "dfs_plus_last": "dfs_plus_last",
    "fdhg_plus_last": "fdhg_plus_last",

    "dmax1_plus_dmax2_topK16": "fdhg_dmax1_dmax2_heuristic_topk16",
    "dmax1_plus_dmax2_topk16": "fdhg_dmax1_dmax2_heuristic_topk16",
    "fdhg_dmax1_plus_dmax2_topK16": "fdhg_dmax1_dmax2_heuristic_topk16",
    "fdhg_dmax1_plus_dmax2_topk16": "fdhg_dmax1_dmax2_heuristic_topk16",

    "dmax1_plus_dmax2_supervised_ap_topK16": "fdhg_dmax1_dmax2_supervised_ap_topk16",
    "dmax1_plus_dmax2_supervised_ap_topk16": "fdhg_dmax1_dmax2_supervised_ap_topk16",
    "fdhg_dmax1_plus_dmax2_supervised_ap_topK16": "fdhg_dmax1_dmax2_supervised_ap_topk16",
    "fdhg_dmax1_plus_dmax2_supervised_ap_topk16": "fdhg_dmax1_dmax2_supervised_ap_topk16",

    "dmax1_plus_dmax2_supervised_auc_topK16": "fdhg_dmax1_dmax2_supervised_auc_topk16",
    "dmax1_plus_dmax2_supervised_auc_topk16": "fdhg_dmax1_dmax2_supervised_auc_topk16",
    "fdhg_dmax1_plus_dmax2_supervised_auc_topK16": "fdhg_dmax1_dmax2_supervised_auc_topk16",
    "fdhg_dmax1_plus_dmax2_supervised_auc_topk16": "fdhg_dmax1_dmax2_supervised_auc_topk16",

    "random_same_budget": "random_same_budget",
    "fdhg_dmax1_random_same_budget": "random_same_budget",

    "shuffle_ambiguity": "shuffle_ambiguity",
    "fdhg_dmax1_shuffle_ambiguity": "shuffle_ambiguity",
}

# 낮을수록 우선순위 높음.
SOURCE_PRIORITY = {
    "clean_main_4task_runs.csv": 0,
    "clean_appendix_7task_seed41.csv": 0,
    "broader_sweep_manual_backfill_seed41.csv": 0,
    "rdblearn_relstack_user_badge_seed41_44.csv": 0,
    "rdblearn_relamazon_item_churn_all_runs.csv": 0,
    "relstack_user_badge_ablation_all_runs.csv": 1,
    "amazon_item_churn_ablation_all_runs.csv": 1,
    "f1_driver_dnf_temporal_diagnostic_all_runs.csv": 1,
    "final_main_runs.csv": 5,
    "final_all_runs.csv": 6,
}

def source_priority(path_str: str) -> int:
    return SOURCE_PRIORITY.get(Path(str(path_str)).name, 10)

def read_csv_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        df["source_priority"] = source_priority(str(path))
        return df
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "roc_auc_mean": "roc_auc",
        "average_precision_mean": "average_precision",
        "log_loss_mean": "log_loss",
        "accuracy_mean": "accuracy",
        "n_features_total": "n_features",
        "n_features_total_mean": "n_features",
        "n_features_mean": "n_features",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "variant" not in df.columns and "method" in df.columns:
        df["variant"] = df["method"]

    for c in ["dataset", "task", "variant"]:
        if c not in df.columns:
            df[c] = np.nan

    if "seed" not in df.columns:
        df["seed"] = np.nan

    if "n_features" not in df.columns:
        df["n_features"] = np.nan

    for c in METRIC_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df["variant_raw"] = df["variant"].astype(str)
    df["variant"] = df["variant_raw"].map(VARIANT_NORMALIZATION).fillna(df["variant_raw"])

    keep = [
        "dataset", "task", "variant", "variant_raw", "seed", "n_features",
        "accuracy", "roc_auc", "average_precision", "log_loss",
        "source_file", "source_priority"
    ]
    return df[keep]

def load_candidate_results(results_dir: Path) -> pd.DataFrame:
    final_tables = results_dir / "final_tables"

    # Run-level files only. Do not read summary/efficiency files here.
    candidate_names = [
        "clean_main_4task_runs.csv",
        "clean_appendix_7task_seed41.csv",
        "broader_sweep_manual_backfill_seed41.csv",
        "rdblearn_relstack_user_badge_seed41_44.csv",
        "rdblearn_relamazon_item_churn_all_runs.csv",
        "relstack_user_badge_ablation_all_runs.csv",
        "amazon_item_churn_ablation_all_runs.csv",
        "f1_driver_dnf_temporal_diagnostic_all_runs.csv",
        "final_main_runs.csv",
        "final_all_runs.csv",
    ]

    dfs = []
    for name in candidate_names:
        df = read_csv_if_exists(final_tables / name)
        if df is not None:
            dfs.append(normalize_columns(df))

    broader_dir = results_dir / "broader_sweep"
    if broader_dir.exists():
        for path in sorted(broader_dir.glob("*.csv")):
            df = read_csv_if_exists(path)
            if df is not None and {"dataset", "task", "variant"}.issubset(df.columns):
                dfs.append(normalize_columns(df))

    if not dfs:
        return pd.DataFrame(columns=[
            "dataset", "task", "variant", "variant_raw", "seed", "n_features",
            "accuracy", "roc_auc", "average_precision", "log_loss",
            "source_file", "source_priority"
        ])

    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["dataset", "task", "variant"])

    metric_nonnull = out[METRIC_COLS].notna().any(axis=1)
    out = out[metric_nonnull].copy()

    # 중요: 같은 dataset/task/variant/seed는 가장 신뢰도 높은 source 하나만 남김.
    out = out.sort_values(
        ["dataset", "task", "variant", "seed", "source_priority"]
    )
    out = out.drop_duplicates(
        subset=["dataset", "task", "variant", "seed"],
        keep="first"
    )

    return out.reset_index(drop=True)

def attach_taxonomy(runs: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    cfg = config[["dataset", "task", "taxonomy", "expected_role", "notes"]].drop_duplicates()
    return runs.merge(cfg, on=["dataset", "task"], how="left")

def make_summary(runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "task", "taxonomy", "variant"]
    rows = []

    for keys, g in runs.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(g)
        row["seeds"] = ",".join(sorted({str(x) for x in g["seed"].dropna().tolist()}))
        row["n_features_mean"] = pd.to_numeric(g["n_features"], errors="coerce").mean()
        row["source_files"] = " | ".join(sorted(set(g["source_file"].astype(str))))

        for m in METRIC_COLS:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std(ddof=1) if vals.notna().sum() > 1 else 0.0

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dataset", "task", "variant"]).reset_index(drop=True)
    return out

def pick_best(summary: pd.DataFrame, variants: list[str]):
    cand = summary[summary["variant"].isin(variants)].copy()
    cand = cand[pd.to_numeric(cand["roc_auc_mean"], errors="coerce").notna()]
    if cand.empty:
        return None
    cand = cand.sort_values(
        ["roc_auc_mean", "average_precision_mean", "log_loss_mean"],
        ascending=[False, False, True]
    )
    return cand.iloc[0]

def classify_outcome(task_summary: pd.DataFrame, taxonomy: str) -> dict:
    non_fdhg_variants = [
        "target_only",
        "naive",
        "naive_latest_flatten",
        "dfs",
        "juice",
        "rdblearn",
        "juice_or_rdblearn",
        "last_only",
        "dfs_plus_last",
    ]

    fdhg_variants = [
        "fdhg_fkagg",
        "fdhg_dmax1",
        "fdhg_dmax1_dmax2_heuristic_topk16",
        "fdhg_dmax1_dmax2_supervised_ap_topk16",
        "fdhg_dmax1_dmax2_supervised_auc_topk16",
        "fdhg_plus_last",
    ]

    base = pick_best(task_summary, non_fdhg_variants)
    fdhg = pick_best(task_summary, fdhg_variants)

    out = {
        "best_non_fdhg_baseline": np.nan,
        "best_non_fdhg_roc_auc": np.nan,
        "best_non_fdhg_ap": np.nan,
        "best_non_fdhg_log_loss": np.nan,
        "best_fdhg_variant": np.nan,
        "best_fdhg_roc_auc": np.nan,
        "best_fdhg_ap": np.nan,
        "best_fdhg_log_loss": np.nan,
        "delta_roc_auc": np.nan,
        "delta_ap": np.nan,
        "delta_log_loss": np.nan,
        "outcome": "insufficient_results",
        "interpretation": "Not enough completed variants to classify.",
    }

    if base is None or fdhg is None:
        return out

    out.update({
        "best_non_fdhg_baseline": base["variant"],
        "best_non_fdhg_roc_auc": base["roc_auc_mean"],
        "best_non_fdhg_ap": base["average_precision_mean"],
        "best_non_fdhg_log_loss": base["log_loss_mean"],
        "best_fdhg_variant": fdhg["variant"],
        "best_fdhg_roc_auc": fdhg["roc_auc_mean"],
        "best_fdhg_ap": fdhg["average_precision_mean"],
        "best_fdhg_log_loss": fdhg["log_loss_mean"],
    })

    d_auc = fdhg["roc_auc_mean"] - base["roc_auc_mean"]
    d_ap = fdhg["average_precision_mean"] - base["average_precision_mean"]
    d_ll = fdhg["log_loss_mean"] - base["log_loss_mean"]

    out["delta_roc_auc"] = d_auc
    out["delta_ap"] = d_ap
    out["delta_log_loss"] = d_ll

    dfs = task_summary[task_summary["variant"].eq("dfs")]
    f1 = task_summary[task_summary["variant"].eq("fdhg_dmax1")]

    if not dfs.empty and not f1.empty:
        dfs_r = dfs.iloc[0]
        f1_r = f1.iloc[0]
        same_auc = abs(f1_r["roc_auc_mean"] - dfs_r["roc_auc_mean"]) < 1e-9
        same_ap = abs(f1_r["average_precision_mean"] - dfs_r["average_precision_mean"]) < 1e-9
        same_ll = abs(f1_r["log_loss_mean"] - dfs_r["log_loss_mean"]) < 1e-9
        same_nf = (
            pd.notna(f1_r["n_features_mean"]) and pd.notna(dfs_r["n_features_mean"]) and
            abs(f1_r["n_features_mean"] - dfs_r["n_features_mean"]) < 1e-9
        )

        if same_auc and same_ap and same_ll and same_nf:
            out["outcome"] = "fallback"
            out["interpretation"] = "FDHG safely degenerates to DFS; no effective dependency-specific gain was added."
            return out

    if taxonomy in {"temporal_local_state", "temporal_event_history"}:
        temporal_base = pick_best(task_summary, ["last_only", "dfs_plus_last"])
        static_fdhg = pick_best(task_summary, ["fdhg_dmax1", "fdhg_fkagg"])
        if temporal_base is not None and static_fdhg is not None:
            if temporal_base["roc_auc_mean"] - static_fdhg["roc_auc_mean"] > 0.01:
                out["outcome"] = "temporal_diagnostic"
                out["interpretation"] = "Explicit last/recency-style temporal features outperform static FDHG; motivates FDHG-T operators."
                return out

    improved = 0
    if d_auc > 0.002:
        improved += 1
    if d_ap > 0.002:
        improved += 1
    if d_ll < -0.002:
        improved += 1

    worsened = 0
    if d_auc < -0.002:
        worsened += 1
    if d_ap < -0.002:
        worsened += 1
    if d_ll > 0.002:
        worsened += 1

    if improved >= 2:
        out["outcome"] = "win"
        out["interpretation"] = "FDHG improves over the strongest available non-FDHG baseline on multiple metrics."
    elif worsened >= 2:
        out["outcome"] = "loss"
        out["interpretation"] = "FDHG underperforms the strongest available non-FDHG baseline; candidate features may be noisy or mismatched."
    else:
        out["outcome"] = "tie"
        out["interpretation"] = "FDHG is broadly comparable to the strongest available non-FDHG baseline."

    return out

def make_outcome_table(summary: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, cfg in config.iterrows():
        dataset = cfg["dataset"]
        task = cfg["task"]
        taxonomy = cfg["taxonomy"]

        g = summary[(summary["dataset"] == dataset) & (summary["task"] == task)]
        cls = classify_outcome(g, taxonomy)

        row = {
            "dataset": dataset,
            "task": task,
            "taxonomy": taxonomy,
            "expected_role": cfg.get("expected_role", ""),
        }
        row.update(cls)
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/broader_relbench_sweep_tasks.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/final_tables")
    args = ap.parse_args()

    config = pd.read_csv(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_candidate_results(Path(args.results_dir))
    runs = attach_taxonomy(runs, config)

    allowed = set(zip(config["dataset"], config["task"]))
    runs = runs[runs.apply(lambda r: (r["dataset"], r["task"]) in allowed, axis=1)].copy()

    summary = make_summary(runs)
    outcome = make_outcome_table(summary, config)

    runs_path = out_dir / "broader_sweep_all_runs.csv"
    summary_path = out_dir / "broader_sweep_summary.csv"
    outcome_path = out_dir / "broader_sweep_outcome_table.csv"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    outcome.to_csv(outcome_path, index=False)

    print(f"[OK] wrote {runs_path} rows={len(runs)}")
    print(f"[OK] wrote {summary_path} rows={len(summary)}")
    print(f"[OK] wrote {outcome_path} rows={len(outcome)}")

    print("\n=== Outcome table ===")
    cols = [
        "dataset", "task", "taxonomy",
        "best_non_fdhg_baseline",
        "best_fdhg_variant",
        "delta_roc_auc",
        "delta_ap",
        "delta_log_loss",
        "outcome",
    ]
    print(outcome[cols].to_string(index=False))

if __name__ == "__main__":
    main()
