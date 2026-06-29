from pathlib import Path
import argparse
import pandas as pd


DEFAULT_METHOD_ORDER = [
    "target_only",
    "naive",
    "dfs",
    "RDBLearn_DFS_TabPFN_no_target_history",
    "fdhg_fkagg",
    "fdhg_dmax1",
    "fdhg_dmax1_plus_dmax2_supervised_ap_topk16",
]


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_main_rows(df):
    variant_col = find_col(df, ["variant", "method"])
    if variant_col is None:
        raise ValueError(f"No variant/method column found. Columns={list(df.columns)}")

    rename = {variant_col: "method"}
    for c in ["roc_auc", "average_precision", "log_loss", "accuracy"]:
        if c in df.columns and f"{c}_mean" not in df.columns:
            # keep raw per-seed columns as-is
            pass

    out = df.rename(columns=rename).copy()
    return out


def summarize_if_needed(df):
    """
    Accepts either per-seed rows or already summarized rows.
    Returns summarized rows with metric means.
    """
    if "roc_auc_mean" in df.columns and "average_precision_mean" in df.columns:
        return df.copy()

    required = ["dataset", "task", "method", "seed", "roc_auc", "average_precision", "log_loss"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot summarize. Missing columns={missing}. Existing={list(df.columns)}")

    agg = {
        "n_runs": ("seed", "count"),
        "seeds": ("seed", lambda x: ",".join(map(str, sorted(pd.Series(x).dropna().astype(int).unique())))),
        "roc_auc_mean": ("roc_auc", "mean"),
        "roc_auc_std": ("roc_auc", "std"),
        "average_precision_mean": ("average_precision", "mean"),
        "average_precision_std": ("average_precision", "std"),
        "log_loss_mean": ("log_loss", "mean"),
        "log_loss_std": ("log_loss", "std"),
    }
    if "accuracy" in df.columns:
        agg["accuracy_mean"] = ("accuracy", "mean")
        agg["accuracy_std"] = ("accuracy", "std")
    if "n_features" in df.columns:
        agg["n_features_mean"] = ("n_features", "mean")
    elif "n_features_total" in df.columns:
        agg["n_features_mean"] = ("n_features_total", "mean")

    keys = ["dataset", "task", "method"]
    if "decoder" in df.columns:
        keys.append("decoder")

    return df.groupby(keys, dropna=False).agg(**agg).reset_index()


def load_optional_csv(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def add_fkagg_alias(summary):
    """
    Creates FDHG-FKAgg rows from DFS rows.
    This is correct only when DFS is the FK/inverse-FK aggregation-only block.
    The output explicitly records source_method=dfs and source_type=dfs_alias.
    """
    if "method" not in summary.columns:
        raise ValueError("summary must contain method column")

    dfs = summary[summary["method"].astype(str).eq("dfs")].copy()
    if dfs.empty:
        # Some tables may use display name
        dfs = summary[summary["method"].astype(str).str.contains("DFS", case=False, na=False)].copy()

    if dfs.empty:
        raise ValueError("No DFS rows found; cannot create FDHG-FKAgg alias.")

    fkagg = dfs.copy()
    fkagg["source_method"] = fkagg["method"]
    fkagg["method"] = "fdhg_fkagg"
    fkagg["method_display"] = "FDHG-FKAgg"
    fkagg["source_type"] = "dfs_alias"
    fkagg["fkagg_definition"] = (
        "FDHG compiler restricted to FK/inverse-FK aggregation only; "
        "FD/AFD/ambiguity/residual/dmax2/ranker features excluded."
    )
    fkagg["paper_note"] = (
        "Materialized as DFS alias when DFS is the FK/inverse-FK aggregation-only block. "
        "Use task-specific feature inspection or actual fkagg_only run to justify this alias."
    )

    summary2 = summary.copy()
    if "source_method" not in summary2.columns:
        summary2["source_method"] = summary2["method"]
    if "method_display" not in summary2.columns:
        summary2["method_display"] = summary2["method"]
    if "source_type" not in summary2.columns:
        summary2["source_type"] = "original"
    if "fkagg_definition" not in summary2.columns:
        summary2["fkagg_definition"] = ""
    if "paper_note" not in summary2.columns:
        summary2["paper_note"] = ""

    return pd.concat([summary2, fkagg], ignore_index=True)


def add_external_rdblearn(summary, rdblearn_summary):
    if rdblearn_summary.empty:
        return summary

    r = rdblearn_summary.copy()
    if "method" not in r.columns and "method_name" in r.columns:
        r = r.rename(columns={"method_name": "method"})

    # Ensure columns align
    for c in summary.columns:
        if c not in r.columns:
            r[c] = None
    for c in r.columns:
        if c not in summary.columns:
            summary[c] = None

    r["method_display"] = "RDBLearn direct DFS+TabPFN"
    r["source_type"] = "external_rdblearn_direct"
    r["paper_note"] = (
        "External direct RDBLearn baseline; uses its own DFS generation/preprocessing pipeline."
    )

    return pd.concat([summary, r[summary.columns]], ignore_index=True)


def add_deltas(summary):
    out = summary.copy()

    for c in [
        "delta_roc_auc_vs_fkagg",
        "delta_ap_vs_fkagg",
        "delta_log_loss_vs_fkagg",
        "interpretation_vs_fkagg",
    ]:
        out[c] = None

    metric_cols = ["roc_auc_mean", "average_precision_mean", "log_loss_mean"]
    for c in metric_cols:
        if c not in out.columns:
            raise ValueError(f"Missing required metric summary column: {c}")

    group_keys = ["dataset", "task"]
    if "decoder" in out.columns:
        # Do not group by decoder for RDBLearn comparison if decoder naming differs.
        pass

    for (dataset, task), g in out.groupby(["dataset", "task"], dropna=False):
        base = g[g["method"].eq("fdhg_fkagg")]
        if base.empty:
            continue
        b = base.iloc[0]

        idxs = (out["dataset"].eq(dataset)) & (out["task"].eq(task))
        out.loc[idxs, "delta_roc_auc_vs_fkagg"] = out.loc[idxs, "roc_auc_mean"] - b["roc_auc_mean"]
        out.loc[idxs, "delta_ap_vs_fkagg"] = out.loc[idxs, "average_precision_mean"] - b["average_precision_mean"]
        out.loc[idxs, "delta_log_loss_vs_fkagg"] = out.loc[idxs, "log_loss_mean"] - b["log_loss_mean"]

        for idx in out[idxs].index:
            method = str(out.loc[idx, "method"])
            if method == "fdhg_fkagg":
                out.loc[idx, "interpretation_vs_fkagg"] = "Aggregation-only FDHG control."
            elif "fdhg_dmax1" in method:
                out.loc[idx, "interpretation_vs_fkagg"] = (
                    "Improvement over FDHG-FKAgg supports dependency/residual contribution; "
                    "if no improvement, dependency block is not helpful for this task."
                )
            elif method == "dfs":
                out.loc[idx, "interpretation_vs_fkagg"] = (
                    "DFS source row used to materialize FDHG-FKAgg when aggregation blocks coincide."
                )
            elif "RDBLearn" in method:
                out.loc[idx, "interpretation_vs_fkagg"] = (
                    "External DFS-style baseline; compare separately because preprocessing/features may differ."
                )
            else:
                out.loc[idx, "interpretation_vs_fkagg"] = ""

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_runs", default="results/intermediate_mvp_20260601/final_tables_normalized/paper_main_runs.csv")
    ap.add_argument("--main_summary", default="")
    ap.add_argument("--rdblearn_summary", default="baselines_repro/results/rdblearn_relstack_user_badge_summary.csv")
    ap.add_argument("--out", default="results/final_tables/generic_fkagg_internal_baseline_comparison.csv")
    args = ap.parse_args()

    if args.main_summary:
        main = load_optional_csv(args.main_summary)
        if main.empty:
            raise SystemExit(f"main_summary not found or empty: {args.main_summary}")
        main = normalize_main_rows(main)
        summary = main.copy()
    else:
        main = load_optional_csv(args.main_runs)
        if main.empty:
            raise SystemExit(f"main_runs not found or empty: {args.main_runs}")
        main = normalize_main_rows(main)
        summary = summarize_if_needed(main)

    summary = add_fkagg_alias(summary)

    rdblearn = load_optional_csv(args.rdblearn_summary)
    if not rdblearn.empty:
        rdblearn = normalize_main_rows(rdblearn)
        summary = add_external_rdblearn(summary, rdblearn)

    summary = add_deltas(summary)

    # Helpful feature-use flags
    summary["uses_fk_inverse_fk_aggregation"] = summary["method"].astype(str).isin(
        ["dfs", "fdhg_fkagg", "fdhg_dmax1", "fdhg_dmax1_full"]
    ) | summary["method"].astype(str).str.contains("fdhg", case=False, na=False)

    summary["uses_fd_afd_or_ambiguity_residual"] = summary["method"].astype(str).str.contains(
        "fdhg_dmax1|dmax2", case=False, na=False
    )

    summary["uses_dmax2_residual"] = summary["method"].astype(str).str.contains(
        "dmax2", case=False, na=False
    )

    summary["uses_fdhg_ranker"] = summary["method"].astype(str).str.contains(
        "supervised|topk", case=False, na=False
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    print(summary.to_string(index=False))
    print("Wrote:", out)


if __name__ == "__main__":
    main()
