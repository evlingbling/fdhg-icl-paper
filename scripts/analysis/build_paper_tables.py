from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path("results/final_tables")
ALL_RUNS_PATH = BASE / "final_all_runs.csv"
FAIL_PATH = BASE / "final_failure_log.csv"

OUT_MAIN_RUNS = BASE / "paper_main_runs.csv"
OUT_MAIN_SUMMARY = BASE / "paper_main_summary.csv"
OUT_STACK_DMAX2 = BASE / "paper_stack_dmax2_summary.csv"
OUT_ABLATION = BASE / "paper_ablation_summary.csv"
OUT_DIAGNOSTIC = BASE / "paper_diagnostic_summary.csv"
OUT_FAILURE = BASE / "paper_failure_log.csv"


METRICS = ["accuracy", "roc_auc", "average_precision", "log_loss"]
SUMMARY_KEYS = ["dataset", "task", "variant"]


def normalize_variant_from_path(row: pd.Series) -> str:
    """
    Paper-table-specific correction.

    The aggregate script already normalizes most variants, but some historical
    phase2 outputs have misleading names because the saved file name is always
    dfs_agg_*. For paper tables, use result_path/protocol evidence to recover
    the intended variant.
    """
    variant = str(row.get("variant", ""))
    path = str(row.get("result_path", ""))

    # rel-f1 FDHG rows from phase2 main were sometimes normalized as dfs
    # because the evaluator output filename starts with dfs_agg_.
    if (
        row.get("dataset") == "rel-f1"
        and row.get("task") == "driver-dnf"
        and "rel-f1_driver-dnf_fdhg" in path
        and "phase2_temporal" not in path
    ):
        return "fdhg_dmax1"

    # rel-amazon item-churn paired protocol rows were historically
    # stored with the broad fdhg_dmax1 variant label. Recover the actual
    # DFS/FDHG arm from the containing directory name.
    if (
        row.get("dataset") == "rel-amazon"
        and row.get("task") == "item-churn"
    ):
        if "_fdhg_full_seed" in path and "_dfs/" in path:
            return "dfs"
        if "_fdhg_full_seed" in path and "_naive/" in path:
            return "naive"
        if "_fdhg_full_seed" in path and "_fdhg/" in path:
            return "fdhg_dmax1"

    # rel-amazon user-churn FDHG fallback rows are stored under *_fdhg path
    # but have the same metrics/features as DFS.
    if (
        row.get("dataset") == "rel-amazon"
        and row.get("task") == "user-churn"
        and "rel-amazon_user-churn_fdhg" in path
    ):
        return "fdhg_dmax1"

    return variant


def clean_all_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only successful metric rows for paper summaries.
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"].copy()

    # Avoid duplicate rows from aggregate CSVs plus individual JSONs.
    # Prefer individual JSON rows over broad phase2_main_runs.csv rows when both exist.
    df["result_path_str"] = df["result_path"].astype(str)
    df["is_aggregate_csv_row"] = df["result_path_str"].str.endswith("phase2_main_runs.csv")

    # Correct a few historical variant labels for paper tables.
    df["variant"] = df.apply(normalize_variant_from_path, axis=1)

    return df


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for keys, g in runs.groupby(SUMMARY_KEYS, dropna=False):
        dataset, task, variant = keys
        row = {
            "dataset": dataset,
            "task": task,
            "variant": variant,
            "n_runs": int(len(g)),
            "seeds": ",".join(str(int(s)) for s in sorted(g["seed"].dropna().unique())),
        }

        if "n_features" in g.columns:
            row["n_features_mean"] = float(g["n_features"].mean())
            row["n_features_std"] = float(g["n_features"].std(ddof=0)) if len(g) > 1 else 0.0

        for m in METRICS:
            if m in g.columns:
                vals = pd.to_numeric(g[m], errors="coerce").dropna()
                if len(vals):
                    row[f"{m}_mean"] = float(vals.mean())
                    row[f"{m}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
                else:
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_std"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["dataset", "task", "variant"]).reset_index(drop=True)
    return out


def select_non_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer individual run JSONs. Drop duplicate aggregate CSV rows."""
    return df[~df["is_aggregate_csv_row"]].copy()


def build_paper_main_runs(df: pd.DataFrame) -> pd.DataFrame:
    d = select_non_aggregate(df)

    parts = []

    # rel-stack/user-badge canonical main:
    # target_only and naive from week1/phase2 individual rows,
    # DFS and FDHG from regenerated dmax1 extension_b rows.
    stack = d[(d.dataset == "rel-stack") & (d.task == "user-badge")].copy()

    parts.append(stack[(stack.variant == "target_only") & stack.result_path_str.str.contains("results/week1/")])
    parts.append(stack[(stack.variant == "naive") & stack.result_path_str.str.contains("phase2_main_missing_relstack_seed42_44|phase2_relbench_new_datasets")])
    parts.append(stack[(stack.variant == "dfs") & stack.result_path_str.str.contains("regen_dmax1_dfs_alone")])
    parts.append(stack[(stack.variant == "fdhg_dmax1") & stack.result_path_str.str.contains("regen_dmax1_fdhg_full_alone")])

    # rel-amazon/item-churn canonical main:
    # target_only week1, naive/dfs/fdhg from phase2 individual protocol.
    item = d[(d.dataset == "rel-amazon") & (d.task == "item-churn")].copy()

    parts.append(item[(item.variant == "target_only") & item.result_path_str.str.contains("results/week1/")])
    # Use the paired four-seed naive arm from the same phase2 protocol
    # as DFS and FDHG. Exclude the older standalone seed-41 naive run.
    parts.append(
        item[
            (item.variant == "naive")
            & item.result_path_str.str.contains(
                "_fdhg_full_seed.*_naive",
                regex=True,
            )
        ]
    )
    # For DFS, canonical n_features=24 phase2 DFS, not fdhg_heuristic n_features=32.
    parts.append(item[(item.variant == "dfs") & (item.n_features == 24)])
    # For FDHG, canonical n_features=32 true FDHG path.
    parts.append(item[(item.variant == "fdhg_dmax1") & (item.n_features == 32) & item.result_path_str.str.contains("_fdhg_full_seed.*_fdhg|rel-amazon_item-churn_fdhg/")])

    # rel-amazon/user-churn canonical fallback:
    # FDHG is identical to DFS and appears under fdhg path after correction.
    user = d[(d.dataset == "rel-amazon") & (d.task == "user-churn")].copy()
    parts.append(user[(user.variant == "target_only") & user.result_path_str.str.contains("results/week1/")])
    parts.append(user[(user.variant == "naive")])
    parts.append(user[(user.variant == "dfs") & user.result_path_str.str.contains("_dfs/")])
    parts.append(user[(user.variant == "fdhg_dmax1") & user.result_path_str.str.contains("_fdhg/")])

    # rel-f1/driver-dnf canonical diagnostic main:
    # target_only/naive/dfs/fdhg_dmax1 only, temporal variants go diagnostic.
    f1 = d[(d.dataset == "rel-f1") & (d.task == "driver-dnf")].copy()
    parts.append(f1[(f1.variant == "target_only") & f1.result_path_str.str.contains("results/week1/")])
    parts.append(f1[(f1.variant == "naive") & f1.result_path_str.str.contains("rel-f1_driver-dnf_naive|phase2_relbench_new_datasets")])
    parts.append(f1[(f1.variant == "dfs") & (f1.n_features == 15) & f1.result_path_str.str.contains("_dfs/|phase2_relbench_new_datasets")])
    parts.append(f1[(f1.variant == "fdhg_dmax1") & (f1.n_features == 23)])

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Drop exact duplicate rows caused by CSV+JSON duplication.
    dedup_cols = ["dataset", "task", "variant", "seed", "n_features", "roc_auc", "average_precision", "log_loss"]
    dedup_cols = [c for c in dedup_cols if c in out.columns]
    out = out.drop_duplicates(subset=dedup_cols).copy()

    return out


def build_stack_dmax2_runs(df: pd.DataFrame) -> pd.DataFrame:
    d = select_non_aggregate(df)
    stack = d[(d.dataset == "rel-stack") & (d.task == "user-badge")].copy()

    keep = [
        "dfs",
        "fdhg_dmax1",
        "dmax2_only_topk16",
        "dmax2_only_random16",
        "dmax2_only_supervised_auc_topk16",
        "dmax2_only_supervised_ap_topk16",
        "fdhg_dmax1_plus_dmax2_topk16",
        "fdhg_dmax1_plus_dmax2_random16",
        "fdhg_dmax1_plus_dmax2_supervised_auc_topk16",
        "fdhg_dmax1_plus_dmax2_supervised_ap_topk16",
    ]

    out = stack[stack.variant.isin(keep)].copy()

    # Canonical DFS/FDHG dmax1 from regenerated extension_b.
    mask_dfs_bad = (out.variant == "dfs") & ~out.result_path_str.str.contains("regen_dmax1_dfs_alone")
    mask_fdhg_bad = (out.variant == "fdhg_dmax1") & ~out.result_path_str.str.contains("regen_dmax1_fdhg_full_alone")
    out = out[~(mask_dfs_bad | mask_fdhg_bad)].copy()

    return out


def summarize_stack_dmax2_evidence(runs: pd.DataFrame) -> pd.DataFrame:
    """Build the portable dmax2 evidence table from final_all_runs.csv."""
    keep = [
        "dmax2_only_topk16",
        "dmax2_only_random16",
        "fdhg_dmax1_plus_dmax2_topk16",
        "fdhg_dmax1_plus_dmax2_random16",
    ]

    runs = runs[runs["variant"].isin(keep)].copy()

    if runs.empty:
        raise AssertionError(
            "No canonical RelStack User-Badge dmax2 rows were found."
        )

    rows = []

    for variant, group in runs.groupby("variant", sort=False):
        group = group.sort_values("seed")

        # Guard against duplicated aggregate/JSON records.
        group = group.drop_duplicates(
            subset=[
                "variant",
                "seed",
                "n_features",
                "accuracy",
                "roc_auc",
                "average_precision",
                "log_loss",
            ]
        )

        seeds = sorted(
            group["seed"].dropna().astype(int).unique().tolist()
        )
        n_runs = len(seeds)

        expected_seeds = (
            [41]
            if variant in {
                "dmax2_only_topk16",
                "dmax2_only_random16",
            }
            else [41, 42, 43, 44]
        )

        if seeds != expected_seeds:
            raise AssertionError(
                f"{variant} seeds={seeds}; "
                f"expected {expected_seeds}."
            )

        row = {
            "dataset": "rel-stack",
            "task": "user-badge",
            "variant": variant,
            "n_runs": n_runs,
            "seeds": ",".join(map(str, seeds)),
            "n_features_mean": float(group["n_features"].mean()),
            "evidence_level": (
                "multiseed_confirmatory"
                if n_runs == 4
                else "single_seed_exploratory"
            ),
            "exact_rerun_possible": n_runs == 4,
        }

        for metric in METRICS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            row[f"{metric}_mean"] = (
                float(values.mean())
                if len(values)
                else np.nan
            )
            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else np.nan
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    order = [
        "dmax2_only_topk16",
        "dmax2_only_random16",
        "fdhg_dmax1_plus_dmax2_topk16",
        "fdhg_dmax1_plus_dmax2_random16",
    ]
    order_map = {
        variant: index
        for index, variant in enumerate(order)
    }

    out["_order"] = out["variant"].map(order_map)

    return (
        out.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

def build_ablation_runs(df: pd.DataFrame) -> pd.DataFrame:
    d = select_non_aggregate(df)

    keep = [
        "dfs",
        "fdhg_dmax1",
        "fdhg_dmax1_no_ambiguity",
        "fdhg_dmax1_random_same_budget",
        "fdhg_dmax1_shuffle_ambiguity",
    ]

    parts = []

    # rel-stack ablation: use phase2_relbench ablation paths for random/shuffle/no_ambiguity,
    # and regenerated extension_b for clean dfs/fdhg full.
    stack = d[(d.dataset == "rel-stack") & (d.task == "user-badge")].copy()
    parts.append(stack[(stack.variant == "dfs") & stack.result_path_str.str.contains("regen_dmax1_dfs_alone")])
    parts.append(stack[(stack.variant == "fdhg_dmax1") & stack.result_path_str.str.contains("regen_dmax1_fdhg_full_alone")])
    parts.append(stack[(stack.variant.isin(["fdhg_dmax1_random_same_budget", "fdhg_dmax1_shuffle_ambiguity"]))])
    # no_ambiguity can be noisy/unsafe; include for appendix only if desired.
    parts.append(stack[(stack.variant == "fdhg_dmax1_no_ambiguity") & (stack.n_features == 16)])

    # rel-amazon item-churn ablation.
    item = d[(d.dataset == "rel-amazon") & (d.task == "item-churn")].copy()
    parts.append(item[(item.variant == "dfs") & (item.n_features == 24)])
    parts.append(item[(item.variant == "fdhg_dmax1") & (item.n_features == 32)])
    parts.append(item[(item.variant == "fdhg_dmax1_no_ambiguity") & (item.n_features == 24)])
    parts.append(item[(item.variant.isin(["fdhg_dmax1_random_same_budget", "fdhg_dmax1_shuffle_ambiguity"]))])

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    dedup_cols = ["dataset", "task", "variant", "seed", "n_features", "roc_auc", "average_precision", "log_loss"]
    dedup_cols = [c for c in dedup_cols if c in out.columns]
    out = out.drop_duplicates(subset=dedup_cols).copy()

    return out


def build_diagnostic_runs(df: pd.DataFrame, fail: pd.DataFrame | None) -> pd.DataFrame:
    d = select_non_aggregate(df)
    parts = []

    # F1 temporal diagnostic.
    f1 = d[(d.dataset == "rel-f1") & (d.task == "driver-dnf")].copy()
    temporal_keep = [
        "target_only", "naive", "dfs", "fdhg_dmax1",
        "last_only", "dfs_plus_last", "fdhg_plus_last",
        "temporal_full", "trend_only",
    ]
    tmp = f1[f1.variant.isin(temporal_keep)].copy()
    # Keep canonical main rows for dfs/fdhg in diagnostic too.
    parts.append(tmp)

    # Fallback cases from main runs will be checked after summary.
    # Add successful target-only rel-hm/user-churn for context if present.
    hm = d[(d.dataset == "rel-hm") & (d.task == "user-churn")].copy()
    if len(hm):
        parts.append(hm)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    dedup_cols = ["dataset", "task", "variant", "seed", "n_features", "roc_auc", "average_precision", "log_loss"]
    dedup_cols = [c for c in dedup_cols if c in out.columns]
    out = out.drop_duplicates(subset=dedup_cols).copy()

    return out


def add_manual_failures(fail: pd.DataFrame) -> pd.DataFrame:
    fail = fail.copy()

    manual = {
        "dataset": "rel-hm",
        "task": "user-churn",
        "variant": "fdhg_dmax1",
        "seed": 41,
        "model": "tabpfn",
        "status": "failed",
        "failure_reason": "TabPFN runtime failure / SIGSEGV after feature generation succeeded",
        "feature_path": "feature_generation_succeeded; exact artifact path not recorded in final aggregate",
        "result_path": "manual_failure_log",
    }

    # Add only if absent.
    exists = (
        (fail.get("dataset", pd.Series(dtype=str)).astype(str) == manual["dataset"])
        & (fail.get("task", pd.Series(dtype=str)).astype(str) == manual["task"])
        & (fail.get("variant", pd.Series(dtype=str)).astype(str) == manual["variant"])
    ).any()

    if not exists:
        fail = pd.concat([fail, pd.DataFrame([manual])], ignore_index=True)

    return fail


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ALL_RUNS_PATH)
    fail = pd.read_csv(FAIL_PATH) if FAIL_PATH.exists() else pd.DataFrame()

    clean = clean_all_runs(df)

    main_runs = build_paper_main_runs(clean)
    main_summary = summarize(main_runs)

    dmax2_runs = build_stack_dmax2_runs(clean)
    dmax2_summary = summarize_stack_dmax2_evidence(dmax2_runs)

    ablation_runs = build_ablation_runs(clean)
    ablation_summary = summarize(ablation_runs)

    diagnostic_runs = build_diagnostic_runs(clean, fail)
    diagnostic_summary = summarize(diagnostic_runs)

    paper_failure = add_manual_failures(fail)

    # Canonical item-churn DFS and FDHG summaries must use the same
    # four validation seeds.
    item_main = main_runs[
        main_runs["dataset"].eq("rel-amazon")
        & main_runs["task"].eq("item-churn")
        & main_runs["variant"].isin(["dfs", "fdhg_dmax1"])
    ]

    for variant in ("dfs", "fdhg_dmax1"):
        variant_seeds = set(
            item_main.loc[
                item_main["variant"].eq(variant),
                "seed",
            ]
            .dropna()
            .astype(int)
        )

        if variant_seeds != {41, 42, 43, 44}:
            raise AssertionError(
                "rel-amazon/item-churn "
                f"{variant} seeds={sorted(variant_seeds)}; "
                "expected [41, 42, 43, 44]."
            )

    main_runs.to_csv(OUT_MAIN_RUNS, index=False)
    main_summary.to_csv(OUT_MAIN_SUMMARY, index=False)
    dmax2_summary.to_csv(OUT_STACK_DMAX2, index=False)
    ablation_summary.to_csv(OUT_ABLATION, index=False)
    diagnostic_summary.to_csv(OUT_DIAGNOSTIC, index=False)
    paper_failure.to_csv(OUT_FAILURE, index=False)

    print("Wrote:")
    for p in [
        OUT_MAIN_RUNS,
        OUT_MAIN_SUMMARY,
        OUT_STACK_DMAX2,
        OUT_ABLATION,
        OUT_DIAGNOSTIC,
        OUT_FAILURE,
    ]:
        print(f"  {p} ({p.stat().st_size} bytes)")

    print("\nPaper main summary:")
    show_cols = [
        "dataset", "task", "variant", "n_runs", "seeds",
        "n_features_mean", "roc_auc_mean", "average_precision_mean", "log_loss_mean",
    ]
    show_cols = [c for c in show_cols if c in main_summary.columns]
    print(main_summary[show_cols].to_string(index=False))

    print("\nStack dmax2 summary:")
    print(dmax2_summary[show_cols].to_string(index=False))

    print("\nAblation summary:")
    print(ablation_summary[show_cols].to_string(index=False))

    print("\nPaper failure log:")
    print(paper_failure.to_string(index=False))


if __name__ == "__main__":
    main()
