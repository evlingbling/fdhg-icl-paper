#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd


ROOT = Path("results")
FINAL_DIR = ROOT / "final_tables"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

PAPER_DIR = (
    ROOT / "sweeps" / "relbench_v2_arxiv_paper_citation"
)
AUTHOR_DIR = (
    ROOT / "sweeps" / "relbench_v2_arxiv_author_category"
)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------
# 1. Load source tables
# ---------------------------------------------------------------------

paper_summary = pd.read_csv(
    require(PAPER_DIR / "summary.csv")
)

paper_all_runs = pd.read_csv(
    require(PAPER_DIR / "all_seed_metrics.csv")
)

author_summary = pd.read_csv(
    require(AUTHOR_DIR / "summary.csv")
)

author_all_runs = pd.read_csv(
    require(AUTHOR_DIR / "all_seed_metrics.csv")
)

author_gate = pd.read_csv(
    require(AUTHOR_DIR / "gate_status.csv")
)

cold_warm = pd.read_csv(
    require(AUTHOR_DIR / "cold_warm_seed41_metrics.csv")
)

cold_warm_gains = pd.read_csv(
    require(AUTHOR_DIR / "cold_warm_seed41_gains.csv")
)


# ---------------------------------------------------------------------
# 2. Build paper-citation task-level row
# ---------------------------------------------------------------------

paper_base = paper_summary[
    paper_summary["variant"] == "temporal_dfs_clean"
].iloc[0]

paper_fallback = paper_summary[
    paper_summary["variant"] == "fdhg_v2_fallback"
].iloc[0]

paper_row = {
    "dataset": "rel-arxiv",
    "task": "paper-citation",
    "task_type": "binary_classification",
    "role_in_paper": "real_world_negative_control",
    "base_variant": "temporal_dfs_clean",
    "selected_variant": "fdhg_v2_fallback",
    "gate_outcome": "FALLBACK",
    "selected_residual_count": 0,
    "seeds": int(paper_fallback["seeds"]),
    "base_n_features": int(paper_base["n_features"]),
    "selected_n_features": int(paper_fallback["n_features"]),
    "accuracy_base": paper_base["accuracy_mean"],
    "accuracy_selected": paper_fallback["accuracy_mean"],
    "accuracy_gain": (
        paper_fallback["accuracy_mean"]
        - paper_base["accuracy_mean"]
    ),
    "roc_auc_base": paper_base["roc_auc_mean"],
    "roc_auc_selected": paper_fallback["roc_auc_mean"],
    "roc_auc_gain": (
        paper_fallback["roc_auc_mean"]
        - paper_base["roc_auc_mean"]
    ),
    "average_precision_base": (
        paper_base["average_precision_mean"]
    ),
    "average_precision_selected": (
        paper_fallback["average_precision_mean"]
    ),
    "average_precision_gain": (
        paper_fallback["average_precision_mean"]
        - paper_base["average_precision_mean"]
    ),
    "log_loss_base": paper_base["log_loss_mean"],
    "log_loss_selected": paper_fallback["log_loss_mean"],
    "log_loss_reduction": (
        paper_base["log_loss_mean"]
        - paper_fallback["log_loss_mean"]
    ),
    "macro_f1_base": pd.NA,
    "macro_f1_selected": pd.NA,
    "macro_f1_gain": pd.NA,
    "mrr_base": pd.NA,
    "mrr_selected": pd.NA,
    "mrr_gain": pd.NA,
    "fallback_exact_match": True,
    "mechanism_summary": (
        "No applicable residual was selected; "
        "FDHG exactly preserved the temporal DFS representation."
    ),
}


# ---------------------------------------------------------------------
# 3. Build author-category task-level row
# ---------------------------------------------------------------------

author_base = author_summary[
    author_summary["variant"]
    == "dfs_without_category_residual"
].iloc[0]

author_selected = author_summary[
    author_summary["variant"]
    == "dfs_plus_category_residual"
].iloc[0]

author_residual_only = author_summary[
    author_summary["variant"]
    == "category_residual_only"
].iloc[0]

gate_row = author_gate.iloc[0]

author_row = {
    "dataset": "rel-arxiv",
    "task": "author-category",
    "task_type": "multiclass_classification",
    "role_in_paper": "strong_positive_mechanism_case",
    "base_variant": "dfs_without_category_residual",
    "selected_variant": "dfs_plus_category_residual",
    "gate_outcome": gate_row["gate_outcome"],
    "selected_residual_count": int(
        gate_row["selected_residual_count"]
    ),
    "seeds": int(author_selected["seeds"]),
    "base_n_features": int(author_base["n_features"]),
    "selected_n_features": int(
        author_selected["n_features"]
    ),
    "accuracy_base": author_base["accuracy_mean"],
    "accuracy_selected": author_selected["accuracy_mean"],
    "accuracy_gain": (
        author_selected["accuracy_mean"]
        - author_base["accuracy_mean"]
    ),
    "roc_auc_base": pd.NA,
    "roc_auc_selected": pd.NA,
    "roc_auc_gain": pd.NA,
    "average_precision_base": pd.NA,
    "average_precision_selected": pd.NA,
    "average_precision_gain": pd.NA,
    "log_loss_base": author_base["log_loss_mean"],
    "log_loss_selected": author_selected["log_loss_mean"],
    "log_loss_reduction": (
        author_base["log_loss_mean"]
        - author_selected["log_loss_mean"]
    ),
    "macro_f1_base": author_base["macro_f1_mean"],
    "macro_f1_selected": (
        author_selected["macro_f1_mean"]
    ),
    "macro_f1_gain": (
        author_selected["macro_f1_mean"]
        - author_base["macro_f1_mean"]
    ),
    "mrr_base": author_base["mrr_mean"],
    "mrr_selected": author_selected["mrr_mean"],
    "mrr_gain": (
        author_selected["mrr_mean"]
        - author_base["mrr_mean"]
    ),
    "fallback_exact_match": False,
    "mechanism_summary": (
        "Historical category state and ambiguity residuals "
        "produce gains only when publication history exists."
    ),
}


task_summary = pd.DataFrame(
    [paper_row, author_row]
)

task_summary_path = (
    FINAL_DIR / "rel_arxiv_extension_task_summary.csv"
)
task_summary.to_csv(task_summary_path, index=False)


# ---------------------------------------------------------------------
# 4. Build compact paper-facing table
# ---------------------------------------------------------------------

paper_table = pd.DataFrame(
    [
        {
            "Dataset": "rel-arxiv",
            "Task": "paper-citation",
            "Setting": "Negative control",
            "Base": "Temporal DFS",
            "FDHG": "Exact fallback",
            "Base metric": (
                f"AUROC {paper_base['roc_auc_mean']:.4f}"
                f" ± {paper_base['roc_auc_std']:.4f}"
            ),
            "FDHG metric": (
                f"AUROC {paper_fallback['roc_auc_mean']:.4f}"
                f" ± {paper_fallback['roc_auc_std']:.4f}"
            ),
            "Delta": (
                f"{paper_row['roc_auc_gain']:+.4f}"
            ),
            "Gate": "FALLBACK",
        },
        {
            "Dataset": "rel-arxiv",
            "Task": "author-category",
            "Setting": "Positive mechanism",
            "Base": "DFS",
            "FDHG": "DFS + 3 residuals",
            "Base metric": (
                f"Macro-F1 "
                f"{author_base['macro_f1_mean']:.4f}"
                f" ± {author_base['macro_f1_std']:.4f}"
            ),
            "FDHG metric": (
                f"Macro-F1 "
                f"{author_selected['macro_f1_mean']:.4f}"
                f" ± {author_selected['macro_f1_std']:.4f}"
            ),
            "Delta": (
                f"{author_row['macro_f1_gain']:+.4f}"
            ),
            "Gate": "SELECT",
        },
    ]
)

paper_table_path = (
    FINAL_DIR / "rel_arxiv_extension_paper_table.csv"
)
paper_table.to_csv(paper_table_path, index=False)


# ---------------------------------------------------------------------
# 5. Consolidate all seed-level runs
# ---------------------------------------------------------------------

paper_all_runs = paper_all_runs.copy()
paper_all_runs["evaluation_family"] = "binary"

author_all_runs = author_all_runs.copy()
author_all_runs["evaluation_family"] = "multiclass"

all_runs = pd.concat(
    [paper_all_runs, author_all_runs],
    ignore_index=True,
    sort=False,
)

all_runs_path = (
    FINAL_DIR / "rel_arxiv_extension_all_runs.csv"
)
all_runs.to_csv(all_runs_path, index=False)


# ---------------------------------------------------------------------
# 6. Save mechanism subgroup tables
# ---------------------------------------------------------------------

cold_warm_path = (
    FINAL_DIR / "rel_arxiv_author_category_cold_warm_metrics.csv"
)
cold_warm.to_csv(cold_warm_path, index=False)

cold_warm_gains_path = (
    FINAL_DIR / "rel_arxiv_author_category_cold_warm_gains.csv"
)
cold_warm_gains.to_csv(cold_warm_gains_path, index=False)


# ---------------------------------------------------------------------
# 7. Merge into existing final_all_runs as a non-destructive copy
# ---------------------------------------------------------------------

existing_all_runs_path = first_existing(
    [
        FINAL_DIR / "final_all_runs.csv",
        ROOT / "final_all_runs.csv",
    ]
)

merged_all_runs_path = (
    FINAL_DIR / "final_all_runs_with_rel_arxiv.csv"
)

if existing_all_runs_path is not None:
    existing = pd.read_csv(existing_all_runs_path)

    merged = pd.concat(
        [existing, all_runs],
        ignore_index=True,
        sort=False,
    )

    dedup_cols = [
        c for c in [
            "dataset",
            "task",
            "variant",
            "decoder",
            "seed",
            "split",
        ]
        if c in merged.columns
    ]

    if dedup_cols:
        merged = merged.drop_duplicates(
            subset=dedup_cols,
            keep="last",
        )

    merged.to_csv(
        merged_all_runs_path,
        index=False,
    )
else:
    all_runs.to_csv(
        merged_all_runs_path,
        index=False,
    )


# ---------------------------------------------------------------------
# 8. Machine-readable manifest
# ---------------------------------------------------------------------

manifest = {
    "dataset": "rel-arxiv",
    "tasks": {
        "paper-citation": {
            "gate_outcome": "FALLBACK",
            "selected_residual_count": 0,
            "fallback_exact_match": True,
            "primary_metric": "roc_auc",
            "primary_metric_mean": float(
                paper_fallback["roc_auc_mean"]
            ),
        },
        "author-category": {
            "gate_outcome": "SELECT",
            "selected_residual_count": 3,
            "primary_metric": "macro_f1",
            "base_primary_metric_mean": float(
                author_base["macro_f1_mean"]
            ),
            "selected_primary_metric_mean": float(
                author_selected["macro_f1_mean"]
            ),
            "primary_metric_gain": float(
                author_row["macro_f1_gain"]
            ),
            "warm_start_accuracy_gain_seed41": float(
                cold_warm_gains.loc[
                    cold_warm_gains["subgroup"]
                    == "warm_start",
                    "accuracy_gain",
                ].iloc[0]
            ),
            "cold_start_accuracy_gain_seed41": float(
                cold_warm_gains.loc[
                    cold_warm_gains["subgroup"]
                    == "cold_start",
                    "accuracy_gain",
                ].iloc[0]
            ),
        },
    },
}

manifest_path = (
    FINAL_DIR / "rel_arxiv_extension_manifest.json"
)
manifest_path.write_text(
    json.dumps(manifest, indent=2)
)


# ---------------------------------------------------------------------
# 9. Assertions
# ---------------------------------------------------------------------

assert paper_row["gate_outcome"] == "FALLBACK"
assert abs(paper_row["roc_auc_gain"]) < 1e-12
assert paper_row["fallback_exact_match"] is True

assert author_row["gate_outcome"] == "SELECT"
assert author_row["macro_f1_gain"] > 0.25
assert author_row["accuracy_gain"] > 0.25
assert author_row["log_loss_reduction"] > 0.8

warm_gain = cold_warm_gains[
    cold_warm_gains["subgroup"] == "warm_start"
].iloc[0]

cold_gain = cold_warm_gains[
    cold_warm_gains["subgroup"] == "cold_start"
].iloc[0]

assert warm_gain["accuracy_gain"] > 0.35
assert abs(cold_gain["accuracy_gain"]) < 1e-12


print("\n=== REL-ARXIV TASK SUMMARY ===")
print(task_summary.to_string(index=False))

print("\n=== PAPER TABLE ===")
print(paper_table.to_string(index=False))

print("\n[OK] rel-arxiv integration complete")
print("\nSaved:")
for path in [
    task_summary_path,
    paper_table_path,
    all_runs_path,
    cold_warm_path,
    cold_warm_gains_path,
    merged_all_runs_path,
    manifest_path,
]:
    print(path)
