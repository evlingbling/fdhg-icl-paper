import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("results/final_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ratebeer_path = Path(
    "results/sweeps/relbench_v2_ratebeer_gate_summary/"
    "ratebeer_fdhg_v2_gate_compact_summary_all_current.csv"
)

salt_path = Path(
    "results/sweeps/relbench_v2_salt_gate_summary/"
    "salt_8task_fdhg_v2_final_compact_summary.csv"
)

arxiv_path = Path(
    "results/final_tables/"
    "rel_arxiv_extension_task_summary.csv"
)

RUNTIME_GATE_DECISIONS = [
    Path(
        "results/rel-ratebeer_beer-churn_tabpfn/"
        "gate_decision.json"
    ),
    Path(
        "results/rel-ratebeer_brewer-dormant_tabpfn/"
        "gate_decision.json"
    ),
]


def norm_gate(x):
    s = str(x).upper()
    if "SELECT" in s and "REJECT" not in s:
        return "SELECT"
    if "FALLBACK" in s or "REJECT" in s:
        return "FALLBACK"
    return s


def apply_runtime_gate_overrides(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Override legacy gate rows with reproducible runtime decisions."""
    frame = frame.copy()

    decision_paths = [
        path
        for path in RUNTIME_GATE_DECISIONS
        if path.exists()
    ]

    for decision_path in decision_paths:
        metrics_path = decision_path.with_name(
            "gated_metrics.csv"
        )

        if not metrics_path.exists():
            print(
                "[runtime-gate] Skipping decision without metrics:",
                decision_path,
            )
            continue

        decision = json.loads(
            decision_path.read_text(encoding="utf-8")
        )
        gated = pd.read_csv(metrics_path)

        if gated.empty:
            raise ValueError(
                f"Runtime gated metrics are empty: {metrics_path}"
            )

        required = {
            "base_metrics_path",
            "candidate_metrics_path",
            "base_score",
            "candidate_score",
            "selected_score",
            "base_n_features",
            "candidate_n_features",
        }
        missing = sorted(required - set(gated.columns))

        if missing:
            raise ValueError(
                f"Missing runtime gate columns in {metrics_path}: "
                f"{missing}"
            )

        metadata_rows = []

        for raw_path in gated["base_metrics_path"]:
            metric_path = Path(str(raw_path))

            if not metric_path.exists():
                raise FileNotFoundError(
                    f"Missing base metrics file: {metric_path}"
                )

            metric_frame = pd.read_csv(metric_path)

            if len(metric_frame) != 1:
                raise ValueError(
                    f"Expected one row in {metric_path}, "
                    f"found {len(metric_frame)}."
                )

            metadata_rows.append(
                metric_frame.iloc[0].to_dict()
            )

        datasets = {
            str(row["dataset"])
            for row in metadata_rows
        }
        tasks = {
            str(row["task"])
            for row in metadata_rows
        }

        if len(datasets) != 1 or len(tasks) != 1:
            raise ValueError(
                "Runtime gate metadata disagrees across seeds: "
                f"datasets={sorted(datasets)}, tasks={sorted(tasks)}"
            )

        dataset = next(iter(datasets))
        task = next(iter(tasks))

        mask = (
            frame["dataset"].eq(dataset)
            & frame["task"].eq(task)
        )

        if int(mask.sum()) != 1:
            print(
                "[runtime-gate] No unique paper-table row for:",
                dataset,
                task,
            )
            continue

        gate_outcome = str(decision["gate_outcome"])
        selected = gate_outcome == "SELECT"

        base_score = float(gated["base_score"].mean())
        candidate_score = float(
            gated["candidate_score"].mean()
        )
        gated_score = float(gated["selected_score"].mean())
        candidate_mean_improvement = float(
            gated["improvement"].mean()
        )
        gated_gain = (
            candidate_mean_improvement
            if selected
            else 0.0
        )

        base_n_features = float(
            gated["base_n_features"].mean()
        )
        candidate_n_features = float(
            gated["candidate_n_features"].mean()
        )

        frame.loc[mask, "gate_outcome"] = gate_outcome
        frame.loc[mask, "gate_reason"] = (
            "runtime_all_seed_improvement"
            if selected
            else "runtime_seed_inconsistency_fallback"
        )
        frame.loc[mask, "primary_metric"] = decision["metric"]
        frame.loc[mask, "base_variant"] = decision[
            "base_variant"
        ]
        frame.loc[mask, "candidate_variant"] = decision[
            "candidate_variant"
        ]
        frame.loc[mask, "selected_variant"] = decision[
            "selected_variant"
        ]
        frame.loc[mask, "base_n_features"] = base_n_features
        frame.loc[
            mask,
            "candidate_n_features",
        ] = candidate_n_features
        frame.loc[
            mask,
            "selected_residual_count",
        ] = (
            candidate_n_features - base_n_features
            if selected
            else 0
        )
        frame.loc[mask, "base_score"] = base_score
        frame.loc[mask, "candidate_score"] = candidate_score
        frame.loc[mask, "gated_score"] = gated_score
        frame.loc[mask, "gain"] = gated_gain
        frame.loc[
            mask,
            "fallback_exact_match",
        ] = not selected

        metric = str(decision["metric"])

        if metric == "log_loss":
            frame.loc[mask, "log_loss_base"] = base_score
            frame.loc[
                mask,
                "log_loss_candidate",
            ] = candidate_score
            frame.loc[mask, "log_loss_gated"] = gated_score
            frame.loc[
                mask,
                "log_loss_reduction",
            ] = gated_gain

        elif metric == "accuracy":
            frame.loc[mask, "accuracy_base"] = base_score
            frame.loc[
                mask,
                "accuracy_candidate",
            ] = candidate_score
            frame.loc[mask, "accuracy_gated"] = gated_score
            frame.loc[mask, "accuracy_gain"] = gated_gain

        elif metric == "rmse":
            frame.loc[mask, "rmse_base"] = base_score
            frame.loc[
                mask,
                "rmse_candidate",
            ] = candidate_score
            frame.loc[mask, "rmse_gated"] = gated_score
            frame.loc[
                mask,
                "rmse_reduction",
            ] = gated_gain

        elif metric == "mrr":
            frame.loc[mask, "mrr_base"] = base_score
            frame.loc[
                mask,
                "mrr_candidate",
            ] = candidate_score
            frame.loc[mask, "mrr_gain"] = gated_gain

        print(
            "[runtime-gate] Applied:",
            dataset,
            task,
            gate_outcome,
            (
                f"candidate_mean_improvement="
                f"{candidate_mean_improvement:.12g}"
            ),
            f"gated_gain={gated_gain:.12g}",
        )

    return frame


rows = []

# ------------------------------------------------------------------
# RateBeer
# ------------------------------------------------------------------
rb = pd.read_csv(ratebeer_path)

for _, r in rb.iterrows():
    task = r["task"]
    task_family = r.get("task_family", "")
    gate = norm_gate(r.get("gate_decision", ""))

    row = {
        "dataset": r["dataset"],
        "task": task,
        "task_family": task_family,
        "decoder": r.get("decoder", ""),
        "gate_outcome": gate,
        "gate_reason": "lower_validation_rmse"
        if "regression" in str(task_family).lower()
        else "lower_validation_log_loss",
        "base_variant": "dfs",
        "candidate_variant": "fdhg_residual",
        "selected_variant": "fdhg_residual"
        if gate == "SELECT"
        else "dfs_fallback",
        "base_n_features": r.get("n_features_dfs", np.nan),
        "candidate_n_features": r.get("n_features_fdhg", np.nan),
        "selected_residual_count": (
            r.get("n_features_fdhg", np.nan)
            - r.get("n_features_dfs", np.nan)
            if gate == "SELECT"
            else 0
        ),
        "fallback_exact_match": gate == "FALLBACK",
        "role_in_paper": "",
    }

    if "regression" in str(task_family).lower():
        row.update({
            "primary_metric": "rmse",
            "base_score": r.get("dfs_auroc", np.nan),
            "candidate_score": r.get("fdhg_auroc", np.nan),
            "gated_score": r.get("gated_auroc", np.nan),
            "gain": (
                r.get("dfs_auroc", np.nan)
                - r.get("gated_auroc", np.nan)
            ),
            "rmse_base": r.get("dfs_auroc", np.nan),
            "rmse_candidate": r.get("fdhg_auroc", np.nan),
            "rmse_gated": r.get("gated_auroc", np.nan),
            "rmse_reduction": (
                r.get("dfs_auroc", np.nan)
                - r.get("gated_auroc", np.nan)
            ),
            "mae_base": r.get("dfs_ap", np.nan),
            "mae_candidate": r.get("fdhg_ap", np.nan),
            "mae_gated": r.get("gated_ap", np.nan),
            "r2_base": r.get("dfs_log_loss", np.nan),
            "r2_candidate": r.get("fdhg_log_loss", np.nan),
            "r2_gated": r.get("gated_log_loss", np.nan),
        })

    else:
        row.update({
            "primary_metric": "log_loss",
            "base_score": r.get("dfs_log_loss", np.nan),
            "candidate_score": r.get("fdhg_log_loss", np.nan),
            "gated_score": r.get("gated_log_loss", np.nan),
            "gain": (
                r.get("dfs_log_loss", np.nan)
                - r.get("gated_log_loss", np.nan)
            ),
            "auroc_base": r.get("dfs_auroc", np.nan),
            "auroc_candidate": r.get("fdhg_auroc", np.nan),
            "auroc_gated": r.get("gated_auroc", np.nan),
            "auroc_gain": (
                r.get("gated_auroc", np.nan)
                - r.get("dfs_auroc", np.nan)
            ),
            "ap_base": r.get("dfs_ap", np.nan),
            "ap_candidate": r.get("fdhg_ap", np.nan),
            "ap_gated": r.get("gated_ap", np.nan),
            "ap_gain": (
                r.get("gated_ap", np.nan)
                - r.get("dfs_ap", np.nan)
            ),
            "log_loss_base": r.get("dfs_log_loss", np.nan),
            "log_loss_candidate": r.get("fdhg_log_loss", np.nan),
            "log_loss_gated": r.get("gated_log_loss", np.nan),
            "log_loss_reduction": (
                r.get("dfs_log_loss", np.nan)
                - r.get("gated_log_loss", np.nan)
            ),
        })

    if task == "beer-churn":
        row["role_in_paper"] = "weak_positive_case"
    elif task == "user-count":
        row["role_in_paper"] = "regression_positive_case"
    elif task == "beer_ratings-total_score_enriched":
        row["role_in_paper"] = "regression_fallback_case"
    elif gate == "FALLBACK":
        row["role_in_paper"] = "fallback_case"

    rows.append(row)


# ------------------------------------------------------------------
# SALT
# ------------------------------------------------------------------
salt = pd.read_csv(salt_path)

for _, r in salt.iterrows():
    gate = norm_gate(r.get("gate_outcome", r.get("gate_decision", "")))

    primary_metric = "accuracy"
    reason = r.get("gate_reason", "")

    if str(reason) == "accuracy_tie_higher_mrr":
        primary_metric = "accuracy_then_mrr"

    row = {
        "dataset": r["dataset"],
        "task": r["task"],
        "task_family": r.get("task_family", "relbench_v2_salt_multiclass"),
        "decoder": r.get("decoder", ""),
        "gate_outcome": gate,
        "gate_reason": reason,
        "base_variant": "dfs",
        "candidate_variant": "fdhg_residual",
        "selected_variant": "fdhg_residual"
        if gate == "SELECT"
        else "dfs_fallback",
        "base_n_features": r.get("n_features_dfs", np.nan),
        "candidate_n_features": r.get("n_features_fdhg", np.nan),
        "selected_residual_count": (
            r.get("n_features_fdhg", np.nan)
            - r.get("n_features_dfs", np.nan)
            if gate == "SELECT"
            and pd.notna(r.get("n_features_dfs", np.nan))
            and pd.notna(r.get("n_features_fdhg", np.nan))
            else np.nan
        ),
        "primary_metric": primary_metric,
        "base_score": r.get("dfs_accuracy", np.nan),
        "candidate_score": r.get("fdhg_accuracy", np.nan),
        "gated_score": r.get("gated_accuracy", np.nan),
        "gain": r.get("delta_accuracy", np.nan),
        "accuracy_base": r.get("dfs_accuracy", np.nan),
        "accuracy_candidate": r.get("fdhg_accuracy", np.nan),
        "accuracy_gated": r.get("gated_accuracy", np.nan),
        "accuracy_gain": r.get("delta_accuracy", np.nan),
        "macro_f1_base": r.get("dfs_macro_f1", np.nan),
        "macro_f1_candidate": r.get("fdhg_macro_f1", np.nan),
        "macro_f1_gated": r.get("gated_macro_f1", np.nan),
        "macro_f1_gain": r.get("delta_macro_f1", np.nan),
        "weighted_f1_base": r.get("dfs_weighted_f1", np.nan),
        "weighted_f1_candidate": r.get("fdhg_weighted_f1", np.nan),
        "weighted_f1_gated": r.get("gated_weighted_f1", np.nan),
        "weighted_f1_gain": r.get("delta_weighted_f1", np.nan),
        "log_loss_base": r.get("dfs_log_loss", np.nan),
        "log_loss_candidate": r.get("fdhg_log_loss", np.nan),
        "log_loss_gated": r.get("gated_log_loss", np.nan),
        "log_loss_reduction": (
            r.get("dfs_log_loss", np.nan)
            - r.get("gated_log_loss", np.nan)
        ),
        "mrr_base": r.get("dfs_mrr", np.nan),
        "mrr_candidate": r.get("fdhg_mrr", np.nan),
        "mrr_gain": r.get("delta_mrr", np.nan),
        "fallback_exact_match": gate == "FALLBACK",
        "role_in_paper": (
            "strong_positive_case"
            if r["task"] == "item-shippoint"
            else "fallback_case"
            if gate == "FALLBACK"
            else "multiclass_positive_case"
        ),
    }

    rows.append(row)


# ------------------------------------------------------------------
# Arxiv
# ------------------------------------------------------------------
arxiv = pd.read_csv(arxiv_path)

for _, r in arxiv.iterrows():
    gate = norm_gate(r["gate_outcome"])

    if r["task_type"] == "binary_classification":
        primary_metric = "log_loss"
        base_score = r.get("log_loss_base", np.nan)
        gated_score = r.get("log_loss_selected", np.nan)
        gain = r.get("log_loss_reduction", np.nan)
    else:
        primary_metric = "accuracy"
        base_score = r.get("accuracy_base", np.nan)
        gated_score = r.get("accuracy_selected", np.nan)
        gain = r.get("accuracy_gain", np.nan)

    row = {
        "dataset": r["dataset"],
        "task": r["task"],
        "task_family": r["task_type"],
        "decoder": "tabpfn",
        "gate_outcome": gate,
        "gate_reason": (
            "no_applicable_residual"
            if gate == "FALLBACK"
            else "positive_validation_gain"
        ),
        "base_variant": r["base_variant"],
        "candidate_variant": r["selected_variant"],
        "selected_variant": r["selected_variant"],
        "base_n_features": r.get("base_n_features", np.nan),
        "candidate_n_features": r.get("selected_n_features", np.nan),
        "selected_residual_count": r.get(
            "selected_residual_count",
            np.nan,
        ),
        "primary_metric": primary_metric,
        "base_score": base_score,
        "candidate_score": gated_score,
        "gated_score": gated_score,
        "gain": gain,
        "accuracy_base": r.get("accuracy_base", np.nan),
        "accuracy_gated": r.get("accuracy_selected", np.nan),
        "accuracy_gain": r.get("accuracy_gain", np.nan),
        "auroc_base": r.get("roc_auc_base", np.nan),
        "auroc_gated": r.get("roc_auc_selected", np.nan),
        "auroc_gain": r.get("roc_auc_gain", np.nan),
        "ap_base": r.get("average_precision_base", np.nan),
        "ap_gated": r.get("average_precision_selected", np.nan),
        "ap_gain": r.get("average_precision_gain", np.nan),
        "log_loss_base": r.get("log_loss_base", np.nan),
        "log_loss_gated": r.get("log_loss_selected", np.nan),
        "log_loss_reduction": r.get(
            "log_loss_reduction",
            np.nan,
        ),
        "macro_f1_base": r.get("macro_f1_base", np.nan),
        "macro_f1_gated": r.get("macro_f1_selected", np.nan),
        "macro_f1_gain": r.get("macro_f1_gain", np.nan),
        "mrr_base": r.get("mrr_base", np.nan),
        "mrr_candidate": r.get("mrr_selected", np.nan),
        "mrr_gain": r.get("mrr_gain", np.nan),
        "fallback_exact_match": r.get(
            "fallback_exact_match",
            gate == "FALLBACK",
        ),
        "role_in_paper": r.get("role_in_paper", ""),
    }

    rows.append(row)


# ------------------------------------------------------------------
# Final unified table
# ------------------------------------------------------------------
out = pd.DataFrame(rows)
out = apply_runtime_gate_overrides(out)

preferred_order = [
    "dataset",
    "task",
    "task_family",
    "decoder",
    "gate_outcome",
    "gate_reason",
    "primary_metric",
    "base_variant",
    "candidate_variant",
    "selected_variant",
    "base_n_features",
    "candidate_n_features",
    "selected_residual_count",
    "base_score",
    "candidate_score",
    "gated_score",
    "gain",
    "accuracy_base",
    "accuracy_candidate",
    "accuracy_gated",
    "accuracy_gain",
    "auroc_base",
    "auroc_candidate",
    "auroc_gated",
    "auroc_gain",
    "ap_base",
    "ap_candidate",
    "ap_gated",
    "ap_gain",
    "macro_f1_base",
    "macro_f1_candidate",
    "macro_f1_gated",
    "macro_f1_gain",
    "weighted_f1_base",
    "weighted_f1_candidate",
    "weighted_f1_gated",
    "weighted_f1_gain",
    "mrr_base",
    "mrr_candidate",
    "mrr_gain",
    "log_loss_base",
    "log_loss_candidate",
    "log_loss_gated",
    "log_loss_reduction",
    "rmse_base",
    "rmse_candidate",
    "rmse_gated",
    "rmse_reduction",
    "mae_base",
    "mae_candidate",
    "mae_gated",
    "r2_base",
    "r2_candidate",
    "r2_gated",
    "fallback_exact_match",
    "role_in_paper",
]

for c in preferred_order:
    if c not in out.columns:
        out[c] = np.nan

out = out[preferred_order].sort_values(
    ["dataset", "task"]
).reset_index(drop=True)

full_path = OUT_DIR / "relbench_v2_fdhg_gate_unified_summary.csv"
out.to_csv(full_path, index=False)


# Compact paper table
paper_cols = [
    "dataset",
    "task",
    "task_family",
    "gate_outcome",
    "primary_metric",
    "base_score",
    "candidate_score",
    "gated_score",
    "gain",
    "selected_residual_count",
    "fallback_exact_match",
    "role_in_paper",
]

paper = out[paper_cols].copy()

paper_path = OUT_DIR / "relbench_v2_fdhg_gate_paper_table.csv"
paper.to_csv(paper_path, index=False)


# Gate outcome summary
gate_summary = (
    out.groupby(["dataset", "gate_outcome"])
       .size()
       .reset_index(name="n_tasks")
)

gate_summary_path = (
    OUT_DIR / "relbench_v2_fdhg_gate_outcome_summary.csv"
)
gate_summary.to_csv(gate_summary_path, index=False)


print("\n=== UNIFIED PAPER TABLE ===")
print(paper.to_string(index=False))

print("\n=== GATE OUTCOME SUMMARY ===")
print(gate_summary.to_string(index=False))

print("\nsaved:", full_path)
print("saved:", paper_path)
print("saved:", gate_summary_path)
