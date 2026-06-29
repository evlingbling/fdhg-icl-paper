from pathlib import Path
import numpy as np
import pandas as pd


FINAL = Path("results/final_tables")

BASE_PATH = FINAL / "relbench_v2_fdhg_gate_unified_summary.csv"

OUT_UNIFIED = FINAL / "relbench_19task_fdhg_gate_unified_summary.csv"
OUT_PAPER = FINAL / "relbench_19task_fdhg_gate_paper_table.csv"
OUT_OUTCOME = FINAL / "relbench_19task_fdhg_gate_outcome_summary.csv"


base = pd.read_csv(BASE_PATH)
schema = list(base.columns)


def blank_row():
    return {c: np.nan for c in schema}


def get_variant(summary, variant):
    row = summary.loc[summary["variant"] == variant]
    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one row for variant={variant}, found {len(row)}"
        )
    return row.iloc[0]


new_rows = []


# ------------------------------------------------------------------
# 1. rel-trial / studies-enrollment
# Primary metric: log1p RMSE, lower is better
# ------------------------------------------------------------------
summary = pd.read_csv(
    "results/rel_trial_studies_enrollment_multiseed/summary.csv"
)

b = get_variant(summary, "dfs_clean")
c = get_variant(summary, "dfs_clean_plus_dmax2")

row = blank_row()

base_score = float(b["rmse_log1p_mean"])
candidate_score = float(c["rmse_log1p_mean"])
gain = base_score - candidate_score

row.update({
    "dataset": "rel-trial",
    "task": "studies-enrollment",
    "task_family": "relbench_autocomplete_regression",
    "decoder": "catboost_regressor",
    "gate_outcome": "SELECT",
    "gate_reason": "lower_multiseed_log1p_rmse",
    "primary_metric": "rmse_log1p",
    "base_variant": "dfs_clean",
    "candidate_variant": "dfs_clean_plus_dmax2",
    "selected_variant": "dfs_clean_plus_dmax2",
    "base_n_features": int(b["n_features"]),
    "candidate_n_features": int(c["n_features"]),
    "selected_residual_count": int(c["n_features"] - b["n_features"]),
    "base_score": base_score,
    "candidate_score": candidate_score,
    "gated_score": candidate_score,
    "gain": gain,

    # Store log1p-scale regression metrics in the existing regression fields.
    "rmse_base": float(b["rmse_log1p_mean"]),
    "rmse_candidate": float(c["rmse_log1p_mean"]),
    "rmse_gated": float(c["rmse_log1p_mean"]),
    "rmse_reduction": (
        float(b["rmse_log1p_mean"])
        - float(c["rmse_log1p_mean"])
    ),
    "mae_base": float(b["mae_log1p_mean"]),
    "mae_candidate": float(c["mae_log1p_mean"]),
    "mae_gated": float(c["mae_log1p_mean"]),

    "fallback_exact_match": False,
    "role_in_paper": "multiseed_regression_positive_case",
})

new_rows.append(row)


# ------------------------------------------------------------------
# Shared helper for binary rel-trial tasks
# ------------------------------------------------------------------
def add_binary_task(task_name, summary_path, role):
    summary = pd.read_csv(summary_path)

    b = get_variant(summary, "dfs_clean")
    c = get_variant(summary, "dfs_clean_plus_dmax2")

    row = blank_row()

    base_score = float(b["auroc_mean"])
    candidate_score = float(c["auroc_mean"])

    row.update({
        "dataset": "rel-trial",
        "task": task_name,
        "task_family": "relbench_binary_classification",
        "decoder": "catboost_classifier",
        "gate_outcome": "SELECT",
        "gate_reason": "higher_multiseed_auroc",
        "primary_metric": "auroc",
        "base_variant": "dfs_clean",
        "candidate_variant": "dfs_clean_plus_dmax2",
        "selected_variant": "dfs_clean_plus_dmax2",
        "base_n_features": int(b["n_features"]),
        "candidate_n_features": int(c["n_features"]),
        "selected_residual_count": int(
            c["n_features"] - b["n_features"]
        ),
        "base_score": base_score,
        "candidate_score": candidate_score,
        "gated_score": candidate_score,
        "gain": candidate_score - base_score,

        "accuracy_base": float(b["accuracy_mean"]),
        "accuracy_candidate": float(c["accuracy_mean"]),
        "accuracy_gated": float(c["accuracy_mean"]),
        "accuracy_gain": (
            float(c["accuracy_mean"])
            - float(b["accuracy_mean"])
        ),

        "auroc_base": float(b["auroc_mean"]),
        "auroc_candidate": float(c["auroc_mean"]),
        "auroc_gated": float(c["auroc_mean"]),
        "auroc_gain": (
            float(c["auroc_mean"])
            - float(b["auroc_mean"])
        ),

        "ap_base": float(b["average_precision_mean"]),
        "ap_candidate": float(c["average_precision_mean"]),
        "ap_gated": float(c["average_precision_mean"]),
        "ap_gain": (
            float(c["average_precision_mean"])
            - float(b["average_precision_mean"])
        ),

        "macro_f1_base": float(b["macro_f1_mean"]),
        "macro_f1_candidate": float(c["macro_f1_mean"]),
        "macro_f1_gated": float(c["macro_f1_mean"]),
        "macro_f1_gain": (
            float(c["macro_f1_mean"])
            - float(b["macro_f1_mean"])
        ),

        "weighted_f1_base": float(b["weighted_f1_mean"]),
        "weighted_f1_candidate": float(c["weighted_f1_mean"]),
        "weighted_f1_gated": float(c["weighted_f1_mean"]),
        "weighted_f1_gain": (
            float(c["weighted_f1_mean"])
            - float(b["weighted_f1_mean"])
        ),

        "log_loss_base": float(b["log_loss_mean"]),
        "log_loss_candidate": float(c["log_loss_mean"]),
        "log_loss_gated": float(c["log_loss_mean"]),
        "log_loss_reduction": (
            float(b["log_loss_mean"])
            - float(c["log_loss_mean"])
        ),

        "fallback_exact_match": False,
        "role_in_paper": role,
    })

    new_rows.append(row)


add_binary_task(
    task_name="studies-has_dmc",
    summary_path=(
        "results/rel_trial_studies_has_dmc_multiseed/summary.csv"
    ),
    role="multiseed_binary_positive_case",
)

add_binary_task(
    task_name="eligibilities-adult",
    summary_path=(
        "results/rel_trial_eligibilities_adult_multiseed/summary.csv"
    ),
    role="bridge_aware_multiseed_positive_case",
)


# ------------------------------------------------------------------
# Combine and validate
# ------------------------------------------------------------------
trial = pd.DataFrame(new_rows, columns=schema)

combined = pd.concat(
    [base, trial],
    ignore_index=True,
)

combined = combined.sort_values(
    ["dataset", "task"],
    kind="stable",
).reset_index(drop=True)


if len(combined) != 19:
    raise ValueError(
        f"Expected 19 rows after integration, got {len(combined)}"
    )

if combined[["dataset", "task"]].duplicated().any():
    dup = combined.loc[
        combined[["dataset", "task"]].duplicated(False),
        ["dataset", "task"],
    ]
    raise ValueError(
        "Duplicate dataset-task pairs:\n"
        + dup.to_string(index=False)
    )


combined.to_csv(
    OUT_UNIFIED,
    index=False,
)


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

paper = combined[paper_cols].copy()

paper.to_csv(
    OUT_PAPER,
    index=False,
)


outcome = (
    combined.groupby("gate_outcome", dropna=False)
    .size()
    .rename("n_tasks")
    .reset_index()
)

outcome["fraction"] = (
    outcome["n_tasks"] / len(combined)
)

outcome.to_csv(
    OUT_OUTCOME,
    index=False,
)


print("=== REL-TRIAL ROWS ADDED ===")
print(
    trial[
        [
            "dataset",
            "task",
            "primary_metric",
            "base_score",
            "candidate_score",
            "gain",
            "gate_outcome",
            "selected_residual_count",
        ]
    ].to_string(index=False)
)

print("\n=== 19-TASK OUTCOME ===")
print(outcome.to_string(index=False))

print("\n=== TASKS PER DATASET ===")
print(
    combined.groupby("dataset")
    .size()
    .sort_values(ascending=False)
    .to_string()
)

print("\nTotal rows:", len(combined))

print("\nSaved:")
print(OUT_UNIFIED)
print(OUT_PAPER)
print(OUT_OUTCOME)
