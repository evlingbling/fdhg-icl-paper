from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


SEEDS = [41, 42, 43, 44]

ROOT = Path(
    "results/rel_trial_eligibilities_child_matched"
)

OUT = Path(
    "results/rel_trial_eligibilities_child_multiseed"
)
OUT.mkdir(parents=True, exist_ok=True)


def normalize_binary(series):
    mapping = {
        "t": 1,
        "true": 1,
        "1": 1,
        "yes": 1,
        "y": 1,
        "f": 0,
        "false": 0,
        "0": 0,
        "no": 0,
        "n": 0,
    }

    out = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if out.isna().any():
        bad = sorted(
            series[out.isna()]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Unknown labels: {bad}"
        )

    return out.astype(int)


train_clean = pd.read_parquet(
    ROOT / "train_dfs_clean.parquet"
)
val_clean = pd.read_parquet(
    ROOT / "val_dfs_clean.parquet"
)

train_residual = pd.read_parquet(
    ROOT / "train_dmax2_residual_only.parquet"
)
val_residual = pd.read_parquet(
    ROOT / "val_dmax2_residual_only.parquet"
)

train_matched = pd.read_parquet(
    ROOT / "train_dfs_clean_plus_dmax2.parquet"
)
val_matched = pd.read_parquet(
    ROOT / "val_dfs_clean_plus_dmax2.parquet"
)

y_train = normalize_binary(
    train_clean["child"]
)
y_val = normalize_binary(
    val_clean["child"]
)

dfs_cols = [
    c for c in train_clean.columns
    if c.startswith("f_dfs_clean_")
]

residual_cols = [
    c for c in train_residual.columns
    if c.startswith("dmax2_")
]

matched_cols = dfs_cols + residual_cols


print("=== CLASS BALANCE ===")
print("train:")
print(y_train.value_counts().sort_index().to_string())
print(y_train.value_counts(normalize=True).sort_index().to_string())

print("\nval:")
print(y_val.value_counts().sort_index().to_string())
print(y_val.value_counts(normalize=True).sort_index().to_string())

print("\nfeature counts:")
print("DFS-clean:", len(dfs_cols))
print("dmax2 residual:", len(residual_cols))
print("combined:", len(matched_cols))


def calculate_metrics(
    seed,
    variant,
    y_prob,
    n_features,
    best_iteration,
):
    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    y_prob = np.clip(
        y_prob,
        1e-7,
        1 - 1e-7,
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    return {
        "dataset": "rel-trial",
        "task": "eligibilities-child",
        "seed": seed,
        "variant": variant,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_features": n_features,
        "best_iteration": best_iteration,
        "accuracy": float(
            accuracy_score(y_val, y_pred)
        ),
        "auroc": float(
            roc_auc_score(y_val, y_prob)
        ),
        "average_precision": float(
            average_precision_score(
                y_val,
                y_prob,
            )
        ),
        "log_loss": float(
            log_loss(y_val, y_prob)
        ),
        "macro_f1": float(
            f1_score(
                y_val,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_val,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def fit_variant(
    seed,
    variant,
    train_df,
    val_df,
    feature_cols,
):
    print(
        f"\n=== seed={seed} "
        f"variant={variant} "
        f"features={len(feature_cols)} ==="
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        random_strength=1.0,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        train_df[feature_cols].astype(float),
        y_train,
        eval_set=(
            val_df[feature_cols].astype(float),
            y_val,
        ),
        early_stopping_rounds=100,
        verbose=False,
    )

    y_prob = model.predict_proba(
        val_df[feature_cols].astype(float)
    )[:, 1]

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.get_feature_importance(),
        "seed": seed,
        "variant": variant,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        OUT / (
            f"feature_importance_"
            f"{variant}_seed{seed}.csv"
        ),
        index=False,
    )

    row = calculate_metrics(
        seed=seed,
        variant=variant,
        y_prob=y_prob,
        n_features=len(feature_cols),
        best_iteration=int(
            model.get_best_iteration()
        ),
    )

    print(row)
    return row


rows = []

prevalence = float(
    y_train.mean()
)

constant_prob = np.full(
    len(y_val),
    prevalence,
)

for seed in SEEDS:
    rows.append(
        calculate_metrics(
            seed=seed,
            variant="constant_prevalence",
            y_prob=constant_prob,
            n_features=0,
            best_iteration=-1,
        )
    )

    rows.append(
        fit_variant(
            seed,
            "dfs_clean",
            train_clean,
            val_clean,
            dfs_cols,
        )
    )

    rows.append(
        fit_variant(
            seed,
            "dmax2_residual_only",
            train_residual,
            val_residual,
            residual_cols,
        )
    )

    rows.append(
        fit_variant(
            seed,
            "dfs_clean_plus_dmax2",
            train_matched,
            val_matched,
            matched_cols,
        )
    )


runs = pd.DataFrame(rows)
runs.to_csv(
    OUT / "all_runs.csv",
    index=False,
)


metric_cols = [
    "accuracy",
    "auroc",
    "average_precision",
    "log_loss",
    "macro_f1",
    "weighted_f1",
]

summary_rows = []

for variant, group in runs.groupby(
    "variant",
    sort=False,
):
    row = {
        "dataset": "rel-trial",
        "task": "eligibilities-child",
        "variant": variant,
        "n_seeds": group["seed"].nunique(),
        "n_features": int(
            group["n_features"].iloc[0]
        ),
    }

    for metric in metric_cols:
        row[f"{metric}_mean"] = float(
            group[metric].mean()
        )
        row[f"{metric}_std"] = float(
            group[metric].std(ddof=1)
        )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    OUT / "summary.csv",
    index=False,
)


base = runs[
    runs["variant"] == "dfs_clean"
].set_index("seed")

candidate = runs[
    runs["variant"]
    == "dfs_clean_plus_dmax2"
].set_index("seed")

gate_rows = []

for seed in SEEDS:
    b = base.loc[seed]
    c = candidate.loc[seed]

    auroc_delta = (
        c["auroc"] - b["auroc"]
    )

    ap_delta = (
        c["average_precision"]
        - b["average_precision"]
    )

    logloss_reduction = (
        b["log_loss"] - c["log_loss"]
    )

    gate_rows.append({
        "dataset": "rel-trial",
        "task": "eligibilities-child",
        "seed": seed,
        "baseline": "dfs_clean",
        "candidate": "dfs_clean_plus_dmax2",
        "auroc_delta": auroc_delta,
        "average_precision_delta": ap_delta,
        "log_loss_reduction": logloss_reduction,
        "accuracy_delta": (
            c["accuracy"] - b["accuracy"]
        ),
        "macro_f1_delta": (
            c["macro_f1"] - b["macro_f1"]
        ),
        "gate_decision_primary_auroc": (
            "SELECT"
            if auroc_delta > 0
            else "FALLBACK"
        ),
    })

gate = pd.DataFrame(gate_rows)
gate.to_csv(
    OUT / "gate_by_seed.csv",
    index=False,
)


gate_summary = pd.DataFrame([{
    "dataset": "rel-trial",
    "task": "eligibilities-child",
    "seeds": "41,42,43,44",
    "n_seeds": len(SEEDS),
    "select_count_auroc": int(
        (
            gate["gate_decision_primary_auroc"]
            == "SELECT"
        ).sum()
    ),
    "fallback_count_auroc": int(
        (
            gate["gate_decision_primary_auroc"]
            == "FALLBACK"
        ).sum()
    ),
    "auroc_delta_mean": float(
        gate["auroc_delta"].mean()
    ),
    "auroc_delta_std": float(
        gate["auroc_delta"].std(ddof=1)
    ),
    "average_precision_delta_mean": float(
        gate["average_precision_delta"].mean()
    ),
    "average_precision_delta_std": float(
        gate["average_precision_delta"].std(ddof=1)
    ),
    "log_loss_reduction_mean": float(
        gate["log_loss_reduction"].mean()
    ),
    "log_loss_reduction_std": float(
        gate["log_loss_reduction"].std(ddof=1)
    ),
    "final_gate_decision_primary_auroc": (
        "SELECT"
        if gate["auroc_delta"].mean() > 0
        else "FALLBACK"
    ),
    "primary_metric": "auroc",
    "secondary_metrics": (
        "average_precision,log_loss"
    ),
}])

gate_summary.to_csv(
    OUT / "gate_summary.csv",
    index=False,
)


metadata = {
    "dataset": "rel-trial",
    "task": "eligibilities-child",
    "seeds": SEEDS,
    "evaluation_split": "official validation",
    "primary_metric": "auroc",
    "positive_rate_train": prevalence,
    "dfs_features": dfs_cols,
    "dmax2_residual_features": residual_cols,
    "excluded_leakage_columns": [
        "child",
        "child",
        "older_child",
        "minimum_age",
        "maximum_age",
        "population",
        "criteria",
        "gender_description",
    ],
}

with open(
    OUT / "metadata.json",
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print("\n=== SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== GATE BY SEED ===")
print(gate.to_string(index=False))

print("\n=== GATE SUMMARY ===")
print(gate_summary.to_string(index=False))

print("\nSaved under:", OUT)
