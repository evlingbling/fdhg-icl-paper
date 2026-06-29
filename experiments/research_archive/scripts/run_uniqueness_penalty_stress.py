import os
import json
import math
import random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

OUT_DIR = "results/uniqueness_stress"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [41, 42, 43, 44, 45, 46, 47, 48]
N_TRAIN = 6000
N_VAL = 3000
TOP_K = 8

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_data(seed, n):
    rng = np.random.default_rng(seed)

    # Surrogate identifiers.
    row_id = np.arange(n)
    user_id = np.arange(n) + seed * 10_000_000

    # Meaningful low-cardinality relational attributes.
    group = rng.integers(0, 20, size=n)
    region = rng.integers(0, 8, size=n)

    # Planted FD-like dependencies:
    # group -> group_pref
    group_pref_map = rng.normal(0, 1, size=20)
    group_pref = group_pref_map[group]

    # region -> region_risk
    region_risk_map = rng.normal(0, 1, size=8)
    region_risk = region_risk_map[region]

    # Noise features.
    noise_1 = rng.normal(0, 1, size=n)
    noise_2 = rng.normal(0, 1, size=n)
    noise_3 = rng.normal(0, 1, size=n)

    # Surrogate-key dependent columns: these are trivially determined by row_id/user_id,
    # but should not be selected as useful relational dependencies.
    row_hash = (row_id * 2654435761) % 9973
    user_hash = (user_id * 1000003) % 7919

    # Label depends on meaningful dependencies, not IDs.
    logits = (
        1.3 * group_pref
        + 1.0 * region_risk
        + 0.3 * noise_1
        - 0.2 * noise_2
    )
    probs = sigmoid(logits)
    y = rng.binomial(1, probs)

    df = pd.DataFrame({
        "row_id": row_id,
        "user_id": user_id,
        "group": group,
        "region": region,
        "group_pref": group_pref,
        "region_risk": region_risk,
        "noise_1": noise_1,
        "noise_2": noise_2,
        "noise_3": noise_3,
        "row_hash": row_hash,
        "user_hash": user_hash,
        "target": y,
    })
    return df

def fd_score(df, lhs, rhs, use_uniqueness_penalty):
    # Approx FD score: average majority confidence rhs within lhs groups.
    g = df.groupby(lhs)[rhs]
    n = len(df)

    maj_counts = g.apply(lambda s: s.value_counts(dropna=False).iloc[0])
    support = g.size()

    confidence = maj_counts.sum() / n

    lhs_unique_ratio = df[lhs].nunique(dropna=False) / n
    rhs_unique_ratio = df[rhs].nunique(dropna=False) / n

    # Penalize identifiers / near-unique LHS columns.
    # This is intentionally simple and interpretable for stress testing.
    if use_uniqueness_penalty:
        penalty = max(0.0, 1.0 - lhs_unique_ratio) ** 2
    else:
        penalty = 1.0

    score = confidence * penalty

    return {
        "lhs": lhs,
        "rhs": rhs,
        "confidence": float(confidence),
        "lhs_unique_ratio": float(lhs_unique_ratio),
        "rhs_unique_ratio": float(rhs_unique_ratio),
        "score": float(score),
        "is_id_like": bool(lhs_unique_ratio > 0.95 or lhs.endswith("_id")),
        "is_meaningful": bool((lhs, rhs) in {
            ("group", "group_pref"),
            ("region", "region_risk"),
        }),
    }

def discover_edges(train_df, use_uniqueness_penalty):
    lhs_cols = ["row_id", "user_id", "group", "region"]
    rhs_cols = ["group_pref", "region_risk", "row_hash", "user_hash", "noise_1", "noise_2", "noise_3"]

    rows = []
    for lhs in lhs_cols:
        for rhs in rhs_cols:
            if lhs == rhs:
                continue
            rows.append(fd_score(train_df, lhs, rhs, use_uniqueness_penalty))

    edges = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    selected = edges.head(TOP_K).copy()
    return edges, selected

def compile_features(df, selected_edges):
    X = pd.DataFrame(index=df.index)

    # Always include weak base features.
    X["group"] = df["group"]
    X["region"] = df["region"]
    X["noise_1"] = df["noise_1"]
    X["noise_2"] = df["noise_2"]

    # Add selected RHS features as if compiled through FD programs.
    for _, e in selected_edges.iterrows():
        lhs = e["lhs"]
        rhs = e["rhs"]
        name = f"fd__{lhs}_to_{rhs}"
        X[name] = df[rhs]

    return X

def evaluate(train_df, val_df, selected_edges, seed):
    X_train = compile_features(train_df, selected_edges)
    X_val = compile_features(val_df, selected_edges)
    y_train = train_df["target"].astype(int)
    y_val = val_df["target"].astype(int)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_val)[:, 1]
    pred_label = (pred >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_val, pred_label),
        "roc_auc": roc_auc_score(y_val, pred),
        "average_precision": average_precision_score(y_val, pred),
        "log_loss": log_loss(y_val, pred, labels=[0, 1]),
        "n_features": X_train.shape[1],
    }

def run():
    all_runs = []
    all_edges = []

    for seed in SEEDS:
        train_df = make_data(seed, N_TRAIN)
        val_df = make_data(seed + 1000, N_VAL)

        for variant, use_penalty in [
            ("without_uniqueness_penalty", False),
            ("with_uniqueness_penalty", True),
        ]:
            edges, selected = discover_edges(train_df, use_penalty)
            metrics = evaluate(train_df, val_df, selected, seed)

            selected_edges = selected[["lhs", "rhs", "score", "confidence", "lhs_unique_ratio", "is_id_like", "is_meaningful"]].to_dict("records")

            id_like_count = int(selected["is_id_like"].sum())
            meaningful_count = int(selected["is_meaningful"].sum())

            row = {
                "seed": seed,
                "variant": variant,
                "use_uniqueness_penalty": use_penalty,
                "top_k": TOP_K,
                "selected_edges": json.dumps(selected_edges),
                "id_like_selected_edges": id_like_count,
                "meaningful_selected_edges": meaningful_count,
                "id_like_ratio": id_like_count / TOP_K,
                "meaningful_recall": meaningful_count / 2.0,
                **metrics,
            }
            all_runs.append(row)

            selected_copy = selected.copy()
            selected_copy["seed"] = seed
            selected_copy["variant"] = variant
            all_edges.append(selected_copy)

    runs = pd.DataFrame(all_runs)
    edges = pd.concat(all_edges, ignore_index=True)

    runs_path = os.path.join(OUT_DIR, "uniqueness_penalty_stress_runs.csv")
    edges_path = os.path.join(OUT_DIR, "uniqueness_penalty_stress_selected_edges.csv")

    runs.to_csv(runs_path, index=False)
    edges.to_csv(edges_path, index=False)

    summary = (
        runs.groupby("variant", as_index=False)
        .agg(
            n_runs=("seed", "count"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(x)))),
            id_like_selected_edges_mean=("id_like_selected_edges", "mean"),
            meaningful_selected_edges_mean=("meaningful_selected_edges", "mean"),
            id_like_ratio_mean=("id_like_ratio", "mean"),
            meaningful_recall_mean=("meaningful_recall", "mean"),
            n_features_mean=("n_features", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
        )
    )

    summary_path = os.path.join(OUT_DIR, "uniqueness_penalty_stress_summary.csv")
    summary.to_csv(summary_path, index=False)

    os.makedirs("results/final_tables", exist_ok=True)
    runs.to_csv("results/final_tables/uniqueness_penalty_stress_runs.csv", index=False)
    edges.to_csv("results/final_tables/uniqueness_penalty_stress_selected_edges.csv", index=False)
    summary.to_csv("results/final_tables/uniqueness_penalty_stress_summary.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(runs_path)
    print(edges_path)
    print(summary_path)
    print("results/final_tables/uniqueness_penalty_stress_runs.csv")
    print("results/final_tables/uniqueness_penalty_stress_selected_edges.csv")
    print("results/final_tables/uniqueness_penalty_stress_summary.csv")

if __name__ == "__main__":
    run()
