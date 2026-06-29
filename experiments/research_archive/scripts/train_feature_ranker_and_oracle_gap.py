from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split


ORACLE_FEATURES = {
    "event_count",
    "rating_mean",
    "brand_category_entropy",
    "price",
}

NUMERIC_CANDIDATES = [
    "event_count",
    "rating_mean",
    "amount_mean",
    "days_since_last_event",
    "brand_category_majconf",
    "brand_category_entropy",
    "brand_category_conflict_count",
    "brand_category_support_count",
    "price",
]


def infer_block(feature: str) -> str:
    if feature in {"event_count", "rating_mean", "amount_mean", "days_since_last_event"}:
        return "aggregation"
    if feature.startswith("brand_category_"):
        return "ambiguity"
    if feature == "price":
        return "pullup_identity"
    return "other"


def infer_operator(feature: str) -> str:
    if feature == "event_count":
        return "count"
    if feature in {"rating_mean", "amount_mean"}:
        return "mean"
    if feature == "days_since_last_event":
        return "last_delta"
    if feature.endswith("_entropy"):
        return "entropy"
    if feature.endswith("_majconf"):
        return "majority_confidence"
    if feature.endswith("_conflict_count"):
        return "conflict_count"
    if feature.endswith("_support_count"):
        return "support_count"
    if feature == "price":
        return "identity"
    return "unknown"


def build_candidate_metadata(seed_dir: Path) -> pd.DataFrame:
    train = pd.read_parquet(seed_dir / "target_train.parquet")
    products = pd.read_parquet(seed_dir / "table_products.parquet")

    rows = []
    for feature in NUMERIC_CANDIDATES:
        # Aggregation / ambiguity features live in target_train.
        # Target-table identity/pull-up features such as products.price live in table_products.
        if feature in train.columns:
            s = train[feature]
        elif feature in products.columns:
            s = products[feature]
        else:
            continue

        block = infer_block(feature)
        operator = infer_operator(feature)

        rows.append(
            {
                "seed_dir": seed_dir.name,
                "feature_name": feature,
                "block": block,
                "operator": operator,
                "coverage": float(1.0 - s.isna().mean()),
                "missing_rate": float(s.isna().mean()),
                "nunique_ratio": float(s.nunique(dropna=True) / max(len(s), 1)),
                "std": float(s.std(skipna=True)) if pd.api.types.is_numeric_dtype(s) else 0.0,
                "mean_abs": float(s.abs().mean(skipna=True)) if pd.api.types.is_numeric_dtype(s) else 0.0,
                "is_aggregation": int(block == "aggregation"),
                "is_ambiguity": int(block == "ambiguity"),
                "is_pullup": int(block == "pullup_identity"),
                "is_count": int(operator == "count"),
                "is_mean": int(operator == "mean"),
                "is_entropy": int(operator == "entropy"),
                "is_identity": int(operator == "identity"),
                "label_oracle_program": int(feature in ORACLE_FEATURES),
            }
        )

    return pd.DataFrame(rows)


def rank_features_for_seed(model, seed_dir: Path, feature_cols: list[str]) -> pd.DataFrame:
    cand = build_candidate_metadata(seed_dir)
    X = cand[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cand["ranker_score"] = model.predict_proba(X)[:, 1]
    cand = cand.sort_values("ranker_score", ascending=False).reset_index(drop=True)
    cand["rank"] = np.arange(1, len(cand) + 1)
    return cand


def recall_at_k(ranked: pd.DataFrame, k: int) -> float:
    topk = set(ranked.head(k)["feature_name"])
    return len(topk.intersection(ORACLE_FEATURES)) / len(ORACLE_FEATURES)


def evaluate_model_on_features(seed_dir: Path, features: list[str]) -> dict:
    train = pd.read_parquet(seed_dir / "target_train.parquet")
    val = pd.read_parquet(seed_dir / "target_val.parquet")
    products = pd.read_parquet(seed_dir / "table_products.parquet")

    # Add target-table identity / pull-up columns, such as products.price :: identity.
    product_cols = [c for c in products.columns if c != "product_id"]
    train = train.merge(products[["product_id"] + product_cols], on="product_id", how="left", suffixes=("", "__prod"))
    val = val.merge(products[["product_id"] + product_cols], on="product_id", how="left", suffixes=("", "__prod"))

    # If a feature already existed in target with the same name, keep the original.
    # If merge created price__prod but price is requested and absent, copy it back.
    for c in product_cols:
        prod_col = f"{c}__prod"
        if c not in train.columns and prod_col in train.columns:
            train[c] = train[prod_col]
        if c not in val.columns and prod_col in val.columns:
            val[c] = val[prod_col]

    missing = [f for f in features if f not in train.columns or f not in val.columns]
    if missing:
        raise KeyError(f"Requested features missing after product pull-up: {missing}")

    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train["churn"].astype(int).to_numpy()

    X_val = val[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_val = val["churn"].astype(int).to_numpy()

    clf = HistGradientBoostingClassifier(random_state=41)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    prob = clf.predict_proba(X_val)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_val, pred)),
        "roc_auc": float(roc_auc_score(y_val, prob)),
        "average_precision": float(average_precision_score(y_val, prob)),
        "log_loss": float(log_loss(y_val, prob, labels=[0, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", default="outputs/synthetic_prior")
    parser.add_argument("--out-dir", default="outputs/ranker/gbt_minimal")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--train-seeds", nargs="*", default=["41", "42", "43", "44", "45", "46", "47"])
    parser.add_argument("--test-seeds", nargs="*", default=["48", "49", "50"])
    parser.add_argument("--ks", nargs="*", type=int, default=[1, 2, 4, 8, 128])
    args = parser.parse_args()

    prior_root = Path(args.prior_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs = [prior_root / f"minimal_seed{s}" for s in args.train_seeds]
    test_dirs = [prior_root / f"minimal_seed{s}" for s in args.test_seeds]

    train_meta = pd.concat(
        [build_candidate_metadata(d) for d in train_dirs],
        ignore_index=True,
    )

    feature_cols = [
        "coverage",
        "missing_rate",
        "nunique_ratio",
        "std",
        "mean_abs",
        "is_aggregation",
        "is_ambiguity",
        "is_pullup",
        "is_count",
        "is_mean",
        "is_entropy",
        "is_identity",
    ]

    X = train_meta[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_meta["label_oracle_program"].astype(int).to_numpy()

    model = GradientBoostingClassifier(random_state=args.seed)
    model.fit(X, y)

    train_prob = model.predict_proba(X)[:, 1]
    train_pred = (train_prob >= 0.5).astype(int)

    ranker_train_metrics = {
        "n_train_examples": int(len(train_meta)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "train_accuracy": float(accuracy_score(y, train_pred)),
        "train_roc_auc": float(roc_auc_score(y, train_prob)),
        "train_average_precision": float(average_precision_score(y, train_prob)),
        "train_log_loss": float(log_loss(y, train_prob, labels=[0, 1])),
    }

    recall_rows = []
    oracle_gap_rows = []
    ranked_all = []

    for seed_dir in test_dirs:
        ranked = rank_features_for_seed(model, seed_dir, feature_cols)
        ranked_all.append(ranked)

        for k in args.ks:
            recall_rows.append(
                {
                    "seed_dir": seed_dir.name,
                    "K": k,
                    "Recall@K": recall_at_k(ranked, k),
                    "topK": ",".join(ranked.head(k)["feature_name"].tolist()),
                }
            )

        selected_top4 = ranked.head(4)["feature_name"].tolist()
        oracle_features = list(ORACLE_FEATURES)

        selected_metrics = evaluate_model_on_features(seed_dir, selected_top4)
        oracle_metrics = evaluate_model_on_features(seed_dir, oracle_features)

        oracle_gap_rows.append(
            {
                "seed_dir": seed_dir.name,
                "selected_features": ",".join(selected_top4),
                "oracle_features": ",".join(oracle_features),
                "selected_log_loss": selected_metrics["log_loss"],
                "oracle_log_loss": oracle_metrics["log_loss"],
                "oracle_gap_log_loss": selected_metrics["log_loss"] - oracle_metrics["log_loss"],
                "selected_roc_auc": selected_metrics["roc_auc"],
                "oracle_roc_auc": oracle_metrics["roc_auc"],
                "oracle_gap_roc_auc": oracle_metrics["roc_auc"] - selected_metrics["roc_auc"],
                "selected_average_precision": selected_metrics["average_precision"],
                "oracle_average_precision": oracle_metrics["average_precision"],
                "oracle_gap_average_precision": oracle_metrics["average_precision"] - selected_metrics["average_precision"],
            }
        )

    recall_df = pd.DataFrame(recall_rows)
    oracle_gap_df = pd.DataFrame(oracle_gap_rows)
    ranked_df = pd.concat(ranked_all, ignore_index=True)

    with open(out_dir / "feature_ranker_gbt.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(out_dir / "ranker_train_metrics.json", "w") as f:
        json.dump(ranker_train_metrics, f, indent=2)

    recall_df.to_csv(out_dir / "synthetic_recall_at_k.csv", index=False)
    oracle_gap_df.to_csv(out_dir / "oracle_gap.csv", index=False)
    ranked_df.to_csv(out_dir / "heldout_ranked_features.csv", index=False)

    summary = {
        "ranker_train_metrics": ranker_train_metrics,
        "mean_recall_by_k": recall_df.groupby("K")["Recall@K"].mean().to_dict(),
        "mean_oracle_gap_log_loss": float(oracle_gap_df["oracle_gap_log_loss"].mean()),
        "mean_oracle_gap_roc_auc": float(oracle_gap_df["oracle_gap_roc_auc"].mean()),
        "mean_oracle_gap_average_precision": float(oracle_gap_df["oracle_gap_average_precision"].mean()),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Feature ranker training complete ===")
    print(json.dumps(summary, indent=2))

    print("\nRecall@K:")
    print(recall_df.to_string(index=False))

    print("\nOracle gap:")
    print(oracle_gap_df.to_string(index=False))

    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
