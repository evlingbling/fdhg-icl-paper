from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score


ORACLE_FEATURES = [
    "event_count",
    "rating_mean",
    "brand_category_entropy",
    "price",
]

WEAK_FEATURES = [
    "price",
]


def get_tabpfn_classifier(device: str, seed: int):
    from tabpfn import TabPFNClassifier

    try:
        return TabPFNClassifier(device=device, random_state=seed)
    except TypeError:
        return TabPFNClassifier(device=device)


def load_target_with_product_pullup(seed_dir: Path, split: str) -> pd.DataFrame:
    target = pd.read_parquet(seed_dir / f"target_{split}.parquet")
    products = pd.read_parquet(seed_dir / "table_products.parquet")

    # Pull up product identity features such as products.price :: identity.
    product_cols = [c for c in products.columns if c != "product_id"]
    out = target.merge(
        products[["product_id"] + product_cols],
        on="product_id",
        how="left",
        suffixes=("", "__prod"),
    )

    for col in product_cols:
        prod_col = f"{col}__prod"
        if col not in out.columns and prod_col in out.columns:
            out[col] = out[prod_col]

    return out


def evaluate_feature_set(
    seed_dir: Path,
    features: list[str],
    *,
    device: str,
    seed: int,
) -> dict:
    train = load_target_with_product_pullup(seed_dir, "train")
    val = load_target_with_product_pullup(seed_dir, "val")

    missing = [f for f in features if f not in train.columns or f not in val.columns]
    if missing:
        raise KeyError(f"Missing features for {seed_dir.name}: {missing}")

    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train["churn"].astype(int).to_numpy()

    X_val = val[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_val = val["churn"].astype(int).to_numpy()

    clf = get_tabpfn_classifier(device, seed)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    proba = clf.predict_proba(X_val)
    score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()

    return {
        "accuracy": float(accuracy_score(y_val, pred)),
        "roc_auc": float(roc_auc_score(y_val, score)),
        "average_precision": float(average_precision_score(y_val, score)),
        "log_loss": float(log_loss(y_val, score, labels=[0, 1])),
    }


def selected_features_for_seed(
    ranked_path: Path,
    seed_name: str,
    k: int,
) -> list[str]:
    ranked = pd.read_csv(ranked_path)
    sub = ranked[ranked["seed_dir"].eq(seed_name)].sort_values("rank")
    if sub.empty:
        raise ValueError(f"No ranked features found for seed_dir={seed_name}")
    return sub.head(k)["feature_name"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", default="outputs/synthetic_prior")
    parser.add_argument("--ranked-features", default="outputs/ranker/gbt_minimal/heldout_ranked_features.csv")
    parser.add_argument("--out-dir", default="outputs/synthetic_tabpfn_oracle_gap")
    parser.add_argument("--test-seeds", nargs="*", default=["48", "49", "50"])
    parser.add_argument("--selected-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    prior_root = Path(args.prior_root)
    ranked_path = Path(args.ranked_features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed_str in args.test_seeds:
        seed_dir = prior_root / f"minimal_seed{seed_str}"
        seed_name = seed_dir.name

        selected = selected_features_for_seed(ranked_path, seed_name, args.selected_k)

        feature_sets = {
            "weak_price_only": WEAK_FEATURES,
            f"ranker_top{args.selected_k}": selected,
            "oracle": ORACLE_FEATURES,
        }

        for method, features in feature_sets.items():
            print(f"\n=== {seed_name} / {method} ===")
            print("features:", features)

            metrics = evaluate_feature_set(
                seed_dir,
                features,
                device=args.device,
                seed=args.seed,
            )

            row = {
                "seed_dir": seed_name,
                "method": method,
                "n_features": len(features),
                "features": ",".join(features),
                **metrics,
            }
            rows.append(row)

            print(json.dumps(metrics, indent=2))

    df = pd.DataFrame(rows)

    # Compute oracle gaps per seed.
    gap_rows = []
    for seed_name, g in df.groupby("seed_dir"):
        oracle = g[g["method"].eq("oracle")].iloc[0]

        for _, row in g.iterrows():
            if row["method"] == "oracle":
                continue

            gap_rows.append(
                {
                    "seed_dir": seed_name,
                    "method": row["method"],
                    "log_loss_gap_vs_oracle": row["log_loss"] - oracle["log_loss"],
                    "roc_auc_gap_vs_oracle": oracle["roc_auc"] - row["roc_auc"],
                    "average_precision_gap_vs_oracle": oracle["average_precision"] - row["average_precision"],
                }
            )

    gap_df = pd.DataFrame(gap_rows)

    result_path = out_dir / "synthetic_tabpfn_oracle_gap_results.csv"
    gap_path = out_dir / "synthetic_tabpfn_oracle_gap_summary.csv"

    df.to_csv(result_path, index=False)
    gap_df.to_csv(gap_path, index=False)

    summary = {
        "mean_metrics": df.groupby("method")[["accuracy", "roc_auc", "average_precision", "log_loss"]].mean().to_dict(),
        "mean_gaps": gap_df.groupby("method")[[
            "log_loss_gap_vs_oracle",
            "roc_auc_gap_vs_oracle",
            "average_precision_gap_vs_oracle",
        ]].mean().to_dict(),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Saved ===")
    print(result_path)
    print(gap_path)
    print(out_dir / "summary.json")

    print("\n=== Results ===")
    print(df.to_string(index=False))

    print("\n=== Gaps vs oracle ===")
    print(gap_df.to_string(index=False))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
