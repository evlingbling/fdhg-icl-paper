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

ALL_CANDIDATE_FEATURES = [
    "event_count",
    "rating_mean",
    "brand_category_entropy",
    "price",
    "amount_mean",
    "days_since_last_event",
    "brand_category_majconf",
    "brand_category_conflict_count",
    "brand_category_support_count",
    "brand",
    "category",
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

    key_candidates = [
        "product_id",
        "product",
        "ProductId",
        "productId",
        "item_id",
        "item",
        "sku",
    ]
    merge_key = None
    for k in key_candidates:
        if k in target.columns and k in products.columns:
            merge_key = k
            break

    if merge_key is None:
        # If there is no obvious product key, return target as-is.
        # This keeps the script robust for already-compiled target tables.
        return target.copy()

    out = target.merge(
        products,
        on=merge_key,
        how="left",
        suffixes=("", "_product"),
    )

    # Pull up product identity features such as products.price :: identity.
    for col in products.columns:
        prod_col = f"{col}_product"
        if col not in out.columns and prod_col in out.columns:
            out[col] = out[prod_col]

    return out


def find_target_col(df: pd.DataFrame) -> str:
    for col in ["target", "label", "y", "churn", "WillGetBadge"]:
        if col in df.columns:
            return col
    raise ValueError(f"No target column found. Columns={list(df.columns)}")


def make_xy(df: pd.DataFrame, features: list[str], target_col: str):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}; available={list(df.columns)}")

    X = df[features].copy()

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col].astype(str), sort=True)[0]
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        if X[col].isna().any():
            median = X[col].median()
            X[col] = X[col].fillna(0.0 if pd.isna(median) else median)

    y = df[target_col].astype(int).to_numpy()
    return X.to_numpy(dtype=np.float32), y


def evaluate_tabpfn(seed_dir: Path, features: list[str], device: str, seed: int):
    train = load_target_with_product_pullup(seed_dir, "train")
    val = load_target_with_product_pullup(seed_dir, "val")

    target_col = find_target_col(train)

    missing = [f for f in features if f not in train.columns or f not in val.columns]
    if missing:
        raise KeyError(f"Missing features for {seed_dir.name}: {missing}")

    X_train, y_train = make_xy(train, features, target_col)
    X_val, y_val = make_xy(val, features, target_col)

    clf = get_tabpfn_classifier(device=device, seed=seed)
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_val, pred)),
        "roc_auc": float(roc_auc_score(y_val, prob)),
        "average_precision": float(average_precision_score(y_val, prob)),
        "log_loss": float(log_loss(y_val, prob, labels=[0, 1])),
    }


def selected_features_for_seed(ranked_path: Path, seed_name: str, k: int) -> list[str]:
    ranked = pd.read_csv(ranked_path)

    if "seed_dir" in ranked.columns:
        sub = ranked[ranked["seed_dir"].astype(str).eq(seed_name)].copy()
    elif "seed" in ranked.columns:
        seed_num = seed_name.replace("minimal_seed", "")
        sub = ranked[ranked["seed"].astype(str).eq(seed_num)].copy()
    else:
        # Fallback: search all string columns for seed name.
        mask = ranked.astype(str).apply(
            lambda row: row.str.contains(seed_name, regex=False).any(),
            axis=1,
        )
        sub = ranked[mask].copy()

    if len(sub) == 0:
        raise ValueError(f"No ranked features found for seed_dir={seed_name}")

    feature_col = None
    for c in ["feature", "feature_name", "program", "program_feature"]:
        if c in sub.columns:
            feature_col = c
            break

    if feature_col is None:
        raise ValueError(f"No feature column found in ranked features. Columns={list(sub.columns)}")

    score_col = None
    for c in ["ranker_score", "score", "pred_score", "importance"]:
        if c in sub.columns:
            score_col = c
            break

    if score_col is not None:
        sub = sub.sort_values(score_col, ascending=False)

    return sub[feature_col].astype(str).head(k).tolist()


def available_candidate_features(seed_dir: Path) -> list[str]:
    train = load_target_with_product_pullup(seed_dir, "train")
    val = load_target_with_product_pullup(seed_dir, "val")
    return [f for f in ALL_CANDIDATE_FEATURES if f in train.columns and f in val.columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", default="outputs/synthetic_prior")
    parser.add_argument("--ranked-features", default="outputs/ranker/gbt_synthetic_prior_seed41_60/heldout_ranked_features.csv")
    parser.add_argument("--out-dir", default="outputs/synthetic_tabpfn_oracle_gap/seed53_60_top4_with_random")
    parser.add_argument("--test-seeds", nargs="*", default=["53", "54", "55", "56", "57", "58", "59", "60"])
    parser.add_argument("--selected-k", type=int, default=4)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    prior_root = Path(args.prior_root)
    ranked_path = Path(args.ranked_features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    random_detail_rows = []

    for seed_str in args.test_seeds:
        seed_name = f"minimal_seed{seed_str}"
        seed_dir = prior_root / seed_name

        print("\n" + "=" * 100)
        print(seed_name)

        selected = selected_features_for_seed(ranked_path, seed_name, args.selected_k)

        feature_sets = {
            "weak_price_only": WEAK_FEATURES,
            f"ranker_top{args.selected_k}": selected,
            "oracle": ORACLE_FEATURES,
        }

        for method, features in feature_sets.items():
            print(method, "features:", features)
            metrics = evaluate_tabpfn(seed_dir, features, device=args.device, seed=args.seed)
            rows.append(
                {
                    "seed_dir": seed_name,
                    "method": method,
                    "n_features": len(features),
                    "features": ",".join(features),
                    **metrics,
                }
            )

        # Random same-budget baseline: average over n_random random top-K draws.
        candidates = available_candidate_features(seed_dir)
        if len(candidates) < args.selected_k:
            raise ValueError(f"Not enough candidate features for random draw: {candidates}")

        rng = np.random.default_rng(int(seed_str) + args.seed)
        random_metrics = []

        for rep in range(args.n_random):
            random_features = list(rng.choice(candidates, size=args.selected_k, replace=False))
            print(f"random_top{args.selected_k} rep={rep} features:", random_features)

            metrics = evaluate_tabpfn(
                seed_dir,
                random_features,
                device=args.device,
                seed=args.seed + 1000 + rep,
            )
            random_metrics.append(metrics)

            random_detail_rows.append(
                {
                    "seed_dir": seed_name,
                    "random_rep": rep,
                    "method": f"random_top{args.selected_k}",
                    "n_features": len(random_features),
                    "features": ",".join(random_features),
                    **metrics,
                }
            )

        random_mean = {
            m: float(np.mean([row[m] for row in random_metrics]))
            for m in ["accuracy", "roc_auc", "average_precision", "log_loss"]
        }

        rows.append(
            {
                "seed_dir": seed_name,
                "method": f"random_top{args.selected_k}",
                "n_features": args.selected_k,
                "features": f"mean_of_{args.n_random}_random_draws",
                **random_mean,
            }
        )

    df = pd.DataFrame(rows)
    random_detail_df = pd.DataFrame(random_detail_rows)

    gap_rows = []
    for seed_dir, g in df.groupby("seed_dir"):
        oracle = g[g["method"].eq("oracle")].iloc[0]
        for _, row in g.iterrows():
            if row["method"] == "oracle":
                continue
            gap_rows.append(
                {
                    "seed_dir": seed_dir,
                    "method": row["method"],
                    "log_loss_gap_vs_oracle": row["log_loss"] - oracle["log_loss"],
                    "roc_auc_gap_vs_oracle": oracle["roc_auc"] - row["roc_auc"],
                    "average_precision_gap_vs_oracle": oracle["average_precision"] - row["average_precision"],
                }
            )

    gap_df = pd.DataFrame(gap_rows)

    result_path = out_dir / "synthetic_tabpfn_oracle_gap_results.csv"
    gap_path = out_dir / "synthetic_tabpfn_oracle_gap_summary.csv"
    random_detail_path = out_dir / "synthetic_tabpfn_random_detail.csv"

    df.to_csv(result_path, index=False)
    gap_df.to_csv(gap_path, index=False)
    random_detail_df.to_csv(random_detail_path, index=False)

    summary = {
        "mean_metrics": df.groupby("method")[["accuracy", "roc_auc", "average_precision", "log_loss"]].mean().to_dict(),
        "mean_gaps": gap_df.groupby("method")[
            ["log_loss_gap_vs_oracle", "roc_auc_gap_vs_oracle", "average_precision_gap_vs_oracle"]
        ].mean().to_dict(),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(result_path)
    print(gap_path)
    print(random_detail_path)
    print(out_dir / "summary.json")

    print("\n=== Mean metrics ===")
    print(df.groupby("method")[["accuracy", "roc_auc", "average_precision", "log_loss"]].mean().to_string())

    print("\n=== Mean gaps vs oracle ===")
    print(gap_df.groupby("method")[["roc_auc_gap_vs_oracle", "average_precision_gap_vs_oracle", "log_loss_gap_vs_oracle"]].mean().to_string())


if __name__ == "__main__":
    main()
