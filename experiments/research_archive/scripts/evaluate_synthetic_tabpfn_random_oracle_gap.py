import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score

try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None


ORACLE_FEATURES = [
    "event_count",
    "rating_mean",
    "brand_category_entropy",
    "price",
]

WEAK_FEATURES = ["price"]

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


def get_model(device, seed):
    if TabPFNClassifier is None:
        raise RuntimeError("TabPFNClassifier import failed. Run inside the fdHG TabPFN env.")
    try:
        return TabPFNClassifier(device=device, random_state=seed)
    except TypeError:
        return TabPFNClassifier(device=device)


def read_seed_table(seed_dir):
    path = Path(seed_dir) / "compiled_features.parquet"
    if path.exists():
        return pd.read_parquet(path)

    path = Path(seed_dir) / "compiled_features.csv"
    if path.exists():
        return pd.read_csv(path)

    candidates = list(Path(seed_dir).glob("*.parquet")) + list(Path(seed_dir).glob("*.csv"))
    for p in candidates:
        if p.name.startswith("compiled") or "feature" in p.name:
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p)

    raise FileNotFoundError(f"No compiled feature table found in {seed_dir}")


def find_target_col(df):
    for c in ["target", "label", "y", "WillGetBadge", "item_churn"]:
        if c in df.columns:
            return c
    raise ValueError(f"No target column found. Columns={list(df.columns)}")


def clean_xy(df, features, target_col):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features {missing}; available={list(df.columns)}")

    X = df[features].copy()
    y = df[target_col].astype(int).to_numpy()

    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c].astype(str), sort=True)[0]
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else 0)

    return X.to_numpy(dtype=np.float32), y


def split_train_val(df, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = min(500, max(1, int(len(df) * 0.25)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) > 2000:
        train_idx = train_idx[:2000]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def eval_features(seed_dir, features, device, model_seed):
    df = read_seed_table(seed_dir)
    target_col = find_target_col(df)

    train_df, val_df = split_train_val(df, model_seed)
    X_train, y_train = clean_xy(train_df, features, target_col)
    X_val, y_val = clean_xy(val_df, features, target_col)

    model = get_model(device, model_seed)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_val)[:, 1]
    else:
        pred = model.predict(X_val)
        prob = np.asarray(pred, dtype=float)

    pred_label = (prob >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_val, pred_label)),
        "roc_auc": float(roc_auc_score(y_val, prob)),
        "average_precision": float(average_precision_score(y_val, prob)),
        "log_loss": float(log_loss(y_val, prob, labels=[0, 1])),
    }


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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = pd.read_csv(args.ranked_features) if Path(args.ranked_features).exists() else pd.DataFrame()

    rows = []
    random_rows = []

    for seed_str in args.test_seeds:
        seed_dir = prior_root / f"minimal_seed{seed_str}"

        # ranker topK from ranked features, fallback to oracle if needed.
        selected = None
        if len(ranked):
            sub = ranked[
                ranked.astype(str).apply(lambda row: row.astype(str).str.contains(f"minimal_seed{seed_str}").any(), axis=1)
            ].copy()
            if len(sub) and "feature" in sub.columns:
                score_col = "ranker_score" if "ranker_score" in sub.columns else None
                if score_col:
                    sub = sub.sort_values(score_col, ascending=False)
                selected = sub["feature"].head(args.selected_k).tolist()

        if not selected:
            # fallback to known selected set from previous experiment
            selected = ORACLE_FEATURES[:args.selected_k]

        methods = {
            "weak_price_only": WEAK_FEATURES,
            f"ranker_top{args.selected_k}": selected,
            "oracle": ORACLE_FEATURES,
        }

        for method, features in methods.items():
            metrics = eval_features(seed_dir, features, args.device, args.seed)
            rows.append({
                "seed_dir": f"minimal_seed{seed_str}",
                "method": method,
                "n_features": len(features),
                "features": ",".join(features),
                **metrics,
            })

        # random selected programs: average n_random random topK selections per seed
        rng = np.random.default_rng(int(seed_str) + args.seed)
        random_metric_list = []
        for rep in range(args.n_random):
            features = list(rng.choice(ALL_CANDIDATE_FEATURES, size=args.selected_k, replace=False))
            metrics = eval_features(seed_dir, features, args.device, args.seed + rep + 1000)
            random_metric_list.append(metrics)
            random_rows.append({
                "seed_dir": f"minimal_seed{seed_str}",
                "random_rep": rep,
                "method": f"random_top{args.selected_k}",
                "n_features": len(features),
                "features": ",".join(features),
                **metrics,
            })

        random_mean = {
            m: float(np.mean([r[m] for r in random_metric_list]))
            for m in ["accuracy", "roc_auc", "average_precision", "log_loss"]
        }
        rows.append({
            "seed_dir": f"minimal_seed{seed_str}",
            "method": f"random_top{args.selected_k}",
            "n_features": args.selected_k,
            "features": f"mean_of_{args.n_random}_random_draws",
            **random_mean,
        })

    df = pd.DataFrame(rows)

    # Oracle gaps
    gap_rows = []
    for seed_dir, g in df.groupby("seed_dir"):
        oracle = g[g["method"] == "oracle"].iloc[0]
        for _, row in g.iterrows():
            if row["method"] == "oracle":
                continue
            gap_rows.append({
                "seed_dir": seed_dir,
                "method": row["method"],
                "log_loss_gap_vs_oracle": row["log_loss"] - oracle["log_loss"],
                "roc_auc_gap_vs_oracle": oracle["roc_auc"] - row["roc_auc"],
                "average_precision_gap_vs_oracle": oracle["average_precision"] - row["average_precision"],
            })

    gap_df = pd.DataFrame(gap_rows)
    random_detail_df = pd.DataFrame(random_rows)

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

    print("Saved:")
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
