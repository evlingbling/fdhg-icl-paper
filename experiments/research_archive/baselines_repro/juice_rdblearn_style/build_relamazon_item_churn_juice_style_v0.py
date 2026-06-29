from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def read_first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return pd.read_parquet(p), p
    raise FileNotFoundError(paths)


def dt(s):
    return pd.to_datetime(s, errors="coerce")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def add_missing_indicators(df, feature_cols):
    for c in list(feature_cols):
        df[c + "__is_missing"] = df[c].isna().astype("int8")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-999.0)
    return df


def encode_product_base(target, product):
    out = target[["timestamp", "product_id", "churn"]].copy()

    p = product.copy()
    if "product_id" not in p.columns:
        raise KeyError(f"product_id not in product columns={list(p.columns)}")

    base = pd.DataFrame({"product_id": p["product_id"]})

    for c in ["brand", "category", "title", "description"]:
        if c in p.columns:
            base[f"juice_product_{c}_len"] = p[c].astype(str).str.len()
            base[f"juice_product_has_{c}"] = p[c].notna().astype("int8")

    for c in ["price", "rank"]:
        if c in p.columns:
            base[f"juice_product_{c}"] = num(p[c])

    out = out.merge(base, on="product_id", how="left")
    return out


def aggregate_reviews_by_product(target, review):
    target = target.copy()
    review = review.copy()
    target["timestamp"] = dt(target["timestamp"])
    review["review_time"] = dt(review["review_time"])
    review = review.dropna(subset=["product_id", "review_time"])

    groups = {k: g for k, g in review.groupby("product_id", sort=False)}

    rows = []
    prefix = "juice1_product_reviews"

    for _, r in target.iterrows():
        pid = r["product_id"]
        cutoff = r["timestamp"]

        row = {
            f"{prefix}_count": 0.0,
            f"{prefix}_customer_nunique": 0.0,
            f"{prefix}_rating_mean": np.nan,
            f"{prefix}_rating_std": np.nan,
            f"{prefix}_rating_max": np.nan,
            f"{prefix}_rating_nunique": 0.0,
            f"{prefix}_verified_mean": np.nan,
            f"{prefix}_days_since_last": np.nan,
        }

        if pd.isna(pid) or pd.isna(cutoff) or pid not in groups:
            rows.append(row)
            continue

        g = groups[pid]
        g = g[g["review_time"] < cutoff]

        row[f"{prefix}_count"] = float(len(g))
        if len(g):
            if "customer_id" in g.columns:
                row[f"{prefix}_customer_nunique"] = float(g["customer_id"].nunique(dropna=True))

            if "rating" in g.columns:
                vals = num(g["rating"])
                row[f"{prefix}_rating_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                row[f"{prefix}_rating_std"] = float(vals.std()) if vals.notna().sum() > 1 else 0.0
                row[f"{prefix}_rating_max"] = float(vals.max()) if vals.notna().any() else np.nan
                row[f"{prefix}_rating_nunique"] = float(vals.nunique(dropna=True))

            if "verified" in g.columns:
                row[f"{prefix}_verified_mean"] = float(g["verified"].astype(float).mean())

            last_time = g["review_time"].max()
            if pd.notna(last_time):
                row[f"{prefix}_days_since_last"] = float((cutoff - last_time).total_seconds() / 86400.0)

        rows.append(row)

    feat = pd.DataFrame(rows)
    return add_missing_indicators(feat, list(feat.columns))


def aggregate_customer_review_history(target, review, max_customers_per_product=500):
    """
    Meta-path:
    product_id -> review.product_id -> customer_id -> review.customer_id
    PK/FK-style traversal only.
    """
    target = target.copy()
    review = review.copy()
    target["timestamp"] = dt(target["timestamp"])
    review["review_time"] = dt(review["review_time"])
    review = review.dropna(subset=["product_id", "customer_id", "review_time"])

    by_product = {k: g for k, g in review.groupby("product_id", sort=False)}
    by_customer = {k: g for k, g in review.groupby("customer_id", sort=False)}

    rows = []
    prefix = "juice2_product_reviewers_history"

    for _, r in target.iterrows():
        pid = r["product_id"]
        cutoff = r["timestamp"]

        row = {
            f"{prefix}_reviewer_nunique": 0.0,
            f"{prefix}_history_count": 0.0,
            f"{prefix}_rating_mean": np.nan,
            f"{prefix}_rating_std": np.nan,
            f"{prefix}_rating_max": np.nan,
            f"{prefix}_rating_nunique": 0.0,
            f"{prefix}_reviewed_product_nunique": 0.0,
        }

        if pd.isna(pid) or pd.isna(cutoff) or pid not in by_product:
            rows.append(row)
            continue

        pg = by_product[pid]
        pg = pg[pg["review_time"] < cutoff]
        customers = pg["customer_id"].dropna().unique()
        if len(customers) > max_customers_per_product:
            customers = customers[:max_customers_per_product]

        row[f"{prefix}_reviewer_nunique"] = float(len(customers))

        parts = []
        for cid in customers:
            cg = by_customer.get(cid)
            if cg is None:
                continue
            cg = cg[cg["review_time"] < cutoff]
            if not cg.empty:
                parts.append(cg)

        if parts:
            h = pd.concat(parts, ignore_index=True)
            row[f"{prefix}_history_count"] = float(len(h))
            if "rating" in h.columns:
                vals = num(h["rating"])
                row[f"{prefix}_rating_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                row[f"{prefix}_rating_std"] = float(vals.std()) if vals.notna().sum() > 1 else 0.0
                row[f"{prefix}_rating_max"] = float(vals.max()) if vals.notna().any() else np.nan
                row[f"{prefix}_rating_nunique"] = float(vals.nunique(dropna=True))
            if "product_id" in h.columns:
                row[f"{prefix}_reviewed_product_nunique"] = float(h["product_id"].nunique(dropna=True))

        rows.append(row)

    feat = pd.DataFrame(rows)
    return add_missing_indicators(feat, list(feat.columns))


def build_split(inspect_root, split, out_path, sample_target_root=None):
    inspect_root = Path(inspect_root)

    sampled_path = None
    if sample_target_root is not None:
        sampled_path = Path(sample_target_root) / f"target_with_dfs_agg_{split}.parquet"

    if sampled_path is not None and sampled_path.exists():
        target_full = pd.read_parquet(sampled_path)
        keep_cols = [c for c in ["timestamp", "product_id", "churn"] if c in target_full.columns]
        target = target_full[keep_cols].copy()
        print(f"[TARGET] using sampled target: {sampled_path} {target.shape}")
    else:
        target = pd.read_parquet(inspect_root / f"target_{split}.parquet")
        print(f"[TARGET] using full target: {inspect_root / f'target_{split}.parquet'} {target.shape}")

    # test split may not have label; for eval we only use train/val.
    if "churn" not in target.columns:
        keep = ["timestamp", "product_id"]
        target = target[keep].copy()
        target["churn"] = np.nan

    review, rp = read_first_existing([
        inspect_root / "table_review.parquet",
        inspect_root / "review.parquet",
    ])
    product, pp = read_first_existing([
        inspect_root / "table_product.parquet",
        inspect_root / "table_products.parquet",
        inspect_root / "product.parquet",
        inspect_root / "products.parquet",
    ])

    print(f"[TABLE] review: {rp} {review.shape}")
    print(f"[TABLE] product: {pp} {product.shape}")

    out = encode_product_base(target, product)
    blocks = [
        aggregate_reviews_by_product(out[["timestamp", "product_id"]], review),
        aggregate_customer_review_history(out[["timestamp", "product_id"]], review),
    ]

    for b in blocks:
        out = pd.concat([out.reset_index(drop=True), b.reset_index(drop=True)], axis=1)

    forbidden = [
        c for c in out.columns
        if c.startswith("f_amb__")
        or "amb__" in c
        or "afd" in c.lower()
        or "fdhg::" in c
        or "ranker" in c.lower()
        or "uniqueness_penalty" in c.lower()
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden FDHG-specific columns found: {forbidden}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print("[WROTE]", out_path, out.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-root", default="outputs/relbench_inspect/rel-amazon_item-churn")
    ap.add_argument("--out-root", default="outputs/phase1_juice_style_matched/rel-amazon_item-churn")
    ap.add_argument("--sample-target-root", default="outputs/dfs_agg/rel-amazon_item-churn_sample")
    args = ap.parse_args()

    for split in ["train", "val"]:
        build_split(
            args.inspect_root,
            split,
            Path(args.out_root) / f"target_with_juice_style_{split}.parquet",
            sample_target_root=args.sample_target_root,
        )


if __name__ == "__main__":
    main()
