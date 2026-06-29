from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_zip_city(rng: np.random.Generator, n_users: int, fd_noise: float) -> tuple[np.ndarray, np.ndarray]:
    zips = np.array([f"zip_{i:03d}" for i in range(80)])
    cities = np.array([f"city_{i // 4:02d}" for i in range(80)])

    user_zip_idx = rng.integers(0, len(zips), size=n_users)
    user_zips = zips[user_zip_idx]
    user_cities = cities[user_zip_idx]

    # Inject FD violations for zip -> city.
    noisy = rng.random(n_users) < fd_noise
    user_cities[noisy] = rng.choice(cities, size=noisy.sum())

    return user_zips, user_cities


def sample_brand_category(
    rng: np.random.Generator,
    n_products: int,
    fd_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_brands = 250
    n_categories = 25

    brands = np.array([f"brand_{i:03d}" for i in range(n_brands)])
    brand_to_category = {
        b: f"cat_{i % n_categories:02d}"
        for i, b in enumerate(brands)
    }

    # Zipf-like brand popularity so some brands repeat and some are rare.
    probs = np.arange(1, n_brands + 1, dtype=float)
    probs = 1.0 / probs
    probs = probs / probs.sum()

    product_brands = rng.choice(brands, size=n_products, p=probs)
    product_categories = np.array([brand_to_category[b] for b in product_brands])

    # Inject FD violations for brand -> category.
    noisy = rng.random(n_products) < fd_noise
    all_categories = np.array([f"cat_{i:02d}" for i in range(n_categories)])
    product_categories[noisy] = rng.choice(all_categories, size=noisy.sum())

    return product_brands, product_categories


def compute_brand_ambiguity(products: pd.DataFrame) -> pd.DataFrame:
    counts = (
        products.groupby(["brand", "category"])
        .size()
        .rename("n")
        .reset_index()
    )

    rows = []
    for brand, g in counts.groupby("brand"):
        total = g["n"].sum()
        p = g["n"].to_numpy(dtype=float) / total
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        majconf = float(g["n"].max() / total)
        conflict_count = int(g["category"].nunique())
        rows.append(
            {
                "brand": brand,
                "brand_category_majconf": majconf,
                "brand_category_entropy": entropy,
                "brand_category_conflict_count": conflict_count,
                "brand_category_support_count": int(total),
            }
        )

    return pd.DataFrame(rows)


def build_target_table(
    rng: np.random.Generator,
    products: pd.DataFrame,
    events: pd.DataFrame,
    cutoff_day: int,
    split: str,
) -> pd.DataFrame:
    target = products[["product_id", "brand", "category", "price"]].copy()
    target["timestamp"] = cutoff_day
    target["split"] = split

    hist = events[events["event_time"] <= cutoff_day].copy()

    agg = (
        hist.groupby("product_id")
        .agg(
            event_count=("event_id", "count"),
            rating_mean=("rating", "mean"),
            amount_mean=("amount", "mean"),
            last_event_time=("event_time", "max"),
        )
        .reset_index()
    )

    target = target.merge(agg, on="product_id", how="left")
    target["event_count"] = target["event_count"].fillna(0)
    target["rating_mean"] = target["rating_mean"].fillna(0.0)
    target["amount_mean"] = target["amount_mean"].fillna(0.0)
    target["days_since_last_event"] = cutoff_day - target["last_event_time"]
    target["days_since_last_event"] = target["days_since_last_event"].fillna(999.0)

    amb = compute_brand_ambiguity(products)
    target = target.merge(amb, on="brand", how="left")

    # Label-generating process.
    # Higher recent activity and higher rating => less churn.
    # Higher ambiguity entropy and higher price => slightly more churn.
    log_event = np.log1p(target["event_count"].to_numpy(dtype=float))
    rating = target["rating_mean"].to_numpy(dtype=float)
    entropy = target["brand_category_entropy"].to_numpy(dtype=float)
    price = target["price"].to_numpy(dtype=float)

    eta = (
        1.2
        - 0.75 * log_event
        - 0.35 * rating
        + 0.40 * entropy
        + 0.008 * price
    )

    p_churn = sigmoid(eta)
    target["churn_prob"] = p_churn
    target["churn"] = rng.binomial(1, p_churn)

    return target[
        [
            "timestamp",
            "product_id",
            "churn",
            "churn_prob",
            "event_count",
            "rating_mean",
            "amount_mean",
            "days_since_last_event",
            "brand_category_majconf",
            "brand_category_entropy",
            "brand_category_conflict_count",
            "brand_category_support_count",
        ]
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/synthetic_prior/minimal_seed41")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n-users", type=int, default=3000)
    parser.add_argument("--n-products", type=int, default=1500)
    parser.add_argument("--n-events", type=int, default=30000)
    parser.add_argument("--zip-city-noise", type=float, default=0.05)
    parser.add_argument("--brand-category-noise", type=float, default=0.10)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_zips, user_cities = sample_zip_city(
        rng,
        n_users=args.n_users,
        fd_noise=args.zip_city_noise,
    )

    users = pd.DataFrame(
        {
            "user_id": np.arange(args.n_users),
            "zip": user_zips,
            "city": user_cities,
            "signup_day": rng.integers(0, 500, size=args.n_users),
        }
    )

    product_brands, product_categories = sample_brand_category(
        rng,
        n_products=args.n_products,
        fd_noise=args.brand_category_noise,
    )

    products = pd.DataFrame(
        {
            "product_id": np.arange(args.n_products),
            "brand": product_brands,
            "category": product_categories,
            "price": np.round(rng.lognormal(mean=3.0, sigma=0.7, size=args.n_products), 2),
        }
    )

    # Events: one event table.
    # Product popularity follows a Zipf-like distribution.
    product_probs = np.arange(1, args.n_products + 1, dtype=float)
    product_probs = 1.0 / product_probs
    product_probs = product_probs / product_probs.sum()

    event_product = rng.choice(products["product_id"].to_numpy(), size=args.n_events, p=product_probs)
    event_user = rng.choice(users["user_id"].to_numpy(), size=args.n_events)

    # Rating depends weakly on product category and price.
    base_rating = rng.normal(loc=4.0, scale=0.8, size=args.n_events)
    base_rating = np.clip(base_rating, 1.0, 5.0)

    events = pd.DataFrame(
        {
            "event_id": np.arange(args.n_events),
            "user_id": event_user,
            "product_id": event_product,
            "rating": np.round(base_rating, 2),
            "amount": np.round(rng.lognormal(mean=3.2, sigma=0.8, size=args.n_events), 2),
            "event_time": rng.integers(0, 1000, size=args.n_events),
        }
    ).sort_values("event_time").reset_index(drop=True)

    train_target = build_target_table(rng, products, events, cutoff_day=650, split="train")
    val_target = build_target_table(rng, products, events, cutoff_day=800, split="val")
    test_target = build_target_table(rng, products, events, cutoff_day=950, split="test")

    # Save DB tables.
    users.to_parquet(out_dir / "table_users.parquet", index=False)
    products.to_parquet(out_dir / "table_products.parquet", index=False)
    events.to_parquet(out_dir / "table_events.parquet", index=False)

    # Save target tables.
    train_target.to_parquet(out_dir / "target_train.parquet", index=False)
    val_target.to_parquet(out_dir / "target_val.parquet", index=False)
    test_target.drop(columns=["churn", "churn_prob"]).to_parquet(out_dir / "target_test.parquet", index=False)

    # Ground truth metadata.
    ground_truth = {
        "seed": args.seed,
        "tables": {
            "users": {
                "primary_key": "user_id",
                "columns": users.columns.tolist(),
            },
            "products": {
                "primary_key": "product_id",
                "columns": products.columns.tolist(),
            },
            "events": {
                "primary_key": "event_id",
                "foreign_keys": {
                    "user_id": "users.user_id",
                    "product_id": "products.product_id",
                },
                "time_col": "event_time",
                "columns": events.columns.tolist(),
            },
        },
        "ground_truth_fds": [
            {
                "lhs": "zip",
                "rhs": "city",
                "table": "users",
                "type": "approx_fd",
                "injected_noise": args.zip_city_noise,
            },
            {
                "lhs": "brand",
                "rhs": "category",
                "table": "products",
                "type": "approx_fd",
                "injected_noise": args.brand_category_noise,
            },
        ],
        "label_generating_programs": [
            "product_id <- events.product_id :: count(event_id) before cutoff",
            "product_id <- events.product_id :: mean(rating) before cutoff",
            "products.brand -> products.category :: ambiguity_entropy",
            "products.price :: identity",
        ],
        "splits": {
            "train_cutoff_day": 650,
            "val_cutoff_day": 800,
            "test_cutoff_day": 950,
        },
    }

    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    summary = {
        "users": users.shape,
        "products": products.shape,
        "events": events.shape,
        "target_train": train_target.shape,
        "target_val": val_target.shape,
        "target_test": test_target.drop(columns=["churn", "churn_prob"]).shape,
        "train_churn_rate": float(train_target["churn"].mean()),
        "val_churn_rate": float(val_target["churn"].mean()),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Synthetic prior task generated ===")
    print("out_dir:", out_dir)
    print(json.dumps(summary, indent=2))
    print("\nGround-truth FDs:")
    for fd in ground_truth["ground_truth_fds"]:
        print(" -", fd)


if __name__ == "__main__":
    main()
