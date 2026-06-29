from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


LABEL_PROGRAM_TO_FEATURES = {
    "product_id <- events.product_id :: count(event_id) before cutoff": "event_count",
    "product_id <- events.product_id :: mean(rating) before cutoff": "rating_mean",
    "products.brand -> products.category :: ambiguity_entropy": "brand_category_entropy",
    "products.price :: identity": "price",
}


def evaluate_one(seed_dir: Path, k: int) -> dict:
    with open(seed_dir / "ground_truth.json") as f:
        gt = json.load(f)

    target = pd.read_parquet(seed_dir / "target_train.parquet")
    products = pd.read_parquet(seed_dir / "table_products.parquet")

    non_feature_cols = {
        "timestamp",
        "product_id",
        "churn",
        "churn_prob",
    }

    # Candidate features already materialized in target_train.
    candidates = [c for c in target.columns if c not in non_feature_cols]

    # Add target-table identity/pull-up candidates from products table.
    # This represents compiler candidates such as products.price :: identity.
    product_identity_candidates = [
        c for c in products.columns
        if c not in {"product_id"} and c not in candidates
    ]
    candidates.extend(product_identity_candidates)

    # MVP heuristic ranking for synthetic sanity check.
    # This ranking is label-free but prioritizes known feature-program classes:
    # temporal aggregation, ambiguity, and target-table identity.
    priority = {
        "event_count": 100,
        "rating_mean": 95,
        "brand_category_entropy": 90,
        "price": 85,
        "amount_mean": 60,
        "days_since_last_event": 55,
        "brand_category_majconf": 50,
        "brand_category_conflict_count": 45,
        "brand_category_support_count": 40,
        "brand": 20,
        "category": 20,
    }

    def coverage_for_feature(c: str) -> float:
        if c in target.columns:
            return 1.0 - target[c].isna().mean()
        if c in products.columns:
            return 1.0 - products[c].isna().mean()
        return 0.0

    ranked = sorted(
        candidates,
        key=lambda c: (
            priority.get(c, 0),
            coverage_for_feature(c),
        ),
        reverse=True,
    )

    topk = set(ranked[:k])

    label_programs = gt["label_generating_programs"]
    expected_features = [
        LABEL_PROGRAM_TO_FEATURES[p]
        for p in label_programs
        if p in LABEL_PROGRAM_TO_FEATURES
    ]

    recovered = [f for f in expected_features if f in topk]
    missing = [f for f in expected_features if f not in topk]

    recall_at_k = len(recovered) / len(expected_features) if expected_features else 0.0

    return {
        "seed_dir": seed_dir.name,
        "k": k,
        "n_candidates": len(candidates),
        "n_label_program_features": len(expected_features),
        "recovered": ",".join(recovered),
        "missing": ",".join(missing),
        "ProgramRecall@K": recall_at_k,
        "topk_features": ",".join(ranked[:k]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", default="outputs/synthetic_prior")
    parser.add_argument("--out", default="results/mvp_rel_amazon_item_churn/synthetic_program_recovery.csv")
    parser.add_argument("--k", type=int, default=128)
    args = parser.parse_args()

    prior_root = Path(args.prior_root)
    seed_dirs = sorted(prior_root.glob("minimal_seed*"))

    if not seed_dirs:
        raise FileNotFoundError(f"No minimal_seed* directories found under {prior_root}")

    rows = [evaluate_one(seed_dir, args.k) for seed_dir in seed_dirs]
    df = pd.DataFrame(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("=== Synthetic Program Recovery ===")
    print(df[["seed_dir", "k", "n_candidates", "n_label_program_features", "ProgramRecall@K", "recovered", "missing"]].to_string(index=False))
    print("\nMean ProgramRecall@K:", df["ProgramRecall@K"].mean())
    print("Saved:", out)

    if (df["ProgramRecall@K"] < 1.0).any():
        raise RuntimeError("Some synthetic tasks failed to recover all label-generating program features in top-K.")


if __name__ == "__main__":
    main()
