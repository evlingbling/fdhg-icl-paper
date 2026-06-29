from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def infer_block(feature_name: str) -> str:
    if feature_name.startswith("f_amb__"):
        return "ambiguity"
    if "days_since_last" in feature_name:
        return "temporal_aggregation"
    if feature_name.startswith("f_review_"):
        return "aggregation"
    return "target_original"


def infer_aggregator(feature_name: str) -> str:
    if feature_name.endswith("__is_missing"):
        return "missing_indicator"
    if "count" in feature_name:
        return "count"
    if "mean" in feature_name:
        return "mean"
    if "std" in feature_name:
        return "std"
    if "max" in feature_name:
        return "max"
    if "days_since_last" in feature_name:
        return "last_delta"
    if "majconf" in feature_name:
        return "majority_confidence"
    if "entropy" in feature_name:
        return "entropy"
    if "conflict_count" in feature_name:
        return "conflict_count"
    if "support_count" in feature_name:
        return "support_count"
    return "unknown"


def cost_for_feature(feature_name: str) -> float:
    if feature_name.endswith("__is_missing"):
        return 0.05
    if "days_since_last" in feature_name:
        return 0.50
    if "std" in feature_name:
        return 0.35
    if "mean" in feature_name or "max" in feature_name:
        return 0.30
    if "count" in feature_name:
        return 0.20
    if feature_name.startswith("f_amb__"):
        return 0.20
    return 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-table",
        default="outputs/ambiguity_features/rel-amazon_item-churn_sample/target_with_dfs_agg_amb_train.parquet",
    )
    parser.add_argument(
        "--dfs-manifest",
        default="outputs/dfs_agg/rel-amazon_item-churn_sample/dfs_agg_feature_manifest.csv",
    )
    parser.add_argument(
        "--amb-manifest",
        default="outputs/ambiguity_features/rel-amazon_item-churn_sample/ambiguity_feature_manifest.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/fdhg_heuristic/rel-amazon_item-churn_sample",
    )
    parser.add_argument("--k", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.feature_table)
    feature_cols = [c for c in train.columns if c.startswith("f_")]

    dfs_manifest = pd.read_csv(args.dfs_manifest)
    amb_manifest = pd.read_csv(args.amb_manifest)

    amb_meta = {
        row["feature_name"]: row.to_dict()
        for _, row in amb_manifest.iterrows()
    }

    rows = []

    for feature in feature_cols:
        block = infer_block(feature)
        aggregator = infer_aggregator(feature)

        missing_rate = float(train[feature].isna().mean())
        coverage = 1.0 - missing_rate

        # Document-style heuristic components.
        if block in {"aggregation", "temporal_aggregation"}:
            reliability = 1.0
            ambiguity_bonus = 0.0
            uniqueness_penalty = 0.0
            path_length = 1.0
            leakage_risk = 0.0
            source = "review"
            edge_name = "product.product_id<-review.product_id"
        elif block == "ambiguity":
            base_feature = feature.replace("__is_missing", "")
            meta = amb_meta.get(base_feature, {})
            reliability = float(meta.get("edge_weight", 0.7))
            ambiguity_bonus = 0.5 if not feature.endswith("__is_missing") else 0.1
            uniqueness_penalty = float(meta.get("lhs_uniqueness", 0.0)) * 0.7
            path_length = 1.0
            leakage_risk = 0.0
            source = "product"
            edge_name = str(meta.get("edge_name", "brand->category"))
        else:
            reliability = 0.8
            ambiguity_bonus = 0.0
            uniqueness_penalty = 0.0
            path_length = 0.0
            leakage_risk = 0.0
            source = "target"
            edge_name = ""

        cost = cost_for_feature(feature)

        # Simple redundancy proxy:
        # Missing indicators get a small penalty because they duplicate the base feature's missingness.
        redundancy = 0.15 if feature.endswith("__is_missing") else 0.0

        score = (
            1.5 * reliability
            + 0.8 * np.log1p(max(coverage, 0.0))
            + ambiguity_bonus
            - uniqueness_penalty
            - 0.2 * path_length
            - 0.3 * cost
            - redundancy
            - leakage_risk
        )

        rows.append(
            {
                "feature_name": feature,
                "block": block,
                "source": source,
                "edge_or_path": edge_name,
                "aggregator": aggregator,
                "reliability": reliability,
                "coverage": coverage,
                "missing_rate": missing_rate,
                "ambiguity_bonus": ambiguity_bonus,
                "uniqueness_penalty": uniqueness_penalty,
                "path_length": path_length,
                "cost": cost,
                "redundancy_penalty": redundancy,
                "leakage_risk": leakage_risk,
                "heuristic_score": float(score),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values("heuristic_score", ascending=False).reset_index(drop=True)
    manifest["rank"] = np.arange(1, len(manifest) + 1)
    manifest["selected"] = manifest["rank"] <= args.k

    out_manifest = out_dir / "fdhg_heuristic_feature_manifest.csv"
    manifest.to_csv(out_manifest, index=False)

    selected = manifest[manifest["selected"]].copy()
    out_selected = out_dir / "selected_features.txt"
    out_selected.write_text("\n".join(selected["feature_name"].tolist()) + "\n")

    print("=== FDHG heuristic feature selection ===")
    print("K:", args.k)
    print("n_candidates:", len(manifest))
    print("n_selected:", len(selected))
    print("\nTop features:")
    print(
        manifest[
            [
                "rank",
                "feature_name",
                "block",
                "aggregator",
                "reliability",
                "coverage",
                "heuristic_score",
                "selected",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(out_manifest)
    print(out_selected)


if __name__ == "__main__":
    main()
