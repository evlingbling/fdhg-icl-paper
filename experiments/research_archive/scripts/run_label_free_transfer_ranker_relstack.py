import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def label_free_score(row):
    name = str(row["feature"])
    low = name.lower()

    score = 0.0

    # Operator preference: robust relational aggregation-like features.
    if "count_distinct" in low:
        score += 2.0
    if "nunique" in low:
        score += 1.8
    if "__count" in low or low.endswith("_count"):
        score += 1.2
    if "__mean" in low:
        score += 0.8
    if "__max" in low or "__min" in low:
        score += 0.5

    # Path/source preference for rel-stack.
    # This is label-free but schema-aware.
    if "posthistory" in low:
        score += 1.0
    if "comments" in low:
        score += 0.8
    if "posts" in low:
        score += 0.6
    if "userid" in low:
        score += 0.5
    if "postid" in low:
        score += 0.4

    # Attribute preference.
    if "body" in low:
        score += 0.6
    if "title" in low:
        score += 0.5
    if "tags" in low:
        score += 0.5

    # Penalize weak/generic metadata-like columns.
    if "contentlicense" in low:
        score -= 0.8
    if "ownerdisplayname" in low:
        score -= 0.6
    if "id_payload" in low or "row_id" in low:
        score -= 2.0

    # Non-null coverage.
    nonnull = float(row.get("nonnull_rate", 1.0))
    score += 1.0 * nonnull

    # Cardinality shape: avoid constants and huge unique IDs.
    nunique = float(row.get("nunique", 0.0))
    if nunique <= 1:
        score -= 2.0
    elif 2 <= nunique <= 500:
        score += 0.7
    elif nunique <= 5000:
        score += 0.2
    else:
        score -= 0.5

    # Mild diversity proxy via shorter path penalty.
    score -= 0.02 * len(name.split("_"))

    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument(
        "--rank-scores",
        default="results/extension_c_ranker/features/rel_stack_user_badge_dmax2_supervised_ap_topk16_seed41/rank_scores.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="results/label_free_transfer_ranker/rel_stack_user_badge_dmax2_label_free_topk16_seed41",
    )
    args = ap.parse_args()

    rank_path = Path(args.rank_scores)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(rank_path)

    # Keep label-derived supervised metrics for analysis, but do NOT use them in label_free_score.
    df["label_free_score"] = df.apply(label_free_score, axis=1)
    df = df.sort_values(
        ["label_free_score", "nonnull_rate", "nunique"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df["label_free_rank"] = np.arange(1, len(df) + 1)
    df["label_free_selected"] = df["label_free_rank"] <= args.k

    selected = df[df["label_free_selected"]].copy()

    score_out = out_dir / "label_free_rank_scores.csv"
    selected_out = out_dir / "selected_features.txt"
    manifest_out = out_dir / "manifest.json"

    df.to_csv(score_out, index=False)
    selected["feature"].to_csv(selected_out, index=False, header=False)

    manifest = {
        "method": "label_free_transfer_ranker",
        "description": (
            "Schema/name/coverage/cardinality based label-free transfer policy. "
            "Does not use RelBench train labels, AUROC, AP, or mutual information for ranking."
        ),
        "seed": args.seed,
        "k": args.k,
        "rank_scores_input": str(rank_path),
        "score_output": str(score_out),
        "selected_features_output": str(selected_out),
        "selected_features": selected["feature"].tolist(),
    }
    json.dump(manifest, open(manifest_out, "w"), indent=2)

    print("=== LABEL-FREE TOPK ===")
    print(selected[["label_free_rank", "feature", "label_free_score", "nunique", "nonnull_rate"]].to_string(index=False))
    print("\nSaved:")
    print(score_out)
    print(selected_out)
    print(manifest_out)

    # Optional diagnostic only: how label-free selection scores under hidden task-local labels.
    # This is for audit/reporting, not for ranking.
    metric_cols = [c for c in ["roc_auc", "average_precision", "mutual_info", "score"] if c in selected.columns]
    if metric_cols:
        print("\n=== DIAGNOSTIC ONLY: task-local utility of selected features ===")
        print(selected[metric_cols].mean().to_string())


if __name__ == "__main__":
    main()
