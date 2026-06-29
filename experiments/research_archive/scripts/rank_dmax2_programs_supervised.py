import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_classif


def read_split(input_dir, split):
    p = Path(input_dir) / f"{split}_combined.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def is_dmax2_col(c):
    c = str(c)
    return c.startswith("dmax2_") or c.startswith("dmax2__")


def stratified_sample_indices(y, max_rows, seed):
    if max_rows is None or max_rows <= 0 or len(y) <= max_rows:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)

    idxs = []
    classes, counts = np.unique(y_arr, return_counts=True)

    for cls, count in zip(classes, counts):
        cls_idx = np.where(y_arr == cls)[0]
        n_take = max(1, int(round(max_rows * count / len(y_arr))))
        n_take = min(n_take, len(cls_idx))
        idxs.extend(rng.choice(cls_idx, size=n_take, replace=False).tolist())

    if len(idxs) > max_rows:
        idxs = rng.choice(idxs, size=max_rows, replace=False).tolist()

    rng.shuffle(idxs)
    return np.asarray(idxs)


def numeric_clean(s):
    s = s.copy()
    if pd.api.types.is_bool_dtype(s):
        s = s.astype(int)
    elif not pd.api.types.is_numeric_dtype(s):
        s = pd.factorize(s.astype(str), sort=True)[0]
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).fillna(0)
    return s


def score_feature(x_raw, y, rank_metric):
    x = numeric_clean(x_raw)

    nunique = int(pd.Series(x).nunique(dropna=False))
    nonnull_rate = float(pd.Series(x_raw).notna().mean())

    if nunique <= 1:
        return {
            "roc_auc": np.nan,
            "roc_auc_oriented": np.nan,
            "average_precision": np.nan,
            "mutual_info": np.nan,
            "score": -1.0,
            "nunique": nunique,
            "nonnull_rate": nonnull_rate,
        }

    try:
        auc = float(roc_auc_score(y, x))
        auc_oriented = max(auc, 1.0 - auc)
    except Exception:
        auc = np.nan
        auc_oriented = np.nan

    try:
        ap = float(average_precision_score(y, x))
    except Exception:
        ap = np.nan

    try:
        mi = float(mutual_info_classif(
            np.asarray(x).reshape(-1, 1),
            np.asarray(y),
            discrete_features=False,
            random_state=0
        )[0])
    except Exception:
        mi = np.nan

    if rank_metric == "auc":
        score = -1.0 if np.isnan(auc) else 2.0 * abs(auc - 0.5)
    elif rank_metric == "ap":
        score = -1.0 if np.isnan(ap) else ap
    elif rank_metric == "mi":
        score = -1.0 if np.isnan(mi) else mi
    else:
        raise ValueError(f"Unknown rank_metric={rank_metric}")

    return {
        "roc_auc": auc,
        "roc_auc_oriented": auc_oriented,
        "average_precision": ap,
        "mutual_info": mi,
        "score": float(score),
        "nunique": nunique,
        "nonnull_rate": nonnull_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--target_col", default="WillGetBadge")
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max_rank_rows", type=int, default=200000)
    parser.add_argument("--rank_metric", choices=["auc", "ap", "mi"], default="auc")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = read_split(args.input_dir, "train")
    val_df = read_split(args.input_dir, "val")
    test_df = read_split(args.input_dir, "test")

    if args.target_col not in train_df.columns:
        raise ValueError(f"target_col={args.target_col} not in train columns")

    dmax2_cols = [c for c in train_df.columns if is_dmax2_col(c)]
    if not dmax2_cols:
        raise ValueError("No dmax2 columns found.")

    y_full = train_df[args.target_col].astype(int).reset_index(drop=True)
    idx = stratified_sample_indices(y_full, args.max_rank_rows, args.seed)

    y = y_full.iloc[idx].reset_index(drop=True)
    rank_df = train_df.iloc[idx].reset_index(drop=True)

    print("Input dir:", args.input_dir)
    print("Rank metric:", args.rank_metric)
    print("Train rows for ranking:", len(rank_df))
    print("Candidate dmax2 features:", len(dmax2_cols))
    print("Label counts:")
    print(y.value_counts(dropna=False))

    rows = []
    for c in dmax2_cols:
        sc = score_feature(rank_df[c], y, args.rank_metric)
        row = {"feature": c}
        row.update(sc)
        rows.append(row)

    score_df = pd.DataFrame(rows)

    tie_cols = {
        "auc": ["score", "roc_auc_oriented", "average_precision", "mutual_info", "nunique"],
        "ap": ["score", "roc_auc_oriented", "mutual_info", "nunique"],
        "mi": ["score", "roc_auc_oriented", "average_precision", "nunique"],
    }[args.rank_metric]

    score_df = score_df.sort_values(
        tie_cols,
        ascending=[False] * len(tie_cols),
    ).reset_index(drop=True)

    selected = score_df.head(args.top_k)["feature"].tolist()

    print("\n=== SELECTED FEATURES ===")
    for i, c in enumerate(selected):
        r = score_df[score_df["feature"] == c].iloc[0]
        print(
            f"{i+1:02d}. {c} "
            f"score={r['score']:.6f} auc={r['roc_auc']:.6f} "
            f"auc_oriented={r['roc_auc_oriented']:.6f} "
            f"ap={r['average_precision']:.6f} mi={r['mutual_info']:.6f}"
        )

    score_df.to_csv(Path(args.out_dir) / "rank_scores.csv", index=False)

    manifest = {
        "input_dir": args.input_dir,
        "target_col": args.target_col,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_rank_rows": args.max_rank_rows,
        "rank_metric": args.rank_metric,
        "n_candidate_features": len(dmax2_cols),
        "selected_features": selected,
    }
    with open(Path(args.out_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    task_cols = [c for c in ["timestamp", "UserId", args.target_col] if c in train_df.columns]
    test_task_cols = [c for c in ["timestamp", "UserId", args.target_col] if c in test_df.columns]

    for split, df, cols in [
        ("train", train_df, task_cols),
        ("val", val_df, task_cols),
        ("test", test_df, test_task_cols),
    ]:
        keep_cols = cols + selected
        keep_cols = [c for c in keep_cols if c in df.columns]
        out = df[keep_cols].copy()
        out_path = Path(args.out_dir) / f"{split}_combined.parquet"
        out.to_parquet(out_path, index=False)
        print(f"Saved {split}: {out.shape} -> {out_path}")

    print("\nDone:", args.out_dir)


if __name__ == "__main__":
    main()
