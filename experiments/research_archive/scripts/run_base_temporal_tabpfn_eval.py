#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def make_target_base(df, label_col):
    base = df.drop(columns=[label_col], errors="ignore").copy()

    for c in list(base.columns):
        if np.issubdtype(base[c].dtype, np.datetime64):
            dt = pd.to_datetime(base[c])
            base[f"{c}__year"] = dt.dt.year
            base[f"{c}__month"] = dt.dt.month
            base[f"{c}__day"] = dt.dt.day
            base[f"{c}__dow"] = dt.dt.dayofweek
            base = base.drop(columns=[c])
        elif c.lower() in {"date", "timestamp", "time"}:
            dt = pd.to_datetime(base[c], errors="coerce")
            if dt.notna().any():
                base[f"{c}__year"] = dt.dt.year
                base[f"{c}__month"] = dt.dt.month
                base[f"{c}__day"] = dt.dt.day
                base[f"{c}__dow"] = dt.dt.dayofweek
                base = base.drop(columns=[c])

    return base


def clean_features(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    for c in X_train.columns:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")

    keep = [c for c in X_train.columns if not X_train[c].isna().all()]
    X_train = X_train[keep]
    X_val = X_val[keep]

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med).fillna(0)
    X_val = X_val.fillna(med).fillna(0)

    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_val = X_val.replace([np.inf, -np.inf], 0)

    return X_train.astype(np.float32), X_val.astype(np.float32), keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-train", required=True)
    parser.add_argument("--target-val", required=True)
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--base-val", required=True)
    parser.add_argument("--temporal-train", required=True)
    parser.add_argument("--temporal-val", required=True)
    parser.add_argument("--label-col", default="did_not_finish")
    parser.add_argument("--row-id-col", default="__target_row_id__")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-train", type=int, default=10000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    target_train = read_table(args.target_train)
    target_val = read_table(args.target_val)

    y_train = target_train[args.label_col].astype(int).to_numpy()
    y_val = target_val[args.label_col].astype(int).to_numpy()

    target_base_train = make_target_base(target_train, args.label_col)
    target_base_val = make_target_base(target_val, args.label_col)

    base_train = read_table(args.base_train)
    base_val = read_table(args.base_val)

    temporal_train = read_table(args.temporal_train).drop(columns=[args.row_id_col], errors="ignore")
    temporal_val = read_table(args.temporal_val).drop(columns=[args.row_id_col], errors="ignore")

    # The generated base files already include target columns, so keep only engineered features from them.
    drop_from_base = {
        args.label_col,
        "target",
        "date",
        "timestamp",
        "time",
        "driverId",
        args.row_id_col,
    }
    base_train = base_train.drop(columns=[c for c in drop_from_base if c in base_train.columns], errors="ignore")
    base_val = base_val.drop(columns=[c for c in drop_from_base if c in base_val.columns], errors="ignore")

    target_base_train = target_base_train.add_prefix("target::")
    target_base_val = target_base_val.add_prefix("target::")
    base_train = base_train.add_prefix("dfs::")
    base_val = base_val.add_prefix("dfs::")

    if not (len(target_base_train) == len(base_train) == len(temporal_train)):
        raise ValueError(
            f"Train row mismatch: target={len(target_base_train)}, "
            f"base={len(base_train)}, temporal={len(temporal_train)}"
        )

    if not (len(target_base_val) == len(base_val) == len(temporal_val)):
        raise ValueError(
            f"Val row mismatch: target={len(target_base_val)}, "
            f"base={len(base_val)}, temporal={len(temporal_val)}"
        )

    X_train = pd.concat(
        [
            target_base_train.reset_index(drop=True),
            base_train.reset_index(drop=True),
            temporal_train.reset_index(drop=True),
        ],
        axis=1,
    )

    X_val = pd.concat(
        [
            target_base_val.reset_index(drop=True),
            base_val.reset_index(drop=True),
            temporal_val.reset_index(drop=True),
        ],
        axis=1,
    )

    X_train, X_val, feature_cols = clean_features(X_train, X_val)

    if len(X_train) > args.max_train:
        idx = np.random.RandomState(args.seed).choice(len(X_train), size=args.max_train, replace=False)
        X_train_used = X_train.iloc[idx].reset_index(drop=True)
        y_train_used = y_train[idx]
    else:
        X_train_used = X_train.reset_index(drop=True)
        y_train_used = y_train

    X_val_used = X_val.reset_index(drop=True)

    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(random_state=args.seed)
    clf.fit(X_train_used, y_train_used)

    proba = clf.predict_proba(X_val_used)
    p1 = proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
    pred = (p1 >= 0.5).astype(int)

    metrics = {
        "variant": args.variant,
        "seed": args.seed,
        "n_train_used": int(len(X_train_used)),
        "n_val_used": int(len(X_val_used)),
        "n_features": int(X_train_used.shape[1]),
        "accuracy": float(accuracy_score(y_val, pred)),
        "roc_auc": float(roc_auc_score(y_val, p1)),
        "average_precision": float(average_precision_score(y_val, p1)),
        "log_loss": float(log_loss(y_val, proba, labels=[0, 1])),
        "feature_cols": feature_cols,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))

    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_cols"}, indent=2))
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
