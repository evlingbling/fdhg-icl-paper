#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, accuracy_score


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def preprocess_train_val(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Convert datetimes if any leaked into features.
    for c in X_train.columns:
        if np.issubdtype(X_train[c].dtype, np.datetime64):
            X_train[c] = X_train[c].astype("int64") / 1e9
            X_val[c] = X_val[c].astype("int64") / 1e9

    # Coerce all to numeric.
    for c in X_train.columns:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")

    # Drop columns all-NaN in train.
    keep_cols = [c for c in X_train.columns if not X_train[c].isna().all()]
    X_train = X_train[keep_cols]
    X_val = X_val[keep_cols]

    # Median impute using train only.
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med).fillna(0)
    X_val = X_val.fillna(med).fillna(0)

    # Replace inf.
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_val = X_val.replace([np.inf, -np.inf], 0)

    return X_train.astype(np.float32), X_val.astype(np.float32), keep_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-train", required=True)
    parser.add_argument("--target-val", required=True)
    parser.add_argument("--temporal-train", required=True)
    parser.add_argument("--temporal-val", required=True)
    parser.add_argument("--label-col", default="did_not_finish")
    parser.add_argument("--variant", default="target_temporal_smoke")
    parser.add_argument("--row-id-col", default="__target_row_id__")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-train", type=int, default=10000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    y_train_df = read_table(args.target_train)
    y_val_df = read_table(args.target_val)

    temporal_train = read_table(args.temporal_train)
    temporal_val = read_table(args.temporal_val)

    if args.label_col not in y_train_df.columns:
        raise ValueError(f"Missing label col in train target: {args.label_col}")
    if args.label_col not in y_val_df.columns:
        raise ValueError(f"Missing label col in val target: {args.label_col}")

    y_train = y_train_df[args.label_col].astype(int).to_numpy()
    y_val = y_val_df[args.label_col].astype(int).to_numpy()

    X_train = temporal_train.drop(columns=[args.row_id_col], errors="ignore")
    X_val = temporal_val.drop(columns=[args.row_id_col], errors="ignore")

    X_train, X_val, feature_cols = preprocess_train_val(X_train, X_val)

    # Match previous TabPFN budget if train is larger than max_train.
    if len(X_train) > args.max_train:
        idx = np.random.RandomState(args.seed).choice(len(X_train), size=args.max_train, replace=False)
        X_train_used = X_train.iloc[idx].reset_index(drop=True)
        y_train_used = y_train[idx]
    else:
        X_train_used = X_train.reset_index(drop=True)
        y_train_used = y_train

    X_val_used = X_val.reset_index(drop=True)

    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        raise RuntimeError(
            "Could not import TabPFNClassifier. Make sure tabpfn is installed in this env."
        ) from e

    clf = TabPFNClassifier(random_state=args.seed)
    clf.fit(X_train_used, y_train_used)

    proba = clf.predict_proba(X_val_used)
    if proba.shape[1] == 2:
        p1 = proba[:, 1]
    else:
        p1 = proba[:, -1]

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
