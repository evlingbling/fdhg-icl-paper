from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sample_like(df, max_rows: int | None, seed: int):
    df = df.reset_index(drop=True)
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)


def get_tabpfn_classifier(device: str, seed: int):
    from tabpfn import TabPFNClassifier
    try:
        return TabPFNClassifier(device=device, random_state=seed)
    except TypeError:
        return TabPFNClassifier(device=device)


def eval_one(X_train, y_train, X_val, y_val, device: str, seed: int) -> dict:
    clf = get_tabpfn_classifier(device, seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    proba = clf.predict_proba(X_val)
    y_score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()

    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_score)),
        "average_precision": float(average_precision_score(y_val, y_score)),
        "log_loss": float(log_loss(y_val, y_score, labels=[0, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/target_only/rel-amazon/item-churn")
    parser.add_argument("--feature-dir", default="outputs/ambiguity_features/rel-amazon_item-churn_sample")
    parser.add_argument("--manifest", default="outputs/fdhg_heuristic/rel-amazon_item-churn_sample/fdhg_heuristic_feature_manifest.csv")
    parser.add_argument("--output-dir", default="results/budget_curve")
    parser.add_argument("--dataset", default="rel-amazon")
    parser.add_argument("--task", default="item-churn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-train-rows", type=int, default=10000)
    parser.add_argument("--max-val-rows", type=int, default=2000)
    parser.add_argument("--ks", nargs="*", type=int, default=[0, 5, 9, 13, 17])
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_full = load_pickle(artifact_dir / "X_train.pkl")
    X_val_full = load_pickle(artifact_dir / "X_val.pkl")
    y_train_full = load_pickle(artifact_dir / "y_train.pkl")
    y_val_full = load_pickle(artifact_dir / "y_val.pkl")

    X_train_base = sample_like(X_train_full, args.max_train_rows, args.seed)
    X_val_base = sample_like(X_val_full, args.max_val_rows, args.seed)
    y_train = sample_like(pd.Series(y_train_full), args.max_train_rows, args.seed).to_numpy().ravel()
    y_val = sample_like(pd.Series(y_val_full), args.max_val_rows, args.seed).to_numpy().ravel()

    train_feat = pd.read_parquet(feature_dir / "target_with_dfs_agg_amb_train.parquet")
    val_feat = pd.read_parquet(feature_dir / "target_with_dfs_agg_amb_val.parquet")

    manifest = pd.read_csv(args.manifest).sort_values("rank")
    ranked_features = [
        f for f in manifest["feature_name"].tolist()
        if f in train_feat.columns
    ]

    rows = []

    for k in args.ks:
        selected = ranked_features[:k]

        X_train = pd.concat(
            [X_train_base.reset_index(drop=True), train_feat[selected].reset_index(drop=True)],
            axis=1,
        )
        X_val = pd.concat(
            [X_val_base.reset_index(drop=True), val_feat[selected].reset_index(drop=True)],
            axis=1,
        )

        for df in [X_train, X_val]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print(f"\n=== K={k} generated features ===")
        print("selected:", selected)
        print("X_train:", X_train.shape)
        print("X_val:", X_val.shape)

        metrics = eval_one(X_train, y_train, X_val, y_val, args.device, args.seed)

        row = {
            "dataset": args.dataset,
            "task": args.task,
            "model": "tabpfn",
            "method": "fdhg_budget_curve",
            "seed": args.seed,
            "K_generated_features": k,
            "n_features_total": int(X_train.shape[1]),
            "selected_features": ",".join(selected),
            **metrics,
        }
        rows.append(row)

        print(json.dumps(metrics, indent=2))

    df = pd.DataFrame(rows)
    out_csv = output_dir / "fdhg_budget_curve_rel-amazon_item-churn.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Saved ===")
    print(out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
