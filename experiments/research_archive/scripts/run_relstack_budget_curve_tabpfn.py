from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score


TARGET_CANDIDATES = ["target", "WillGetBadge", "label", "y"]


def get_tabpfn_classifier(device: str, seed: int):
    from tabpfn import TabPFNClassifier
    try:
        return TabPFNClassifier(device=device, random_state=seed)
    except TypeError:
        return TabPFNClassifier(device=device)


def find_target_col(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No target column found. Columns={list(df.columns)}")


def sample_df_xy(df: pd.DataFrame, target_col: str, max_rows: int | None, seed: int):
    df = df.reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
    y = df[target_col].astype(int).to_numpy().ravel()
    return df, y


def clean_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()

    for c in X.columns:
        # Convert all feature columns to float-safe numeric arrays before filling.
        # This avoids pandas nullable Int64 fillna errors when the median is non-integer.
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c].astype(str), sort=True)[0]

        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)

        if X[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(0.0 if pd.isna(med) else float(med))

    return X.astype(np.float32)


def eval_one(X_train, y_train, X_val, y_val, device: str, seed: int):
    clf = get_tabpfn_classifier(device, seed)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    proba = clf.predict_proba(X_val)
    score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()

    return {
        "accuracy": float(accuracy_score(y_val, pred)),
        "roc_auc": float(roc_auc_score(y_val, score)),
        "average_precision": float(average_precision_score(y_val, score)),
        "log_loss": float(log_loss(y_val, score, labels=[0, 1])),
    }


def get_ranked_manifest_features(manifest_path: Path, train_cols: set[str]) -> list[str]:
    manifest = pd.read_csv(manifest_path)

    if "rank" in manifest.columns:
        manifest = manifest.sort_values("rank")

    feature_col = None
    for c in ["feature_name", "feature", "column", "program_feature"]:
        if c in manifest.columns:
            feature_col = c
            break

    if feature_col is None:
        raise ValueError(f"No feature column in manifest. Columns={list(manifest.columns)}")

    return [f for f in manifest[feature_col].astype(str).tolist() if f in train_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="rel-stack")
    parser.add_argument("--task", default="user-badge")
    parser.add_argument("--variant", default="fdhg_topK")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-train-rows", type=int, default=10000)
    parser.add_argument("--max-val-rows", type=int, default=2000)
    parser.add_argument("--ks", nargs="*", type=int, default=[0, 8, 16, 32, 64])
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(feature_dir / "target_with_dfs_agg_train.parquet")
    val = pd.read_parquet(feature_dir / "target_with_dfs_agg_val.parquet")

    target_col = find_target_col(train)

    train, y_train = sample_df_xy(train, target_col, args.max_train_rows, args.seed)
    val, y_val = sample_df_xy(val, target_col, args.max_val_rows, args.seed)

    # Candidate FDHG-specific feature programs from manifest.
    # Each program may have a value column and a matching missing indicator.
    ranked_program_features = get_ranked_manifest_features(Path(args.manifest), set(train.columns))

    def expand_program_feature(f):
        cols = []
        if f in train.columns and f in val.columns:
            cols.append(f)
        miss = f + "__is_missing"
        if miss in train.columns and miss in val.columns:
            cols.append(miss)
        return cols

    # All f_amb__ columns are FDHG-specific and must be excluded from K=0.
    all_fdhg_specific_cols = [
        c for c in train.columns
        if c.startswith("f_amb__")
    ]

    # Base DFS columns = all usable non-target, non-FDHG-specific columns.
    blocked = set(all_fdhg_specific_cols) | {target_col}
    base_cols = [
        c for c in train.columns
        if c not in blocked and c in val.columns
    ]

    rows = []

    for k in args.ks:
        selected_programs = ranked_program_features[:k]
        selected = []
        for f in selected_programs:
            selected.extend(expand_program_feature(f))

        # preserve order while deduplicating
        selected = list(dict.fromkeys(selected))
        feature_cols = base_cols + selected

        X_train = clean_X(train, feature_cols)
        X_val = clean_X(val, feature_cols)

        print("\n" + "=" * 100)
        print(f"seed={args.seed} K={k}")
        print("target_col:", target_col)
        print("base_cols:", len(base_cols))
        print("available_fdhg_programs:", len(ranked_program_features))
        print("selected_programs:", selected_programs)
        print("selected_columns:", selected)
        print("X_train:", X_train.shape)
        print("X_val:", X_val.shape)

        metrics = eval_one(X_train, y_train, X_val, y_val, args.device, args.seed)

        row = {
            "dataset": args.dataset,
            "task": args.task,
            "variant": args.variant,
            "seed": args.seed,
            "K": k,
            "n_base_features": len(base_cols),
            "n_selected_fdhg_programs": len(selected_programs),
            "n_selected_fdhg_columns": len(selected),
            "n_features_total": int(X_train.shape[1]),
            "selected_features": ",".join(selected),
            **metrics,
        }
        rows.append(row)

        print(json.dumps(metrics, indent=2))

    out = pd.DataFrame(rows)
    out_csv = output_dir / f"feature_budget_curve_{args.dataset}_{args.task}_{args.variant}_seed{args.seed}.csv"
    out_csv = Path(str(out_csv).replace("/", "_"))
    out.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
