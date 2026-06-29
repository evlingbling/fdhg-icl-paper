import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURES = [
    "r_pair",
    "r_tuple",
    "r_del",
    "r_ent",
    "lhs_collision_rate",
    "lhs_uniqueness",
    "lhs_support_repeated",
    "coverage",
    "rhs_entropy",
    "lhs_arity",
    "schema_flag",
    "bootstrap_stability",
]


def heuristic_score(df):
    # 문서의 minimal heuristic spirit:
    # reliability high, support/stability high, uniqueness/constant RHS penalty.
    score = (
        1.5 * df["r_del"].fillna(0)
        + 1.0 * df["r_ent"].fillna(0)
        + 0.8 * df["lhs_support_repeated"].fillna(0)
        + 0.5 * df["bootstrap_stability"].fillna(0)
        + 0.3 * df["schema_flag"].fillna(0)
        - 1.2 * df["lhs_uniqueness"].fillna(1)
        - 0.8 * (df["rhs_entropy"].fillna(0) < 0.05).astype(float)
    )
    return score.to_numpy()


def recall_precision_at_k(y_true, scores, k):
    order = np.argsort(-scores)
    top = order[:k]
    tp = int(y_true[top].sum())
    total_pos = int(y_true.sum())
    precision = tp / max(k, 1)
    recall = tp / max(total_pos, 1)
    return {
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        f"tp_at_{k}": tp,
        f"k_{k}": k,
    }


def surrogate_fp_rate(df, scores, k):
    order = np.argsort(-scores)
    top = df.iloc[order[:k]]
    if len(top) == 0:
        return 0.0
    return float((top["edge_type"] == "surrogate_key_like").mean())


def constant_fp_rate(df, scores, k):
    order = np.argsort(-scores)
    top = df.iloc[order[:k]]
    if len(top) == 0:
        return 0.0
    return float((top["edge_type"] == "constant_rhs_trivial").mean())


def eval_scores(name, df, y, scores, ks):
    out = {
        "method": name,
        "edge_auroc": float(roc_auc_score(y, scores)),
        "edge_ap": float(average_precision_score(y, scores)),
    }

    # log loss only meaningful if score is probability-like.
    eps = 1e-6
    if scores.min() >= 0 and scores.max() <= 1:
        out["edge_log_loss"] = float(log_loss(y, np.clip(scores, eps, 1 - eps)))
    else:
        out["edge_log_loss"] = None

    for k in ks:
        k = min(k, len(y))
        out.update(recall_precision_at_k(y, scores, k))
        out[f"surrogate_fp_rate_at_{k}"] = surrogate_fp_rate(df, scores, k)
        out[f"constant_fp_rate_at_{k}"] = constant_fp_rate(df, scores, k)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_csv", required=True)
    parser.add_argument("--out_dir", default="results/calibrator_synthetic/calibrator")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--max_iter", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.edge_csv)
    df = df.copy()

    # Clean numeric features.
    for c in FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    y = df["is_true_structural_fd"].astype(int).to_numpy()
    groups = df["task_id"].to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.seed,
    )
    train_idx, test_idx = next(splitter.split(df, y, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[FEATURES].to_numpy()
    y_train = train_df["is_true_structural_fd"].astype(int).to_numpy()

    X_test = test_df[FEATURES].to_numpy()
    y_test = test_df["is_true_structural_fd"].astype(int).to_numpy()

    print("train shape:", X_train.shape, "positives:", int(y_train.sum()))
    print("test shape:", X_test.shape, "positives:", int(y_test.sum()))
    print("test edge types:")
    print(test_df["edge_type"].value_counts())

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=args.seed,
        )),
    ])

    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]

    raw_rdel = test_df["r_del"].to_numpy()
    raw_rent = test_df["r_ent"].to_numpy()
    heur = heuristic_score(test_df)

    total_pos = int(y_test.sum())
    ks = sorted(set([
        20,
        50,
        100,
        total_pos,
        min(2 * total_pos, len(y_test)),
    ]))

    results = []
    results.append(eval_scores("raw_r_del", test_df, y_test, raw_rdel, ks))
    results.append(eval_scores("raw_r_ent", test_df, y_test, raw_rent, ks))
    results.append(eval_scores("heuristic_score", test_df, y_test, heur, ks))
    results.append(eval_scores("learned_logreg_calibrator", test_df, y_test, pred, ks))

    res_df = pd.DataFrame(results)
    res_path = out_dir / "calibrator_metrics.csv"
    res_df.to_csv(res_path, index=False)

    # Coefficients
    clf = model.named_steps["clf"]
    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coef": clf.coef_[0],
    }).sort_values("coef", ascending=False)
    coef_df.to_csv(out_dir / "calibrator_coefficients.csv", index=False)

    # Save scored test edges
    scored = test_df.copy()
    scored["score_raw_r_del"] = raw_rdel
    scored["score_raw_r_ent"] = raw_rent
    scored["score_heuristic"] = heur
    scored["score_learned_calibrator"] = pred
    scored.to_csv(out_dir / "test_edges_scored.csv", index=False)

    joblib.dump(model, out_dir / "fd_calibrator_logreg.joblib")

    summary = {
        "edge_csv": args.edge_csv,
        "out_dir": str(out_dir),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_train_pos": int(y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "features": FEATURES,
        "metrics_csv": str(res_path),
    }
    with open(out_dir / "calibrator_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== METRICS ===")
    print(res_df.to_string(index=False))

    print("\n=== COEFFICIENTS ===")
    print(coef_df.to_string(index=False))

    print("\n=== Top learned calibrator edges ===")
    cols = [
        "edge_type", "lhs", "rhs", "is_true_structural_fd",
        "score_learned_calibrator", "score_raw_r_del", "score_heuristic",
        "r_del", "r_ent", "lhs_uniqueness", "lhs_support_repeated",
        "rhs_entropy", "bootstrap_stability", "schema_flag"
    ]
    print(scored.sort_values("score_learned_calibrator", ascending=False)[cols].head(30).to_string(index=False))

    print("\n=== Top raw r_del edges ===")
    print(scored.sort_values("score_raw_r_del", ascending=False)[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
