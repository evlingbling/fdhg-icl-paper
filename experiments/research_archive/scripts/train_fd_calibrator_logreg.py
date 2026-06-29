from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "r_del",
    "r_ent",
    "r_pair",
    "r_tuple",
    "lhs_uniqueness",
    "lhs_collision",
    "lhs_support_repeated",
    "coverage",
    "rhs_entropy",
    "n_lhs_values",
    "n_rhs_values",
]


def load_ground_truth(path: Path) -> tuple[set[tuple[str, str, str]], set[str]]:
    with open(path) as f:
        gt = json.load(f)

    true_fds = set()
    id_cols = set()

    for table_name, meta in gt["tables"].items():
        pk = meta.get("primary_key")
        if pk:
            id_cols.add(pk)

    for fd in gt["ground_truth_fds"]:
        true_fds.add((fd["table"], fd["lhs"], fd["rhs"]))

    return true_fds, id_cols


def load_one_seed(seed_dir: Path) -> pd.DataFrame:
    true_fds, id_cols = load_ground_truth(seed_dir / "ground_truth.json")

    frames = []

    for table_name, fname in [
        ("products", "products_afd_dmax1.csv"),
        ("users", "users_afd_dmax1.csv"),
    ]:
        path = seed_dir / fname
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["table"] = table_name
        df["seed_dir"] = seed_dir.name

        # MVP calibrator: train non-ID AFD reliability.
        # Schema/key FDs are handled separately by schema_fd edges and uniqueness penalty.
        df = df[~df["lhs"].isin(id_cols)].copy()
        df = df[~df["rhs"].isin(id_cols)].copy()

        df["label_structural_afd"] = [
            int((table_name, lhs, rhs) in true_fds)
            for lhs, rhs in zip(df["lhs"], df["rhs"])
        ]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prior-root",
        default="outputs/synthetic_prior",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/calibrator/logreg_minimal",
    )
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    prior_root = Path(args.prior_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(prior_root.glob("minimal_seed*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No minimal_seed* dirs found under {prior_root}")

    all_rows = []
    for seed_dir in seed_dirs:
        df = load_one_seed(seed_dir)
        if not df.empty:
            all_rows.append(df)

    data = pd.concat(all_rows, ignore_index=True)

    missing = set(FEATURE_COLS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    X = data[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = data["label_structural_afd"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both positive and negative examples to train calibrator.")

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        data[["seed_dir", "table", "lhs", "rhs", "label_structural_afd"]].copy(),
        test_size=0.30,
        random_state=args.seed,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=args.seed,
                    max_iter=1000,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "n_examples": int(len(data)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
        "test_average_precision": float(average_precision_score(y_test, prob)),
        "test_log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
    }

    test_out = meta_test.reset_index(drop=True).copy()
    test_out["calibrator_prob"] = prob
    test_out["pred"] = pred
    test_out = pd.concat([test_out, X_test.reset_index(drop=True)], axis=1)
    test_out = test_out.sort_values("calibrator_prob", ascending=False)

    data_out = data.copy()
    data_out["calibrator_prob_full"] = model.predict_proba(X)[:, 1]
    data_out = data_out.sort_values("calibrator_prob_full", ascending=False)

    with open(out_dir / "fd_calibrator_logreg.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    test_out.to_csv(out_dir / "heldout_predictions.csv", index=False)
    data_out.to_csv(out_dir / "all_calibrator_training_examples.csv", index=False)

    print("=== FD calibrator logistic regression ===")
    print(json.dumps(metrics, indent=2))
    print("\nTop full-data examples:")
    print(
        data_out[
            [
                "seed_dir",
                "table",
                "lhs",
                "rhs",
                "label_structural_afd",
                "calibrator_prob_full",
                "r_del",
                "r_ent",
                "lhs_uniqueness",
                "lhs_support_repeated",
            ]
        ].head(30).to_string(index=False)
    )

    print("\nSaved:")
    print(out_dir / "fd_calibrator_logreg.pkl")
    print(out_dir / "metrics.json")
    print(out_dir / "heldout_predictions.csv")
    print(out_dir / "all_calibrator_training_examples.csv")


if __name__ == "__main__":
    main()
