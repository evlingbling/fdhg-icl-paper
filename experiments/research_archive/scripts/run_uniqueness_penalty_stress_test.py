import argparse
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split


def make_synthetic(seed: int, n: int = 6000):
    rng = np.random.default_rng(seed)

    # True surrogate-key-like determinants.
    row_id = np.arange(n)
    user_id = np.array([f"user_{i}" for i in row_id])
    event_id = np.array([f"event_{i}" for i in row_id])

    # These RHS columns are perfectly determined by IDs, creating tempting id -> all FDs.
    id_payload_1 = np.array([f"payloadA_{i}" for i in row_id])
    id_payload_2 = np.array([f"payloadB_{i}" for i in row_id])
    id_payload_3 = rng.normal(size=n) + row_id * 1e-6

    # Meaningful repeated determinants.
    region = rng.integers(0, 8, size=n)
    product_group = rng.integers(0, 12, size=n)

    city = region.copy()
    city_noise = rng.random(n) < 0.08
    city[city_noise] = rng.integers(0, 8, size=city_noise.sum())

    category = product_group.copy()
    cat_noise = rng.random(n) < 0.10
    category[cat_noise] = rng.integers(0, 12, size=cat_noise.sum())

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    # Label depends on meaningful dependencies, not surrogate IDs.
    eta = (
        -0.8
        + 0.55 * (region == city)
        + 0.45 * (product_group == category)
        + 0.25 * (region <= 2)
        - 0.20 * (product_group >= 8)
        + 0.20 * x1
    )
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p, size=n)

    return pd.DataFrame({
        "row_id": row_id,
        "user_id": user_id,
        "event_id": event_id,
        "id_payload_1": id_payload_1,
        "id_payload_2": id_payload_2,
        "id_payload_3": id_payload_3,
        "region": region,
        "product_group": product_group,
        "city": city,
        "category": category,
        "x1": x1,
        "x2": x2,
        "target": y,
    })


def fd_stats(df, lhs, rhs):
    n = len(df)

    lhs_counts = df.groupby(lhs, dropna=False).size()
    uniqueness_ratio = df[lhs].nunique(dropna=False) / max(n, 1)
    support = lhs_counts[lhs_counts >= 2].sum() / max(n, 1)

    counts = df.groupby([lhs, rhs], dropna=False).size().reset_index(name="cnt")
    majority_kept = counts.groupby(lhs)["cnt"].max().sum()
    rdel = majority_kept / max(n, 1)

    # Evidence proxy. Unique LHS can look deceptively perfect unless penalized.
    # This intentionally creates the stress condition.
    apparent_reliability = rdel

    return {
        "lhs": lhs,
        "rhs": rhs,
        "rdel": rdel,
        "apparent_reliability": apparent_reliability,
        "uniqueness_ratio": uniqueness_ratio,
        "support": support,
    }


def score_edge(row, use_uniqueness_penalty: bool):
    # Intentionally stress-test bad behavior:
    # without penalty, perfect id -> payload FDs rank very high.
    score = 3.0 * row["apparent_reliability"]

    # Small support bonus only, not enough to suppress id edges by itself.
    score += 0.2 * np.log1p(row["support"] * 100)

    if use_uniqueness_penalty:
        # Strong penalty for near-unique determinants.
        score -= 4.0 * row["uniqueness_ratio"]

    return score


def materialize_features(df, selected):
    out = pd.DataFrame(index=df.index)
    out["x1"] = df["x1"]
    out["x2"] = df["x2"]

    for e in selected:
        lhs, rhs = e["lhs"], e["rhs"]

        maj = (
            df.groupby([lhs, rhs], dropna=False).size()
              .reset_index(name="cnt")
              .sort_values(["cnt"], ascending=False)
              .drop_duplicates(lhs)
              .set_index(lhs)[rhs]
        )

        out[f"fd_majority__{lhs}_to_{rhs}"] = df[lhs].map(maj)

        rhs_nunique = df.groupby(lhs, dropna=False)[rhs].nunique(dropna=False)
        out[f"fd_conflict__{lhs}_to_{rhs}"] = df[lhs].map(rhs_nunique).fillna(0)

    return out


def encode_train_test(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for c in X_train.columns:
        if X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("string"):
            combined = pd.concat([X_train[c], X_test[c]], axis=0).astype("string").fillna("__NA__")
            codes, _ = pd.factorize(combined, sort=True)
            X_train[c] = codes[:len(X_train)]
            X_test[c] = codes[len(X_train):]

    for c in X_train.columns:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0)

    return X_train, X_test


def evaluate(df, selected, seed):
    X = materialize_features(df, selected)
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=seed, stratify=y
    )
    X_train, X_test = encode_train_test(X_train, X_test)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "average_precision": average_precision_score(y_test, prob),
        "log_loss": log_loss(y_test, prob, labels=[0, 1]),
        "n_features": X_train.shape[1],
    }


def run_seed(seed, top_k):
    df = make_synthetic(seed)

    lhs_cols = ["row_id", "user_id", "event_id", "region", "product_group"]
    rhs_cols = [
        "id_payload_1",
        "id_payload_2",
        "id_payload_3",
        "city",
        "category",
        "x1",
        "x2",
    ]

    candidates = []
    for lhs in lhs_cols:
        for rhs in rhs_cols:
            if lhs == rhs:
                continue
            candidates.append(fd_stats(df, lhs, rhs))

    cand = pd.DataFrame(candidates)

    meaningful_edges = {("region", "city"), ("product_group", "category")}

    rows = []
    selected_rows = []

    for use_penalty in [False, True]:
        c = cand.copy()
        c["score"] = c.apply(lambda r: score_edge(r, use_penalty), axis=1)
        c = c.sort_values("score", ascending=False).head(top_k)

        selected = c.to_dict("records")
        metrics = evaluate(df, selected, seed)

        id_like = sum(e["uniqueness_ratio"] > 0.98 for e in selected)
        meaningful = sum((e["lhs"], e["rhs"]) in meaningful_edges for e in selected)

        row = {
            "seed": seed,
            "use_uniqueness_penalty": use_penalty,
            "top_k": top_k,
            "id_like_selected_edges": id_like,
            "meaningful_selected_edges": meaningful,
            "id_like_ratio": id_like / top_k,
            "meaningful_recall": meaningful / len(meaningful_edges),
            "selected_edges": ";".join([f"{e['lhs']}->{e['rhs']}" for e in selected]),
            **metrics,
        }
        rows.append(row)

        for rank, e in enumerate(selected, start=1):
            selected_rows.append({
                "seed": seed,
                "use_uniqueness_penalty": use_penalty,
                "rank": rank,
                "lhs": e["lhs"],
                "rhs": e["rhs"],
                "score": e["score"],
                "rdel": e["rdel"],
                "uniqueness_ratio": e["uniqueness_ratio"],
                "support": e["support"],
                "is_id_like": e["uniqueness_ratio"] > 0.98,
                "is_meaningful": (e["lhs"], e["rhs"]) in meaningful_edges,
            })

    return rows, selected_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="41,42,43,44,45,46,47,48")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default="results/final_tables/uniqueness_penalty_stress_test_by_seed.csv")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]

    all_rows = []
    all_selected = []
    for seed in seeds:
        rows, selected = run_seed(seed, args.top_k)
        all_rows.extend(rows)
        all_selected.extend(selected)

    out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    selected_path = args.out.replace("_by_seed.csv", "_selected_edges.csv")
    pd.DataFrame(all_selected).to_csv(selected_path, index=False)

    summary = (
        out.groupby("use_uniqueness_penalty", as_index=False)
        .agg(
            n_runs=("seed", "count"),
            id_like_selected_edges_mean=("id_like_selected_edges", "mean"),
            meaningful_selected_edges_mean=("meaningful_selected_edges", "mean"),
            id_like_ratio_mean=("id_like_ratio", "mean"),
            meaningful_recall_mean=("meaningful_recall", "mean"),
            n_features_mean=("n_features", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
        )
    )

    summary_path = args.out.replace("_by_seed.csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== SELECTED EDGE EXAMPLES ===")
    selected_df = pd.DataFrame(all_selected)
    print(selected_df.head(30).to_string(index=False))

    print("\nSaved:")
    print(args.out)
    print(summary_path)
    print(selected_path)


if __name__ == "__main__":
    main()
