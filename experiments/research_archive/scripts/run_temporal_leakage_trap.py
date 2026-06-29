import argparse
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split


def make_task(seed: int, n_users: int = 5000, n_events_per_user: int = 12):
    rng = np.random.default_rng(seed)

    users = np.arange(n_users)
    signup_age = rng.normal(0, 1, size=n_users)
    user_quality = rng.normal(0, 1, size=n_users)

    # Prediction cutoff time.
    cutoff = np.full(n_users, 50)

    # Target label depends partly on hidden user_quality.
    eta = -0.2 + 0.8 * user_quality + 0.3 * signup_age
    prob = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, prob, size=n_users)

    target = pd.DataFrame({
        "user_id": users,
        "cutoff_time": cutoff,
        "signup_age": signup_age,
        "target": y,
    })

    events = []
    for u in users:
        # Past events before cutoff.
        for _ in range(n_events_per_user):
            t = rng.integers(1, 50)
            signal = 0.3 * user_quality[u] + rng.normal(0, 1)
            events.append((u, t, signal, 0))

        # Future event after cutoff.
        # This is intentionally label-derived leakage.
        future_t = rng.integers(51, 80)
        future_signal = y[u] + rng.normal(0, 0.03)
        events.append((u, future_t, future_signal, 1))

    events = pd.DataFrame(
        events,
        columns=["user_id", "event_time", "event_signal", "is_future_label_proxy"]
    )

    return target, events


def build_features(target, events, mode):
    rows = []

    for _, r in target.iterrows():
        uid = r["user_id"]
        cutoff = r["cutoff_time"]

        if mode == "safe_cutoff":
            obs = events[(events["user_id"] == uid) & (events["event_time"] <= cutoff)]
        elif mode == "leaky_future":
            obs = events[events["user_id"] == uid]
        elif mode == "target_only":
            obs = None
        else:
            raise ValueError(mode)

        row = {
            "user_id": uid,
            "signup_age": r["signup_age"],
            "target": r["target"],
        }

        if obs is not None:
            row["event_count"] = len(obs)
            row["event_signal_mean"] = obs["event_signal"].mean()
            row["event_signal_max"] = obs["event_signal"].max()
            row["event_signal_last"] = obs.sort_values("event_time")["event_signal"].iloc[-1]
            row["days_since_last"] = cutoff - obs["event_time"].max()
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def eval_df(df, seed):
    y = df["target"].astype(int)
    X = df.drop(columns=["target", "user_id"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=seed, stratify=y
    )

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
        "n_features": X.shape[1],
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "average_precision": average_precision_score(y_test, prob),
        "log_loss": log_loss(y_test, prob, labels=[0, 1]),
    }


def run_seed(seed):
    target, events = make_task(seed)
    rows = []

    for mode in ["target_only", "safe_cutoff", "leaky_future"]:
        df = build_features(target, events, mode)
        metrics = eval_df(df, seed)

        rows.append({
            "seed": seed,
            "mode": mode,
            "n_target_rows": len(target),
            "n_event_rows": len(events),
            **metrics,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="41,42,43,44,45,46,47,48")
    ap.add_argument("--out", default="results/final_tables/temporal_leakage_trap_by_seed.csv")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    all_rows = []
    for seed in seeds:
        all_rows.extend(run_seed(seed))

    out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    summary = (
        out.groupby("mode", as_index=False)
        .agg(
            n_runs=("seed", "count"),
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

    # Make explicit delta table.
    base = summary[summary["mode"] == "safe_cutoff"].iloc[0]
    rows = []
    for _, r in summary.iterrows():
        if r["mode"] == "safe_cutoff":
            continue
        rows.append({
            "comparison": f"{r['mode']} minus safe_cutoff",
            "delta_accuracy": r["accuracy_mean"] - base["accuracy_mean"],
            "delta_roc_auc": r["roc_auc_mean"] - base["roc_auc_mean"],
            "delta_average_precision": r["average_precision_mean"] - base["average_precision_mean"],
            "delta_log_loss": r["log_loss_mean"] - base["log_loss_mean"],
        })
    delta = pd.DataFrame(rows)
    delta_path = args.out.replace("_by_seed.csv", "_deltas_vs_safe_cutoff.csv")
    delta.to_csv(delta_path, index=False)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== DELTA VS SAFE CUTOFF ===")
    print(delta.to_string(index=False))
    print("\nSaved:")
    print(args.out)
    print(summary_path)
    print(delta_path)


if __name__ == "__main__":
    main()
