import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("results/final_tables/final_all_runs.csv")
MAIN_CLEAN_PATH = Path("results/final_tables/clean_main_4task_runs.csv")
OUT_DIR = Path("results/final_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "rel-f1"
TASK = "driver-dnf"

METRICS = ["accuracy", "roc_auc", "average_precision", "log_loss"]

VARIANT_MAP = {
    "target_only": "target_only",
    "naive": "naive_latest_flatten",
    "dfs": "dfs",
    "fdhg_dmax1": "fdhg_dmax1",
    "last_only": "last_only",
    "dfs_plus_last": "dfs_plus_last",
    "fdhg_plus_last": "fdhg_plus_last",
}

ORDER = [
    "target_only",
    "naive_latest_flatten",
    "dfs",
    "fdhg_dmax1",
    "last_only",
    "dfs_plus_last",
    "fdhg_plus_last",
]


def summarize(df):
    rows = []
    for variant, g in df.groupby("paper_variant"):
        row = {
            "paper_variant": variant,
            "n_runs": len(g),
            "seeds": ",".join(map(str, sorted(g["seed"].dropna().astype(int).unique()))),
            "n_features_mean": g["n_features"].mean() if "n_features" in g else np.nan,
            "n_features_std": g["n_features"].std(ddof=1) if len(g) > 1 else 0.0,
        }
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    order_map = {v: i for i, v in enumerate(ORDER)}
    out["_order"] = out["paper_variant"].map(order_map).fillna(999).astype(int)
    return out.sort_values(["_order", "paper_variant"]).drop(columns=["_order"])


def main():
    all_df = pd.read_csv(IN_PATH)

    # Use clean main rows for target_only / naive / dfs / fdhg_dmax1.
    clean = pd.read_csv(MAIN_CLEAN_PATH)
    clean = clean[
        (clean["dataset"] == DATASET)
        & (clean["task"] == TASK)
        & (clean["variant"].isin(["target_only", "naive", "dfs", "fdhg_dmax1"]))
    ].copy()
    clean["paper_variant"] = clean["variant"].map(VARIANT_MAP)
    clean["source_table"] = "clean_main_4task_runs"

    # Use final_all_runs for temporal diagnostic variants.
    temporal = all_df[
        (all_df["dataset"] == DATASET)
        & (all_df["task"] == TASK)
        & (all_df["variant"].isin(["last_only", "dfs_plus_last", "fdhg_plus_last"]))
    ].copy()
    temporal["paper_variant"] = temporal["variant"].map(VARIANT_MAP)
    temporal["source_table"] = "final_all_runs"

    combined = pd.concat([clean, temporal], ignore_index=True, sort=False)

    if "status" in combined.columns:
        combined = combined[combined["status"].astype(str).str.lower().isin(["ok", "success"])].copy()

    combined = combined[combined["seed"].isin([41, 42, 43, 44])].copy()

    # Deduplicate if both summary CSV and JSON rows exist.
    if "result_path" in combined.columns:
        combined["_json_prefer"] = combined["result_path"].astype(str).str.endswith(".json").astype(int)
    else:
        combined["_json_prefer"] = 0

    combined = (
        combined.sort_values(["paper_variant", "seed", "_json_prefer"], ascending=[True, True, False])
        .drop_duplicates(subset=["paper_variant", "seed"], keep="first")
        .drop(columns=["_json_prefer"])
    )

    runs_path = OUT_DIR / "f1_driver_dnf_temporal_diagnostic_all_runs.csv"
    combined.to_csv(runs_path, index=False)

    summary = summarize(combined)
    summary_path = OUT_DIR / "f1_driver_dnf_temporal_diagnostic_summary.csv"
    summary.to_csv(summary_path, index=False)

    s = summary.set_index("paper_variant")
    delta_rows = []

    def add_delta(name, a, b, description):
        if a not in s.index or b not in s.index:
            return
        row = {
            "comparison": name,
            "variant_a": a,
            "variant_b": b,
            "description": description,
        }
        for m in METRICS:
            row[f"delta_{m}_mean"] = float(s.loc[a, f"{m}_mean"] - s.loc[b, f"{m}_mean"])
        delta_rows.append(row)

    add_delta(
        "fdhg_over_dfs",
        "fdhg_dmax1",
        "dfs",
        "FDHG dmax1 versus DFS without explicit last/recent temporal operators",
    )
    add_delta(
        "naive_over_fdhg",
        "naive_latest_flatten",
        "fdhg_dmax1",
        "Naive latest flattening versus FDHG dmax1",
    )
    add_delta(
        "last_only_over_fdhg",
        "last_only",
        "fdhg_dmax1",
        "Explicit last-state temporal feature versus FDHG dmax1",
    )
    add_delta(
        "dfs_plus_last_over_dfs",
        "dfs_plus_last",
        "dfs",
        "Adding last-state temporal features to DFS",
    )
    add_delta(
        "fdhg_plus_last_over_fdhg",
        "fdhg_plus_last",
        "fdhg_dmax1",
        "Adding last-state temporal features to FDHG",
    )
    add_delta(
        "fdhg_plus_last_vs_dfs_plus_last",
        "fdhg_plus_last",
        "dfs_plus_last",
        "FDHG plus last-state features versus DFS plus last-state features",
    )
    add_delta(
        "dfs_plus_last_over_naive",
        "dfs_plus_last",
        "naive_latest_flatten",
        "DFS plus last-state features versus naive latest flattening",
    )

    deltas = pd.DataFrame(delta_rows)
    delta_path = OUT_DIR / "f1_driver_dnf_temporal_diagnostic_deltas.csv"
    deltas.to_csv(delta_path, index=False)

    print("Saved:")
    print(f"  {runs_path}")
    print(f"  {summary_path}")
    print(f"  {delta_path}")

    print("\n=== F1 DRIVER-DNF TEMPORAL DIAGNOSTIC SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== DELTAS ===")
    print(deltas.to_string(index=False))

    print("\n=== RUN COUNTS CHECK ===")
    print(summary[["paper_variant", "n_runs", "seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()
