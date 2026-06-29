import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("results/final_tables/final_all_runs.csv")
MAIN_CLEAN_PATH = Path("results/final_tables/clean_main_4task_runs.csv")
OUT_DIR = Path("results/final_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "rel-amazon"
TASK = "item-churn"

METRICS = ["accuracy", "roc_auc", "average_precision", "log_loss"]

VARIANT_MAP = {
    "target_only": "target_only",
    "naive": "naive",
    "dfs": "dfs",
    "fdhg_dmax1": "fdhg_dmax1_full",
    "fdhg_dmax1_no_ambiguity": "fdhg_dmax1_no_ambiguity",
    "fdhg_dmax1_random_same_budget": "random_same_budget",
    "fdhg_dmax1_shuffle_ambiguity": "shuffle_ambiguity",
}

ORDER = [
    "target_only",
    "naive",
    "dfs",
    "fdhg_dmax1_full",
    "fdhg_dmax1_no_ambiguity",
    "random_same_budget",
    "shuffle_ambiguity",
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

    # Use clean main rows for target_only/naive/dfs/fdhg_dmax1_full.
    clean = pd.read_csv(MAIN_CLEAN_PATH)
    clean = clean[
        (clean["dataset"] == DATASET)
        & (clean["task"] == TASK)
        & (clean["variant"].isin(["target_only", "naive", "dfs", "fdhg_dmax1"]))
    ].copy()
    clean["paper_variant"] = clean["variant"].map(VARIANT_MAP)
    clean["source_table"] = "clean_main_4task_runs"

    # Use final_all_runs for ablation controls.
    rest = all_df[
        (all_df["dataset"] == DATASET)
        & (all_df["task"] == TASK)
        & (all_df["variant"].isin([
            "fdhg_dmax1_no_ambiguity",
            "fdhg_dmax1_random_same_budget",
            "fdhg_dmax1_shuffle_ambiguity",
        ]))
    ].copy()
    rest["paper_variant"] = rest["variant"].map(VARIANT_MAP)
    rest["source_table"] = "final_all_runs"

    combined = pd.concat([clean, rest], ignore_index=True, sort=False)

    if "status" in combined.columns:
        combined = combined[combined["status"].astype(str).str.lower().isin(["ok", "success"])].copy()

    combined = combined[combined["seed"].isin([41, 42, 43, 44])].copy()

    # Deduplicate if the aggregator has both csv-summary and json rows for same variant/seed.
    # Prefer explicit JSON result paths over phase summary CSV rows when duplicate.
    if "result_path" in combined.columns:
        combined["_json_prefer"] = combined["result_path"].astype(str).str.endswith(".json").astype(int)
    else:
        combined["_json_prefer"] = 0

    combined["_order"] = combined["paper_variant"].map({v: i for i, v in enumerate(ORDER)}).fillna(999)
    combined = (
        combined.sort_values(["paper_variant", "seed", "_json_prefer"], ascending=[True, True, False])
        .drop_duplicates(subset=["paper_variant", "seed"], keep="first")
        .drop(columns=["_json_prefer", "_order"])
    )

    runs_path = OUT_DIR / "amazon_item_churn_ablation_all_runs.csv"
    combined.to_csv(runs_path, index=False)

    summary = summarize(combined)
    summary_path = OUT_DIR / "amazon_item_churn_ablation_summary.csv"
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
        "fdhg_full_over_dfs",
        "fdhg_dmax1_full",
        "dfs",
        "FDHG dmax1 full versus DFS-style aggregation",
    )
    add_delta(
        "fdhg_full_over_no_ambiguity",
        "fdhg_dmax1_full",
        "fdhg_dmax1_no_ambiguity",
        "Contribution of FDHG ambiguity features",
    )
    add_delta(
        "fdhg_full_over_random_same_budget",
        "fdhg_dmax1_full",
        "random_same_budget",
        "FDHG full versus random same-budget feature programs",
    )
    add_delta(
        "fdhg_full_over_shuffle_ambiguity",
        "fdhg_dmax1_full",
        "shuffle_ambiguity",
        "FDHG full versus shuffled ambiguity control",
    )

    deltas = pd.DataFrame(delta_rows)
    delta_path = OUT_DIR / "amazon_item_churn_ablation_deltas.csv"
    deltas.to_csv(delta_path, index=False)

    print("Saved:")
    print(f"  {runs_path}")
    print(f"  {summary_path}")
    print(f"  {delta_path}")

    print("\n=== AMAZON ITEM-CHURN ABLATION SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== DELTAS ===")
    print(deltas.to_string(index=False))

    print("\n=== RUN COUNTS CHECK ===")
    print(summary[["paper_variant", "n_runs", "seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()
