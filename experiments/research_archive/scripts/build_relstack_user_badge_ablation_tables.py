import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("results/final_tables/final_all_runs.csv")
MAIN_CLEAN_PATH = Path("results/final_tables/clean_main_4task_runs.csv")
OUT_DIR = Path("results/final_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["accuracy", "roc_auc", "average_precision", "log_loss"]

# Final paper-facing names.
VARIANT_MAP = {
    # dmax1 main / ablation
    "target_only": "target_only",
    "naive": "naive",
    "dfs": "dfs",
    "fdhg_dmax1": "fdhg_dmax1_full",
    "fdhg_dmax1_no_ambiguity": "fdhg_dmax1_no_ambiguity",
    "fdhg_dmax1_random_same_budget": "random_same_budget",
    "fdhg_dmax1_shuffle_ambiguity": "shuffle_ambiguity",

    # Extension B
    "dmax2_only_topk16": "dmax2_only_topK16",
    "dmax2_only_random16": "dmax2_only_random16",
    "fdhg_dmax1_plus_dmax2_topk16": "dmax1_plus_dmax2_topK16",
    "fdhg_dmax1_plus_dmax2_random16": "dmax1_plus_dmax2_random16",

    # Extension C
    "fdhg_dmax1_plus_dmax2_supervised_auc_topk16": "dmax1_plus_dmax2_supervised_auc_topK16",
    "fdhg_dmax1_plus_dmax2_supervised_ap_topk16": "dmax1_plus_dmax2_supervised_ap_topK16",
    "dmax2_only_supervised_auc_topk16": "dmax2_only_supervised_auc_topK16",
    "dmax2_only_supervised_ap_topk16": "dmax2_only_supervised_ap_topK16",
}

DMAX1_ORDER = [
    "target_only",
    "naive",
    "dfs",
    "fdhg_dmax1_full",
    "random_same_budget",
    "shuffle_ambiguity",
]

EXT_B_ORDER = [
    "fdhg_dmax1_full",
    "dmax2_only_topK16",
    "dmax2_only_random16",
    "dmax1_plus_dmax2_topK16",
    "dmax1_plus_dmax2_random16",
]

EXT_C_ORDER = [
    "fdhg_dmax1_full",
    "dmax1_plus_dmax2_random16",
    "dmax1_plus_dmax2_topK16",
    "dmax1_plus_dmax2_supervised_auc_topK16",
    "dmax1_plus_dmax2_supervised_ap_topK16",
    "dmax2_only_supervised_auc_topK16",
    "dmax2_only_supervised_ap_topK16",
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
    return pd.DataFrame(rows)


def order_summary(summary, order):
    order_map = {v: i for i, v in enumerate(order)}
    summary = summary.copy()
    summary["_order"] = summary["paper_variant"].map(order_map).fillna(999).astype(int)
    return summary.sort_values(["_order", "paper_variant"]).drop(columns=["_order"])


def add_clean_main_relstack(df_all):
    """
    Use clean_main_4task_runs.csv for target_only/naive/dfs/fdhg_dmax1_full,
    because rel-stack DFS main row is the regenerated clean dmax1 DFS.
    Use final_all_runs.csv for the rest of ablations/extensions.
    """
    clean = pd.read_csv(MAIN_CLEAN_PATH)
    clean = clean[
        (clean["dataset"] == "rel-stack")
        & (clean["task"] == "user-badge")
        & (clean["variant"].isin(["target_only", "naive", "dfs", "fdhg_dmax1"]))
    ].copy()
    clean["paper_variant"] = clean["variant"].map(VARIANT_MAP)
    clean["source_table"] = "clean_main_4task_runs"

    rest = df_all[
        (df_all["dataset"] == "rel-stack")
        & (df_all["task"] == "user-badge")
        & (~df_all["variant"].isin(["target_only", "naive", "dfs", "fdhg_dmax1"]))
    ].copy()
    rest["paper_variant"] = rest["variant"].map(VARIANT_MAP)
    rest = rest[rest["paper_variant"].notna()].copy()
    rest["source_table"] = "final_all_runs"

    combined = pd.concat([clean, rest], ignore_index=True, sort=False)

    # Normalize status.
    if "status" in combined.columns:
        combined = combined[combined["status"].astype(str).str.lower().isin(["ok", "success"])].copy()

    # Keep seeds 41-44 only for paper summary.
    combined = combined[combined["seed"].isin([41, 42, 43, 44])].copy()

    return combined


def compute_delta_rows(summary):
    s = summary.set_index("paper_variant")
    rows = []

    def delta(name, a, b, description):
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
        rows.append(row)

    # Extension B requested deltas
    delta(
        "topK16_over_dmax1_alone",
        "dmax1_plus_dmax2_topK16",
        "fdhg_dmax1_full",
        "Complementary gain from adding heuristic dmax2 topK16 to dmax1 FDHG",
    )
    delta(
        "topK16_over_random16",
        "dmax1_plus_dmax2_topK16",
        "dmax1_plus_dmax2_random16",
        "Heuristic dmax2 topK16 versus same-budget random dmax2 when added to dmax1",
    )
    delta(
        "dmax2_only_topK16_vs_random16",
        "dmax2_only_topK16",
        "dmax2_only_random16",
        "dmax2-only heuristic topK16 versus dmax2-only random16",
    )

    # Extension C requested deltas
    delta(
        "supervised_auc_over_heuristic_topK16",
        "dmax1_plus_dmax2_supervised_auc_topK16",
        "dmax1_plus_dmax2_topK16",
        "Train-only supervised AUC ranker versus heuristic topK16",
    )
    delta(
        "supervised_ap_over_heuristic_topK16",
        "dmax1_plus_dmax2_supervised_ap_topK16",
        "dmax1_plus_dmax2_topK16",
        "Train-only supervised AP ranker versus heuristic topK16",
    )
    delta(
        "supervised_auc_over_random16",
        "dmax1_plus_dmax2_supervised_auc_topK16",
        "dmax1_plus_dmax2_random16",
        "Train-only supervised AUC ranker versus random16",
    )
    delta(
        "supervised_ap_over_random16",
        "dmax1_plus_dmax2_supervised_ap_topK16",
        "dmax1_plus_dmax2_random16",
        "Train-only supervised AP ranker versus random16",
    )
    delta(
        "supervised_ap_over_supervised_auc",
        "dmax1_plus_dmax2_supervised_ap_topK16",
        "dmax1_plus_dmax2_supervised_auc_topK16",
        "AP ranker versus AUC ranker",
    )

    return pd.DataFrame(rows)


def main():
    df_all = pd.read_csv(IN_PATH)
    combined = add_clean_main_relstack(df_all)

    # Save all selected run-level rows.
    combined.to_csv(OUT_DIR / "relstack_user_badge_ablation_all_runs.csv", index=False)

    # Dmax1 ablation
    dmax1 = combined[combined["paper_variant"].isin(DMAX1_ORDER)].copy()
    dmax1_summary = order_summary(summarize(dmax1), DMAX1_ORDER)
    dmax1_summary.to_csv(OUT_DIR / "relstack_user_badge_dmax1_ablation_summary.csv", index=False)

    # Extension B
    ext_b = combined[combined["paper_variant"].isin(EXT_B_ORDER)].copy()
    ext_b_summary = order_summary(summarize(ext_b), EXT_B_ORDER)
    ext_b_summary.to_csv(OUT_DIR / "relstack_user_badge_extension_b_summary.csv", index=False)

    # Extension C
    ext_c = combined[combined["paper_variant"].isin(EXT_C_ORDER)].copy()
    ext_c_summary = order_summary(summarize(ext_c), EXT_C_ORDER)
    ext_c_summary.to_csv(OUT_DIR / "relstack_user_badge_extension_c_summary.csv", index=False)

    all_order = DMAX1_ORDER + [v for v in EXT_B_ORDER + EXT_C_ORDER if v not in DMAX1_ORDER]
    all_summary = order_summary(summarize(combined), all_order)
    all_summary.to_csv(OUT_DIR / "relstack_user_badge_ablation_all_summary.csv", index=False)

    deltas = compute_delta_rows(all_summary)
    deltas.to_csv(OUT_DIR / "relstack_user_badge_ablation_deltas.csv", index=False)

    print("Saved:")
    print("  results/final_tables/relstack_user_badge_ablation_all_runs.csv")
    print("  results/final_tables/relstack_user_badge_dmax1_ablation_summary.csv")
    print("  results/final_tables/relstack_user_badge_extension_b_summary.csv")
    print("  results/final_tables/relstack_user_badge_extension_c_summary.csv")
    print("  results/final_tables/relstack_user_badge_ablation_all_summary.csv")
    print("  results/final_tables/relstack_user_badge_ablation_deltas.csv")

    print("\n=== DMAX1 ABLATION SUMMARY ===")
    print(dmax1_summary.to_string(index=False))

    print("\n=== EXTENSION B SUMMARY ===")
    print(ext_b_summary.to_string(index=False))

    print("\n=== EXTENSION C SUMMARY ===")
    print(ext_c_summary.to_string(index=False))

    print("\n=== DELTAS ===")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
