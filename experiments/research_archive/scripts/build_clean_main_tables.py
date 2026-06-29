import pandas as pd
from pathlib import Path
import numpy as np

IN_PATH = Path("results/final_tables/final_all_runs.csv")
OUT_DIR = Path("results/final_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_TASKS = [
    ("rel-stack", "user-badge"),
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "user-churn"),
    ("rel-f1", "driver-dnf"),
]

APPENDIX_TASKS = [
    ("rel-stack", "user-badge"),
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "user-churn"),
    ("rel-f1", "driver-dnf"),
    ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    ("rel-trial", "study-outcome"),
]

MAIN_VARIANTS = ["target_only", "naive", "dfs", "fdhg_dmax1"]
SEEDS = [41, 42, 43, 44]

METRICS = ["accuracy", "roc_auc", "average_precision", "log_loss"]


def norm_status(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    if x in {"ok", "success"}:
        return "success"
    return x


def is_success_row(row):
    return norm_status(row.get("status", "")) == "success"


def pick_one(g, dataset, task, variant, seed):
    """
    Pick one clean row among duplicates.

    Rules:
    - Prefer success rows.
    - Prefer exact requested seed.
    - Exclude dmax2 extension variants from main.
    - For rel-stack regenerated dmax1 rows, prefer rows with n_train/n_val filled.
    - For fdhg_dmax1, prefer AFD/ambiguity rows when available.
    - For dfs, prefer no AFD rows.
    """

    g = g.copy()

    if len(g) == 0:
        return None

    # status normalize
    if "status" in g.columns:
        g["_status_norm"] = g["status"].apply(norm_status)
        success = g[g["_status_norm"] == "success"]
        if len(success):
            g = success

    # hard exclude plus dmax2 rows from main
    g = g[~g["variant"].astype(str).str.contains("plus_dmax2", na=False)]

    if len(g) == 0:
        return None

    # exact seed only
    if "seed" in g.columns:
        exact = g[g["seed"] == seed]
        if len(exact):
            g = exact

    # variant-specific cleaning
    if variant == "target_only":
        g = g[g["variant"] == "target_only"]

    elif variant == "naive":
        g = g[g["variant"] == "naive"]

    elif variant == "dfs":
        g = g[g["variant"] == "dfs"]

        # Prefer rows without AFD.
        if "uses_afd" in g.columns:
            no_afd = g[g["uses_afd"].fillna(False).astype(bool) == False]
            if len(no_afd):
                g = no_afd

        # Prefer rows without ambiguity unless rel-stack regenerated row has n_train filled.
        if not (dataset == "rel-stack" and task == "user-badge"):
            if "uses_ambiguity" in g.columns:
                no_amb = g[g["uses_ambiguity"].fillna(False).astype(bool) == False]
                if len(no_amb):
                    g = no_amb

        # For rel-stack/user-badge, the main DFS baseline must be the
        # regenerated dmax1 DFS-alone row. If n_features == 15 is absent,
        # mark this seed as missing rather than selecting old/dirty DFS rows.
        if dataset == "rel-stack" and task == "user-badge" and "n_features" in g.columns:
            nf15 = g[g["n_features"] == 15]
            if len(nf15):
                g = nf15
            else:
                return None

    elif variant == "fdhg_dmax1":
        g = g[g["variant"] == "fdhg_dmax1"]

        # Prefer FDHG rows using AFD/ambiguity if present.
        if "uses_afd" in g.columns:
            afd = g[g["uses_afd"].fillna(False).astype(bool) == True]
            if len(afd):
                g = afd

        if "uses_ambiguity" in g.columns:
            amb = g[g["uses_ambiguity"].fillna(False).astype(bool) == True]
            if len(amb):
                g = amb

    if len(g) == 0:
        return None

    # Prefer rows with n_train/n_val filled when available.
    score = pd.Series(0, index=g.index, dtype=float)

    for col in ["n_train", "n_val"]:
        if col in g.columns:
            score += g[col].notna().astype(float) * 10

    # Prefer larger n_features for FDHG when duplicate rows include partial variants.
    if "n_features" in g.columns:
        if variant == "fdhg_dmax1":
            score += g["n_features"].fillna(-1).astype(float) / 1000
        elif variant == "dfs":
            # For rel-stack regenerated DFS, n_features=15 is expected.
            # Else prefer smaller clean DFS when duplicates include DFS+ambiguity.
            if dataset == "rel-stack" and task == "user-badge":
                score -= (g["n_features"].fillna(999) - 15).abs() / 1000
            else:
                score -= g["n_features"].fillna(999).astype(float) / 1000

    # Prefer rows with real metric values.
    if "roc_auc" in g.columns:
        score += g["roc_auc"].notna().astype(float)

    chosen_idx = score.sort_values(ascending=False).index[0]
    row = g.loc[chosen_idx].copy()

    # Main table metadata correction for known regenerated rel-stack dmax1 rows.
    if dataset == "rel-stack" and task == "user-badge" and variant in {"dfs", "fdhg_dmax1"}:
        row["uses_dmax2"] = False
        row["dmax"] = 1

    row["status"] = norm_status(row.get("status", "success"))
    row["main_selector_note"] = "selected_clean_main_row"

    return row


def build_clean_rows(df, tasks, seeds, variants):
    rows = []
    missing = []

    for dataset, task in tasks:
        for variant in variants:
            for seed in seeds:
                sub = df[
                    (df["dataset"] == dataset)
                    & (df["task"] == task)
                    & (df["variant"] == variant)
                    & (df["seed"] == seed)
                ]

                # Some Phase 2 FDHG fallback/specific rows were aggregated as variant=dfs
                # because the JSON filename is dfs_agg_*. If the result_path clearly
                # comes from an _fdhg run directory, recover it as fdhg_dmax1.
                if len(sub) == 0 and variant == "fdhg_dmax1" and "result_path" in df.columns:
                    sub = df[
                        (df["dataset"] == dataset)
                        & (df["task"] == task)
                        & (df["seed"] == seed)
                        & (df["result_path"].astype(str).str.contains("_fdhg/", regex=False, na=False))
                    ].copy()
                    if len(sub):
                        sub["variant"] = "fdhg_dmax1"
                        sub["uses_afd"] = True
                        sub["uses_ambiguity"] = True
                        sub["main_selector_note"] = "recovered_fdhg_from_fdhg_path"

                        # If FDHG has exactly same feature count/metrics as DFS, treat as fallback.
                        dfs_ref = df[
                            (df["dataset"] == dataset)
                            & (df["task"] == task)
                            & (df["seed"] == seed)
                            & (df["variant"] == "dfs")
                            & (~df["result_path"].astype(str).str.contains("_fdhg/", regex=False, na=False))
                        ]
                        if len(dfs_ref):
                            for idx in sub.index:
                                same = False
                                for _, drow in dfs_ref.iterrows():
                                    same = (
                                        abs(float(sub.loc[idx, "roc_auc"]) - float(drow["roc_auc"])) < 1e-12
                                        and abs(float(sub.loc[idx, "log_loss"]) - float(drow["log_loss"])) < 1e-12
                                        and int(sub.loc[idx, "n_features"]) == int(drow["n_features"])
                                    )
                                    if same:
                                        break
                                sub.loc[idx, "is_fdhg_fallback"] = bool(same)

                picked = pick_one(sub, dataset, task, variant, seed)

                if picked is None:
                    missing.append({
                        "dataset": dataset,
                        "task": task,
                        "variant": variant,
                        "seed": seed,
                        "reason": "missing_clean_row",
                    })
                else:
                    rows.append(picked)

    clean = pd.DataFrame(rows)
    missing = pd.DataFrame(missing)

    return clean, missing


def summarize(clean):
    rows = []

    if len(clean) == 0:
        return pd.DataFrame()

    for (dataset, task, variant), g in clean.groupby(["dataset", "task", "variant"]):
        row = {
            "dataset": dataset,
            "task": task,
            "variant": variant,
            "n_runs": len(g),
            "seeds": ",".join(map(str, sorted(g["seed"].dropna().astype(int).unique()))),
            "n_features_mean": g["n_features"].mean() if "n_features" in g else np.nan,
            "n_features_std": g["n_features"].std(ddof=1) if "n_features" in g and len(g) > 1 else 0.0,
        }

        for m in METRICS:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["dataset", "task", "variant"])


def add_delta(summary):
    rows = []

    for dataset, task in MAIN_TASKS:
        s = summary[(summary["dataset"] == dataset) & (summary["task"] == task)]
        dfs = s[s["variant"] == "dfs"]
        fdhg = s[s["variant"] == "fdhg_dmax1"]

        if len(dfs) and len(fdhg):
            row = {
                "dataset": dataset,
                "task": task,
                "dfs_n_runs": int(dfs["n_runs"].iloc[0]),
                "fdhg_n_runs": int(fdhg["n_runs"].iloc[0]),
            }

            for m in METRICS:
                dm = f"{m}_mean"
                if dm in s.columns:
                    row[f"delta_fdhg_minus_dfs_{m}"] = (
                        float(fdhg[dm].iloc[0]) - float(dfs[dm].iloc[0])
                    )

            rows.append(row)

    return pd.DataFrame(rows)


def build_appendix_seed41(df):
    rows = []
    missing = []

    for dataset, task in APPENDIX_TASKS:
        for variant in MAIN_VARIANTS:
            sub = df[
                (df["dataset"] == dataset)
                & (df["task"] == task)
                & (df["variant"] == variant)
                & (df["seed"] == 41)
            ]

            # Same recovery for appendix seed41.
            if len(sub) == 0 and variant == "fdhg_dmax1" and "result_path" in df.columns:
                sub = df[
                    (df["dataset"] == dataset)
                    & (df["task"] == task)
                    & (df["seed"] == 41)
                    & (df["result_path"].astype(str).str.contains("_fdhg/", regex=False, na=False))
                ].copy()
                if len(sub):
                    sub["variant"] = "fdhg_dmax1"
                    sub["uses_afd"] = True
                    sub["uses_ambiguity"] = True
                    sub["main_selector_note"] = "recovered_fdhg_from_fdhg_path"

                    dfs_ref = df[
                        (df["dataset"] == dataset)
                        & (df["task"] == task)
                        & (df["seed"] == 41)
                        & (df["variant"] == "dfs")
                        & (~df["result_path"].astype(str).str.contains("_fdhg/", regex=False, na=False))
                    ]
                    if len(dfs_ref):
                        for idx in sub.index:
                            same = False
                            for _, drow in dfs_ref.iterrows():
                                same = (
                                    abs(float(sub.loc[idx, "roc_auc"]) - float(drow["roc_auc"])) < 1e-12
                                    and abs(float(sub.loc[idx, "log_loss"]) - float(drow["log_loss"])) < 1e-12
                                    and int(sub.loc[idx, "n_features"]) == int(drow["n_features"])
                                )
                                if same:
                                    break
                            sub.loc[idx, "is_fdhg_fallback"] = bool(same)

            picked = pick_one(sub, dataset, task, variant, 41)
            if picked is None:
                missing.append({
                    "dataset": dataset,
                    "task": task,
                    "variant": variant,
                    "seed": 41,
                    "reason": "missing_seed41_appendix_row",
                })
            else:
                rows.append(picked)

    return pd.DataFrame(rows), pd.DataFrame(missing)


def main():
    df = pd.read_csv(IN_PATH)

    # Normalize variant names if needed.
    df["variant"] = df["variant"].astype(str)

    clean_main, missing_main = build_clean_rows(df, MAIN_TASKS, SEEDS, MAIN_VARIANTS)
    main_summary = summarize(clean_main)
    main_deltas = add_delta(main_summary)

    appendix41, missing_appendix41 = build_appendix_seed41(df)

    clean_main.to_csv(OUT_DIR / "clean_main_4task_runs.csv", index=False)
    missing_main.to_csv(OUT_DIR / "clean_main_4task_missing.csv", index=False)
    main_summary.to_csv(OUT_DIR / "clean_main_4task_summary.csv", index=False)
    main_deltas.to_csv(OUT_DIR / "clean_main_4task_fdhg_minus_dfs.csv", index=False)

    appendix41.to_csv(OUT_DIR / "clean_appendix_7task_seed41.csv", index=False)
    missing_appendix41.to_csv(OUT_DIR / "clean_appendix_7task_seed41_missing.csv", index=False)

    print("Saved:")
    print("  results/final_tables/clean_main_4task_runs.csv")
    print("  results/final_tables/clean_main_4task_missing.csv")
    print("  results/final_tables/clean_main_4task_summary.csv")
    print("  results/final_tables/clean_main_4task_fdhg_minus_dfs.csv")
    print("  results/final_tables/clean_appendix_7task_seed41.csv")
    print("  results/final_tables/clean_appendix_7task_seed41_missing.csv")

    print("\n=== MAIN SUMMARY ===")
    if len(main_summary):
        print(main_summary.to_string(index=False))
    else:
        print("EMPTY")

    print("\n=== FDHG - DFS DELTAS ===")
    if len(main_deltas):
        print(main_deltas.to_string(index=False))
    else:
        print("EMPTY")

    print("\n=== MISSING MAIN ROWS ===")
    if len(missing_main):
        print(missing_main.to_string(index=False))
    else:
        print("No missing main rows.")

    print("\n=== MISSING APPENDIX SEED41 ROWS ===")
    if len(missing_appendix41):
        print(missing_appendix41.to_string(index=False))
    else:
        print("No missing appendix seed41 rows.")


if __name__ == "__main__":
    main()
