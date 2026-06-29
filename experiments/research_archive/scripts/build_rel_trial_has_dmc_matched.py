from pathlib import Path

import numpy as np
import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task


DATASET = "rel-trial"
TASK = "studies-has_dmc"

DMAX2_ROOT = Path(
    "results/rel_trial_studies_has_dmc_dmax2_all_seed41/"
    "rel_trial_studies_has_dmc_dmax2_all_seed41_topk64"
)

OUT = Path("results/rel_trial_studies_has_dmc_matched")
OUT.mkdir(parents=True, exist_ok=True)

BRIDGES = [
    ("interventions_studies", "intervention_id"),
    ("facilities_studies", "facility_id"),
    ("conditions_studies", "condition_id"),
    ("sponsors_studies", "sponsor_id"),
]

dataset = get_dataset(DATASET, download=False)
db = dataset.get_db()
task = get_task(DATASET, TASK, download=False)


def get_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "df"):
        return obj.df.copy()
    raise TypeError(type(obj))


for split in ["train", "val", "test"]:
    task_df = get_df(task.get_table(split))
    task_df["start_date"] = pd.to_datetime(
        task_df["start_date"],
        errors="coerce",
    )

    clean = task_df.copy()

    for bridge_table, second_key in BRIDGES:
        child = get_df(db.table_dict[bridge_table])[
            ["nct_id", second_key, "date"]
        ].copy()

        child["date"] = pd.to_datetime(
            child["date"],
            errors="coerce",
        )

        merged = task_df[
            ["nct_id", "start_date"]
        ].merge(
            child,
            on="nct_id",
            how="left",
        )

        merged = merged[
            merged["date"].isna()
            | merged["start_date"].isna()
            | (merged["date"] <= merged["start_date"])
        ].copy()

        grouped = merged.groupby("nct_id", sort=False)

        prefix = bridge_table.removesuffix("_studies")

        agg = grouped.agg(
            **{
                f"f_dfs_clean_{prefix}_count": (
                    second_key,
                    "nunique",
                ),
                f"f_dfs_clean_{prefix}_last_date": (
                    "date",
                    "max",
                ),
            }
        ).reset_index()

        clean = clean.merge(
            agg,
            on="nct_id",
            how="left",
        )

        count_col = f"f_dfs_clean_{prefix}_count"
        last_col = f"f_dfs_clean_{prefix}_last_date"
        days_col = f"f_dfs_clean_{prefix}_days_since_last"

        clean[count_col] = (
            clean[count_col]
            .fillna(0)
            .astype(float)
        )

        clean[days_col] = (
            clean["start_date"] - clean[last_col]
        ).dt.total_seconds() / 86400.0

        clean[f"{days_col}__is_missing"] = (
            clean[days_col]
            .isna()
            .astype(int)
        )

        clean[days_col] = (
            clean[days_col]
            .fillna(0)
            .clip(lower=0)
        )

        clean = clean.drop(columns=[last_col])

    dmax2_combined = pd.read_parquet(
        DMAX2_ROOT / f"{split}_combined.parquet"
    ).reset_index(drop=True)

    clean = clean.reset_index(drop=True)

    if len(clean) != len(dmax2_combined):
        raise ValueError(
            f"{split}: row-count mismatch: "
            f"{len(clean)} vs {len(dmax2_combined)}"
        )

    if not clean["nct_id"].equals(dmax2_combined["nct_id"]):
        raise ValueError(
            f"{split}: nct_id row alignment mismatch"
        )

    all_dmax2_cols = [
        c for c in dmax2_combined.columns
        if c.startswith("dmax2_")
    ]

    residual_cols = [
        c for c in all_dmax2_cols
        if "__count_distinct_" not in c
    ]

    matched = pd.concat(
        [
            clean,
            dmax2_combined[residual_cols],
        ],
        axis=1,
    )

    clean.to_parquet(
        OUT / f"{split}_dfs_clean.parquet",
        index=False,
    )

    matched.to_parquet(
        OUT / f"{split}_dfs_clean_plus_dmax2.parquet",
        index=False,
    )

    dmax2_only = pd.concat(
        [
            task_df.reset_index(drop=True),
            dmax2_combined[residual_cols],
        ],
        axis=1,
    )

    dmax2_only.to_parquet(
        OUT / f"{split}_dmax2_residual_only.parquet",
        index=False,
    )

    dfs_cols = [
        c for c in clean.columns
        if c.startswith("f_dfs_clean_")
    ]

    print(f"\n=== {split} ===")
    print("task rows:", len(task_df))
    print("DFS-clean features:", len(dfs_cols))
    print("dmax2 residual features:", len(residual_cols))
    print("candidate features:", len(dfs_cols) + len(residual_cols))

    print("\nDFS-clean:")
    for c in dfs_cols:
        print(" -", c)

    print("\ndmax2 residual:")
    for c in residual_cols:
        print(" -", c)
