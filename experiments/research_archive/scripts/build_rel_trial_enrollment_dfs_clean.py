from pathlib import Path

import numpy as np
import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task

DATASET = "rel-trial"
TASK = "studies-enrollment"

DMAX2_ROOT = Path(
    "results/rel_trial_studies_enrollment_dmax2_all_seed41/"
    "rel_trial_studies_enrollment_dmax2_all_seed41_topk64"
)

OUT = Path("results/rel_trial_studies_enrollment_matched_seed41")
OUT.mkdir(parents=True, exist_ok=True)

dataset = get_dataset(DATASET, download=False)
db = dataset.get_db()
task = get_task(DATASET, TASK, download=False)

facilities_studies = db.table_dict["facilities_studies"].df[
    ["nct_id", "facility_id", "date"]
].copy()

facilities_studies["date"] = pd.to_datetime(
    facilities_studies["date"],
    errors="coerce",
)

for split in ["train", "val", "test"]:
    task_df = task.get_table(split).df.copy()
    task_df["start_date"] = pd.to_datetime(
        task_df["start_date"],
        errors="coerce",
    )

    # One task row per nct_id. Attach the corresponding target timestamp
    # before applying the temporal cutoff.
    child = facilities_studies.merge(
        task_df[["nct_id", "start_date"]],
        on="nct_id",
        how="inner",
    )

    child = child[
        child["date"].isna()
        | child["start_date"].isna()
        | (child["date"] <= child["start_date"])
    ].copy()

    grouped = child.groupby("nct_id", sort=False)

    agg = grouped.agg(
        f_dfs_clean_facility_count=("facility_id", "count"),
        f_dfs_clean_last_facility_date=("date", "max"),
    ).reset_index()

    clean = task_df.merge(
        agg,
        on="nct_id",
        how="left",
    )

    clean["f_dfs_clean_facility_count"] = (
        clean["f_dfs_clean_facility_count"]
        .fillna(0)
        .astype(float)
    )

    clean["f_dfs_clean_days_since_last_facility"] = (
        clean["start_date"]
        - clean["f_dfs_clean_last_facility_date"]
    ).dt.total_seconds() / 86400.0

    clean["f_dfs_clean_days_since_last_facility__is_missing"] = (
        clean["f_dfs_clean_days_since_last_facility"]
        .isna()
        .astype(int)
    )

    clean["f_dfs_clean_days_since_last_facility"] = (
        clean["f_dfs_clean_days_since_last_facility"]
        .fillna(0)
        .clip(lower=0)
    )

    clean = clean.drop(
        columns=["f_dfs_clean_last_facility_date"]
    )

    dmax2 = pd.read_parquet(
        DMAX2_ROOT / f"{split}_features.parquet"
    ).reset_index(drop=True)

    if len(clean) != len(dmax2):
        raise ValueError(
            f"{split}: row mismatch: "
            f"clean={len(clean)}, dmax2={len(dmax2)}"
        )

    dmax2_cols = [
        c for c in dmax2.columns
        if c.startswith("dmax2_")
    ]

    matched = pd.concat(
        [
            clean.reset_index(drop=True),
            dmax2[dmax2_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    clean_path = OUT / f"{split}_dfs_clean.parquet"
    matched_path = OUT / f"{split}_dfs_clean_plus_dmax2.parquet"

    clean.to_parquet(clean_path, index=False)
    matched.to_parquet(matched_path, index=False)

    print(f"\n[{split}]")
    print("DFS-clean:", clean.shape, clean_path)
    print("DFS-clean+dmax2:", matched.shape, matched_path)
    print("DFS features:", [
        c for c in clean.columns
        if c.startswith("f_dfs_clean_")
    ])
    print("dmax2 features:", len(dmax2_cols))
