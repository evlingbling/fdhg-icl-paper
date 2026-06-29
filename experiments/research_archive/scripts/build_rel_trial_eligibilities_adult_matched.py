from pathlib import Path

import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task


DATASET = "rel-trial"
TASK = "eligibilities-adult"

OUT = Path("results/rel_trial_eligibilities_adult_matched")
OUT.mkdir(parents=True, exist_ok=True)

BRIDGES = [
    ("interventions_studies", "intervention_id", "interventions", "intervention_id", ["mesh_term"]),
    ("facilities_studies", "facility_id", "facilities", "facility_id", ["name", "city", "state", "zip", "country"]),
    ("conditions_studies", "condition_id", "conditions", "condition_id", ["mesh_term"]),
    ("sponsors_studies", "sponsor_id", "sponsors", "sponsor_id", ["name", "agency_class"]),
]


def get_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "df"):
        return obj.df.copy()
    raise TypeError(f"Cannot extract DataFrame from {type(obj)}")


dataset = get_dataset(DATASET, download=False)
db = dataset.get_db()
task = get_task(DATASET, TASK, download=False)

eligibility_lookup = get_df(db.table_dict["eligibilities"])[
    ["id", "nct_id"]
].copy()

if not eligibility_lookup["id"].is_unique:
    raise ValueError("eligibilities.id is not unique")

if not eligibility_lookup["nct_id"].is_unique:
    raise ValueError("eligibilities.nct_id is not unique")


for split in ["train", "val", "test"]:
    task_df = get_df(task.get_table(split))
    task_df["date"] = pd.to_datetime(
        task_df["date"],
        errors="coerce",
    )

    # Preserve official split-row order and map id -> nct_id.
    base = task_df.merge(
        eligibility_lookup,
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if base["nct_id"].isna().any():
        missing = int(base["nct_id"].isna().sum())
        raise ValueError(
            f"{split}: {missing} rows failed id -> nct_id mapping"
        )

    if len(base) != len(task_df):
        raise ValueError(
            f"{split}: row count changed during id -> nct_id mapping"
        )

    clean = base.copy()
    residual_frames = []

    for (
        bridge_table,
        bridge_second_key,
        second_table,
        second_pkey,
        second_usable_cols,
    ) in BRIDGES:
        bridge = get_df(db.table_dict[bridge_table])[
            ["nct_id", bridge_second_key, "date"]
        ].copy()

        bridge["date"] = pd.to_datetime(
            bridge["date"],
            errors="coerce",
        )

        merged = base[
            ["id", "nct_id", "date"]
        ].merge(
            bridge,
            on="nct_id",
            how="left",
            suffixes=("_task", "_bridge"),
            sort=False,
        )

        merged = merged[
            merged["date_bridge"].isna()
            | merged["date_task"].isna()
            | (merged["date_bridge"] <= merged["date_task"])
        ].copy()

        prefix = bridge_table.removesuffix("_studies")

        grouped = merged.groupby("id", sort=False)

        dfs_agg = grouped.agg(
            **{
                f"f_dfs_clean_{prefix}_count": (
                    bridge_second_key,
                    "nunique",
                ),
                f"f_dfs_clean_{prefix}_last_date": (
                    "date_bridge",
                    "max",
                ),
            }
        ).reset_index()

        clean = clean.merge(
            dfs_agg,
            on="id",
            how="left",
            validate="one_to_one",
            sort=False,
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
            clean["date"] - clean[last_col]
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

        second = get_df(db.table_dict[second_table])[
            [second_pkey] + second_usable_cols
        ].copy()

        enriched = merged.merge(
            second,
            left_on=bridge_second_key,
            right_on=second_pkey,
            how="left",
            sort=False,
        )

        enriched_grouped = enriched.groupby("id", sort=False)

        residual = pd.DataFrame({
            "id": base["id"].to_numpy()
        })

        for col in second_usable_cols:
            residual[
                f"dmax2_{bridge_table}_{bridge_second_key}_{second_table}"
                f"__nunique_{col}"
            ] = (
                residual["id"]
                .map(
                    enriched_grouped[col]
                    .nunique(dropna=True)
                    .to_dict()
                )
                .fillna(0)
                .astype(float)
            )

        residual_frames.append(
            residual.set_index("id")
        )

    residual_all = pd.concat(
        residual_frames,
        axis=1,
    ).reset_index()

    matched = clean.merge(
        residual_all,
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    residual_only = base.merge(
        residual_all,
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    dfs_cols = [
        c for c in clean.columns
        if c.startswith("f_dfs_clean_")
    ]

    residual_cols = [
        c for c in matched.columns
        if c.startswith("dmax2_")
    ]

    # Keep nct_id only as an alignment key, never as a model feature.
    clean.to_parquet(
        OUT / f"{split}_dfs_clean.parquet",
        index=False,
    )

    residual_only.to_parquet(
        OUT / f"{split}_dmax2_residual_only.parquet",
        index=False,
    )

    matched.to_parquet(
        OUT / f"{split}_dfs_clean_plus_dmax2.parquet",
        index=False,
    )

    print(f"\n=== {split} ===")
    print("rows:", len(base))
    print("DFS-clean features:", len(dfs_cols))
    print("dmax2 residual features:", len(residual_cols))
    print("combined features:", len(dfs_cols) + len(residual_cols))

    print("\nDFS-clean:")
    for c in dfs_cols:
        print(" -", c)

    print("\ndmax2 residual:")
    for c in residual_cols:
        print(" -", c)
