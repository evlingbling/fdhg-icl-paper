from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", required=True)
    args = parser.parse_args()

    root = Path(args.inspect_dir)

    child_path = root / "table_beer_ratings.parquet"
    child = pd.read_parquet(child_path)

    required_child = {
        "created_at",
        "beer_id",
        "aroma",
    }
    missing = required_child - set(child.columns)
    if missing:
        raise ValueError(
            f"Child table missing columns: {sorted(missing)}"
        )

    safe_child = (
        child.loc[
            child["beer_id"].notna(),
            ["created_at", "beer_id", "aroma"],
        ]
        .rename(
            columns={
                "created_at": "timestamp",
                "aroma": "aroma_history_value",
            }
        )
        .reset_index(drop=True)
    )

    safe_child["beer_id"] = (
        safe_child["beer_id"].astype("int64")
    )
    safe_child["aroma_history_value"] = (
        safe_child["aroma_history_value"].astype("float32")
    )

    safe_child.to_parquet(child_path, index=False)

    print(
        f"child: saved {child_path} "
        f"shape={safe_child.shape}"
    )

    for split in ["train", "val"]:
        target_path = root / f"target_{split}.parquet"
        target = pd.read_parquet(target_path)

        if "beer_id" not in target.columns:
            raise ValueError(
                f"beer_id missing from {target_path}"
            )

        n_missing = int(target["beer_id"].isna().sum())

        target["beer_id"] = (
            target["beer_id"]
            .fillna(-1)
            .astype("int64")
        )

        target.to_parquet(target_path, index=False)

        print(
            f"{split}: saved {target_path} "
            f"shape={target.shape} "
            f"filled_missing_beer_id={n_missing}"
        )


if __name__ == "__main__":
    main()
