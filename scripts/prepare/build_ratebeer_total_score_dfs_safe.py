from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-parquet",
        required=True,
    )
    parser.add_argument(
        "--output-parquet",
        required=True,
    )

    args = parser.parse_args()

    source_path = Path(args.source_parquet)
    output_path = Path(args.output_parquet)

    raw = pd.read_parquet(source_path)

    required = {
        "created_at",
        "beer_id",
        "aroma",
    }
    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    safe = (
        raw.loc[
            raw["beer_id"].notna(),
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

    # Match the legacy DFS-safe schema exactly.
    safe["beer_id"] = safe["beer_id"].astype("int64")
    safe["aroma_history_value"] = (
        safe["aroma_history_value"].astype("float32")
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe.to_parquet(
        output_path,
        index=False,
    )

    print("source:", source_path)
    print("saved:", output_path)
    print("shape:", safe.shape)
    print("columns:", safe.columns.tolist())
    print("dtypes:")
    print(safe.dtypes.to_string())


if __name__ == "__main__":
    main()
