from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an enriched inspect bundle by attaching an entity key "
            "from a source table using a row-index primary key."
        )
    )
    parser.add_argument("--inspect-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--primary-key-col", default="primary_key")
    parser.add_argument("--entity-key", required=True)
    parser.add_argument("--source-entity-col", default=None)
    parser.add_argument("--verify-time-col", default=None)
    parser.add_argument("--verify-label-col", default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Target splits to enrich. Defaults to train and val.",
    )
    args = parser.parse_args()

    inspect_dir = Path(args.inspect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_entity_col = args.source_entity_col or args.entity_key
    source_path = inspect_dir / f"table_{args.source_table}.parquet"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source table: {source_path}")

    source_df = pd.read_parquet(source_path).reset_index(drop=True)

    if source_entity_col not in source_df.columns:
        raise KeyError(
            f"Column {source_entity_col!r} not found in {source_path}. "
            f"Available columns: {list(source_df.columns)}"
        )

    # Copy standardized raw tables and metadata unchanged.
    for path in inspect_dir.iterdir():
        if path.is_file() and not path.name.startswith("target_"):
            shutil.copy2(path, out_dir / path.name)

    for split in args.splits:
        target_path = inspect_dir / f"target_{split}.parquet"

        if not target_path.exists():
            continue

        target_df = pd.read_parquet(target_path).reset_index(drop=True)

        if args.primary_key_col not in target_df.columns:
            raise KeyError(
                f"Column {args.primary_key_col!r} not found in {target_path}"
            )

        keys = pd.to_numeric(
            target_df[args.primary_key_col],
            errors="raise",
        ).astype("int64")

        valid = (keys >= 0) & (keys < len(source_df))
        if not bool(valid.all()):
            bad = keys[~valid].head(10).tolist()
            raise IndexError(
                f"{split}: primary keys outside source table range: {bad}"
            )

        selected = source_df.iloc[keys.to_numpy()].reset_index(drop=True)

        if args.verify_time_col:
            if args.verify_time_col not in target_df.columns:
                raise KeyError(
                    f"{args.verify_time_col!r} missing from target {split}"
                )
            if args.verify_time_col not in selected.columns:
                raise KeyError(
                    f"{args.verify_time_col!r} missing from source table"
                )

            target_time = pd.to_datetime(target_df[args.verify_time_col])
            source_time = pd.to_datetime(selected[args.verify_time_col])

            if not target_time.equals(source_time):
                raise ValueError(f"{split}: timestamp verification failed")

        if args.verify_label_col:
            if args.verify_label_col not in target_df.columns:
                raise KeyError(
                    f"{args.verify_label_col!r} missing from target {split}"
                )
            if args.verify_label_col not in selected.columns:
                raise KeyError(
                    f"{args.verify_label_col!r} missing from source table"
                )

            if not target_df[args.verify_label_col].reset_index(drop=True).equals(
                selected[args.verify_label_col].reset_index(drop=True)
            ):
                raise ValueError(f"{split}: label verification failed")

        target_df[args.entity_key] = selected[source_entity_col].to_numpy()

        out_path = out_dir / f"target_{split}.parquet"
        target_df.to_parquet(out_path, index=False)

        print(
            f"[{split}] saved {out_path} "
            f"shape={target_df.shape} entity_key={args.entity_key}"
        )


if __name__ == "__main__":
    main()
