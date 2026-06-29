#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--row-id-col", default="__target_row_id__")
    args = parser.parse_args()

    merged = None

    for input_path in args.inputs:
        df = read_table(input_path)

        if args.row_id_col not in df.columns:
            raise ValueError(f"{input_path} missing row id column: {args.row_id_col}")

        if merged is None:
            merged = df.copy()
        else:
            overlap = set(merged.columns).intersection(set(df.columns))
            overlap = overlap - {args.row_id_col}

            if overlap:
                raise ValueError(
                    f"Overlapping feature columns found in {input_path}: "
                    f"{sorted(overlap)[:20]}"
                )

            merged = merged.merge(df, on=args.row_id_col, how="outer")

    merged = merged.sort_values(args.row_id_col).reset_index(drop=True)

    write_table(merged, args.out_path)

    print(f"[OK] wrote merged temporal features: {args.out_path}")
    print(f"[OK] shape: {merged.shape}")
    print("[OK] first columns:")
    for c in merged.columns[:20]:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
