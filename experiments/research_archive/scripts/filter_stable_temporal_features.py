#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def read_parquet(path):
    return pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--row-id-col", default="__target_row_id__")
    parser.add_argument("--max-nan-ratio", type=float, default=0.95)
    args = parser.parse_args()

    train = read_parquet(args.train_path)
    val = read_parquet(args.val_path)
    test = read_parquet(args.test_path)

    feature_cols = [c for c in train.columns if c != args.row_id_col]

    keep_cols = []
    drop_info = []

    for c in feature_cols:
        train_nan = train[c].isna().mean()
        val_nan = val[c].isna().mean()
        test_nan = test[c].isna().mean()

        max_nan = max(train_nan, val_nan, test_nan)

        if max_nan <= args.max_nan_ratio:
            keep_cols.append(c)
        else:
            drop_info.append(
                {
                    "feature": c,
                    "train_nan": train_nan,
                    "val_nan": val_nan,
                    "test_nan": test_nan,
                    "max_nan": max_nan,
                }
            )

    selected_cols = [args.row_id_col] + keep_cols

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train[selected_cols].to_parquet(out_dir / "train_temporal_stable.parquet", index=False)
    val[selected_cols].to_parquet(out_dir / "val_temporal_stable.parquet", index=False)
    test[selected_cols].to_parquet(out_dir / "test_temporal_stable.parquet", index=False)

    pd.DataFrame(drop_info).to_csv(out_dir / "dropped_temporal_features.csv", index=False)

    print("[OK] stable temporal features written")
    print(f"[OK] kept features: {len(keep_cols)}")
    print(f"[OK] dropped features: {len(drop_info)}")
    print(f"[OK] output dir: {out_dir}")

    if drop_info:
        print("\nDropped examples:")
        print(pd.DataFrame(drop_info).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
