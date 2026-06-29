import argparse
from pathlib import Path
import os
import pandas as pd


def read_split(base_dir, split):
    base = Path(base_dir)
    candidates = [
        base / f"{split}_combined.parquet",
        base / f"{split}_features.parquet",
        base / f"{split}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p), p
    raise FileNotFoundError(f"No parquet for split={split} in {base_dir}")


def is_dmax2_col(c):
    c = str(c)
    return c.startswith("dmax2_") or c.startswith("dmax2__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--extra_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        base_df, base_path = read_split(args.base_dir, split)
        extra_df, extra_path = read_split(args.extra_dir, split)

        extra_cols = [c for c in extra_df.columns if is_dmax2_col(c)]

        print(f"\n[{split}]")
        print("base:", base_path, base_df.shape)
        print("extra:", extra_path, extra_df.shape)
        print("found dmax2 cols:", len(extra_cols))

        if len(extra_cols) == 0:
            print("extra columns:", extra_df.columns.tolist())
            raise ValueError("No dmax2 columns found in extra_df.")

        extra_only = extra_df[extra_cols].copy()

        if len(base_df) != len(extra_only):
            raise ValueError(
                f"Row mismatch for {split}: base={len(base_df)}, extra={len(extra_only)}"
            )

        duplicate = [c for c in extra_only.columns if c in base_df.columns]
        if duplicate:
            extra_only = extra_only.rename(columns={c: f"{c}__dmax2" for c in duplicate})

        merged = pd.concat(
            [base_df.reset_index(drop=True), extra_only.reset_index(drop=True)],
            axis=1,
        )

        out_path = Path(args.out_dir) / f"{split}_combined.parquet"
        merged.to_parquet(out_path, index=False)

        merged_dmax2 = [c for c in merged.columns if is_dmax2_col(c)]
        print("merged:", merged.shape)
        print("merged dmax2 cols:", len(merged_dmax2))
        print("saved:", out_path)


if __name__ == "__main__":
    main()
