#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd


def read_df(path):
    return pd.read_parquet(path)


def select_cols(cols, variant, row_id_col):
    feat_cols = [c for c in cols if c != row_id_col]

    if variant == "temporal_full":
        keep = feat_cols

    elif variant == "last_only":
        keep = [c for c in feat_cols if c.endswith("::last_value")]

    elif variant == "history_only":
        keep = [
            c for c in feat_cols
            if c.endswith("::all_past_count")
            or c.endswith("::all_past_mean")
            or c.endswith("::all_past_std")
        ]

    elif variant == "recency_only":
        keep = [
            c for c in feat_cols
            if c.endswith("::time_since_last_days")
            or c.endswith("::recent_30d_count")
            or c.endswith("::recent_90d_count")
        ]

    elif variant == "trend_only":
        keep = [c for c in feat_cols if c.endswith("::trend_last_minus_mean")]

    elif variant == "standings_only":
        keep = [c for c in feat_cols if "<-standings." in c]

    elif variant == "results_only":
        keep = [c for c in feat_cols if "<-results." in c]

    elif variant == "no_statusId":
        keep = [c for c in feat_cols if "::statusId::" not in c]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return [row_id_col] + keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--row-id-col", default="__target_row_id__")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "temporal_full",
            "last_only",
            "history_only",
            "recency_only",
            "trend_only",
            "standings_only",
            "results_only",
            "no_statusId",
        ],
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]

    for variant in args.variants:
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            in_path = in_dir / f"{split}_temporal_stable.parquet"
            df = read_df(in_path)

            selected = select_cols(df.columns, variant, args.row_id_col)
            sub = df[selected].copy()

            out_path = variant_dir / f"{split}_{variant}.parquet"
            sub.to_parquet(out_path, index=False)

            print(f"[OK] {variant} {split}: shape={sub.shape} -> {out_path}")


if __name__ == "__main__":
    main()
