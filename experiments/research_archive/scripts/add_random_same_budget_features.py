import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_random_features(df: pd.DataFrame, n_random: int, seed: int, split: str):
    out = df.copy()
    rng = np.random.default_rng(seed)

    n = len(out)

    # Random Gaussian features. These preserve feature budget but contain no row-aligned relational signal.
    for j in range(n_random):
        out[f"f_random_same_budget_{j:02d}"] = rng.normal(loc=0.0, scale=1.0, size=n)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="Input DFS/w-o ambiguity feature directory")
    parser.add_argument("--out-dir", required=True, help="Output random same-budget feature directory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-random", type=int, default=8)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_i, split in enumerate(["train", "val", "test"]):
        src = in_dir / f"target_with_dfs_agg_{split}.parquet"
        dst = out_dir / f"target_with_dfs_agg_{split}.parquet"

        if not src.exists():
            raise FileNotFoundError(f"Missing input file: {src}")

        df = pd.read_parquet(src)
        out = add_random_features(
            df,
            n_random=args.n_random,
            seed=args.seed + split_i * 10000,
            split=split,
        )
        out.to_parquet(dst, index=False)
        print(f"Wrote {dst} | rows={len(out)} | added_random_features={args.n_random}")

    pd.DataFrame({
        "feature_name": [f"f_random_same_budget_{j:02d}" for j in range(args.n_random)],
        "block": ["random_same_budget"] * args.n_random,
    }).to_csv(out_dir / "random_same_budget_manifest.csv", index=False)

    print(f"Saved manifest: {out_dir / 'random_same_budget_manifest.csv'}")


if __name__ == "__main__":
    main()
