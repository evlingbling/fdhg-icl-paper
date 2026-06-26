from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def is_listlike(x):
    return isinstance(x, (list, tuple, np.ndarray))


def make_pairwise(df, universe_places, split, seed=41, neg_per_pos=1, max_rows=None):
    rng = np.random.default_rng(seed)
    rows = []

    feature_cols = [c for c in df.columns if c not in {"place_id"}]

    for _, r in df.iterrows():
        positives = r["place_id"]

        if not is_listlike(positives):
            positives = [] if pd.isna(positives) else [positives]

        positives = [int(x) for x in positives]
        pos_set = set(positives)

        base = {c: r[c] for c in feature_cols}

        for pid in positives:
            rr = dict(base)
            rr["candidate_place_id"] = pid
            rr["label"] = 1
            rows.append(rr)

        n_neg = max(1, len(positives) * neg_per_pos)
        if len(universe_places) > 0:
            negs = []
            tries = 0
            while len(negs) < n_neg and tries < n_neg * 50:
                cand = int(universe_places[rng.integers(0, len(universe_places))])
                if cand not in pos_set:
                    negs.append(cand)
                tries += 1

            for pid in negs:
                rr = dict(base)
                rr["candidate_place_id"] = pid
                rr["label"] = 0
                rows.append(rr)

        if max_rows and len(rows) >= max_rows:
            break

    out = pd.DataFrame(rows)

    if max_rows and len(out) > max_rows:
        out = out.sample(max_rows, random_state=seed)

    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfs-dir", required=True)
    ap.add_argument("--fdhg-dir", required=True)
    ap.add_argument("--inspect-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--max-train-rows", type=int, default=10000)
    ap.add_argument("--max-val-rows", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    places_path = Path(args.inspect_dir) / "table_places.parquet"
    if places_path.exists():
        places = pd.read_parquet(places_path)
        universe_places = places["place_id"].dropna().astype(int).unique()
    else:
        # fallback from target lists
        target = pd.read_parquet(Path(args.dfs_dir) / "target_with_dfs_agg_train.parquet")
        vals = []
        for x in target["place_id"].dropna():
            if is_listlike(x):
                vals.extend(list(x))
        universe_places = np.array(sorted(set(map(int, vals))))

    for variant, in_dir in [
        ("dfs", Path(args.dfs_dir)),
        ("fdhg", Path(args.fdhg_dir)),
    ]:
        for split, max_rows in [
            ("train", args.max_train_rows),
            ("val", args.max_val_rows),
        ]:
            p = in_dir / f"target_with_dfs_agg_{split}.parquet"
            df = pd.read_parquet(p)

            pair = make_pairwise(
                df,
                universe_places=universe_places,
                split=split,
                seed=args.seed,
                neg_per_pos=args.neg_per_pos,
                max_rows=max_rows,
            )

            out_path = out_dir / f"{variant}_{split}_pairwise.parquet"
            pair.to_parquet(out_path, index=False)

            print(variant, split, pair.shape, "saved:", out_path)
            print(pair["label"].value_counts().to_string())
            print(pair.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
