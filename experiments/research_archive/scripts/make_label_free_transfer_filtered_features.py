import argparse
import os
from pathlib import Path

import pandas as pd


def find_first_existing(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def autodetect_all_dir(seed):
    patterns = [
        f"results/**/rel_stack_user_badge*dmax2*all*seed{seed}*/train_combined.parquet",
        f"results/**/rel-stack_user-badge*dmax2*all*seed{seed}*/train_combined.parquet",
        f"results/**/dmax2_all*seed{seed}*/train_combined.parquet",
        f"results/**/all_only*seed{seed}*/train_combined.parquet",
    ]
    hits = []
    for pat in patterns:
        hits.extend(Path(".").glob(pat))
    hits = sorted(set(hits))
    if not hits:
        return None
    return hits[0].parent


def autodetect_dmax1_dir(seed):
    patterns = [
        f"results/gbdt_filtered_features/rel_stack_user_badge_fdhg_dmax1_full_seed{seed}/train_combined.parquet",
        f"results/**/rel_stack_user_badge_fdhg_dmax1_full*seed{seed}*/train_combined.parquet",
        f"results/**/regen_dmax1_fdhg_full*seed{seed}*/train_combined.parquet",
    ]
    hits = []
    for pat in patterns:
        hits.extend(Path(".").glob(pat))
    hits = sorted(set(hits))
    if not hits:
        return None
    return hits[0].parent


def load_selected(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def pick_base_cols(df):
    base_cols = []
    for c in df.columns:
        low = c.lower()
        if c in ["target", "WillGetBadge", "willgetbadge", "label", "y"]:
            base_cols.append(c)
        elif c in ["timestamp", "UserId", "user_id", "index"]:
            base_cols.append(c)
        elif low.startswith("f_user_") or low.startswith("f_badges_") or low.startswith("f_amb_"):
            base_cols.append(c)
    return base_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--selected", default="results/label_free_transfer_ranker/rel_stack_user_badge_dmax2_label_free_topk16_seed41/selected_features.txt")
    ap.add_argument("--all-dir", default="")
    ap.add_argument("--dmax1-dir", default="")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    seed = args.seed
    selected = load_selected(args.selected)

    all_dir = Path(args.all_dir) if args.all_dir else autodetect_all_dir(seed)
    dmax1_dir = Path(args.dmax1_dir) if args.dmax1_dir else autodetect_dmax1_dir(seed)

    if all_dir is None:
        raise FileNotFoundError(
            "Could not autodetect all-candidate dmax2 directory. "
            "Pass --all-dir path/to/dir containing train_combined.parquet and val_combined.parquet."
        )
    if dmax1_dir is None:
        raise FileNotFoundError(
            "Could not autodetect dmax1 FDHG full directory. "
            "Pass --dmax1-dir path/to/dir containing train_combined.parquet and val_combined.parquet."
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"results/label_free_transfer_ranker/rel_stack_user_badge_dmax1_plus_dmax2_label_free_topk16_seed{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("all_dir:", all_dir)
    print("dmax1_dir:", dmax1_dir)
    print("out_dir:", out_dir)

    for split in ["train", "val"]:
        dmax1_path = dmax1_dir / f"{split}_combined.parquet"
        all_path = all_dir / f"{split}_combined.parquet"

        if not dmax1_path.exists():
            raise FileNotFoundError(dmax1_path)
        if not all_path.exists():
            raise FileNotFoundError(all_path)

        d1 = pd.read_parquet(dmax1_path)
        da = pd.read_parquet(all_path)

        missing = [c for c in selected if c not in da.columns]
        if missing:
            print("Missing selected columns in all matrix:")
            for c in missing:
                print("  ", c)
            raise ValueError(f"{len(missing)} selected columns missing")

        base_cols = pick_base_cols(d1)
        if not base_cols:
            raise ValueError("No base columns detected from dmax1 matrix.")

        out = pd.concat(
            [
                d1[base_cols].reset_index(drop=True),
                da[selected].reset_index(drop=True),
            ],
            axis=1,
        )

        # Deduplicate columns if any overlap.
        out = out.loc[:, ~out.columns.duplicated()]
        out.to_parquet(out_dir / f"{split}_combined.parquet", index=False)

        print(split, "shape:", out.shape)
        print("n_selected_dmax2:", len(selected))
        print("n_total_cols:", out.shape[1])

    with open(out_dir / "selected_features.txt", "w") as f:
        for c in selected:
            f.write(c + "\n")

    print("\nSaved filtered features to:", out_dir)


if __name__ == "__main__":
    main()
