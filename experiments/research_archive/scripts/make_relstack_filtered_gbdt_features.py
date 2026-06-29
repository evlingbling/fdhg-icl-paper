import os
from pathlib import Path
import pandas as pd

SEEDS = [41, 42, 43, 44]

BASE_TEMPLATE = (
    "results/extension_c_ranker/features/"
    "rel_stack_user_badge_regen_dmax1_fdhg_full_plus_dmax2_supervised_ap_topk16_seed{seed}"
)

OUT_ROOT = Path("results/gbdt_filtered_features")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ID_COLS = ["timestamp", "UserId"]
TARGET_COL = "WillGetBadge"

def make_variant(df, variant):
    keep = []

    for c in df.columns:
        if c == TARGET_COL:
            keep.append(c)
        elif c in ID_COLS:
            keep.append(c)
        elif variant == "dfs":
            # DFS/base aggregation features only.
            if c.startswith("f_user_") or c.startswith("f_badges_"):
                keep.append(c)
        elif variant == "fdhg_dmax1_full":
            # DFS + dmax1 ambiguity features.
            if c.startswith("f_user_") or c.startswith("f_badges_") or c.startswith("f_amb_"):
                keep.append(c)
        elif variant == "fdhg_dmax1_plus_dmax2_supervised_ap_topk16":
            # Full matrix except no extra filtering.
            if c not in []:
                keep.append(c)
        else:
            raise ValueError(f"Unknown variant: {variant}")

    out = df[keep].copy()
    return out

variants = [
    "dfs",
    "fdhg_dmax1_full",
    "fdhg_dmax1_plus_dmax2_supervised_ap_topk16",
]

for seed in SEEDS:
    base = Path(BASE_TEMPLATE.format(seed=seed))
    train_path = base / "train_combined.parquet"
    val_path = base / "val_combined.parquet"

    if not train_path.exists() or not val_path.exists():
        print(f"[SKIP] seed={seed} missing {base}")
        continue

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    for variant in variants:
        out_dir = OUT_ROOT / f"rel_stack_user_badge_{variant}_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_out = make_variant(train_df, variant)
        val_out = make_variant(val_df, variant)

        train_out.to_parquet(out_dir / "train_combined.parquet", index=False)
        val_out.to_parquet(out_dir / "val_combined.parquet", index=False)

        print(
            f"[OK] seed={seed} variant={variant} "
            f"train_shape={train_out.shape} val_shape={val_out.shape} "
            f"out={out_dir}"
        )
