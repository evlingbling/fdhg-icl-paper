from pathlib import Path
import pandas as pd


def normalize_cols(df):
    df = df.copy()
    if "target" in df.columns and "WillGetBadge" not in df.columns:
        df = df.rename(columns={"target": "WillGetBadge"})
    if "user" in df.columns and "UserId" not in df.columns:
        df = df.rename(columns={"user": "UserId"})
    return df


def add_key_occurrence(df, keys):
    df = df.copy()
    df["__key_occ"] = df.groupby(keys, sort=False).cumcount()
    return df


def build_split(split):
    juice_path = Path("outputs/phase1_juice_style_matched/rel-stack_user-badge") / f"target_with_juice_style_{split}.parquet"
    fdhg_path = Path("outputs/fdhg_heuristic/rel-stack_user-badge_sample") / f"target_with_dfs_agg_{split}.parquet"

    if not juice_path.exists():
        raise FileNotFoundError(juice_path)
    if not fdhg_path.exists():
        raise FileNotFoundError(fdhg_path)

    juice = normalize_cols(pd.read_parquet(juice_path))
    fdhg = normalize_cols(pd.read_parquet(fdhg_path))

    keys = ["timestamp", "UserId"]
    label = "WillGetBadge"

    for c in keys:
        if c not in juice.columns or c not in fdhg.columns:
            raise KeyError(f"missing key={c}; juice={list(juice.columns)} fdhg={list(fdhg.columns)}")

    residual_cols = [
        c for c in fdhg.columns
        if c.startswith("f_amb__")
    ]

    if not residual_cols:
        raise RuntimeError(f"No f_amb residual columns found in {fdhg_path}")

    # If duplicate timestamp/UserId exists, use within-key occurrence to avoid many-to-many explosion.
    juice = add_key_occurrence(juice, keys)
    fdhg = add_key_occurrence(fdhg, keys)

    merge_keys = keys + ["__key_occ"]

    keep_from_fdhg = merge_keys + residual_cols
    if label in fdhg.columns and label not in juice.columns:
        keep_from_fdhg.append(label)

    # Use FDHG rows as the left side, so train uses the same sampled rows as prior FDHG/DFS runs.
    left_cols = merge_keys + [c for c in juice.columns if c not in merge_keys]
    out = fdhg[merge_keys].merge(
        juice[left_cols],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        fdhg[keep_from_fdhg],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    out = out.drop(columns=["__key_occ"], errors="ignore")

    # Allow standard aggregation operators such as nunique.
    # Forbid only FDHG-specific machinery beyond the intended f_amb residual block.
    forbidden_bad = [
        c for c in out.columns
        if (
            "afd" in c.lower()
            or "ranker" in c.lower()
            or "uniqueness_penalty" in c.lower()
            or "fdhg::" in c
        )
    ]
    if forbidden_bad:
        raise RuntimeError(f"Unexpected forbidden non-residual columns: {forbidden_bad}")

    out_dir = Path("outputs/phase1_juice_style_plus_fdhg_residual/rel-stack_user-badge")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"target_with_juice_plus_fdhg_residual_{split}.parquet"
    out.to_parquet(out_path, index=False)

    print("\n" + "="*100)
    print(split)
    print("juice:", juice_path, juice.shape)
    print("fdhg:", fdhg_path, fdhg.shape)
    print("out:", out_path, out.shape)
    print("residual_cols:", residual_cols)
    print("missing after merge:", out.isna().sum().sum())


for split in ["train", "val"]:
    build_split(split)
