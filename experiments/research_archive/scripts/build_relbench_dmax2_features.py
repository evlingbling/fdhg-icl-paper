import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


ID_PATTERNS = [
    r"(^id$)",
    r"(_id$)",
    r"(^id_)",
    r"(uuid)",
    r"(key$)",
    r"(^index$)",
    r"(unnamed)",
]

TIME_PATTERNS = [
    r"time",
    r"date",
    r"timestamp",
    r"created",
    r"updated",
]


def safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^0-9a-zA-Z_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def is_id_like(col: str) -> bool:
    c = str(col).lower()
    return any(re.search(p, c) for p in ID_PATTERNS)


def is_time_like(col: str) -> bool:
    c = str(col).lower()
    return any(re.search(p, c) for p in TIME_PATTERNS)


def get_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "df"):
        return obj.df
    raise TypeError(f"Cannot extract df from object type: {type(obj)}")


def get_table_dict(db):
    if hasattr(db, "table_dict"):
        return db.table_dict
    if hasattr(db, "tables"):
        return db.tables
    raise AttributeError("Cannot find table dictionary in RelBench db object.")


def get_table_metadata(table):
    fks = getattr(table, "fkey_col_to_pkey_table", None)
    pkey = getattr(table, "pkey_col", None)
    time_col = getattr(table, "time_col", None)

    if fks is None:
        fks = {}

    return {
        "fkey_col_to_pkey_table": fks,
        "pkey_col": pkey,
        "time_col": time_col,
    }


def infer_pkey_col(table_name, table_obj, df):
    meta_pkey = getattr(table_obj, "pkey_col", None)
    if meta_pkey is not None and meta_pkey in df.columns:
        return meta_pkey

    candidates = [
        table_name,
        table_name.rstrip("s"),
        "id",
        f"{table_name}_id",
        f"{table_name.rstrip('s')}_id",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: first ID-like unique-ish column
    n = len(df)
    for c in df.columns:
        if is_id_like(c):
            nunq = df[c].nunique(dropna=True)
            if n > 0 and nunq >= 0.7 * n:
                return c

    return None


def infer_time_col(table_obj, df):
    meta_time = getattr(table_obj, "time_col", None)
    if meta_time is not None and meta_time in df.columns:
        return meta_time

    for c in df.columns:
        if is_time_like(c):
            return c
    return None


def load_relbench(dataset_name, task_name, download=False):
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset = get_dataset(dataset_name, download=download)
    db = dataset.get_db()
    task = get_task(dataset_name, task_name, download=download)
    return db, task


def get_task_split_df(task, split):
    try:
        table = task.get_table(split)
    except TypeError:
        table = task.get_table(split, mask_input_cols=False)
    return get_df(table).copy()


def discover_dmax2_paths(db, target_key, target_time_col=None, verbose=True):
    table_dict = get_table_dict(db)

    # Materialize table info
    infos = {}
    for tname, tobj in table_dict.items():
        df = get_df(tobj)
        infos[tname] = {
            "obj": tobj,
            "df": df,
            "columns": list(df.columns),
            "pkey_col": infer_pkey_col(tname, tobj, df),
            "time_col": infer_time_col(tobj, df),
            "fks": get_table_metadata(tobj)["fkey_col_to_pkey_table"],
        }

    paths = []

    for bridge_name, bridge_info in infos.items():
        bridge_df = bridge_info["df"]
        bridge_cols = set(bridge_df.columns)

        if target_key not in bridge_cols:
            continue

        # Prefer FK metadata if available.
        candidate_second_keys = []

        fks = bridge_info["fks"] or {}
        for fk_col, pkey_table in fks.items():
            if fk_col == target_key:
                continue
            if fk_col in bridge_cols:
                candidate_second_keys.append((fk_col, pkey_table, "fk_metadata"))

        # Fallback: ID-like columns other than target_key.
        for c in bridge_df.columns:
            if c == target_key:
                continue
            if is_id_like(c) or c in infos:
                # Try matching to table by fk col name or pkey col.
                matched_table = None

                # Exact table name match.
                if c in infos:
                    matched_table = c

                # Remove suffixes.
                if matched_table is None:
                    base = re.sub(r"_id$", "", c)
                    if base in infos:
                        matched_table = base
                    elif base + "s" in infos:
                        matched_table = base + "s"

                # Match table pkey.
                if matched_table is None:
                    for tname, info in infos.items():
                        if tname == bridge_name:
                            continue
                        if info["pkey_col"] == c:
                            matched_table = tname
                            break

                if matched_table is not None:
                    tup = (c, matched_table, "name_or_pkey_inference")
                    if tup not in candidate_second_keys:
                        candidate_second_keys.append(tup)

        for second_key, second_table, source in candidate_second_keys:
            if second_table not in infos:
                continue

            second_info = infos[second_table]
            second_df = second_info["df"]
            second_pkey = second_info["pkey_col"]

            if second_pkey is None:
                continue
            if second_pkey not in second_df.columns:
                continue

            # Avoid degenerate path target <- bridge -> target table if possible.
            if second_key == target_key:
                continue

            # Determine usable second entity columns.
            usable_cols = []
            for col in second_df.columns:
                if col == second_pkey:
                    continue
                if col == target_key:
                    continue
                if col.lower() == "target":
                    continue
                if is_time_like(col):
                    continue
                if is_id_like(col):
                    continue
                usable_cols.append(col)

            if len(usable_cols) == 0:
                continue

            paths.append({
                "bridge_table": bridge_name,
                "bridge_target_key": target_key,
                "bridge_second_key": second_key,
                "bridge_time_col": bridge_info["time_col"],
                "second_table": second_table,
                "second_pkey": second_pkey,
                "second_time_col": second_info["time_col"],
                "second_usable_cols": usable_cols,
                "source": source,
            })

    if verbose:
        print("\n=== DISCOVERED DMAX=2 PATHS ===")
        if not paths:
            print("No dmax=2 paths discovered.")
        for i, p in enumerate(paths):
            print(f"\n[{i}] target <- {p['bridge_table']} -> {p['second_table']}")
            print(f"    target key: {p['bridge_target_key']}")
            print(f"    second key: bridge.{p['bridge_second_key']} -> {p['second_table']}.{p['second_pkey']}")
            print(f"    bridge time col: {p['bridge_time_col']}")
            print(f"    usable second cols ({len(p['second_usable_cols'])}): {p['second_usable_cols'][:20]}")
            print(f"    source: {p['source']}")

    return paths


def feature_score_from_train(series: pd.Series, name: str):
    nonnull = series.notna().mean()
    nunique = series.nunique(dropna=True)
    n = len(series)
    nunique_norm = math.log1p(nunique) / math.log1p(max(n, 2))
    id_penalty = 0.5 if is_id_like(name) else 0.0
    constant_penalty = 0.5 if nunique <= 1 else 0.0
    return float(nonnull + 0.25 * nunique_norm - id_penalty - constant_penalty)


def prepare_for_merge(df, cols):
    out = df[cols].copy()
    return out


def build_features_for_split(
    task_df,
    table_dict,
    paths,
    target_key,
    target_time_col,
    max_bridge_rows_per_path=None,
):
    task_df = task_df.copy()
    task_df["__row_id__"] = np.arange(len(task_df))

    # Convert target time once.
    has_target_time = target_time_col is not None and target_time_col in task_df.columns
    if has_target_time:
        task_df[target_time_col] = pd.to_datetime(task_df[target_time_col], errors="coerce")

    all_feats = pd.DataFrame({"__row_id__": task_df["__row_id__"].values})

    for path_idx, p in enumerate(paths):
        bridge_name = p["bridge_table"]
        second_name = p["second_table"]
        bridge_second_key = p["bridge_second_key"]
        second_pkey = p["second_pkey"]
        bridge_time_col = p["bridge_time_col"]

        bridge_df = get_df(table_dict[bridge_name]).copy()
        second_df = get_df(table_dict[second_name]).copy()

        if max_bridge_rows_per_path is not None and len(bridge_df) > max_bridge_rows_per_path:
            bridge_df = bridge_df.sample(max_bridge_rows_per_path, random_state=0)

        needed_bridge_cols = [target_key, bridge_second_key]
        if bridge_time_col is not None and bridge_time_col in bridge_df.columns:
            needed_bridge_cols.append(bridge_time_col)
        needed_bridge_cols = list(dict.fromkeys(needed_bridge_cols))
        bridge_df = bridge_df[needed_bridge_cols].copy()

        # task row -> bridge
        left_cols = ["__row_id__", target_key]
        if has_target_time:
            left_cols.append(target_time_col)

        merged = task_df[left_cols].merge(
            bridge_df,
            on=target_key,
            how="left",
            suffixes=("", "__bridge"),
        )

        # Temporal safety:
        # keep bridge rows whose bridge timestamp <= target timestamp.
        if has_target_time and bridge_time_col is not None and bridge_time_col in merged.columns:
            merged[bridge_time_col] = pd.to_datetime(merged[bridge_time_col], errors="coerce")
            before_mask = merged[bridge_time_col].isna() | merged[target_time_col].isna() | (
                merged[bridge_time_col] <= merged[target_time_col]
            )
            merged = merged[before_mask].copy()

        second_cols = [second_pkey] + p["second_usable_cols"]
        second_cols = [c for c in second_cols if c in second_df.columns]
        second_df = second_df[second_cols].copy()

        merged = merged.merge(
            second_df,
            left_on=bridge_second_key,
            right_on=second_pkey,
            how="left",
        )

        prefix = safe_name(
            f"dmax2__{bridge_name}__{bridge_second_key}__{second_name}"
        )

        grouped = merged.groupby("__row_id__", sort=False)
        path_feats = pd.DataFrame({"__row_id__": task_df["__row_id__"].values}).set_index("__row_id__")

        # Always include count distinct of second entity.
        cd_name = f"{prefix}__count_distinct_{safe_name(bridge_second_key)}"
        cd = grouped[bridge_second_key].nunique(dropna=True)
        path_feats[cd_name] = cd

        for col in p["second_usable_cols"]:
            if col not in merged.columns:
                continue

            s = merged[col]
            col_safe = safe_name(col)

            if pd.api.types.is_numeric_dtype(s):
                agg = grouped[col].agg(["mean", "std", "min", "max"])
                for op in ["mean", "std", "min", "max"]:
                    fname = f"{prefix}__{op}_{col_safe}"
                    path_feats[fname] = agg[op]
            else:
                # categorical/object/bool: only nunique for MVP
                fname = f"{prefix}__nunique_{col_safe}"
                path_feats[fname] = grouped[col].nunique(dropna=True)

        path_feats = path_feats.reset_index()
        all_feats = all_feats.merge(path_feats, on="__row_id__", how="left")

    all_feats = all_feats.drop(columns=["__row_id__"])
    all_feats = all_feats.replace([np.inf, -np.inf], np.nan)
    all_feats = all_feats.fillna(0)

    # Cast bool/object that slipped through.
    for c in all_feats.columns:
        if all_feats[c].dtype == "bool":
            all_feats[c] = all_feats[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(all_feats[c]):
            all_feats[c] = pd.to_numeric(all_feats[c], errors="coerce").fillna(0)

    return all_feats


def select_features(train_feats, variant, top_k, seed):
    cols = list(train_feats.columns)

    if variant == "dmax2_all":
        return cols

    scores = [(c, feature_score_from_train(train_feats[c], c)) for c in cols]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    if variant == "dmax2_topk":
        return [c for c, _ in scores[:top_k]]

    if variant == "dmax2_random_same_budget":
        rng = random.Random(seed)
        cols_copy = cols[:]
        rng.shuffle(cols_copy)
        return cols_copy[:top_k]

    raise ValueError(f"Unknown variant: {variant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target_key", required=True)
    parser.add_argument("--target_time_col", default=None)
    parser.add_argument("--variant", default="discover",
                        choices=["discover", "dmax2_all", "dmax2_topk", "dmax2_random_same_budget"])
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out_dir", default="results/extension_b_dmax2/features")
    parser.add_argument("--max_bridge_rows_per_path", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    db, task = load_relbench(args.dataset, args.task, download=args.download)
    table_dict = get_table_dict(db)

    paths = discover_dmax2_paths(
        db=db,
        target_key=args.target_key,
        target_time_col=args.target_time_col,
        verbose=True,
    )

    stem = f"{args.dataset}_{args.task}_{args.variant}_seed{args.seed}_topk{args.top_k}"
    stem = safe_name(stem)
    out_base = Path(args.out_dir) / stem
    out_base.mkdir(parents=True, exist_ok=True)

    with open(out_base / "paths.json", "w") as f:
        json.dump(paths, f, indent=2, default=str)

    if args.variant == "discover":
        print(f"\nSaved discovered paths to: {out_base / 'paths.json'}")
        return

    split_features = {}
    split_task_dfs = {}

    for split in ["train", "val", "test"]:
        print(f"\n=== Building {split} features ===")
        split_df = get_task_split_df(task, split)
        split_task_dfs[split] = split_df.copy()

        feats = build_features_for_split(
            task_df=split_df,
            table_dict=table_dict,
            paths=paths,
            target_key=args.target_key,
            target_time_col=args.target_time_col,
            max_bridge_rows_per_path=args.max_bridge_rows_per_path,
        )
        split_features[split] = feats
        print(f"{split} dmax2 raw feature shape: {feats.shape}")

    selected_cols = select_features(
        train_feats=split_features["train"],
        variant=args.variant,
        top_k=args.top_k,
        seed=args.seed,
    )

    print(f"\nSelected {len(selected_cols)} features for variant={args.variant}")
    print("First selected columns:")
    for c in selected_cols[:20]:
        print("  ", c)

    # Save feature-only files and target-attached files.
    manifest = {
        "dataset": args.dataset,
        "task": args.task,
        "target_key": args.target_key,
        "target_time_col": args.target_time_col,
        "variant": args.variant,
        "top_k": args.top_k,
        "seed": args.seed,
        "n_paths": len(paths),
        "n_raw_features": int(split_features["train"].shape[1]),
        "n_selected_features": len(selected_cols),
        "selected_features": selected_cols,
    }

    with open(out_base / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    for split in ["train", "val", "test"]:
        feats = split_features[split][selected_cols].copy()

        feature_path = out_base / f"{split}_features.parquet"
        feats.to_parquet(feature_path, index=False)

        # Also save a combined file with original task columns + dmax2 features.
        combined = pd.concat(
            [
                split_task_dfs[split].reset_index(drop=True),
                feats.reset_index(drop=True),
            ],
            axis=1,
        )
        combined_path = out_base / f"{split}_combined.parquet"
        combined.to_parquet(combined_path, index=False)

        print(f"Saved {split}:")
        print(f"  feature-only: {feature_path}")
        print(f"  combined:     {combined_path}")

    print(f"\nDone. Output directory: {out_base}")


if __name__ == "__main__":
    main()
