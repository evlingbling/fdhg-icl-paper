import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^0-9a-zA-Z_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def get_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "df"):
        return obj.df
    raise TypeError(f"Cannot extract df from {type(obj)}")


def get_table_dict(db):
    if hasattr(db, "table_dict"):
        return db.table_dict
    if hasattr(db, "tables"):
        return db.tables
    raise AttributeError("Cannot find db.table_dict or db.tables")


def is_time_like(col):
    c = str(col).lower()
    return any(k in c for k in ["time", "date", "timestamp", "created", "creation"])


def is_id_like(col):
    c = str(col).lower()
    return (
        c == "id"
        or c.endswith("id")
        or c.endswith("_id")
        or "userid" in c
        or "postid" in c
        or "badgeid" in c
        or "parentid" in c
        or "acceptedanswerid" in c
        or "relatedpostid" in c
        or c in ["index", "unnamed: 0"]
    )


def infer_pkey(table_name, df):
    for c in ["Id", "id", table_name, table_name.rstrip("s")]:
        if c in df.columns:
            return c

    n = len(df)
    for c in df.columns:
        if is_id_like(c) and n > 0:
            nunq = df[c].nunique(dropna=True)
            if nunq >= 0.7 * n:
                return c
    return None


def infer_time_col(df):
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
    return get_df(task.get_table(split)).copy()


def parse_csv(x):
    if x is None or str(x).strip() == "":
        return []
    return [v.strip() for v in str(x).split(",") if v.strip()]


def build_infos(db):
    infos = {}
    for name, table in get_table_dict(db).items():
        df = get_df(table)
        infos[name] = {
            "df": df,
            "pkey": infer_pkey(name, df),
            "time_col": infer_time_col(df),
            "columns": list(df.columns),
        }
    return infos


def match_second_table(infos, bridge_name, second_key):
    if second_key in infos:
        return second_key

    key = str(second_key)

    # Stack schema specific aliases
    alias = {
        "PostId": "posts",
        "ParentId": "posts",
        "AcceptedAnswerId": "posts",
        "RelatedPostId": "posts",
        "UserId": "users",
        "OwnerUserId": "users",
        "BadgeId": "badges",
    }
    if key in alias and alias[key] in infos:
        return alias[key]

    base = re.sub(r"(_id|Id|ID)$", "", key)
    candidates = [
        base,
        base.lower(),
        base.capitalize(),
        base + "s",
        base.lower() + "s",
        base.capitalize() + "s",
    ]
    for c in candidates:
        if c in infos:
            return c

    for tname, info in infos.items():
        if tname == bridge_name:
            continue
        if info["pkey"] == second_key:
            return tname

    return None


def discover_paths(db, task_target_key, bridge_target_keys):
    infos = build_infos(db)
    paths = []

    for bridge_name, binfo in infos.items():
        bdf = binfo["df"]
        bcols = set(bdf.columns)

        valid_target_keys = [c for c in bridge_target_keys if c in bcols]
        if not valid_target_keys:
            continue

        for bridge_target_key in valid_target_keys:
            for second_key in bdf.columns:
                if second_key == bridge_target_key:
                    continue
                if not is_id_like(second_key):
                    continue

                # Reject generic row identifiers, but allow meaningful FK-like
                # columns such as PostId, facility_id, sponsor_id, condition_id,
                # and intervention_id when they match another table's primary key.
                if str(second_key).lower() == "id":
                    continue

                second_table = match_second_table(infos, bridge_name, second_key)
                if second_table is None:
                    continue

                sinfo = infos[second_table]
                sdf = sinfo["df"]
                second_pkey = sinfo["pkey"]

                if second_pkey is None or second_pkey not in sdf.columns:
                    continue

                # Critical anti-spurious-join condition:
                # the bridge FK must match the inferred PK of the second table.
                if str(second_pkey).lower() != str(second_key).lower():
                    continue

                usable_cols = []
                for col in sdf.columns:
                    if col == second_pkey:
                        continue
                    if col in [task_target_key, bridge_target_key, second_key]:
                        continue
                    if str(col).lower() in ["target", "willgetbadge"]:
                        continue
                    if is_time_like(col):
                        continue
                    if is_id_like(col):
                        continue
                    usable_cols.append(col)

                if not usable_cols:
                    continue

                paths.append({
                    "bridge_table": bridge_name,
                    "task_target_key": task_target_key,
                    "bridge_target_key": bridge_target_key,
                    "bridge_second_key": second_key,
                    "bridge_time_col": binfo["time_col"],
                    "second_table": second_table,
                    "second_pkey": second_pkey,
                    "second_usable_cols": usable_cols,
                })

    print("\n=== DISCOVERED DMAX=2 PATHS ===")
    if not paths:
        print("No dmax=2 paths discovered.")
    for i, p in enumerate(paths):
        print(f"\n[{i}] task.{p['task_target_key']} <- {p['bridge_table']}.{p['bridge_target_key']} -> {p['second_table']}")
        print(f"    second key: {p['bridge_table']}.{p['bridge_second_key']} -> {p['second_table']}.{p['second_pkey']}")
        print(f"    bridge time col: {p['bridge_time_col']}")
        print(f"    usable cols: {p['second_usable_cols'][:20]}")
    return paths


def build_features_for_split(task_df, table_dict, paths, task_target_key, target_time_col):
    task_df = task_df.copy()
    task_df["__row_id__"] = np.arange(len(task_df))

    has_target_time = target_time_col in task_df.columns
    if has_target_time:
        task_df[target_time_col] = pd.to_datetime(task_df[target_time_col], errors="coerce")

    out = pd.DataFrame({"__row_id__": task_df["__row_id__"].values})

    for p in paths:
        bridge_df = get_df(table_dict[p["bridge_table"]]).copy()
        second_df = get_df(table_dict[p["second_table"]]).copy()

        bridge_cols = [p["bridge_target_key"], p["bridge_second_key"]]
        if p["bridge_time_col"] in bridge_df.columns:
            bridge_cols.append(p["bridge_time_col"])
        bridge_cols = list(dict.fromkeys(bridge_cols))
        bridge_df = bridge_df[bridge_cols].copy()

        left_cols = ["__row_id__", task_target_key]
        if has_target_time:
            left_cols.append(target_time_col)

        merged = task_df[left_cols].merge(
            bridge_df,
            left_on=task_target_key,
            right_on=p["bridge_target_key"],
            how="left",
        )

        if has_target_time and p["bridge_time_col"] in merged.columns:
            merged[p["bridge_time_col"]] = pd.to_datetime(merged[p["bridge_time_col"]], errors="coerce")
            merged = merged[
                merged[p["bridge_time_col"]].isna()
                | merged[target_time_col].isna()
                | (merged[p["bridge_time_col"]] <= merged[target_time_col])
            ].copy()

        second_cols = [p["second_pkey"]] + p["second_usable_cols"]
        second_cols = [c for c in second_cols if c in second_df.columns]
        second_df = second_df[second_cols].copy()

        merged = merged.merge(
            second_df,
            left_on=p["bridge_second_key"],
            right_on=p["second_pkey"],
            how="left",
        )

        prefix = safe_name(
            f"dmax2__{p['bridge_table']}__{p['bridge_target_key']}__{p['bridge_second_key']}__{p['second_table']}"
        )

        grouped = merged.groupby("__row_id__", sort=False)
        feat = pd.DataFrame({"__row_id__": task_df["__row_id__"].values}).set_index("__row_id__")

        feat[f"{prefix}__count_distinct_{safe_name(p['bridge_second_key'])}"] = grouped[p["bridge_second_key"]].nunique(dropna=True)

        for col in p["second_usable_cols"]:
            if col not in merged.columns:
                continue

            if pd.api.types.is_numeric_dtype(merged[col]):
                agg = grouped[col].agg(["mean", "std", "min", "max"])
                for op in ["mean", "std", "min", "max"]:
                    feat[f"{prefix}__{op}_{safe_name(col)}"] = agg[op]
            else:
                feat[f"{prefix}__nunique_{safe_name(col)}"] = grouped[col].nunique(dropna=True)

        feat = feat.reset_index()
        out = out.merge(feat, on="__row_id__", how="left")

    out = out.drop(columns=["__row_id__"])
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)

    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    return out


def score_feature(s, name):
    nonnull = s.notna().mean()
    nunq = s.nunique(dropna=True)
    const_penalty = 0.5 if nunq <= 1 else 0.0
    return float(nonnull + 0.1 * np.log1p(nunq) - const_penalty)


def select_cols(train_feats, variant, top_k, seed):
    cols = list(train_feats.columns)

    if variant == "dmax2_all":
        return cols

    if variant == "dmax2_topk":
        scored = sorted(
            [(c, score_feature(train_feats[c], c)) for c in cols],
            key=lambda x: x[1],
            reverse=True,
        )
        return [c for c, _ in scored[:top_k]]

    if variant == "dmax2_random_same_budget":
        rng = random.Random(seed)
        cols = cols[:]
        rng.shuffle(cols)
        return cols[:top_k]

    raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task_target_key", required=True)
    parser.add_argument("--bridge_target_key_candidates", required=True)
    parser.add_argument("--target_time_col", default="timestamp")
    parser.add_argument("--variant", default="discover",
                        choices=["discover", "dmax2_all", "dmax2_topk", "dmax2_random_same_budget"])
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out_dir", default="results/extension_b_dmax2/features")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    db, task = load_relbench(args.dataset, args.task, args.download)
    table_dict = get_table_dict(db)

    bridge_target_keys = parse_csv(args.bridge_target_key_candidates)

    paths = discover_paths(db, args.task_target_key, bridge_target_keys)

    stem = safe_name(f"{args.dataset}_{args.task}_{args.variant}_seed{args.seed}_topk{args.top_k}")
    out_base = Path(args.out_dir) / stem
    out_base.mkdir(parents=True, exist_ok=True)

    with open(out_base / "paths.json", "w") as f:
        json.dump(paths, f, indent=2, default=str)

    if args.variant == "discover":
        print(f"\nSaved discovered paths to: {out_base / 'paths.json'}")
        return

    split_feats = {}
    split_task = {}

    for split in ["train", "val", "test"]:
        print(f"\n=== Building {split} ===")
        task_df = get_task_split_df(task, split)
        split_task[split] = task_df.copy()

        if args.task_target_key not in task_df.columns:
            raise ValueError(f"{args.task_target_key} not found in {split}: {list(task_df.columns)}")

        feats = build_features_for_split(
            task_df=task_df,
            table_dict=table_dict,
            paths=paths,
            task_target_key=args.task_target_key,
            target_time_col=args.target_time_col,
        )
        split_feats[split] = feats
        print(split, feats.shape)

    selected = select_cols(split_feats["train"], args.variant, args.top_k, args.seed)
    print(f"\nSelected {len(selected)} features")
    for c in selected[:20]:
        print(" ", c)

    manifest = {
        "dataset": args.dataset,
        "task": args.task,
        "variant": args.variant,
        "seed": args.seed,
        "top_k": args.top_k,
        "n_paths": len(paths),
        "n_raw_features": split_feats["train"].shape[1],
        "n_selected_features": len(selected),
        "selected_features": selected,
    }
    with open(out_base / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    for split in ["train", "val", "test"]:
        feats = split_feats[split][selected].copy()
        feats.to_parquet(out_base / f"{split}_features.parquet", index=False)

        combined = pd.concat(
            [split_task[split].reset_index(drop=True), feats.reset_index(drop=True)],
            axis=1,
        )
        combined.to_parquet(out_base / f"{split}_combined.parquet", index=False)

        print(f"Saved {split}: {out_base / f'{split}_combined.parquet'}")

    print(f"\nDone: {out_base}")


if __name__ == "__main__":
    main()
