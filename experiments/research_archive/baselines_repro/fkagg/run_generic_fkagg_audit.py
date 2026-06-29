from pathlib import Path
import argparse
import json
import re
import pandas as pd


DROP_PATTERNS = [
    r"^f_amb__",
    r"__majconf$",
    r"__entropy$",
    r"__conflict_count$",
    r"__support_count$",
    r"^f_amb__.*__is_missing$",
    r"dmax2",
    r"twohop",
    r"2hop",
    r"afd",
    r"fd_score",
    r"fdhg_rank",
    r"ranker",
    r"uniqueness",
    r"surrogate",
    r"residual",
    r"ambiguity",
]

KEY_OR_TARGET_PATTERNS = [
    r"^target$",
    r"^label$",
    r"^y$",
    r"^timestamp$",
    r"^time$",
    r"^date$",
    r"^index$",
    r"(^|_)(id|key)$",
    r"(Id|ID)$",
]


def norm_name(s: str) -> str:
    return str(s).replace("-", "_").replace("/", "_")


def is_key_or_target_col(col: str) -> bool:
    c = str(col)
    for pat in KEY_OR_TARGET_PATTERNS:
        if re.search(pat, c, flags=re.IGNORECASE):
            return True
    return False


def is_dependency_col(col: str) -> bool:
    c = str(col)
    for pat in DROP_PATTERNS:
        if re.search(pat, c, flags=re.IGNORECASE):
            return True
    return False


def infer_fkagg_columns(columns):
    keep, drop = [], []
    for col in columns:
        c = str(col)

        if is_key_or_target_col(c):
            keep.append(c)
        elif is_dependency_col(c):
            drop.append(c)
        else:
            keep.append(c)

    return keep, drop


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def compare_dataframes(a: pd.DataFrame, b: pd.DataFrame):
    cols_equal = list(a.columns) == list(b.columns)
    a_only = sorted(set(a.columns) - set(b.columns))
    b_only = sorted(set(b.columns) - set(a.columns))

    common = [c for c in a.columns if c in b.columns]
    values_equal = True
    diff_cols = []

    if len(a) != len(b):
        values_equal = False
        diff_cols.append("__ROW_COUNT_DIFF__")
    else:
        for c in common:
            if not a[c].equals(b[c]):
                values_equal = False
                diff_cols.append(c)

    return {
        "columns_equal": bool(cols_equal),
        "values_equal": bool(values_equal),
        "a_only_cols": a_only,
        "b_only_cols": b_only,
        "different_common_cols": diff_cols,
    }


def find_feature_dirs(root: Path, dataset: str, task: str, seed: int):
    """
    Searches for existing DFS and FDHG dmax1 feature directories.
    Expected examples:
      results/gbdt_filtered_features/rel_stack_user_badge_dfs_seed41
      results/gbdt_filtered_features/rel_stack_user_badge_fdhg_dmax1_full_seed41
    """
    prefix = f"{norm_name(dataset)}_{norm_name(task)}"
    candidates = list(root.glob(f"{prefix}*seed{seed}"))

    dfs_dirs = [
        p for p in candidates
        if p.is_dir()
        and "dfs" in p.name.lower()
        and "fdhg" not in p.name.lower()
    ]

    fdhg_dirs = [
        p for p in candidates
        if p.is_dir()
        and "fdhg" in p.name.lower()
        and "dmax1" in p.name.lower()
        and "dmax2" not in p.name.lower()
    ]

    if not dfs_dirs:
        raise FileNotFoundError(f"No DFS feature dir found for {dataset}/{task}/seed{seed} under {root}")
    if not fdhg_dirs:
        raise FileNotFoundError(f"No FDHG dmax1 feature dir found for {dataset}/{task}/seed{seed} under {root}")

    # Prefer explicit names.
    dfs_dirs = sorted(dfs_dirs, key=lambda p: len(p.name))
    fdhg_dirs = sorted(fdhg_dirs, key=lambda p: len(p.name))

    return dfs_dirs[0], fdhg_dirs[0]


def process_one(root, out_root, dataset, task, seed, split):
    root = Path(root)
    out_root = Path(out_root)

    dfs_dir, fdhg_dir = find_feature_dirs(root, dataset, task, seed)

    dfs_path = dfs_dir / f"{split}_combined.parquet"
    fdhg_path = fdhg_dir / f"{split}_combined.parquet"

    dfs = read_parquet(dfs_path)
    fdhg = read_parquet(fdhg_path)

    keep_cols, drop_cols = infer_fkagg_columns(fdhg.columns)

    fkagg = fdhg[keep_cols].copy()

    task_key = f"{norm_name(dataset)}_{norm_name(task)}"
    out_dir = out_root / task_key / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{split}_combined.parquet"
    manifest_path = out_dir / f"{split}_manifest.json"

    fkagg.to_parquet(out_path, index=False)

    cmp = compare_dataframes(fkagg, dfs)

    manifest = {
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "split": split,
        "dfs_path": str(dfs_path),
        "fdhg_input_path": str(fdhg_path),
        "fkagg_output_path": str(out_path),
        "mode": "generic_fdhg_fkagg_filter",
        "definition": (
            "FDHG-FKAgg keeps base/FK/inverse-FK aggregation columns and removes "
            "FD/AFD ambiguity, dependency residual, dmax2, uniqueness, and ranker-derived columns."
        ),
        "n_dfs_columns": int(dfs.shape[1]),
        "n_fdhg_input_columns": int(fdhg.shape[1]),
        "n_fkagg_output_columns": int(fkagg.shape[1]),
        "n_dropped_columns": int(len(drop_cols)),
        "dropped_columns": drop_cols,
        "kept_columns": keep_cols,
        "comparison_to_dfs": cmp,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))

    row = {
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "split": split,
        "dfs_dir": str(dfs_dir),
        "fdhg_dir": str(fdhg_dir),
        "dfs_path": str(dfs_path),
        "fdhg_input_path": str(fdhg_path),
        "fkagg_output_path": str(out_path),
        "manifest_path": str(manifest_path),
        "n_dfs_columns": int(dfs.shape[1]),
        "n_fdhg_input_columns": int(fdhg.shape[1]),
        "n_fkagg_output_columns": int(fkagg.shape[1]),
        "n_dropped_columns": int(len(drop_cols)),
        "dropped_columns": "|".join(drop_cols),
        "fkagg_columns_equal_dfs": cmp["columns_equal"],
        "fkagg_values_equal_dfs": cmp["values_equal"],
        "fkagg_only_cols": "|".join(cmp["a_only_cols"]),
        "dfs_only_cols": "|".join(cmp["b_only_cols"]),
        "different_common_cols": "|".join(cmp["different_common_cols"][:50]),
        "status": "ok",
        "needs_separate_fkagg_evaluation": not (cmp["columns_equal"] and cmp["values_equal"]),
    }

    print(json.dumps(row, indent=2))
    return row


def summarize(rows, out_csv):
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = df.groupby(["dataset", "task"], dropna=False).agg(
        n_rows=("seed", "count"),
        seeds=("seed", lambda x: ",".join(map(str, sorted(pd.Series(x).astype(int).unique())))),
        n_splits=("split", "nunique"),
        all_exact_match=("needs_separate_fkagg_evaluation", lambda x: not bool(pd.Series(x).any())),
        any_needs_eval=("needs_separate_fkagg_evaluation", "any"),
        n_dfs_columns_mean=("n_dfs_columns", "mean"),
        n_fdhg_input_columns_mean=("n_fdhg_input_columns", "mean"),
        n_fkagg_output_columns_mean=("n_fkagg_output_columns", "mean"),
        n_dropped_columns_mean=("n_dropped_columns", "mean"),
    ).reset_index()

    summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== per-task summary ===")
    print(summary.to_string(index=False))
    print("\nWrote:")
    print(out_csv)
    print(summary_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="CSV with dataset,task,seeds")
    ap.add_argument("--feature_root", default="results/gbdt_filtered_features")
    ap.add_argument("--out_root", default="results/fkagg_only_generic")
    ap.add_argument("--out_csv", default="results/final_tables/generic_fkagg_audit_rows.csv")
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    cfg = pd.read_csv(args.config)
    rows = []

    for _, r in cfg.iterrows():
        dataset = str(r["dataset"])
        task = str(r["task"])
        seeds = [int(x.strip()) for x in str(r["seeds"]).split(",") if x.strip()]
        splits = [x.strip() for x in args.splits.split(",") if x.strip()]

        for seed in seeds:
            for split in splits:
                try:
                    rows.append(process_one(
                        root=args.feature_root,
                        out_root=args.out_root,
                        dataset=dataset,
                        task=task,
                        seed=seed,
                        split=split,
                    ))
                except Exception as e:
                    row = {
                        "dataset": dataset,
                        "task": task,
                        "seed": seed,
                        "split": split,
                        "status": "failed",
                        "error": repr(e),
                        "needs_separate_fkagg_evaluation": True,
                    }
                    print(json.dumps(row, indent=2))
                    rows.append(row)

    summarize(rows, args.out_csv)


if __name__ == "__main__":
    main()
