from pathlib import Path
import argparse
import json
import re
import pandas as pd


DROP_PATTERNS = [
    r"^f_amb__",
    r"^f_amb__.*__is_missing$",
    r"^amb_",
    r"^amb_.*__is_missing$",
    r"amb_",
    r"ambiguity",
    r"__majconf$",
    r"__entropy$",
    r"__conflict_count$",
    r"__support_count$",
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


def norm_name(s):
    return str(s).replace("-", "_").replace("/", "_")


def is_key_or_target_col(col):
    c = str(col)
    return any(re.search(p, c, flags=re.IGNORECASE) for p in KEY_OR_TARGET_PATTERNS)


def is_dependency_col(col):
    c = str(col)
    return any(re.search(p, c, flags=re.IGNORECASE) for p in DROP_PATTERNS)


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


def compare(a, b):
    cols_equal = list(a.columns) == list(b.columns)
    a_only = sorted(set(a.columns) - set(b.columns))
    b_only = sorted(set(b.columns) - set(a.columns))

    common = [c for c in a.columns if c in b.columns]
    values_equal = len(a) == len(b)
    diff_cols = []

    if values_equal:
        for c in common:
            if not a[c].equals(b[c]):
                values_equal = False
                diff_cols.append(c)
    else:
        diff_cols.append("__ROW_COUNT_DIFF__")

    return {
        "columns_equal": bool(cols_equal),
        "values_equal": bool(values_equal),
        "a_only_cols": a_only,
        "b_only_cols": b_only,
        "different_common_cols": diff_cols,
    }


def load_fdhg_matrix(row):
    """
    Option A: fdhg_path already contains full FDHG matrix.
    Option B: fdhg_base_path + ambiguity_path are concatenated column-wise.
    """
    if pd.notna(row.get("fdhg_path", None)) and str(row.get("fdhg_path")).strip():
        return pd.read_parquet(row["fdhg_path"]), [row["fdhg_path"]]

    parts = []
    used = []

    for key in ["fdhg_base_path", "ambiguity_path", "dmax2_path"]:
        val = row.get(key, "")
        if pd.notna(val) and str(val).strip():
            p = str(val)
            parts.append(pd.read_parquet(p))
            used.append(p)

    if not parts:
        raise ValueError("No FDHG input path found. Need fdhg_path or fdhg_base_path/ambiguity_path.")

    out = parts[0].copy()
    for part in parts[1:]:
        # Avoid duplicate columns when target/key columns are repeated.
        add_cols = [c for c in part.columns if c not in out.columns]
        out = pd.concat([out, part[add_cols]], axis=1)

    return out, used


def process_row(row, out_root):
    dataset = row["dataset"]
    task = row["task"]
    seed = int(row["seed"])
    split = row["split"]

    dfs_path = Path(row["dfs_path"])
    dfs = pd.read_parquet(dfs_path)

    fdhg, fdhg_used_paths = load_fdhg_matrix(row)

    keep_cols, drop_cols = infer_fkagg_columns(fdhg.columns)
    fkagg = fdhg[keep_cols].copy()

    task_key = f"{norm_name(dataset)}_{norm_name(task)}"
    out_dir = Path(out_root) / task_key / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{split}_combined.parquet"
    manifest_path = out_dir / f"{split}_manifest.json"

    fkagg.to_parquet(out_path, index=False)

    cmp = compare(fkagg, dfs)

    manifest = {
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "split": split,
        "dfs_path": str(dfs_path),
        "fdhg_used_paths": fdhg_used_paths,
        "fkagg_output_path": str(out_path),
        "mode": "path_config_fdhg_fkagg_filter",
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

    return {
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "split": split,
        "status": "ok",
        "dfs_path": str(dfs_path),
        "fdhg_used_paths": "|".join(map(str, fdhg_used_paths)),
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
        "needs_separate_fkagg_evaluation": not (cmp["columns_equal"] and cmp["values_equal"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_root", default="results/fkagg_only_path_config")
    ap.add_argument("--out_csv", default="results/final_tables/fkagg_path_config_audit_rows.csv")
    args = ap.parse_args()

    cfg = pd.read_csv(args.config)
    rows = []

    for _, row in cfg.iterrows():
        try:
            res = process_row(row, args.out_root)
            print(json.dumps(res, indent=2))
            rows.append(res)
        except Exception as e:
            res = {
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "split": row.get("split"),
                "status": "failed",
                "error": repr(e),
                "needs_separate_fkagg_evaluation": True,
            }
            print(json.dumps(res, indent=2))
            rows.append(res)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    ok = df[df["status"].eq("ok")].copy()
    if len(ok):
        summary = ok.groupby(["dataset", "task"], dropna=False).agg(
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
    else:
        summary = pd.DataFrame()

    summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== summary ===")
    print(summary.to_string(index=False) if len(summary) else "no successful rows")
    print("\nWrote:")
    print(out_csv)
    print(summary_path)


if __name__ == "__main__":
    main()
