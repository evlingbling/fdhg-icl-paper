from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")


def summarize_values(vals):
    vals = pd.to_numeric(vals, errors="coerce")
    if len(vals) == 0:
        return {
            "count": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "max": np.nan,
            "nunique": 0.0,
        }
    return {
        "count": float(vals.notna().sum()),
        "mean": float(vals.mean()) if vals.notna().any() else np.nan,
        "std": float(vals.std()) if vals.notna().sum() > 1 else 0.0,
        "max": float(vals.max()) if vals.notna().any() else np.nan,
        "nunique": float(vals.nunique(dropna=True)),
    }


def build_dmax2_for_split(
    target_path,
    base_child_path,
    second_path,
    out_path,
    target_key,
    target_time_col,
    base_child_key,
    base_child_time_col,
    pivot_key,
    second_key,
    second_time_col,
    numeric_col,
    source_name,
    max_pivots_per_row=500,
    strict_before=True,
):
    target = pd.read_parquet(target_path).copy()

    base_cols = list(dict.fromkeys([base_child_key, base_child_time_col, pivot_key]))
    second_cols = list(dict.fromkeys([second_key, second_time_col, numeric_col]))

    base = pd.read_parquet(base_child_path, columns=base_cols).copy()
    second = pd.read_parquet(second_path, columns=second_cols).copy()

    target[target_time_col] = to_datetime_safe(target[target_time_col])
    base[base_child_time_col] = to_datetime_safe(base[base_child_time_col])
    second[second_time_col] = to_datetime_safe(second[second_time_col])

    base = base.dropna(subset=[base_child_key, base_child_time_col, pivot_key])
    second = second.dropna(subset=[second_key, second_time_col])

    base_groups = {k: g for k, g in base.groupby(base_child_key, sort=False)}
    second_groups = {k: g for k, g in second.groupby(second_key, sort=False)}

    prefix = f"fdhg::dmax2::{target_key}<-{source_name}.{base_child_key}->{pivot_key}->{source_name}.{second_key}::{numeric_col}"

    rows = []
    for _, row in target.iterrows():
        key = row.get(target_key)
        cutoff = row.get(target_time_col)

        out = {
            f"{prefix}::pivot_nunique": 0.0,
            f"{prefix}::second_count": 0.0,
            f"{prefix}::second_mean": np.nan,
            f"{prefix}::second_std": np.nan,
            f"{prefix}::second_max": np.nan,
            f"{prefix}::second_nunique": 0.0,
            f"{prefix}::days_since_second_last": np.nan,
        }

        if pd.isna(key) or pd.isna(cutoff) or key not in base_groups:
            rows.append(out)
            continue

        bg = base_groups[key]
        if strict_before:
            bg = bg[bg[base_child_time_col] < cutoff]
        else:
            bg = bg[bg[base_child_time_col] <= cutoff]

        if bg.empty:
            rows.append(out)
            continue

        pivots = bg[pivot_key].dropna().unique()
        if len(pivots) > max_pivots_per_row:
            pivots = pivots[:max_pivots_per_row]

        out[f"{prefix}::pivot_nunique"] = float(len(pivots))

        second_parts = []
        for pv in pivots:
            sg = second_groups.get(pv)
            if sg is None or sg.empty:
                continue
            if strict_before:
                sg = sg[sg[second_time_col] < cutoff]
            else:
                sg = sg[sg[second_time_col] <= cutoff]
            if not sg.empty:
                second_parts.append(sg)

        if not second_parts:
            rows.append(out)
            continue

        ss = pd.concat(second_parts, axis=0, ignore_index=True)
        stats = summarize_values(ss[numeric_col])

        out[f"{prefix}::second_count"] = stats["count"]
        out[f"{prefix}::second_mean"] = stats["mean"]
        out[f"{prefix}::second_std"] = stats["std"]
        out[f"{prefix}::second_max"] = stats["max"]
        out[f"{prefix}::second_nunique"] = stats["nunique"]

        last_time = ss[second_time_col].max()
        if pd.notna(last_time):
            out[f"{prefix}::days_since_second_last"] = float((cutoff - last_time).total_seconds() / 86400.0)

        rows.append(out)

    feat = pd.DataFrame(rows)

    # Missing indicators.
    for c in list(feat.columns):
        feat[f"{c}__is_missing"] = feat[c].isna().astype(int)
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(-999.0)

    out_df = pd.concat([target.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print(f"[OK] wrote {out_path} shape={out_df.shape} dmax2_cols={feat.shape[1]}")
    return out_df


def merge_base_plus_dmax2(base_path, dmax2_path, out_path):
    base = pd.read_parquet(base_path)
    dmax2 = pd.read_parquet(dmax2_path)

    add_cols = [c for c in dmax2.columns if c not in base.columns]
    out = pd.concat(
        [base.reset_index(drop=True), dmax2[add_cols].reset_index(drop=True)],
        axis=1,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[OK merge] {out_path} shape={out.shape} added={len(add_cols)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default="outputs/stage_c_dmax2_sampled")
    ap.add_argument("--max-pivots-per-row", type=int, default=500)
    ap.add_argument("--strict-before", action="store_true")
    args = ap.parse_args()

    cfg = pd.read_csv(args.config)
    failures = []

    for _, r in cfg.iterrows():
        dataset = r["dataset"]
        task = r["task"]
        slug = f"{dataset}_{task}"

        print("\n" + "=" * 100)
        print(f"STAGE C DMAX2: {slug}")
        print("=" * 100)

        try:
            inspect_dir = Path("outputs/relbench_inspect") / slug
            base_child_path = inspect_dir / f"table_{r['base_child_table']}.parquet"
            second_path = inspect_dir / f"table_{r['second_table']}.parquet"

            dfs_base_dir = Path("outputs/dfs_agg") / f"{slug}_sample"
            fdhg_base_dir = Path("outputs/fdhg_heuristic") / f"{slug}_sample"

            for split in ["train", "val", "test"]:
                # Prefer sampled DFS matrix as target path so row count aligns with sampled eval.
                sampled_target = dfs_base_dir / f"target_with_dfs_agg_{split}.parquet"
                if not sampled_target.exists():
                    sampled_target = inspect_dir / f"target_{split}.parquet"
                    if not sampled_target.exists():
                        print(f"[SKIP] missing target for {slug} {split}")
                        continue

                dmax2_out = (
                    Path(args.out_root)
                    / slug
                    / "dmax2_only"
                    / f"target_with_dmax2_{split}.parquet"
                )

                build_dmax2_for_split(
                    target_path=sampled_target,
                    base_child_path=base_child_path,
                    second_path=second_path,
                    out_path=dmax2_out,
                    target_key=r["target_key"],
                    target_time_col=r["target_time_col"],
                    base_child_key=r["base_child_key"],
                    base_child_time_col=r["base_child_time_col"],
                    pivot_key=r["pivot_key"],
                    second_key=r["second_key"],
                    second_time_col=r["second_time_col"],
                    numeric_col=r["numeric_col"],
                    source_name=r["second_table"],
                    max_pivots_per_row=args.max_pivots_per_row,
                    strict_before=args.strict_before,
                )

                for base_dir, variant in [
                    (dfs_base_dir, "dfs_plus_dmax2"),
                    (fdhg_base_dir, "fdhg_plus_dmax2"),
                ]:
                    base_path = base_dir / f"target_with_dfs_agg_{split}.parquet"
                    if base_path.exists():
                        out_path = (
                            Path(args.out_root)
                            / slug
                            / variant
                            / f"target_with_dfs_agg_{split}.parquet"
                        )
                        merge_base_plus_dmax2(base_path, dmax2_out, out_path)

        except Exception as e:
            print(f"[FAILED] {slug}: {type(e).__name__}: {e}")
            failures.append({
                "dataset": dataset,
                "task": task,
                "error_type": type(e).__name__,
                "error": str(e),
            })

    if failures:
        fail_path = Path(args.out_root) / "stage_c_dmax2_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print("\nSaved failures:", fail_path)


if __name__ == "__main__":
    main()
