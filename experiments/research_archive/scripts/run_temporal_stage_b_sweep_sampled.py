from pathlib import Path
import argparse
import subprocess
import pandas as pd


def run(cmd, dry_run=False):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, cmd)), check=True)


def ensure_binary_labels(task_slug):
    label_specs = [
        ("rel-stack_post-votes", "popularity", "popularity_binary", lambda y: (y > 0).astype(int)),
        ("rel-stack_post-votes", "popularity", "popularity_ge2", lambda y: (y >= 2).astype(int)),
        ("rel-trial_study-adverse", "num_of_adverse_events", "adverse_binary", lambda y: (y > 0).astype(int)),
        ("rel-trial_study-adverse", "num_of_adverse_events", "adverse_ge2", lambda y: (y >= 2).astype(int)),
    ]

    roots = [
        Path("outputs/naive_flatten") / f"{task_slug}_sample",
        Path("outputs/dfs_agg") / f"{task_slug}_sample",
        Path("outputs/fdhg_heuristic") / f"{task_slug}_sample",
    ]

    for slug, src_col, new_col, fn in label_specs:
        if slug != task_slug:
            continue
        for root in roots:
            for p in root.glob("target_with_dfs_agg_*.parquet"):
                df = pd.read_parquet(p)
                if src_col in df.columns and new_col not in df.columns:
                    y = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
                    df[new_col] = fn(y)
                    df.to_parquet(p, index=False)
                    print(f"[OK label] {p} added {new_col}: {df[new_col].value_counts().to_dict()}")


def merge_on_position(base_path, temporal_path, out_path):
    base = pd.read_parquet(base_path)
    temp = pd.read_parquet(temporal_path)

    drop_cols = [c for c in temp.columns if c in base.columns]
    temp2 = temp.drop(columns=drop_cols, errors="ignore")

    if len(base) != len(temp2):
        raise ValueError(
            f"row mismatch after sampled build: base={base_path} {base.shape}, "
            f"temp={temporal_path} {temp.shape}"
        )

    out = pd.concat([base.reset_index(drop=True), temp2.reset_index(drop=True)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[OK merge] {out_path} shape={out.shape} added={len(temp2.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default="outputs/temporal_stage_b_sampled")
    ap.add_argument("--windows-days", default="30,90")
    ap.add_argument("--strict-before", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = pd.read_csv(args.config)
    out_root = Path(args.out_root)
    failures = []

    for _, row in cfg.iterrows():
        dataset = str(row["dataset"])
        task = str(row["task"])
        task_slug = f"{dataset}_{task}"

        print("\n" + "=" * 100)
        print(f"STAGE B TEMPORAL SAMPLED: {task_slug}")
        print("=" * 100)

        try:
            ensure_binary_labels(task_slug)

            inspect_dir = Path("outputs/relbench_inspect") / task_slug
            child_path = inspect_dir / f"table_{row['child_table']}.parquet"

            if not child_path.exists():
                raise FileNotFoundError(f"missing child table: {child_path}")

            dfs_base_dir = Path("outputs/dfs_agg") / f"{task_slug}_sample"
            fdhg_base_dir = Path("outputs/fdhg_heuristic") / f"{task_slug}_sample"

            temporal_dir = out_root / task_slug / "temporal_only"
            dfs_plus_dir = out_root / task_slug / "dfs_plus_temporal"
            fdhg_plus_dir = out_root / task_slug / "fdhg_plus_temporal"

            for split in ["train", "val", "test"]:
                dfs_base_path = dfs_base_dir / f"target_with_dfs_agg_{split}.parquet"
                fdhg_base_path = fdhg_base_dir / f"target_with_dfs_agg_{split}.parquet"

                # Use sampled DFS base as temporal target so row order/count matches sampled matrices.
                if not dfs_base_path.exists():
                    print(f"[SKIP] missing sampled DFS base: {dfs_base_path}")
                    continue

                temp_out = temporal_dir / f"target_with_temporal_{split}.parquet"

                cmd = [
                    "python", "scripts/build_temporal_features.py",
                    "--target-path", dfs_base_path,
                    "--child-path", child_path,
                    "--out-path", temp_out,
                    "--target-key", row["target_key"],
                    "--target-time-col", row["target_time_col"],
                    "--child-key", row["child_key"],
                    "--child-time-col", row["child_time_col"],
                    "--numeric-col", row["numeric_col"],
                    "--source-name", row["child_table"],
                    "--windows-days", args.windows_days,
                    "--row-id-col", "__row_id",
                ]
                if args.strict_before:
                    cmd.append("--strict-before")

                run(cmd, dry_run=args.dry_run)

                if dfs_base_path.exists() and temp_out.exists():
                    merge_on_position(
                        dfs_base_path,
                        temp_out,
                        dfs_plus_dir / f"target_with_dfs_agg_{split}.parquet",
                    )

                if fdhg_base_path.exists() and temp_out.exists():
                    merge_on_position(
                        fdhg_base_path,
                        temp_out,
                        fdhg_plus_dir / f"target_with_dfs_agg_{split}.parquet",
                    )

        except Exception as e:
            print(f"[FAILED] {task_slug}: {type(e).__name__}: {e}")
            failures.append({
                "dataset": dataset,
                "task": task,
                "error_type": type(e).__name__,
                "error": str(e),
            })

    if failures:
        fail_path = out_root / "stage_b_temporal_failures.csv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print("\nSaved failures:", fail_path)


if __name__ == "__main__":
    main()
