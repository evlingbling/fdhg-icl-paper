import os
import time
import tracemalloc
from pathlib import Path

import pandas as pd


VARIANTS = [
    "dfs",
    "fdhg_dmax1_full",
    "fdhg_dmax1_plus_dmax2_supervised_ap_topk16",
]

SEEDS = [41, 42, 43, 44]

BASE = Path("results/gbdt_filtered_features")
OUT = Path("results/final_tables/efficiency_audit_relstack_user_badge.csv")


def file_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def audit_one(seed, variant):
    d = BASE / f"rel_stack_user_badge_{variant}_seed{seed}"
    train_path = d / "train_combined.parquet"
    val_path = d / "val_combined.parquet"

    if not train_path.exists() or not val_path.exists():
        print(f"[SKIP] missing {d}")
        return None

    tracemalloc.start()
    t0 = time.perf_counter()

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    target_col = "WillGetBadge"
    n_features = len([c for c in train_df.columns if c != target_col])

    return {
        "dataset": "rel-stack",
        "task": "user-badge",
        "seed": seed,
        "variant": variant,
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "n_columns_total": train_df.shape[1],
        "n_features_in_file": n_features,
        "train_file_mb": file_mb(train_path),
        "val_file_mb": file_mb(val_path),
        "read_materialized_features_sec": elapsed,
        "peak_memory_read_mb": peak / (1024 ** 2),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }


def main():
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            row = audit_one(seed, variant)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    summary = (
        df.groupby(["dataset", "task", "variant"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            n_train_rows_mean=("n_train_rows", "mean"),
            n_val_rows_mean=("n_val_rows", "mean"),
            n_features_mean=("n_features_in_file", "mean"),
            train_file_mb_mean=("train_file_mb", "mean"),
            val_file_mb_mean=("val_file_mb", "mean"),
            read_materialized_features_sec_mean=("read_materialized_features_sec", "mean"),
            peak_memory_read_mb_mean=("peak_memory_read_mb", "mean"),
        )
    )

    summary_path = OUT.with_name("efficiency_audit_relstack_user_badge_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== EFFICIENCY BY SEED ===")
    print(df.to_string(index=False))

    print("\n=== EFFICIENCY SUMMARY ===")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(OUT)
    print(summary_path)


if __name__ == "__main__":
    main()
