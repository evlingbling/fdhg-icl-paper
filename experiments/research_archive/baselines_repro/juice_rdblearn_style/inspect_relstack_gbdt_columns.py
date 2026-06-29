from pathlib import Path
import pandas as pd

OUT_DIR = Path("baselines_repro/juice_rdblearn_style/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATHS = {
    "dfs_seed41_train": Path("results/gbdt_filtered_features/rel_stack_user_badge_dfs_seed41/train_combined.parquet"),
    "dfs_seed41_val": Path("results/gbdt_filtered_features/rel_stack_user_badge_dfs_seed41/val_combined.parquet"),
    "fdhg_dmax1_seed41_train": Path("results/gbdt_filtered_features/rel_stack_user_badge_fdhg_dmax1_full_seed41/train_combined.parquet"),
    "fdhg_dmax1_seed41_val": Path("results/gbdt_filtered_features/rel_stack_user_badge_fdhg_dmax1_full_seed41/val_combined.parquet"),
    "fdhg_dmax1_dmax2_ap_seed41_train": Path("results/gbdt_filtered_features/rel_stack_user_badge_fdhg_dmax1_plus_dmax2_supervised_ap_topk16_seed41/train_combined.parquet"),
    "fdhg_dmax1_dmax2_ap_seed41_val": Path("results/gbdt_filtered_features/rel_stack_user_badge_fdhg_dmax1_plus_dmax2_supervised_ap_topk16_seed41/val_combined.parquet"),
}

rows = []
cols_by_name = {}

for name, path in PATHS.items():
    if not path.exists():
        print(f"MISSING: {name}: {path}")
        continue

    df = pd.read_parquet(path).head(20)
    cols = list(df.columns)
    cols_by_name[name] = cols

    for i, c in enumerate(cols):
        rows.append({
            "matrix": name,
            "path": str(path),
            "col_idx": i,
            "column": c,
        })

    print("\n===", name, "===")
    print("path:", path)
    print("shape_head:", df.shape)
    print("columns:")
    for i, c in enumerate(cols):
        print(f"{i:03d} {c}")

pd.DataFrame(rows).to_csv(OUT_DIR / "relstack_gbdt_column_inventory_seed41.csv", index=False)

# Compare DFS and FDHG dmax1 train columns
dfs = set(cols_by_name.get("dfs_seed41_train", []))
fdhg = set(cols_by_name.get("fdhg_dmax1_seed41_train", []))
dmax2 = set(cols_by_name.get("fdhg_dmax1_dmax2_ap_seed41_train", []))

diff_rows = []

for c in sorted(fdhg - dfs):
    diff_rows.append({
        "comparison": "fdhg_dmax1_minus_dfs",
        "column": c,
    })

for c in sorted(dfs - fdhg):
    diff_rows.append({
        "comparison": "dfs_minus_fdhg_dmax1",
        "column": c,
    })

for c in sorted(dmax2 - fdhg):
    diff_rows.append({
        "comparison": "dmax1_dmax2_ap_minus_fdhg_dmax1",
        "column": c,
    })

pd.DataFrame(diff_rows).to_csv(OUT_DIR / "relstack_gbdt_column_diffs_seed41.csv", index=False)

print("\nWrote:")
print(" -", OUT_DIR / "relstack_gbdt_column_inventory_seed41.csv")
print(" -", OUT_DIR / "relstack_gbdt_column_diffs_seed41.csv")
