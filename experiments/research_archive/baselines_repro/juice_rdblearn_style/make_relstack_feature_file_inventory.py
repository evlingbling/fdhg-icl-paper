from pathlib import Path
import pandas as pd
import re

OUT_DIR = Path("baselines_repro/juice_rdblearn_style/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

paths = sorted(Path("results").rglob("*.parquet"))

rows = []
for p in paths:
    s = str(p)
    low = s.lower()

    if not (("rel_stack" in low or "rel-stack" in low or "relstack" in low) and ("user_badge" in low or "user-badge" in low)):
        continue

    split = "unknown"
    name = p.name.lower()
    if name.startswith("train_") or "/train" in low:
        split = "train"
    elif name.startswith("val_") or "/val" in low:
        split = "val"
    elif name.startswith("test_") or "/test" in low:
        split = "test"

    kind = "unknown"
    if "combined" in name:
        kind = "combined"
    elif "features" in name:
        kind = "features"

    variant_guess = "unknown"
    parent = str(p.parent).lower()

    if "dmax2_random_same_budget" in parent:
        variant_guess = "dmax2_random_same_budget"
    elif "dmax2_topk" in parent:
        variant_guess = "dmax2_topk"
    elif "dmax2_all" in parent:
        variant_guess = "dmax2_all"
    elif "supervised_ap" in parent:
        variant_guess = "supervised_ap"
    elif "supervised_auc" in parent:
        variant_guess = "supervised_auc"
    elif "fdhg" in parent and "dmax1" in parent:
        variant_guess = "fdhg_dmax1"
    elif "dfs" in parent:
        variant_guess = "dfs"
    elif "naive" in parent:
        variant_guess = "naive"
    elif "target" in parent:
        variant_guess = "target_only"

    seed = None
    m = re.search(r"seed(\d+)", s)
    if m:
        seed = int(m.group(1))

    topk = None
    m = re.search(r"topk(\d+)", s.lower())
    if m:
        topk = int(m.group(1))

    try:
        df_head = pd.read_parquet(p).head(5)
        n_cols = len(df_head.columns)
        columns = list(df_head.columns)
        status = "ok"
        err = ""
    except Exception as e:
        n_cols = None
        columns = []
        status = "read_failed"
        err = f"{type(e).__name__}: {e}"

    rows.append({
        "path": s,
        "parent": str(p.parent),
        "filename": p.name,
        "split": split,
        "kind": kind,
        "variant_guess": variant_guess,
        "seed": seed,
        "topk": topk,
        "n_cols_head": n_cols,
        "columns_head": ";".join(map(str, columns[:80])),
        "status": status,
        "error": err,
    })

out = pd.DataFrame(rows).sort_values(["variant_guess", "seed", "split", "kind", "path"])
out.to_csv(OUT_DIR / "relstack_feature_file_inventory.csv", index=False)

print("n_files:", len(out))
if len(out):
    print(out[["variant_guess","seed","topk","split","kind","n_cols_head","path"]].to_string(index=False))
print("Wrote:", OUT_DIR / "relstack_feature_file_inventory.csv")
