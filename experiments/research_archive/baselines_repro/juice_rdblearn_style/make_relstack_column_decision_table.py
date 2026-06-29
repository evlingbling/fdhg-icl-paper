from pathlib import Path
import pandas as pd

IN = Path("baselines_repro/juice_rdblearn_style/results/relstack_gbdt_column_diffs_seed41.csv")
OUT = Path("baselines_repro/juice_rdblearn_style/results/relstack_fdhg_residual_column_summary.csv")

df = pd.read_csv(IN)

rows = []
for _, r in df.iterrows():
    col = r["column"]
    comp = r["comparison"]

    if col.startswith("f_amb__"):
        block = "ambiguity_residual"
        interpretation = "FDHG dmax1 dependency/ambiguity-derived residual feature"
    elif col.startswith("dmax2_"):
        block = "dmax2_residual"
        interpretation = "FDHG dmax2 typed second-hop residual feature"
    elif "__is_missing" in col:
        block = "missing_indicator"
        interpretation = "Missingness indicator for generated feature"
    else:
        block = "other"
        interpretation = "Needs manual inspection"

    rows.append({
        "comparison": comp,
        "column": col,
        "block": block,
        "interpretation": interpretation,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print("Wrote:", OUT)
