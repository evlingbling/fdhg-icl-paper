import argparse
import json
from pathlib import Path
import pandas as pd


def infer_variant(name):
    if "regen_dmax1_fdhg_full_plus_dmax2_supervised_ap_topk16" in name:
        return "regen_dmax1_fdhg_full_plus_dmax2_supervised_ap_topk16"
    if "regen_dmax1_fdhg_full_plus_dmax2_supervised_auc_topk16" in name:
        return "regen_dmax1_fdhg_full_plus_dmax2_supervised_auc_topk16"
    if "dmax2_supervised_ap_topk16_only" in name:
        return "dmax2_supervised_ap_topk16_only"
    if "dmax2_supervised_auc_topk16_only" in name:
        return "dmax2_supervised_auc_topk16_only"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="results/extension_c_ranker/runs")
    parser.add_argument("--out_csv", default="results/extension_c_ranker/extension_c_summary.csv")
    args = parser.parse_args()

    rows = []
    for p in sorted(Path(args.runs_dir).glob("*.json")):
        with open(p) as f:
            d = json.load(f)

        row = {"file": str(p), "variant": infer_variant(p.name)}
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No result JSON files found.")
        return

    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)

    cols = [
        c for c in [
            "variant", "seed", "n_features",
            "accuracy", "roc_auc", "average_precision", "log_loss", "file"
        ]
        if c in df.columns
    ]

    print("\n=== RAW ===")
    print(df[cols].sort_values(["variant", "seed"]).to_string(index=False))

    metric_cols = [
        c for c in ["accuracy", "roc_auc", "average_precision", "log_loss"]
        if c in df.columns
    ]

    print("\n=== SUMMARY ===")
    print(df.groupby("variant")[metric_cols].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
