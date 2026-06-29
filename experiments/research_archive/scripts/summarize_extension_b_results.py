import argparse
import json
from pathlib import Path
import pandas as pd


def infer_variant(path):
    name = path.name

    if "regen_dmax1_fdhg_full_plus_dmax2_topk16" in name:
        return "regen_dmax1_fdhg_full_plus_dmax2_topk16"
    if "regen_dmax1_fdhg_full_plus_dmax2_random16" in name:
        return "regen_dmax1_fdhg_full_plus_dmax2_random16"
    if "regen_dmax1_fdhg_full_alone" in name:
        return "regen_dmax1_fdhg_full_alone"
    if "regen_dmax1_dfs_alone" in name:
        return "regen_dmax1_dfs_alone"

    if "dmax2_topk16" in name:
        return "dmax2_topk16_only"
    if "dmax2_random16" in name:
        return "dmax2_random16_only"
    if "dmax2_topk64" in name:
        return "dmax2_topk64_only"
    if "dmax2_random64" in name:
        return "dmax2_random64_only"
    if "dmax2_all" in name:
        return "dmax2_all_only"

    return "unknown"


def flatten_result(d):
    row = {}

    for k, v in d.items():
        if k == "metrics" and isinstance(v, dict):
            for mk, mv in v.items():
                row[mk] = mv
        elif isinstance(v, (str, int, float, bool)) or v is None:
            row[k] = v

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="results/extension_b_dmax2/runs")
    parser.add_argument("--out_csv", default="results/extension_b_dmax2/extension_b_summary.csv")
    args = parser.parse_args()

    rows = []
    for p in sorted(Path(args.runs_dir).glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue

        row = flatten_result(d)
        row["file"] = str(p)
        row["variant"] = infer_variant(p)
        rows.append(row)

    if not rows:
        print("No JSON result files found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)

    show_cols = [
        c for c in [
            "variant", "seed", "n_features", "n_train", "n_val",
            "accuracy", "roc_auc", "average_precision", "log_loss", "file"
        ]
        if c in df.columns
    ]

    print("\n=== RAW RESULTS ===")
    print(df[show_cols].sort_values(["variant", "seed", "file"]).to_string(index=False))

    metric_cols = [
        c for c in ["n_features", "accuracy", "roc_auc", "average_precision", "log_loss"]
        if c in df.columns
    ]

    print("\n=== SUMMARY BY VARIANT ===")
    print(df.groupby("variant")[metric_cols].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
