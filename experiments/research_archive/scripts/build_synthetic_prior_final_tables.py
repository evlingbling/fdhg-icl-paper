import json
from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/final_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
PROGRAM_RECOVERY_GLOB = sorted(Path("results/synthetic_prior").glob("program_recovery_k*.csv"))
TABPFN_DIR = Path("outputs/synthetic_tabpfn_oracle_gap/seed53_60_top4_with_random")
TABPFN_RESULTS = TABPFN_DIR / "synthetic_tabpfn_oracle_gap_results.csv"
TABPFN_GAPS = TABPFN_DIR / "synthetic_tabpfn_oracle_gap_summary.csv"
TABPFN_SUMMARY_JSON = TABPFN_DIR / "summary.json"

# Outputs
RECOVERY_BY_SEED = OUT_DIR / "synthetic_program_recovery_by_seed.csv"
RECOVERY_SUMMARY = OUT_DIR / "synthetic_program_recovery_summary.csv"
ORACLE_GAP_BY_SEED = OUT_DIR / "synthetic_oracle_gap_by_seed.csv"
ORACLE_GAP_SUMMARY = OUT_DIR / "synthetic_oracle_gap_summary.csv"

# -------------------------
# Program recovery aggregation
# -------------------------
recovery_frames = []
for p in PROGRAM_RECOVERY_GLOB:
    df = pd.read_csv(p)
    # Extract K from filename if not reliable.
    if "k" not in df.columns:
        k = int(p.stem.split("k")[-1])
        df["k"] = k
    recovery_frames.append(df)

if recovery_frames:
    recovery = pd.concat(recovery_frames, ignore_index=True)
    recovery.to_csv(RECOVERY_BY_SEED, index=False)

    agg = recovery.groupby("k", as_index=False).agg(
        n_seeds=("seed_dir", "nunique"),
        n_candidates_mean=("n_candidates", "mean"),
        n_candidates_std=("n_candidates", "std"),
        n_label_program_features_mean=("n_label_program_features", "mean"),
        ProgramRecall_at_K_mean=("ProgramRecall@K", "mean"),
        ProgramRecall_at_K_std=("ProgramRecall@K", "std"),
    )

    # FDRecall/FDP placeholders only if columns exist.
    for col in ["FDRecall@K", "FDP@K", "OracleGap"]:
        if col in recovery.columns:
            agg[f"{col}_mean"] = recovery.groupby("k")[col].mean().values
            agg[f"{col}_std"] = recovery.groupby("k")[col].std().values

    agg.to_csv(RECOVERY_SUMMARY, index=False)
else:
    recovery = pd.DataFrame()
    pd.DataFrame().to_csv(RECOVERY_BY_SEED, index=False)
    pd.DataFrame().to_csv(RECOVERY_SUMMARY, index=False)

# -------------------------
# Oracle gap aggregation
# -------------------------
if TABPFN_RESULTS.exists():
    results = pd.read_csv(TABPFN_RESULTS)
    gaps = pd.read_csv(TABPFN_GAPS) if TABPFN_GAPS.exists() else pd.DataFrame()

    # Save method-level rows.
    results.to_csv(ORACLE_GAP_BY_SEED, index=False)

    summary_rows = []
    for method, g in results.groupby("method"):
        row = {
            "method": method,
            "n_seeds": g["seed_dir"].nunique(),
            "n_features_mean": g["n_features"].mean(),
            "accuracy_mean": g["accuracy"].mean(),
            "accuracy_std": g["accuracy"].std(),
            "roc_auc_mean": g["roc_auc"].mean(),
            "roc_auc_std": g["roc_auc"].std(),
            "average_precision_mean": g["average_precision"].mean(),
            "average_precision_std": g["average_precision"].std(),
            "log_loss_mean": g["log_loss"].mean(),
            "log_loss_std": g["log_loss"].std(),
        }

        if len(gaps) and method != "oracle":
            gm = gaps[gaps["method"] == method]
            row["roc_auc_gap_vs_oracle_mean"] = gm["roc_auc_gap_vs_oracle"].mean()
            row["average_precision_gap_vs_oracle_mean"] = gm["average_precision_gap_vs_oracle"].mean()
            row["log_loss_gap_vs_oracle_mean"] = gm["log_loss_gap_vs_oracle"].mean()
        else:
            row["roc_auc_gap_vs_oracle_mean"] = 0.0 if method == "oracle" else None
            row["average_precision_gap_vs_oracle_mean"] = 0.0 if method == "oracle" else None
            row["log_loss_gap_vs_oracle_mean"] = 0.0 if method == "oracle" else None

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    order = {"weak_price_only": 0, "random_top4": 1, "ranker_top4": 2, "oracle": 3}
    summary["_order"] = summary["method"].map(order).fillna(999)
    summary = summary.sort_values(["_order", "method"]).drop(columns=["_order"])
    summary.to_csv(ORACLE_GAP_SUMMARY, index=False)

    if TABPFN_SUMMARY_JSON.exists():
        with open(TABPFN_SUMMARY_JSON) as f:
            print("=== Existing summary.json ===")
            print(json.dumps(json.load(f), indent=2))
else:
    pd.DataFrame().to_csv(ORACLE_GAP_BY_SEED, index=False)
    pd.DataFrame().to_csv(ORACLE_GAP_SUMMARY, index=False)

print("Saved:")
print(f"  {RECOVERY_BY_SEED}")
print(f"  {RECOVERY_SUMMARY}")
print(f"  {ORACLE_GAP_BY_SEED}")
print(f"  {ORACLE_GAP_SUMMARY}")

print("\n=== Synthetic oracle gap summary ===")
if ORACLE_GAP_SUMMARY.exists():
    print(pd.read_csv(ORACLE_GAP_SUMMARY).to_string(index=False))

print("\n=== Synthetic recovery summary ===")
if RECOVERY_SUMMARY.exists():
    rec_sum = pd.read_csv(RECOVERY_SUMMARY)
    if len(rec_sum):
        print(rec_sum.to_string(index=False))
    else:
        print("No recovery files found under results/synthetic_prior/program_recovery_k*.csv")
