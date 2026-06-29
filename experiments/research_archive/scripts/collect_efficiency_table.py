import os
import re
import json
import glob
import pandas as pd
from pathlib import Path

OUT_DIR = Path("results/efficiency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DIR = Path("results/final_tables")
FINAL_DIR.mkdir(parents=True, exist_ok=True)

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def infer_dataset_task_variant(path):
    s = str(path)

    dataset = "unknown"
    task = "unknown"
    variant = "unknown"
    seed = None

    m = re.search(r"(rel-[a-z0-9]+)[_/.-]([a-z0-9-]+)", s)
    if m:
        dataset = m.group(1)
        task = m.group(2)

    m2 = re.search(r"seed(\d+)", s)
    if m2:
        seed = int(m2.group(1))

    lower = s.lower()
    if "target" in lower and "target_only" in lower:
        variant = "target_only"
    elif "naive" in lower:
        variant = "naive"
    elif "dfs" in lower and "fdhg" not in lower:
        variant = "dfs"
    elif "fdhg" in lower and "dmax2" in lower:
        variant = "fdhg_dmax1_plus_dmax2"
    elif "fdhg" in lower:
        variant = "fdhg_dmax1"
    elif "xgb" in lower:
        variant = "xgb"
    elif "catboost" in lower:
        variant = "catboost"

    return dataset, task, variant, seed

def collect_feature_matrix_rows():
    rows = []

    patterns = [
        "results/**/*.parquet",
        "results/**/*.csv",
    ]

    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            p = Path(path)

            # Avoid massive/final summary CSVs except feature-like files.
            lower = str(p).lower()
            if p.suffix == ".csv":
                if not any(k in lower for k in ["features", "combined", "runs", "summary"]):
                    continue
                if "final_tables" in lower and "gbdt" not in lower and "main" not in lower:
                    continue

            if not any(k in lower for k in [
                "train_combined",
                "val_combined",
                "test_combined",
                "gbdt_compiled_features",
                "clean_main",
                "ablation",
                "extension",
                "phase2_main_runs",
            ]):
                continue

            dataset, task, variant, seed = infer_dataset_task_variant(p)

            n_rows = None
            n_cols = None
            n_features = None
            size_mb = None

            try:
                size_mb = p.stat().st_size / (1024 ** 2)
            except Exception:
                pass

            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                    n_rows, n_cols = df.shape
                    target_cols = [c for c in ["target", "WillGetBadge", "label", "y", "outcome"] if c in df.columns]
                    n_features = n_cols - len(target_cols)
                elif p.suffix == ".csv" and p.stat().st_size < 50 * 1024 * 1024:
                    df = pd.read_csv(p)
                    n_rows, n_cols = df.shape
                    if "n_features" in df.columns:
                        n_features = pd.to_numeric(df["n_features"], errors="coerce").mean()
                    elif "n_features_total_mean" in df.columns:
                        n_features = pd.to_numeric(df["n_features_total_mean"], errors="coerce").mean()
            except Exception:
                pass

            rows.append({
                "source_type": "feature_or_result_file",
                "path": str(p),
                "dataset": dataset,
                "task": task,
                "variant": variant,
                "seed": seed,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_features_observed": n_features,
                "file_size_mb": size_mb,
            })

    return pd.DataFrame(rows)

def collect_json_metric_rows():
    rows = []
    for path in glob.glob("results/**/*.json", recursive=True):
        p = Path(path)
        data = safe_read_json(p)
        if data is None:
            continue

        dataset, task, variant, seed = infer_dataset_task_variant(p)

        flat = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    flat[k] = v

        rows.append({
            "path": str(p),
            "dataset": dataset,
            "task": task,
            "variant": variant,
            "seed": seed,
            **flat,
        })

    return pd.DataFrame(rows)

def grep_times():
    rows = []
    time_patterns = [
        r"FD discovery.*?([0-9.]+)\s*(s|sec|seconds|min|minutes)",
        r"feature materialization.*?([0-9.]+)\s*(s|sec|seconds|min|minutes)",
        r"materialization.*?([0-9.]+)\s*(s|sec|seconds|min|minutes)",
        r"inference.*?([0-9.]+)\s*(s|sec|seconds|min|minutes)",
        r"evaluation.*?([0-9.]+)\s*(s|sec|seconds|min|minutes)",
        r"elapsed.*?([0-9.]+)\s*(s|sec|seconds|min|minutes)",
        r"runtime.*?([0-9.]+)\s*(s|sec|seconds|min|minutes)",
        r"time.*?([0-9.]+)\s*(s|sec|seconds|min|minutes)",
    ]

    for pat in ["logs/**/*.log", "results/**/*.log", "results/**/*.txt"]:
        for path in glob.glob(pat, recursive=True):
            p = Path(path)
            dataset, task, variant, seed = infer_dataset_task_variant(p)

            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue

            lower = text.lower()
            if not any(k in lower for k in ["time", "elapsed", "runtime", "seconds", "materialization", "inference", "discovery"]):
                continue

            for line in text.splitlines():
                l = line.strip()
                ll = l.lower()
                if not any(k in ll for k in ["time", "elapsed", "runtime", "seconds", "materialization", "inference", "discovery"]):
                    continue

                for rgx in time_patterns:
                    m = re.search(rgx, l, flags=re.IGNORECASE)
                    if m:
                        value = float(m.group(1))
                        unit = m.group(2).lower()
                        seconds = value * 60 if unit.startswith("min") else value

                        rows.append({
                            "path": str(p),
                            "dataset": dataset,
                            "task": task,
                            "variant": variant,
                            "seed": seed,
                            "line": l[:500],
                            "time_seconds_extracted": seconds,
                        })
                        break

    return pd.DataFrame(rows)

def build_minimal_efficiency_summary(feature_df, json_df, time_df):
    rows = []

    # Prefer final summary CSVs for stable feature counts.
    candidate_summary_paths = [
        "results/final_tables/clean_main_4task_summary.csv",
        "results/final_tables/relstack_user_badge_ablation_all_summary.csv",
        "results/final_tables/amazon_item_churn_ablation_summary.csv",
        "results/final_tables/f1_driver_dnf_temporal_diagnostic_summary.csv",
        "results/final_tables/priority1_gbdt_relstack_summary.csv",
        "results/final_tables/gbdt_compiled_features_summary.csv",
    ]

    for path in candidate_summary_paths:
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        for _, r in df.iterrows():
            dataset = r.get("dataset", "unknown")
            task = r.get("task", "unknown")
            variant = r.get("variant", "unknown")

            n_features = None
            for c in ["n_features_total_mean", "n_features_mean", "n_features"]:
                if c in df.columns:
                    n_features = r.get(c)
                    break

            rows.append({
                "dataset": dataset,
                "task": task,
                "variant": variant,
                "source_summary": path,
                "n_runs": r.get("n_runs", None),
                "seeds": r.get("seeds", None),
                "n_features_mean": n_features,
                "candidate_programs": "NA_not_logged",
                "selected_programs": n_features,
                "fd_discovery_time_sec": "NA_not_logged",
                "feature_materialization_time_sec": "NA_not_logged",
                "decoder_eval_time_sec": "NA_not_logged",
                "peak_memory_mb": "NA_not_logged",
                "note": "Feature counts collected from final result summaries; timing fields require explicit instrumentation if absent from logs.",
            })

    summary = pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "task", "variant", "source_summary"],
        keep="first"
    )

    return summary

def main():
    print("[1/4] Collecting feature/result file metadata...")
    feature_df = collect_feature_matrix_rows()
    feature_df.to_csv(OUT_DIR / "efficiency_feature_file_inventory.csv", index=False)
    feature_df.to_csv(FINAL_DIR / "efficiency_feature_file_inventory.csv", index=False)

    print("[2/4] Collecting JSON metric metadata...")
    json_df = collect_json_metric_rows()
    json_df.to_csv(OUT_DIR / "efficiency_json_metric_inventory.csv", index=False)
    json_df.to_csv(FINAL_DIR / "efficiency_json_metric_inventory.csv", index=False)

    print("[3/4] Grepping logs for timing lines...")
    time_df = grep_times()
    time_df.to_csv(OUT_DIR / "efficiency_timing_lines_extracted.csv", index=False)
    time_df.to_csv(FINAL_DIR / "efficiency_timing_lines_extracted.csv", index=False)

    print("[4/4] Building minimal efficiency summary...")
    summary = build_minimal_efficiency_summary(feature_df, json_df, time_df)
    summary.to_csv(OUT_DIR / "efficiency_minimal_summary.csv", index=False)
    summary.to_csv(FINAL_DIR / "efficiency_minimal_summary.csv", index=False)

    print("\n=== Minimal efficiency summary ===")
    if len(summary):
        print(summary.to_string(index=False))
    else:
        print("[WARN] No summary rows found.")

    print("\n=== Timing lines extracted ===")
    if len(time_df):
        print(time_df.head(80).to_string(index=False))
    else:
        print("[WARN] No timing lines found in logs.")

    print("\nSaved:")
    print(OUT_DIR / "efficiency_feature_file_inventory.csv")
    print(OUT_DIR / "efficiency_json_metric_inventory.csv")
    print(OUT_DIR / "efficiency_timing_lines_extracted.csv")
    print(OUT_DIR / "efficiency_minimal_summary.csv")
    print(FINAL_DIR / "efficiency_minimal_summary.csv")

if __name__ == "__main__":
    main()
