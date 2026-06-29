#!/usr/bin/env python3
"""
Aggregate FDHG-ICL experiment outputs into standardized final CSV tables.

Outputs:
  results/final_tables/final_all_runs.csv
  results/final_tables/final_task_summary.csv
  results/final_tables/final_failure_log.csv

This script is intentionally read-only with respect to experiment outputs.
It scans JSON/CSV result files under results/ and normalizes them into one schema.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


FINAL_COLUMNS = [
    "dataset",
    "task",
    "variant",
    "seed",
    "model",
    "split",
    "n_train",
    "n_val",
    "n_features",
    "feature_budget",
    "dmax",
    "uses_afd",
    "uses_ambiguity",
    "uses_dmax2",
    "uses_supervised_ranker",
    "is_fdhg_fallback",
    "accuracy",
    "roc_auc",
    "average_precision",
    "log_loss",
    "runtime_sec",
    "feature_path",
    "result_path",
    "status",
    "failure_reason",
]


METRIC_ALIASES = {
    "accuracy": ["accuracy", "acc", "acc_0.5"],
    "roc_auc": ["roc_auc", "auroc", "auc"],
    "average_precision": ["average_precision", "avg_precision", "ap", "pr_auc"],
    "log_loss": ["log_loss", "loss"],
    "runtime_sec": ["runtime_sec", "runtime_seconds", "elapsed_sec", "elapsed_seconds", "time_sec"],
}


VARIANT_PATTERNS = [
    ("fdhg_dmax1_plus_dmax2_supervised_ap_topk16", [
        "supervised_ap", "ap_topk16", "dmax2_supervised_ap", "plus_supervised_ap"
    ]),
    ("fdhg_dmax1_plus_dmax2_supervised_auc_topk16", [
        "supervised_auc", "auc_topk16", "dmax2_supervised_auc", "plus_supervised_auc"
    ]),
    ("fdhg_dmax1_plus_dmax2_topk16", [
        "dmax1_plus_dmax2_topk16", "fdhg_plus_dmax2_topk16", "plus_dmax2_topk16"
    ]),
    ("fdhg_dmax1_plus_dmax2_random16", [
        "dmax1_plus_dmax2_random16", "fdhg_plus_dmax2_random16", "plus_dmax2_random16"
    ]),
    ("dmax2_only_topk16", [
        "dmax2_topk16_only", "dmax2_only_topk16", "dmax2_topK16_only", "topk16_only"
    ]),
    ("dmax2_only_random16", [
        "dmax2_random16_only", "dmax2_only_random16", "random16_only"
    ]),
    ("fdhg_dmax1_shuffle_ambiguity", [
        "shuffle_ambiguity", "shuffled_ambiguity"
    ]),
    ("fdhg_dmax1_random_same_budget", [
        "random_same_budget", "same_budget_random"
    ]),
    ("fdhg_dmax1_no_ambiguity", [
        "no_ambiguity", "without_ambiguity", "noamb"
    ]),
    ("fdhg_dmax1", [
        "fdhg_full", "fdhg_dmax1", "fdhg"
    ]),
    ("target_only", [
        "target_only", "target-only"
    ]),
    ("naive", [
        "naive", "latest"
    ]),
    ("dfs", [
        "dfs", "dfs_agg"
    ]),
]


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return {"records": obj}
    except Exception:
        return None


def get_nested(obj: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, new_key))
        else:
            out[new_key] = v
    return out


def find_metric(flat: Dict[str, Any], canonical: str) -> Any:
    aliases = METRIC_ALIASES[canonical]
    for alias in aliases:
        for key in flat:
            if key == alias or key.endswith("." + alias):
                return flat[key]
    return None


def as_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def as_int(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def infer_seed(text: str, obj: Dict[str, Any]) -> Optional[int]:
    direct = get_nested(obj, ["seed", "random_seed"])
    if direct is not None:
        return as_int(direct)
    m = re.search(r"seed[_-]?(\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def infer_dataset_task(text: str, obj: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    dataset = get_nested(obj, ["dataset", "dataset_name"])
    task = get_nested(obj, ["task", "task_name"])

    if dataset and task:
        return str(dataset), str(task)

    # RelBench-style names such as rel-stack_user-badge, rel-amazon_item-churn
    m = re.search(
        r"(rel-[a-zA-Z0-9]+)[_/.-]+([a-zA-Z0-9]+[-_][a-zA-Z0-9]+)",
        text,
    )
    if m:
        dataset = dataset or m.group(1)
        task = task or m.group(2).replace("_", "-")

    return (str(dataset) if dataset else None, str(task) if task else None)


def normalize_variant(raw: Optional[str], path_text: str) -> str:
    raw_str = str(raw).strip() if raw is not None else ""
    raw_lower = raw_str.lower()
    source = f"{raw_str} {path_text}".lower()

    alias = {
        "shuffle_all_ambiguity": "fdhg_dmax1_shuffle_ambiguity",
        "random_same_budget": "fdhg_dmax1_random_same_budget",
        "fdhg_full": "fdhg_dmax1",
        "fdhg_wo_ambiguity": "fdhg_dmax1_no_ambiguity",
        "dfs_or_wo_ambiguity": "fdhg_dmax1_no_ambiguity",

        "dmax2_topk16_only": "dmax2_only_topk16",
        "dmax2_random16_only": "dmax2_only_random16",
        "dmax2_supervised_auc_topk16_only": "dmax2_only_supervised_auc_topk16",
        "dmax2_supervised_ap_topk16_only": "dmax2_only_supervised_ap_topk16",

        "regen_dmax1_dfs_alone": "dfs",
        "regen_dmax1_fdhg_full_alone": "fdhg_dmax1",
        "regen_dmax1_fdhg_full_plus_dmax2_topk16": "fdhg_dmax1_plus_dmax2_topk16",
        "regen_dmax1_fdhg_full_plus_dmax2_random16": "fdhg_dmax1_plus_dmax2_random16",
        "regen_dmax1_fdhg_full_plus_dmax2_supervised_auc_topk16": "fdhg_dmax1_plus_dmax2_supervised_auc_topk16",
        "regen_dmax1_fdhg_full_plus_dmax2_supervised_ap_topk16": "fdhg_dmax1_plus_dmax2_supervised_ap_topk16",
    }

    if raw_lower in alias:
        return alias[raw_lower]

    # Main dmax1 + dmax2 integrated variants first.
    if "regen_dmax1_fdhg_full_plus_dmax2_supervised_ap_topk16" in source:
        return "fdhg_dmax1_plus_dmax2_supervised_ap_topk16"
    if "regen_dmax1_fdhg_full_plus_dmax2_supervised_auc_topk16" in source:
        return "fdhg_dmax1_plus_dmax2_supervised_auc_topk16"
    if "regen_dmax1_fdhg_full_plus_dmax2_topk16" in source:
        return "fdhg_dmax1_plus_dmax2_topk16"
    if "regen_dmax1_fdhg_full_plus_dmax2_random16" in source:
        return "fdhg_dmax1_plus_dmax2_random16"

    if "dmax1" in source and "dmax2" in source and "supervised_ap" in source:
        return "fdhg_dmax1_plus_dmax2_supervised_ap_topk16"
    if "dmax1" in source and "dmax2" in source and "supervised_auc" in source:
        return "fdhg_dmax1_plus_dmax2_supervised_auc_topk16"
    if "dmax1" in source and "dmax2" in source and "random16" in source:
        return "fdhg_dmax1_plus_dmax2_random16"
    if "dmax1" in source and "dmax2" in source and "topk16" in source:
        return "fdhg_dmax1_plus_dmax2_topk16"

    # dmax2-only variants.
    if "dmax2_supervised_ap_topk16_only" in source:
        return "dmax2_only_supervised_ap_topk16"
    if "dmax2_supervised_auc_topk16_only" in source:
        return "dmax2_only_supervised_auc_topk16"
    if "dmax2_random16_only" in source or "dmax2_only_random16" in source:
        return "dmax2_only_random16"
    if "dmax2_topk16_only" in source or "dmax2_only_topk16" in source:
        return "dmax2_only_topk16"

    # Regenerated standalone.
    if "regen_dmax1_fdhg_full_alone" in source:
        return "fdhg_dmax1"
    if "regen_dmax1_dfs_alone" in source:
        return "dfs"

    # FDHG dmax1 ablations before DFS, because FDHG result paths often contain dfs_agg.
    if "shuffle_all_ambiguity" in source or "shuffle_ambiguity" in source:
        return "fdhg_dmax1_shuffle_ambiguity"
    if "random_same_budget" in source or "same_budget_random" in source:
        return "fdhg_dmax1_random_same_budget"
    if "fdhg_wo_ambiguity" in source or "no_ambiguity" in source or "without_ambiguity" in source:
        return "fdhg_dmax1_no_ambiguity"
    if "fdhg_full" in source or "fdhg_dmax1" in source:
        return "fdhg_dmax1"

    # Main baselines.
    if "target_only" in source or "target-only" in source:
        return "target_only"
    if "naive" in source:
        return "naive"

    # Temporal/operator ablations.
    known_extension_variants = [
        "last_only",
        "dfs_plus_last",
        "fdhg_plus_last",
        "temporal_full",
        "history_only",
        "trend_only",
        "recency_only",
        "results_only",
        "standings_only",
        "target_temporal_smoke",
        "no_statusid",
        "no_ambiguity_missing_indicators",
        "drop_entropy",
        "drop_conflict_count",
        "drop_majconf",
        "drop_support_count",
        "value_only_entropy",
        "value_only_conflict_count",
        "value_only_majconf",
        "value_only_support_count",
        "only_entropy",
        "only_conflict_count",
        "only_majconf",
        "only_support_count",
        "single_value_only_best_entropy",
        "single_operator_only_best_conflict_count",
        "dmax2_all",
        "dmax2_all_only",
        "dmax2_topk",
        "dmax2_topk64_only",
        "dmax2_random64_only",
    ]

    for v in known_extension_variants:
        if raw_lower == v or v in source:
            if v == "no_statusid":
                return "no_statusId"
            return v

    # Plain DFS last, because many FDHG result files include dfs_agg in path.
    if raw_lower == "dfs" or raw_lower == "dfs_agg":
        return "dfs"
    if re.search(r"(^|[_/.-])dfs([_/.-]|$)", source):
        return "dfs"

    if re.search(r"(^|[_/.-])fdhg([_/.-]|$)", source) and "dmax2" not in source:
        return "fdhg_dmax1"

    return raw_str if raw_str else "unknown"


def infer_dmax(variant: str, text: str, obj: Dict[str, Any]) -> Optional[int]:
    direct = get_nested(obj, ["dmax", "max_depth"])
    if direct is not None:
        return as_int(direct)
    if "dmax2" in variant or "dmax2" in text.lower():
        return 2
    if "fdhg" in variant or "dfs" in variant:
        return 1
    return None


def infer_bool_flags(variant: str, text: str, obj: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    lower = text.lower() + " " + variant.lower()

    def direct_bool(*keys: str) -> Optional[bool]:
        for k in keys:
            if k in obj:
                v = obj[k]
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.lower() in ["true", "1", "yes", "y"]
                if isinstance(v, (int, float)):
                    return bool(v)
        return None

    uses_dmax2 = direct_bool("uses_dmax2")
    if uses_dmax2 is None:
        uses_dmax2 = "dmax2" in lower

    uses_supervised_ranker = direct_bool("uses_supervised_ranker")
    if uses_supervised_ranker is None:
        uses_supervised_ranker = "supervised" in lower

    uses_ambiguity = direct_bool("uses_ambiguity")
    if uses_ambiguity is None:
        if "no_ambiguity" in lower or "noamb" in lower:
            uses_ambiguity = False
        elif "fdhg" in lower or "ambiguity" in lower:
            uses_ambiguity = True
        else:
            uses_ambiguity = False

    uses_afd = direct_bool("uses_afd")
    if uses_afd is None:
        if "target_only" in lower or "naive" in lower or variant == "dfs":
            uses_afd = False
        elif "fdhg" in lower or "afd" in lower or "ambiguity" in lower:
            uses_afd = True
        else:
            uses_afd = False

    is_fdhg_fallback = direct_bool("is_fdhg_fallback", "fdhg_fallback", "fallback")
    if is_fdhg_fallback is None:
        is_fdhg_fallback = "fallback" in lower

    return {
        "uses_afd": uses_afd,
        "uses_ambiguity": uses_ambiguity,
        "uses_dmax2": uses_dmax2,
        "uses_supervised_ranker": uses_supervised_ranker,
        "is_fdhg_fallback": is_fdhg_fallback,
    }


def infer_feature_budget(variant: str, obj: Dict[str, Any]) -> Optional[int]:
    direct = get_nested(obj, ["feature_budget", "k", "top_k", "topK"])
    if direct is not None:
        return as_int(direct)

    m = re.search(r"topk(\d+)|random(\d+)", variant.lower())
    if m:
        return int(m.group(1) or m.group(2))

    return None


def extract_feature_path(obj: Dict[str, Any]) -> Optional[str]:
    for key in [
        "feature_path",
        "features_path",
        "feature_matrix_path",
        "train_feature_path",
        "input_path",
    ]:
        if key in obj:
            return str(obj[key])
    return None


def normalize_json_result(path: Path, obj: Dict[str, Any]) -> Dict[str, Any]:
    flat = flatten_dict(obj)
    path_text = str(path)

    dataset, task = infer_dataset_task(path_text, obj)
    raw_variant = get_nested(obj, ["variant", "setting", "method", "name"])
    variant = normalize_variant(raw_variant, path_text)

    row = {col: None for col in FINAL_COLUMNS}
    row["dataset"] = dataset
    row["task"] = task
    row["variant"] = variant
    row["seed"] = infer_seed(path_text, obj)
    row["model"] = get_nested(obj, ["model", "model_name"]) or ("tabpfn" if "tabpfn" in path_text.lower() else None)
    row["split"] = get_nested(obj, ["split"]) or "val"

    row["n_train"] = as_int(get_nested(obj, ["n_train", "num_train", "train_rows"]))
    row["n_val"] = as_int(get_nested(obj, ["n_val", "num_val", "val_rows", "eval_rows"]))
    row["n_features"] = as_int(get_nested(obj, ["n_features", "num_features", "feature_count"]))
    row["feature_budget"] = infer_feature_budget(variant, obj)
    row["dmax"] = infer_dmax(variant, path_text, obj)

    flags = infer_bool_flags(variant, path_text, obj)
    row.update(flags)

    row["accuracy"] = as_float(find_metric(flat, "accuracy"))
    row["roc_auc"] = as_float(find_metric(flat, "roc_auc"))
    row["average_precision"] = as_float(find_metric(flat, "average_precision"))
    row["log_loss"] = as_float(find_metric(flat, "log_loss"))
    row["runtime_sec"] = as_float(find_metric(flat, "runtime_sec"))

    row["feature_path"] = extract_feature_path(obj)
    row["result_path"] = str(path)

    failure_reason = get_nested(obj, ["failure_reason", "error", "exception", "message"])
    row["failure_reason"] = str(failure_reason) if failure_reason else None

    has_metric = any(row[m] is not None for m in ["accuracy", "roc_auc", "average_precision", "log_loss"])
    row["status"] = "ok" if has_metric and not row["failure_reason"] else "failed"

    return row


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    rows = []
    for _, r in df.iterrows():
        obj = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
        row = normalize_json_result(path, obj)

        # Preserve explicit CSV columns if already close to final schema.
        # Do NOT overwrite normalized fields with old/raw CSV names.
        protected_cols = {
            "variant",
            "status",
            "uses_afd",
            "uses_ambiguity",
            "uses_dmax2",
            "uses_supervised_ranker",
            "is_fdhg_fallback",
            "dmax",
            "feature_budget",
        }

        for col in FINAL_COLUMNS:
            if col in protected_cols:
                continue
            if col in obj and obj[col] is not None:
                row[col] = obj[col]

        rows.append(row)

    return rows


def discover_result_files(results_root: Path) -> List[Path]:
    candidates = []

    # Files that are useful artifacts, but should NOT become final_all_runs rows.
    exclude_names = {
        "paths.json",
        "manifest.json",
        "rank_scores.csv",
        "generation_summary.json",
        "calibrator_summary.json",
        "dropped_temporal_features.csv",
        "afd_edges_product_dmax1.csv",
        "phase2_task_summary.csv",
        "phase2_win_tie_loss.csv",
        "extension_b_summary.csv",
        "extension_c_summary.csv",
    }

    exclude_names.add("phase2_failures.csv")

    for p in results_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".json", ".csv"]:
            continue

        name = p.name.lower()
        full = str(p).lower()

        # Avoid re-reading final outputs.
        if "final_tables" in full:
            continue

        # Exclude metadata / feature-discovery artifacts.
        if name in exclude_names:
            continue
        if "/features/" in full:
            continue

        # Always keep actual dmax2 / supervised-ranker run JSONs.
        if (
            "results/extension_b_dmax2/runs/" in full
            and p.suffix.lower() == ".json"
        ):
            candidates.append(p)
            continue

        if (
            "results/extension_c_ranker/runs/" in full
            and p.suffix.lower() == ".json"
        ):
            candidates.append(p)
            continue

        # Keep likely model-evaluation outputs.
        keep = (
            "tabpfn" in name
            or "result" in name
            or name.endswith("_metrics.json")
            or name.endswith("_eval.json")
            or "phase2_main_runs" in name
        )

        if keep:
            candidates.append(p)

    return sorted(set(candidates))


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[FINAL_COLUMNS].copy()

    # Fill missing status after CSV merging.
    metric_cols = ["accuracy", "roc_auc", "average_precision", "log_loss"]

    def infer_status_from_row(r):
        cur = r.get("status", None)
        if cur is not None and not pd.isna(cur) and str(cur).strip() != "":
            return str(cur)

        failure = r.get("failure_reason", None)
        if failure is not None and not pd.isna(failure) and str(failure).strip() != "":
            return "failed"

        has_metric = any(
            r.get(m, None) is not None and not pd.isna(r.get(m, None))
            for m in metric_cols
        )
        return "ok" if has_metric else "failed"

    df["status"] = df.apply(infer_status_from_row, axis=1)

    # Type cleanup.
    int_cols = ["seed", "n_train", "n_val", "n_features", "feature_budget", "dmax"]
    float_cols = ["accuracy", "roc_auc", "average_precision", "log_loss", "runtime_sec"]
    bool_cols = [
        "uses_afd",
        "uses_ambiguity",
        "uses_dmax2",
        "uses_supervised_ranker",
        "is_fdhg_fallback",
    ]

    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in bool_cols:
        df[c] = df[c].map(lambda x: None if pd.isna(x) else bool(x)).astype("boolean")

    # Standardize strings.
    for c in ["dataset", "task", "variant", "model", "split", "status", "failure_reason"]:
        df[c] = df[c].astype("string")

    # Remove obvious duplicate rows by result path first.
    df = df.drop_duplicates(subset=["result_path", "variant", "seed"], keep="last")

    # Sort for readability.
    df = df.sort_values(
        by=["dataset", "task", "variant", "seed", "result_path"],
        na_position="last",
    ).reset_index(drop=True)

    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].fillna("") == "ok"].copy()

    group_cols = [
        "dataset",
        "task",
        "variant",
        "model",
        "split",
        "feature_budget",
        "dmax",
        "uses_afd",
        "uses_ambiguity",
        "uses_dmax2",
        "uses_supervised_ranker",
        "is_fdhg_fallback",
    ]

    metrics = ["accuracy", "roc_auc", "average_precision", "log_loss", "runtime_sec", "n_features"]

    summary = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("seed", "count"),
            seeds=("seed", lambda x: ",".join(str(int(v)) for v in sorted(x.dropna().unique()))),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            n_features_mean=("n_features", "mean"),
            n_features_std=("n_features", "std"),
        )
        .reset_index()
    )

    return summary.sort_values(["dataset", "task", "variant"]).reset_index(drop=True)


def make_failure_log(df: pd.DataFrame) -> pd.DataFrame:
    failed = df[df["status"].fillna("") != "ok"].copy()
    cols = [
        "dataset",
        "task",
        "variant",
        "seed",
        "model",
        "status",
        "failure_reason",
        "feature_path",
        "result_path",
    ]
    return failed[cols].sort_values(["dataset", "task", "variant", "seed"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default="results/final_tables")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    files = discover_result_files(results_root)
    print(f"[aggregate] discovered {len(files)} candidate result files")

    for path in files:
        if path.suffix.lower() == ".json":
            obj = safe_read_json(path)
            if obj is None:
                continue
            rows.append(normalize_json_result(path, obj))

        elif path.suffix.lower() == ".csv":
            rows.extend(read_csv_rows(path))

    if not rows:
        raise RuntimeError(f"No result rows found under {results_root}")

    df = postprocess(pd.DataFrame(rows))
    summary = make_summary(df)
    failures = make_failure_log(df)

    all_path = out_dir / "final_all_runs.csv"
    summary_path = out_dir / "final_task_summary.csv"
    failure_path = out_dir / "final_failure_log.csv"

    df.to_csv(all_path, index=False)
    summary.to_csv(summary_path, index=False)
    failures.to_csv(failure_path, index=False)

    print(f"[aggregate] wrote {all_path} with {len(df)} rows")
    print(f"[aggregate] wrote {summary_path} with {len(summary)} rows")
    print(f"[aggregate] wrote {failure_path} with {len(failures)} rows")

    print("\n[aggregate] variant counts:")
    print(df["variant"].value_counts(dropna=False).to_string())

    print("\n[aggregate] status counts:")
    print(df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
