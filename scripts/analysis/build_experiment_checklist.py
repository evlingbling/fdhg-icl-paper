from pathlib import Path
import re
import pandas as pd
from relbench.tasks import get_task_names, get_task


FINAL = Path("results/final_tables")
OUT = FINAL / "full_target_experiment_checklist.csv"

# ------------------------------------------------------------------
# 1. 목표 실험군 정의
# ------------------------------------------------------------------

REL_V1_DATASETS = [
    "rel-amazon",
    "rel-avito",
    "rel-event",
    "rel-f1",
    "rel-hm",
    "rel-stack",
    "rel-trial",
    "rel-arxiv",
    "rel-ratebeer",
]

# v2 AutoComplete tasks가 들어 있는 dataset도 동일 registry에서 탐색
REL_V2_DATASETS = REL_V1_DATASETS

SALT_DATASET = "rel-salt"

DBINFER_DATASETS = [
    "dbinfer-avs",
    "dbinfer-diginetica",
    "dbinfer-retailrocket",
    "dbinfer-seznam",
    "dbinfer-amazon",
    "dbinfer-stackexchange",
    "dbinfer-outbrain-small",
]

TARGET_TASK_TYPES = {
    "binary_classification",
    "regression",
    "multiclass_classification",
}


def normalize_task_type(value):
    if value is None:
        return None

    text = str(value).lower()

    if "binary_classification" in text:
        return "binary_classification"
    if "multiclass_classification" in text:
        return "multiclass_classification"
    if "regression" in text:
        return "regression"

    return text


def classify_family(dataset, task):
    """
    프로젝트에서 사용한 구분:
    - rel-salt: SALT multiclass
    - dbinfer-*: 4DBInfer/DBInfer binary
    - AutoCompleteTask: RelBench v2
    - 기존 named task class: RelBench v1
    """
    obj = get_task(dataset, task, download=False)
    class_name = type(obj).__name__

    if dataset == "rel-salt":
        return "SALT"

    if dataset.startswith("dbinfer-"):
        return "4DBInfer"

    if class_name == "AutoCompleteTask":
        return "RelBench_v2"

    return "RelBench_v1"


# ------------------------------------------------------------------
# 2. registry에서 목표 task 수집
# ------------------------------------------------------------------

registry_rows = []

datasets = sorted(set(
    REL_V1_DATASETS
    + REL_V2_DATASETS
    + [SALT_DATASET]
    + DBINFER_DATASETS
))

for dataset in datasets:
    try:
        task_names = get_task_names(dataset)
    except Exception as exc:
        print(f"[WARN] registry unavailable: {dataset}: {exc}")
        continue

    for task_name in task_names:
        try:
            task = get_task(
                dataset,
                task_name,
                download=False,
            )
        except Exception as exc:
            print(
                f"[WARN] task unavailable: "
                f"{dataset}/{task_name}: {exc}"
            )
            continue

        task_type = normalize_task_type(
            getattr(task, "task_type", None)
        )

        family = classify_family(
            dataset,
            task_name,
        )

        # 원하는 실험 범위만 포함
        include = False

        if family in {"RelBench_v1", "RelBench_v2"}:
            include = task_type in {
                "binary_classification",
                "regression",
            }

        elif family == "SALT":
            include = (
                task_type
                == "multiclass_classification"
            )

        elif family == "4DBInfer":
            include = (
                task_type
                == "binary_classification"
            )

        if not include:
            continue

        registry_rows.append({
            "benchmark_family": family,
            "dataset": dataset,
            "task": task_name,
            "task_class": type(task).__name__,
            "task_type": task_type,
            "target_col": getattr(
                task,
                "target_col",
                None,
            ),
            "entity_table": getattr(
                task,
                "entity_table",
                None,
            ),
            "entity_col": getattr(
                task,
                "entity_col",
                None,
            ),
            "time_col": getattr(
                task,
                "time_col",
                None,
            ),
        })

registry = pd.DataFrame(registry_rows).drop_duplicates(
    ["benchmark_family", "dataset", "task"]
)

# DBInfer registry loading fails in fdhg310 because of the
# DGL GraphBolt / PyTorch mismatch. Add the two project tasks
# manually from their persisted final-table artifacts.
manual_dbinfer = pd.DataFrame([
    {
        "benchmark_family": "4DBInfer",
        "dataset": "dbinfer-avs",
        "task": "repeater",
        "task_class": "AVSRepeaterTask",
        "task_type": "binary_classification",
        "target_col": None,
        "entity_table": None,
        "entity_col": None,
        "time_col": None,
    },
    {
        "benchmark_family": "4DBInfer",
        "dataset": "dbinfer-retailrocket",
        "task": "cvr",
        "task_class": "RetailRocketCVRTask",
        "task_type": "binary_classification",
        "target_col": None,
        "entity_table": None,
        "entity_col": None,
        "time_col": None,
    },
])

registry = pd.concat(
    [registry, manual_dbinfer],
    ignore_index=True,
).drop_duplicates(
    ["benchmark_family", "dataset", "task"]
)

print("=== TARGET REGISTRY COUNTS ===")
print(
    registry.groupby(
        ["benchmark_family", "task_type"]
    ).size().to_string()
)


# ------------------------------------------------------------------
# 3. final_tables 전체에서 task별 증거 수집
# ------------------------------------------------------------------

result_rows = []

for _, task_row in registry.iterrows():
    family = task_row["benchmark_family"]
    dataset = task_row["dataset"]
    task_name = task_row["task"]
    task_type = task_row["task_type"]

    matched_files = set()
    gate_files = set()
    fallback_files = set()
    status_files = set()
    run_files = set()

    gate_outcomes = set()
    statuses = set()
    variants = set()
    decoders = set()
    seeds = set()

    has_rmse = False
    has_mae = False
    has_r2 = False

    has_accuracy = False
    has_auroc = False
    has_ap = False
    has_log_loss = False
    has_macro_f1 = False
    has_weighted_f1 = False

    selected_residual_count = None
    fallback_exact_match = None

    for path in sorted(FINAL.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if not {"dataset", "task"}.issubset(df.columns):
            continue

        mask = (
            df["dataset"].astype(str).eq(dataset)
            & df["task"].astype(str).eq(task_name)
        )

        if not mask.any():
            continue

        sub = df.loc[mask].copy()
        matched_files.add(str(path))
        lower_name = path.name.lower()

        if "gate" in lower_name:
            gate_files.add(str(path))
        if "fallback" in lower_name:
            fallback_files.add(str(path))
        if "status" in lower_name:
            status_files.add(str(path))
        if "run" in lower_name:
            run_files.add(str(path))

        if "gate_outcome" in sub.columns:
            gate_outcomes.update(
                sub["gate_outcome"]
                .dropna()
                .astype(str)
                .str.upper()
                .tolist()
            )

        if "status" in sub.columns:
            statuses.update(
                sub["status"]
                .dropna()
                .astype(str)
                .tolist()
            )

        for variant_col in [
            "variant",
            "baseline_variant",
            "candidate_variant",
            "base_variant",
        ]:
            if variant_col in sub.columns:
                variants.update(
                    sub[variant_col]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

        for decoder_col in [
            "decoder",
            "model",
        ]:
            if decoder_col in sub.columns:
                decoders.update(
                    sub[decoder_col]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

        if "seed" in sub.columns:
            for value in sub["seed"].dropna():
                try:
                    seeds.add(int(float(value)))
                except Exception:
                    pass

        columns = {
            str(c).lower()
            for c in sub.columns
        }

        has_rmse |= any(
            "rmse" in c
            for c in columns
        )
        has_mae |= any(
            re.search(r"(^|_)mae($|_)", c)
            for c in columns
        )
        has_r2 |= any(
            c == "r2"
            or c.startswith("r2_")
            or c.endswith("_r2")
            for c in columns
        )

        has_accuracy |= any(
            c == "accuracy"
            or "accuracy_" in c
            or c.endswith("_accuracy")
            for c in columns
        )
        has_auroc |= any(
            "auroc" in c
            or "roc_auc" in c
            for c in columns
        )
        has_ap |= any(
            c == "ap"
            or "average_precision" in c
            for c in columns
        )
        has_log_loss |= any(
            "log_loss" in c
            for c in columns
        )
        has_macro_f1 |= any(
            "macro_f1" in c
            for c in columns
        )
        has_weighted_f1 |= any(
            "weighted_f1" in c
            for c in columns
        )

        if "selected_residual_count" in sub.columns:
            vals = pd.to_numeric(
                sub["selected_residual_count"],
                errors="coerce",
            ).dropna()

            if len(vals):
                selected_residual_count = int(vals.max())

        if "fallback_exact_match" in sub.columns:
            vals = (
                sub["fallback_exact_match"]
                .dropna()
                .astype(str)
                .str.lower()
            )

            if len(vals):
                fallback_exact_match = any(
                    v in {"true", "1", "yes"}
                    for v in vals
                )

    has_four_seeds = {
        41, 42, 43, 44
    }.issubset(seeds)

    # --------------------------------------------------------------
    # 4. task type별 metric completeness
    # --------------------------------------------------------------

    if task_type == "regression":
        metric_complete = (
            has_rmse
            and has_mae
            and has_r2
        )

    elif task_type == "binary_classification":
        metric_complete = (
            has_auroc
            and has_ap
            and has_log_loss
        )

    elif task_type == "multiclass_classification":
        metric_complete = (
            has_accuracy
            and has_macro_f1
            and has_weighted_f1
        )

    else:
        metric_complete = False

    # --------------------------------------------------------------
    # 5. 완료 상태 판정
    # --------------------------------------------------------------

    if "SELECT" in gate_outcomes:
        completion_status = (
            "COMPLETE_EVALUATED_SELECT"
        )

    elif "FALLBACK" in gate_outcomes:
        completion_status = (
            "COMPLETE_FALLBACK"
        )

    elif "NOT_EVALUATED" in gate_outcomes:
        completion_status = (
            "COMPLETE_NOT_EVALUATED"
        )

    elif metric_complete and has_four_seeds:
        completion_status = (
            "EVALUATED_MISSING_GATE"
        )

    elif metric_complete and len(seeds) > 0:
        completion_status = (
            "PARTIAL_SEEDS_OR_BASELINE_ONLY"
        )

    elif matched_files:
        target_only = any(
            "target_only" in x.lower()
            or "efficiency_json" in x.lower()
            for x in matched_files
        )

        completion_status = (
            "TARGET_ONLY_ONLY"
            if target_only
            else "INCOMPLETE_ARTIFACTS"
        )

    else:
        completion_status = "NOT_STARTED"

    result_rows.append({
        **task_row.to_dict(),
        "completion_status": completion_status,
        "gate_outcomes": ";".join(
            sorted(gate_outcomes)
        ),
        "statuses_found": ";".join(
            sorted(statuses)
        ),
        "seeds_found": ";".join(
            map(str, sorted(seeds))
        ),
        "has_seed_41_44": has_four_seeds,
        "metric_complete": metric_complete,
        "has_rmse": has_rmse,
        "has_mae": has_mae,
        "has_r2": has_r2,
        "has_accuracy": has_accuracy,
        "has_auroc": has_auroc,
        "has_ap": has_ap,
        "has_log_loss": has_log_loss,
        "has_macro_f1": has_macro_f1,
        "has_weighted_f1": has_weighted_f1,
        "selected_residual_count": (
            selected_residual_count
        ),
        "fallback_exact_match": (
            fallback_exact_match
        ),
        "variants_found": ";".join(
            sorted(variants)
        ),
        "decoders_found": ";".join(
            sorted(decoders)
        ),
        "n_matched_files": len(matched_files),
        "gate_files": ";".join(
            sorted(gate_files)
        ),
        "fallback_files": ";".join(
            sorted(fallback_files)
        ),
        "status_files": ";".join(
            sorted(status_files)
        ),
    })

out = pd.DataFrame(result_rows).sort_values(
    [
        "benchmark_family",
        "task_type",
        "dataset",
        "task",
    ]
)

out.to_csv(OUT, index=False)

print("\n=== COMPLETION BY FAMILY ===")
print(
    out.groupby(
        [
            "benchmark_family",
            "completion_status",
        ]
    ).size().to_string()
)

print("\n=== FULL CHECKLIST ===")
print(
    out[
        [
            "benchmark_family",
            "dataset",
            "task",
            "task_type",
            "completion_status",
            "seeds_found",
            "gate_outcomes",
        ]
    ].to_string(index=False)
)

print("\nsaved:", OUT)
