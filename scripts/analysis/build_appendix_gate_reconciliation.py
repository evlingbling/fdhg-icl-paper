#!/usr/bin/env python3
"""Build and verify the reconciled 51-task appendix gate artifacts.

This script:

1. Recovers canonical DFS/FDHG metrics for the 18 tasks with complete
   paired seed-41--44 artifacts.
2. Applies each task's paper gate metric.
3. Recomputes strict all-seed DFS-to-FDHG decisions.
4. Distinguishes stale archived labels from comparison-scope differences.
5. Produces the reconciled 51-task appendix table.
6. Verifies all expected counts and provenance invariants.

The script does not modify configs/benchmark_tasks.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "final_tables"

INVENTORY_PATH = ROOT / "configs" / "benchmark_tasks.csv"

SEED_METRICS_PATH = (
    RESULTS / "appendix_strict_gate_seed_metrics.csv"
)
METRIC_COVERAGE_PATH = (
    RESULTS / "appendix_strict_gate_metric_coverage.csv"
)
TASK_AUDIT_PATH = (
    RESULTS / "appendix_strict_dfs_fdhg_gate_audit.csv"
)
SEED_AUDIT_PATH = (
    RESULTS / "appendix_strict_dfs_fdhg_seed_audit.csv"
)
RECONCILIATION_PATH = (
    RESULTS / "appendix_strict_gate_reconciliation.csv"
)
FINAL_RECONCILED_PATH = (
    RESULTS / "appendix_51task_gate_reconciled.csv"
)

REQUIRED_SEEDS = (41, 42, 43, 44)
TOLERANCE = 1e-12


# These are the 18 tasks for which complete paired DFS/FDHG seed-level
# artifacts have been recovered.
AUDITED_TASKS: tuple[tuple[str, str], ...] = (
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "user-churn"),
    ("rel-arxiv", "paper-citation"),
    ("rel-f1", "driver-dnf"),
    ("rel-ratebeer", "beer-churn"),
    ("rel-ratebeer", "beer_ratings-total_score"),
    ("rel-ratebeer", "brewer-dormant"),
    ("rel-ratebeer", "user-churn"),
    ("rel-ratebeer", "user-count"),
    ("rel-salt", "item-incoterms"),
    ("rel-salt", "item-plant"),
    ("rel-salt", "item-shippoint"),
    ("rel-salt", "sales-group"),
    ("rel-salt", "sales-incoterms"),
    ("rel-salt", "sales-office"),
    ("rel-salt", "sales-payterms"),
    ("rel-salt", "sales-shipcond"),
    ("rel-stack", "user-badge"),
)


# Archived labels before strict reconciliation. These are retained only
# to document which historical decisions changed.
ARCHIVED_GATE: dict[tuple[str, str], str] = {
    ("rel-amazon", "item-churn"): "SELECT",
    ("rel-amazon", "user-churn"): "FALLBACK",
    ("rel-arxiv", "paper-citation"): "FALLBACK",
    ("rel-f1", "driver-dnf"): "FALLBACK",
    ("rel-ratebeer", "beer-churn"): "SELECT",
    ("rel-ratebeer", "beer_ratings-total_score"): "FALLBACK",
    ("rel-ratebeer", "brewer-dormant"): "FALLBACK",
    ("rel-ratebeer", "user-churn"): "FALLBACK",
    ("rel-ratebeer", "user-count"): "SELECT",
    ("rel-salt", "item-incoterms"): "FALLBACK",
    ("rel-salt", "item-plant"): "FALLBACK",
    ("rel-salt", "item-shippoint"): "SELECT",
    ("rel-salt", "sales-group"): "SELECT",
    ("rel-salt", "sales-incoterms"): "SELECT",
    ("rel-salt", "sales-office"): "SELECT",
    ("rel-salt", "sales-payterms"): "SELECT",
    ("rel-salt", "sales-shipcond"): "SELECT",
    ("rel-stack", "user-badge"): "FALLBACK",
}


# Gate metric used for each audited task.
PRIMARY_METRIC: dict[tuple[str, str], str] = {
    ("rel-amazon", "item-churn"): "roc_auc",
    ("rel-amazon", "user-churn"): "roc_auc",
    ("rel-arxiv", "paper-citation"): "log_loss",
    ("rel-f1", "driver-dnf"): "roc_auc",
    ("rel-ratebeer", "beer-churn"): "log_loss",
    ("rel-ratebeer", "beer_ratings-total_score"): "rmse",
    ("rel-ratebeer", "brewer-dormant"): "log_loss",
    ("rel-ratebeer", "user-churn"): "log_loss",
    ("rel-ratebeer", "user-count"): "rmse",
    ("rel-salt", "item-incoterms"): "accuracy",
    ("rel-salt", "item-plant"): "accuracy",
    ("rel-salt", "item-shippoint"): "accuracy",
    ("rel-salt", "sales-group"): "accuracy",
    ("rel-salt", "sales-incoterms"): "accuracy",
    ("rel-salt", "sales-office"): "accuracy",
    ("rel-salt", "sales-payterms"): "accuracy",
    ("rel-salt", "sales-shipcond"): "accuracy_then_mrr",
    ("rel-stack", "user-badge"): "roc_auc",
}


DIRECTION: dict[str, str] = {
    "accuracy": "maximize",
    "macro_f1": "maximize",
    "weighted_f1": "maximize",
    "roc_auc": "maximize",
    "average_precision": "maximize",
    "mrr": "maximize",
    "r2": "maximize",
    "log_loss": "minimize",
    "rmse": "minimize",
    "mae": "minimize",
}


SOURCE_PRIORITY: dict[str, int] = {
    "task_root": 0,
    "reproduced": 1,
    "legacy": 2,
    "paper_main": 3,
    "final_all": 4,
}


METRIC_COLUMNS = (
    "accuracy",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "roc_auc",
    "average_precision",
    "log_loss",
    "mrr",
    "rmse",
    "mae",
    "r2",
)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV while safely ignoring missing or empty optional inputs."""
    if not path.exists():
        print(f"[skip missing] {path.relative_to(ROOT)}")
        return None

    if path.stat().st_size == 0:
        print(f"[skip empty] {path.relative_to(ROOT)}")
        return None

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[skip no columns] {path.relative_to(ROOT)}")
        return None
    except Exception as exc:
        print(
            f"[skip unreadable] {path.relative_to(ROOT)}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def normalize_variant(value: Any) -> str | None:
    """Map known artifact variant names to canonical DFS/FDHG labels."""
    text = str(value).strip().lower()

    if text in {
        "dfs",
        "temporal_dfs_clean",
    }:
        return "dfs"

    if text in {
        "fdhg",
        "fdhg_dmax1",
        "fdhg_dmax1_full",
        "fdhg_full",
        "fdhg_residual",
        "fdhg_v2",
    }:
        return "fdhg"

    return None


def add_frame_records(
    records: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    source_path: Path,
    source_kind: str,
) -> None:
    """Append usable seed-level rows from one source frame."""
    if frame.empty:
        return

    required = {"dataset", "task", "seed"}
    if not required.issubset(frame.columns):
        return

    variant_col = next(
        (
            column
            for column in (
                "variant",
                "artifact_variant",
                "method",
            )
            if column in frame.columns
        ),
        None,
    )

    if variant_col is None:
        return

    audited_keys = set(AUDITED_TASKS)

    for _, row in frame.iterrows():
        dataset = str(row.get("dataset"))
        task = str(row.get("task"))
        key = (dataset, task)

        if key not in audited_keys:
            continue

        variant = normalize_variant(row.get(variant_col))
        if variant is None:
            continue

        try:
            seed = int(float(row.get("seed")))
        except (TypeError, ValueError):
            continue

        if seed not in REQUIRED_SEEDS:
            continue

        record: dict[str, Any] = {
            "dataset": dataset,
            "task": task,
            "variant": variant,
            "seed": seed,
            "source_kind": source_kind,
            "source_priority": SOURCE_PRIORITY[source_kind],
            "source_path": str(source_path.relative_to(ROOT)),
        }

        for metric in METRIC_COLUMNS:
            record[metric] = pd.to_numeric(
                row.get(metric, np.nan),
                errors="coerce",
            )

        records.append(record)


def build_canonical_seed_metrics() -> pd.DataFrame:
    """Recover one canonical DFS and FDHG row per task and seed."""
    records: list[dict[str, Any]] = []

    direct_patterns = (
        "rel-*_*_*/dfs/seed*/metrics.csv",
        "rel-*_*_*/fdhg_dmax1/seed*/metrics.csv",
    )

    results_root = ROOT / "results"

    for pattern in direct_patterns:
        for path in sorted(results_root.glob(pattern)):
            frame = safe_read_csv(path)
            if frame is None:
                continue

            add_frame_records(
                records,
                frame,
                source_path=path,
                source_kind="task_root",
            )

    aggregate_sources = (
        (
            RESULTS / "reproduced_validation_gate_seed_metrics.csv",
            "reproduced",
        ),
        (
            RESULTS
            / "legacy_validation_gate_seed_metrics_normalized.csv",
            "legacy",
        ),
        (
            RESULTS / "paper_main_runs.csv",
            "paper_main",
        ),
        (
            RESULTS / "final_all_runs.csv",
            "final_all",
        ),
    )

    for path, source_kind in aggregate_sources:
        frame = safe_read_csv(path)
        if frame is None:
            continue

        add_frame_records(
            records,
            frame,
            source_path=path,
            source_kind=source_kind,
        )

    raw = pd.DataFrame(records)

    if raw.empty:
        raise RuntimeError(
            "No usable DFS/FDHG seed records were recovered."
        )

    raw = raw.sort_values(
        [
            "dataset",
            "task",
            "variant",
            "seed",
            "source_priority",
        ]
    )

    canonical = raw.drop_duplicates(
        subset=[
            "dataset",
            "task",
            "variant",
            "seed",
        ],
        keep="first",
    ).reset_index(drop=True)

    expected_rows = (
        len(AUDITED_TASKS)
        * 2
        * len(REQUIRED_SEEDS)
    )

    if len(canonical) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} canonical rows, "
            f"found {len(canonical)}."
        )

    group_counts = canonical.groupby(
        ["dataset", "task", "variant"]
    )["seed"].nunique()

    if not (group_counts == len(REQUIRED_SEEDS)).all():
        raise AssertionError(
            "At least one task/variant lacks four canonical seeds."
        )

    canonical.to_csv(SEED_METRICS_PATH, index=False)
    return canonical


def build_metric_coverage(
    canonical: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired seed and metric coverage for the 18 tasks."""
    task_type_lookup = {
        (str(row.dataset), str(row.task)): str(row.task_type)
        for row in inventory.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []

    for dataset, task in AUDITED_TASKS:
        subset = canonical[
            canonical["dataset"].eq(dataset)
            & canonical["task"].eq(task)
        ]

        dfs_seeds = sorted(
            subset.loc[
                subset["variant"].eq("dfs"),
                "seed",
            ].unique()
        )
        fdhg_seeds = sorted(
            subset.loc[
                subset["variant"].eq("fdhg"),
                "seed",
            ].unique()
        )
        paired_seeds = sorted(
            set(dfs_seeds) & set(fdhg_seeds)
        )

        available_metrics: list[str] = []

        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "roc_auc",
            "average_precision",
            "log_loss",
            "mrr",
            "rmse",
            "mae",
            "r2",
        ):
            pivot = subset.pivot_table(
                index="seed",
                columns="variant",
                values=metric,
                aggfunc="first",
            )

            if not {"dfs", "fdhg"}.issubset(pivot.columns):
                continue

            complete = pivot.loc[
                pivot.index.isin(REQUIRED_SEEDS),
                ["dfs", "fdhg"],
            ].dropna()

            if set(complete.index) == set(REQUIRED_SEEDS):
                available_metrics.append(metric)

        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "task_type": task_type_lookup[(dataset, task)],
                "legacy_gate": ARCHIVED_GATE[(dataset, task)],
                "dfs_seeds": ",".join(map(str, dfs_seeds)),
                "fdhg_seeds": ",".join(map(str, fdhg_seeds)),
                "paired_seeds": ",".join(map(str, paired_seeds)),
                "canonical_strict_ready": (
                    paired_seeds == list(REQUIRED_SEEDS)
                ),
                "available_paired_metrics": ",".join(
                    available_metrics
                ),
            }
        )

    coverage = pd.DataFrame(rows)
    coverage.to_csv(METRIC_COVERAGE_PATH, index=False)

    if not coverage["canonical_strict_ready"].all():
        raise AssertionError(
            "At least one audited task lacks complete paired seeds."
        )

    return coverage


def metric_improvement(
    *,
    base: float,
    candidate: float,
    metric: str,
) -> float:
    """Return positive values when the candidate improves."""
    if DIRECTION[metric] == "maximize":
        return candidate - base

    return base - candidate


def build_strict_gate_audit(
    canonical: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute strict all-seed DFS-to-FDHG gate outcomes."""
    task_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for coverage_row in coverage.itertuples(index=False):
        dataset = str(coverage_row.dataset)
        task = str(coverage_row.task)
        key = (dataset, task)

        metric_spec = PRIMARY_METRIC[key]

        if metric_spec == "accuracy_then_mrr":
            gate_mode = "lexicographic"
            primary_metric = "accuracy"
            secondary_metric: str | None = "mrr"
        else:
            gate_mode = "single_metric"
            primary_metric = metric_spec
            secondary_metric = None

        subset = canonical[
            canonical["dataset"].eq(dataset)
            & canonical["task"].eq(task)
        ]

        dfs = (
            subset[subset["variant"].eq("dfs")]
            .set_index("seed")
            .sort_index()
        )
        fdhg = (
            subset[subset["variant"].eq("fdhg")]
            .set_index("seed")
            .sort_index()
        )

        primary_improvements: list[float] = []
        secondary_improvements: list[float] = []
        seed_passes: list[bool] = []

        for seed in REQUIRED_SEEDS:
            base_primary = float(
                dfs.loc[seed, primary_metric]
            )
            candidate_primary = float(
                fdhg.loc[seed, primary_metric]
            )

            if not (
                np.isfinite(base_primary)
                and np.isfinite(candidate_primary)
            ):
                raise RuntimeError(
                    f"Missing {primary_metric} for "
                    f"{dataset}/{task}/seed{seed}."
                )

            primary_gain = metric_improvement(
                base=base_primary,
                candidate=candidate_primary,
                metric=primary_metric,
            )

            secondary_gain = np.nan

            if secondary_metric is None:
                seed_pass = primary_gain > TOLERANCE
            else:
                base_secondary = float(
                    dfs.loc[seed, secondary_metric]
                )
                candidate_secondary = float(
                    fdhg.loc[seed, secondary_metric]
                )

                if not (
                    np.isfinite(base_secondary)
                    and np.isfinite(candidate_secondary)
                ):
                    raise RuntimeError(
                        f"Missing {secondary_metric} for "
                        f"{dataset}/{task}/seed{seed}."
                    )

                secondary_gain = metric_improvement(
                    base=base_secondary,
                    candidate=candidate_secondary,
                    metric=secondary_metric,
                )

                seed_pass = bool(
                    primary_gain > TOLERANCE
                    or (
                        abs(primary_gain) <= TOLERANCE
                        and secondary_gain > TOLERANCE
                    )
                )

            primary_improvements.append(primary_gain)
            secondary_improvements.append(secondary_gain)
            seed_passes.append(bool(seed_pass))

            seed_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "task_type": coverage_row.task_type,
                    "comparison_scope": "dfs_vs_fdhg",
                    "gate_mode": gate_mode,
                    "primary_metric": primary_metric,
                    "secondary_metric": secondary_metric,
                    "seed": seed,
                    "base_primary_score": base_primary,
                    "candidate_primary_score": candidate_primary,
                    "primary_improvement": primary_gain,
                    "secondary_improvement": secondary_gain,
                    "seed_pass": bool(seed_pass),
                    "base_source": dfs.loc[seed, "source_path"],
                    "candidate_source": fdhg.loc[
                        seed,
                        "source_path",
                    ],
                }
            )

        all_seeds_pass = all(seed_passes)
        strict_gate = (
            "SELECT" if all_seeds_pass else "FALLBACK"
        )

        if all_seeds_pass:
            gate_reason = "all_seeds_strictly_pass"
        elif gate_mode == "lexicographic":
            gate_reason = (
                "lexicographic_seed_inconsistency_fallback"
            )
        else:
            gate_reason = (
                "primary_metric_seed_inconsistency_fallback"
            )

        task_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "task_type": coverage_row.task_type,
                "legacy_gate": ARCHIVED_GATE[key],
                "comparison_scope": "dfs_vs_fdhg",
                "gate_mode": gate_mode,
                "primary_metric": primary_metric,
                "secondary_metric": secondary_metric,
                "strict_gate": strict_gate,
                "gate_matches_legacy": (
                    strict_gate == ARCHIVED_GATE[key]
                ),
                "all_seeds_pass": all_seeds_pass,
                "missing_seeds": "",
                "mean_primary_improvement": float(
                    np.mean(primary_improvements)
                ),
                "primary_improvements": "|".join(
                    f"{gain:.12g}"
                    for gain in primary_improvements
                ),
                "secondary_improvements": (
                    "|".join(
                        f"{gain:.12g}"
                        for gain in secondary_improvements
                    )
                    if secondary_metric is not None
                    else ""
                ),
                "gate_reason": gate_reason,
            }
        )

    task_audit = pd.DataFrame(task_rows).sort_values(
        ["dataset", "task"]
    ).reset_index(drop=True)

    seed_audit = pd.DataFrame(seed_rows).sort_values(
        ["dataset", "task", "seed"]
    ).reset_index(drop=True)

    task_audit.to_csv(TASK_AUDIT_PATH, index=False)
    seed_audit.to_csv(SEED_AUDIT_PATH, index=False)

    return task_audit, seed_audit


def build_reconciliation(
    task_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Classify archived-vs-strict differences and recommend final gates."""
    special_cases: dict[
        tuple[str, str],
        dict[str, str],
    ] = {
        ("rel-f1", "driver-dnf"): {
            "mismatch_class": "comparison_scope_mismatch",
            "recommended_appendix_gate": "FALLBACK",
            "recommended_action": (
                "Keep fallback because the archived decision uses "
                "a stronger temporal baseline than DFS alone."
            ),
        },
        ("rel-stack", "user-badge"): {
            "mismatch_class": "stale_legacy_gate",
            "recommended_appendix_gate": "SELECT",
            "recommended_action": (
                "Update to SELECT under strict all-seed "
                "DFS-to-FDHG AUROC gating."
            ),
        },
        ("rel-salt", "sales-office"): {
            "mismatch_class": "stale_legacy_gate",
            "recommended_appendix_gate": "FALLBACK",
            "recommended_action": (
                "Update to FALLBACK because tied seeds do not "
                "satisfy strict improvement."
            ),
        },
        ("rel-salt", "sales-payterms"): {
            "mismatch_class": "stale_legacy_gate",
            "recommended_appendix_gate": "FALLBACK",
            "recommended_action": (
                "Update to FALLBACK because tied seeds do not "
                "satisfy strict improvement."
            ),
        },
        ("rel-salt", "sales-shipcond"): {
            "mismatch_class": "stale_legacy_gate",
            "recommended_appendix_gate": "FALLBACK",
            "recommended_action": (
                "Update to FALLBACK because seed 42 regresses "
                "on primary accuracy."
            ),
        },
    }

    rows: list[dict[str, Any]] = []

    for row in task_audit.itertuples(index=False):
        key = (str(row.dataset), str(row.task))
        info = special_cases.get(key)

        if info is None:
            mismatch_class = "agreement"
            recommended_gate = str(row.strict_gate)
            recommended_action = "No change."
        else:
            mismatch_class = info["mismatch_class"]
            recommended_gate = info[
                "recommended_appendix_gate"
            ]
            recommended_action = info[
                "recommended_action"
            ]

        rows.append(
            {
                "dataset": row.dataset,
                "task": row.task,
                "legacy_gate": row.legacy_gate,
                "strict_dfs_fdhg_gate": row.strict_gate,
                "mismatch_class": mismatch_class,
                "recommended_appendix_gate": recommended_gate,
                "primary_metric": row.primary_metric,
                "secondary_metric": row.secondary_metric,
                "gate_reason": row.gate_reason,
                "recommended_action": recommended_action,
            }
        )

    reconciliation = pd.DataFrame(rows).sort_values(
        ["dataset", "task"]
    ).reset_index(drop=True)

    reconciliation.to_csv(
        RECONCILIATION_PATH,
        index=False,
    )

    return reconciliation


def build_final_reconciled_table(
    inventory: pd.DataFrame,
    task_audit: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the 18-task audit and reconciliation into the 51-task inventory."""
    audit_columns = [
        "dataset",
        "task",
        "comparison_scope",
        "gate_mode",
        "primary_metric",
        "secondary_metric",
        "strict_gate",
        "mean_primary_improvement",
        "primary_improvements",
        "secondary_improvements",
        "gate_reason",
    ]

    reconciliation_columns = [
        "dataset",
        "task",
        "mismatch_class",
        "recommended_appendix_gate",
        "recommended_action",
    ]

    output = inventory.merge(
        task_audit[audit_columns],
        on=["dataset", "task"],
        how="left",
    )

    output = output.merge(
        reconciliation[reconciliation_columns],
        on=["dataset", "task"],
        how="left",
    )

    output["strict_seed_audit_available"] = (
        output["strict_gate"].notna()
    )

    output["appendix_gate_provenance"] = (
        "legacy_inventory"
    )

    output.loc[
        output["strict_seed_audit_available"],
        "appendix_gate_provenance",
    ] = "strict_seed_audit"

    output.loc[
        output["mismatch_class"].eq(
            "comparison_scope_mismatch"
        ),
        "appendix_gate_provenance",
    ] = "stronger_baseline_retained"

    output.to_csv(FINAL_RECONCILED_PATH, index=False)
    return output


def verify_outputs(
    inventory: pd.DataFrame,
    canonical: pd.DataFrame,
    coverage: pd.DataFrame,
    task_audit: pd.DataFrame,
    seed_audit: pd.DataFrame,
    reconciliation: pd.DataFrame,
    final_table: pd.DataFrame,
) -> None:
    """Verify all paper-facing reconciliation invariants."""
    if len(inventory) != 51:
        raise AssertionError(
            f"Expected 51 inventory rows, found {len(inventory)}."
        )

    if (
        inventory[["dataset", "task"]]
        .drop_duplicates()
        .shape[0]
        != 51
    ):
        raise AssertionError(
            "Inventory dataset/task keys are not unique."
        )

    inventory_counts = (
        inventory["gate_outcomes"]
        .value_counts()
        .to_dict()
    )

    expected_inventory_counts = {
        "FALLBACK": 26,
        "SELECT": 24,
        "NOT_EVALUATED": 1,
    }

    if inventory_counts != expected_inventory_counts:
        raise AssertionError(
            "Unexpected final gate counts: "
            f"{inventory_counts}"
        )

    if len(canonical) != 144:
        raise AssertionError(
            f"Expected 144 seed rows, found {len(canonical)}."
        )

    if len(coverage) != 18:
        raise AssertionError(
            f"Expected 18 coverage rows, found {len(coverage)}."
        )

    if len(task_audit) != 18:
        raise AssertionError(
            f"Expected 18 task audit rows, found {len(task_audit)}."
        )

    if len(seed_audit) != 72:
        raise AssertionError(
            f"Expected 72 seed audit rows, found {len(seed_audit)}."
        )

    raw_strict_counts = (
        task_audit["strict_gate"]
        .value_counts()
        .to_dict()
    )

    if raw_strict_counts != {
        "FALLBACK": 10,
        "SELECT": 8,
    }:
        raise AssertionError(
            f"Unexpected raw strict counts: {raw_strict_counts}"
        )

    recommended_counts = (
        reconciliation["recommended_appendix_gate"]
        .value_counts()
        .to_dict()
    )

    if recommended_counts != {
        "FALLBACK": 11,
        "SELECT": 7,
    }:
        raise AssertionError(
            "Unexpected recommended audited-subset counts: "
            f"{recommended_counts}"
        )

    mismatch_counts = (
        reconciliation["mismatch_class"]
        .value_counts()
        .to_dict()
    )

    if mismatch_counts != {
        "agreement": 13,
        "stale_legacy_gate": 4,
        "comparison_scope_mismatch": 1,
    }:
        raise AssertionError(
            f"Unexpected mismatch classes: {mismatch_counts}"
        )

    if len(final_table) != 51:
        raise AssertionError(
            f"Expected 51 final rows, found {len(final_table)}."
        )

    coverage_counts = (
        final_table["strict_seed_audit_available"]
        .value_counts()
        .to_dict()
    )

    if coverage_counts != {
        False: 33,
        True: 18,
    }:
        raise AssertionError(
            f"Unexpected audit coverage: {coverage_counts}"
        )

    provenance_counts = (
        final_table["appendix_gate_provenance"]
        .value_counts()
        .to_dict()
    )

    if provenance_counts != {
        "legacy_inventory": 33,
        "strict_seed_audit": 17,
        "stronger_baseline_retained": 1,
    }:
        raise AssertionError(
            f"Unexpected provenance counts: {provenance_counts}"
        )

    driver = final_table[
        final_table["dataset"].eq("rel-f1")
        & final_table["task"].eq("driver-dnf")
    ]

    if len(driver) != 1:
        raise AssertionError(
            "Expected exactly one driver-dnf row."
        )

    driver_row = driver.iloc[0]

    if driver_row["gate_outcomes"] != "FALLBACK":
        raise AssertionError(
            "Driver-DNF must retain final FALLBACK status."
        )

    if (
        driver_row["appendix_gate_provenance"]
        != "stronger_baseline_retained"
    ):
        raise AssertionError(
            "Driver-DNF provenance is incorrect."
        )

    expected_final_labels = {
        ("rel-stack", "user-badge"): "SELECT",
        ("rel-salt", "sales-office"): "FALLBACK",
        ("rel-salt", "sales-payterms"): "FALLBACK",
        ("rel-salt", "sales-shipcond"): "FALLBACK",
    }

    for (dataset, task), expected_gate in (
        expected_final_labels.items()
    ):
        row = final_table[
            final_table["dataset"].eq(dataset)
            & final_table["task"].eq(task)
        ]

        if len(row) != 1:
            raise AssertionError(
                f"Expected one row for {dataset}/{task}."
            )

        actual_gate = str(row.iloc[0]["gate_outcomes"])

        if actual_gate != expected_gate:
            raise AssertionError(
                f"{dataset}/{task}: expected {expected_gate}, "
                f"found {actual_gate}."
            )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(INVENTORY_PATH)

    canonical = build_canonical_seed_metrics()
    coverage = build_metric_coverage(
        canonical,
        inventory,
    )
    task_audit, seed_audit = build_strict_gate_audit(
        canonical,
        coverage,
    )
    reconciliation = build_reconciliation(task_audit)
    final_table = build_final_reconciled_table(
        inventory,
        task_audit,
        reconciliation,
    )

    verify_outputs(
        inventory,
        canonical,
        coverage,
        task_audit,
        seed_audit,
        reconciliation,
        final_table,
    )

    print("[OK] Appendix 51-task reconciliation regenerated.")
    print(
        "[OK] Gate count: "
        "24 SELECT / 26 FALLBACK / 1 NOT_EVALUATED."
    )
    print(
        "[OK] Coverage: "
        "18 strict-audited / 33 legacy-only."
    )
    print(
        "[OK] Provenance: "
        "17 strict / 1 stronger baseline / 33 legacy."
    )

    print("\nGenerated artifacts:")
    for path in (
        SEED_METRICS_PATH,
        METRIC_COVERAGE_PATH,
        TASK_AUDIT_PATH,
        SEED_AUDIT_PATH,
        RECONCILIATION_PATH,
        FINAL_RECONCILED_PATH,
    ):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
