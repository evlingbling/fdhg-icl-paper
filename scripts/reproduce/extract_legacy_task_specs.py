from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = Path.home() / "fdhg_icl"
LEGACY_ROOT_TOKEN = "<LEGACY_REPO>"


def portable_path(path: Path | str) -> str:
    """Convert a legacy-repository path into a portable provenance path."""
    path = Path(path)

    try:
        relative = path.relative_to(OLD_ROOT)
    except ValueError:
        return str(path)

    return f"{LEGACY_ROOT_TOKEN}/{relative.as_posix()}"
LEGACY_ROOT_TOKEN = "<LEGACY_REPO>"


def portable_path(path: Path | str) -> str:
    """Convert a legacy-repository path into a portable provenance path."""
    path = Path(path)

    try:
        relative = path.relative_to(OLD_ROOT)
    except ValueError:
        return str(path)

    return f"{LEGACY_ROOT_TOKEN}/{relative.as_posix()}"

INVENTORY_PATH = (
    ROOT / "configs/reproduction/task_inventory_with_artifacts.csv"
)
OUT_JSON = ROOT / "configs/reproduction/legacy_task_specs.json"
OUT_CSV = ROOT / "configs/reproduction/legacy_task_specs.csv"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {
            "_read_error": str(exc),
            "_path": portable_path(path),
        }


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        return pd.read_csv(path).to_dict(orient="records")
    except Exception as exc:
        return [{
            "_read_error": str(exc),
            "_path": portable_path(path),
        }]


def exact_artifact_dir(
    category: str,
    dataset: str,
    task: str,
) -> Path:
    return (
        OLD_ROOT
        / "outputs"
        / category
        / f"{dataset}_{task}_sample"
    )


def find_gate_tables(dataset: str, task: str) -> list[Path]:
    results_root = OLD_ROOT / "results"

    dataset_token = dataset.replace("rel-", "")
    task_tokens = {
        task.lower(),
        task.replace("-", "_").lower(),
    }

    matches: list[Path] = []

    for path in results_root.rglob("*.csv"):
        path_text = str(path).lower()

        if dataset_token not in path_text:
            continue

        if not any(token in path_text for token in task_tokens):
            continue

        if "gate" in path.name.lower() or "summary" in path.name.lower():
            matches.append(path)

    return sorted(set(matches))


def feature_names(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []

    candidate_columns = [
        "feature",
        "feature_name",
        "column",
        "name",
    ]

    for key in candidate_columns:
        values = [
            row.get(key)
            for row in records
            if row.get(key) is not None
        ]

        if values:
            return [str(value) for value in values]

    return []


def main() -> None:
    inventory = pd.read_csv(INVENTORY_PATH)

    specs: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for row in inventory.itertuples(index=False):
        dataset = str(row.dataset)
        task = str(row.task)
        task_key = str(row.task_key)

        dfs_dir = exact_artifact_dir("dfs_agg", dataset, task)
        fdhg_dir = exact_artifact_dir("fdhg_heuristic", dataset, task)
        naive_dir = exact_artifact_dir("naive_flatten", dataset, task)

        dfs_config_path = dfs_dir / "dfs_feature_config.json"
        dfs_manifest_path = dfs_dir / "dfs_feature_manifest.csv"

        ambiguity_manifest_path = (
            fdhg_dir / "ambiguity_feature_manifest.csv"
        )

        naive_config_path = naive_dir / "naive_feature_config.json"
        naive_manifest_path = naive_dir / "naive_feature_manifest.csv"

        dfs_config = read_json(dfs_config_path)
        dfs_manifest = read_csv_records(dfs_manifest_path)
        ambiguity_manifest = read_csv_records(
            ambiguity_manifest_path
        )
        naive_config = read_json(naive_config_path)
        naive_manifest = read_csv_records(
            naive_manifest_path
        )

        gate_tables = find_gate_tables(dataset, task)
        gate_payload = {
            portable_path(path): read_csv_records(path)
            for path in gate_tables
        }

        spec = {
            "dataset": dataset,
            "task": task,
            "gate_outcome": str(row.gate_outcome),
            "task_family": (
                None
                if pd.isna(row.task_family)
                else str(row.task_family)
            ),
            "primary_metric": str(row.primary_metric),
            "dfs_dir": portable_path(dfs_dir),
            "dfs_config_path": portable_path(dfs_config_path),
            "dfs_config": dfs_config,
            "dfs_manifest_path": portable_path(dfs_manifest_path),
            "dfs_manifest": dfs_manifest,
            "dfs_features": feature_names(dfs_manifest),
            "fdhg_dir": portable_path(fdhg_dir),
            "ambiguity_manifest_path": portable_path(
                ambiguity_manifest_path
            ),
            "ambiguity_manifest": ambiguity_manifest,
            "ambiguity_features": feature_names(
                ambiguity_manifest
            ),
            "naive_config_path": portable_path(naive_config_path),
            "naive_config": naive_config,
            "naive_manifest": naive_manifest,
            "gate_tables": gate_payload,
        }

        specs[task_key] = spec

        summary_rows.append({
            "task_key": task_key,
            "gate_outcome": row.gate_outcome,
            "has_dfs_config": dfs_config_path.exists(),
            "has_dfs_manifest": dfs_manifest_path.exists(),
            "n_dfs_features": len(spec["dfs_features"]),
            "has_ambiguity_manifest": (
                ambiguity_manifest_path.exists()
            ),
            "n_ambiguity_features": len(
                spec["ambiguity_features"]
            ),
            "n_gate_tables": len(gate_tables),
            "needs_manual_recovery": not (
                dfs_config_path.exists()
                and dfs_manifest_path.exists()
            ),
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    OUT_JSON.write_text(
        json.dumps(
            specs,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_CSV, index=False)

    print("=== Legacy task-spec extraction ===")
    print(summary.to_string(index=False))

    print()
    print("tasks:", len(summary))
    print(
        "automatic:",
        int((~summary["needs_manual_recovery"]).sum()),
    )
    print(
        "manual recovery needed:",
        int(summary["needs_manual_recovery"].sum()),
    )
    print("saved:", OUT_JSON)
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
