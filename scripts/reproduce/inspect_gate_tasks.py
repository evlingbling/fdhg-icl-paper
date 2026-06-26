from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "configs/reproduction/task_inventory.csv"
OUT_ROOT = ROOT / "outputs/gate_task_inspect"
REPORT = ROOT / "configs/reproduction/gate_task_schema_report.csv"

TASK_ALIASES = {
    (
        "rel-ratebeer",
        "beer_ratings-total_score_enriched",
    ): "beer_ratings-total_score",
}


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    inventory = pd.read_csv(INVENTORY)
    rows: list[dict[str, object]] = []

    for item in inventory.itertuples(index=False):
        dataset = str(item.dataset)
        task = str(item.task)
        inspect_task = TASK_ALIASES.get(
            (dataset, task),
            task,
        )

        task_key = f"{dataset}_{task}"
        inspect_key = f"{dataset}_{inspect_task}"

        parent = OUT_ROOT / task_key
        inspect_dir = parent / inspect_key

        print()
        print("=" * 80)
        print(f"{dataset}/{task}")
        print("=" * 80)

        if inspect_task != task:
            print(
                f"inspection alias: {task} -> {inspect_task}"
            )

        if not (inspect_dir / "target_train.parquet").exists():
            try:
                run([
                    "fdhg-inspect",
                    "--dataset", dataset,
                    "--task", inspect_task,
                    "--out-dir", str(parent),
                ])
            except subprocess.CalledProcessError as exc:
                rows.append({
                    "task_key": f"{dataset}/{task}",
                    "dataset": dataset,
                    "task": task,
                    "inspect_task": inspect_task,
                    "gate_outcome": item.gate_outcome,
                    "status": "INSPECT_FAILED",
                    "error": str(exc),
                    "train_rows": None,
                    "val_rows": None,
                    "target_train_columns": "",
                    "target_val_columns": "",
                    "table_schemas": "",
                })
                continue

        target_train = pd.read_parquet(
            inspect_dir / "target_train.parquet"
        )
        target_val = pd.read_parquet(
            inspect_dir / "target_val.parquet"
        )

        table_schemas: list[str] = []

        for table_path in sorted(inspect_dir.glob("table_*.parquet")):
            table_name = table_path.stem.removeprefix("table_")
            table = pd.read_parquet(table_path)

            table_schemas.append(
                f"{table_name}:"
                + ",".join(str(col) for col in table.columns)
            )

        rows.append({
            "task_key": f"{dataset}/{task}",
            "dataset": dataset,
            "task": task,
            "inspect_task": inspect_task,
            "gate_outcome": item.gate_outcome,
            "status": "OK",
            "error": "",
            "train_rows": len(target_train),
            "val_rows": len(target_val),
            "target_train_columns": ",".join(target_train.columns),
            "target_val_columns": ",".join(target_val.columns),
            "table_schemas": " | ".join(table_schemas),
        })

    report = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False)

    print()
    print("=== Gate task schema report ===")
    print(
        report[
            [
                "task_key",
                "gate_outcome",
                "train_rows",
                "val_rows",
                "target_train_columns",
            ]
        ].to_string(index=False)
    )
    print()
    print("saved:", REPORT)


if __name__ == "__main__":
    main()
