from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/reproduction/tasks.yaml"


def run(command: list[str]) -> None:
    print()
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def load_config(dataset: str, task: str) -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)

    key = f"{dataset}/{task}"

    try:
        return config["tasks"][key]
    except KeyError as exc:
        available = "\n".join(
            f"  - {name}" for name in sorted(config.get("tasks", {}))
        )
        raise SystemExit(
            f"No reproduction configuration for {key!r}.\n"
            f"Available tasks:\n{available}"
        ) from exc


def add_entity_alias(
    inspect_dir: Path,
    *,
    entity_table: str,
    source_key: str,
    target_key: str,
) -> None:
    path = inspect_dir / f"table_{entity_table}.parquet"
    df = pd.read_parquet(path)

    if target_key not in df.columns:
        if source_key not in df.columns:
            raise KeyError(
                f"Neither {target_key!r} nor {source_key!r} exists in {path}"
            )
        df[target_key] = df[source_key]

    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end FDHG reproduction task."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build DFS/FDHG artifacts without model evaluation.",
    )
    args = parser.parse_args()

    cfg = load_config(args.dataset, args.task)

    task_key = f"{args.dataset}_{args.task}"
    work_root = ROOT / "outputs" / "e2e" / task_key

    inspect_parent = work_root / "inspect"
    raw_inspect = inspect_parent / task_key
    enriched_inspect = work_root / "inspect_enriched"
    fdhg_inspect = work_root / "inspect_fdhg"
    dfs_dir = work_root / "dfs"
    afd_dir = work_root / "afd"
    fdhg_dir = work_root / "fdhg"

    if args.force and work_root.exists():
        shutil.rmtree(work_root)

    # 1. RelBench inspection
    if not (raw_inspect / "target_train.parquet").exists():
        run([
            "fdhg-inspect",
            "--dataset", args.dataset,
            "--task", args.task,
            "--out-dir", str(inspect_parent),
        ])

    target_cfg = cfg.get("target", {})

    # 2. Optional row-key enrichment
    if target_cfg.get("primary_key_col"):
        if not (enriched_inspect / "target_train.parquet").exists():
            command = [
                sys.executable,
                "scripts/prepare/enrich_target_from_primary_key.py",
                "--inspect-dir", str(raw_inspect),
                "--out-dir", str(enriched_inspect),
                "--source-table", target_cfg["source_table"],
                "--primary-key-col", target_cfg["primary_key_col"],
                "--entity-key", target_cfg["entity_key"],
                "--source-entity-col", target_cfg["source_entity_col"],
                "--verify-time-col", target_cfg["time_col"],
                "--splits", "train", "val",
            ]

            verify_label = target_cfg.get("verify_label_col")
            if verify_label:
                command.extend(["--verify-label-col", verify_label])

            run(command)
    else:
        enriched_inspect = raw_inspect

    dfs_cfg = cfg["dfs"]

    # 3. DFS
    if not (dfs_dir / "target_with_dfs_agg_train.parquet").exists():
        run([
            sys.executable,
            "scripts/prepare/build_relbench_features.py",
            "--inspect-dir", str(enriched_inspect),
            "--out-dir", str(dfs_dir),
            "--mode", "dfs",
            "--max-train", str(dfs_cfg["max_train"]),
            "--max-val", str(dfs_cfg["max_val"]),
            "--seed", str(args.seed),
            "--target-key", target_cfg["entity_key"],
            "--target-time-col", target_cfg["time_col"],
            "--child-table", dfs_cfg["child_table"],
            "--numeric-col", dfs_cfg["numeric_col"],
            "--splits", "train", "val",
        ])

    # 4. Optional AFD/FDHG block
    afd_cfg = cfg.get("afd")

    if afd_cfg:
        if not fdhg_inspect.exists():
            shutil.copytree(enriched_inspect, fdhg_inspect)

        add_entity_alias(
            fdhg_inspect,
            entity_table=afd_cfg["entity_table"],
            source_key=afd_cfg["entity_table_key"],
            target_key=target_cfg["entity_key"],
        )

        afd_dir.mkdir(parents=True, exist_ok=True)
        afd_path = afd_dir / "afd_dmax1.csv"

        if not afd_path.exists():
            run([
                sys.executable,
                "scripts/prepare/compute_afd_dmax1.py",
                "--table",
                str(
                    fdhg_inspect
                    / f"table_{afd_cfg['entity_table']}.parquet"
                ),
                "--out", str(afd_path),
                "--seed", str(args.seed),
                "--columns", *afd_cfg["columns"],
                "--exclude-columns",
                target_cfg["entity_key"],
                afd_cfg["entity_table_key"],
            ])

        if not (fdhg_dir / "target_with_dfs_agg_train.parquet").exists():
            run([
                sys.executable,
                "scripts/prepare/build_fdhg_ambiguity.py",
                "--inspect-dir", str(fdhg_inspect),
                "--dfs-dir", str(dfs_dir),
                "--afd-stats", str(afd_path),
                "--out-dir", str(fdhg_dir),
                "--entity-key", target_cfg["entity_key"],
                "--target-table", afd_cfg["entity_table"],
            ])
    else:
        # No usable FDHG edge: exact DFS fallback.
        fdhg_dir = dfs_dir

    if args.prepare_only:
        print(f"\nPrepared task artifacts under: {work_root}")
        return

    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    evaluator = {
        "binary": "scripts/evaluate/evaluate_binary_tabpfn.py",
        "regression": "scripts/evaluate/evaluate_regression_tabpfn.py",
    }[cfg["problem_type"]]

    drop_cols = ",".join(cfg["evaluation"]["drop_cols"])
    result_root = ROOT / "results" / f"{task_key}_tabpfn"

    for variant, feature_root in [
        ("dfs", dfs_dir),
        ("fdhg_dmax1", fdhg_dir),
    ]:
        out_dir = result_root / variant / f"seed{args.seed}"

        run([
            sys.executable,
            evaluator,
            "--train-parquet",
            str(feature_root / "target_with_dfs_agg_train.parquet"),
            "--val-parquet",
            str(feature_root / "target_with_dfs_agg_val.parquet"),
            "--output-dir", str(out_dir),
            "--dataset", args.dataset,
            "--task", args.task,
            "--variant", variant,
            "--label-col", cfg["label_col"],
            "--drop-cols", drop_cols,
            "--device", args.device,
            "--seed", str(args.seed),
        ])

    print(f"\nCompleted end-to-end reproduction: {args.dataset}/{args.task}")


if __name__ == "__main__":
    main()
