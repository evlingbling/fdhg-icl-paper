from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from fdhg.gate import (
    consistent_improvement_gate,
    lexicographic_improvement_gate,
)


SEED_PATTERN = re.compile(r"seed(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select FDHG or exact DFS fallback using paired "
            "multi-seed validation metrics."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metric",
        required=True,
    )
    parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        required=True,
    )
    parser.add_argument(
        "--secondary-metric",
        default=None,
    )
    parser.add_argument(
        "--secondary-direction",
        choices=["maximize", "minimize"],
        default=None,
    )
    parser.add_argument(
        "--base-variant",
        default="dfs",
    )
    parser.add_argument(
        "--candidate-variant",
        default="fdhg_dmax1",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--required-seeds",
        nargs="*",
        type=int,
        default=None,
    )
    return parser.parse_args()


def discover_seed_metrics(
    variant_dir: Path,
) -> dict[int, Path]:
    metrics_by_seed: dict[int, Path] = {}

    for path in sorted(variant_dir.glob("seed*/metrics.csv")):
        match = SEED_PATTERN.fullmatch(path.parent.name)

        if match is None:
            continue

        seed = int(match.group(1))

        if seed in metrics_by_seed:
            raise ValueError(
                f"Duplicate metrics for seed {seed}: "
                f"{metrics_by_seed[seed]} and {path}"
            )

        metrics_by_seed[seed] = path

    return metrics_by_seed


def read_metric(
    path: Path,
    *,
    metric: str,
    expected_seed: int,
) -> tuple[float, dict]:
    frame = pd.read_csv(path)

    if len(frame) != 1:
        raise ValueError(
            f"Expected exactly one row in {path}, found {len(frame)}."
        )

    if metric not in frame.columns:
        raise ValueError(
            f"Metric {metric!r} not found in {path}. "
            f"Available columns: {list(frame.columns)}"
        )

    row = frame.iloc[0].to_dict()

    if "seed" in row and pd.notna(row["seed"]):
        row_seed = int(row["seed"])

        if row_seed != expected_seed:
            raise ValueError(
                f"Seed mismatch for {path}: directory says "
                f"{expected_seed}, metrics row says {row_seed}."
            )

    score = float(row[metric])

    if pd.isna(score):
        raise ValueError(
            f"Metric {metric!r} is NaN in {path}."
        )

    return score, row


def main() -> None:
    args = parse_args()

    result_root = args.result_root.resolve()
    base_dir = result_root / args.base_variant
    candidate_dir = result_root / args.candidate_variant

    base_paths = discover_seed_metrics(base_dir)
    candidate_paths = discover_seed_metrics(candidate_dir)

    base_seeds = set(base_paths)
    candidate_seeds = set(candidate_paths)

    if base_seeds != candidate_seeds:
        missing_candidate = sorted(base_seeds - candidate_seeds)
        missing_base = sorted(candidate_seeds - base_seeds)

        raise ValueError(
            "Base/candidate seed mismatch. "
            f"Missing candidate seeds: {missing_candidate}; "
            f"missing base seeds: {missing_base}."
        )

    if not base_seeds:
        raise ValueError(
            f"No paired seed metrics found under {result_root}."
        )

    seeds = sorted(base_seeds)

    if args.required_seeds is not None:
        required = sorted(set(args.required_seeds))

        if seeds != required:
            raise ValueError(
                f"Expected seeds {required}, found {seeds}."
            )

    if bool(args.secondary_metric) != bool(
        args.secondary_direction
    ):
        raise ValueError(
            "Provide both --secondary-metric and "
            "--secondary-direction, or neither."
        )

    base_scores: list[float] = []
    candidate_scores: list[float] = []
    secondary_base_scores: list[float] = []
    secondary_candidate_scores: list[float] = []
    per_seed_rows: list[dict] = []

    for seed in seeds:
        base_score, base_row = read_metric(
            base_paths[seed],
            metric=args.metric,
            expected_seed=seed,
        )
        candidate_score, candidate_row = read_metric(
            candidate_paths[seed],
            metric=args.metric,
            expected_seed=seed,
        )

        if args.direction == "maximize":
            improvement = candidate_score - base_score
        else:
            improvement = base_score - candidate_score

        base_scores.append(base_score)
        candidate_scores.append(candidate_score)

        secondary_base_score = None
        secondary_candidate_score = None

        if args.secondary_metric:
            secondary_base_score, _ = read_metric(
                base_paths[seed],
                metric=args.secondary_metric,
                expected_seed=seed,
            )
            secondary_candidate_score, _ = read_metric(
                candidate_paths[seed],
                metric=args.secondary_metric,
                expected_seed=seed,
            )
            secondary_base_scores.append(
                secondary_base_score
            )
            secondary_candidate_scores.append(
                secondary_candidate_score
            )

        per_seed_rows.append({
            "seed": seed,
            "metric": args.metric,
            "direction": args.direction,
            "base_variant": args.base_variant,
            "candidate_variant": args.candidate_variant,
            "base_score": base_score,
            "candidate_score": candidate_score,
            "improvement": improvement,
            "secondary_metric": args.secondary_metric,
            "secondary_direction": args.secondary_direction,
            "secondary_base_score": secondary_base_score,
            "secondary_candidate_score": (
                secondary_candidate_score
            ),
            "secondary_improvement": (
                (
                    secondary_candidate_score
                    - secondary_base_score
                )
                if (
                    args.secondary_metric
                    and args.secondary_direction == "maximize"
                )
                else (
                    secondary_base_score
                    - secondary_candidate_score
                )
                if args.secondary_metric
                else None
            ),
            "base_metrics_path": str(base_paths[seed]),
            "candidate_metrics_path": str(
                candidate_paths[seed]
            ),
            "base_n_features": base_row.get("n_features"),
            "candidate_n_features": candidate_row.get(
                "n_features"
            ),
        })

    if args.secondary_metric:
        selected = lexicographic_improvement_gate(
            base_scores,
            candidate_scores,
            secondary_base_scores,
            secondary_candidate_scores,
            primary_direction=args.direction,
            secondary_direction=args.secondary_direction,
        )
    else:
        selected = consistent_improvement_gate(
            base_scores,
            candidate_scores,
            direction=args.direction,
            min_improvement=args.min_improvement,
        )

    gate_outcome = "SELECT" if selected else "FALLBACK"
    selected_variant = (
        args.candidate_variant
        if selected
        else args.base_variant
    )

    for row in per_seed_rows:
        row["gate_outcome"] = gate_outcome
        row["selected_variant"] = selected_variant
        row["selected_score"] = (
            row["candidate_score"]
            if selected
            else row["base_score"]
        )

    decision = {
        "result_root": str(result_root),
        "base_variant": args.base_variant,
        "candidate_variant": args.candidate_variant,
        "metric": args.metric,
        "direction": args.direction,
        "secondary_metric": args.secondary_metric,
        "secondary_direction": args.secondary_direction,
        "gate_mode": (
            "lexicographic"
            if args.secondary_metric
            else "single_metric"
        ),
        "min_improvement": args.min_improvement,
        "seeds": seeds,
        "base_scores": base_scores,
        "candidate_scores": candidate_scores,
        "secondary_base_scores": (
            secondary_base_scores
            if args.secondary_metric
            else None
        ),
        "secondary_candidate_scores": (
            secondary_candidate_scores
            if args.secondary_metric
            else None
        ),
        "per_seed_improvements": [
            row["improvement"]
            for row in per_seed_rows
        ],
        "require_all_seeds": True,
        "gate_outcome": gate_outcome,
        "selected_variant": selected_variant,
        "fallback_exact_match": not selected,
    }

    decision_path = result_root / "gate_decision.json"
    metrics_path = result_root / "gated_metrics.csv"

    decision_path.write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(per_seed_rows).to_csv(
        metrics_path,
        index=False,
    )

    print("gate_outcome:", gate_outcome)
    print("selected_variant:", selected_variant)
    print("metric:", args.metric)
    print("direction:", args.direction)
    print("seeds:", seeds)
    print("improvements:", decision["per_seed_improvements"])
    print("decision:", decision_path)
    print("gated metrics:", metrics_path)


if __name__ == "__main__":
    main()
