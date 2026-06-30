from __future__ import annotations

from collections.abc import Iterable
from math import isclose, isfinite
from typing import Literal


Direction = Literal["maximize", "minimize"]


def consistent_improvement_gate(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    direction: Direction,
    min_improvement: float = 0.0,
) -> bool:
    """
    Select the candidate only when it improves on every paired seed.

    Improvement is defined as:

    - maximize: candidate - baseline
    - minimize: baseline - candidate

    A tie is rejected when ``min_improvement == 0`` because the
    improvement must be strictly greater than the threshold.
    """
    pairs = list(zip(baseline, candidate, strict=True))

    if not pairs:
        raise ValueError("At least one seed is required.")

    if direction not in {"maximize", "minimize"}:
        raise ValueError(
            "direction must be either 'maximize' or 'minimize'."
        )

    if not isfinite(min_improvement):
        raise ValueError("min_improvement must be finite.")

    improvements: list[float] = []

    for baseline_value, candidate_value in pairs:
        baseline_float = float(baseline_value)
        candidate_float = float(candidate_value)

        if not isfinite(baseline_float):
            raise ValueError("Baseline scores must be finite.")

        if not isfinite(candidate_float):
            raise ValueError("Candidate scores must be finite.")

        if direction == "maximize":
            improvement = candidate_float - baseline_float
        else:
            improvement = baseline_float - candidate_float

        improvements.append(improvement)

    return all(
        improvement > min_improvement
        and not isclose(
            improvement,
            min_improvement,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for improvement in improvements
    )


def classification_gate(
    baseline: Iterable[float],
    candidate: Iterable[float],
) -> bool:
    """Select the candidate only if accuracy improves on every seed."""
    return consistent_improvement_gate(
        baseline,
        candidate,
        direction="maximize",
    )


def regression_gate(
    baseline_rmse: Iterable[float],
    candidate_rmse: Iterable[float],
) -> bool:
    """Select the candidate only if RMSE decreases on every seed."""
    return consistent_improvement_gate(
        baseline_rmse,
        candidate_rmse,
        direction="minimize",
    )
