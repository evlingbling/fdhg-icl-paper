from __future__ import annotations

from collections.abc import Iterable


def classification_gate(
    baseline: Iterable[float],
    candidate: Iterable[float],
) -> bool:
    """Select the candidate only if it improves on every seed."""
    pairs = list(zip(baseline, candidate, strict=True))
    if not pairs:
        raise ValueError("At least one seed is required.")
    return all(candidate_value > baseline_value
               for baseline_value, candidate_value in pairs)


def regression_gate(
    baseline_rmse: Iterable[float],
    candidate_rmse: Iterable[float],
) -> bool:
    """Select the candidate only if RMSE decreases on every seed."""
    pairs = list(zip(baseline_rmse, candidate_rmse, strict=True))
    if not pairs:
        raise ValueError("At least one seed is required.")
    return all(candidate_value < baseline_value
               for baseline_value, candidate_value in pairs)
