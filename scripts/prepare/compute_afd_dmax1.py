from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_value(x: Any) -> str:
    """
    Make values hashable and stable for FD grouping.
    This handles lists/arrays from RelBench sequence-valued columns.
    """
    if x is None:
        return "__NULL__"

    try:
        if pd.isna(x):
            return "__NULL__"
    except Exception:
        pass

    if isinstance(x, (list, tuple, np.ndarray)):
        return "|".join(map(str, list(x)))

    return str(x)


def normalize_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_value).astype("string")


def entropy_from_counts(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def compute_pair_stats(df: pd.DataFrame, lhs_col: str, rhs_col: str) -> dict[str, float | int | str]:
    """
    Compute dmax=1 approximate FD scores for lhs_col -> rhs_col.

    Rows with null-like normalized values are excluded from the core FD score.
    """
    x = normalize_series(df[lhs_col])
    a = normalize_series(df[rhs_col])

    valid = (x != "__NULL__") & (a != "__NULL__")
    x = x[valid].reset_index(drop=True)
    a = a[valid].reset_index(drop=True)

    n_total = len(df)
    n = len(x)

    if n == 0:
        return {
            "lhs": lhs_col,
            "rhs": rhs_col,
            "n_total": n_total,
            "n_valid": 0,
            "coverage": 0.0,
            "r_del": np.nan,
            "r_ent": np.nan,
            "r_pair": np.nan,
            "r_tuple": np.nan,
            "lhs_uniqueness": np.nan,
            "lhs_collision": np.nan,
            "lhs_support_repeated": np.nan,
            "rhs_entropy": np.nan,
            "n_lhs_values": 0,
            "n_rhs_values": 0,
            "exact_fd": False,
            "heuristic_score": np.nan,
        }

    tmp = pd.DataFrame({"x": x, "a": a})

    # n_xa
    xa_counts = (
        tmp.groupby(["x", "a"], dropna=False)
        .size()
        .rename("n_xa")
        .reset_index()
    )

    # n_x, max_a n_xa, number of RHS values per X
    x_stats = (
        xa_counts.groupby("x", dropna=False)
        .agg(
            n_x=("n_xa", "sum"),
            max_n_xa=("n_xa", "max"),
            n_rhs_for_x=("a", "nunique"),
        )
        .reset_index()
    )

    # deletion / majority reliability
    majority_kept = float(x_stats["max_n_xa"].sum())
    r_del = majority_kept / n

    # tuple violation reliability
    violating_rows = float(x_stats.loc[x_stats["n_rhs_for_x"] > 1, "n_x"].sum())
    eps_tuple = violating_rows / n
    r_tuple = 1.0 - eps_tuple

    # pair violation reliability
    def comb2(v: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        return arr * (arr - 1.0) / 2.0

    lhs_pairs = float(comb2(x_stats["n_x"]).sum())
    same_rhs_pairs = float(comb2(xa_counts["n_xa"]).sum())
    all_pairs = n * (n - 1.0) / 2.0

    if all_pairs > 0:
        lhs_collision = lhs_pairs / all_pairs
    else:
        lhs_collision = 0.0

    if lhs_pairs > 0:
        eps_pair_conditional = (lhs_pairs - same_rhs_pairs) / lhs_pairs
        r_pair = 1.0 - eps_pair_conditional
    else:
        r_pair = np.nan

    # entropy score
    rhs_counts = tmp["a"].value_counts(dropna=False)
    h_a = entropy_from_counts(rhs_counts)

    h_a_given_x = 0.0
    for _, row in x_stats.iterrows():
        x_val = row["x"]
        n_x = float(row["n_x"])
        sub = xa_counts[xa_counts["x"] == x_val]["n_xa"]
        h_sub = entropy_from_counts(sub)
        h_a_given_x += (n_x / n) * h_sub

    if h_a > 1e-12:
        r_ent = 1.0 - (h_a_given_x / h_a)
    else:
        r_ent = 1.0

    n_lhs_values = int(x.nunique(dropna=False))
    n_rhs_values = int(a.nunique(dropna=False))

    lhs_uniqueness = n_lhs_values / n
    lhs_support_repeated = float(x_stats.loc[x_stats["n_x"] >= 2, "n_x"].sum()) / n
    coverage = n / n_total if n_total > 0 else 0.0

    exact_fd = bool(math.isclose(r_del, 1.0, rel_tol=0.0, abs_tol=1e-12))

    # MVP heuristic only for ranking inspection, not final FDHG selection yet.
    # Higher r_del/r_ent/support good; high uniqueness bad.
    heuristic_score = (
        1.2 * float(r_del)
        + 0.8 * float(r_ent)
        + 0.5 * float(lhs_support_repeated)
        + 0.3 * float(coverage)
        - 0.9 * float(lhs_uniqueness)
    )

    return {
        "lhs": lhs_col,
        "rhs": rhs_col,
        "n_total": int(n_total),
        "n_valid": int(n),
        "coverage": float(coverage),
        "r_del": float(r_del),
        "r_ent": float(r_ent),
        "r_pair": float(r_pair) if not pd.isna(r_pair) else np.nan,
        "r_tuple": float(r_tuple),
        "lhs_uniqueness": float(lhs_uniqueness),
        "lhs_collision": float(lhs_collision),
        "lhs_support_repeated": float(lhs_support_repeated),
        "rhs_entropy": float(h_a),
        "n_lhs_values": int(n_lhs_values),
        "n_rhs_values": int(n_rhs_values),
        "exact_fd": exact_fd,
        "heuristic_score": float(heuristic_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="Input parquet/csv table path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional subset of columns to evaluate",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=[],
        help="Columns to exclude from both LHS and RHS",
    )
    args = parser.parse_args()

    table_path = Path(args.table)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if table_path.suffix == ".parquet":
        df = pd.read_parquet(table_path)
    elif table_path.suffix == ".csv":
        df = pd.read_csv(table_path)
    else:
        raise ValueError("Only parquet/csv supported.")

    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).sort_index().reset_index(drop=True)

    if args.columns is not None and len(args.columns) > 0:
        missing = set(args.columns) - set(df.columns)
        if missing:
            raise ValueError(f"Requested columns missing from table: {sorted(missing)}")
        cols = list(args.columns)
        df = df[cols].copy()
    else:
        cols = list(df.columns)

    exclude = set(args.exclude_columns)
    cols = [c for c in cols if c not in exclude]

    print("=== AFD dmax=1 computation ===")
    print("table:", table_path)
    print("shape:", df.shape)
    print("columns:", cols)

    rows = []
    for lhs in cols:
        for rhs in cols:
            if lhs == rhs:
                continue
            print(f"computing {lhs} -> {rhs}")
            rows.append(compute_pair_stats(df, lhs, rhs))

    result = pd.DataFrame(rows)

    # Put strongest-looking rows first, but keep all rows.
    result = result.sort_values(
        ["heuristic_score", "r_del", "r_ent"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    result.to_csv(out_path, index=False)

    print("\n=== Done ===")
    print("saved:", out_path)
    print("\nTop 20 by heuristic_score:")
    display_cols = [
        "lhs", "rhs", "r_del", "r_ent", "lhs_uniqueness",
        "lhs_support_repeated", "coverage", "exact_fd", "heuristic_score"
    ]
    print(result[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
