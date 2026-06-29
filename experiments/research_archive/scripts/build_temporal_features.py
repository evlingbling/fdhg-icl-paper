#!/usr/bin/env python3
"""
Leakage-safe temporal feature builder for FDHG-ICL.

This script/function builds temporal features such as:
- all_past_count
- all_past_mean
- all_past_std
- last_value
- time_since_last_days
- recent_30d_count / recent_30d_mean
- recent_90d_count / recent_90d_mean
- trend_last_minus_mean
- trend_recent30_minus_all

Core leakage rule:
    child_time <= target_time

This file can be imported by the generic RelBench feature builder,
or tested directly with small CSV/parquet inputs later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {path}")


def build_temporal_features(
    target_df: pd.DataFrame,
    child_df: pd.DataFrame,
    target_key: str,
    target_time_col: str,
    child_key: str,
    child_time_col: str,
    numeric_col: str,
    windows_days: Iterable[int] = (30, 90),
    row_id_col: str = "__target_row_id__",
    strict_before: bool = False,
    source_name: str | None = None,
) -> pd.DataFrame:
    """
    Build leakage-safe temporal features.

    For each target row, this only uses child rows satisfying:
        child_time <= target_time

    Returns:
        One row per target row, aligned by row_id_col.
    """

    required_target_cols = {target_key, target_time_col}
    required_child_cols = {child_key, child_time_col, numeric_col}

    missing_target = required_target_cols - set(target_df.columns)
    missing_child = required_child_cols - set(child_df.columns)

    if missing_target:
        raise ValueError(f"Missing target columns: {sorted(missing_target)}")
    if missing_child:
        raise ValueError(f"Missing child columns: {sorted(missing_child)}")

    target = target_df.copy()
    child = child_df.copy()

    if row_id_col not in target.columns:
        target[row_id_col] = np.arange(len(target), dtype=np.int64)

    target[target_time_col] = pd.to_datetime(target[target_time_col], errors="coerce")
    child[child_time_col] = pd.to_datetime(child[child_time_col], errors="coerce")
    child[numeric_col] = pd.to_numeric(child[numeric_col], errors="coerce")

    # Rename time/key columns before merge to avoid pandas suffixes
    # when target_time_col == child_time_col, e.g. both are "date".
    target_time_tmp = "__target_time__"
    child_time_tmp = "__child_time__"
    target_key_tmp = "__target_key__"
    child_key_tmp = "__child_key__"

    target_small = target[[row_id_col, target_key, target_time_col]].copy()
    target_small = target_small.rename(
        columns={
            target_key: target_key_tmp,
            target_time_col: target_time_tmp,
        }
    )

    child_small = child[[child_key, child_time_col, numeric_col]].copy()
    child_small = child_small.rename(
        columns={
            child_key: child_key_tmp,
            child_time_col: child_time_tmp,
        }
    )

    merged = target_small.merge(
        child_small,
        left_on=target_key_tmp,
        right_on=child_key_tmp,
        how="left",
    )

    # Leakage guard: only past child rows are allowed.
    # strict_before=True uses child_time < target_time.
    # This is safer for tasks where same-date child rows may contain current-event outcomes.
    if strict_before:
        time_ok = merged[child_time_tmp] < merged[target_time_tmp]
    else:
        time_ok = merged[child_time_tmp] <= merged[target_time_tmp]

    merged = merged[
        merged[child_time_tmp].notna()
        & merged[target_time_tmp].notna()
        & time_ok
    ].copy()

    if source_name is None:
        source_name = child_key

    feature_prefix = (
        f"fdhg::temporal::{target_key}<-{source_name}.{child_key}"
        f"::{numeric_col}"
    )

    out = target_small[[row_id_col]].copy()

    # Define all expected columns up front for stable schema.
    expected_cols = [
        f"{feature_prefix}::all_past_count",
        f"{feature_prefix}::all_past_mean",
        f"{feature_prefix}::all_past_std",
        f"{feature_prefix}::last_value",
        f"{feature_prefix}::time_since_last_days",
        f"{feature_prefix}::trend_last_minus_mean",
    ]

    for w in windows_days:
        expected_cols.extend(
            [
                f"{feature_prefix}::recent_{w}d_count",
                f"{feature_prefix}::recent_{w}d_mean",
            ]
        )

    if 30 in set(windows_days):
        expected_cols.append(f"{feature_prefix}::trend_recent30_minus_all")

    if merged.empty:
        for c in expected_cols:
            out[c] = 0 if c.endswith("_count") else np.nan
        return out

    # All-past aggregation.
    all_past = (
        merged.groupby(row_id_col)[numeric_col]
        .agg(
            all_past_count="count",
            all_past_mean="mean",
            all_past_std="std",
        )
        .reset_index()
        .rename(
            columns={
                "all_past_count": f"{feature_prefix}::all_past_count",
                "all_past_mean": f"{feature_prefix}::all_past_mean",
                "all_past_std": f"{feature_prefix}::all_past_std",
            }
        )
    )

    out = out.merge(all_past, on=row_id_col, how="left")

    # Last value and time since last.
    merged_sorted = merged.sort_values([row_id_col, child_time_tmp])
    last_rows = merged_sorted.groupby(row_id_col, as_index=False).tail(1).copy()

    last_rows[f"{feature_prefix}::last_value"] = last_rows[numeric_col]
    last_rows[f"{feature_prefix}::time_since_last_days"] = (
        last_rows[target_time_tmp] - last_rows[child_time_tmp]
    ).dt.total_seconds() / 86400.0

    out = out.merge(
        last_rows[
            [
                row_id_col,
                f"{feature_prefix}::last_value",
                f"{feature_prefix}::time_since_last_days",
            ]
        ],
        on=row_id_col,
        how="left",
    )

    # Recent-window features.
    for w in windows_days:
        lower_bound = merged[target_time_tmp] - pd.to_timedelta(w, unit="D")
        if strict_before:
            recent_upper_ok = merged[child_time_tmp] < merged[target_time_tmp]
        else:
            recent_upper_ok = merged[child_time_tmp] <= merged[target_time_tmp]

        recent = merged[
            (merged[child_time_tmp] > lower_bound)
            & recent_upper_ok
        ].copy()

        count_col = f"{feature_prefix}::recent_{w}d_count"
        mean_col = f"{feature_prefix}::recent_{w}d_mean"

        if recent.empty:
            recent_agg = target_small[[row_id_col]].copy()
            recent_agg[count_col] = 0
            recent_agg[mean_col] = np.nan
        else:
            recent_agg = (
                recent.groupby(row_id_col)[numeric_col]
                .agg(**{count_col: "count", mean_col: "mean"})
                .reset_index()
            )

        out = out.merge(recent_agg, on=row_id_col, how="left")

    # Fill count columns.
    count_cols = [c for c in out.columns if c.endswith("_count")]
    for c in count_cols:
        out[c] = out[c].fillna(0).astype(np.float32)

    # Trend features.
    all_mean_col = f"{feature_prefix}::all_past_mean"
    last_col = f"{feature_prefix}::last_value"

    out[f"{feature_prefix}::trend_last_minus_mean"] = (
        out[last_col] - out[all_mean_col]
    )

    recent30_col = f"{feature_prefix}::recent_30d_mean"
    if recent30_col in out.columns:
        out[f"{feature_prefix}::trend_recent30_minus_all"] = (
            out[recent30_col] - out[all_mean_col]
        )

    # Ensure stable column order and existence.
    for c in expected_cols:
        if c not in out.columns:
            out[c] = 0 if c.endswith("_count") else np.nan

    return out[[row_id_col] + expected_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--child-path", required=True)
    parser.add_argument("--out-path", required=True)

    parser.add_argument("--target-key", required=True)
    parser.add_argument("--target-time-col", required=True)
    parser.add_argument("--child-key", required=True)
    parser.add_argument("--child-time-col", required=True)
    parser.add_argument("--numeric-col", required=True)
    parser.add_argument("--source-name", default=None)

    parser.add_argument("--windows-days", default="30,90")
    parser.add_argument("--row-id-col", default="__target_row_id__")
    parser.add_argument("--strict-before", action="store_true")

    args = parser.parse_args()

    windows_days = tuple(
        int(x.strip()) for x in args.windows_days.split(",") if x.strip()
    )

    target_df = read_table(args.target_path)
    child_df = read_table(args.child_path)

    feat_df = build_temporal_features(
        target_df=target_df,
        child_df=child_df,
        target_key=args.target_key,
        target_time_col=args.target_time_col,
        child_key=args.child_key,
        child_time_col=args.child_time_col,
        numeric_col=args.numeric_col,
        windows_days=windows_days,
        row_id_col=args.row_id_col,
        strict_before=args.strict_before,
        source_name=args.source_name,
    )

    write_table(feat_df, args.out_path)
    print(f"[OK] wrote temporal features: {args.out_path}")
    print(f"[OK] shape: {feat_df.shape}")
    print("[OK] columns:")
    for c in feat_df.columns:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
