from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def maybe_sample(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)


def infer_time_col(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if lc in {"timestamp", "time", "date", "datetime", "t_dat"}:
            candidates.append(c)
        elif "time" in lc or "date" in lc:
            candidates.append(c)
    return candidates[0] if candidates else None


def infer_target_key(target_df: pd.DataFrame) -> str:
    excluded = {"index", "unnamed: 0", "Unnamed: 0", "timestamp", "time", "date", "datetime", "churn", "label", "target", "y"}
    id_cols = [
        c for c in target_df.columns
        if c.lower().endswith("_id") and c.lower() not in excluded
    ]
    if len(id_cols) == 1:
        return id_cols[0]
    if id_cols:
        return id_cols[0]

    candidates = [c for c in target_df.columns if c.lower() not in excluded]
    if not candidates:
        raise ValueError("Could not infer target key.")
    return candidates[0]


def is_numeric_non_id(s: pd.Series, col: str, target_key: str | None = None) -> bool:
    lc = col.lower()
    if target_key is not None and col == target_key:
        return False
    if lc.endswith("_id") or lc.endswith("id") or lc in {"id", "index", "unnamed: 0"}:
        return False
    return pd.api.types.is_numeric_dtype(s)


def infer_child_config(
    inspect_dir: Path,
    target_key: str,
    target_time_col: Optional[str],
    preferred_child: Optional[str] = None,
    preferred_numeric: Optional[str] = None,
) -> dict:
    table_paths = sorted(inspect_dir.glob("table_*.parquet"))
    candidates = []

    for path in table_paths:
        table_name = path.stem.replace("table_", "")
        df = load_parquet(path)

        if target_key not in df.columns:
            continue

        time_col = infer_time_col(df)
        if target_time_col is not None and time_col is None:
            # RelBench target splits often use standardized cutoff column
            # "timestamp", while raw DB tables use dataset-specific names
            # such as CreationDate or Date. Try common aliases.
            for cand in ["timestamp", "CreationDate", "Date", "t_dat", "datetime", "time"]:
                if cand in df.columns:
                    time_col = cand
                    break
            if time_col is None:
                continue

        numeric_cols = [
            c for c in df.columns
            if is_numeric_non_id(df[c], c, target_key=target_key)
        ]

        if preferred_numeric is not None and preferred_numeric in df.columns:
            numeric_cols = [preferred_numeric]

        if not numeric_cols:
            continue

        candidates.append(
            {
                "table_name": table_name,
                "path": path,
                "n_rows": len(df),
                "time_col": time_col,
                "numeric_cols": numeric_cols,
            }
        )

    if preferred_child is not None:
        for c in candidates:
            if c["table_name"] == preferred_child:
                return c
        raise ValueError(f"Preferred child table {preferred_child} not found among candidates: {candidates}")

    if not candidates:
        raise ValueError(
            f"No child table found with target_key={target_key}, time_col={target_time_col}"
        )

    # Prefer large event-like tables.
    candidates = sorted(candidates, key=lambda x: x["n_rows"], reverse=True)
    return candidates[0]


def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__fdhg_row_id"] = range(len(out))
    return out


def build_dfs_split(
    con: duckdb.DuckDBPyConnection,
    target_df: pd.DataFrame,
    *,
    split: str,
    target_key: str,
    target_time_col: str,
    child_table: str,
    child_time_col: str,
    numeric_col: str,
    prefix: str,
) -> pd.DataFrame:
    target_df = add_row_id(target_df)
    con.register("target_split", target_df)

    query = f"""
    WITH target_cast AS (
        SELECT
            __fdhg_row_id,
            {q(target_key)} AS target_id,
            CAST({q(target_time_col)} AS TIMESTAMP) AS target_timestamp
        FROM target_split
    ),

    joined AS (
        SELECT
            t.__fdhg_row_id,
            t.target_id,
            t.target_timestamp,
            c.child_time,
            c.cum_count,
            c.cum_sum,
            c.cum_sumsq,
            c.cum_max
        FROM target_cast t
        ASOF LEFT JOIN cum_child c
          ON t.target_id = c.target_id
         AND t.target_timestamp >= c.child_time
    )

    SELECT
        __fdhg_row_id,
        target_id AS {q(target_key)},

        COALESCE(cum_count, 0) AS {q(f"f_{prefix}_count")},

        CASE
            WHEN cum_count IS NULL OR cum_count = 0 THEN NULL
            ELSE cum_sum / cum_count
        END AS {q(f"f_{prefix}_{numeric_col}_mean")},

        CASE
            WHEN cum_count IS NULL OR cum_count = 0 THEN NULL
            ELSE SQRT(GREATEST(cum_sumsq / cum_count - POWER(cum_sum / cum_count, 2), 0))
        END AS {q(f"f_{prefix}_{numeric_col}_std")},

        cum_max AS {q(f"f_{prefix}_{numeric_col}_max")},

        CASE
            WHEN child_time IS NULL THEN NULL
            ELSE DATE_DIFF('day', child_time, target_timestamp)
        END AS {q(f"f_{prefix}_days_since_last")}

    FROM joined
    ORDER BY __fdhg_row_id
    """

    features = con.execute(query).df()

    if len(features) != len(target_df):
        raise RuntimeError(f"{split}: row count changed target={len(target_df)}, features={len(features)}")

    for col in [
        f"f_{prefix}_{numeric_col}_mean",
        f"f_{prefix}_{numeric_col}_std",
        f"f_{prefix}_{numeric_col}_max",
        f"f_{prefix}_days_since_last",
    ]:
        features[f"{col}__is_missing"] = features[col].isna().astype("int8")

    return features


def build_naive_split(
    con: duckdb.DuckDBPyConnection,
    target_df: pd.DataFrame,
    *,
    split: str,
    target_key: str,
    target_time_col: str,
    numeric_col: str,
    prefix: str,
) -> pd.DataFrame:
    target_df = add_row_id(target_df)
    con.register("target_split", target_df)

    query = f"""
    WITH target_cast AS (
        SELECT
            __fdhg_row_id,
            {q(target_key)} AS target_id,
            CAST({q(target_time_col)} AS TIMESTAMP) AS target_timestamp
        FROM target_split
    ),

    joined AS (
        SELECT
            t.__fdhg_row_id,
            t.target_id,
            t.target_timestamp,
            c.child_time,
            c.value
        FROM target_cast t
        ASOF LEFT JOIN child_latest c
          ON t.target_id = c.target_id
         AND t.target_timestamp >= c.child_time
    )

    SELECT
        __fdhg_row_id,
        target_id AS {q(target_key)},
        value AS {q(f"f_naive_latest_{prefix}_{numeric_col}")},
        CASE
            WHEN child_time IS NULL THEN NULL
            ELSE DATE_DIFF('day', child_time, target_timestamp)
        END AS {q(f"f_naive_days_since_latest_{prefix}")}
    FROM joined
    ORDER BY __fdhg_row_id
    """

    features = con.execute(query).df()

    if len(features) != len(target_df):
        raise RuntimeError(f"{split}: row count changed target={len(target_df)}, features={len(features)}")

    for col in [
        f"f_naive_latest_{prefix}_{numeric_col}",
        f"f_naive_days_since_latest_{prefix}",
    ]:
        features[f"{col}__is_missing"] = features[col].isna().astype("int8")

    return features


def build_features(
    *,
    inspect_dir: Path,
    out_dir: Path,
    mode: str,
    max_train: int,
    max_val: int,
    max_test: int,
    seed: int,
    target_key: Optional[str],
    target_time_col: Optional[str],
    child_table: Optional[str],
    numeric_col: Optional[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_parquet(inspect_dir / "target_train.parquet")
    val = load_parquet(inspect_dir / "target_val.parquet")
    test = load_parquet(inspect_dir / "target_test.parquet")

    if target_key is None:
        target_key = infer_target_key(train)
    if target_time_col is None:
        target_time_col = infer_time_col(train)

    if target_time_col is None:
        raise ValueError("Could not infer target time column. Pass --target-time-col.")

    child_cfg = infer_child_config(
        inspect_dir,
        target_key=target_key,
        target_time_col=target_time_col,
        preferred_child=child_table,
        preferred_numeric=numeric_col,
    )

    child_table = child_cfg["table_name"]
    child_path = child_cfg["path"]
    child_time_col = child_cfg["time_col"]

    # RelBench v2 RateBeer fix:
    # updated_by / created_by are numeric user-id audit columns, not timestamps.
    if str(child_time_col).lower() in {"updated_by", "created_by", "updatedby", "createdby"}:
        print(f"[time-fix] overriding child_time_col {child_time_col} -> created_at")
        child_time_col = "created_at"
    numeric_col = numeric_col or child_cfg["numeric_cols"][0]
    prefix = child_table

    child = load_parquet(child_path)

    train = maybe_sample(train, max_train, seed)
    val = maybe_sample(val, max_val, seed)
    test = maybe_sample(test, max_test, seed)

    print("=== Generic RelBench feature builder ===")
    print("mode:", mode)
    print("inspect_dir:", inspect_dir)
    print("target_key:", target_key)
    print("target_time_col:", target_time_col)
    print("child_table:", child_table)
    print("child_path:", child_path)
    print("child_time_col:", child_time_col)
    print("numeric_col:", numeric_col)
    print("train:", train.shape)
    print("val:", val.shape)
    print("test:", test.shape)
    print("child:", child.shape)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8")
    con.register("child_raw", child)

    if mode == "dfs":
        con.execute(f"""
        CREATE TEMP TABLE child_daily AS
        SELECT
            {q(target_key)} AS target_id,
            CAST({q(child_time_col)} AS TIMESTAMP) AS child_time,
            COUNT(*) AS n,
            SUM(TRY_CAST({q(numeric_col)} AS DOUBLE)) AS sum_value,
            SUM(POWER(TRY_CAST({q(numeric_col)} AS DOUBLE), 2)) AS sumsq_value,
            MAX(TRY_CAST({q(numeric_col)} AS DOUBLE)) AS max_value
        FROM child_raw
        WHERE {q(target_key)} IS NOT NULL
          AND {q(child_time_col)} IS NOT NULL
          AND {q(numeric_col)} IS NOT NULL
        GROUP BY {q(target_key)}, CAST({q(child_time_col)} AS TIMESTAMP)
        """)

        con.execute("""
        CREATE TEMP TABLE cum_child AS
        SELECT
            target_id,
            child_time,
            SUM(n) OVER (
                PARTITION BY target_id
                ORDER BY child_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_count,
            SUM(sum_value) OVER (
                PARTITION BY target_id
                ORDER BY child_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_sum,
            SUM(sumsq_value) OVER (
                PARTITION BY target_id
                ORDER BY child_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_sumsq,
            MAX(max_value) OVER (
                PARTITION BY target_id
                ORDER BY child_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_max
        FROM child_daily
        ORDER BY target_id, child_time
        """)

    elif mode == "naive":
        con.execute(f"""
        CREATE TEMP TABLE child_latest AS
        SELECT
            {q(target_key)} AS target_id,
            CAST({q(child_time_col)} AS TIMESTAMP) AS child_time,
            AVG(TRY_CAST({q(numeric_col)} AS DOUBLE)) AS value
        FROM child_raw
        WHERE {q(target_key)} IS NOT NULL
          AND {q(child_time_col)} IS NOT NULL
          AND {q(numeric_col)} IS NOT NULL
        GROUP BY {q(target_key)}, CAST({q(child_time_col)} AS TIMESTAMP)
        ORDER BY target_id, child_time
        """)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    split_to_df = {
        "train": train,
        "val": val,
        "test": test,
    }

    feature_cols_ref = None

    for split, target_df in split_to_df.items():
        if mode == "dfs":
            features = build_dfs_split(
                con,
                target_df,
                split=split,
                target_key=target_key,
                target_time_col=target_time_col,
                child_table=child_table,
                child_time_col=child_time_col,
                numeric_col=numeric_col,
                prefix=prefix,
            )
        else:
            features = build_naive_split(
                con,
                target_df,
                split=split,
                target_key=target_key,
                target_time_col=target_time_col,
                numeric_col=numeric_col,
                prefix=prefix,
            )

        feature_cols = [c for c in features.columns if c.startswith("f_")]
        if feature_cols_ref is None:
            feature_cols_ref = feature_cols
        elif feature_cols != feature_cols_ref:
            raise RuntimeError(f"{split}: feature columns not aligned")

        features_path = out_dir / f"{mode}_features_{split}.parquet"
        features.drop(columns=["__fdhg_row_id"]).to_parquet(features_path, index=False)

        combined = pd.concat(
            [
                target_df.reset_index(drop=True),
                features.drop(columns=["__fdhg_row_id", target_key]).reset_index(drop=True),
            ],
            axis=1,
        )

        # Standard name expected by existing TabPFN runner.
        combined_path = out_dir / f"target_with_dfs_agg_{split}.parquet"
        combined.to_parquet(combined_path, index=False)

        print(f"\n[{split}]")
        print("features:", features.shape)
        print("combined:", combined.shape)
        print("saved:", combined_path)

    manifest_rows = []
    for f in feature_cols_ref:
        manifest_rows.append(
            {
                "feature_name": f,
                "mode": mode,
                "child_table": child_table,
                "target_key": target_key,
                "child_time_col": child_time_col,
                "numeric_col": numeric_col,
                "path_signature": f"{target_key}<-{child_table}.{target_key}",
                "temporal_predicate": f"{child_table}.{child_time_col} <= target.{target_time_col}",
                "temporal_safe": True,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(out_dir / f"{mode}_feature_manifest.csv", index=False)

    config = {
        "mode": mode,
        "inspect_dir": str(inspect_dir),
        "out_dir": str(out_dir),
        "target_key": target_key,
        "target_time_col": target_time_col,
        "child_table": child_table,
        "child_time_col": child_time_col,
        "numeric_col": numeric_col,
        "feature_cols": feature_cols_ref,
    }

    with open(out_dir / f"{mode}_feature_config.json", "w") as f:
        import json
        json.dump(config, f, indent=2)

    con.close()

    print("\n=== Done ===")
    print("out_dir:", out_dir)
    print("features:")
    for f in feature_cols_ref:
        print(" -", f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["dfs", "naive"], required=True)
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-val", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=41)

    parser.add_argument("--target-key", default=None)
    parser.add_argument("--target-time-col", default=None)
    parser.add_argument("--child-table", default=None)
    parser.add_argument("--numeric-col", default=None)

    args = parser.parse_args()

    build_features(
        inspect_dir=Path(args.inspect_dir),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
        target_key=args.target_key,
        target_time_col=args.target_time_col,
        child_table=args.child_table,
        numeric_col=args.numeric_col,
    )


if __name__ == "__main__":
    main()
