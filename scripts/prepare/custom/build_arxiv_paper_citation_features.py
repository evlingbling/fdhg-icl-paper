from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-val", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
    )
    return parser.parse_args()


def sample_target(
    path: Path,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if len(df) > limit:
        df = df.sample(n=limit, random_state=seed)

    df = df.reset_index(drop=True)

    if "__row_id" in df.columns:
        df = df.drop(columns=["__row_id"])

    df.insert(0, "__row_id", range(len(df)))
    return df


def build_features(
    con: duckdb.DuckDBPyConnection,
    target: pd.DataFrame,
) -> pd.DataFrame:
    con.register("target_split", target)

    query = """
    WITH target_rows AS (
        SELECT
            __row_id,
            Paper_ID,
            CAST(date AS TIMESTAMP) AS target_date
        FROM target_split
    ),

    incoming_stats AS (
        SELECT
            t.__row_id,
            COUNT(c.Paper_ID) AS all_past_count,
            DATE_DIFF(
                'day',
                MAX(CAST(c.Submission_Date AS TIMESTAMP)),
                MAX(t.target_date)
            ) AS time_since_last_days,
            COUNT(
                CASE
                    WHEN CAST(c.Submission_Date AS TIMESTAMP)
                         > t.target_date - INTERVAL '30 days'
                    THEN 1
                END
            ) AS recent_30d_count,
            COUNT(
                CASE
                    WHEN CAST(c.Submission_Date AS TIMESTAMP)
                         > t.target_date - INTERVAL '90 days'
                    THEN 1
                END
            ) AS recent_90d_count,
            COUNT(
                CASE
                    WHEN CAST(c.Submission_Date AS TIMESTAMP)
                         > t.target_date - INTERVAL '365 days'
                    THEN 1
                END
            ) AS recent_365d_count
        FROM target_rows t
        LEFT JOIN citations c
          ON c.References_Paper_ID = t.Paper_ID
         AND CAST(c.Submission_Date AS TIMESTAMP) < t.target_date
        GROUP BY t.__row_id
    ),

    outgoing_stats AS (
        SELECT
            t.__row_id,

            COUNT(c.References_Paper_ID)
            AS all_past_count,

            DATE_DIFF(
                'day',
                MAX(CAST(c.Submission_Date AS TIMESTAMP)),
                t.target_date
            ) AS time_since_last_days,

            COUNT(
                CASE
                    WHEN CAST(c.Submission_Date AS TIMESTAMP)
                         > t.target_date - INTERVAL '30 days'
                    THEN c.References_Paper_ID
                END
            ) AS recent_30d_count,

            COUNT(
                CASE
                    WHEN CAST(c.Submission_Date AS TIMESTAMP)
                         > t.target_date - INTERVAL '90 days'
                    THEN c.References_Paper_ID
                END
            ) AS recent_90d_count,

            COUNT(
                CASE
                    WHEN CAST(c.Submission_Date AS TIMESTAMP)
                         > t.target_date - INTERVAL '365 days'
                    THEN c.References_Paper_ID
                END
            ) AS recent_365d_count

        FROM target_rows t

        LEFT JOIN citations c
          ON c.Paper_ID = t.Paper_ID
         AND CAST(c.Submission_Date AS TIMESTAMP) < t.target_date

        GROUP BY t.__row_id, t.target_date
    ),

    author_counts AS (
        SELECT
            t.__row_id,
            COUNT(DISTINCT pa.Author_ID) AS n_authors
        FROM target_rows t
        LEFT JOIN paper_authors pa
          ON pa.Paper_ID = t.Paper_ID
        GROUP BY t.__row_id
    ),

    category_counts AS (
        SELECT
            t.__row_id,
            COUNT(DISTINCT pc.Category_ID) AS n_categories
        FROM target_rows t
        LEFT JOIN paper_categories pc
          ON pc.Paper_ID = t.Paper_ID
        GROUP BY t.__row_id
    ),

    paper_static AS (
        SELECT
            t.__row_id,
            p.Primary_Category_ID,
            DATE_DIFF(
                'day',
                CAST(p.Submission_Date AS TIMESTAMP),
                t.target_date
            ) AS age_days
        FROM target_rows t
        LEFT JOIN papers p
          ON p.Paper_ID = t.Paper_ID
    )

    SELECT
        t.__row_id,

        CAST(COALESCE(i.all_past_count, 0) AS FLOAT)
        AS "fdhg::temporal::Paper_ID<-incoming_citations.References_Paper_ID::Paper_ID::all_past_count",

        i.time_since_last_days
        AS "fdhg::temporal::Paper_ID<-incoming_citations.References_Paper_ID::Paper_ID::time_since_last_days",

        CAST(COALESCE(i.recent_30d_count, 0) AS FLOAT)
        AS "fdhg::temporal::Paper_ID<-incoming_citations.References_Paper_ID::Paper_ID::recent_30d_count",

        CAST(COALESCE(i.recent_90d_count, 0) AS FLOAT)
        AS "fdhg::temporal::Paper_ID<-incoming_citations.References_Paper_ID::Paper_ID::recent_90d_count",

        CAST(COALESCE(i.recent_365d_count, 0) AS FLOAT)
        AS "fdhg::temporal::Paper_ID<-incoming_citations.References_Paper_ID::Paper_ID::recent_365d_count",

        CAST(COALESCE(o.all_past_count, 0) AS FLOAT)
        AS "fdhg::temporal::Paper_ID<-outgoing_references.Paper_ID::References_Paper_ID::all_past_count",

        o.time_since_last_days
        AS "fdhg::temporal::Paper_ID<-outgoing_references.Paper_ID::References_Paper_ID::time_since_last_days",

        CAST(COALESCE(o.recent_30d_count, 0) AS FLOAT)
        AS "fdhg::temporal::Paper_ID<-outgoing_references.Paper_ID::References_Paper_ID::recent_30d_count",

        CAST(COALESCE(o.recent_90d_count, 0) AS FLOAT)
        AS "fdhg::temporal::Paper_ID<-outgoing_references.Paper_ID::References_Paper_ID::recent_90d_count",

        CAST(COALESCE(o.recent_365d_count, 0) AS FLOAT)
        AS "fdhg::temporal::Paper_ID<-outgoing_references.Paper_ID::References_Paper_ID::recent_365d_count",

        p.Primary_Category_ID
        AS "dfs::paper::primary_category_id",

        CAST(COALESCE(a.n_authors, 0) AS DOUBLE)
        AS "dfs::paper::n_authors",

        CAST(COALESCE(c.n_categories, 0) AS DOUBLE)
        AS "dfs::paper::n_categories",

        p.age_days
        AS "dfs::paper::age_days"

    FROM target_rows t
    LEFT JOIN incoming_stats i USING (__row_id)
    LEFT JOIN outgoing_stats o USING (__row_id)
    LEFT JOIN author_counts a USING (__row_id)
    LEFT JOIN category_counts c USING (__row_id)
    LEFT JOIN paper_static p USING (__row_id)
    ORDER BY t.__row_id
    """

    features = con.execute(query).df()

    if len(features) != len(target):
        raise RuntimeError(
            f"Row count mismatch: target={len(target)}, "
            f"features={len(features)}"
        )

    if not features["__row_id"].equals(target["__row_id"]):
        raise RuntimeError("Row alignment changed during feature generation")

    return target.merge(
        features,
        on="__row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )


def main() -> None:
    args = parse_args()

    inspect_dir = Path(args.inspect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    limits = {
        "train": args.max_train,
        "val": args.max_val,
        "test": args.max_test,
    }

    con = duckdb.connect()

    con.execute(
        f"""
        CREATE VIEW citations AS
        SELECT * FROM read_parquet(
            '{inspect_dir / "table_citations.parquet"}'
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW papers AS
        SELECT * FROM read_parquet(
            '{inspect_dir / "table_papers.parquet"}'
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW paper_authors AS
        SELECT * FROM read_parquet(
            '{inspect_dir / "table_paperAuthors.parquet"}'
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW paper_categories AS
        SELECT * FROM read_parquet(
            '{inspect_dir / "table_paperCategories.parquet"}'
        )
        """
    )

    for split in args.splits:
        target = sample_target(
            inspect_dir / f"target_{split}.parquet",
            limits[split],
            args.seed,
        )

        combined = build_features(con, target)

        output_path = (
            out_dir
            / f"target_with_dfs_agg_{split}.parquet"
        )

        combined.to_parquet(output_path, index=False)

        print(
            split,
            combined.shape,
            "saved:",
            output_path,
        )


if __name__ == "__main__":
    main()
