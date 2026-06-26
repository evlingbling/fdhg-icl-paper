#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-val", type=int, default=2000)
    p.add_argument("--max-test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=41)
    return p.parse_args()


def sample_target(path: Path, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if len(df) > n:
        df = df.sample(n=n, random_state=seed)

    df = df.reset_index(drop=True)

    if "__row_id" in df.columns:
        df = df.drop(columns=["__row_id"])

    df.insert(0, "__row_id", range(len(df)))
    return df


def build_split_features(
    con: duckdb.DuckDBPyConnection,
    target: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    con.register("target_split", target)

    query = """
    WITH target_cast AS (
        SELECT
            __row_id,
            Author_ID,
            CAST(date AS TIMESTAMP) AS target_date
        FROM target_split
    ),

    author_papers AS (
        SELECT
            t.__row_id,
            t.Author_ID,
            t.target_date,
            pa.Paper_ID,
            pa.Author_ID AS historical_author_id,
            CAST(pa.Submission_Date AS TIMESTAMP) AS paper_date,
            p.Primary_Category_ID
        FROM target_cast t
        LEFT JOIN paper_authors pa
          ON pa.Author_ID = t.Author_ID
         AND CAST(pa.Submission_Date AS TIMESTAMP) < t.target_date
        LEFT JOIN papers p
          ON p.Paper_ID = pa.Paper_ID
         AND CAST(p.Submission_Date AS TIMESTAMP) < t.target_date
    ),

    paper_counts AS (
        SELECT
            __row_id,
            COUNT(DISTINCT Paper_ID) AS author_past_paper_count,
            COUNT(
                DISTINCT CASE
                    WHEN paper_date >= target_date - INTERVAL '30 days'
                    THEN Paper_ID
                END
            ) AS author_past_paper_count_30d,
            COUNT(
                DISTINCT CASE
                    WHEN paper_date >= target_date - INTERVAL '90 days'
                    THEN Paper_ID
                END
            ) AS author_past_paper_count_90d,
            COUNT(
                DISTINCT CASE
                    WHEN paper_date >= target_date - INTERVAL '365 days'
                    THEN Paper_ID
                END
            ) AS author_past_paper_count_365d,
            DATE_DIFF(
                'day',
                MAX(paper_date),
                MAX(target_date)
            ) AS author_days_since_last_paper,
            COUNT(DISTINCT Primary_Category_ID)
                AS author_past_unique_primary_categories
        FROM author_papers
        GROUP BY __row_id
    ),

    category_counts AS (
        SELECT
            __row_id,
            Primary_Category_ID,
            COUNT(DISTINCT Paper_ID) AS category_paper_count
        FROM author_papers
        WHERE Primary_Category_ID IS NOT NULL
          AND Paper_ID IS NOT NULL
        GROUP BY __row_id, Primary_Category_ID
    ),

    category_totals AS (
        SELECT
            __row_id,
            SUM(category_paper_count) AS total_category_papers,
            MAX(category_paper_count) AS max_category_papers
        FROM category_counts
        GROUP BY __row_id
    ),

    category_stats AS (
        SELECT
            c.__row_id,
            MAX(c.category_paper_count)::DOUBLE
                / NULLIF(MAX(t.total_category_papers), 0)
                AS author_primary_category_majority_confidence,
            -SUM(
                (
                    c.category_paper_count::DOUBLE
                    / NULLIF(t.total_category_papers, 0)
                )
                * LN(
                    c.category_paper_count::DOUBLE
                    / NULLIF(t.total_category_papers, 0)
                )
            ) AS author_primary_category_entropy
        FROM category_counts c
        JOIN category_totals t USING (__row_id)
        GROUP BY c.__row_id
    ),

    latest_category AS (
        SELECT
            __row_id,
            Primary_Category_ID AS author_last_primary_category
        FROM (
            SELECT
                __row_id,
                Primary_Category_ID,
                ROW_NUMBER() OVER (
                    PARTITION BY __row_id
                    ORDER BY paper_date DESC, Paper_ID DESC
                ) AS rn
            FROM author_papers
            WHERE Paper_ID IS NOT NULL
              AND Primary_Category_ID IS NOT NULL
        )
        WHERE rn = 1
    ),

    historical_paper_ids AS (
        SELECT DISTINCT
            __row_id,
            Paper_ID,
            target_date
        FROM author_papers
        WHERE Paper_ID IS NOT NULL
    ),

    coauthor_rows AS (
        SELECT
            h.__row_id,
            h.Paper_ID,
            pa2.Author_ID AS coauthor_id
        FROM historical_paper_ids h
        JOIN paper_authors pa2
          ON pa2.Paper_ID = h.Paper_ID
         AND CAST(pa2.Submission_Date AS TIMESTAMP) < h.target_date
    ),

    coauthor_stats AS (
        SELECT
            c.__row_id,
            COUNT(DISTINCT c.coauthor_id)
                AS author_past_unique_collaborators,
            AVG(p.n_authors_on_paper)
                AS author_mean_authors_per_paper,
            MAX(p.n_authors_on_paper)
                AS author_max_authors_per_paper
        FROM coauthor_rows c
        JOIN (
            SELECT
                __row_id,
                Paper_ID,
                COUNT(DISTINCT coauthor_id) AS n_authors_on_paper
            FROM coauthor_rows
            GROUP BY __row_id, Paper_ID
        ) p
          ON p.__row_id = c.__row_id
         AND p.Paper_ID = c.Paper_ID
        GROUP BY c.__row_id
    ),

    citation_stats AS (
        SELECT
            h.__row_id,
            COUNT(c.References_Paper_ID)
                AS author_past_incoming_citation_count,
            COUNT(DISTINCT c.Paper_ID)
                AS author_past_unique_citing_papers,
            COUNT(
                CASE
                    WHEN CAST(c.Submission_Date AS TIMESTAMP)
                         >= h.target_date - INTERVAL '365 days'
                    THEN 1
                END
            ) AS author_past_incoming_citation_count_365d
        FROM historical_paper_ids h
        LEFT JOIN citations c
          ON c.References_Paper_ID = h.Paper_ID
         AND CAST(c.Submission_Date AS TIMESTAMP) < h.target_date
        GROUP BY h.__row_id
    )

    SELECT
        t.__row_id,

        COALESCE(pc.author_past_paper_count, 0)
            AS "dfs::author::past_paper_count",
        COALESCE(pc.author_past_paper_count_30d, 0)
            AS "dfs::author::past_paper_count_30d",
        COALESCE(pc.author_past_paper_count_90d, 0)
            AS "dfs::author::past_paper_count_90d",
        COALESCE(pc.author_past_paper_count_365d, 0)
            AS "dfs::author::past_paper_count_365d",
        pc.author_days_since_last_paper
            AS "dfs::author::days_since_last_paper",
        COALESCE(pc.author_past_unique_primary_categories, 0)
            AS "dfs::author::past_unique_primary_categories",

        COALESCE(cs.author_primary_category_majority_confidence, 0.0)
            AS "fdhg::author_category::majority_confidence",
        COALESCE(cs.author_primary_category_entropy, 0.0)
            AS "fdhg::author_category::entropy",
        lc.author_last_primary_category
            AS "fdhg::author_category::last_primary_category",

        COALESCE(co.author_past_unique_collaborators, 0)
            AS "dfs::author::past_unique_collaborators",
        COALESCE(co.author_mean_authors_per_paper, 0.0)
            AS "dfs::author::mean_authors_per_paper",
        COALESCE(co.author_max_authors_per_paper, 0)
            AS "dfs::author::max_authors_per_paper",

        COALESCE(ci.author_past_incoming_citation_count, 0)
            AS "dfs::author::past_incoming_citation_count",
        COALESCE(ci.author_past_unique_citing_papers, 0)
            AS "dfs::author::past_unique_citing_papers",
        COALESCE(ci.author_past_incoming_citation_count_365d, 0)
            AS "dfs::author::past_incoming_citation_count_365d"

    FROM target_cast t
    LEFT JOIN paper_counts pc USING (__row_id)
    LEFT JOIN category_stats cs USING (__row_id)
    LEFT JOIN latest_category lc USING (__row_id)
    LEFT JOIN coauthor_stats co USING (__row_id)
    LEFT JOIN citation_stats ci USING (__row_id)
    ORDER BY t.__row_id
    """

    features = con.execute(query).df()

    if len(features) != len(target):
        raise RuntimeError(
            f"{split}: row count mismatch "
            f"target={len(target)} features={len(features)}"
        )

    if not features["__row_id"].equals(target["__row_id"]):
        raise RuntimeError(f"{split}: row alignment changed")

    combined = target.merge(
        features,
        on="__row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    return combined


def main():
    args = parse_args()

    inspect_dir = Path(args.inspect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_limits = {
        "train": args.max_train,
        "val": args.max_val,
        "test": args.max_test,
    }

    con = duckdb.connect()

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
        CREATE VIEW papers AS
        SELECT * FROM read_parquet(
            '{inspect_dir / "table_papers.parquet"}'
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW citations AS
        SELECT * FROM read_parquet(
            '{inspect_dir / "table_citations.parquet"}'
        )
        """
    )

    for split, limit in split_limits.items():
        target = sample_target(
            inspect_dir / f"target_{split}.parquet",
            limit,
            args.seed,
        )

        combined = build_split_features(
            con=con,
            target=target,
            split=split,
        )

        out = out_dir / f"target_with_dfs_agg_{split}.parquet"
        combined.to_parquet(out, index=False)

        feature_cols = [
            c for c in combined.columns
            if c not in {
                "__row_id",
                "date",
                "Author_ID",
                "primary_category",
            }
        ]

        print("\n" + "=" * 100)
        print(split)
        print("saved:", out)
        print("shape:", combined.shape)
        print("n_features:", len(feature_cols))
        print(
            "duplicate row ids:",
            combined["__row_id"].duplicated().sum(),
        )

        if split != "test":
            print(
                "n_classes:",
                combined["primary_category"].nunique(),
            )

        print("features:")
        for c in feature_cols:
            print(" ", c)

        print("non-null rates:")
        print(
            combined[feature_cols]
            .notna()
            .mean()
            .sort_values()
            .to_string()
        )

    con.close()


if __name__ == "__main__":
    main()
