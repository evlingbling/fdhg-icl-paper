from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_value(x: Any) -> str:
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
        return np.nan
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum())


def infer_target_table_from_inspect(inspect_dir: Path) -> str:
    # Prefer common RelBench naming from folder/task; fallback to the smallest entity table.
    table_paths = sorted(inspect_dir.glob("table_*.parquet"))
    if not table_paths:
        raise FileNotFoundError(f"No table_*.parquet found in {inspect_dir}")

    # If target key appears in exactly one non-child table, this is handled later by CLI target-table.
    # Fallback: choose smallest table except huge event/review table.
    candidates = []
    for p in table_paths:
        name = p.stem.replace("table_", "")
        df = pd.read_parquet(p)
        candidates.append((name, len(df), p))
    candidates = sorted(candidates, key=lambda x: x[1])
    return candidates[0][0]


def select_edge(
    afd_df: pd.DataFrame,
    *,
    id_columns: set[str],
    target_df: pd.DataFrame,
    min_r_del: float,
    min_coverage: float,
    min_lhs_support_repeated: float,
    max_lhs_uniqueness: float,
) -> pd.DataFrame:
    df = afd_df.copy()

    # RHS should be categorical/object-like for ambiguity features.
    categorical_rhs = []
    for c in target_df.columns:
        if c in id_columns:
            continue
        if target_df[c].dtype == "object" or str(target_df[c].dtype).startswith("category"):
            categorical_rhs.append(c)

    if not categorical_rhs:
        return pd.DataFrame()

    df = df[~df["rhs"].isin(id_columns)].copy()
    df = df[~df["lhs"].isin(id_columns)].copy()
    df = df[df["rhs"].isin(categorical_rhs)].copy()
    df = df[df["lhs"] != df["rhs"]].copy()
    df = df[df["r_del"] >= min_r_del].copy()
    df = df[df["coverage"] >= min_coverage].copy()
    df = df[df["lhs_support_repeated"] >= min_lhs_support_repeated].copy()
    df = df[df["lhs_uniqueness"] < max_lhs_uniqueness].copy()

    if df.empty:
        return df

    raw = (
        1.5 * df["r_del"].astype(float)
        + 1.0 * df["r_ent"].astype(float)
        + 0.5 * df["lhs_support_repeated"].astype(float)
        + 0.3 * df["coverage"].astype(float)
        - 1.0 * df["lhs_uniqueness"].astype(float)
    )
    df["edge_weight"] = 1.0 / (1.0 + np.exp(-(raw - 1.0)))
    df["edge_type"] = "afd"
    df["edge_name"] = df["lhs"].astype(str) + "->" + df["rhs"].astype(str)

    return df.sort_values(["edge_weight", "r_del", "r_ent"], ascending=False).head(1).reset_index(drop=True)


def compute_ambiguity_map(entity_df: pd.DataFrame, lhs: str, rhs: str) -> pd.DataFrame:
    tmp = entity_df[[lhs, rhs]].copy()
    tmp["lhs_norm"] = normalize_series(tmp[lhs])
    tmp["rhs_norm"] = normalize_series(tmp[rhs])
    tmp = tmp[(tmp["lhs_norm"] != "__NULL__") & (tmp["rhs_norm"] != "__NULL__")]

    if tmp.empty:
        return pd.DataFrame(columns=["lhs_norm", "majconf", "entropy", "conflict_count", "support_count"])

    counts = tmp.groupby(["lhs_norm", "rhs_norm"]).size().rename("n").reset_index()

    rows = []
    for lhs_value, g in counts.groupby("lhs_norm"):
        total = int(g["n"].sum())
        max_count = int(g["n"].max())
        p = g["n"].to_numpy(dtype=float) / total
        rows.append({
            "lhs_norm": lhs_value,
            "majconf": max_count / total,
            "entropy": float(-(p * np.log(p + 1e-12)).sum()),
            "conflict_count": int(g["rhs_norm"].nunique()),
            "support_count": total,
        })
    return pd.DataFrame(rows)


def build_split(
    target_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    fit_entity_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    *,
    entity_key: str,
) -> pd.DataFrame:
    out = target_df[[entity_key]].copy().reset_index(drop=True)
    out["__fdhg_row_id"] = range(len(out))

    if edge_df.empty:
        return out

    product_cols = [entity_key] + sorted(set(edge_df["lhs"]).union(set(edge_df["rhs"])))
    lookup = entity_df[product_cols].drop_duplicates(subset=[entity_key]).copy()
    merged = out.merge(lookup, on=entity_key, how="left")

    result = out.copy()

    for _, edge in edge_df.iterrows():
        lhs = str(edge["lhs"])
        rhs = str(edge["rhs"])
        edge_name = f"{lhs}_to_{rhs}"

        amb_map = compute_ambiguity_map(
            fit_entity_df,
            lhs,
            rhs,
        )
        tmp = pd.DataFrame({
            "__fdhg_row_id": merged["__fdhg_row_id"],
            "lhs_norm": normalize_series(merged[lhs]),
        }).merge(amb_map, on="lhs_norm", how="left")

        feat_cols = {
            f"f_amb__{edge_name}__majconf": tmp["majconf"],
            f"f_amb__{edge_name}__entropy": tmp["entropy"],
            f"f_amb__{edge_name}__conflict_count": tmp["conflict_count"],
            f"f_amb__{edge_name}__support_count": tmp["support_count"],
        }

        for name, values in feat_cols.items():
            result[name] = values
            result[f"{name}__is_missing"] = values.isna().astype("int8")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", required=True)
    parser.add_argument("--dfs-dir", required=True)
    parser.add_argument("--afd-stats", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--entity-key", required=True)
    parser.add_argument("--target-table", default=None)
    parser.add_argument(
        "--fit-table",
        default=None,
        help=(
            "Train-only entity table used to fit ambiguity mappings. "
            "If omitted, the full entity table is used for backward "
            "compatibility."
        ),
    )
    parser.add_argument("--min-r-del", type=float, default=0.6)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-lhs-support-repeated", type=float, default=0.05)
    parser.add_argument("--max-lhs-uniqueness", type=float, default=0.98)
    args = parser.parse_args()

    inspect_dir = Path(args.inspect_dir)
    dfs_dir = Path(args.dfs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_table = args.target_table or infer_target_table_from_inspect(inspect_dir)
    entity_path = inspect_dir / f"table_{target_table}.parquet"

    if not entity_path.exists():
        raise FileNotFoundError(f"Could not find entity table: {entity_path}")

    entity_df = pd.read_parquet(entity_path)

    fit_entity_path = (
        Path(args.fit_table)
        if args.fit_table
        else entity_path
    )
    if not fit_entity_path.exists():
        raise FileNotFoundError(
            f"Could not find AFD fit table: {fit_entity_path}"
        )

    fit_entity_df = pd.read_parquet(fit_entity_path)
    afd_df = pd.read_csv(args.afd_stats)

    edge_df = select_edge(
        afd_df,
        id_columns={args.entity_key},
        target_df=fit_entity_df,
        min_r_del=args.min_r_del,
        min_coverage=args.min_coverage,
        min_lhs_support_repeated=args.min_lhs_support_repeated,
        max_lhs_uniqueness=args.max_lhs_uniqueness,
    )

    edge_path = out_dir / "afd_edges_dmax1.csv"
    edge_df.to_csv(edge_path, index=False)

    print("=== FDHG ambiguity fitting scope ===")
    print("entity lookup table:", entity_path)
    print("ambiguity fit table:", fit_entity_path)
    print("entity lookup shape:", entity_df.shape)
    print("ambiguity fit shape:", fit_entity_df.shape)

    print("=== Selected AFD edges ===")
    if edge_df.empty:
        print("No usable AFD edge found. FDHG will equal DFS for this task.")
    else:
        print(edge_df[["lhs", "rhs", "edge_weight", "r_del", "r_ent", "lhs_uniqueness", "lhs_support_repeated", "coverage"]])

    for split in ["train", "val"]:
        target_path = dfs_dir / f"target_with_dfs_agg_{split}.parquet"
        target_df = pd.read_parquet(target_path)

        amb = build_split(
            target_df,
            entity_df,
            fit_entity_df,
            edge_df,
            entity_key=args.entity_key,
        )
        amb_path = out_dir / f"ambiguity_features_{split}.parquet"
        amb.drop(columns=["__fdhg_row_id"]).to_parquet(amb_path, index=False)

        amb_only = amb.drop(columns=["__fdhg_row_id", args.entity_key])
        combined = pd.concat([target_df.reset_index(drop=True), amb_only.reset_index(drop=True)], axis=1)

        combined_path = out_dir / f"target_with_dfs_agg_{split}.parquet"
        combined.to_parquet(combined_path, index=False)

        print(f"\n[{split}]")
        print("target:", target_df.shape)
        print("amb:", amb.shape)
        print("combined:", combined.shape)
        print("saved:", combined_path)

    manifest_rows = []
    if not edge_df.empty:
        for _, edge in edge_df.iterrows():
            lhs, rhs = edge["lhs"], edge["rhs"]
            edge_name = f"{lhs}_to_{rhs}"
            for suffix in ["majconf", "entropy", "conflict_count", "support_count"]:
                manifest_rows.append({
                    "feature_name": f"f_amb__{edge_name}__{suffix}",
                    "block": "ambiguity",
                    "edge_name": f"{lhs}->{rhs}",
                    "lhs": lhs,
                    "rhs": rhs,
                    "edge_weight": edge["edge_weight"],
                    "r_del": edge["r_del"],
                    "r_ent": edge["r_ent"],
                    "lhs_uniqueness": edge["lhs_uniqueness"],
                    "lhs_support_repeated": edge["lhs_support_repeated"],
                    "coverage": edge["coverage"],
                })

    pd.DataFrame(manifest_rows).to_csv(out_dir / "ambiguity_feature_manifest.csv", index=False)

    print("\n=== Done ===")
    print("out_dir:", out_dir)
    print("edge_path:", edge_path)


if __name__ == "__main__":
    main()
