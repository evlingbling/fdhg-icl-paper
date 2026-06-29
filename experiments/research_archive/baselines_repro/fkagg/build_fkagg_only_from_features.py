from pathlib import Path
import argparse
import json
import re
import pandas as pd


DROP_PATTERNS = [
    # FDHG ambiguity / residual features
    r"^f_amb__",
    r"__majconf$",
    r"__entropy$",
    r"__conflict_count$",
    r"__support_count$",

    # dmax2 / second-hop dependency programs
    r"dmax2",
    r"twohop",
    r"2hop",

    # explicit FD / AFD / dependency markers
    r"afd",
    r"fd_score",
    r"fdhg_rank",
    r"ranker",
    r"uniqueness",
    r"surrogate",
    r"residual",
    r"ambiguity",

    # paired missing indicators for dependency-derived features
    r"^f_amb__.*__is_missing$",
]


KEEP_EXACT = {
    "target",
    "timestamp",
    "time",
    "UserId",
    "user",
    "item",
    "item_id",
    "product",
    "product_id",
    "driverId",
    "constructorId",
    "raceId",
    "date",
}


def is_target_or_key_col(col: str) -> bool:
    c = str(col)

    if c in KEEP_EXACT:
        return True

    if c.lower() in {
        "target",
        "label",
        "y",
        "timestamp",
        "time",
        "date",
        "index",
    }:
        return True

    # common key columns
    if re.search(r"(^|_)(id|key)$", c, flags=re.IGNORECASE):
        return True
    if re.search(r"(Id|ID)$", c):
        return True

    return False


def is_dependency_derived_col(col: str) -> bool:
    c = str(col)
    low = c.lower()

    for pat in DROP_PATTERNS:
        if re.search(pat, c, flags=re.IGNORECASE):
            return True

    # Conservative FDHG feature naming: f_amb and dmax are dependency-specific.
    if low.startswith("f_amb__"):
        return True

    return False


def infer_fkagg_columns(cols):
    keep = []
    drop = []

    for col in cols:
        c = str(col)

        if is_target_or_key_col(c):
            keep.append(c)
            continue

        if is_dependency_derived_col(c):
            drop.append(c)
            continue

        # Everything else is treated as base / FK / inverse-FK aggregation.
        keep.append(c)

    return keep, drop


def build_one(input_path, output_path, manifest_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)

    df = pd.read_parquet(input_path)

    keep_cols, drop_cols = infer_fkagg_columns(df.columns)

    out = df[keep_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(output_path, index=False)

    manifest = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "mode": "fdhg_fkagg_only_generic_filter",
        "definition": (
            "Keep base/FK/inverse-FK aggregation columns; drop FD/AFD ambiguity, "
            "dependency residual, uniqueness, dmax2, and ranker-derived columns."
        ),
        "n_input_columns": len(df.columns),
        "n_output_columns": len(out.columns),
        "n_dropped_columns": len(drop_cols),
        "kept_columns": keep_cols,
        "dropped_columns": drop_cols,
        "drop_patterns": DROP_PATTERNS,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps({
        "input_path": str(input_path),
        "output_path": str(output_path),
        "n_input_columns": len(df.columns),
        "n_output_columns": len(out.columns),
        "n_dropped_columns": len(drop_cols),
        "dropped_columns": drop_cols[:20],
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    build_one(args.input, args.output, args.manifest)


if __name__ == "__main__":
    main()
