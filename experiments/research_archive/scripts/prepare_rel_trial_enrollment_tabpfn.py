from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(
    "results/rel_trial_studies_enrollment_matched_seed41"
)
OUT = Path(
    "results/rel_trial_studies_enrollment_tabpfn_parquets"
)
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "dfs_clean",
    "dfs_clean_plus_dmax2",
]
SPLITS = ["train", "val"]


def detect_label(df: pd.DataFrame) -> str:
    preferred = [
        "target",
        "enrollment",
        "label",
        "sales",
    ]

    for col in preferred:
        if col in df.columns:
            return col

    raise RuntimeError(
        "Could not detect label column. "
        f"Columns: {list(df.columns)}"
    )


detected_label = None

for split in SPLITS:
    loaded = {}

    for variant in VARIANTS:
        path = SRC / f"{split}_{variant}.parquet"
        df = pd.read_parquet(path)

        label = detect_label(df)

        if detected_label is None:
            detected_label = label
        elif label != detected_label:
            raise RuntimeError(
                f"Inconsistent labels: "
                f"{detected_label!r} vs {label!r}"
            )

        y = pd.to_numeric(
            df[label],
            errors="coerce",
        )

        if y.isna().any():
            raise RuntimeError(
                f"{path}: label contains "
                f"{int(y.isna().sum())} NaNs"
            )

        if (y < 0).any():
            raise RuntimeError(
                f"{path}: log1p target has negative values"
            )

        df = df.copy()
        df["target_log1p"] = np.log1p(y.astype(float))

        raw_out = OUT / f"{split}_{variant}_raw.parquet"
        log_out = OUT / f"{split}_{variant}_log1p.parquet"

        # Same dataframe is usable for both evaluations:
        # raw run uses the original label,
        # log run uses target_log1p and drops the original label.
        df.to_parquet(raw_out, index=False)
        df.to_parquet(log_out, index=False)

        loaded[variant] = df

        print(
            split,
            variant,
            "shape=", df.shape,
            "label=", label,
            "raw_range=",
            (float(y.min()), float(y.max())),
            "log1p_range=",
            (
                float(df["target_log1p"].min()),
                float(df["target_log1p"].max()),
            ),
        )

    base = loaded["dfs_clean"]
    cand = loaded["dfs_clean_plus_dmax2"]

    assert len(base) == len(cand)
    assert np.array_equal(
        base[detected_label].to_numpy(),
        cand[detected_label].to_numpy(),
    ), f"{split}: target mismatch"

print("\nlabel_col:", detected_label)
print("saved:", OUT)
