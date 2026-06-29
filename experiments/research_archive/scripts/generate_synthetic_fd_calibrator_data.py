import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


PROFILES = {
    "train_base": {
        "fd_noise_choices": [0.00, 0.02, 0.05, 0.10, 0.20],
        "missing_choices": [0.00, 0.01, 0.03, 0.05],
        "n_zip": 80,
        "n_city": 25,
        "n_sku": 250,
        "n_group": 70,
        "extra_surrogate_cols": 0,
        "skew": False,
    },
    "test_high_noise": {
        "fd_noise_choices": [0.25, 0.30, 0.35, 0.40, 0.45],
        "missing_choices": [0.00, 0.03, 0.05, 0.08],
        "n_zip": 80,
        "n_city": 25,
        "n_sku": 250,
        "n_group": 70,
        "extra_surrogate_cols": 0,
        "skew": False,
    },
    "test_high_cardinality": {
        "fd_noise_choices": [0.02, 0.05, 0.10, 0.20],
        "missing_choices": [0.00, 0.01, 0.03, 0.05],
        "n_zip": 400,
        "n_city": 120,
        "n_sku": 1200,
        "n_group": 350,
        "extra_surrogate_cols": 0,
        "skew": False,
    },
    "test_missing_skew": {
        "fd_noise_choices": [0.05, 0.10, 0.20, 0.30],
        "missing_choices": [0.10, 0.15, 0.20, 0.30],
        "n_zip": 100,
        "n_city": 30,
        "n_sku": 300,
        "n_group": 90,
        "extra_surrogate_cols": 0,
        "skew": True,
    },
    "test_surrogate_trap": {
        "fd_noise_choices": [0.00, 0.02, 0.05, 0.10, 0.20],
        "missing_choices": [0.00, 0.01, 0.03, 0.05],
        "n_zip": 80,
        "n_city": 25,
        "n_sku": 250,
        "n_group": 70,
        "extra_surrogate_cols": 4,
        "skew": False,
    },
}


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def sample_domain(domain, size, rng, skew=False):
    if not skew:
        return rng.choice(domain, size=size, replace=True)
    ranks = np.arange(1, len(domain) + 1)
    probs = 1 / ranks
    probs = probs / probs.sum()
    return rng.choice(domain, size=size, replace=True, p=probs)


def maybe_corrupt_values(values, domain, noise_rate, rng, skew=False):
    values = np.asarray(values, dtype=object).copy()
    if noise_rate <= 0:
        return values
    mask = rng.random(len(values)) < noise_rate
    if mask.sum() > 0:
        values[mask] = sample_domain(domain, mask.sum(), rng, skew=skew)
    return values


def maybe_missing(values, missing_rate, rng):
    values = np.asarray(values, dtype=object).copy()
    if missing_rate <= 0:
        return values
    mask = rng.random(len(values)) < missing_rate
    values[mask] = None
    return values


def generate_one_table(task_id: int, n_rows: int, rng: np.random.Generator, profile: str) -> Tuple[pd.DataFrame, Dict]:
    cfg = PROFILES[profile]
    skew = cfg["skew"]

    zips = np.array([f"zip_{i:04d}" for i in range(cfg["n_zip"])])
    cities = np.array([f"city_{i:04d}" for i in range(cfg["n_city"])])
    regions = np.array([f"region_{i:02d}" for i in range(max(6, cfg["n_city"] // 5))])

    skus = np.array([f"sku_{i:05d}" for i in range(cfg["n_sku"])])
    categories = np.array([f"cat_{i:03d}" for i in range(max(20, cfg["n_sku"] // 60))])
    brands = np.array([f"brand_{i:03d}" for i in range(max(30, cfg["n_sku"] // 50))])

    groups = np.array([f"group_{i:04d}" for i in range(cfg["n_group"])])
    risk_types = np.array(["low", "mid", "high", "very_high"])

    zip_to_city = {z: rng.choice(cities) for z in zips}
    city_to_region = {c: rng.choice(regions) for c in cities}
    sku_to_category = {s: rng.choice(categories) for s in skus}
    sku_to_brand = {s: rng.choice(brands) for s in skus}
    group_to_risk = {g: rng.choice(risk_types, p=[0.45, 0.30, 0.18, 0.07]) for g in groups}

    zip_col = sample_domain(zips, n_rows, rng, skew=skew)
    city_clean = np.array([zip_to_city[z] for z in zip_col])
    region_clean = np.array([city_to_region[c] for c in city_clean])

    sku_col = sample_domain(skus, n_rows, rng, skew=skew)
    category_clean = np.array([sku_to_category[s] for s in sku_col])
    brand_clean = np.array([sku_to_brand[s] for s in sku_col])

    group_col = sample_domain(groups, n_rows, rng, skew=skew)
    risk_clean = np.array([group_to_risk[g] for g in group_col])

    fd_noise = float(rng.choice(cfg["fd_noise_choices"]))
    missing_rate = float(rng.choice(cfg["missing_choices"]))

    city_noisy = maybe_missing(maybe_corrupt_values(city_clean, cities, fd_noise, rng, skew=skew), missing_rate, rng)
    region_noisy = maybe_missing(maybe_corrupt_values(region_clean, regions, fd_noise, rng, skew=skew), missing_rate, rng)
    category_noisy = maybe_missing(maybe_corrupt_values(category_clean, categories, fd_noise, rng, skew=skew), missing_rate, rng)
    brand_noisy = maybe_missing(maybe_corrupt_values(brand_clean, brands, fd_noise, rng, skew=skew), missing_rate, rng)
    risk_noisy = maybe_missing(maybe_corrupt_values(risk_clean, risk_types, fd_noise, rng, skew=skew), missing_rate, rng)

    row_id = np.array([f"row_{task_id}_{i}" for i in range(n_rows)])
    user_id = np.array([f"user_{task_id}_{i}" for i in range(n_rows)])

    random_highcard = np.array([f"hc_{x}" for x in rng.integers(0, n_rows * 3, size=n_rows)])
    random_lowcard = np.array([f"lc_{x}" for x in rng.integers(0, 5, size=n_rows)])
    constant_col = np.array(["CONST"] * n_rows)
    noise_col = np.array([f"noise_{x}" for x in rng.integers(0, 100, size=n_rows)])
    accidental_rhs = np.array([f"acc_{x}" for x in rng.integers(0, max(10, n_rows // 2), size=n_rows)])

    df = pd.DataFrame({
        "row_id": row_id,
        "user_id": user_id,
        "zip": zip_col,
        "city": city_noisy,
        "region": region_noisy,
        "sku": sku_col,
        "category": category_noisy,
        "brand": brand_noisy,
        "group_id": group_col,
        "risk_type": risk_noisy,
        "random_highcard": random_highcard,
        "random_lowcard": random_lowcard,
        "constant_col": constant_col,
        "noise_col": noise_col,
        "accidental_rhs": accidental_rhs,
    })

    for j in range(cfg["extra_surrogate_cols"]):
        df[f"quasi_id_{j}"] = np.array([f"qid{j}_{task_id}_{i}" for i in range(n_rows)])
        df[f"quasi_value_{j}"] = np.array([f"qval{j}_{x}" for x in rng.integers(0, n_rows * 2, size=n_rows)])

    true_fds = {
        ("zip", "city"),
        ("zip", "region"),
        ("city", "region"),
        ("sku", "category"),
        ("sku", "brand"),
        ("group_id", "risk_type"),
    }
    declared_fds = {
        ("zip", "city"),
        ("sku", "category"),
    }

    meta = {
        "task_id": task_id,
        "profile": profile,
        "n_rows": n_rows,
        "fd_noise": fd_noise,
        "missing_rate": missing_rate,
        "true_fds": sorted(list(true_fds)),
        "declared_fds": sorted(list(declared_fds)),
    }
    return df, meta


def afd_stats(df: pd.DataFrame, lhs: str, rhs: str, n_bootstrap: int, rng: np.random.Generator) -> Dict:
    sub = df[[lhs, rhs]].copy()
    valid = sub[lhs].notna() & sub[rhs].notna()
    coverage = float(valid.mean())
    sub = sub.loc[valid].copy()
    n = len(sub)

    if n <= 1:
        return {
            "r_pair": 0.0,
            "r_tuple": 0.0,
            "r_del": 0.0,
            "r_ent": 0.0,
            "lhs_collision_rate": 0.0,
            "lhs_uniqueness": 1.0,
            "lhs_support_repeated": 0.0,
            "coverage": coverage,
            "rhs_entropy": 0.0,
            "lhs_arity": 1,
            "bootstrap_stability": 0.0,
        }

    grouped = sub.groupby(lhs, dropna=False)[rhs]
    nx = grouped.size()
    nunique_lhs = int(nx.shape[0])

    max_per_lhs = grouped.value_counts(dropna=False).groupby(level=0).max()
    r_del = float(max_per_lhs.sum() / n)

    rhs_nunique_per_lhs = grouped.nunique(dropna=False)
    non_conflict_lhs = rhs_nunique_per_lhs[rhs_nunique_per_lhs <= 1].index
    non_conflict_rows = int(nx.loc[non_conflict_lhs].sum()) if len(non_conflict_lhs) else 0
    r_tuple = float(non_conflict_rows / n)

    total_pairs = n * (n - 1) / 2
    lhs_pairs = float(((nx * (nx - 1)) / 2).sum())

    same_rhs_pairs = 0.0
    for _, g in sub.groupby([lhs, rhs], dropna=False):
        m = len(g)
        same_rhs_pairs += m * (m - 1) / 2

    violating_pairs = max(lhs_pairs - same_rhs_pairs, 0.0)
    r_pair = float(1.0 - violating_pairs / total_pairs)
    lhs_collision_rate = float(lhs_pairs / total_pairs)

    rhs_counts = sub[rhs].value_counts(dropna=False).values.astype(float)
    h_rhs = entropy_from_counts(rhs_counts)

    h_cond = 0.0
    for _, group in grouped:
        counts = group.value_counts(dropna=False).values.astype(float)
        h_cond += (len(group) / n) * entropy_from_counts(counts)

    if h_rhs <= 1e-12:
        r_ent = 0.0
        rhs_entropy = 0.0
    else:
        r_ent = float(1.0 - h_cond / (h_rhs + 1e-12))
        rhs_entropy = float(h_rhs / (math.log(len(rhs_counts)) + 1e-12)) if len(rhs_counts) > 1 else 0.0

    lhs_uniqueness = float(nunique_lhs / n)
    lhs_support_repeated = float(nx[nx >= 2].sum() / n)

    stable = 0
    if n_bootstrap > 0 and n >= 20:
        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, n, size=n)
            bdf = sub.iloc[sample_idx]
            bg = bdf.groupby(lhs, dropna=False)[rhs]
            bnx = bg.size()
            if len(bnx) == 0:
                continue
            bmax = bg.value_counts(dropna=False).groupby(level=0).max()
            br_del = float(bmax.sum() / len(bdf))
            brepeated = float(bnx[bnx >= 2].sum() / len(bdf))
            buniq = float(len(bnx) / len(bdf))
            if br_del >= 0.80 and brepeated >= 0.10 and buniq <= 0.98:
                stable += 1
        bootstrap_stability = float(stable / n_bootstrap)
    else:
        bootstrap_stability = 0.0

    return {
        "r_pair": r_pair,
        "r_tuple": r_tuple,
        "r_del": r_del,
        "r_ent": r_ent,
        "lhs_collision_rate": lhs_collision_rate,
        "lhs_uniqueness": lhs_uniqueness,
        "lhs_support_repeated": lhs_support_repeated,
        "coverage": coverage,
        "rhs_entropy": rhs_entropy,
        "lhs_arity": 1,
        "bootstrap_stability": bootstrap_stability,
    }


def classify_edge(lhs: str, rhs: str, true_fds: set, declared_fds: set) -> Dict:
    is_true = int((lhs, rhs) in true_fds)
    schema_flag = int((lhs, rhs) in declared_fds)

    if lhs in {"row_id", "user_id"} or lhs.startswith("quasi_id_"):
        edge_type = "surrogate_key_like"
    elif rhs == "constant_col":
        edge_type = "constant_rhs_trivial"
    elif is_true:
        edge_type = "true_structural_fd"
    elif lhs == "random_highcard" or lhs.startswith("quasi_value_"):
        edge_type = "high_cardinality_accidental"
    else:
        edge_type = "negative_candidate"

    return {
        "is_true_structural_fd": is_true,
        "schema_flag": schema_flag,
        "edge_type": edge_type,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(PROFILES.keys()), required=True)
    parser.add_argument("--n_tasks", type=int, default=200)
    parser.add_argument("--min_rows", type=int, default=800)
    parser.add_argument("--max_rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n_bootstrap", type=int, default=5)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_rows = []
    all_meta = []

    for task_id in range(args.n_tasks):
        n_rows = int(rng.integers(args.min_rows, args.max_rows + 1))
        df, meta = generate_one_table(task_id, n_rows, rng, args.profile)
        true_fds = set(tuple(x) for x in meta["true_fds"])
        declared_fds = set(tuple(x) for x in meta["declared_fds"])

        cols = list(df.columns)
        for lhs in cols:
            for rhs in cols:
                if lhs == rhs:
                    continue
                stats = afd_stats(df, lhs, rhs, args.n_bootstrap, rng)
                label_info = classify_edge(lhs, rhs, true_fds, declared_fds)

                row = {
                    "task_id": task_id,
                    "profile": args.profile,
                    "table_name": "synthetic_entity_table",
                    "lhs": lhs,
                    "rhs": rhs,
                    "fd_noise": meta["fd_noise"],
                    "missing_rate": meta["missing_rate"],
                    "n_rows": n_rows,
                }
                row.update(label_info)
                row.update(stats)
                all_rows.append(row)

        all_meta.append(meta)

        if (task_id + 1) % 20 == 0:
            print(f"[{args.profile}] Generated {task_id + 1}/{args.n_tasks}; edges={len(all_rows)}")

    edge_df = pd.DataFrame(all_rows)
    edge_path = out_dir / "edge_dataset.csv"
    edge_df.to_csv(edge_path, index=False)

    with open(out_dir / "synthetic_task_metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    summary = {
        "profile": args.profile,
        "n_tasks": args.n_tasks,
        "n_edges": int(len(edge_df)),
        "n_positive_edges": int(edge_df["is_true_structural_fd"].sum()),
        "positive_rate": float(edge_df["is_true_structural_fd"].mean()),
        "edge_type_counts": edge_df["edge_type"].value_counts().to_dict(),
        "out_csv": str(edge_path),
    }

    with open(out_dir / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GENERATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
