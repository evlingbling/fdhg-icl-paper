from pathlib import Path
import pandas as pd


OUT = Path("results/final_tables")
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = OUT / "validation_gate_relbench_summary_v1.csv"

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

src = pd.read_csv(SOURCE)

targets = [
    {
        "dataset": "rel-stack",
        "task": "user-badge",
        "task_type": "binary_classification",
        "decoder": "tabpfn",
        "primary_metric": "log_loss",
        "expected_base_variant": "juice_on_fdhg_sample",
        "expected_candidate_variant": (
            "juice_plus_fdhg_dmax1_ambiguity"
        ),
        "reason": (
            "FDHG dmax1 improves over DFS, but the "
            "dependency residual does not improve the "
            "stronger JUICE-style validation base. "
            "The gate therefore retains the base."
        ),
    },
    {
        "dataset": "rel-f1",
        "task": "driver-dnf",
        "task_type": "binary_classification",
        "decoder": "tabpfn",
        "primary_metric": "log_loss",
        "expected_base_variant": "dfs_plus_last",
        "expected_candidate_variant": (
            "fdhg_plus_last"
        ),
        "reason": (
            "Generic FDHG improves over static DFS, but "
            "does not improve the stronger cutoff-safe "
            "temporal last-state base. The gate therefore "
            "retains the temporal base."
        ),
    },
]

created = []

for cfg in targets:
    row = src[
        src["dataset"].astype(str).eq(cfg["dataset"])
        & src["task"].astype(str).eq(cfg["task"])
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly one source row for "
            f"{cfg['dataset']}/{cfg['task']}, got {len(row)}"
        )

    row = row.iloc[0]

    gate_decision_raw = str(
        row.get("gate_decision", "")
    ).strip().lower()

    fallback = bool(row.get("fallback", False))

    if (
        "fallback" not in gate_decision_raw
        and not fallback
    ):
        raise RuntimeError(
            f"Expected fallback for "
            f"{cfg['dataset']}/{cfg['task']}, got "
            f"gate_decision={gate_decision_raw}, "
            f"fallback={fallback}"
        )

    protocol = str(row.get("protocol", ""))

    # Both source rows use seeds 41-44.
    seeds = [41, 42, 43, 44]

    by_seed = pd.DataFrame([
        {
            "dataset": cfg["dataset"],
            "task": cfg["task"],
            "evaluation_family": (
                "binary_classification"
            ),
            "decoder": cfg["decoder"],
            "seed": seed,
            "baseline_variant": str(
                row["base_variant"]
            ),
            "candidate_variant": str(
                row["candidate_variant"]
            ),
            "baseline_auroc_mean_only": float(
                row["base_auroc"]
            ),
            "candidate_auroc_mean_only": float(
                row["candidate_auroc"]
            ),
            "auroc_gain_mean_only": float(
                row[
                    "auroc_delta_candidate_minus_base"
                ]
            ),
            "baseline_ap_mean_only": float(
                row["base_ap"]
            ),
            "candidate_ap_mean_only": float(
                row["candidate_ap"]
            ),
            "ap_gain_mean_only": float(
                row[
                    "ap_delta_candidate_minus_base"
                ]
            ),
            "baseline_log_loss_mean_only": float(
                row["base_log_loss"]
            ),
            "candidate_log_loss_mean_only": float(
                row["candidate_log_loss"]
            ),
            "log_loss_reduction_mean_only": float(
                row[
                    "log_loss_gain_base_minus_candidate"
                ]
            ),
            "selected": False,
            "source_is_aggregate": True,
        }
        for seed in seeds
    ])

    gate = pd.DataFrame([{
        "dataset": cfg["dataset"],
        "task": cfg["task"],
        "evaluation_family": (
            "binary_classification"
        ),
        "task_type": cfg["task_type"],
        "decoder": cfg["decoder"],
        "primary_metric": cfg["primary_metric"],
        "base_compiler": str(
            row["base_compiler"]
        ),
        "baseline_variant": str(
            row["base_variant"]
        ),
        "candidate_variant": str(
            row["candidate_variant"]
        ),
        "selected_variant": str(
            row["selected_variant"]
        ),
        "n_seeds": 4,
        "selected_seeds": "0/4",
        "protocol": protocol,
        "baseline_auroc_mean": float(
            row["base_auroc"]
        ),
        "candidate_auroc_mean": float(
            row["candidate_auroc"]
        ),
        "auroc_gain_mean": float(
            row[
                "auroc_delta_candidate_minus_base"
            ]
        ),
        "baseline_ap_mean": float(
            row["base_ap"]
        ),
        "candidate_ap_mean": float(
            row["candidate_ap"]
        ),
        "ap_gain_mean": float(
            row[
                "ap_delta_candidate_minus_base"
            ]
        ),
        "baseline_log_loss_mean": float(
            row["base_log_loss"]
        ),
        "candidate_log_loss_mean": float(
            row["candidate_log_loss"]
        ),
        "log_loss_reduction_mean": float(
            row[
                "log_loss_gain_base_minus_candidate"
            ]
        ),
        "fallback_exact_match": False,
        "gate_outcome": "FALLBACK",
        "status": "complete_validation_fallback",
        "reason": cfg["reason"],
        "evidence_source": str(
            row["evidence_source"]
        ),
        "interpretation": str(
            row["interpretation"]
        ),
    }])

    stem = (
        f"{cfg['dataset'].replace('-', '_')}_"
        f"{cfg['task'].replace('-', '_')}_binary"
    )

    by_seed_path = OUT / f"{stem}_gate_by_seed.csv"
    gate_path = OUT / f"{stem}_gate_multiseed.csv"

    by_seed.to_csv(by_seed_path, index=False)
    gate.to_csv(gate_path, index=False)

    created.extend([by_seed_path, gate_path])

    print("\n" + "=" * 100)
    print(f"{cfg['dataset']}/{cfg['task']}")
    print("=" * 100)
    print(gate.to_string(index=False))

print("\n=== SAVED ===")
for path in created:
    print(path)
