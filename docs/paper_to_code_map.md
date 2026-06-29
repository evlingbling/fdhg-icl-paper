# Paper-to-Code Map

This document maps paper claims and result tables to the code and artifacts in this repository.

Status labels:

- **Reproducible with external data/model dependencies**: generator and configuration are present; RelBench or decoder assets must be installed separately.
- **Recovered research implementation**: the original experiment script and compact outputs are present under `experiments/research_archive/`, but the script has not yet been refactored into the canonical package.
- **Curated artifact only**: a final table exists but the full generating path still requires an uncommitted large intermediate artifact.

## Core configured RelBench path

| Component | Status | Implementation |
|---|---|---|
| RelBench loading and raw export | Reproducible with external data | `src/fdhg/data/relbench_loader.py`, `src/fdhg/data/inspect_bundle.py` |
| Configured leakage-safe aggregation | Reproducible with external data | `scripts/prepare/build_relbench_features.py` |
| dmax1 AFD discovery | Reproducible with external data | `scripts/prepare/compute_afd_dmax1.py` |
| dmax1 ambiguity residuals | Reproducible with external data | `scripts/prepare/build_fdhg_ambiguity.py` |
| Binary, multiclass, and regression evaluation | Reproducible with decoder dependencies | `scripts/evaluate/` |
| Configured end-to-end runner | Partially reproducible | `src/fdhg/cli/reproduce.py`, `configs/reproduction/tasks.yaml` |
| Gate utility | Reproducible | `src/fdhg/gate.py` |

## Recovered broader experiment code

| Paper component | Recovered implementation | Compact evidence |
|---|---|---|
| dmax2 feature generation | `experiments/research_archive/scripts/build_relbench_dmax2_features_v2.py` | `results/research_archive/extension_b_dmax2/` |
| Supervised dmax2 top-K | `experiments/research_archive/scripts/rank_dmax2_programs_supervised.py` | dmax2 manifests and per-seed JSON under `results/research_archive/extension_b_dmax2/` |
| Random residual same-budget | `experiments/research_archive/scripts/add_random_same_budget_features.py` | `results/research_archive/phase1_item_churn_random_residual_same_budget_eval/` |
| Feature-budget curve | `experiments/research_archive/scripts/run_fdhg_budget_curve_tabpfn.py`, `run_relstack_budget_curve_tabpfn.py`, `build_relstack_budget_curve_summary.py` | `results/research_archive/budget_curve/` and curated paper table |
| Cutoff-safe temporal features | `experiments/research_archive/scripts/build_temporal_features.py` | `results/research_archive/phase2_temporal/` |
| Temporal leakage stress | `experiments/research_archive/scripts/run_temporal_leakage_trap.py` | curated temporal leakage table |
| Uniqueness-penalty stress | `experiments/research_archive/scripts/run_uniqueness_penalty_stress.py` | `results/research_archive/uniqueness_stress/` |
| Synthetic relational generator | `experiments/research_archive/scripts/generate_minimal_synthetic_prior.py` | `results/synthetic_prior/` |
| Synthetic program recovery | `experiments/research_archive/scripts/evaluate_synthetic_program_recovery.py` | `results/synthetic_prior/program_recovery_k*.csv` |
| Feature ranker and oracle gap | `experiments/research_archive/scripts/train_feature_ranker_and_oracle_gap.py` | `outputs/synthetic_tabpfn_oracle_gap/seed53_60_top4_with_random/` |
| Arxiv extension and cold/warm integration | `experiments/research_archive/scripts/integrate_rel_arxiv_results.py` | `results/final_tables/rel_arxiv_extension_task_summary.csv` and paper tables |
| GBDT/RDBLearn/JUICE-style comparisons | `experiments/research_archive/baselines_repro/` and related archive scripts | compact final tables |

## Paper table regeneration

The raw compact inputs required by the paper-table builders are now present:

- `results/final_tables/final_all_runs.csv`
- `results/final_tables/final_failure_log.csv`
- `results/final_tables/rel_arxiv_extension_task_summary.csv`
- `results/sweeps/relbench_v2_ratebeer_gate_summary/`
- `results/sweeps/relbench_v2_salt_gate_summary/`
- `results/synthetic_prior/program_recovery_k*.csv`
- `outputs/synthetic_tabpfn_oracle_gap/seed53_60_top4_with_random/`

Run:

```bash
bash scripts/reproduce/reproduce_paper_tables.sh
```

Then audit:

```bash
python scripts/reproduce/audit_paper_artifacts.py
```

## Remaining engineering gap

The broader experiments are now traceable and their original scripts are preserved, but they are not yet all exposed through one canonical CLI. Some recovered scripts still contain development-era path assumptions and depend on large feature Parquets that should be regenerated rather than committed. The next refactor should extract shared dmax2, temporal, selection, and synthetic utilities into `src/fdhg/` while keeping the archive unchanged for provenance.
