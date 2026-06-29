# Recovered Research Experiment Archive

This directory preserves experiment scripts and configurations recovered from the development repository used during the FDHG-ICL study.

The files are retained with their original names so that paper results can be traced to the code that produced them. They are not yet presented as a fully unified public API. Several scripts expect development-repository paths, cached RelBench datasets, or intermediate Parquet artifacts that are intentionally not committed.

## Main recovered components

- `scripts/build_relbench_dmax2_features_v2.py`: dmax2 relational feature generation.
- `scripts/rank_dmax2_programs_supervised.py`: supervised dmax2 candidate ranking and top-K selection.
- `scripts/add_random_same_budget_features.py`: random residual same-budget control.
- `scripts/build_temporal_features.py`: cutoff-safe temporal aggregates and last-state features.
- `scripts/run_temporal_leakage_trap.py`: temporal leakage stress experiment.
- `scripts/run_fdhg_budget_curve_tabpfn.py` and `scripts/run_relstack_budget_curve_tabpfn.py`: feature-budget experiments.
- `scripts/generate_minimal_synthetic_prior.py`: synthetic relational generator.
- `scripts/evaluate_synthetic_program_recovery.py`: program-recovery evaluation.
- `scripts/train_feature_ranker_and_oracle_gap.py`: learned ranker and oracle-gap experiment.
- `scripts/evaluate_synthetic_tabpfn_oracle_gap_with_random.py`: oracle/random residual evaluation.
- `scripts/run_uniqueness_penalty_stress.py`: surrogate-key uniqueness-penalty stress test.
- `scripts/integrate_rel_arxiv_results.py`: Arxiv extension and cold/warm result integration.

## Publication policy

Canonical, reusable components should gradually be moved into `src/fdhg/` or the normal `scripts/` hierarchy. Until that refactor is complete, this archive is the evidence-preserving source of truth for the broader experiments.

Do not commit downloaded RelBench databases, model caches, or large intermediate feature matrices. Compact CSV/JSON metrics and selected-feature manifests are stored under `results/research_archive/`.
