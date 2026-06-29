# Reproducibility Status Report

## Executive finding

The second development-repository bundle recovered the experiment scripts and compact raw metrics that were missing from the first public snapshot. The repository can now rebuild the curated paper tables from committed compact inputs, and the broader dmax2, temporal, budget, random-control, synthetic, and uniqueness-stress results are traceable to concrete scripts.

The remaining issue is software consolidation, not absence of evidence. Several recovered scripts are development-era standalone programs rather than modules in the canonical `src/fdhg/` package.

## Resolved gaps

The following previously missing inputs are now present:

- `results/final_tables/final_all_runs.csv`
- `results/final_tables/final_failure_log.csv`
- RateBeer and SALT validation-gate sweep summaries
- `results/final_tables/rel_arxiv_extension_task_summary.csv`
- synthetic program-recovery CSVs
- synthetic oracle-gap result files

The following original implementations were recovered:

- dmax2 generation
- supervised top-K ranking
- random same-budget residual generation
- feature-budget sweeps
- cutoff-safe temporal features and temporal diagnostics
- uniqueness-penalty stress tests
- synthetic data generation, program recovery, ranker, and oracle-gap evaluation
- baseline comparison and broader sweep aggregation scripts

## Remaining high-priority work

### 1. Canonical module refactor

Move stable shared logic from `experiments/research_archive/scripts/` into modules such as:

```text
src/fdhg/
├── discovery/dmax2.py
├── features/temporal.py
├── features/random_control.py
├── selection/topk.py
├── synthetic/generators.py
└── analysis/subgroups.py
```

The archive should remain unchanged as provenance.

### 2. Remove machine-specific assumptions

Recovered scripts should be checked for:

- absolute paths
- assumed cache locations
- hard-coded dataset/task names
- reliance on pre-existing intermediate Parquets
- implicit environment variables

### 3. Add canonical experiment manifests

Each paper experiment should record:

- dataset and task
- source artifact paths
- seed list
- feature budget
- dmax setting
- model and primary metric
- gate rule
- expected output files

### 4. Add smoke tests

Small synthetic tests should cover:

- dmax2 candidate construction
- train-only top-K selection
- random same-budget dimensionality
- temporal cutoff safety
- uniqueness penalty behavior
- program-recovery output schema

## Artifact policy

Commit:

- source code and configurations
- compact per-seed CSV/JSON metrics
- selected-feature manifests
- paper tables

Do not commit:

- RelBench databases
- model caches
- large intermediate Parquet matrices
- credentials
- duplicated logs and backups
