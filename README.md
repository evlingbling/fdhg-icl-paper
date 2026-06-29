# FDHG-ICL

FDHG-ICL is a schema-guided feature compiler for relational prediction with frozen tabular models.

It augments a strong relational aggregation baseline with feature programs derived from functional dependencies (FDs), approximate functional dependencies (AFDs), foreign-key structure, dependency violations, and task-specific relational structure. A validation gate selects FDHG residual features only when they improve the primary validation metric; otherwise, the pipeline returns the baseline features unchanged.

## Pipeline

1. Inspect the relational schema and prediction target.
2. Build leakage-safe relational aggregations.
3. Discover candidate FDs and AFDs using training data only.
4. Construct dependency-aware residual and ambiguity features.
5. Evaluate a frozen or lightweight tabular decoder.
6. Apply validation-gated selection or exact fallback to the baseline.

## Benchmark summary

The completed experiment checklist contains 51 relational prediction targets:

* 26 validation-gated FDHG selections
* 24 exact baseline fallbacks
* 1 target marked not evaluated
* 0 incomplete targets

The benchmark includes binary classification, multiclass classification, and regression tasks.

Curated result tables are available under:

```text
results/paper_tables/
```

The complete target-level checklist is:

```text
results/paper_tables/full_target_experiment_checklist.csv
```

## Requirements

* Python 3.10
* Linux recommended for full RelBench and TabPFN experiments
* RelBench datasets are downloaded separately and are not committed to this repository

## Installation

### Minimal development environment

Using micromamba:

```bash
git clone https://github.com/evlingbling/fdhg-icl-paper.git
cd fdhg-icl-paper

micromamba create -n fdhg-repro python=3.10 -y
micromamba activate fdhg-repro

python -m pip install --upgrade pip
python -m pip install -e ".[dev,relbench]"
```

Alternatively, use the provided environment file:

```bash
micromamba create -f environment.yml
micromamba activate fdhg-repro
```

### Optional dependencies

Install RelBench support:

```bash
python -m pip install -e ".[dev,relbench]"
```

Install gradient-boosted tree baselines:

```bash
python -m pip install -e ".[dev,relbench,gbdt]"
```

Install TabPFN support:

```bash
python -m pip install -e ".[dev,relbench,tabpfn]"
```

Install all optional components:

```bash
python -m pip install -e ".[dev,relbench,gbdt,tabpfn]"
```

The exact package versions used to validate the minimal environment are recorded in:

```text
requirements-repro.txt
```

## Quick start

Inspect a RelBench dataset and task:

```bash
fdhg-inspect \
  --dataset rel-stack \
  --task user-badge \
  --out-dir outputs/inspect
```

Show command-line options:

```bash
fdhg-inspect --help
```

## Reproduction examples

Representative reproduction scripts are provided under:

```text
scripts/reproduce/
```

Binary classification example:

```bash
bash scripts/reproduce/reproduce_binary_example.sh
```

Regression example:

```bash
bash scripts/reproduce/reproduce_regression_example.sh
```

Fallback example:

```bash
bash scripts/reproduce/reproduce_fallback_example.sh
```

These scripts are intended to demonstrate the preparation, evaluation, and validation-gate workflow. Full experiments may require downloaded RelBench data, cached feature artifacts, and optional model dependencies.

## Validation gate

Let the baseline feature matrix be (Z_{\mathrm{base}}), and let (Z_{\Delta}) contain dependency-aware residual candidates.

FDHG evaluates:

```text
baseline:       Z_base
FDHG candidate: [Z_base, Z_delta]
```

The candidate is selected only when the configured primary validation metric improves under the required evaluation seeds. Otherwise, the pipeline falls back to the baseline.

A fallback is exact: the selected feature matrix and prediction configuration revert to the baseline rather than retaining non-selected residual columns.

## Repository structure

```text
fdhg-icl-paper/
├── configs/
│   ├── benchmark_tasks.csv
│   └── main.yaml
├── results/
│   └── paper_tables/
├── scripts/
│   ├── analysis/
│   ├── evaluate/
│   ├── prepare/
│   └── reproduce/
├── src/
│   └── fdhg/
│       ├── cli/
│       ├── data/
│       └── gate.py
├── tests/
│   ├── integration/
│   └── unit/
├── environment.yml
├── requirements-repro.txt
└── pyproject.toml
```

## Testing

Run unit tests:

```bash
python -m pytest tests/unit -q
```

Run the complete test suite:

```bash
python -m pytest -q
```

The integration tests that require generated experiment artifacts are skipped when those artifacts are not present.

Validate installed dependencies:

```bash
python -m pip check
```

Expected minimal test status:

```text
4 passed, 9 skipped
```

## Result tables

Important curated tables include:

```text
results/paper_tables/paper_main_summary.csv
results/paper_tables/paper_ablation_summary.csv
results/paper_tables/paper_diagnostic_summary.csv
results/paper_tables/method_comparison_table_paper.csv
results/paper_tables/relbench_v2_fdhg_gate_paper_table.csv
results/paper_tables/relbench_v2_fdhg_gate_outcome_summary.csv
results/paper_tables/full_target_experiment_checklist.csv
```


## End-to-End Reproduction

Install the RelBench and TabPFN dependencies before running full reproduction:

```bash
python -m pip install -e ".[dev,relbench,tabpfn]"
```


Run any configured task with the same command:

```bash
fdhg-reproduce \
  --dataset <dataset> \
  --task <task> \
  --device cpu
```

Preparation only:

```bash
fdhg-reproduce \
  --dataset <dataset> \
  --task <task> \
  --prepare-only
```

Rebuild from raw RelBench data:

```bash
fdhg-reproduce \
  --dataset <dataset> \
  --task <task> \
  --device cpu \
  --force
```

### Supported examples

#### Binary classification

```bash
fdhg-reproduce \
  --dataset rel-event \
  --task event_interest-interested \
  --device cpu
```

| Variant | Features | Accuracy | AUROC | AP | Log Loss |
|---|---:|---:|---:|---:|---:|
| DFS | 2 | 0.731343 | 0.461327 | 0.247115 | 0.583912 |
| FDHG dmax1 | 10 | 0.731343 | 0.526927 | 0.274117 | 0.581789 |

#### Regression

```bash
fdhg-reproduce \
  --dataset rel-hm \
  --task transactions-price \
  --device cpu
```

| Variant | Features | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| DFS | 4 | 0.024492 | 0.014646 | 0.005122 |
| FDHG dmax1 | 5 | 0.023939 | 0.014225 | 0.049580 |

#### Validation-gated fallback

```bash
fdhg-reproduce \
  --dataset rel-ratebeer \
  --task beer_ratings-total_score \
  --device cpu
```

| Variant | Features | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| DFS | 8 | 0.565587 | 0.385917 | 0.331664 |
| FDHG fallback | 8 | 0.565587 | 0.385917 | 0.331664 |

Task-specific configurations are stored in `configs/reproduction/tasks.yaml`.
Generated artifacts are written under `outputs/e2e/<dataset>_<task>/` and `results/<dataset>_<task>_tabpfn/`.

## Paper-to-code reproducibility map

The repository contains both executable reproduction paths and curated tables imported from the broader development project. These are not currently equivalent in coverage.

Before claiming full paper-table reproducibility, run:

```bash
python scripts/reproduce/audit_paper_artifacts.py
```

The audit reports which raw metric files and experiment generators are present. The detailed mapping is documented in:

```text
docs/paper_to_code_map.md
docs/reproducibility_gap_report.md
configs/experiments/paper_artifact_manifest.yaml
```

To regenerate all paper tables after restoring the declared raw inputs:

```bash
bash scripts/reproduce/reproduce_paper_tables.sh
```

At the time this map was added, the configured dmax1 RelBench path was executable with external data and model dependencies, while several broader dmax2, feature-budget, synthetic, temporal-diagnostic, and subgroup results were available only as curated tables or aggregation references. See the mapping document for the exact status of each claim.

## Recovered broader experiment archive

The broader research scripts recovered from the development repository are preserved under:

```text
experiments/research_archive/
```

This archive includes the original dmax2, supervised top-K, random same-budget, temporal, feature-budget, synthetic, oracle-gap, and uniqueness-stress experiment scripts. Compact raw metrics needed to rebuild the paper tables are committed; large RelBench caches and intermediate Parquet matrices are not.

See:

```text
docs/paper_to_code_map.md
docs/reproducibility_gap_report.md
```

Rebuild and audit the compact paper artifacts with:

```bash
bash scripts/reproduce/reproduce_paper_tables.sh
python scripts/reproduce/audit_paper_artifacts.py
```
