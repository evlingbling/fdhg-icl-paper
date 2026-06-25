# FDHG-ICL

FDHG-ICL compiles functional dependencies, foreign-key structure, and approximate functional dependencies from normalized relational databases into flat feature programs for frozen tabular prediction models.

## Overview

1. Relational schema inspection
2. Leakage-safe temporal aggregation
3. Approximate functional-dependency discovery
4. Ambiguity residual construction
5. Frozen decoder evaluation
6. Validation-gated selection with exact DFS fallback

## Main result

The completed benchmark contains 51 relational prediction targets:

- 26 validation-gated FDHG selections
- 24 exact DFS fallbacks
- 1 task marked not evaluated
- 0 incomplete targets

## Installation

```bash
micromamba create -n fdhg-icl python=3.10
micromamba activate fdhg-icl
pip install -e .
```

## Quick start

```bash
fdhg-inspect --dataset rel-stack --task user-badge --out-dir outputs/inspect
```

## Results

Curated result tables are stored in `results/paper_tables/`.

## Validation gate

FDHG is selected only when the primary validation metric improves on every evaluation seed. Otherwise, the system falls back exactly to the DFS baseline.
