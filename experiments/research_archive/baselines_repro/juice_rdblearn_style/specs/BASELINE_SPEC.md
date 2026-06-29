# JUICE / RDBLearn-style Baseline Specification

## Status
Pending implementation / reproduction.

## Purpose
This baseline is required to compare FDHG-ICL against a column-wise relational meta-path aggregation method beyond the current DFS-style aggregation proxy.

## Important interpretation rule
Existing Phase 0 results should be described as comparisons against DFS-style aggregation, not as exact JUICE or RDBLearn-style results.

## Target first task
- Dataset: rel-stack
- Task: user-badge
- Primary decoder: TabPFN
- Additional decoders if feasible: XGBoost, CatBoost

## Matched comparison methods
1. target_only
2. naive
3. dfs_style_aggregation
4. juice_rdblearn_style
5. fdhg_fkagg
6. fdhg_dmax1

## Fair comparison requirements
- Same train/validation/test split.
- Same feature budget K where applicable.
- Same decoder.
- Same preprocessing.
- Same categorical handling.
- Same leakage guard.
- Same evaluation metrics.

## Allowed feature class for JUICE/RDBLearn-style
The baseline may use:
- PK/FK meta-path traversal.
- Inverse-FK aggregation.
- Column-wise aggregations such as count, mean, sum, min, max, std, nunique.
- Same target cutoff / temporal leakage guard as FDHG.

The baseline must not use:
- FD or AFD reliability scores.
- FDHG ambiguity features.
- Uniqueness penalty.
- FDHG ranker or supervised program selector.
- Dependency-derived non-FK hyperpaths.

## Required output columns
Every result row must include:
- dataset
- task
- method
- seed
- split
- decoder
- feature_budget
- n_features
- accuracy
- roc_auc
- average_precision
- log_loss
- runtime_sec
- feature_path
- result_path
- status
- failure_reason

## Paper-use rule
Until this baseline is completed, the paper may say:
"JUICE/RDBLearn-style comparison is pending."

The paper must not say:
"FDHG beats JUICE."
