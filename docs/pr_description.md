## Summary

This PR reconciles the appendix-level validation-gate outcomes with the strict seed-level gate policy and fixes an inconsistency in the rel-amazon/item-churn paper summary.

The main changes are:

- Recomputed strict DFS-to-FDHG gate outcomes for the 18 tasks with complete paired seed-level artifacts over seeds 41–44.
- Reconciled the 51-task appendix inventory to:
  - 24 SELECT
  - 26 FALLBACK
  - 1 NOT_EVALUATED
- Corrected four stale archived gate labels:
  - rel-stack/user-badge: FALLBACK → SELECT
  - rel-salt/sales-office: SELECT → FALLBACK
  - rel-salt/sales-payterms: SELECT → FALLBACK
  - rel-salt/sales-shipcond: SELECT → FALLBACK
- Retained rel-f1/driver-dnf as FALLBACK because its archived decision compares against a stronger temporal baseline rather than DFS alone.
- Added deterministic appendix gate reconciliation and provenance artifacts.
- Integrated appendix reconciliation into the paper-table reproduction pipeline.
- Fixed rel-amazon/item-churn so that both DFS and FDHG are summarized over the same four seeds rather than comparing one DFS seed against four FDHG seeds.
- Updated the README benchmark summary to match the reconciled appendix inventory.

## Appendix gate reconciliation

The final 51-task inventory is:

- 24 validation-gated FDHG selections
- 26 fallbacks to DFS or a stronger task-specific baseline
- 1 task not evaluated

Complete paired four-seed artifacts were available for 18 tasks:

- 17 decisions use the strict DFS-to-FDHG seed-level audit.
- 1 decision, rel-f1/driver-dnf, retains a stronger task-specific temporal baseline.
- The remaining 33 tasks retain their archived inventory decisions.

The strict DFS-to-FDHG audit over the 18 fully reconstructed tasks yielded:

- 8 SELECT
- 10 FALLBACK

After preserving the stronger temporal-baseline decision for Driver-DNF, the reconciled audited subset contains:

- 7 SELECT
- 11 FALLBACK

## Item-churn correction

The previous main table used only seed 41 for the rel-amazon/item-churn DFS summary, while FDHG used seeds 41–44.

The corrected paired four-seed results are:

| Variant | Accuracy | AUROC | Average Precision | Log Loss |
|---|---:|---:|---:|---:|
| DFS | 0.671250 | 0.757267 | 0.631661 | 0.603545 |
| FDHG dmax1 | 0.675500 | 0.758332 | 0.632755 | 0.602888 |

FDHG therefore improves:

- Accuracy by 0.004250
- AUROC by 0.001065
- Average precision by 0.001094
- Log loss by 0.000658

## Reproducibility

Added:

- `scripts/analysis/build_appendix_gate_reconciliation.py`
- `results/final_tables/appendix_strict_gate_seed_metrics.csv`
- `results/final_tables/appendix_strict_gate_metric_coverage.csv`
- `results/final_tables/appendix_strict_dfs_fdhg_gate_audit.csv`
- `results/final_tables/appendix_strict_dfs_fdhg_seed_audit.csv`
- `results/final_tables/appendix_strict_gate_reconciliation.csv`
- `results/final_tables/appendix_51task_gate_reconciled.csv`

The appendix reconciliation script is now called from:

- `scripts/reproduce/reproduce_paper_tables.sh`

## Validation

The following checks pass:

- Full paper-table reproduction
- Appendix reconciliation assertions
- 51 unique task inventory rows
- 24 SELECT / 26 FALLBACK / 1 NOT_EVALUATED
- 18 strict-audited / 33 legacy-only tasks
- 17 strict-seed provenance / 1 stronger-baseline provenance
- rel-amazon/item-churn four-seed parity between DFS and FDHG
- 13 validation-gate unit tests
