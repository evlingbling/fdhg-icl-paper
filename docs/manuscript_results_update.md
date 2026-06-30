# Manuscript Results Update

## Main benchmark summary

Across the 51-task benchmark inventory, FDHG was selected for 24 tasks, while 26 tasks fell back to DFS or a stronger task-specific baseline, and one task was not evaluated. Complete paired four-seed artifacts were normalized for 18 tasks. Among these, 17 decisions were reconstructed using the strict seed-level DFS-to-FDHG validation gate, while Driver-DNF retained its archived fallback decision because the relevant comparison used a stronger temporal baseline rather than DFS alone. The remaining 33 tasks retain their archived inventory decisions.

Strict seed-level reconciliation revised four archived labels. User-Badge changed from FALLBACK to SELECT because FDHG improved AUROC on every required seed. Sales-Office and Sales-Payterms changed from SELECT to FALLBACK because one or more seeds tied the baseline, which does not satisfy the strict-improvement rule. Sales-Shipcond also changed to FALLBACK because the primary accuracy metric regressed on one seed; improvements in the secondary MRR metric cannot override a primary-metric regression under the lexicographic gate.

## rel-amazon/item-churn

For rel-amazon/item-churn, the corrected main comparison uses matched seeds 41–44 for both DFS and FDHG. FDHG increased mean accuracy from 0.6713 to 0.6755, AUROC from 0.7573 to 0.7583, and average precision from 0.6317 to 0.6328, while reducing log loss from 0.6035 to 0.6029. This corresponds to gains of 0.0043 in accuracy, 0.0011 in AUROC, and 0.0011 in average precision, together with a log-loss reduction of 0.0007.

## Appendix caption

The appendix reports outcomes for 51 relational prediction tasks: 24 SELECT, 26 FALLBACK, and one NOT_EVALUATED. Complete paired four-seed artifacts were available for 18 tasks. Seventeen were reconciled using the strict seed-level DFS-to-FDHG gate, while Driver-DNF retained a stronger temporal-baseline fallback. The remaining 33 tasks retain their archived inventory decisions.
