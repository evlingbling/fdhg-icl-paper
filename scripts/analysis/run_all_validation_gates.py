from __future__ import annotations

import subprocess
import sys
from pathlib import Path


TASKS = [
    (
        "results/rel-arxiv_author-category_tabpfn",
        "accuracy",
        "maximize",
        None,
        None,
    ),
    (
        "results/rel-arxiv_paper-citation_tabpfn",
        "log_loss",
        "minimize",
        None,
        None,
    ),
    (
        "results/rel-ratebeer_beer-churn_tabpfn",
        "log_loss",
        "minimize",
        None,
        None,
    ),
    (
        "results/rel-ratebeer_brewer-dormant_tabpfn",
        "log_loss",
        "minimize",
        None,
        None,
    ),
    (
        "results/rel-ratebeer_user-churn_tabpfn",
        "log_loss",
        "minimize",
        None,
        None,
    ),
    (
        "results/rel-ratebeer_user-count_tabpfn",
        "rmse",
        "minimize",
        None,
        None,
    ),
    (
        "results/rel-ratebeer_user-place-liked_pairwise_tabpfn",
        "log_loss",
        "minimize",
        None,
        None,
    ),
    (
        "results/rel-salt_item-incoterms_tabpfn",
        "accuracy",
        "maximize",
        None,
        None,
    ),
    (
        "results/rel-salt_item-plant_tabpfn",
        "accuracy",
        "maximize",
        None,
        None,
    ),
    (
        "results/rel-salt_item-shippoint_tabpfn",
        "accuracy",
        "maximize",
        None,
        None,
    ),
    (
        "results/rel-salt_sales-group_catboost",
        "accuracy",
        "maximize",
        None,
        None,
    ),
    (
        "results/rel-salt_sales-incoterms_tabpfn",
        "accuracy",
        "maximize",
        None,
        None,
    ),
    (
        "results/rel-salt_sales-office_tabpfn",
        "accuracy",
        "maximize",
        None,
        None,
    ),
    (
        "results/rel-salt_sales-payterms_tabpfn",
        "accuracy",
        "maximize",
        None,
        None,
    ),
    (
        "results/rel-salt_sales-shipcond_tabpfn",
        "accuracy",
        "maximize",
        "mrr",
        "maximize",
    ),
]


def main() -> None:
    selector = Path(
        "scripts/analysis/select_validation_gate.py"
    )

    for (
        result_root,
        metric,
        direction,
        secondary_metric,
        secondary_direction,
    ) in TASKS:
        command = [
            sys.executable,
            str(selector),
            "--result-root",
            result_root,
            "--metric",
            metric,
            "--direction",
            direction,
            "--required-seeds",
            "41",
            "42",
            "43",
            "44",
        ]

        if secondary_metric is not None:
            command.extend([
                "--secondary-metric",
                secondary_metric,
                "--secondary-direction",
                secondary_direction,
            ])

        print("\n$", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
