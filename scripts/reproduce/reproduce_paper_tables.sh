#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python scripts/reproduce/audit_paper_artifacts.py
python scripts/analysis/build_paper_tables.py

PYTHONPATH=src python scripts/analysis/select_validation_gate.py \
  --result-root results/rel-ratebeer_beer-churn_tabpfn \
  --metric log_loss \
  --direction minimize \
  --required-seeds 41 42 43 44

PYTHONPATH=src python scripts/analysis/select_validation_gate.py \
  --result-root results/rel-ratebeer_brewer-dormant_tabpfn \
  --metric log_loss \
  --direction minimize \
  --required-seeds 41 42 43 44

PYTHONPATH=src python scripts/analysis/build_validation_gate_table.py
python scripts/analysis/build_synthetic_tables.py

echo "Paper tables regenerated successfully."
