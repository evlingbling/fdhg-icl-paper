#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python scripts/reproduce/audit_paper_artifacts.py
python scripts/analysis/build_paper_tables.py
python scripts/analysis/build_validation_gate_table.py
python scripts/analysis/build_synthetic_tables.py

echo "Paper tables regenerated successfully."
