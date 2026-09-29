#!/usr/bin/env bash
# Regenerate every golden notebook: build (.py -> .ipynb) then execute in place.
# Run from the repo root:  bash scripts/golden/run_all.sh
# Heavy notebooks (06 dominance, 07 limit cycles, 10 linguistic, 11 robustness) take several
# minutes each (B=1000 nulls / cluster bootstraps / 420-cell metric recompute).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

NBS=(
  01_setup_and_datasets
  02_detection_worked_examples
  03_system_performance
  04_self_reinforcement
  05_bistability
  06_dominance
  07_limit_cycles
  08_high_repetition
  09_seeded_causal
  10_linguistic
  11_appendix_robustness
)

for nb in "${NBS[@]}"; do
  echo "=== build $nb ==="
  python "scripts/golden/build_${nb}.py"
done

for nb in "${NBS[@]}"; do
  echo "=== execute $nb ==="
  python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=2400 "notebooks/golden/${nb}.ipynb"
done

echo "All golden notebooks regenerated."
