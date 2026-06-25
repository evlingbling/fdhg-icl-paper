#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEEDS=(41 42 43 44)

run_task () {
  task="$1"
  label="$2"

  result_root="results/rel-event_${task}_tabpfn"
  log_root="logs/rel-event_${task}_tabpfn"

  mkdir -p "$result_root" "$log_root"

  for variant in dfs fdhg_dmax1; do
    if [[ "$variant" == "dfs" ]]; then
      root="outputs/dfs_agg/rel-event_${task}_userkey_sample"
    else
      root="outputs/fdhg_heuristic/rel-event_${task}_sample"
    fi

    for seed in "${SEEDS[@]}"; do
      out="${result_root}/${variant}/seed${seed}"
      log="${log_root}/${variant}_seed${seed}.log"

      mkdir -p "$out"

      echo
      echo "============================================================"
      echo "rel-event/${task} | ${variant} | seed=${seed}"
      echo "============================================================"

      python scripts/run_generic_parquet_tabpfn_eval_with_predictions.py \
        --train-parquet \
        "${root}/target_with_dfs_agg_train.parquet" \
        --val-parquet \
        "${root}/target_with_dfs_agg_val.parquet" \
        --output-dir "$out" \
        --dataset rel-event \
        --task "$task" \
        --variant "$variant" \
        --label-col "$label" \
        --drop-cols timestamp,primary_key,user \
        --seed "$seed" \
        2>&1 | tee "$log"
    done
  done
}

run_task \
  event_interest-interested \
  interested

run_task \
  event_interest-not_interested \
  not_interested

echo
echo "============================================================"
echo "ALL REL-EVENT INTEREST RUNS COMPLETE"
echo "============================================================"
