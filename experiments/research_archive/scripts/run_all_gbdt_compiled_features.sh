#!/usr/bin/env bash
set -u

MODEL="${1:-xgb}"
SEEDS="41 42 43 44"

OUTDIR="results/gbdt_compiled_features"
LOGDIR="logs/gbdt_compiled_features"
mkdir -p "$OUTDIR" "$LOGDIR"

run_one () {
  local dataset="$1"
  local task="$2"
  local variant="$3"
  local seed="$4"
  local train="$5"
  local val="$6"
  local target_col="${7:-target}"

  if [[ ! -f "$train" ]]; then
    echo "[SKIP] missing train: $train"
    return 0
  fi
  if [[ ! -f "$val" ]]; then
    echo "[SKIP] missing val: $val"
    return 0
  fi

  local safe_dataset="${dataset//-/_}"
  local safe_task="${task//-/_}"
  local safe_variant="${variant//-/_}"

  local outfile="${OUTDIR}/${safe_dataset}_${safe_task}_${safe_variant}_${MODEL}_seed${seed}.csv"
  local logfile="${LOGDIR}/${safe_dataset}_${safe_task}_${safe_variant}_${MODEL}_seed${seed}.log"

  if [[ -f "$outfile" ]]; then
    echo "[DONE] exists: $outfile"
    return 0
  fi

  echo "[RUN] $dataset/$task | $variant | $MODEL | seed=$seed"
  echo "      train=$train"
  echo "      val=$val"

  python scripts/run_gbdt_compiled_features.py \
    --train "$train" \
    --val "$val" \
    --dataset "$dataset" \
    --task "$task" \
    --variant "$variant" \
    --model "$MODEL" \
    --seed "$seed" \
    --target-col "$target_col" \
    --out "$outfile" \
    > "$logfile" 2>&1

  status=$?
  if [[ $status -ne 0 ]]; then
    echo "[FAIL] $dataset/$task | $variant | seed=$seed"
    echo "       see $logfile"
    return 0
  fi

  echo "[OK] saved: $outfile"
}

pick_dir () {
  local include="$1"
  local seed="$2"
  local exclude="${3:-}"

  if [[ -z "$exclude" ]]; then
    find ./results -type d \
      | grep -Ei "$include" \
      | grep -Ei "seed${seed}" \
      | sort \
      | head -1
  else
    find ./results -type d \
      | grep -Ei "$include" \
      | grep -Ei "seed${seed}" \
      | grep -Eiv "$exclude" \
      | sort \
      | head -1
  fi
}

find_pair_and_run () {
  local dataset="$1"
  local task="$2"
  local variant="$3"
  local seed="$4"
  local include="$5"
  local exclude="${6:-}"

  local dir
  dir=$(pick_dir "$include" "$seed" "$exclude" || true)

  if [[ -z "${dir:-}" ]]; then
    echo "[SKIP] no dir found for $dataset/$task | $variant | seed=$seed"
    echo "       include=$include"
    echo "       exclude=$exclude"
    return 0
  fi

  local train="${dir}/train_combined.parquet"
  local val="${dir}/val_combined.parquet"

  local target_col="target"
  if [[ "$dataset" == "rel-stack" && "$task" == "user-badge" ]]; then
    target_col="WillGetBadge"
  fi
  if [[ "$dataset" == "rel-amazon" && "$task" == "item-churn" ]]; then
    target_col="target"
  fi
  if [[ "$dataset" == "rel-f1" && "$task" == "driver-dnf" ]]; then
    target_col="target"
  fi

  run_one "$dataset" "$task" "$variant" "$seed" "$train" "$val" "$target_col"
}

echo "============================================================"
echo "Running GBDT compiled-feature experiments"
echo "MODEL=$MODEL"
echo "SEEDS=$SEEDS"
echo "============================================================"

for SEED in $SEEDS; do
  echo ""
  echo "==================== seed $SEED ===================="

  # rel-stack/user-badge
  find_pair_and_run \
    "rel-stack" "user-badge" "dfs" "$SEED" \
    "rel_stack_user_badge.*dfs" \
    "dmax2|random|shuffle|supervised|topk"

  find_pair_and_run \
    "rel-stack" "user-badge" "fdhg_dmax1_full" "$SEED" \
    "rel_stack_user_badge.*dmax1.*fdhg.*full" \
    "dmax2|random|shuffle|supervised|topk"

  find_pair_and_run \
    "rel-stack" "user-badge" "fdhg_dmax1_plus_dmax2_supervised_ap_topk16" "$SEED" \
    "rel_stack_user_badge.*dmax1.*dmax2.*supervised_ap_topk16" \
    ""

  # rel-amazon/item-churn
  find_pair_and_run \
    "rel-amazon" "item-churn" "dfs" "$SEED" \
    "amazon.*item.*churn.*dfs|rel_amazon_item_churn.*dfs" \
    "fdhg|dmax2|random|shuffle"

  find_pair_and_run \
    "rel-amazon" "item-churn" "fdhg_dmax1_full" "$SEED" \
    "amazon.*item.*churn.*fdhg|rel_amazon_item_churn.*fdhg" \
    "dmax2|random|shuffle"

  find_pair_and_run \
    "rel-amazon" "item-churn" "fdhg_dmax1_plus_dmax2" "$SEED" \
    "amazon.*item.*churn.*dmax2|rel_amazon_item_churn.*dmax2" \
    ""

  # rel-f1/driver-dnf
  find_pair_and_run \
    "rel-f1" "driver-dnf" "dfs" "$SEED" \
    "f1.*driver.*dnf.*dfs|rel_f1_driver_dnf.*dfs" \
    "fdhg|last|dmax2|random|shuffle"

  find_pair_and_run \
    "rel-f1" "driver-dnf" "fdhg_dmax1_full" "$SEED" \
    "f1.*driver.*dnf.*fdhg|rel_f1_driver_dnf.*fdhg" \
    "last|dmax2|random|shuffle"

  find_pair_and_run \
    "rel-f1" "driver-dnf" "fdhg_dmax1_plus_dmax2" "$SEED" \
    "f1.*driver.*dnf.*dmax2|rel_f1_driver_dnf.*dmax2" \
    ""

done

echo ""
echo "============================================================"
echo "Finished running available jobs."
echo "Now aggregating results..."
echo "============================================================"

python - <<'PY'
import glob
import os
import pandas as pd

paths = sorted(glob.glob("results/gbdt_compiled_features/*.csv"))

if not paths:
    print("[WARN] No result CSVs found.")
    raise SystemExit(0)

df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
os.makedirs("results/final_tables", exist_ok=True)

all_path = "results/final_tables/gbdt_compiled_features_all_runs.csv"
summary_path = "results/final_tables/gbdt_compiled_features_summary.csv"
delta_path = "results/final_tables/gbdt_compiled_features_deltas_vs_dfs.csv"

df.to_csv(all_path, index=False)

summary = (
    df.groupby(["dataset", "task", "model", "variant"], as_index=False)
      .agg(
          n_runs=("seed", "count"),
          seeds=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
          n_features_mean=("n_features", "mean"),
          accuracy_mean=("accuracy", "mean"),
          accuracy_std=("accuracy", "std"),
          roc_auc_mean=("roc_auc", "mean"),
          roc_auc_std=("roc_auc", "std"),
          average_precision_mean=("average_precision", "mean"),
          average_precision_std=("average_precision", "std"),
          log_loss_mean=("log_loss", "mean"),
          log_loss_std=("log_loss", "std"),
      )
)
summary.to_csv(summary_path, index=False)

rows = []
for (dataset, task, model), g in summary.groupby(["dataset", "task", "model"]):
    dfs = g[g["variant"] == "dfs"]
    if len(dfs) == 0:
        continue
    base = dfs.iloc[0]

    for _, r in g.iterrows():
        if r["variant"] == "dfs":
            continue
        rows.append({
            "dataset": dataset,
            "task": task,
            "model": model,
            "variant": r["variant"],
            "delta_vs_dfs_accuracy": r["accuracy_mean"] - base["accuracy_mean"],
            "delta_vs_dfs_roc_auc": r["roc_auc_mean"] - base["roc_auc_mean"],
            "delta_vs_dfs_average_precision": r["average_precision_mean"] - base["average_precision_mean"],
            "delta_vs_dfs_log_loss": r["log_loss_mean"] - base["log_loss_mean"],
            "variant_accuracy": r["accuracy_mean"],
            "dfs_accuracy": base["accuracy_mean"],
            "variant_roc_auc": r["roc_auc_mean"],
            "dfs_roc_auc": base["roc_auc_mean"],
            "variant_average_precision": r["average_precision_mean"],
            "dfs_average_precision": base["average_precision_mean"],
            "variant_log_loss": r["log_loss_mean"],
            "dfs_log_loss": base["log_loss_mean"],
        })

delta = pd.DataFrame(rows)
delta.to_csv(delta_path, index=False)

print("\n=== SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== DELTA VS DFS ===")
if len(delta):
    print(delta.to_string(index=False))
else:
    print("[WARN] No DFS baseline found for delta calculation.")

print("\nSaved:")
print(all_path)
print(summary_path)
print(delta_path)
PY
