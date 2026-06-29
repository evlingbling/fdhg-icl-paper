from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(
    "results/rel_trial_studies_enrollment_dmax2_all_seed41/"
    "rel_trial_studies_enrollment_dmax2_all_seed41_topk64"
)

OUT = Path("results/rel_trial_studies_enrollment_eval_seed41")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 41

train = pd.read_parquet(ROOT / "train_combined.parquet")
val = pd.read_parquet(ROOT / "val_combined.parquet")

feature_cols = [
    c for c in train.columns
    if c.startswith("dmax2_")
]

X_train = train[feature_cols].astype(float)
X_val = val[feature_cols].astype(float)

y_train = train["enrollment"].astype(float).clip(lower=0)
y_val = val["enrollment"].astype(float).clip(lower=0)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)


def evaluate(name, pred_raw, pred_log=None):
    pred_raw = np.asarray(pred_raw, dtype=float)
    pred_raw = np.clip(pred_raw, 0, None)

    if pred_log is None:
        pred_log = np.log1p(pred_raw)
    else:
        pred_log = np.asarray(pred_log, dtype=float)

    return {
        "variant": name,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_features": 0 if name == "median_baseline" else len(feature_cols),
        "rmse_raw": np.sqrt(
            mean_squared_error(y_val, pred_raw)
        ),
        "mae_raw": mean_absolute_error(y_val, pred_raw),
        "r2_raw": r2_score(y_val, pred_raw),
        "rmse_log1p": np.sqrt(
            mean_squared_error(y_val_log, pred_log)
        ),
        "mae_log1p": mean_absolute_error(y_val_log, pred_log),
        "median_absolute_error_raw": float(
            np.median(np.abs(y_val.to_numpy() - pred_raw))
        ),
    }


results = []

# 1. Constant median baseline
train_median = float(y_train.median())
baseline_raw = np.full(len(y_val), train_median)
baseline_log = np.full(len(y_val), np.log1p(train_median))

results.append(
    evaluate(
        "median_baseline",
        baseline_raw,
        baseline_log,
    )
)

# 2. DMAX2 CatBoost, trained on log1p target
model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=5.0,
    random_seed=SEED,
    verbose=100,
    allow_writing_files=False,
)

model.fit(
    X_train,
    y_train_log,
    eval_set=(X_val, y_val_log),
    early_stopping_rounds=100,
    verbose=100,
)

pred_log = model.predict(X_val)
pred_raw = np.expm1(pred_log)
pred_raw = np.clip(pred_raw, 0, None)

results.append(
    evaluate(
        "dmax2_all_catboost_log1p",
        pred_raw,
        pred_log,
    )
)

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUT / "metrics.csv",
    index=False,
)

pred_df = pd.DataFrame({
    "nct_id": val["nct_id"],
    "target_enrollment": y_val,
    "prediction_enrollment": pred_raw,
    "target_log1p": y_val_log,
    "prediction_log1p": pred_log,
})

pred_df.to_parquet(
    OUT / "val_predictions.parquet",
    index=False,
)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False)

importance_df.to_csv(
    OUT / "feature_importance.csv",
    index=False,
)

metadata = {
    "dataset": "rel-trial",
    "task": "studies-enrollment",
    "seed": SEED,
    "training_target": "log1p(enrollment)",
    "evaluation_split": "official validation",
    "n_features": len(feature_cols),
    "feature_columns": feature_cols,
    "best_iteration": int(model.get_best_iteration()),
}

with open(OUT / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n=== METRICS ===")
print(result_df.to_string(index=False))

print("\n=== FEATURE IMPORTANCE ===")
print(importance_df.to_string(index=False))

print("\nSaved:")
print(OUT / "metrics.csv")
print(OUT / "feature_importance.csv")
print(OUT / "val_predictions.parquet")
print(OUT / "metadata.json")
