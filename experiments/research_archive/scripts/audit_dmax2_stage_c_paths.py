from pathlib import Path
import pandas as pd

cfg = pd.read_csv("configs/relbench_v1_dmax2_stage_c_candidates.csv")
rows = []

for _, r in cfg.iterrows():
    dataset = r["dataset"]
    task = r["task"]
    slug = f"{dataset}_{task}"
    inspect_dir = Path("outputs/relbench_inspect") / slug

    print("\n" + "="*100)
    print(slug)

    target_train = inspect_dir / "target_train.parquet"
    base_child = inspect_dir / f"table_{r['base_child_table']}.parquet"
    second = inspect_dir / f"table_{r['second_table']}.parquet"

    status = "ok"
    problems = []

    for name, p in [("target_train", target_train), ("base_child", base_child), ("second_table", second)]:
        if not p.exists():
            status = "missing_file"
            problems.append(f"missing {name}: {p}")

    if status == "ok":
        try:
            target = pd.read_parquet(target_train).head(5)
            b = pd.read_parquet(base_child).head(5)
            s = pd.read_parquet(second).head(5)

            checks = [
                ("target_key", r["target_key"], target.columns),
                ("target_time_col", r["target_time_col"], target.columns),
                ("base_child_key", r["base_child_key"], b.columns),
                ("base_child_time_col", r["base_child_time_col"], b.columns),
                ("pivot_key_in_base", r["pivot_key"], b.columns),
                ("second_key", r["second_key"], s.columns),
                ("second_time_col", r["second_time_col"], s.columns),
                ("numeric_col", r["numeric_col"], s.columns),
            ]

            for label, col, cols in checks:
                if pd.isna(col) or str(col).strip() == "":
                    continue
                if col not in cols:
                    status = "missing_column"
                    problems.append(f"{label}={col} not in columns={list(cols)}")

            print("target columns:", list(target.columns))
            print("base child columns:", list(b.columns))
            print("second table columns:", list(s.columns))

        except Exception as e:
            status = "read_error"
            problems.append(repr(e))

    print("status:", status)
    if problems:
        for p in problems:
            print(" -", p)

    rows.append({
        "dataset": dataset,
        "task": task,
        "status": status,
        "problems": " | ".join(problems),
        "inspect_dir": str(inspect_dir),
    })

out = pd.DataFrame(rows)
out_path = Path("results/stage_c_dmax2_path_audit.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print("\nSaved:", out_path)
print(out.to_string(index=False))
