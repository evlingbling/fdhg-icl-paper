import re
import glob
import pandas as pd
from pathlib import Path

rows = []

def get_value(pattern, text):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None

def parse_elapsed_from_gnu_time(text):
    m = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text)
    if not m:
        return None
    t = m.group(1).strip()
    parts = t.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return None

def parse_elapsed_from_wrapper(text):
    s = re.search(r"\[TIME_WRAPPER\] start_unix:\s*([0-9]+)", text)
    e = re.search(r"\[TIME_WRAPPER\] end_unix:\s*([0-9]+)", text)
    if s and e:
        return int(e.group(1)) - int(s.group(1))
    return None

for path in sorted(glob.glob("logs/timing/*.log")):
    p = Path(path)
    text = p.read_text(errors="ignore")
    name = p.stem

    elapsed_seconds = parse_elapsed_from_gnu_time(text)
    if elapsed_seconds is None:
        elapsed_seconds = parse_elapsed_from_wrapper(text)

    max_rss_kb = get_value(r"Maximum resident set size \(kbytes\):\s*([0-9.]+)", text)
    user_time = get_value(r"User time \(seconds\):\s*([0-9.]+)", text)
    sys_time = get_value(r"System time \(seconds\):\s*([0-9.]+)", text)

    dataset = "rel-stack" if "relstack" in name else "unknown"
    task = "user-badge" if "relstack" in name else "unknown"

    if "feature_filter_materialization" in name:
        stage = "feature_materialization"
        variant = "filtered_feature_export_all_variants"
        model = "NA"
    else:
        stage = "decoder_eval"
        model = "catboost" if "catboost" in name else "xgb" if "xgb" in name else "unknown"

        if "dmax1_plus_dmax2" in name:
            variant = "fdhg_dmax1_plus_dmax2_supervised_ap_topk16"
        elif "fdhg_dmax1_full" in name:
            variant = "fdhg_dmax1_full"
        elif "dfs" in name:
            variant = "dfs"
        else:
            variant = "unknown"

    rows.append({
        "dataset": dataset,
        "task": task,
        "stage": stage,
        "variant": variant,
        "model": model,
        "log_path": str(p),
        "elapsed_wall_time_sec": elapsed_seconds,
        "user_time_sec": user_time,
        "system_time_sec": sys_time,
        "peak_memory_mb": max_rss_kb / 1024 if max_rss_kb is not None else None,
    })

df = pd.DataFrame(rows)
df.to_csv("results/final_tables/efficiency_timing_summary_relstack.csv", index=False)
df.to_csv("results/efficiency/efficiency_timing_summary_relstack.csv", index=False)

print(df.to_string(index=False))
print("\nSaved: results/final_tables/efficiency_timing_summary_relstack.csv")
