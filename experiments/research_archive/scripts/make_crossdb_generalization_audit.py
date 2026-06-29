import os
from pathlib import Path
import pandas as pd


OUT_DIR = Path("results/final_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    rows = []

    # 1. Main RelBench multi-task evidence.
    main = read_csv_if_exists(OUT_DIR / "clean_main_4task_summary.csv")
    if main is not None:
        tasks = sorted(
            main[["dataset", "task"]]
            .drop_duplicates()
            .apply(lambda r: f"{r['dataset']}/{r['task']}", axis=1)
            .tolist()
        )
        variants = sorted(main["variant"].dropna().unique().tolist())
        rows.append({
            "requirement": "RelBench multi-database/task evaluation",
            "status": "done",
            "evidence_file": "clean_main_4task_summary.csv",
            "evidence_summary": f"{len(tasks)} tasks: {', '.join(tasks)}; variants: {', '.join(variants)}",
            "paper_claim_allowed": "FDHG is evaluated across multiple RelBench databases/tasks.",
            "paper_claim_not_allowed": "This is not by itself leave-one-database-out learned-policy generalization.",
            "next_action": "Use as main RelBench coverage table."
        })
    else:
        rows.append({
            "requirement": "RelBench multi-database/task evaluation",
            "status": "missing",
            "evidence_file": "",
            "evidence_summary": "",
            "paper_claim_allowed": "",
            "paper_claim_not_allowed": "",
            "next_action": "Find or regenerate clean_main_4task_summary.csv."
        })

    # 2. Heuristic fallback evidence.
    fallback = read_csv_if_exists(OUT_DIR / "fdhg_fallback_audit_main_4task.csv")
    if fallback is not None:
        n_fallback = int(fallback.get("fallback", pd.Series(dtype=bool)).sum()) if "fallback" in fallback.columns else ""
        rows.append({
            "requirement": "Heuristic fallback behavior",
            "status": "done",
            "evidence_file": "fdhg_fallback_audit_main_4task.csv",
            "evidence_summary": f"Fallback audit exists; fallback_count={n_fallback}.",
            "paper_claim_allowed": "FDHG safely degenerates to DFS-style aggregation when no reliable FDHG-specific program is selected.",
            "paper_claim_not_allowed": "Fallback audit is not a learned-policy transfer experiment.",
            "next_action": "Report in ablation/failure-analysis section."
        })
    else:
        rows.append({
            "requirement": "Heuristic fallback behavior",
            "status": "missing",
            "evidence_file": "",
            "evidence_summary": "",
            "paper_claim_allowed": "",
            "paper_claim_not_allowed": "",
            "next_action": "Create fallback audit from DFS/FDHG identical rows."
        })

    # 3. Synthetic prior evidence.
    syn_recovery = read_csv_if_exists(OUT_DIR / "synthetic_program_recovery_summary.csv")
    syn_oracle = read_csv_if_exists(OUT_DIR / "synthetic_oracle_gap_summary.csv")
    if syn_recovery is not None and syn_oracle is not None:
        rows.append({
            "requirement": "Synthetic prior sanity check",
            "status": "done",
            "evidence_file": "synthetic_program_recovery_summary.csv; synthetic_oracle_gap_summary.csv",
            "evidence_summary": "Program recovery and oracle gap tables exist.",
            "paper_claim_allowed": "Synthetic prior validates controlled program recovery/oracle-gap behavior.",
            "paper_claim_not_allowed": "This does not prove synthetic-only ranker transfer to RelBench.",
            "next_action": "Report as synthetic sanity/theory support."
        })
    else:
        rows.append({
            "requirement": "Synthetic prior sanity check",
            "status": "partial_or_missing",
            "evidence_file": "",
            "evidence_summary": "One or both synthetic summary files missing.",
            "paper_claim_allowed": "",
            "paper_claim_not_allowed": "",
            "next_action": "Regenerate synthetic recovery/oracle-gap summaries."
        })

    # 4. Learned/supervised ranker on RelBench.
    ext_c = read_csv_if_exists(OUT_DIR / "relstack_user_badge_extension_c_summary.csv")
    if ext_c is not None:
        variants = sorted(ext_c["variant"].dropna().unique().tolist()) if "variant" in ext_c.columns else []
        rows.append({
            "requirement": "Learned/supervised policy vs heuristic on RelBench",
            "status": "partial",
            "evidence_file": "relstack_user_badge_extension_c_summary.csv",
            "evidence_summary": f"Learned/supervised topK evidence exists only for rel-stack/user-badge; variants: {', '.join(variants)}",
            "paper_claim_allowed": "Train-only supervised program filtering improves the rel-stack/user-badge dmax2 integration over heuristic topK.",
            "paper_claim_not_allowed": "Do not call this cross-database learned-policy generalization.",
            "next_action": "Either run learned-policy transfer on additional DBs or explicitly limit claim to rel-stack ablation."
        })
    else:
        rows.append({
            "requirement": "Learned/supervised policy vs heuristic on RelBench",
            "status": "missing",
            "evidence_file": "",
            "evidence_summary": "",
            "paper_claim_allowed": "",
            "paper_claim_not_allowed": "",
            "next_action": "Find Extension C summary or run supervised ranker ablation."
        })

    # 5. Leave-one-database-out.
    rows.append({
        "requirement": "Leave-one-database-out learned policy",
        "status": "not_done",
        "evidence_file": "",
        "evidence_summary": "No current evidence that a learned ranker/calibrator trained on one set of RelBench databases transfers to a held-out database.",
        "paper_claim_allowed": "Can be listed as future work/limitation if not run.",
        "paper_claim_not_allowed": "Do not claim cross-database learned-policy generalization.",
        "next_action": "Optional heavy experiment: train ranker on synthetic or source DBs and evaluate selected programs on held-out RelBench DB."
    })

    # 6. Synthetic-only pretraining to RelBench.
    rows.append({
        "requirement": "Synthetic-only pretraining then RelBench test",
        "status": "not_done",
        "evidence_file": "",
        "evidence_summary": "Synthetic prior was used for controlled recovery/oracle-gap, not yet for a policy trained only on synthetic tasks and transferred to RelBench.",
        "paper_claim_allowed": "Can claim synthetic prior sanity support only.",
        "paper_claim_not_allowed": "Do not claim synthetic-only learned policy transfers to RelBench.",
        "next_action": "Optional heavy experiment: train calibrator/ranker on synthetic tasks, freeze it, select RelBench features."
    })

    audit = pd.DataFrame(rows)
    out = OUT_DIR / "crossdb_generalization_coverage_audit.csv"
    audit.to_csv(out, index=False)

    print(audit.to_string(index=False))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
