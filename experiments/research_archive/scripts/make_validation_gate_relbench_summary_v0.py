from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("results/final_tables")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

def add(
    dataset,
    task,
    base_compiler,
    base_variant,
    candidate_variant,
    base_log_loss,
    candidate_log_loss,
    base_auroc=None,
    candidate_auroc=None,
    base_ap=None,
    candidate_ap=None,
    protocol="",
    evidence_source="",
    interpretation="",
):
    gain_log_loss = base_log_loss - candidate_log_loss if pd.notna(base_log_loss) and pd.notna(candidate_log_loss) else np.nan

    if pd.notna(gain_log_loss) and gain_log_loss > 0:
        selected_variant = candidate_variant
        gate_decision = "select_residual"
        fallback = False
    else:
        selected_variant = base_variant
        gate_decision = "fallback_to_base"
        fallback = True

    rows.append({
        "dataset": dataset,
        "task": task,
        "base_compiler": base_compiler,
        "base_variant": base_variant,
        "candidate_variant": candidate_variant,
        "protocol": protocol,
        "base_log_loss": base_log_loss,
        "candidate_log_loss": candidate_log_loss,
        "log_loss_gain_base_minus_candidate": gain_log_loss,
        "base_auroc": base_auroc,
        "candidate_auroc": candidate_auroc,
        "auroc_delta_candidate_minus_base": (
            candidate_auroc - base_auroc
            if pd.notna(base_auroc) and pd.notna(candidate_auroc)
            else np.nan
        ),
        "base_ap": base_ap,
        "candidate_ap": candidate_ap,
        "ap_delta_candidate_minus_base": (
            candidate_ap - base_ap
            if pd.notna(base_ap) and pd.notna(candidate_ap)
            else np.nan
        ),
        "selected_variant": selected_variant,
        "gate_decision": gate_decision,
        "fallback": fallback,
        "evidence_source": evidence_source,
        "interpretation": interpretation,
    })

# ---------------------------------------------------------------------
# Case 1: rel-stack/user-badge
# Strong JUICE-style base. FDHG residual hurts on same sampled rows.
# Gate should fallback.
# ---------------------------------------------------------------------

add(
    dataset="rel-stack",
    task="user-badge",
    base_compiler="JUICE-style v0",
    base_variant="juice_on_fdhg_sample",
    candidate_variant="juice_plus_fdhg_dmax1_ambiguity",
    base_log_loss=0.085063,
    candidate_log_loss=0.086046,
    base_auroc=0.949759,
    candidate_auroc=0.945768,
    base_ap=0.480929,
    candidate_ap=0.480335,
    protocol="same sampled rows, seeds 41-44, n_val=2000",
    evidence_source="phase1_relstack_juice_residual_complementarity",
    interpretation=(
        "FDHG ambiguity residual improves over DFS/FKAgg, but not over broad JUICE-style meta-path features; "
        "validation gate should fallback to JUICE-style base."
    ),
)

# ---------------------------------------------------------------------
# Case 2: rel-amazon/item-churn
# Simple JUICE-style base. Current best recorded residual-like variant is FDHG+dmax2.
# This is not yet a literal JUICE+dmax2 matrix; mark as proxy.
# Gate should select residual under proxy comparison.
# ---------------------------------------------------------------------

add(
    dataset="rel-amazon",
    task="item-churn",
    base_compiler="JUICE-style v0",
    base_variant="juice_style_matched_v0",
    candidate_variant="fdhg_dmax1_plus_dmax2_proxy",
    base_log_loss=0.606485,
    candidate_log_loss=0.591138,
    base_auroc=0.760865,
    candidate_auroc=0.764607,
    base_ap=0.631319,
    candidate_ap=0.638943,
    protocol="sampled 10k/2k, seeds 41-44",
    evidence_source="phase1_item_churn_juice_style_and_stage_c_dmax2",
    interpretation=(
        "FDHG+dmax2 proxy beats simple JUICE-style v0 on AUROC/AP/log-loss; "
        "next step is to build literal JUICE+dmax2 residual matrix."
    ),
)

# ---------------------------------------------------------------------
# Case 3: rel-amazon/user-churn
# FDHG exactly equals DFS. Gate should fallback.
# ---------------------------------------------------------------------

add(
    dataset="rel-amazon",
    task="user-churn",
    base_compiler="DFS/FKAgg",
    base_variant="dfs",
    candidate_variant="fdhg_dmax1",
    base_log_loss=0.615203,
    candidate_log_loss=0.615203,
    base_auroc=0.641902,
    candidate_auroc=0.641902,
    base_ap=0.729593,
    candidate_ap=0.729593,
    protocol="seeds 41-44",
    evidence_source="fallback_audit_main_4task",
    interpretation=(
        "No usable non-ID FDHG-specific residual was selected; FDHG safely degenerates to DFS/FKAgg."
    ),
)

# ---------------------------------------------------------------------
# Case 4: rel-f1/driver-dnf
# Temporal base dominates. Adding FDHG on top of last-state hurts.
# Gate should fallback to temporal base.
# ---------------------------------------------------------------------

add(
    dataset="rel-f1",
    task="driver-dnf",
    base_compiler="DFS+last temporal",
    base_variant="dfs_plus_last",
    candidate_variant="fdhg_plus_last",
    base_log_loss=0.421222,
    candidate_log_loss=0.431066,
    base_auroc=0.807701,
    candidate_auroc=0.798771,
    base_ap=0.935101,
    candidate_ap=0.927954,
    protocol="seeds 41-44",
    evidence_source="f1_driver_dnf_temporal_diagnostic",
    interpretation=(
        "Temporal/local-state features dominate; generic FDHG ambiguity residual should not be added on top of temporal base."
    ),
)

df = pd.DataFrame(rows)

out_csv = OUT / "validation_gate_relbench_summary_v0.csv"
df.to_csv(out_csv, index=False)

print("=== validation_gate_relbench_summary_v0 ===")
print(df.to_string(index=False))
print("\nSaved:", out_csv)

# Also write a small markdown interpretation.
md = OUT / "VALIDATION_GATE_RELBENCH_SUMMARY_V0.md"
with md.open("w") as f:
    f.write("# Validation-Gated RelBench Summary v0\n\n")
    f.write("This table connects existing RelBench results to the FDHG-v2 validation-gated residual compiler story.\n\n")
    f.write("Important caveat: the item-churn row currently uses FDHG+dmax2 as a proxy candidate, not yet a literal JUICE+dmax2 residual matrix. The next implementation step is to materialize that matrix.\n\n")
    for _, r in df.iterrows():
        f.write(f"## {r['dataset']}/{r['task']}\n\n")
        f.write(f"- Base: `{r['base_variant']}`\n")
        f.write(f"- Candidate: `{r['candidate_variant']}`\n")
        f.write(f"- Base log-loss: {r['base_log_loss']:.6f}\n")
        f.write(f"- Candidate log-loss: {r['candidate_log_loss']:.6f}\n")
        f.write(f"- Log-loss gain, base minus candidate: {r['log_loss_gain_base_minus_candidate']:.6f}\n")
        f.write(f"- Gate decision: `{r['gate_decision']}`\n")
        f.write(f"- Selected variant: `{r['selected_variant']}`\n")
        f.write(f"- Interpretation: {r['interpretation']}\n\n")

print("Saved:", md)
