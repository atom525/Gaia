#!/usr/bin/env python3
"""Experiment: Silence vs Tunable-Leak Entailment Potential.

Compares three entailment C4 models on the Galileo coarse and fine graphs:
  1. silence  (current):  premise false → potential = 1.0
  2. noisy-AND (ε=1e-3):  premise false → potential = ε (same as induction)
  3. soft-leak (ε=0.01):  premise false → potential = 0.01 (intermediate)

Uses exact inference (brute-force enumeration) so results are mathematically
precise — no BP approximation artifacts.

Key questions:
  - Does the choice of entailment C4 model affect belief in conclusions?
  - In which graph topology (coarse vs fine) is the difference most significant?
  - Does the silence model violate C4 in practice (premise false → conclusion
    belief should decrease below prior)?
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libs.inference_v2.factor_graph import CROMWELL_EPS, FactorGraph, FactorType, Factor
from libs.inference_v2.exact import exact_inference


# =====================================================================
# Monkey-patch helpers: swap entailment potential model at runtime
# =====================================================================

def _patched_factor_log_potentials(factor, states, var_idx, *, leak):
    """Like exact.py's _factor_log_potentials but with configurable entailment leak."""
    cs = states.shape[0]
    eps = CROMWELL_EPS
    ft = factor.factor_type

    if ft == FactorType.ENTAILMENT:
        p = factor.p
        premise_idxs = [var_idx[v] for v in factor.premises]
        conclusion_idxs = [var_idx[v] for v in factor.conclusions]
        all_prem_true = np.ones(cs, dtype=bool)
        for pi in premise_idxs:
            all_prem_true &= states[:, pi] == 1
        log_pot = np.zeros(cs, dtype=np.float64)
        for ci in conclusion_idxs:
            c_val = states[:, ci]
            if leak is None:
                # silence: potential = 1.0 when premises false
                pot_val = np.where(
                    all_prem_true,
                    np.where(c_val == 1, p, 1.0 - p),
                    1.0,
                )
            else:
                # tunable leak: potential = leak when premises false, conclusion=1
                pot_val = np.where(
                    all_prem_true,
                    np.where(c_val == 1, p, 1.0 - p),
                    np.where(c_val == 1, leak, 1.0 - leak),
                )
            log_pot += np.log(pot_val)
        return log_pot

    elif ft in (FactorType.INDUCTION, FactorType.ABDUCTION):
        p = factor.p
        premise_idxs = [var_idx[v] for v in factor.premises]
        conclusion_idxs = [var_idx[v] for v in factor.conclusions]
        all_prem_true = np.ones(cs, dtype=bool)
        for pi in premise_idxs:
            all_prem_true &= states[:, pi] == 1
        log_pot = np.zeros(cs, dtype=np.float64)
        for ci in conclusion_idxs:
            c_val = states[:, ci]
            pot_val = np.where(
                all_prem_true,
                np.where(c_val == 1, p, 1.0 - p),
                np.where(c_val == 1, eps, 1.0 - eps),
            )
            log_pot += np.log(pot_val)
        return log_pot

    elif ft == FactorType.CONTRADICTION:
        r_idx = var_idx[factor.relation_var]
        claim_idxs = [var_idx[v] for v in factor.premises]
        r_val = states[:, r_idx]
        all_claims_true = np.ones(cs, dtype=bool)
        for ci in claim_idxs:
            all_claims_true &= states[:, ci] == 1
        pot_val = np.where((r_val == 1) & all_claims_true, eps, 1.0)
        return np.log(pot_val)

    elif ft == FactorType.EQUIVALENCE:
        r_idx = var_idx[factor.relation_var]
        a_idx = var_idx[factor.premises[0]]
        b_idx = var_idx[factor.premises[1]]
        r_val = states[:, r_idx]
        a_val = states[:, a_idx]
        b_val = states[:, b_idx]
        pot_val = np.where(
            r_val == 0, 1.0,
            np.where(a_val == b_val, 1.0 - eps, eps),
        )
        return np.log(pot_val)

    else:
        raise ValueError(f"Unknown FactorType: {ft}")


def exact_inference_with_leak(graph: FactorGraph, leak) -> dict[str, float]:
    """Run exact inference with a specific entailment leak parameter."""
    import libs.inference_v2.exact as exact_mod

    original_fn = exact_mod._factor_log_potentials

    def patched(factor, states, var_idx):
        return _patched_factor_log_potentials(factor, states, var_idx, leak=leak)

    exact_mod._factor_log_potentials = patched
    try:
        beliefs, Z = exact_inference(graph)
    finally:
        exact_mod._factor_log_potentials = original_fn
    return beliefs


# =====================================================================
# Graph builders
# =====================================================================

def build_coarse_galileo() -> FactorGraph:
    """Galileo coarse factor graph from science-formalization.md §3.3."""
    fg = FactorGraph()

    # Variables — all claims
    fg.add_variable("A", prior=0.5)          # Aristotle: heavier falls faster
    fg.add_variable("V", prior=0.5)          # Galileo: equal speed in vacuum
    fg.add_variable("O_daily", prior=0.9)    # observation: stones > feathers
    fg.add_variable("O_media", prior=0.9)    # observation: denser medium → bigger gap
    fg.add_variable("O_air", prior=0.9)      # observation: in air, gold ≈ copper
    fg.add_variable("T1", prior=0.5)         # drag argument
    fg.add_variable("T2", prior=0.5)         # total-weight argument
    fg.add_variable("S_vac", prior=0.95)     # setting: vacuum = zero resistance
    fg.add_variable("S_plane", prior=0.95)   # setting: inclined plane ≈ diluted gravity
    fg.add_variable("E_theta", prior=0.9)    # experiment: s ∝ t² for all angles
    fg.add_variable("A_vac", prior=0.5)      # A applied to vacuum
    fg.add_variable("R_T1T2", prior=0.9)     # contradiction relation T1⊗T2
    fg.add_variable("R_AV", prior=0.9)       # contradiction relation A_vac⊗V

    # W₁: O_daily → A (induction, p<1)
    fg.add_factor("W1", FactorType.INDUCTION,
                  premises=["O_daily"], conclusions=["A"], p=0.7)

    # A → T₁, A → T₂ (entailment)
    fg.add_factor("f_A_T1", FactorType.ENTAILMENT,
                  premises=["A"], conclusions=["T1"], p=0.999)
    fg.add_factor("f_A_T2", FactorType.ENTAILMENT,
                  premises=["A"], conclusions=["T2"], p=0.999)

    # T₁ ⊗ T₂ contradiction
    fg.add_factor("contra_T1T2", FactorType.CONTRADICTION,
                  premises=["T1", "T2"], conclusions=[], p=0.5,
                  relation_var="R_T1T2")

    # A + S_vac → A_vac (entailment)
    fg.add_factor("f_A_Avac", FactorType.ENTAILMENT,
                  premises=["A", "S_vac"], conclusions=["A_vac"], p=0.999)

    # A_vac ⊗ V contradiction
    fg.add_factor("contra_AV", FactorType.CONTRADICTION,
                  premises=["A_vac", "V"], conclusions=[], p=0.5,
                  relation_var="R_AV")

    # W₂: O_media + O_air + S_vac → V (abduction, p<1)
    fg.add_factor("W2", FactorType.ABDUCTION,
                  premises=["O_media", "O_air", "S_vac"], conclusions=["V"], p=0.8)

    # W₃: E_theta + S_plane → V (induction, p<1)
    fg.add_factor("W3", FactorType.INDUCTION,
                  premises=["E_theta", "S_plane"], conclusions=["V"], p=0.85)

    return fg


def build_fine_galileo() -> FactorGraph:
    """Galileo fine factor graph from science-formalization.md §3.5."""
    fg = FactorGraph()

    # --- Variables ---
    fg.add_variable("A", prior=0.5)
    fg.add_variable("V", prior=0.5)
    fg.add_variable("O_daily", prior=0.9)
    fg.add_variable("O_media", prior=0.9)
    fg.add_variable("O_air", prior=0.9)
    fg.add_variable("T1", prior=0.5)
    fg.add_variable("T2", prior=0.5)
    fg.add_variable("S_vac", prior=0.95)
    fg.add_variable("S_plane", prior=0.95)
    fg.add_variable("A_vac", prior=0.5)
    fg.add_variable("R_T1T2", prior=0.9)
    fg.add_variable("R_AV", prior=0.9)

    # W₁ expanded: A → B_daily ≡ O_daily
    fg.add_variable("B_daily", prior=0.5)
    fg.add_variable("R_eq_daily", prior=0.9)

    # W₂ expanded: G → B_media ≡ O_media, G → B_air ≡ O_air, G+S_vac → V
    fg.add_variable("G", prior=0.5)
    fg.add_variable("B_media", prior=0.5)
    fg.add_variable("B_air", prior=0.5)
    fg.add_variable("R_eq_media", prior=0.9)
    fg.add_variable("R_eq_air", prior=0.9)

    # W₃ expanded: V+S_plane → B_t1 ≡ E_t1, V+S_plane → B_t2 ≡ E_t2
    fg.add_variable("B_t1", prior=0.5)
    fg.add_variable("B_t2", prior=0.5)
    fg.add_variable("E_t1", prior=0.9)
    fg.add_variable("E_t2", prior=0.9)
    fg.add_variable("R_eq_t1", prior=0.9)
    fg.add_variable("R_eq_t2", prior=0.9)

    # --- Factors ---
    # W₁ expanded
    fg.add_factor("f_A_Bdaily", FactorType.ENTAILMENT,
                  premises=["A"], conclusions=["B_daily"], p=0.999)
    fg.add_factor("eq_daily", FactorType.EQUIVALENCE,
                  premises=["B_daily", "O_daily"], conclusions=[], p=0.5,
                  relation_var="R_eq_daily")

    # A → T₁, A → T₂
    fg.add_factor("f_A_T1", FactorType.ENTAILMENT,
                  premises=["A"], conclusions=["T1"], p=0.999)
    fg.add_factor("f_A_T2", FactorType.ENTAILMENT,
                  premises=["A"], conclusions=["T2"], p=0.999)
    fg.add_factor("contra_T1T2", FactorType.CONTRADICTION,
                  premises=["T1", "T2"], conclusions=[], p=0.5,
                  relation_var="R_T1T2")

    # A + S_vac → A_vac
    fg.add_factor("f_A_Avac", FactorType.ENTAILMENT,
                  premises=["A", "S_vac"], conclusions=["A_vac"], p=0.999)
    fg.add_factor("contra_AV", FactorType.CONTRADICTION,
                  premises=["A_vac", "V"], conclusions=[], p=0.5,
                  relation_var="R_AV")

    # W₂ expanded
    fg.add_factor("f_G_Bm", FactorType.ENTAILMENT,
                  premises=["G"], conclusions=["B_media"], p=0.999)
    fg.add_factor("f_G_Ba", FactorType.ENTAILMENT,
                  premises=["G"], conclusions=["B_air"], p=0.999)
    fg.add_factor("eq_media", FactorType.EQUIVALENCE,
                  premises=["B_media", "O_media"], conclusions=[], p=0.5,
                  relation_var="R_eq_media")
    fg.add_factor("eq_air", FactorType.EQUIVALENCE,
                  premises=["B_air", "O_air"], conclusions=[], p=0.5,
                  relation_var="R_eq_air")
    fg.add_factor("f_GV", FactorType.ENTAILMENT,
                  premises=["G", "S_vac"], conclusions=["V"], p=0.999)

    # W₃ expanded
    fg.add_factor("f_V_Bt1", FactorType.ENTAILMENT,
                  premises=["V", "S_plane"], conclusions=["B_t1"], p=0.999)
    fg.add_factor("f_V_Bt2", FactorType.ENTAILMENT,
                  premises=["V", "S_plane"], conclusions=["B_t2"], p=0.999)
    fg.add_factor("eq_t1", FactorType.EQUIVALENCE,
                  premises=["B_t1", "E_t1"], conclusions=[], p=0.5,
                  relation_var="R_eq_t1")
    fg.add_factor("eq_t2", FactorType.EQUIVALENCE,
                  premises=["B_t2", "E_t2"], conclusions=[], p=0.5,
                  relation_var="R_eq_t2")

    return fg


def build_c4_test_graph() -> FactorGraph:
    """Minimal graph to test C4 violation directly.

    Structure: A → ent → B, with A's prior varied.
    C4 says: A goes from high to low → B should decrease.
    Silence model: B won't change at all (C4 violated).
    """
    fg = FactorGraph()
    fg.add_variable("A", prior=0.5)
    fg.add_variable("B", prior=0.5)
    fg.add_factor("f", FactorType.ENTAILMENT,
                  premises=["A"], conclusions=["B"], p=0.999)
    return fg


# =====================================================================
# Experiment runner
# =====================================================================

def run_comparison(graph: FactorGraph, title: str, key_vars: list[str]) -> str:
    """Run exact inference with three entailment models, return formatted table."""
    models = [
        ("silence (current)", None),
        ("soft-leak (ε=0.01)", 0.01),
        ("noisy-AND (ε=1e-3)", CROMWELL_EPS),
    ]

    results = {}
    for name, leak in models:
        results[name] = exact_inference_with_leak(graph, leak)

    lines = []
    lines.append(f"\n{'='*90}")
    lines.append(f"  {title}")
    lines.append(f"  Variables: {len(graph.variables)} | Factors: {len(graph.factors)}")
    lines.append(f"{'='*90}")

    header = f"  {'Variable':20s}  {'Prior':>7}  {'Silence':>10}  {'Soft-leak':>10}  {'Noisy-AND':>10}  {'Δ(sil-nAND)':>12}"
    lines.append(header)
    lines.append("  " + "-" * 82)

    for vid in key_vars:
        prior = graph.variables[vid]
        sil = results["silence (current)"][vid]
        soft = results["soft-leak (ε=0.01)"][vid]
        nand = results["noisy-AND (ε=1e-3)"][vid]
        delta = sil - nand
        mark = "  ← DIFF" if abs(delta) > 0.01 else ""
        lines.append(
            f"  {vid:20s}  {prior:7.4f}  {sil:10.6f}  {soft:10.6f}  {nand:10.6f}  {delta:+12.6f}{mark}"
        )

    lines.append(f"{'='*90}")
    return "\n".join(lines)


def run_c4_sweep() -> str:
    """Sweep A's prior from 0.9 to 0.1, check if B's belief decreases (C4 test)."""
    lines = []
    lines.append(f"\n{'='*90}")
    lines.append("  C4 Direct Test: A → ent → B")
    lines.append("  Sweep A's prior from 0.9 down to 0.1.")
    lines.append("  C4 requires: as A's prior decreases, B's belief should decrease.")
    lines.append(f"{'='*90}")

    header = f"  {'π(A)':>7}  {'B(silence)':>12}  {'B(soft-leak)':>14}  {'B(noisy-AND)':>14}  {'C4:silence?':>12}  {'C4:nAND?':>10}"
    lines.append(header)
    lines.append("  " + "-" * 80)

    prev_sil = prev_soft = prev_nand = None

    for pi_a in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        fg = FactorGraph()
        fg.add_variable("A", prior=pi_a)
        fg.add_variable("B", prior=0.5)
        fg.add_factor("f", FactorType.ENTAILMENT,
                      premises=["A"], conclusions=["B"], p=0.999)

        sil = exact_inference_with_leak(fg, None)["B"]
        soft = exact_inference_with_leak(fg, 0.01)["B"]
        nand = exact_inference_with_leak(fg, CROMWELL_EPS)["B"]

        c4_sil = "✓" if (prev_sil is None or sil < prev_sil) else "✗ VIOLATION"
        c4_nand = "✓" if (prev_nand is None or nand < prev_nand) else "✗ VIOLATION"

        lines.append(
            f"  {pi_a:7.2f}  {sil:12.6f}  {soft:14.6f}  {nand:14.6f}  {c4_sil:>12}  {c4_nand:>10}"
        )
        prev_sil, prev_soft, prev_nand = sil, soft, nand

    lines.append(f"{'='*90}")
    return "\n".join(lines)


def run_premise_refuted_test() -> str:
    """Test: A is refuted (prior drops from 0.9 to 0.1), how does B change?

    This simulates discovering that a premise is wrong. Under C4, B should drop.
    With silence, B won't care.
    """
    lines = []
    lines.append(f"\n{'='*90}")
    lines.append("  Premise Refutation Test: A → ent → B")
    lines.append("  A's prior changes from 0.9 (believed) to 0.1 (refuted).")
    lines.append("  B's prior = 0.5 throughout.")
    lines.append(f"{'='*90}")

    for scenario, pi_a in [("A believed (π=0.9)", 0.9), ("A refuted  (π=0.1)", 0.1)]:
        fg = FactorGraph()
        fg.add_variable("A", prior=pi_a)
        fg.add_variable("B", prior=0.5)
        fg.add_factor("f", FactorType.ENTAILMENT,
                      premises=["A"], conclusions=["B"], p=0.999)

        sil = exact_inference_with_leak(fg, None)["B"]
        soft = exact_inference_with_leak(fg, 0.01)["B"]
        nand = exact_inference_with_leak(fg, CROMWELL_EPS)["B"]

        lines.append(f"\n  {scenario}:")
        lines.append(f"    B(silence)   = {sil:.6f}")
        lines.append(f"    B(soft-leak) = {soft:.6f}")
        lines.append(f"    B(noisy-AND) = {nand:.6f}")

    lines.append(f"\n{'='*90}")
    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================

def main():
    output_lines = []
    output_lines.append("# Entailment C4 Model Comparison Experiment")
    output_lines.append("# =========================================")
    output_lines.append(f"# CROMWELL_EPS = {CROMWELL_EPS}")
    output_lines.append("")

    # Experiment 1: C4 direct sweep
    output_lines.append(run_c4_sweep())

    # Experiment 2: Premise refutation
    output_lines.append(run_premise_refuted_test())

    # Experiment 3: Coarse Galileo graph
    coarse = build_coarse_galileo()
    coarse_vars = ["A", "V", "T1", "T2", "A_vac", "O_daily", "O_media", "O_air", "E_theta"]
    output_lines.append(run_comparison(coarse, "Coarse Galileo Factor Graph (§3.3)", coarse_vars))

    # Experiment 4: Fine Galileo graph
    fine = build_fine_galileo()
    fine_vars = ["A", "V", "G", "B_daily", "B_media", "B_air", "B_t1", "B_t2",
                 "T1", "T2", "A_vac", "O_daily", "O_media", "O_air", "E_t1", "E_t2"]
    output_lines.append(run_comparison(fine, "Fine Galileo Factor Graph (§3.5)", fine_vars))

    # Experiment 5: Coarse graph with A refuted (O_daily low)
    coarse_refuted = build_coarse_galileo()
    coarse_refuted.variables["O_daily"] = 0.1  # daily observation refuted
    output_lines.append(run_comparison(
        coarse_refuted,
        "Coarse Galileo — O_daily REFUTED (π=0.1): does A drop more with leak?",
        coarse_vars
    ))

    result = "\n".join(output_lines)
    print(result)

    out_path = Path(__file__).parent / "entailment_leak_results.txt"
    out_path.write_text(result, encoding="utf-8")
    print(f"\n>>> Results saved to {out_path}")


if __name__ == "__main__":
    main()
