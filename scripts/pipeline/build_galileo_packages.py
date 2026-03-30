#!/usr/bin/env python3
"""Build Galileo coarse + fine Gaia IR packages using the new BP engine.

Constructs the Galileo coarse and fine factor graphs (from theory docs
05-formalization) using the new gaia.bp API, converts to gaia.gaia_ir
Pydantic models, runs Junction Tree exact inference, and writes standard
packages.

Usage:
    python scripts/pipeline/build_galileo_packages.py \
        --output-dir tests/fixtures/gaia_ir_packages
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gaia.bp import FactorGraph, FactorType, InferenceEngine
from gaia.bp.factor_graph import CROMWELL_EPS
from gaia.gaia_ir import (
    BeliefState,
    Knowledge,
    KnowledgeType,
    LocalCanonicalGraph,
    Operator,
    OperatorType,
    PriorRecord,
    Strategy,
    StrategyParamRecord,
    StrategyType,
)

PACKAGE_ID = "galileo_falling_bodies"


# ---------------------------------------------------------------------------
# Intermediate representation for graph definition
# ---------------------------------------------------------------------------


@dataclass
class VarDef:
    id: str
    prior: float
    ktype: KnowledgeType
    content: str


@dataclass
class FactorDef:
    """Semantic factor definition — carries both gaia_ir type and BP mapping info."""

    id: str
    semantic_type: (
        str  # "implication" | "conjunction" | "contradiction" | "equivalence" | "noisy_and"
    )
    premises: list[str]
    conclusion: str | None = None
    p: float = 0.999
    relation_var: str | None = None  # auto-created for contradiction/equivalence


@dataclass
class GraphDef:
    name: str
    variables: list[VarDef] = field(default_factory=list)
    factors: list[FactorDef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Galileo coarse graph
# ---------------------------------------------------------------------------


def define_galileo_coarse() -> GraphDef:
    g = GraphDef(name="galileo_coarse")

    g.variables = [
        VarDef("A", 0.5, KnowledgeType.CLAIM, "物体下落速度与其重量成正比：重物比轻物下落更快"),
        VarDef("V", 0.5, KnowledgeType.CLAIM, "在真空中，不同重量的物体以相同速率下落"),
        VarDef(
            "O_daily", 0.90, KnowledgeType.CLAIM, "日常生活中，石头比羽毛下落更快，铁球比木球更快"
        ),
        VarDef("O_media", 0.90, KnowledgeType.CLAIM, "不同介质中轻重物体下落差异越稀薄越不明显"),
        VarDef(
            "O_air",
            0.90,
            KnowledgeType.CLAIM,
            "空气中金球铅球铜球从100腕尺落下，金球领先铜球不超过四指",
        ),
        VarDef(
            "E_theta",
            0.90,
            KnowledgeType.CLAIM,
            "斜面实验：不同重量球滚下距离之比等于时间之比的平方，与重量无关",
        ),
        VarDef("S_vac", 1.0 - CROMWELL_EPS, KnowledgeType.SETTING, "真空是完全没有阻力的介质"),
        VarDef(
            "S_plane",
            1.0 - CROMWELL_EPS,
            KnowledgeType.SETTING,
            "斜面经抛光处理，摩擦和空气阻力可忽略",
        ),
        VarDef("T1", 0.5, KnowledgeType.CLAIM, "假设重物下落更快，绑球复合体HL速度应慢于H单独下落"),
        VarDef(
            "T2", 0.5, KnowledgeType.CLAIM, "假设重物下落更快，复合体HL总重量大于H，速度应快于H"
        ),
        VarDef("A_vac", 0.5, KnowledgeType.CLAIM, "在真空中也应重者更快（A的推论应用于真空）"),
        VarDef("W2_M", 0.5, KnowledgeType.CLAIM, "O_media、O_air与S_vac的合取成立"),
        VarDef("W3_M", 0.5, KnowledgeType.CLAIM, "E_theta与S_plane的合取成立"),
        VarDef("AV_M", 0.5, KnowledgeType.CLAIM, "A与S_vac的合取成立"),
        VarDef("R_T1_T2", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "T1与T2之间存在矛盾关系"),
        VarDef("R_Avac_V", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "A_vac与V之间存在矛盾关系"),
    ]

    g.factors = [
        FactorDef("W1", "noisy_and", ["O_daily"], "A", p=0.70),
        FactorDef("A_to_T1", "implication", ["A"], "T1"),
        FactorDef("A_to_T2", "implication", ["A"], "T2"),
        FactorDef("T1_contra_T2", "contradiction", ["T1", "T2"], relation_var="R_T1_T2"),
        FactorDef("A_and_Svac", "conjunction", ["A", "S_vac"], "AV_M"),
        FactorDef("AVM_to_Avac", "implication", ["AV_M"], "A_vac"),
        FactorDef("Avac_contra_V", "contradiction", ["A_vac", "V"], relation_var="R_Avac_V"),
        FactorDef("W2_conj", "conjunction", ["O_media", "O_air", "S_vac"], "W2_M"),
        FactorDef("W2_soft", "noisy_and", ["W2_M"], "V", p=0.80),
        FactorDef("W3_conj", "conjunction", ["E_theta", "S_plane"], "W3_M"),
        FactorDef("W3_soft", "noisy_and", ["W3_M"], "V", p=0.78),
    ]

    return g


# ---------------------------------------------------------------------------
# Galileo fine graph
# ---------------------------------------------------------------------------


def define_galileo_fine() -> GraphDef:
    g = GraphDef(name="galileo_fine")

    g.variables = [
        VarDef("A", 0.5, KnowledgeType.CLAIM, "物体下落速度与其重量成正比：重物比轻物下落更快"),
        VarDef("V", 0.5, KnowledgeType.CLAIM, "在真空中，不同重量的物体以相同速率下落"),
        VarDef("G", 0.5, KnowledgeType.CLAIM, "空气阻力是造成下落速度差异的主要原因"),
        VarDef(
            "O_daily", 0.90, KnowledgeType.CLAIM, "日常生活中，石头比羽毛下落更快，铁球比木球更快"
        ),
        VarDef("O_media", 0.90, KnowledgeType.CLAIM, "不同介质中轻重物体下落差异越稀薄越不明显"),
        VarDef(
            "O_air",
            0.90,
            KnowledgeType.CLAIM,
            "空气中金球铅球铜球从100腕尺落下，金球领先铜球不超过四指",
        ),
        VarDef(
            "E_theta1",
            0.90,
            KnowledgeType.CLAIM,
            "斜面实验1：不同重量球滚下所经距离之比等于时间之比的平方",
        ),
        VarDef(
            "E_theta2",
            0.90,
            KnowledgeType.CLAIM,
            "斜面实验2：重复实验百余次，结果偏差不超过脉搏一次",
        ),
        VarDef("S_vac", 1.0 - CROMWELL_EPS, KnowledgeType.SETTING, "真空是完全没有阻力的介质"),
        VarDef(
            "S_plane",
            1.0 - CROMWELL_EPS,
            KnowledgeType.SETTING,
            "斜面经抛光处理，摩擦和空气阻力可忽略",
        ),
        VarDef("T1", 0.5, KnowledgeType.CLAIM, "假设重物下落更快，绑球复合体HL速度应慢于H单独下落"),
        VarDef(
            "T2", 0.5, KnowledgeType.CLAIM, "假设重物下落更快，复合体HL总重量大于H，速度应快于H"
        ),
        VarDef("A_vac", 0.5, KnowledgeType.CLAIM, "在真空中也应重者更快（A的推论应用于真空）"),
        VarDef("B_daily", 0.5, KnowledgeType.CLAIM, "A的预测：日常应观察到重者更快"),
        VarDef("B_media", 0.5, KnowledgeType.CLAIM, "G的预测：介质密度影响下落速度差异"),
        VarDef("B_air", 0.5, KnowledgeType.CLAIM, "G的预测：空气中重材料球落差极小"),
        VarDef("B_theta1", 0.5, KnowledgeType.CLAIM, "V的预测1：斜面实验中不同重量球加速一致"),
        VarDef("B_theta2", 0.5, KnowledgeType.CLAIM, "V的预测2：重复实验结果高度一致"),
        VarDef("GV_M", 0.5, KnowledgeType.CLAIM, "G与S_vac的合取成立"),
        VarDef("Vt1_M", 0.5, KnowledgeType.CLAIM, "V与S_plane的合取成立（实验1）"),
        VarDef("Vt2_M", 0.5, KnowledgeType.CLAIM, "V与S_plane的合取成立（实验2）"),
        VarDef("AV_M", 0.5, KnowledgeType.CLAIM, "A与S_vac的合取成立"),
        # Relation variables (high prior: the structural relationships are established)
        VarDef("R_T1_T2", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "T1与T2之间存在矛盾关系"),
        VarDef("R_Avac_V", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "A_vac与V之间存在矛盾关系"),
        VarDef(
            "R_Bdaily_Odaily", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "B_daily与O_daily是等价命题"
        ),
        VarDef(
            "R_Bmedia_Omedia", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "B_media与O_media是等价命题"
        ),
        VarDef("R_Bair_Oair", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "B_air与O_air是等价命题"),
        VarDef(
            "R_Bt1_Et1", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "B_theta1与E_theta1是等价命题"
        ),
        VarDef(
            "R_Bt2_Et2", 1.0 - CROMWELL_EPS, KnowledgeType.CLAIM, "B_theta2与E_theta2是等价命题"
        ),
    ]

    g.factors = [
        # W1 expansion
        FactorDef("A_to_Bdaily", "implication", ["A"], "B_daily"),
        FactorDef(
            "Bdaily_eq_Odaily",
            "equivalence",
            ["B_daily", "O_daily"],
            relation_var="R_Bdaily_Odaily",
        ),
        # Contradictions around A
        FactorDef("A_to_T1", "implication", ["A"], "T1"),
        FactorDef("A_to_T2", "implication", ["A"], "T2"),
        FactorDef("T1_contra_T2", "contradiction", ["T1", "T2"], relation_var="R_T1_T2"),
        FactorDef("A_Svac_conj", "conjunction", ["A", "S_vac"], "AV_M"),
        FactorDef("AVM_to_Avac", "implication", ["AV_M"], "A_vac"),
        FactorDef("Avac_contra_V", "contradiction", ["A_vac", "V"], relation_var="R_Avac_V"),
        # W2 expansion
        FactorDef("G_to_Bmedia", "implication", ["G"], "B_media"),
        FactorDef(
            "Bmedia_eq_Omedia",
            "equivalence",
            ["B_media", "O_media"],
            relation_var="R_Bmedia_Omedia",
        ),
        FactorDef("G_to_Bair", "implication", ["G"], "B_air"),
        FactorDef("Bair_eq_Oair", "equivalence", ["B_air", "O_air"], relation_var="R_Bair_Oair"),
        FactorDef("G_Svac_conj", "conjunction", ["G", "S_vac"], "GV_M"),
        FactorDef("GVM_to_V", "implication", ["GV_M"], "V"),
        # W3 expansion
        FactorDef("V_Splane_conj_1", "conjunction", ["V", "S_plane"], "Vt1_M"),
        FactorDef("Vt1_to_Btheta1", "implication", ["Vt1_M"], "B_theta1"),
        FactorDef(
            "Btheta1_eq_Etheta1", "equivalence", ["B_theta1", "E_theta1"], relation_var="R_Bt1_Et1"
        ),
        FactorDef("V_Splane_conj_2", "conjunction", ["V", "S_plane"], "Vt2_M"),
        FactorDef("Vt2_to_Btheta2", "implication", ["Vt2_M"], "B_theta2"),
        FactorDef(
            "Btheta2_eq_Etheta2", "equivalence", ["B_theta2", "E_theta2"], relation_var="R_Bt2_Et2"
        ),
    ]

    return g


# ---------------------------------------------------------------------------
# Conversion: GraphDef → FactorGraph (for BP)
# ---------------------------------------------------------------------------

BP_TYPE_MAP = {
    "implication": FactorType.ENTAILMENT,
    "conjunction": FactorType.ENTAILMENT,
    "noisy_and": FactorType.INDUCTION,
    "contradiction": FactorType.CONTRADICTION,
    "equivalence": FactorType.EQUIVALENCE,
}


def build_factor_graph(gdef: GraphDef) -> FactorGraph:
    fg = FactorGraph()
    for v in gdef.variables:
        fg.add_variable(v.id, v.prior)

    for f in gdef.factors:
        bp_type = BP_TYPE_MAP[f.semantic_type]
        if f.semantic_type in ("contradiction", "equivalence"):
            fg.add_factor(f.id, bp_type, f.premises, [], 0.5, relation_var=f.relation_var)
        else:
            fg.add_factor(f.id, bp_type, f.premises, [f.conclusion] if f.conclusion else [], f.p)
    return fg


# ---------------------------------------------------------------------------
# Conversion: GraphDef → gaia.gaia_ir models
# ---------------------------------------------------------------------------

IR_OP_MAP = {
    "implication": OperatorType.IMPLICATION,
    "conjunction": OperatorType.CONJUNCTION,
    "contradiction": OperatorType.CONTRADICTION,
    "equivalence": OperatorType.EQUIVALENCE,
}


def build_gaia_ir(gdef: GraphDef):
    """Convert GraphDef to gaia.gaia_ir models."""
    knowledges = []
    for v in gdef.variables:
        knowledges.append(
            Knowledge(id=f"lcn_{v.id}", type=v.ktype, content=v.content, package_id=PACKAGE_ID)
        )

    operators: list[Operator] = []
    strategies: list[Strategy] = []

    for f in gdef.factors:
        if f.semantic_type in IR_OP_MAP:
            op_type = IR_OP_MAP[f.semantic_type]
            variables = [f"lcn_{p}" for p in f.premises]
            conclusion = f"lcn_{f.conclusion}" if f.conclusion else None
            if conclusion and conclusion not in variables:
                variables.append(conclusion)
            operators.append(
                Operator(
                    operator_id=f"lco_{f.id}",
                    scope="local",
                    operator=op_type,
                    variables=variables,
                    conclusion=conclusion,
                )
            )
        elif f.semantic_type == "noisy_and":
            claim_premises = [f"lcn_{p}" for p in f.premises]
            strategies.append(
                Strategy(
                    strategy_id=f"lcs_{f.id}",
                    scope="local",
                    type=StrategyType.NOISY_AND,
                    premises=claim_premises,
                    conclusion=f"lcn_{f.conclusion}" if f.conclusion else None,
                )
            )

    lcg = LocalCanonicalGraph(knowledges=knowledges, operators=operators, strategies=strategies)
    return lcg, strategies


def build_parameterization(gdef: GraphDef, lcg: LocalCanonicalGraph, strategies: list[Strategy]):
    source_id = f"pipeline:{PACKAGE_ID}"
    priors = []
    for v in gdef.variables:
        if v.ktype == KnowledgeType.CLAIM:
            priors.append(PriorRecord(gcn_id=f"lcn_{v.id}", value=v.prior, source_id=source_id))

    strat_params = []
    factor_map = {f.id: f for f in gdef.factors}
    for s in strategies:
        fid = s.strategy_id.replace("lcs_", "")
        fdef = factor_map.get(fid)
        p = fdef.p if fdef else 0.5
        strat_params.append(
            StrategyParamRecord(
                strategy_id=s.strategy_id,
                conditional_probabilities=[p],
                source_id=source_id,
            )
        )

    return priors, strat_params


# ---------------------------------------------------------------------------
# Main: build + infer + write
# ---------------------------------------------------------------------------


def build_one_package(gdef: GraphDef, output_dir: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"Building: {gdef.name}")
    print(f"{'=' * 60}")

    lcg, strategies = build_gaia_ir(gdef)
    priors, strat_params = build_parameterization(gdef, lcg, strategies)

    fg = build_factor_graph(gdef)
    errors = fg.validate()
    if errors:
        print(f"  WARNING: {errors}")

    engine = InferenceEngine()
    result = engine.run(fg)

    kn_type_map = {v.id: v.ktype for v in gdef.variables}
    claim_beliefs = {
        f"lcn_{vid}": round(b, 6)
        for vid, b in result.beliefs.items()
        if kn_type_map.get(vid) == KnowledgeType.CLAIM
    }

    now = datetime.now(timezone.utc)
    belief_state = BeliefState(
        bp_run_id=str(uuid.uuid4()),
        created_at=now,
        resolution_policy="latest",
        prior_cutoff=now,
        beliefs=claim_beliefs,
        compilation_summary={
            "method_used": result.method_used,
            "treewidth": result.treewidth,
            "elapsed_ms": round(result.elapsed_ms, 2),
            "is_exact": result.is_exact,
        },
        converged=result.bp_result.diagnostics.converged,
        iterations=result.bp_result.diagnostics.iterations_run,
        max_residual=result.bp_result.diagnostics.max_change_at_stop,
    )

    pkg_dir = output_dir / gdef.name / "gaia_ir"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    def _dump(obj, filename):
        data = obj.model_dump(mode="json")
        (pkg_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )

    _dump(lcg, "local_canonical_graph.json")

    param_data = {
        "scope": "local",
        "ir_hash": lcg.ir_hash,
        "priors": [p.model_dump(mode="json") for p in priors],
        "strategy_params": [sp.model_dump(mode="json") for sp in strat_params],
    }
    (pkg_dir / "local_parameterization.json").write_text(
        json.dumps(param_data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )

    _dump(belief_state, "local_belief_state.json")

    n_k = len(lcg.knowledges)
    n_o = len(lcg.operators)
    n_s = len(lcg.strategies)
    print(
        f"  {n_k} knowledges, {n_o} operators, {n_s} strategies"
        f" | BP: {result.method_used} (exact={result.is_exact},"
        f" {result.elapsed_ms:.1f}ms)"
    )
    print(f"  Output: {pkg_dir}")

    for v in gdef.variables:
        if v.ktype != KnowledgeType.CLAIM:
            continue
        b = claim_beliefs.get(f"lcn_{v.id}", v.prior)
        delta = b - v.prior
        arrow = "↑" if delta > 0.01 else "↓" if delta < -0.01 else "="
        print(f"    {arrow} {v.id:20s} prior={v.prior:.3f} → belief={b:.4f} ({delta:+.4f})")


def main():
    parser = argparse.ArgumentParser(description="Build Galileo coarse + fine Gaia IR packages")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures/gaia_ir_packages"),
        help="Output directory (default: tests/fixtures/gaia_ir_packages)",
    )
    args = parser.parse_args()

    build_one_package(define_galileo_coarse(), args.output_dir)
    build_one_package(define_galileo_fine(), args.output_dir)

    print(f"\nDone. Packages written to {args.output_dir}/")


if __name__ == "__main__":
    main()
