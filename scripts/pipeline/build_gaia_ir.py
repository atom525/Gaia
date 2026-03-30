#!/usr/bin/env python3
"""Convert old graph_ir format → new gaia.gaia_ir format + run new BP engine.

Takes a Gaia Language Typst package that has already been compiled to old
graph_ir/ format (by build_graph_ir.py) and produces a new gaia_ir/ package:

  gaia_ir/local_canonical_graph.json   — Knowledge + Operator + Strategy
  gaia_ir/local_parameterization.json  — PriorRecord + StrategyParamRecord
  gaia_ir/local_belief_state.json      — BP posteriors

Usage:
    # First, compile Typst to old format:
    python scripts/pipeline/build_graph_ir.py tests/fixtures/gaia_language_packages/galileo_falling_bodies_v4

    # Then, convert + run inference:
    python scripts/pipeline/build_gaia_ir.py tests/fixtures/gaia_language_packages/galileo_falling_bodies_v4
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gaia.bp import InferenceEngine, FactorGraph, FactorType
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


OLD_TYPE_TO_KNOWLEDGE_TYPE: dict[str, KnowledgeType] = {
    "claim": KnowledgeType.CLAIM,
    "setting": KnowledgeType.SETTING,
    "question": KnowledgeType.QUESTION,
    "action": KnowledgeType.CLAIM,
    "contradiction": KnowledgeType.CLAIM,
}

OLD_FACTOR_TO_OPERATOR: dict[str, OperatorType] = {
    "contradiction": OperatorType.CONTRADICTION,
    "equivalence": OperatorType.EQUIVALENCE,
}


def _load_old_graph(pkg_dir: Path):
    """Load old-format local_canonical_graph.json + local_parameterization.json."""
    graph_dir = pkg_dir / "graph_ir"
    lcg = json.loads((graph_dir / "local_canonical_graph.json").read_text())
    params = json.loads((graph_dir / "local_parameterization.json").read_text())
    return lcg, params


def _convert_knowledge(old_node: dict, package_id: str) -> Knowledge:
    """Convert old LocalCanonicalNode to new Knowledge."""
    old_type = old_node["knowledge_type"]
    new_type = OLD_TYPE_TO_KNOWLEDGE_TYPE.get(old_type)
    if new_type is None:
        raise ValueError(f"Unknown knowledge_type: {old_type}")

    metadata = old_node.get("metadata") or {}
    source_refs = old_node.get("source_refs", [])
    if source_refs:
        metadata["source_refs"] = source_refs

    return Knowledge(
        id=old_node["local_canonical_id"],
        type=new_type,
        content=old_node["representative_content"],
        parameters=[],
        metadata=metadata or None,
        package_id=package_id,
    )


def _convert_graph(
    old_lcg: dict, old_params: dict
) -> tuple[LocalCanonicalGraph, list[PriorRecord], list[StrategyParamRecord]]:
    """Convert old format to new gaia.gaia_ir models."""
    package_id = old_lcg["package"]

    knowledges = [_convert_knowledge(n, package_id) for n in old_lcg["knowledge_nodes"]]
    kn_type_map = {k.id: k.type for k in knowledges}

    operators: list[Operator] = []
    strategies: list[Strategy] = []

    for factor in old_lcg["factor_nodes"]:
        ftype = factor["type"]
        if ftype in OLD_FACTOR_TO_OPERATOR:
            op_type = OLD_FACTOR_TO_OPERATOR[ftype]
            variables = list(factor["premises"])
            operators.append(
                Operator(
                    operator_id=factor["factor_id"].replace("f_", "lco_"),
                    scope="local",
                    operator=op_type,
                    variables=variables,
                    conclusion=None,
                )
            )
        else:
            all_premises = list(factor["premises"])
            conclusion = factor.get("conclusion")
            claim_premises = [p for p in all_premises if kn_type_map.get(p) == KnowledgeType.CLAIM]
            bg_premises = [p for p in all_premises if kn_type_map.get(p) != KnowledgeType.CLAIM]

            strategies.append(
                Strategy(
                    strategy_id=factor["factor_id"].replace("f_", "lcs_"),
                    scope="local",
                    type=StrategyType.NOISY_AND,
                    premises=claim_premises,
                    conclusion=conclusion,
                    background=bg_premises or None,
                )
            )

    lcg = LocalCanonicalGraph(knowledges=knowledges, operators=operators, strategies=strategies)

    priors: list[PriorRecord] = []
    strat_params: list[StrategyParamRecord] = []
    source_id = f"pipeline:{package_id}"

    for kid, pval in old_params.get("node_priors", {}).items():
        if kn_type_map.get(kid) == KnowledgeType.CLAIM:
            priors.append(PriorRecord(gcn_id=kid, value=pval, source_id=source_id))

    for fid, fp in old_params.get("factor_parameters", {}).items():
        strat_id = fid.replace("f_", "lcs_")
        cp = fp.get("conditional_probability", 0.5)
        strat_params.append(
            StrategyParamRecord(
                strategy_id=strat_id,
                conditional_probabilities=[cp],
                source_id=source_id,
            )
        )

    return lcg, priors, strat_params


def _build_factor_graph(old_lcg: dict, old_params: dict) -> FactorGraph:
    """Build gaia.bp.FactorGraph from old format for inference."""
    fg = FactorGraph()

    for node in old_lcg["knowledge_nodes"]:
        nid = node["local_canonical_id"]
        prior = old_params.get("node_priors", {}).get(nid, 0.5)
        fg.add_variable(nid, prior)

    for factor in old_lcg["factor_nodes"]:
        fid = factor["factor_id"]
        ftype = factor["type"]
        premises = list(factor.get("premises", []))
        conclusion = factor.get("conclusion")
        cp = (
            old_params.get("factor_parameters", {}).get(fid, {}).get("conditional_probability", 0.5)
        )

        if ftype == "contradiction":
            if conclusion:
                fg.add_factor(
                    fid, FactorType.CONTRADICTION, premises, [], cp, relation_var=conclusion
                )
            else:
                fg.add_factor(fid, FactorType.CONTRADICTION, premises, [], cp, relation_var=None)
        elif ftype == "equivalence":
            if conclusion:
                fg.add_factor(
                    fid, FactorType.EQUIVALENCE, premises, [], cp, relation_var=conclusion
                )
        else:
            fg.add_factor(
                fid,
                FactorType.ENTAILMENT,
                premises,
                [conclusion] if conclusion else [],
                cp,
            )

    return fg


def build_gaia_ir_package(pkg_dir: Path, output_dir: Path | None = None) -> bool:
    """Convert old graph_ir + run new BP, write new gaia_ir package."""
    graph_dir = pkg_dir / "graph_ir"
    if not (graph_dir / "local_canonical_graph.json").exists():
        print(f"  SKIP: no graph_ir/local_canonical_graph.json in {pkg_dir.name}")
        return False

    old_lcg, old_params = _load_old_graph(pkg_dir)

    lcg, priors, strat_params = _convert_graph(old_lcg, old_params)

    fg = _build_factor_graph(old_lcg, old_params)
    errors = fg.validate()
    if errors:
        print(f"  WARNING: FactorGraph validation errors: {errors}")

    engine = InferenceEngine()
    result = engine.run(fg)

    kn_type_map = {k.id: k.type for k in lcg.knowledges}
    claim_beliefs = {
        vid: round(b, 6)
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

    out_dir = output_dir or (pkg_dir / "gaia_ir")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _dump(obj, filename: str):
        data = obj.model_dump(mode="json")
        (out_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )

    _dump(lcg, "local_canonical_graph.json")

    param_data = {
        "scope": "local",
        "ir_hash": lcg.ir_hash,
        "priors": [p.model_dump(mode="json") for p in priors],
        "strategy_params": [sp.model_dump(mode="json") for sp in strat_params],
    }
    (out_dir / "local_parameterization.json").write_text(
        json.dumps(param_data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )

    _dump(belief_state, "local_belief_state.json")

    n_k = len(lcg.knowledges)
    n_o = len(lcg.operators)
    n_s = len(lcg.strategies)
    print(
        f"  OK: {n_k} knowledges, {n_o} operators, {n_s} strategies"
        f" | BP: {result.method_used} (exact={result.is_exact},"
        f" {result.elapsed_ms:.1f}ms)"
    )

    for vid, b in sorted(claim_beliefs.items()):
        kn = next((k for k in lcg.knowledges if k.id == vid), None)
        name = (kn.content or vid)[:50] if kn else vid
        prior_val = next((p.value for p in priors if p.gcn_id == vid), 0.5)
        delta = b - prior_val
        arrow = "↑" if delta > 0.01 else "↓" if delta < -0.01 else "="
        print(f"    {arrow} {name:50s} prior={prior_val:.3f} → belief={b:.4f} ({delta:+.4f})")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build Gaia IR from old graph_ir + run new BP")
    parser.add_argument("pkg_dirs", type=Path, nargs="+", help="Package directories")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory override")
    args = parser.parse_args()

    succeeded = 0
    for pkg_dir in args.pkg_dirs:
        if not pkg_dir.is_dir():
            continue
        print(f"Processing: {pkg_dir.name}")
        if build_gaia_ir_package(pkg_dir, args.output_dir):
            succeeded += 1

    print(f"\nDone: {succeeded}/{len(args.pkg_dirs)} packages.")


if __name__ == "__main__":
    main()
