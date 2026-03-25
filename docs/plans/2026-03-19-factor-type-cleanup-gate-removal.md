# Factor Type Cleanup + Gate Removal Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify factor types to 5 clean types, remove gate variables, make relation nodes full BP participants, eliminate `metadata.edge_type` indirection.

**Architecture:** (1) Replace 4 old types (`reasoning`, `mutex_constraint`, `equiv_constraint` + dead `retraction`) with 5 clean types (`infer`, `instantiation`, `abstraction`, `contradiction`, `equivalence`); (2) Move relation node from `conclusion` (read-only gate) to `premises` (full BP participant); (3) Remove `metadata.edge_type` — `FactorNode.type` is the single source of truth for BP semantics.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy (BP), LanceDB, Neo4j/Kuzu

**Spec:** `docs/superpowers/specs/2026-03-16-noisy-and-leak-unified-factor-design.md`, `docs/foundations/theory/inference-theory.md` §4

---

## Context

The current factor system has two problems:

1. **Gate variables** on contradiction/equivalence factors violate Jaynes consistency — blocking bidirectional info flow means BP can't "question the relationship itself" when both claims have strong evidence. v2.0 spec says: make relation nodes normal participants, loopy BP + damping handles feedback.

2. **Naming confusion** — `FactorNode.type` has 4 values (`reasoning`, `instantiation`, `mutex_constraint`, `equiv_constraint`) but actual semantics are carried by `metadata.edge_type` (8 values: `deduction`, `induction`, `abstraction`, `retraction`, `instantiation`, `contradiction`, `relation_contradiction`, `relation_equivalence`). Two layers expressing the same thing with different names.

## Type Migration Table

| Old `type` | Old `edge_type` | New `type` | Notes |
|------------|-----------------|------------|-------|
| `reasoning` | `deduction` | `infer` | Rename |
| `reasoning` | `induction` | `infer` | Merge (p<1.0 is a parameter constraint, not a type) |
| `reasoning` | `abstraction` | `abstraction` | Promote to top-level type |
| `reasoning` | `retraction` | — | **Delete** (dead code, `_build_retraction_factor` never called) |
| `instantiation` | `instantiation` | `instantiation` | Keep |
| `mutex_constraint` | `relation_contradiction` | `contradiction` | Rename + remove gate |
| `equiv_constraint` | `relation_equivalence` | `equivalence` | Rename + remove gate |

After migration, `metadata.edge_type` is **removed entirely**. `FactorNode.type` is the only field BP needs.

## Structural Change: Gate Removal

**Before (v1.0):**
```
Factor: type=mutex_constraint, premises=[A, B], conclusion=gate_123
BP: reads gate_123 belief as constraint strength, never writes back
```

**After (v2.0):**
```
Factor: type=contradiction, premises=[C_contra, A, B], conclusion=None
BP: C_contra is normal participant, receives messages bidirectionally
```

For curation-created constraints: `create_constraint()` also creates a `GlobalCanonicalNode` for the relation (type=contradiction/equivalence), consistent with Gaia Language package behavior.

## File Structure

### Models
- Modify: `libs/storage/models.py` — FactorNode Literal, conclusion optional, remove is_gate_factor/bp_participant_ids
- Modify: `libs/graph_ir/models.py` — conclusion optional

### Inference Engine
- Modify: `libs/inference/factor_graph.py` — remove gate_var param
- Modify: `libs/inference/bp.py` — new potentials keyed on type not edge_type, remove gate_beliefs

### Graph IR
- Modify: `libs/graph_ir/adapter.py` — remove gate_var, use type directly
- Modify: `libs/graph_ir/build.py` — new types, relation in premises, remove edge_type metadata, delete `_build_retraction_factor`

### Language
- Modify: `libs/lang/compiler.py` — remove gate_var, relation in premises
- Modify: `libs/lang/runtime.py` — remove gate_var

### Curation
- Modify: `libs/curation/operations.py` — new types, create relation node, handle None conclusion
- Modify: `libs/curation/scheduler.py` — new types
- Modify: `libs/curation/structure.py` — handle None conclusion
- Modify: `libs/curation/cleanup.py` — handle create_constraint returning (factor, node)

### Storage
- Modify: `libs/storage/neo4j_graph_store.py` — remove is_gate
- Modify: `libs/storage/kuzu_graph_store.py` — remove is_gate

### Scripts
- Modify: `scripts/pipeline/run_local_bp.py` — remove gate_var
- Modify: `scripts/pipeline/run_global_bp.py` — remove gate_var

### Tests (~20 files)

---

## Chunk 1: Models + Inference Engine

### Task 1: Update FactorNode models

**Files:**
- Modify: `libs/storage/models.py:150-173`
- Modify: `libs/graph_ir/models.py:45-52`
- Test: `tests/libs/storage/test_models.py`

- [ ] **Step 1: Update storage FactorNode**

```python
class FactorNode(BaseModel):
    """Persistent factor from Graph IR."""
    factor_id: str
    type: Literal["infer", "instantiation", "abstraction", "contradiction", "equivalence"]
    premises: list[str] = []
    contexts: list[str] = []
    conclusion: str | None = None
    package_id: str
    source_ref: SourceRef | None = None
    metadata: dict | None = None
```

Delete `is_gate_factor` and `bp_participant_ids` properties.

- [ ] **Step 2: Update graph_ir FactorNode**

Change `conclusion: str` to `conclusion: str | None = None`.

- [ ] **Step 3: Update test_models.py**

Rename types in test data. Remove is_gate_factor / bp_participant_ids tests. Update constraint factors: `conclusion=None`, relation node in premises.

- [ ] **Step 4: Run tests, fix compilation errors**

```bash
pytest tests/libs/storage/test_models.py -v
```

- [ ] **Step 5: Commit**

### Task 2: Update BP engine

**Files:**
- Modify: `libs/inference/factor_graph.py:48-78`
- Modify: `libs/inference/bp.py:69-221`
- Test: `tests/libs/inference/`

- [ ] **Step 1: Remove gate_var from FactorGraph.add_factor()**

Delete `gate_var` parameter and storage.

- [ ] **Step 2: Update `_evaluate_potential` — key on type, not edge_type**

The function currently uses `edge_type` strings. Change to use the same strings as `FactorNode.type`:

```python
# contradiction: relation node is premises[0], claims are premises[1:]
if edge_type == "contradiction":
    c_val = assignment[premise_ids[0]]
    if c_val == 0:
        return 1.0
    all_claims_true = all(assignment[c] == 1 for c in premise_ids[1:])
    return CROMWELL_EPS if all_claims_true else 1.0

# equivalence: relation node is premises[0], two claims are premises[1:3]
if edge_type == "equivalence":
    c_val = assignment[premise_ids[0]]
    if c_val == 0:
        return 1.0
    a_val = assignment[premise_ids[1]]
    b_val = assignment[premise_ids[2]]
    return (1.0 - CROMWELL_EPS) if a_val == b_val else CROMWELL_EPS

# retraction: delete entirely (dead code)

# "infer", "instantiation", "abstraction": all use standard reasoning potential
# (existing deduction handler, just accept these type names)
```

- [ ] **Step 3: Remove gate_beliefs from `_compute_factor_to_var`**

Remove `gate_beliefs` parameter and gate substitution block. Update all call sites in `_run_core`.

- [ ] **Step 4: Run inference tests**

```bash
pytest tests/libs/inference/ -v
```

- [ ] **Step 5: Commit**

---

## Chunk 2: Graph IR + Language Layer

### Task 3: Update graph_ir adapter

**Files:**
- Modify: `libs/graph_ir/adapter.py:40-81`

- [ ] **Step 1: Simplify adapt_local_graph**

Remove gate_var branch. Use `factor.type` directly as edge_type:

```python
if factor.type in ("contradiction", "equivalence"):
    factor_graph.add_factor(
        edge_id=factor_index,
        premises=premise_ids,
        conclusions=[],
        probability=1.0,
        edge_type=factor.type,
    )
elif factor.type == "instantiation":
    factor_graph.add_factor(
        edge_id=factor_index,
        premises=premise_ids,
        conclusions=[local_id_to_var_id[factor.conclusion]],
        probability=1.0,
        edge_type=factor.type,
    )
# reasoning → infer/abstraction use params.conditional_probability
```

- [ ] **Step 2: Run graph_ir tests**
- [ ] **Step 3: Commit**

### Task 4: Update graph_ir build

**Files:**
- Modify: `libs/graph_ir/build.py:380-527`

- [ ] **Step 1: Update `_build_reasoning_factor`**

Change `type="reasoning"` to `type="infer"`. Remove `metadata={"edge_type": ...}` — no longer needed.

For chains with `chain.edge_type == "abstraction"`, use `type="abstraction"` instead.

- [ ] **Step 2: Delete `_build_retraction_factor`**

Dead code — never called.

- [ ] **Step 3: Update `_build_relation_factors`**

- Type: `"contradiction"` / `"equivalence"`
- Relation node moves from `conclusion` to `premises[0]`
- `conclusion=None`
- Remove `metadata={"edge_type": ...}`

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

### Task 5: Update language compiler + runtime

**Files:**
- Modify: `libs/lang/compiler.py:147-203`
- Modify: `libs/lang/runtime.py`

- [ ] **Step 1: Update `_compile_relation`**

- `edge_type = rel.type` (was `f"relation_{rel.type}"`)
- Relation variable in premises: `"premises": [var_name] + related_vars`
- Remove `"gate_var": var_name`

- [ ] **Step 2: Update runtime — remove gate_var from add_factor call**
- [ ] **Step 3: Run lang tests**

```bash
pytest tests/libs/lang/ -v
```

- [ ] **Step 4: Commit**

---

## Chunk 3: Curation + Storage + Scripts

### Task 6: Update curation operations + cleanup

**Files:**
- Modify: `libs/curation/operations.py`
- Modify: `libs/curation/cleanup.py`
- Modify: `libs/curation/structure.py`

- [ ] **Step 1: Update `create_constraint()` — return (factor, relation_node)**

```python
def create_constraint(node_a_id, node_b_id, constraint_type) -> tuple[FactorNode, GlobalCanonicalNode]:
    relation_node = GlobalCanonicalNode(
        global_canonical_id=f"gcn_rel_{digest}",
        knowledge_type=constraint_type,
        representative_content=f"{constraint_type} between {node_a_id} and {node_b_id}",
    )
    factor = FactorNode(
        type=constraint_type,  # "contradiction" or "equivalence"
        premises=[relation_node.global_canonical_id, node_a_id, node_b_id],
        conclusion=None,
        package_id="__curation__",
    )
    return factor, relation_node
```

- [ ] **Step 2: Update callers in cleanup.py**

Handle tuple return, add relation_node to node_map.

- [ ] **Step 3: Handle None conclusion in merge_nodes and structure.py**

- [ ] **Step 4: Update `create_abstraction()` — type="abstraction" for instantiation factors? No — instantiation factors stay as "instantiation"**

Actually, review: `create_abstraction` creates instantiation factors (`type="instantiation"`). The type name doesn't change. But remove `metadata={"edge_type": "instantiation"}` since we're eliminating edge_type.

- [ ] **Step 5: Run curation tests**
- [ ] **Step 6: Commit**

### Task 7: Update scheduler + storage + scripts

**Files:**
- Modify: `libs/curation/scheduler.py:30-70`
- Modify: `libs/storage/neo4j_graph_store.py`
- Modify: `libs/storage/kuzu_graph_store.py`
- Modify: `scripts/pipeline/run_local_bp.py`
- Modify: `scripts/pipeline/run_global_bp.py`

- [ ] **Step 1: Update `_build_factor_graph_from_storage`**

```python
if factor.type in ("contradiction", "equivalence"):
    graph.add_factor(fi, premises_int, [], 0.9, factor.type)
```

Use `factor.type` as edge_type directly. Handle `conclusion=None`.

- [ ] **Step 2: Remove is_gate from storage backends**

Neo4j/Kuzu: remove `is_gate` property. Handle None conclusion in relationship creation.

- [ ] **Step 3: Update scripts — remove gate_var handling**

- [ ] **Step 4: Run storage tests + full test suite**
- [ ] **Step 5: Commit**

---

## Chunk 4: Tests + Docs

### Task 8: Update all remaining tests

Global replacements across ~20 test files:

| Find | Replace |
|------|---------|
| `type="reasoning"` | `type="infer"` (or `"abstraction"` where edge_type was abstraction) |
| `type="mutex_constraint"` | `type="contradiction"` |
| `type="equiv_constraint"` | `type="equivalence"` |
| `"edge_type": "deduction"` | remove (metadata.edge_type eliminated) |
| `"edge_type": "relation_contradiction"` | remove |
| `"edge_type": "relation_equivalence"` | remove |
| `is_gate_factor` | delete assertions |
| `gate_var` | remove from factor dicts |

For constraint factor test data: relation node from `conclusion` to `premises[0]`, `conclusion=None`.

Key test files:
- `tests/libs/curation/conftest.py`
- `tests/libs/curation/test_coverage_gaps.py`
- `tests/libs/curation/test_integration.py`
- `tests/libs/curation/test_physics_graph.py`
- `tests/libs/global_graph/test_canonicalize.py`
- `tests/libs/storage/test_lance_content.py`
- `tests/libs/storage/test_three_write.py`
- `tests/libs/lang/test_relation_compiler.py`
- `tests/libs/lang/test_relation_runtime.py`
- `tests/libs/lang/test_integration.py`
- `tests/fixtures/global_graph/global_graph.json`

- [ ] **Step 1: Search-and-replace types across all test files**
- [ ] **Step 2: Fix constraint factor test data (conclusion→premises)**
- [ ] **Step 3: Remove edge_type from metadata in test fixtures**
- [ ] **Step 4: Remove gate_var and is_gate assertions**
- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -x --ignore=tests/libs/storage/test_vector_store.py -q
```

- [ ] **Step 6: Commit**

### Task 9: Update docs

**Files:**
- Modify: `docs/foundations/bp-on-graph-ir.md` — remove Gate Semantics section, update constraint descriptions
- Modify: `CLAUDE.md` — if factor type references exist

- [ ] **Step 1: Update bp-on-graph-ir.md**
- [ ] **Step 2: Commit**

---

## Verification

```bash
# Full test suite
pytest tests/ -x --ignore=tests/libs/storage/test_vector_store.py -v

# Lint
ruff check libs/ tests/ scripts/
ruff format --check libs/ tests/ scripts/

# Smoke test
uv run python scripts/smoke_curation.py
```

## Summary

| Before | After |
|--------|-------|
| `type="reasoning"` | `type="infer"` or `type="abstraction"` |
| `type="mutex_constraint"` | `type="contradiction"` |
| `type="equiv_constraint"` | `type="equivalence"` |
| `type="instantiation"` | `type="instantiation"` (unchanged) |
| `metadata.edge_type=*` | **deleted** — type is the single source of truth |
| `conclusion=gate_id` (constraints) | `conclusion=None`, relation in `premises[0]` |
| `gate_var` in add_factor | **deleted** |
| `is_gate_factor` property | **deleted** |
| `_build_retraction_factor` | **deleted** (dead code) |
