# Typst → Graph IR Compiler Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a compiler that takes Typst loader output and produces Graph IR (`RawGraph` → `LocalCanonicalGraph`), replacing the YAML-era `build_raw_graph()`.

**Architecture:** Phase 1 only — new `typst_compiler.py` + extracted `build_utils.py` + Typst `export-graph()` changes + `pipeline_build()` rewrite. YAML code is NOT deleted in this phase (Phase 2). All work happens on branch `feat/typst-gaia-lang-poc` which has a worktree at `.worktrees/typst-gaia-lang-poc/`.

**Tech Stack:** Python 3.12+, Typst, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-03-19-typst-graph-ir-compiler-design.md`

**Base branch:** `feat/typst-gaia-lang-poc`

**Important context:**
- The Typst library and test files exist in the worktree at `.worktrees/typst-gaia-lang-poc/`, not on the main checkout. All implementation work MUST happen in that worktree.
- `declarations.typ` already emits `module` per node (line 14 of `.worktrees/typst-gaia-lang-poc/libs/typst/gaia-lang/declarations.typ`).
- `module.typ`'s `#package()` function does NOT currently store name/version in state — it only renders the title block and stores exports. This needs to change.
- Existing Graph IR models (`libs/graph_ir/models.py`) are unchanged — `RawGraph`, `RawKnowledgeNode`, `FactorNode`, `SourceRef`, etc. are used as-is.
- `extract_parameters()` is extracted into `build_utils.py` for future use but is NOT called by `typst_compiler.py` in Phase 1 — the schema parameter mechanism is a future extension.
- `derive_local_parameterization()` is intentionally NOT called in `pipeline_build_typst()` — in v3, priors (π) and conditional probabilities (p) come from the review stage, not build time. This is a Phase 2 concern.
- Node names within a single Typst package are unique (Typst's `#use()` imports use aliases). The compiler relies on this: `name_to_raw_id` is a flat dict keyed by name. If a future Typst package produces duplicate names, the compiler should raise an error.

---

## File Map

### Typst Library (modify)
| File | Responsibility | Change |
|------|---------------|--------|
| `.worktrees/typst-gaia-lang-poc/libs/typst/gaia-lang/module.typ` | State + export-graph | Add `_gaia_package_name` / `_gaia_package_version` state; `#package()` stores them; `export-graph()` includes them in metadata output |

### Python — New Files
| File | Responsibility |
|------|---------------|
| `libs/graph_ir/build_utils.py` | Source-agnostic IR utilities extracted from `build.py`: ID generation (`_raw_node_id`, `_local_canonical_id`, `_factor_id`, `_extract_parameters`), `build_singleton_local_graph()`, `CanonicalizationResult` |
| `libs/graph_ir/typst_compiler.py` | `compile_typst_to_raw_graph(graph_data: dict) → RawGraph` — the core compiler |

### Python — Modify
| File | Change |
|------|--------|
| `libs/lang/typst_loader.py` | Ensure `package` and `version` keys are passed through from Typst metadata (currently they're not in the output because Typst doesn't emit them yet) |
| `libs/pipeline.py` | Rewrite `pipeline_build()` to use Typst path; simplify `BuildResult` |

### Python — No Change
| File | Why |
|------|-----|
| `libs/graph_ir/models.py` | IR models are source-agnostic |
| `libs/graph_ir/adapter.py` | Downstream of RawGraph, unchanged |
| `libs/graph_ir/serialize.py` | Operates on graph_ir.models only |
| `libs/graph_ir/build.py` | Left intact for Phase 2 cleanup (still used by YAML path) |
| `libs/inference/` | Downstream, unchanged |

### Tests
| File | Purpose |
|------|---------|
| `tests/libs/graph_ir/test_build_utils.py` | Tests for extracted utility functions |
| `tests/libs/graph_ir/test_typst_compiler.py` | Tests for `compile_typst_to_raw_graph()` |
| `tests/libs/lang/test_typst_loader.py` | Add test for `package`/`version` fields |

### Housekeeping
| Action | Files |
|--------|-------|
| Move to `future/` | `libs/lang/plausible_core.py`, `tests/libs/lang/test_plausible_core.py` |

---

## Chunk 1: Extract build_utils.py from build.py

### Task 1: Create build_utils.py with ID generation functions

**Files:**
- Create: `libs/graph_ir/build_utils.py`
- Test: `tests/libs/graph_ir/test_build_utils.py`

- [ ] **Step 1: Write failing tests for ID generation functions**

Create `tests/libs/graph_ir/test_build_utils.py`:

```python
"""Tests for Graph IR build utilities."""

from libs.graph_ir.build_utils import (
    CanonicalizationResult,
    build_singleton_local_graph,
    extract_parameters,
    factor_id,
    local_canonical_id,
    raw_node_id,
)
from libs.graph_ir.models import (
    FactorNode,
    RawGraph,
    RawKnowledgeNode,
    SourceRef,
)


def test_raw_node_id_is_deterministic():
    id1 = raw_node_id(
        package="pkg",
        version="1.0.0",
        module_name="mod",
        knowledge_name="claim_a",
        knowledge_type="claim",
        kind=None,
        content="hello",
        parameters=[],
    )
    id2 = raw_node_id(
        package="pkg",
        version="1.0.0",
        module_name="mod",
        knowledge_name="claim_a",
        knowledge_type="claim",
        kind=None,
        content="hello",
        parameters=[],
    )
    assert id1 == id2
    assert id1.startswith("raw_")
    assert len(id1) == 4 + 16  # "raw_" + 16 hex chars


def test_raw_node_id_differs_on_content():
    id1 = raw_node_id("p", "1", "m", "k", "claim", None, "aaa", [])
    id2 = raw_node_id("p", "1", "m", "k", "claim", None, "bbb", [])
    assert id1 != id2


def test_local_canonical_id_is_deterministic():
    lcn1 = local_canonical_id("raw_abc123")
    lcn2 = local_canonical_id("raw_abc123")
    assert lcn1 == lcn2
    assert lcn1.startswith("lcn_")


def test_factor_id_is_deterministic():
    fid1 = factor_id("reasoning", "mod", "chain_a")
    fid2 = factor_id("reasoning", "mod", "chain_a")
    assert fid1 == fid2
    assert fid1.startswith("f_")


def test_factor_id_with_suffix():
    fid1 = factor_id("equiv", "mod", "name", suffix="1")
    fid2 = factor_id("equiv", "mod", "name", suffix="2")
    assert fid1 != fid2


def test_extract_parameters_empty():
    assert extract_parameters("no placeholders here") == []


def test_extract_parameters_finds_placeholders():
    params = extract_parameters("For all {X}, if {Y} then {X}")
    names = [p.name for p in params]
    assert names == ["X", "Y"]  # sorted, deduplicated


def test_build_singleton_local_graph():
    raw = RawGraph(
        package="test_pkg",
        version="1.0.0",
        knowledge_nodes=[
            RawKnowledgeNode(
                raw_node_id="raw_a",
                knowledge_type="claim",
                content="A is true",
                source_refs=[SourceRef(package="test_pkg", version="1.0.0", module="m", knowledge_name="a")],
            ),
            RawKnowledgeNode(
                raw_node_id="raw_b",
                knowledge_type="observation",
                content="B observed",
                source_refs=[SourceRef(package="test_pkg", version="1.0.0", module="m", knowledge_name="b")],
            ),
        ],
        factor_nodes=[
            FactorNode(
                factor_id="f_test",
                type="reasoning",
                premises=["raw_b"],
                conclusion="raw_a",
                source_ref=SourceRef(package="test_pkg", version="1.0.0", module="m", knowledge_name="chain_a"),
            ),
        ],
    )
    result = build_singleton_local_graph(raw)
    assert isinstance(result, CanonicalizationResult)
    assert len(result.local_graph.knowledge_nodes) == 2
    assert len(result.local_graph.factor_nodes) == 1
    assert result.local_graph.package == "test_pkg"

    # Factor premises/conclusion should be remapped to local IDs
    factor = result.local_graph.factor_nodes[0]
    assert factor.conclusion.startswith("lcn_")
    assert all(p.startswith("lcn_") for p in factor.premises)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/libs/graph_ir/test_build_utils.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'libs.graph_ir.build_utils'`

- [ ] **Step 3: Create build_utils.py by extracting from build.py**

Create `libs/graph_ir/build_utils.py`. Extract these functions from `libs/graph_ir/build.py` WITHOUT modifying `build.py` (it stays intact for Phase 2):

```python
"""Source-agnostic Graph IR build utilities.

Extracted from build.py. These functions operate on Graph IR models
and do not depend on any source language (YAML or Typst).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

from .models import (
    CanonicalizationLogEntry,
    FactorNode,
    LocalCanonicalGraph,
    LocalCanonicalNode,
    Parameter,
    RawGraph,
    RawKnowledgeNode,
)

_PLACEHOLDER_RE = re.compile(r"{([A-Za-z_][A-Za-z0-9_]*)}")


@dataclass
class CanonicalizationResult:
    local_graph: LocalCanonicalGraph
    log: list[CanonicalizationLogEntry]


def raw_node_id(
    package: str,
    version: str,
    module_name: str,
    knowledge_name: str,
    knowledge_type: str,
    kind: str | None,
    content: str,
    parameters: list[Parameter],
) -> str:
    """Generate a deterministic raw node ID from its identity fields."""
    payload = {
        "package": package,
        "version": version,
        "module_name": module_name,
        "knowledge_name": knowledge_name,
        "knowledge_type": knowledge_type,
        "kind": kind,
        "content": content,
        "parameters": [p.model_dump(mode="json") for p in parameters],
    }
    digest = sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"raw_{digest[:16]}"


def local_canonical_id(raw_id: str) -> str:
    """Generate a deterministic local canonical ID from a raw node ID."""
    digest = sha256(raw_id.encode("utf-8")).hexdigest()
    return f"lcn_{digest[:16]}"


def factor_id(kind: str, module_name: str, name: str, suffix: str | None = None) -> str:
    """Generate a deterministic factor ID."""
    raw = f"{kind}:{module_name}:{name}"
    if suffix is not None:
        raw = f"{raw}:{suffix}"
    digest = sha256(raw.encode("utf-8")).hexdigest()
    return f"f_{digest[:16]}"


def extract_parameters(content: str) -> list[Parameter]:
    """Extract {X}-style parameter placeholders from content."""
    names = sorted({match.group(1) for match in _PLACEHOLDER_RE.finditer(content)})
    return [Parameter(name=name, constraint="unknown") for name in names]


def build_singleton_local_graph(raw_graph: RawGraph) -> CanonicalizationResult:
    """Build a singleton local canonical graph from the raw graph.

    Each raw node maps to exactly one local canonical node (no merging).
    """
    raw_to_local: dict[str, str] = {}
    local_nodes: list[LocalCanonicalNode] = []
    log: list[CanonicalizationLogEntry] = []

    for raw_node in raw_graph.knowledge_nodes:
        local_id = local_canonical_id(raw_node.raw_node_id)
        raw_to_local[raw_node.raw_node_id] = local_id
        local_nodes.append(
            LocalCanonicalNode(
                local_canonical_id=local_id,
                package=raw_graph.package,
                knowledge_type=raw_node.knowledge_type,
                kind=raw_node.kind,
                representative_content=raw_node.content,
                parameters=raw_node.parameters,
                member_raw_node_ids=[raw_node.raw_node_id],
                source_refs=raw_node.source_refs,
                metadata=raw_node.metadata,
            )
        )
        log.append(
            CanonicalizationLogEntry(
                local_canonical_id=local_id,
                members=[raw_node.raw_node_id],
                reason="singleton: no local semantic merge applied",
            )
        )

    local_factors = [
        FactorNode(
            factor_id=f.factor_id,
            type=f.type,
            premises=[raw_to_local[node_id] for node_id in f.premises],
            contexts=[raw_to_local[node_id] for node_id in f.contexts],
            conclusion=raw_to_local[f.conclusion],
            source_ref=f.source_ref,
            metadata=f.metadata,
        )
        for f in raw_graph.factor_nodes
    ]

    return CanonicalizationResult(
        local_graph=LocalCanonicalGraph(
            package=raw_graph.package,
            version=raw_graph.version,
            knowledge_nodes=local_nodes,
            factor_nodes=local_factors,
        ),
        log=log,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/libs/graph_ir/test_build_utils.py -v`

Expected: All 8 tests PASS.

- [ ] **Step 5: Run ruff**

Run: `ruff check libs/graph_ir/build_utils.py tests/libs/graph_ir/test_build_utils.py && ruff format --check libs/graph_ir/build_utils.py tests/libs/graph_ir/test_build_utils.py`

- [ ] **Step 6: Commit**

```bash
git add libs/graph_ir/build_utils.py tests/libs/graph_ir/test_build_utils.py
git commit -m "refactor(graph-ir): extract build_utils.py from build.py"
```

---

## Chunk 2: Typst export-graph() — add package/version metadata

### Task 2: Add package name/version state to module.typ

**Files:**
- Modify: `.worktrees/typst-gaia-lang-poc/libs/typst/gaia-lang/module.typ`

- [ ] **Step 1: Add `_gaia_package_name` and `_gaia_package_version` state variables**

At the top of `module.typ`, after the existing state declarations (line 8), add:

```typst
#let _gaia_package_name = state("gaia-package-name", none)
#let _gaia_package_version = state("gaia-package-version", none)
```

- [ ] **Step 2: Update `#package()` to store name and version in state**

In the `#package()` function (currently starts at line 45), add state updates at the beginning of the function body, before the existing `_gaia_exports.update(...)`:

```typst
  _gaia_package_name.update(_ => name)
  if version != none {
    _gaia_package_version.update(_ => str(version))
  }
```

- [ ] **Step 3: Update `export-graph()` to include package/version in metadata**

In `export-graph()` (starts at line 95), add package/version reads at the top:

```typst
#let export-graph() = context {
  let pkg_name = _gaia_package_name.final()
  let pkg_version = _gaia_package_version.final()
```

Then update the metadata dict to include them:

```typst
  [#metadata((
    package: pkg_name,
    version: pkg_version,
    nodes: _gaia_nodes.final(),
    factors: _gaia_factors.final() + proof_factors,
    refs: _gaia_refs.final(),
    modules: _gaia_modules.final(),
    module-titles: _gaia_module_titles.final(),
    exports: _gaia_exports.final(),
    constraints: raw_constraints,
  )) <gaia-graph>]
```

- [ ] **Step 4: Verify Typst compilation of v3 fixture**

Run from worktree:
```bash
cd .worktrees/typst-gaia-lang-poc
python -c "
from libs.lang.typst_loader import load_typst_package
from pathlib import Path
g = load_typst_package(Path('tests/fixtures/gaia_language_packages/galileo_falling_bodies_v3'))
print('package:', g.get('package'))
print('version:', g.get('version'))
print('nodes:', len(g['nodes']))
"
```

Expected:
```
package: galileo_falling_bodies
version: 3.0.0
nodes: 10
```

(If `package` shows `None`, the state update in `#package()` is not working. Check the fixture's `lib.typ` calls `#package("galileo_falling_bodies", version: "3.0.0", ...)`.)

- [ ] **Step 5: Commit**

```bash
git add libs/typst/gaia-lang/module.typ
git commit -m "feat(typst): export package name and version in graph metadata"
```

### Task 3: Add defensive defaults to typst_loader + test

**Files:**
- Modify: `libs/lang/typst_loader.py`
- Modify: `tests/libs/lang/test_typst_loader.py`

- [ ] **Step 1: Add setdefault for package/version in typst_loader.py**

In `libs/lang/typst_loader.py`, after the existing `data.setdefault("constraints", [])` line, add:

```python
    data.setdefault("package", None)
    data.setdefault("version", None)
```

This ensures backward compatibility with older Typst packages that don't emit `package`/`version`.

- [ ] **Step 2: Add test**

Append to `tests/libs/lang/test_typst_loader.py`:

```python
def test_v3_package_metadata():
    """v3 export-graph() should include package name and version."""
    graph = load_typst_package(GALILEO_V3)
    assert graph["package"] == "galileo_falling_bodies"
    assert graph["version"] == "3.0.0"
```

- [ ] **Step 3: Run test**

Run: `pytest tests/libs/lang/test_typst_loader.py::test_v3_package_metadata -v`

Expected: PASS (the Typst change from Task 2 already emits these fields, and `load_typst_package()` passes them through with the setdefault fallback).

- [ ] **Step 4: Commit**

```bash
git add libs/lang/typst_loader.py tests/libs/lang/test_typst_loader.py
git commit -m "feat(loader): add package/version defaults, verify in v3 test"
```

---

## Chunk 3: typst_compiler.py — Core Compilation

### Task 4: Create typst_compiler.py with compile_typst_to_raw_graph()

**Files:**
- Create: `libs/graph_ir/typst_compiler.py`
- Create: `tests/libs/graph_ir/test_typst_compiler.py`

- [ ] **Step 1: Write failing tests**

Create `tests/libs/graph_ir/test_typst_compiler.py`:

```python
"""Tests for Typst → RawGraph compiler."""

from libs.graph_ir.typst_compiler import compile_typst_to_raw_graph
from libs.graph_ir.models import RawGraph


def _make_graph_data(
    nodes=None, factors=None, constraints=None,
    package="test_pkg", version="1.0.0",
):
    """Build a minimal graph_data dict for testing."""
    return {
        "package": package,
        "version": version,
        "nodes": nodes or [],
        "factors": factors or [],
        "constraints": constraints or [],
    }


# ── Node compilation ──


def test_empty_graph():
    data = _make_graph_data()
    raw = compile_typst_to_raw_graph(data)
    assert isinstance(raw, RawGraph)
    assert raw.package == "test_pkg"
    assert raw.version == "1.0.0"
    assert raw.knowledge_nodes == []
    assert raw.factor_nodes == []


def test_single_observation_node():
    data = _make_graph_data(nodes=[
        {"name": "obs_a", "type": "observation", "content": "A observed", "module": "mod1"},
    ])
    raw = compile_typst_to_raw_graph(data)
    assert len(raw.knowledge_nodes) == 1
    node = raw.knowledge_nodes[0]
    assert node.raw_node_id.startswith("raw_")
    assert node.knowledge_type == "observation"
    assert node.content == "A observed"
    assert node.kind is None
    assert node.parameters == []
    assert len(node.source_refs) == 1
    sr = node.source_refs[0]
    assert sr.package == "test_pkg"
    assert sr.version == "1.0.0"
    assert sr.module == "mod1"
    assert sr.knowledge_name == "obs_a"


def test_claim_node():
    data = _make_graph_data(nodes=[
        {"name": "claim_x", "type": "claim", "content": "X is true", "module": "mod1"},
    ])
    raw = compile_typst_to_raw_graph(data)
    node = raw.knowledge_nodes[0]
    assert node.knowledge_type == "claim"


def test_constraint_node_has_between_metadata():
    data = _make_graph_data(
        nodes=[
            {"name": "a", "type": "claim", "content": "A", "module": "m"},
            {"name": "b", "type": "claim", "content": "B", "module": "m"},
            {"name": "c_rel", "type": "contradiction", "content": "C", "module": "m"},
        ],
        constraints=[
            {"name": "c_rel", "type": "contradiction", "between": ["a", "b"]},
        ],
    )
    raw = compile_typst_to_raw_graph(data)
    rel_node = [n for n in raw.knowledge_nodes if n.knowledge_type == "contradiction"][0]
    assert rel_node.metadata == {"between": ["a", "b"]}


def test_node_ids_are_deterministic():
    data = _make_graph_data(nodes=[
        {"name": "obs_a", "type": "observation", "content": "A", "module": "m"},
    ])
    raw1 = compile_typst_to_raw_graph(data)
    raw2 = compile_typst_to_raw_graph(data)
    assert raw1.knowledge_nodes[0].raw_node_id == raw2.knowledge_nodes[0].raw_node_id


# ── Factor compilation ──


def test_reasoning_factor():
    data = _make_graph_data(
        nodes=[
            {"name": "obs_a", "type": "observation", "content": "A", "module": "m"},
            {"name": "claim_b", "type": "claim", "content": "B", "module": "m"},
        ],
        factors=[
            {"type": "reasoning", "premise": ["obs_a"], "conclusion": "claim_b"},
        ],
    )
    raw = compile_typst_to_raw_graph(data)
    assert len(raw.factor_nodes) == 1
    factor = raw.factor_nodes[0]
    assert factor.type == "reasoning"
    assert factor.factor_id.startswith("f_")
    assert factor.contexts == []
    assert factor.metadata == {"edge_type": "deduction"}

    # premises and conclusion should be raw_node_ids, not names
    node_ids = {n.raw_node_id for n in raw.knowledge_nodes}
    assert factor.conclusion in node_ids
    assert all(p in node_ids for p in factor.premises)

    # source_ref should point to conclusion
    assert factor.source_ref is not None
    assert factor.source_ref.knowledge_name == "claim_b"


def test_reasoning_factor_multiple_premises():
    data = _make_graph_data(
        nodes=[
            {"name": "a", "type": "observation", "content": "A", "module": "m"},
            {"name": "b", "type": "observation", "content": "B", "module": "m"},
            {"name": "c", "type": "claim", "content": "C", "module": "m"},
        ],
        factors=[
            {"type": "reasoning", "premise": ["a", "b"], "conclusion": "c"},
        ],
    )
    raw = compile_typst_to_raw_graph(data)
    factor = raw.factor_nodes[0]
    assert len(factor.premises) == 2


# ── Constraint compilation ──


def test_contradiction_constraint():
    data = _make_graph_data(
        nodes=[
            {"name": "a", "type": "claim", "content": "A", "module": "m"},
            {"name": "b", "type": "claim", "content": "B", "module": "m"},
            {"name": "c", "type": "contradiction", "content": "C", "module": "m"},
        ],
        constraints=[
            {"name": "c", "type": "contradiction", "between": ["a", "b"]},
        ],
    )
    raw = compile_typst_to_raw_graph(data)
    constraint_factors = [f for f in raw.factor_nodes if f.type == "mutex_constraint"]
    assert len(constraint_factors) == 1
    cf = constraint_factors[0]
    assert len(cf.premises) == 2
    assert cf.metadata == {"edge_type": "relation_contradiction"}

    # conclusion should be the constraint node's ID
    node_map = {n.knowledge_type: n.raw_node_id for n in raw.knowledge_nodes}
    assert cf.conclusion == node_map["contradiction"]


def test_equivalence_constraint():
    data = _make_graph_data(
        nodes=[
            {"name": "a", "type": "claim", "content": "A", "module": "m"},
            {"name": "b", "type": "claim", "content": "B", "module": "m"},
            {"name": "eq", "type": "equivalence", "content": "E", "module": "m"},
        ],
        constraints=[
            {"name": "eq", "type": "equivalence", "between": ["a", "b"]},
        ],
    )
    raw = compile_typst_to_raw_graph(data)
    equiv_factors = [f for f in raw.factor_nodes if f.type == "equiv_constraint"]
    assert len(equiv_factors) == 1
    assert equiv_factors[0].metadata == {"edge_type": "relation_equivalence"}


# ── Cross-module references ──


def test_duplicate_node_name_raises():
    """Duplicate node names within a package should raise ValueError."""
    import pytest

    data = _make_graph_data(nodes=[
        {"name": "obs", "type": "observation", "content": "Same", "module": "mod_a"},
        {"name": "obs", "type": "observation", "content": "Same", "module": "mod_b"},
    ])
    with pytest.raises(ValueError, match="Duplicate node name"):
        compile_typst_to_raw_graph(data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/libs/graph_ir/test_typst_compiler.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'libs.graph_ir.typst_compiler'`

- [ ] **Step 3: Implement typst_compiler.py**

Create `libs/graph_ir/typst_compiler.py`:

```python
"""Compile Typst loader output to Graph IR RawGraph.

Takes the dict produced by typst_loader.load_typst_package() and
produces a RawGraph with deterministic IDs and full source refs.
"""

from __future__ import annotations

from .build_utils import factor_id, raw_node_id
from .models import (
    FactorNode,
    RawGraph,
    RawKnowledgeNode,
    SourceRef,
)

_CONSTRAINT_TYPE_TO_FACTOR_TYPE = {
    "contradiction": "mutex_constraint",
    "equivalence": "equiv_constraint",
}


def compile_typst_to_raw_graph(graph_data: dict) -> RawGraph:
    """Compile typst_loader output dict to RawGraph.

    Args:
        graph_data: Dict with keys: package, version, nodes, factors, constraints.
                    Produced by typst_loader.load_typst_package().

    Returns:
        A RawGraph with deterministic node/factor IDs and source refs.
    """
    package = graph_data.get("package", "unknown")
    version = graph_data.get("version", "0.0.0")

    # Build constraint lookup for metadata injection
    constraint_map: dict[str, dict] = {}
    for constraint in graph_data.get("constraints", []):
        constraint_map[constraint["name"]] = constraint

    # 1. Compile nodes
    knowledge_nodes: list[RawKnowledgeNode] = []
    name_to_raw_id: dict[str, str] = {}

    for node in graph_data.get("nodes", []):
        name = node["name"]
        knowledge_type = node["type"]
        content = node.get("content", "")
        module = node.get("module", "unknown")

        node_id = raw_node_id(
            package=package,
            version=version,
            module_name=module,
            knowledge_name=name,
            knowledge_type=knowledge_type,
            kind=None,
            content=content,
            parameters=[],
        )

        metadata = None
        if name in constraint_map:
            metadata = {"between": list(constraint_map[name]["between"])}

        knowledge_nodes.append(
            RawKnowledgeNode(
                raw_node_id=node_id,
                knowledge_type=knowledge_type,
                kind=None,
                content=content,
                parameters=[],
                source_refs=[
                    SourceRef(
                        package=package,
                        version=version,
                        module=module,
                        knowledge_name=name,
                    )
                ],
                metadata=metadata,
            )
        )
        if name in name_to_raw_id:
            raise ValueError(
                f"Duplicate node name '{name}' in package '{package}'. "
                "Node names must be unique within a package."
            )
        name_to_raw_id[name] = node_id

    # 2. Compile reasoning factors
    factor_nodes: list[FactorNode] = []

    for factor in graph_data.get("factors", []):
        if factor.get("type") != "reasoning":
            continue

        conclusion_name = factor["conclusion"]
        premise_names = factor.get("premise", [])

        if conclusion_name not in name_to_raw_id:
            continue
        premise_ids = [
            name_to_raw_id[p] for p in premise_names if p in name_to_raw_id
        ]
        if not premise_ids:
            continue

        # Determine module from the conclusion node
        conclusion_module = _find_node_module(graph_data["nodes"], conclusion_name)

        factor_nodes.append(
            FactorNode(
                factor_id=factor_id("reasoning", conclusion_module, conclusion_name),
                type="reasoning",
                premises=premise_ids,
                contexts=[],
                conclusion=name_to_raw_id[conclusion_name],
                source_ref=SourceRef(
                    package=package,
                    version=version,
                    module=conclusion_module,
                    knowledge_name=conclusion_name,
                ),
                metadata={"edge_type": "deduction"},
            )
        )

    # 3. Compile constraint factors
    for constraint in graph_data.get("constraints", []):
        constraint_name = constraint["name"]
        constraint_type = constraint["type"]
        between = constraint.get("between", [])

        if constraint_name not in name_to_raw_id:
            continue
        related_ids = [
            name_to_raw_id[b] for b in between if b in name_to_raw_id
        ]
        if len(related_ids) < 2:
            continue

        ft = _CONSTRAINT_TYPE_TO_FACTOR_TYPE.get(constraint_type)
        if ft is None:
            continue

        constraint_module = _find_node_module(graph_data["nodes"], constraint_name)

        factor_nodes.append(
            FactorNode(
                factor_id=factor_id(ft, constraint_module, constraint_name),
                type=ft,
                premises=related_ids,
                contexts=[],
                conclusion=name_to_raw_id[constraint_name],
                source_ref=SourceRef(
                    package=package,
                    version=version,
                    module=constraint_module,
                    knowledge_name=constraint_name,
                ),
                metadata={"edge_type": f"relation_{constraint_type}"},
            )
        )

    return RawGraph(
        package=package,
        version=version,
        knowledge_nodes=knowledge_nodes,
        factor_nodes=factor_nodes,
    )


def _find_node_module(nodes: list[dict], name: str) -> str:
    """Find the module of a node by name."""
    for node in nodes:
        if node["name"] == name:
            return node.get("module", "unknown")
    return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/libs/graph_ir/test_typst_compiler.py -v`

Expected: All 12 tests PASS.

- [ ] **Step 5: Run ruff**

Run: `ruff check libs/graph_ir/typst_compiler.py tests/libs/graph_ir/test_typst_compiler.py && ruff format --check libs/graph_ir/typst_compiler.py tests/libs/graph_ir/test_typst_compiler.py`

- [ ] **Step 6: Commit**

```bash
git add libs/graph_ir/typst_compiler.py tests/libs/graph_ir/test_typst_compiler.py
git commit -m "feat(graph-ir): add typst_compiler — Typst dict to RawGraph"
```

### Task 5: Integration test — compile real Typst fixture to RawGraph

**Files:**
- Modify: `tests/libs/graph_ir/test_typst_compiler.py`

- [ ] **Step 1: Add integration test using galileo v3 fixture**

Append to `tests/libs/graph_ir/test_typst_compiler.py`:

```python
from pathlib import Path

from libs.lang.typst_loader import load_typst_package
from libs.graph_ir.build_utils import build_singleton_local_graph

GALILEO_V3 = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "gaia_language_packages"
    / "galileo_falling_bodies_v3"
)


def test_galileo_v3_full_compile():
    """End-to-end: load Typst fixture → compile → verify RawGraph structure."""
    graph_data = load_typst_package(GALILEO_V3)
    raw = compile_typst_to_raw_graph(graph_data)

    assert raw.package == "galileo_falling_bodies"
    assert raw.version == "3.0.0"

    # Should have nodes for observations, settings, claims, and constraints
    types = {n.knowledge_type for n in raw.knowledge_nodes}
    assert "observation" in types
    assert "claim" in types

    # Should have reasoning factors
    reasoning = [f for f in raw.factor_nodes if f.type == "reasoning"]
    assert len(reasoning) >= 3  # vacuum_prediction, composite_is_slower, etc.

    # Should have constraint factor for tied_balls_contradiction
    constraints = [f for f in raw.factor_nodes if f.type == "mutex_constraint"]
    assert len(constraints) >= 1

    # All factor premises/conclusions should reference valid node IDs
    node_ids = {n.raw_node_id for n in raw.knowledge_nodes}
    for factor in raw.factor_nodes:
        assert factor.conclusion in node_ids, f"conclusion {factor.conclusion} not in nodes"
        for p in factor.premises:
            assert p in node_ids, f"premise {p} not in nodes"


def test_galileo_v3_through_canonicalization():
    """End-to-end: Typst → RawGraph → LocalCanonicalGraph."""
    graph_data = load_typst_package(GALILEO_V3)
    raw = compile_typst_to_raw_graph(graph_data)
    result = build_singleton_local_graph(raw)

    assert len(result.local_graph.knowledge_nodes) == len(raw.knowledge_nodes)
    assert len(result.local_graph.factor_nodes) == len(raw.factor_nodes)

    # All local IDs should start with "lcn_"
    for node in result.local_graph.knowledge_nodes:
        assert node.local_canonical_id.startswith("lcn_")
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/libs/graph_ir/test_typst_compiler.py::test_galileo_v3_full_compile tests/libs/graph_ir/test_typst_compiler.py::test_galileo_v3_through_canonicalization -v`

Expected: Both PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/libs/graph_ir/test_typst_compiler.py
git commit -m "test(graph-ir): add integration tests for Typst → RawGraph compilation"
```

---

## Chunk 4: Pipeline rewrite + housekeeping

### Task 6: Rewrite pipeline_build()

**Files:**
- Modify: `libs/pipeline.py`

- [ ] **Step 1: Read current pipeline.py**

Read: `libs/pipeline.py`

Understand the current `BuildResult` dataclass and `pipeline_build()` function.

- [ ] **Step 2: Add new `pipeline_build_typst()` function**

Do NOT modify the existing `pipeline_build()` — it's still used by the YAML path (Phase 2 cleanup). Instead, add a new function alongside it:

```python
from libs.graph_ir.typst_compiler import compile_typst_to_raw_graph
from libs.graph_ir.build_utils import build_singleton_local_graph
from libs.lang.typst_loader import load_typst_package


@dataclass
class TypstBuildResult:
    graph_data: dict
    raw_graph: RawGraph
    local_graph: LocalCanonicalGraph
    canonicalization_log: list
    source_files: dict[str, str] = field(default_factory=dict)


async def pipeline_build_typst(pkg_path: Path) -> TypstBuildResult:
    """Load, compile, and canonicalize a Typst package — all in memory.

    Args:
        pkg_path: Path to the Typst package directory (contains typst.toml + lib.typ).
    """
    graph_data = load_typst_package(pkg_path)
    raw_graph = compile_typst_to_raw_graph(graph_data)
    canonicalization = build_singleton_local_graph(raw_graph)
    source_files = {p.name: p.read_text() for p in pkg_path.glob("*.typ") if p.is_file()}

    return TypstBuildResult(
        graph_data=graph_data,
        raw_graph=raw_graph,
        local_graph=canonicalization.local_graph,
        canonicalization_log=canonicalization.log,
        source_files=source_files,
    )
```

Add the necessary imports at the top of the file. Keep all existing functions unchanged.

- [ ] **Step 3: Write test for pipeline_build_typst**

Create `tests/test_pipeline_typst.py`:

```python
"""Tests for Typst pipeline_build_typst()."""

from pathlib import Path

import pytest

from libs.pipeline import TypstBuildResult, pipeline_build_typst

GALILEO_V3 = (
    Path(__file__).parent
    / "fixtures"
    / "gaia_language_packages"
    / "galileo_falling_bodies_v3"
)


@pytest.mark.asyncio
async def test_pipeline_build_typst_returns_result():
    result = await pipeline_build_typst(GALILEO_V3)
    assert isinstance(result, TypstBuildResult)


@pytest.mark.asyncio
async def test_pipeline_build_typst_has_graph_data():
    result = await pipeline_build_typst(GALILEO_V3)
    assert "nodes" in result.graph_data
    assert "factors" in result.graph_data
    assert result.graph_data["package"] == "galileo_falling_bodies"


@pytest.mark.asyncio
async def test_pipeline_build_typst_has_raw_graph():
    result = await pipeline_build_typst(GALILEO_V3)
    assert result.raw_graph.package == "galileo_falling_bodies"
    assert len(result.raw_graph.knowledge_nodes) > 0
    assert len(result.raw_graph.factor_nodes) > 0


@pytest.mark.asyncio
async def test_pipeline_build_typst_has_local_graph():
    result = await pipeline_build_typst(GALILEO_V3)
    assert len(result.local_graph.knowledge_nodes) > 0
    assert result.local_graph.package == "galileo_falling_bodies"


@pytest.mark.asyncio
async def test_pipeline_build_typst_collects_source_files():
    result = await pipeline_build_typst(GALILEO_V3)
    assert "lib.typ" in result.source_files
    assert "galileo.typ" in result.source_files
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pipeline_typst.py -v`

Expected: All 5 tests PASS.

- [ ] **Step 5: Run ruff**

Run: `ruff check libs/pipeline.py tests/test_pipeline_typst.py && ruff format --check libs/pipeline.py tests/test_pipeline_typst.py`

- [ ] **Step 6: Commit**

```bash
git add libs/pipeline.py tests/test_pipeline_typst.py
git commit -m "feat(pipeline): add pipeline_build_typst() for Typst packages"
```

### Task 7: Move plausible_core.py to future/

**Files:**
- Move: `libs/lang/plausible_core.py` → `future/lang/plausible_core.py`
- Move: `tests/libs/lang/test_plausible_core.py` → `future/lang/test_plausible_core.py`

- [ ] **Step 1: Create future directory and move files**

```bash
mkdir -p future/lang
git mv libs/lang/plausible_core.py future/lang/plausible_core.py
git mv tests/libs/lang/test_plausible_core.py future/lang/test_plausible_core.py
```

Note: These files are currently untracked (not yet committed). If `git mv` fails because they're untracked, use regular `mv` instead:

```bash
mkdir -p future/lang
mv libs/lang/plausible_core.py future/lang/plausible_core.py
mv tests/libs/lang/test_plausible_core.py future/lang/test_plausible_core.py
```

- [ ] **Step 2: Commit**

```bash
git add future/lang/
git commit -m "chore: move plausible_core.py to future/ (not needed for v3)"
```

### Task 8: Full test suite verification

- [ ] **Step 1: Run all new tests**

```bash
pytest tests/libs/graph_ir/test_build_utils.py tests/libs/graph_ir/test_typst_compiler.py tests/test_pipeline_typst.py -v
```

Expected: All tests PASS.

- [ ] **Step 2: Run existing tests to verify no regressions**

```bash
pytest tests/ -v --ignore=tests/integration
```

Expected: All tests PASS. Existing YAML-path tests should be unaffected since we did not modify `build.py` or any YAML-era code.

- [ ] **Step 3: Run ruff on all changed files**

```bash
ruff check libs/graph_ir/build_utils.py libs/graph_ir/typst_compiler.py libs/pipeline.py
ruff format --check libs/graph_ir/build_utils.py libs/graph_ir/typst_compiler.py libs/pipeline.py
```

Expected: No errors.

- [ ] **Step 4: Fix any issues and commit**

If any test failures or lint issues, fix them:

```bash
git add -A
git commit -m "fix: address test and lint issues from graph IR compiler"
```
