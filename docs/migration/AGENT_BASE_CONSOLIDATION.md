# Agent Base Consolidation — Migration Checklist

**Status:** Phase 1.3 — planning document.
**Prereqs:** Phase 1.1 audit ([`../architecture/agent_inheritance_audit.md`](../architecture/agent_inheritance_audit.md)) and Phase 1.2 canonical declaration (`greenlang.agent_runtime.CANONICAL_V3 = True`, `EnhancedBaseAgent` emits runtime `DeprecationWarning`).
**Companion doc:** `greenlang/agents/MIGRATION_TO_AGENTSPECV2.md` — per-pattern migration steps (referenced but not duplicated here).

---

## 0. Scope and definition of done

"Consolidation" means three things, in order:

1. **Import surface cleanup.** Every caller imports bases from `greenlang.agent_runtime` (the CANONICAL_V3 surface). No direct imports from `greenlang.agents.base`, `base_agents`, `intelligent_base`, `async_agent_base`, `agentspec_v2_base`, or `enhanced_base_agent`.
2. **Category discipline.** Every agent declares a category (DETERMINISTIC / REASONING / INSIGHT) by inheriting the matching base or composing the matching mixin onto `AgentSpecV2Base`. No bare `class X(BaseAgent)` without a category, except documented utility scaffolding.
3. **Dead code removal.** `EnhancedBaseAgent` removed at v1.0. Ad-hoc mixins not exported by `agent_runtime` (`IntelligenceMixin`, `OperationalMonitoringMixin`) either promoted to canonical exports or retired.

Definition of done ships in **three milestones**: M1 (reference exemplars, this sprint), M2 (high-traffic FY27 agents), M3 (remainder + v1.0 removal).

---

## 1. Canonical import pattern

**Old (direct):**
```python
from greenlang.agents.base import BaseAgent, AgentResult
from greenlang.agents.base_agents import DeterministicAgent
```

**New (canonical):**
```python
from greenlang.agent_runtime import BaseAgent, AgentResult, DeterministicAgent
```

Everything a caller needs — base classes, result types, mixins, exec contexts — is re-exported by `greenlang.agent_runtime`. The direct imports still work for backward compatibility through v0.9.x; they will be marked `DeprecationWarning` in v0.8.x (M2 milestone) and removed in v1.0 (M3 milestone).

---

## 2. Category decision tree

Before migrating an agent, decide its category. The audit ([§5](../architecture/agent_inheritance_audit.md#5-inheritance-counts-files-that-subclass-each-base)) shows 146 agents on plain `BaseAgent` that lack a declared category — that is the backlog.

| You need… | Use |
|---|---|
| Pure calculation, zero LLM, regulator-auditable | **`DeterministicAgent`** (or `AgentSpecV2Base[InT, OutT]` + `DeterministicMixin`) |
| Strategic/creative LLM reasoning, non-regulatory | **`ReasoningAgent`** (or mixin variant) |
| Deterministic numbers + AI explanation/recommendation | **`InsightAgent`** (or mixin variant) |
| Typed schema-validated agent for a pack | `AgentSpecV2Base[InT, OutT]` + one of the above mixins |
| High-throughput async I/O (Connect extracts, batch ingests) | `AsyncAgentBase[InT, OutT]` |
| LLM-enhanced agent that wants ChatSession + RAG + budgets | `IntelligentAgentBase` |

For full worked examples per path, see `greenlang/agents/MIGRATION_TO_AGENTSPECV2.md`.

---

## 3. Migration recipes

### 3.1 Simple `BaseAgent` → typed + categorized

```python
# Before
from greenlang.agents.base import BaseAgent, AgentResult

class MyAgent(BaseAgent):
    def execute(self, input_data):
        return AgentResult(success=True, data={"x": 1})
```

```python
# After
from pydantic import BaseModel
from greenlang.agent_runtime import (
    AgentSpecV2Base, DeterministicMixin, AgentResult,
)

class MyInput(BaseModel):
    value: float

class MyOutput(BaseModel):
    x: float

class MyAgent(DeterministicMixin, AgentSpecV2Base[MyInput, MyOutput]):
    def execute(self, input_data: MyInput) -> AgentResult:
        return AgentResult(success=True, data={"x": input_data.value})
```

### 3.2 `Agent[InT, OutT]` protocol → `AgentSpecV2Base[InT, OutT]`

```python
# Before
from greenlang.types import Agent
class FuelAgent(Agent[FuelInput, FuelOutput]):
    ...
```

```python
# After
from greenlang.agent_runtime import AgentSpecV2Base, DeterministicMixin
class FuelAgent(DeterministicMixin, AgentSpecV2Base[FuelInput, FuelOutput]):
    ...
```

### 3.3 Ad-hoc `IntelligenceMixin` + `BaseAgent` → `IntelligentAgentBase`

```python
# Before
class CarbonAgent(IntelligenceMixin, OperationalMonitoringMixin, BaseAgent):
    ...
```

```python
# After (for LLM-enhanced agents)
from greenlang.agent_runtime import IntelligentAgentBase

class CarbonAgent(IntelligentAgentBase):
    # LLM provider, ChatSession, and RAG are attributes on self
    ...
```

### 3.4 `EnhancedBaseAgent` → any canonical base

Pick a replacement from §2, move overrides of `_process` into `execute`, drop unused `_get_safety_functions` / `_get_api_routes` stubs. The legacy capabilities (circuit breakers, SIL validators, event bus) now live behind opt-in features on the canonical lifecycle, not as mandatory abstract methods.

Subclassing `EnhancedBaseAgent` emits a runtime `DeprecationWarning` as of Phase 1.2. CI can fail on deprecation with `-W error::DeprecationWarning:greenlang.agents.enhanced_base_agent`.

---

## 4. Milestone M1 — reference exemplars (this sprint)

Ship one tested, migrated exemplar per canonical path. These become the copy-paste templates for M2.

| # | Exemplar agent | Target base | Purpose |
|---|---|---|---|
| 1 | `greenlang/agents/carbon_agent_ai.py` → `CarbonAgentAI` | `IntelligentAgentBase` | Canonical LLM-enhanced agent |
| 2 | `greenlang/agents/gl_010_emissions_guardian.py` → `EmissionsGuardian` | `DeterministicAgent` | Canonical deterministic agent |
| 3 | `greenlang/agents/policy/cbam_compliance_agent.py` → `CBAMComplianceAgent` | `AgentSpecV2Base[In, Out]` + `DeterministicMixin` | Canonical typed + categorized agent |
| 4 | `greenlang/agents/fuel_agent.py` → `FuelAgent` | `AgentSpecV2Base[FuelInput, FuelOutput]` + `DeterministicMixin` | Migration off `Agent[InT, OutT]` protocol |
| 5 | (future) `greenlang/connect/sap_s4hana.py` extract job | `AsyncAgentBase[In, Out]` | Canonical async exemplar — lands with Phase 2.5 |

Each M1 exemplar must include:

- Updated imports to `greenlang.agent_runtime`.
- Type-annotated input/output Pydantic models.
- A unit test in `tests/agents/test_<agent>_canonical.py` that imports via the canonical surface only.
- An entry in `docs/migration/AGENT_BASE_CONSOLIDATION.md` §8 with the diff.

---

## 5. Milestone M2 — high-traffic FY27 agents

The 10 agents identified in the audit are the minimum-viable set for the first paid CBAM / Comply / Scope Engine pilots. Migrate after M1 exemplars are merged.

| # | Agent | File | Current | Target |
|---|---|---|---|---|
| 1 | CBAM compliance | `greenlang/agents/policy/cbam_compliance_agent.py` | `BaseAgent` | `AgentSpecV2Base + DeterministicMixin` |
| 2 | CSRD compliance | `greenlang/agents/policy/csrd_compliance_agent.py` | `BaseAgent` | `AgentSpecV2Base + DeterministicMixin` |
| 3 | Emissions guardian | `greenlang/agents/gl_010_emissions_guardian.py` | `BaseAgent` | `DeterministicAgent` |
| 4 | Carbon sync | `greenlang/agents/carbon_agent.py` | `IntelligenceMixin + OperationalMonitoringMixin + BaseAgent` | `IntelligentAgentBase` |
| 5 | Carbon AI | `greenlang/agents/carbon_agent_ai.py` | `BaseAgent` | `IntelligentAgentBase` |
| 6 | Fuel | `greenlang/agents/fuel_agent.py` | `Agent[FuelInput, FuelOutput]` | `AgentSpecV2Base + DeterministicMixin` |
| 7 | Grid factor | `greenlang/agents/grid_factor_agent.py` | `Agent[...]` | `AgentSpecV2Base + DeterministicMixin` |
| 8 | Intensity | `greenlang/agents/intensity_agent.py` | `IntelligenceMixin + BaseAgent` | `InsightAgent` or `AgentSpecV2Base + InsightMixin` |
| 9 | Report | `greenlang/agents/report_agent.py` | `OperationalMonitoringMixin + BaseAgent` | `ReasoningAgent` or `InsightAgent` depending on LLM use |
| 10 | Scope 3 category mapper | `greenlang/agents/scope3/…` | assorted | `AgentSpecV2Base + DeterministicMixin` |

**Per-agent checklist:**

- [ ] Replace direct imports with `from greenlang.agent_runtime import …`.
- [ ] Declare typed `Input` / `Output` Pydantic models (if not already present).
- [ ] Switch base to target from table above.
- [ ] Remove ad-hoc mixins (`IntelligenceMixin`, `OperationalMonitoringMixin`) — if the capability is still needed, raise a PR to promote that mixin into `greenlang.agent_runtime` exports.
- [ ] Update callers (any pack YAML agent reference, any CLI registration, any application `app.py`).
- [ ] Run `pytest tests/agents/test_<agent>*.py` green.
- [ ] Add a passing canonical-only smoke test.

---

## 6. Milestone M3 — remainder + v1.0 removal

After M2 merges, sweep the remaining ~190 class-bearing files grouped by sub-directory. Migrate in the order below so that regulated-compliance agents are converted first.

| Order | Directory | Approximate files | Dominant target | Notes |
|---|---|---|---|---|
| 1 | `greenlang/agents/policy/` | 9 | `AgentSpecV2Base + DeterministicMixin` | CBAM, CSRD, SB 253, TCFD, SBTi rules |
| 2 | `greenlang/agents/mrv/` | 13 | `DeterministicAgent` | Scope 1/2 primitives |
| 3 | `greenlang/agents/scope3/` | 19 | `DeterministicAgent` | Scope 3 Cat 1–15 |
| 4 | `greenlang/agents/foundation/` | 8 | `BaseAgent` (stay) | Cross-cutting utilities |
| 5 | `greenlang/agents/data/` | 20 | `AgentSpecV2Base + DeterministicMixin` or `AsyncAgentBase` | Data intake/quality |
| 6 | `greenlang/agents/intelligence/` | 17 | `IntelligentAgentBase` | LLM-heavy paths |
| 7 | `greenlang/agents/reporting/` | 8 | `ReasoningAgent` or `InsightAgent` | Narratives + templates |
| 8 | `greenlang/agents/ecosystem/` | 8 | `InsightAgent` | Benchmarks, scoring |
| 9 | `greenlang/agents/formulas/` | 7 | `DeterministicAgent` | Pure math |
| 10 | `greenlang/agents/procurement/` | 5 | `AgentSpecV2Base + InsightMixin` | FY28 SupplierOS precursor |
| 11 | `greenlang/agents/finance/` | 10 | case-by-case | FY29 FinanceOS precursor |
| 12 | `greenlang/agents/operations/` | 7 | archive candidate | FY28 PlantOS — may be archived instead of migrated per Phase 6 |
| 13 | `greenlang/agents/process_heat/` | 1 | archive candidate | FY28 PlantOS |
| 14 | `greenlang/agents/adaptation/` | 13 | leave as stub | FY30 Adaptation Planner; stub per §3.5 of FY27 analysis |
| 15 | `greenlang/agents/decarbonization/` | 1 | `ReasoningAgent` | FY29+ |
| 16 | `greenlang/agents/calculation/` | 1 | `DeterministicAgent` | — |
| 17 | `greenlang/agents/eudr/` | 1 | conditional on Phase 6.1 scope decision | See FY27 analysis §6.1 |
| 18 | `greenlang/agents/sb253/` | 1 | `AgentSpecV2Base + DeterministicMixin` | US Comply |

**After M3 completes:**

- [ ] Zero subclasses of `EnhancedBaseAgent` across the repo (CI grep-fails if any reappear).
- [ ] Zero imports of `greenlang.agents.base` / `base_agents` / `intelligent_base` / `async_agent_base` / `agentspec_v2_base` / `enhanced_base_agent` from outside `greenlang/agent_runtime/`.
- [ ] `greenlang/agents/enhanced_base_agent.py` moved to `_archive/05_duplicate_agent_versions/` and its re-export dropped from `greenlang/agent_runtime/__init__.py` (backwards-incompatible — bump to v1.0).
- [ ] Deprecation warnings removed.

---

## 7. Tooling to keep us honest

- **Static check (M2):** ship a `scripts/check_canonical_imports.py` that grep-fails any file outside `greenlang/agent_runtime/` importing the underlying base modules directly. Wire it into `.pre-commit-config.yaml`.
- **CI gate (M3):** add a pytest fixture `tests/conftest.py::_fail_on_deprecation` that upgrades `DeprecationWarning` from `greenlang.agents.enhanced_base_agent` to errors, so any new subclass of `EnhancedBaseAgent` breaks CI.
- **Inventory refresh:** re-run the audit script and update `docs/architecture/agent_inheritance_audit.md` §5 counts before each milestone PR.

---

## 8. Per-agent diff log

Track migration per agent here. Filled in as M1 / M2 land.

| Milestone | Agent | PR | Before | After | Notes |
|---|---|---|---|---|---|
| M1 | `carbon_agent_ai.py` | _pending_ | `BaseAgent` | `IntelligentAgentBase` | reference exemplar |
| M1 | `gl_010_emissions_guardian.py` | _pending_ | `BaseAgent` | `DeterministicAgent` | reference exemplar |
| M1 | `cbam_compliance_agent.py` | _pending_ | `BaseAgent` | `AgentSpecV2Base + DeterministicMixin` | reference exemplar |
| M1 | `fuel_agent.py` | _pending_ | `Agent[InT, OutT]` | `AgentSpecV2Base + DeterministicMixin` | reference exemplar |
| … | … | … | … | … | … |

---

## 9. Open questions

1. Do we promote `IntelligenceMixin` + `OperationalMonitoringMixin` into `greenlang.agent_runtime` exports, or fold their behaviour into `IntelligentAgentBase` / `AgentSpecV2Base` instance hooks? Decide before M2 kickoff.
2. Does `Agent[InT, OutT]` (the typed protocol in `greenlang/types.py`) stay as-is, or do we deprecate it in favour of the typed `AgentSpecV2Base[InT, OutT]`? Recommendation: keep the protocol for duck-typed interop, deprecate direct subclassing.
3. Final fate of `adaptation/` (13 files) and `operations/` (7 files) — migrate or archive per Phase 6 scope decision?

All three are tracked in `FY27_vs_Reality_Analysis.md` Phase 6 and can be resolved outside of the migration itself.
