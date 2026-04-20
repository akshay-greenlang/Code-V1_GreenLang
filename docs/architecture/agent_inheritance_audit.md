# Agent Inheritance Audit

**Date:** 2026-04-20
**Scope:** `greenlang/agents/**` and `greenlang/agent_runtime/`
**Purpose:** Inventory the current agent inheritance graph as input to Phase 1.2 canonical-base declaration and Phase 1.3 migration.

---

## 1. Headline finding

The "six overlapping base classes" risk called out in `FY27_vs_Reality_Analysis.md` has already been **partially resolved**. A unified namespace at `greenlang/agent_runtime/__init__.py` (v0.1.0) now re-exports every base class and mixin under a documented **v3 canonical hierarchy**. The legacy `EnhancedBaseAgent` is marked deprecated as of v0.1.0.

The remaining work is not "collapse six classes into one" — it is:

1. Migrate callers to import from `greenlang.agent_runtime` instead of the individual `greenlang.agents.*` modules.
2. Burn down the **146 direct `BaseAgent` subclasses** by routing them to a category-specific mixin (`DeterministicMixin` / `ReasoningMixin` / `InsightMixin` on top of `AgentSpecV2Base[InT, OutT]`).
3. Add a runtime `DeprecationWarning` on `EnhancedBaseAgent` so stragglers are forced onto the canonical path before the v1.0 removal.

Full plan: [`../migration/AGENT_BASE_CONSOLIDATION.md`](../migration/AGENT_BASE_CONSOLIDATION.md).

---

## 2. File-count inventory

| Scope | Count |
|---|---|
| `.py` files directly under `greenlang/agents/` | **45** |
| `.py` files recursively under `greenlang/agents/**` | **~2,966** (most are tool files / subagent helpers; class-bearing files are a small subset) |

Sub-directory breakdown:

| Directory | Files | v3 grouping |
|---|---|---|
| `adaptation/` | 13 | FY30 stub |
| `calculation/` | 1 | L3 |
| `data/` | 20 | L1 (data intake agents) |
| `decarbonization/` | 1 | FY28+ |
| `ecosystem/` | 8 | L4 |
| `eudr/` | 1 | FY27 scope-decision pending |
| `finance/` | 10 | FY29 FinanceOS precursor |
| `formulas/` | 7 | L3 |
| `foundation/` | 8 | L3 foundation agents |
| `intelligence/` | 17 | L3 LLM layer |
| `mrv/` | 13 | L3 Scope engine precursor |
| `operations/` | 7 | FY28 PlantOS precursor |
| `policy/` | 9 | L3 (CBAM, CSRD, …) |
| `process_heat/` | 1 | FY28 PlantOS |
| `procurement/` | 5 | FY28 SupplierOS |
| `reporting/` | 8 | L4 |
| `sb253/` | 1 | L4 (US Comply) |
| `scope3/` | 19 | L3 Scope 3 engine precursor |

---

## 3. Canonical v3 hierarchy (already documented in `agent_runtime/__init__.py`)

```
BaseAgent                                   # lifecycle, metrics, provenance
├── DeterministicAgent                      # zero-hallucination (CRITICAL PATH)
├── ReasoningAgent                          # LLM reasoning (RECOMMENDATION PATH)
└── InsightAgent                            # hybrid deterministic + AI (INSIGHT PATH)

AgentSpecV2Base[InT, OutT]                  # typed, schema-validated
+ DeterministicMixin | ReasoningMixin | InsightMixin

AsyncAgentBase[InT, OutT]                   # async/await lifecycle

IntelligentAgentBase                        # BaseAgent + LLM providers + ChatSession + RAG
```

**Source of truth:** `greenlang.agent_runtime` (re-exports everything from the six underlying modules).

**Deprecated:** `EnhancedBaseAgent` (removal scheduled for v1.0 per docstring).

---

## 4. Base class inventory

| File | Primary export | Status |
|---|---|---|
| `greenlang/agents/base.py` | `BaseAgent` (+ `AgentConfig`, `AgentResult`, `AgentMetrics`, `StatsTracker`) | **Canonical lifecycle root** |
| `greenlang/agents/base_agents.py` | `DeterministicAgent`, `ReasoningAgent`, `InsightAgent`, `AuditEntry` | **Canonical category paths** |
| `greenlang/agents/intelligent_base.py` | `IntelligentAgentBase`, `IntelligentAgentConfig`, `IntelligenceLevel` | Canonical LLM-enhanced path |
| `greenlang/agents/agentspec_v2_base.py` | `AgentSpecV2Base[InT, OutT]`, `AgentExecutionContext`, `AgentLifecycleState` | Canonical typed path (migration target) |
| `greenlang/agents/async_agent_base.py` | `AsyncAgentBase[InT, OutT]` + context | Canonical async path |
| `greenlang/agents/mixins.py` | `DeterministicMixin`, `ReasoningMixin`, `InsightMixin` (+ `get_category_mixin`, `validate_mixin_usage`) | Category composition helpers |
| `greenlang/agents/enhanced_base_agent.py` | `EnhancedBaseAgent` | **DEPRECATED v0.1.0** — remove in v1.0 |

Parallel `Agent[InT, OutT]` protocol types exist outside `greenlang/agents/`:

| Location | Export | Use |
|---|---|---|
| `greenlang/types.py:174` | `Agent(Protocol[InT, OutT])` | Generic protocol — used by `fuel_agent.py`, `grid_factor_agent.py`, `anomaly_agent_iforest.py`, `boiler_replacement_agent_ai.py`, `forecast_agent_sarima.py` (5 files) |
| `greenlang/integration/sdk/base.py:82` | `Agent(ABC, Generic[TInput, TOutput])` | Integration-SDK ABC |
| `greenlang/config/greenlang_registry/db/models.py:47` | `Agent(Base)` | SQLAlchemy model (registry DB) — unrelated to the runtime base class |
| `greenlang/integration/api/graphql/types.py:388` | `Agent` (GraphQL type) | GraphQL schema — unrelated |
| `greenlang/cli/agent_factory/create_command.py:1324`, `utilities/factory/sdk/python/agent_factory.py:1115` | `Agent` (local classes) | Agent Factory scaffolding |

> **Implication for Phase 1.2:** we do NOT need to unify `greenlang.types.Agent` with `greenlang.agents.base.BaseAgent`. `Agent[InT, OutT]` is the typed protocol; `BaseAgent` is the lifecycle class. They are orthogonal today. The 5 agents that inherit `Agent[InT, OutT]` should migrate to `AgentSpecV2Base[InT, OutT]` which composes both concerns.

---

## 5. Inheritance counts (files that subclass each base)

| Base class | Files with a subclass | Assessment |
|---|---|---|
| `BaseAgent` | **146** | Dominant. Migration target. Most of these should route to a category mixin on `AgentSpecV2Base` or stay on `BaseAgent` with an explicit reason. |
| `DeterministicAgent` | **68** | Healthy — CRITICAL PATH (CBAM, emissions, factor lookups). |
| `ReasoningAgent` | **2** | Minimal. |
| `InsightAgent` | **6** | Minor. Hybrid agents (anomaly + explanation, forecast + interpretation). |
| `IntelligentAgentBase` | **4** | Emerging. Expected to grow as LLM-enhanced agents migrate off plain `BaseAgent`. |
| `EnhancedBaseAgent` | **0** | Dead. Safe to remove in v1.0 — add runtime `DeprecationWarning` now. |
| `AsyncAgentBase` | **0** | Unused. Available for new async-heavy agents (Connect extracts, ERP ingests). |
| `AgentSpecV2Base` | **0** | Unused. **Formal migration target** — see Phase 1.3 checklist. |
| `Agent[InT, OutT]` (protocol from `greenlang/types.py`) | 5 | Typed agents on a non-lifecycle protocol. Should migrate to `AgentSpecV2Base[InT, OutT]`. |

**Import-site counts** (from the `agent_runtime/__init__.py` header comment): **183+ importers of `BaseAgent`**, **62 importers of the Intelligence-Paradox bases** (`DeterministicAgent` / `ReasoningAgent` / `InsightAgent`). These are the surface we need to redirect toward `greenlang.agent_runtime` so that future base-class moves stay backwards-compatible.

---

## 6. High-traffic FY27 agents — current signatures

These agents are on the critical path for CBAM, Comply, Scope Engine, and Factors pilots. Document their current base class so the Phase 1.3 migration can track per-agent diffs.

| # | Agent | Path | Current class signature |
|---|---|---|---|
| 1 | CBAM compliance | `greenlang/agents/policy/cbam_compliance_agent.py` | `class CBAMComplianceAgent(BaseAgent):` |
| 2 | CSRD compliance | `greenlang/agents/policy/csrd_compliance_agent.py` | `class CSRDComplianceAgent(BaseAgent):` |
| 3 | Emissions guardian | `greenlang/agents/gl_010_emissions_guardian.py` | `class EmissionsGuardian(BaseAgent):` |
| 4 | Carbon (sync) | `greenlang/agents/carbon_agent.py` | `class CarbonAgent(IntelligenceMixin, OperationalMonitoringMixin, BaseAgent):` |
| 5 | Carbon (AI) | `greenlang/agents/carbon_agent_ai.py` | `class CarbonAgentAI(BaseAgent):` |
| 6 | Fuel | `greenlang/agents/fuel_agent.py` | `class FuelAgent(Agent[FuelInput, FuelOutput]):` |
| 7 | Grid factor | `greenlang/agents/grid_factor_agent.py` | `class GridFactorAgent(Agent[GridFactorInput, GridFactorOutput]):` |
| 8 | Intensity | `greenlang/agents/intensity_agent.py` | `class IntensityAgent(IntelligenceMixin, BaseAgent):` |
| 9 | Report | `greenlang/agents/report_agent.py` | `class ReportAgent(OperationalMonitoringMixin, BaseAgent):` |
| 10 | Fuel (AI) | `greenlang/agents/fuel_agent_ai.py` | **DEPRECATED** — callers should use `FuelAgent` |

Most high-traffic agents mix `BaseAgent` with ad-hoc mixins (`IntelligenceMixin`, `OperationalMonitoringMixin`) that are not part of the canonical `agent_runtime` export. **This is the mixin cleanup that Phase 1.3 formalizes.**

---

## 7. Duplicate-file archival check

The doc predicted 16 duplicate variants. All 16 are **confirmed archived** under `_archive/05_duplicate_agent_versions/`:

| File | Status |
|---|---|
| `fuel_agent_v2.py`, `fuel_agent_ai_v2.py`, `fuel_agent_ai_async.py`, `fuel_agent_ai_sync.py`, `fuel_agent_intelligent.py`, `fuel_tools_v2.py` | Archived |
| `recommendation_agent_ai.py`, `recommendation_agent_ai_v2.py`, `recommendation_agent_intelligent.py` | Archived |
| `boiler_replacement_agent_ai_v3.py`, `boiler_replacement_agent_ai_v4.py` | Archived |
| `industrial_heat_pump_agent_ai_v3.py`, `industrial_heat_pump_agent_ai_v4.py` | Archived |
| `grid_factor_agent_ai.py`, `grid_factor_agent_intelligent.py` | Archived |
| `carbon_agent_intelligent.py` | Archived |

No active duplicates remain in `greenlang/agents/`.

---

## 8. What changes in Phase 1.2 and 1.3

Phase 1.2 (code — next):

- Upgrade `EnhancedBaseAgent` deprecation from docstring-only to a **runtime `DeprecationWarning` at class creation** (via `__init_subclass__`).
- Update `greenlang/agents/__init__.py` to route import paths through `greenlang.agent_runtime` (backwards-compatible re-export with a soft deprecation note for direct paths).
- Add a public, version-pinned declaration of the canonical hierarchy in `greenlang/agent_runtime/__init__.py` (tag as `CANONICAL_V3 = True`).

Phase 1.3 (docs — after 1.2):

- File-by-file migration checklist in `docs/migration/AGENT_BASE_CONSOLIDATION.md`.
- Concrete migration diff for 10 high-traffic agents (§6 above) as copy-paste templates.
- Staged plan for the remaining ~190 class-bearing files in subdirectories (policy, mrv, scope3, data, operations, foundation, finance, procurement, reporting, intelligence, ecosystem).

---

## 9. References

- `greenlang/agent_runtime/__init__.py` — canonical import surface.
- `greenlang/agents/MIGRATION_TO_AGENTSPECV2.md` — existing per-pattern migration steps (keep as supplementary reference).
- `greenlang/agents/enhanced_base_agent.py` — deprecation docstring (Phase 1.2 upgrades to runtime warning).
- `_archive/05_duplicate_agent_versions/` — parked legacy variants.
- [`docs/CLI_REFERENCE.md`](../CLI_REFERENCE.md) — CLI surface that ultimately dispatches these agents.
