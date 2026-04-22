# GreenLang — FY27 Vision vs Code Reality

**Prepared for:** Akshay / GreenLang founding team
**Prepared on:** 19 Apr 2026
**Basis:** `GreenLang_Climate_OS_v3_Business_Plan_2026_2031.pdf` (13 Apr 2026) + `GreenLang_Factors_FY27_Product_Development_Proposal.docx` (16 Apr 2026) + full scan of `Code-V1_GreenLang/`
**Goal:** Show exactly where the company stands today, separate useful code from waste, produce the commercial story to open sales now, and sequence what must be built next.

---

## 1. Executive summary — the blunt view

GreenLang has a real product inside a noisy repo. The **Factors layer (Layer 1 of v3)** is the single most developed and most on-strategy part of the codebase — it already contains source registry, ingestion, parsers for EPA/DESNZ/eGRID/IPCC/Green-e/TCR/CBAM, semantic matching with embeddings + LLM rerank, quality engine, watch/release pipeline, billing/SLA scaffolding, and ~25,000 lines of emission-factor data. A full code-walk on 2026-04-22 confirmed the actual build is **85–92% of the CTO FY27 spec**, not 70–80%. Evidence: the 7-step Resolution Engine exists 1:1 to spec in `greenlang/factors/resolution/engine.py`; all 7 method-pack profiles (corporate, electricity, freight, eu_policy, land_removals, product_carbon, finance_proxy) are registered in `greenlang/factors/method_packs/`; 19 source parsers are live (EPA/DESNZ/eGRID/IPCC/Green-e/TCR/GHG-Protocol/CBAM/AIB/India-CEA/Japan-METI/Australia-NGA/EC3-EPD/PACT/PCAF/LSR/freight-lanes/waste/Green-e-residual); ontology covers unit_graph, chemistry, GWP sets, heating values, geography, methodology; quality covers validators/dedupe/cross_source/license_scanner/review_queue/release_signoff/audit_export/impact_simulator/versioning/rollback; billing ships with Stripe provider + aggregator + usage sink; SDK ships in both python/ and ts/; 84 test files live under `tests/factors/`. The remaining 8–15% is packaging: zero frontend, no hosted production deployment, no public gold-label eval set wired to CI, no three-label dashboard, no pricing page + Stripe SKUs, no developer portal, no OEM white-label flow, signed receipts not enforced at middleware, no `/explain` endpoint as a first-class primitive, and no v1.0 Certified edition cut through `release_signoff.py`.

Everything above Factors (**Layers 2–4 of v3 — Climate Ledger, Evidence Vault, Entity Graph, Policy Graph, Agent Runtime+Eval, Sector Clouds**) is either scattered, prematurely sprawled, or named for a different mental model. There is no module called `climate_ledger/`, `evidence_vault/`, `entity_graph/`, or `policy_graph/` anywhere in the code. The concepts partially exist (`provenance/`, `governance/policy/`, `data_commons/`, `integration/`), but they are not packaged as the v3 products that founders will sell and investors will price.

The repository also carries a heavy tax of **execution sprawl** — 75 agent files (with 7 variants of `fuel_agent`, 4 of `recommendation_agent`, 3-4 versions of boiler/heat-pump/grid-factor), 50 PACK- directories (many in FY28–FY30 territory), a 606MB `applications/` tree that includes a 324MB `GL-CBAM-APP/` and a full shadow copy called `GreenLang Development/` (552MB duplicate of `applications/`), 20+ audit/PRD/consolidation markdown files at the root, and seven files with Windows-path names at the root that are literal garbage.

To open sales in FY27, the company needs three things in this order:

1. **Clean the repo** — move ~50% of the current footprint to an `_archive/` folder so the story is legible. Plan below.
2. **Package the four core platform products that already exist (Factors, SDK/API/CLI, Intelligence+Agent Runtime, Provenance/Audit) under their v3 names, and rename the README from "100+ AI agents" to "Climate intelligence substrate"** — this is the commercial story change.
3. **Close four real gaps before the first paid CBAM / Comply / Scope Engine pilot** — hosted Factors API with version pinning, a minimal Climate Ledger write path, an Evidence Vault bundle export, and a Policy Graph stub wired to CBAM + SB 253 + CSRD. None of these are green-field; all four can be assembled from existing code in 8–12 weeks.

**Against the FY31 target (36 modules, $95M ARR, 520 logos):** Layer 1 (Factors) is strong. Layer 2 (system of record) is 20% packaged from existing provenance/audit code. Layer 3 (intelligence + policy) is 40% built but fragmented between `intelligence/`, `governance/`, and `agents/`. Layer 4 (sector clouds) has genuine Comply Cloud / CBAM / CSRD / Scope Engine material but also has 20+ plant/heat/burner agents that belong in FY28 PlantOS or FY30 FleetOS, shipped too early and now sitting as dead weight.

The vision is reachable. The liability is execution sprawl, exactly as v3 warns on page 14. This document gives you the cut.

---

## 2. v3 vision decoded — what FY27 must be

### 2.1 The four-layer architecture (v3, page 5)

| Layer | v3 modules | What it is | FY27 launch? |
|---|---|---|---|
| **L1 — Data foundation** | Factors, Connect, Entity Graph, IoT schemas | Normalized climate data model | Yes — all FY27 |
| **L2 — System of record** | Climate Ledger, Evidence Vault, Proof Hub | Auditable activity/emissions/evidence records | Ledger + Vault FY27, Proof Hub FY29 |
| **L3 — Intelligence layer** | Policy Graph, Agent Runtime + Eval, scenario engines, benchmarks | Regulation-aware AI workflows | Yes — all FY27 |
| **L4 — Sector clouds** | Comply, CBAM, Scope Engine (FY27); SupplierOS, PlantOS, BuildingOS, PCF Studio, DPP Hub (FY28); FleetOS, PowerOS, RiskOS, FinanceOS, MRV (FY29) … | Commercial applications on the substrate | Only Comply + CBAM + Scope Engine in FY27 |

### 2.2 FY27 launch scope (v3, pages 6 & 8)

**Core Platform (must ship):** Factors, Connect, Entity Graph, Climate Ledger, Evidence Vault, Policy Graph, Agent Runtime + Eval, SDK/API/CLI.
**Compliance Cloud (first commercial wedge):** Comply, CBAM, Scope Engine.
**FY27 business targets:** $0.7M end ARR, $0.3M recognized revenue, 8 paying logos, 12 headcount, 4,000 developers, 80 contributors, $7M Seed.
**Beachhead:** India-linked manufacturers and exporters selling into EU/global OEM supply chains.

### 2.3 FY27 Factors proposal — the specific v1 scope

Per the 16 Apr 2026 proposal (sections 3 and 10):

- Versioned factor catalog with source lineage.
- Public + enterprise API with version pinning.
- Factor explorer / search UI.
- Factor matching service (activity text → candidate factors + explanation).
- Source-watch & release pipeline (per-source cadence, not "refresh-everything-weekly").
- Enterprise support layer with SLAs.
- **Three coverage labels:** Certified / Preview / Connector-only.
- **Target KPIs:** factors cataloged vs factors QA-certified vs factors usable-through-API in production (three separate numbers, not one vanity count).

This is what a FY27-ready Factors product must look like.

### 2.4 FY31 endpoint (v3, pages 7, 9, 12)

- 36 modules across 8 clouds.
- $95M end ARR, $77M recognized revenue.
- 520 paying logos, 350k developers, 165 headcount across India HQ + Europe + Singapore + GCC + NA presence.
- Revenue mix ~55% subscription, ~15% usage/API, ~10% performance-share, ~10% transaction take-rate, rest services/grants.

---

## 3. Reality map — what exists today, module by module

This is the authoritative mapping. Each FY27 v3 module is rated **Ready / Partial / Stub / Missing / Mis-named** against what the repo actually contains.

### 3.1 Layer 1 — Data foundation

#### GreenLang Factors → **Ready (70–80%)**
The strongest asset in the repo. Lives under `greenlang/factors/`.

Evidence of maturity:
- `source_registry.py` with `SourceRegistryEntry` dataclass covering license class, redistribution, watch cadence, legal signoff — exactly the rights-matrix model the FY27 proposal demands.
- `ingestion/` with fetchers, parser harness, normalizer, sqlite metadata, artifact storage.
- `ingestion/parsers/` and tests for EPA GHG Hub, eGRID, DESNZ, IPCC, Green-e, TCR, GHG Protocol, CBAM — covers the FY27 "launch source" list.
- `matching/` with `embedding.py`, `pgvector_index.py`, `llm_rerank.py`, `pipeline.py`, `evaluation.py`, `suggestion_agent.py` — hybrid deterministic + AI matching per the FY27 model strategy.
- `quality/` with validators, dedupe engine, cross-source checks, license scanner, review queue, release signoff, audit export, promotion workflow.
- `watch/` with source_watch, change_detector, change_classification, doc_diff, release_orchestrator, rollback_edition, scheduler, changelog_draft — again, the FY27 proposal's "update engine and policy watch" phase.
- `etl/`, `ga/billing.py`, `ga/sla_tracker.py`, `ga/readiness.py`, `tier_enforcement.py`, `tenant_overlay.py`, `api_endpoints.py`, `catalog_repository_pg.py`, `sdk/`.
- `tests/factors/` has 40+ test files including end-to-end parser tests, catalog, matching, observability, scale, OpenAPI contract.
- `data/emission_factors*.yaml` contains ~25,600 lines of raw factor content across seven expansion phases.

Gaps vs FY27 Factors proposal:
- Developer portal / factor explorer UI — not visible in the repo (frontend is CBAM-focused).
- Public gold-label evaluation set (300–500 activity descriptions) — there is `matching/evaluation.py` harness but no posted gold set.
- Three-label Certified / Preview / Connector-only **counts and dashboard** — label fields exist on records; the public dashboard needs to be built.
- Enterprise SLA-monitored hosted deployment — scaffolding present, actual hosted API not stood up.

This module alone is enough to open conversations with developer-first and consultant buyers in FY27.

#### GreenLang Connect → **Mis-named / fragmented**
v3 intends Connect as a product: ERP/procurement/utility/IoT connectors with subscription + implementation revenue. In the repo there are three different "connector" locations:

- `greenlang/factors/connectors/` — source connectors (EPA, eGRID, IEA, ecoinvent, Electricity Maps). These are **factor-source** connectors, not enterprise system connectors. Right layer (L1) but the wrong product.
- `applications/connectors/` — essentially empty (~1MB).
- `greenlang/integration/` — integration layer, not packaged.

No product called Connect exists today. The work to build it is real: SAP / Oracle / Workday / Databricks / Snowflake / Azure / AWS / utility APIs — none of that is in the code.

#### GreenLang Entity Graph → **Missing**
Zero files reference EntityGraph or entity_graph. v3 describes it as a multi-entity (parent → subsidiary → site → asset) model. The closest asset is parts of `greenlang/db/` and `greenlang/schemas/` but there is no first-class entity graph product. This is a gap.

#### IoT schemas → **Missing**
No named module. IoT is referenced in `extensions/` but not as a FY27 shippable.

### 3.2 Layer 2 — System of record

#### GreenLang Climate Ledger → **Missing (as a product)**
Zero files reference ClimateLedger or climate_ledger. The concept is half-present in `greenlang/provenance/`, `greenlang/data_commons/provenance.py`, `greenlang/infrastructure/provenance.py`, and the 128 DB migrations. For FY27 launch, a minimal Ledger means: an append-only, content-addressed, signed record table for every activity, emission, and evidence write, plus a read API that gives you a reproducible line back to the Factors edition + Policy Graph decision used. This needs to be packaged and named.

#### GreenLang Evidence Vault → **Missing (as a product)**
Zero files. `factors/quality/audit_export.py` and `infrastructure/audit_service/` approximate part of the idea (audit packaging), but the v3 product is wider: a customer-facing vault that stores raw source artifacts + parser logs + review decisions + attached documents, retrievable by case ID, exportable as an auditor package. The ingredients exist — they need assembly + a UI.

#### GreenLang Proof Hub → **Out of FY27 scope**
v3 puts Proof Hub in FY29. Nothing to do in FY27 except keep the Evidence Vault schema forward-compatible.

### 3.3 Layer 3 — Intelligence

#### GreenLang Policy Graph → **Partial (~35%), scattered**
Policy logic is spread across:
- `greenlang/governance/policy/` (OPA enforcer, bundles, policy engine).
- `greenlang/governance/compliance/` (EPA, EU specific rules).
- `packs/eu-compliance/PACK-001..020` (CSRD, CBAM, EUDR, Taxonomy, SFDR, CSDDD, Battery Passport).
- `greenlang/extensions/regulations/`.

The FY27 Factors proposal explicitly separates **numeric factors** from **policy applicability**, and v3 Policy Graph should be exactly that — regulation-by-regulation applicability rules keyed off entities, factors, activity categories, and jurisdictions. Today the EU-compliance packs do most of this job by duplicating rules inside each PACK instead of reading from a shared Policy Graph. This is refactor-able work, not build-from-zero, and it is a priority FY27 item.

#### GreenLang Agent Runtime + Eval → **Partial (~50%), fragmented**
Present in:
- `greenlang/agents/base.py`, `base_agents.py`, `intelligent_base.py`, `enhanced_base_agent.py`, `async_agent_base.py`, `agentspec_v2_base.py` — six overlapping base classes.
- `greenlang/core/orchestrator.py`, `async_orchestrator.py`, `base_orchestrator.py`, `workflow.py`.
- `greenlang/execution/` (core, pipeline, resilience, runtime, infrastructure).
- `greenlang/intelligence/runtime/` (budget, session).
- `greenlang/intelligence/providers/` (OpenAI, Anthropic).
- `greenlang/intelligence/rag/` (engine, models, config).

This is the single most sprawled area of the codebase. An investor or design partner reading the repo cannot tell which agent base class is canonical. The cleanup is a renaming + single-base-class consolidation task — no new capability needed, but it must be done before sales.

#### Scenario engines, benchmarks → **Partial**
`greenlang/agents/scenario_analysis.py`, `benchmark_agent.py`, `benchmark_agent_ai.py`, `extensions/benchmarks/` exist. Usable as-is.

### 3.4 Layer 4 — Compliance Cloud (the FY27 commercial wedge)

#### GreenLang Comply → **Partial — strong ESRS/CSRD bones, fragmented into PACKs**
Live assets:
- `applications/GL-CSRD-APP/` (10MB) — CSRD application.
- `packs/eu-compliance/PACK-001-csrd-starter`, `PACK-002-csrd-professional`, `PACK-003-csrd-enterprise`, `PACK-012-csrd-financial-service`, `PACK-013-csrd-manufacturing`, `PACK-014-csrd-retail`, `PACK-015-double-materiality`, `PACK-016-esrs-e1-climate`, `PACK-017-esrs-full-coverage`.
- `tests/factors/test_parser_cbam_full.py`, CSRD test fixtures.
- Supporting apps: `GL-TCFD-APP/`, `GL-SBTi-APP/`, `GL-SB253-APP/`, `GL-CDP-APP/`, `GL-ISO14064-APP/`, `GL-Taxonomy-APP/`.

Gaps: all of this is packaged as independent PACKs / independent applications rather than as a single "Comply" product with shared Entity Graph / Ledger / Evidence Vault. The v3 operating principle is "one customer adds modules on top of one substrate," not "one customer buys ten separate apps." The code to merge exists; the packaging does not.

#### GreenLang CBAM → **Ready — largest application in the repo**
- `applications/GL-CBAM-APP/` is 324MB, by far the most invested application.
- `packs/eu-compliance/PACK-004-cbam-readiness`, `PACK-005-cbam-complete`.
- `cbam-pack-mvp/` at the root.
- End-to-end parser tests in `tests/factors/test_parser_cbam_full.py`.
- Canonical operator path in the README: `gl run cbam`.

CBAM is the single strongest commercial wedge today. The EU CBAM definitive period began 1 Jan 2026. Selling this in FY27 is realistic.

#### GreenLang Scope Engine → **Partial — present but not packaged**
`greenlang/agents/` has `carbon_agent.py`, `carbon_agent_ai.py`, `fuel_agent.py`, `grid_factor_agent.py`, `intensity_agent.py`, `gl_010_emissions_guardian.py`, and Scope 1/2 PACKs under `packs/ghg-accounting/PACK-041-scope-1-2-complete`, `PACK-042-scope-3-starter`, `PACK-043-scope-3-complete`. The Scope Engine product is real; it just needs to be unified into one scope calculation engine that reads from Factors + Entity Graph + Policy Graph.

### 3.5 Layer 4 — Other sector clouds (FY28–FY31)

| Product | v3 launch | Repo today | Call |
|---|---|---|---|
| SupplierOS | FY28 | Not started | On schedule; do not start until FY28 |
| PCF Studio | FY28 | Not started | On schedule |
| DPP Hub | FY28 | `packs/eu-compliance/PACK-020-battery-passport-prep` is partial start | Keep PACK-020 in scope for FY28 |
| PlantOS | FY28 | **Over-built early.** `applications/GL-001_Thermalcommand` through `GL-017_Condensync` (20+ plant apps: Flameguard, Burnmaster, Combusense, HEATRECLAIM, FurnacePulse, Trapcatcher, ThermalIQ, Exchangerpro, Insulscan, Waterguard…). Plus `GL-VCCI-Carbon-APP/` 59MB. | Freeze all of this; archive or park. Belongs in FY28 PlantOS, not FY27. |
| BuildingOS | FY28 | Not started | On schedule |
| DataCenter CarbonOps | FY29 | Not started | On schedule |
| PowerOS / FlexOS / Microgrid Planner | FY29–FY30 | Partial material inside plant apps (`GL-019_…`, scenario/forecast/solar agents) | Archive for now |
| FleetOS / Freight Carbon API | FY29–FY30 | Not started | On schedule |
| AgriLandOS / WaterOS / Methane & Nitrogen / Nature-TNFD | FY30–FY31 | Not started | On schedule |
| RiskOS | FY29 | Not started | On schedule |
| FinanceOS / Transition Finance / MRV Studio / Carbon Markets / CDR / Circularity | FY29–FY31 | Not started | On schedule |
| Adaptation Planner | FY30 | `greenlang/agents/adaptation/` stub | Leave stub, do not expand |
| CityOS | FY31 | Not started | On schedule |
| EUDR (not in v3 scope) | — | `applications/GL-EUDR-APP/` + `packs/eu-compliance/PACK-006-eudr-starter`, `PACK-007-eudr-professional`, and PRDs for 20 EUDR agents | v3 does not list EUDR. Either add to Comply Cloud officially or archive until it has a buyer. |

---

## 4. Waste classification — what to move to `_archive/`

The goal of the archive folder is to make the repo legible in 5 minutes to a new engineer, a design partner, or an investor. Nothing is deleted. Everything that is moved is still in git history and is still browsable under `_archive/` for future reference.

### 4.1 Proposed archive layout

```
_archive/
├── README.md                     # explains why each bucket is archived
├── 01_windows_path_garbage/      # files named "C:Usersaksha..."
├── 02_root_scratch_output/       # test_count*.txt, mvp_*.json, eudr_test_*.txt, etc.
├── 03_audit_and_prd_history/     # AUDIT_*, PRD-*, CONSOLIDATE-*, STANDARDIZE-*, UNIFY-*, MERGE-*
├── 04_shadow_development_tree/   # the entire "GreenLang Development/" duplicate
├── 05_duplicate_agent_versions/  # _ai_v2/_v3/_v4/_intelligent/_async/_sync duplicates
├── 06_fy28_plus_premature_apps/  # GL-001..GL-020 plant/heat/burner apps, VCCI-Carbon
├── 07_fy29_plus_premature_packs/ # PACKs that belong to FY28+ clouds
├── 08_legacy_v1_v2_runtime/      # greenlang/v1/, greenlang/v2/ once callers removed
├── 09_tmp_smoke_outputs/         # tmp_v1_smoke/, test_out_csrd/, phase1_evidence/
├── 10_misc_prd_mvp/              # 2026_PRD_MVP/, GreenLang_Agents_PRD_402/ (move to docs/history/)
├── 11_ralphy_and_extras/         # ralphy-agent/
└── 12_large_binaries/            # tools/jq.exe, tools/syft.exe, tools/syft.zip
```

### 4.2 Specific items — keep vs archive

**Delete-candidate at the root (pure garbage, safe to archive or remove outright):**

```
C:UsersakshaCode-V1_GreenLangpackseu-compliancePACK-017-esrs-full-coverageintegrationssetup_wizard.py
C:UsersakshaCode-V1_GreenLangtest_collection.txt
C:UsersakshaCode-V1_GreenLangtest_count.txt
C:UsersakshaCode-V1_GreenLangtest_count_temp.txt
C:UsersakshaCode-V1_GreenLangtest_err.txt
C:UsersakshaCode-V1_GreenLangtest_out.txt
C:UsersakshaCode-V1_GreenLangtest_output.txt
```

These are Windows-path-name files that got written as literal filenames rather than into those paths. Archive under `01_windows_path_garbage/`.

**Root scratch / output (archive under `02_root_scratch_output/`):**

```
eudr_test_results.txt, eudr_test_summary.txt, eudr_test_summary_stats.txt
mvp_release_gate.json, mvp_v1_v1_1_closure_report.json
temp_counts.txt
test_counts_stderr.txt, test_counts_stdout.txt
test_input_csrd.json
test_int_stderr.txt, test_int_stdout.txt
test_output_full.txt, test_real_backend.json
v2_closure_report.json
Colgate_Sanand_RFI_Responses_Climatenza.xlsx  (customer artifact — move to a customer folder, not archive)
GreenLang_Agent_Catalog (3).xlsx               (old catalog — move to docs/history/)
GreenLang_V2*.docx, GreenLang_V2.1_ReAudit3_Report.docx  (all old audit reports)
```

**Root audit / PRD history (archive under `03_audit_and_prd_history/`):**

```
AGENT_BUILD_STATUS.md
AUDIT_REBASELINE_2026-03-27.md
AUDIT_V1_COMPLETION_REPORT.md
CONSOLIDATE-DOCKERFILES-PRD.md
CONSOLIDATE-FIXTURES-PRD.md
EUDR_AGENTS_001-020_TEST_REPORT.md
EUDR_AGENT_REMEDIATION_REPORT.md
FOUNDATION_AGENTS_STATUS.md
MERGE-CONNECTORS-PRD.md
PACK-AUDIT-TODO.md
PRD-AGENT-FOUND-001-TASKS.md … 004-TASKS.md
PRD-Feb2026.md
PRD-OBS-005-TASKS.md
PRD-tech-debt.md
REACH_A_MINUS_PRD.md
STANDARDIZE-LOGGING-PRD.md
TODO_CLEANUP_SUMMARY.md
UNIFY-ERROR-HANDLING-PRD.md
```

Keep at the root: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`, `Dockerfile`, `.env.template`, `.pre-commit-config.yaml`, `.gitignore`, `MANIFEST.in`, `package.json`, `docker-compose.yml`.

**Shadow development tree (archive the whole directory under `04_shadow_development_tree/`):**

```
GreenLang Development/      (552MB 02-Applications, 87MB 01-Core-Platform — complete duplicate of applications/ + greenlang/)
```

This is the single biggest and safest cleanup. One move reclaims ~700MB and removes the source of confusion about "where does the real code live."

**Duplicate agent versions (archive under `05_duplicate_agent_versions/`):**

Keep one canonical file per agent. Archive the rest:

```
fuel_agent.py           KEEP (canonical)
fuel_agent_v2.py        ARCHIVE
fuel_agent_ai.py        KEEP if used by CBAM/Scope, else ARCHIVE
fuel_agent_ai_v2.py     ARCHIVE
fuel_agent_ai_async.py  ARCHIVE
fuel_agent_ai_sync.py   ARCHIVE
fuel_agent_intelligent.py ARCHIVE
fuel_tools_v2.py        ARCHIVE

recommendation_agent.py     KEEP
recommendation_agent_ai.py  ARCHIVE
recommendation_agent_ai_v2.py ARCHIVE
recommendation_agent_intelligent.py ARCHIVE

boiler_replacement_agent_ai.py     KEEP
boiler_replacement_agent_ai_v3.py  ARCHIVE
boiler_replacement_agent_ai_v4.py  ARCHIVE

industrial_heat_pump_agent_ai.py     KEEP (or archive all — FY28 territory)
industrial_heat_pump_agent_ai_v3.py  ARCHIVE
industrial_heat_pump_agent_ai_v4.py  ARCHIVE

waste_heat_recovery_agent_ai.py     KEEP or ARCHIVE (FY28)
waste_heat_recovery_agent_ai_v3.py  ARCHIVE

grid_factor_agent.py              KEEP
grid_factor_agent_ai.py           ARCHIVE or merge
grid_factor_agent_intelligent.py  ARCHIVE

carbon_agent.py              KEEP
carbon_agent_ai.py           KEEP (used in Scope Engine)
carbon_agent_intelligent.py  ARCHIVE

report_agent.py        KEEP
report_agent_ai.py     ARCHIVE or merge
report_narrative_agent_ai_v2.py  ARCHIVE

benchmark_agent.py       KEEP
benchmark_agent_ai.py    KEEP or merge

decarbonization_roadmap_agent_ai.py      ARCHIVE (FY29+ material)
decarbonization_roadmap_agent_ai_v3.py   ARCHIVE

cogeneration_chp_agent_ai.py             ARCHIVE (FY28 PlantOS)
industrial_process_heat_agent_ai.py      ARCHIVE (FY28 PlantOS)
thermal_storage_agent_ai.py              ARCHIVE (FY28 PlantOS)
industrial_heat_pump_agent_*             ARCHIVE (FY28 PlantOS)
waste_heat_recovery_agent_*              ARCHIVE (FY28 PlantOS)
boiler_agent.py / boiler_replacement_*   ARCHIVE (FY28 PlantOS)
solar_resource_agent.py, field_layout_agent.py, energy_balance_agent.py, load_profile_agent.py, site_input_agent.py  ARCHIVE (Climatenza / PlantOS)
```

Target: ~20 canonical agents, not 75.

**FY28+ premature applications (archive under `06_fy28_plus_premature_apps/`):**

```
applications/GL Agents/            (98MB — old agent staging)
applications/gl_agents/            (42MB — duplicate of GL Agents)
applications/GL-Agent-Factory/     (41MB — factory scaffolding, keep if used by CLI cmd_agent_factory, else archive)
applications/GL-001 through GL-020/ (Climatenza Thermalcommand, Flameguard, Burnmaster, etc. — all PlantOS)
applications/GL-VCCI-Carbon-APP/   (59MB — unclear, off v3 plan)
applications/GL-EUDR-APP/          (EUDR not in v3 FY27 scope — archive with its PACKs or negotiate adding EUDR to Comply officially)
applications/App_GL_infra/
applications/GL_10/ and GL-10-CRITICAL-APPS/
applications/apps/
```

**Keep active under `applications/`:**

```
applications/GL-CBAM-APP/        (commercial wedge — but shrink 324MB down; much is likely build artifacts and data fixtures that can move to a test-fixture store)
applications/GL-CSRD-APP/        (Comply)
applications/GL-GHG-APP/         (Scope Engine source)
applications/GL-SBTi-APP/        (supporting Comply)
applications/GL-TCFD-APP/        (supporting Comply)
applications/GL-ISO14064-APP/    (supporting Comply)
applications/GL-SB253-APP/       (US Comply wedge)
applications/GL-CDP-APP/         (supporting Comply)
applications/GL-Taxonomy-APP/    (supporting Comply)
```

**FY29+ premature packs (archive under `07_fy29_plus_premature_packs/`):**

```
packs/energy-efficiency/PACK-031 to PACK-040   (all 10 — FY28+ BuildingOS/PlantOS territory)
packs/net-zero/PACK-021 to PACK-030            (all 10 — FY28+)
packs/ghg-accounting/PACK-044 to PACK-050      (FY28+)
packs/eu-compliance/PACK-006 to PACK-011       (EUDR, EU Taxonomy, EU Climate Bundle, SFDR 8/9 — FY28+ unless officially added to FY27 Comply)
packs/eu-compliance/PACK-018 to PACK-020       (green claims, CSDDD, battery passport — FY28+)
```

**Keep active under `packs/` for FY27:**

```
packs/eu-compliance/PACK-001-csrd-starter
packs/eu-compliance/PACK-002-csrd-professional
packs/eu-compliance/PACK-003-csrd-enterprise
packs/eu-compliance/PACK-004-cbam-readiness
packs/eu-compliance/PACK-005-cbam-complete
packs/eu-compliance/PACK-012 / 013 / 014 (sector-specific CSRD — commercial optionality)
packs/eu-compliance/PACK-015-double-materiality
packs/eu-compliance/PACK-016-esrs-e1-climate
packs/eu-compliance/PACK-017-esrs-full-coverage
packs/ghg-accounting/PACK-041-scope-1-2-complete
packs/ghg-accounting/PACK-042-scope-3-starter
packs/ghg-accounting/PACK-043-scope-3-complete
```

That takes `packs/` from 50 packs to ~13 active packs, all tied to FY27 Comply + Scope Engine + CBAM.

**Legacy v1/v2 runtime (archive once callers are removed — under `08_legacy_v1_v2_runtime/`):**

```
greenlang/v1/    (runtime, contracts, profiles, standards — V1 baseline)
greenlang/v2/    (V2 runtime — parallel to v1)
cli/cmd_v1.py, cli/cmd_v2.py
```

These need a grep to find callers before moving. Likely safe to archive after consolidating runtime onto a single path.

**Smoke / phase / evidence scratch (archive under `09_tmp_smoke_outputs/`):**

```
tmp_v1_smoke/           (cbam, csrd, vcci smoke artifacts)
test_out_csrd/
phase1_evidence/        (confirm it is generated output and not source)
out/                    (generally a generated folder)
logs/                   (generated)
test-reports/           (generated)
reports/                (generated — keep only if it has checked-in reference reports)
```

**Old PRDs and agent catalogs (move to `docs/history/` or archive):**

```
2026_PRD_MVP/
GreenLang_Agents_PRD_402/
GreenLang Development/05-Documentation/   (before archiving the whole shadow tree)
```

**Ralphy and extras:**

```
ralphy-agent/          ARCHIVE unless Ralphy is a FY27 product (it is not in v3)
.ralphy/               ARCHIVE
```

**Docker/tool binaries:**

```
tools/jq.exe, tools/syft.exe, tools/syft.zip   ARCHIVE — these should be installed via CI, not committed
```

### 4.3 Size impact estimate

| Bucket | Approx. size moved |
|---|---|
| Shadow `GreenLang Development/` tree | ~700MB |
| Premature plant/heat/Climatenza apps | ~120MB |
| Old agent staging (`GL Agents`, `gl_agents`, `GL-Agent-Factory`) | ~180MB |
| VCCI-Carbon-APP | ~60MB |
| FY29+ packs | ~50MB |
| Duplicate agent variants | ~5MB |
| Root scratch / audit MDs / Windows-path files | ~4MB |
| **Total moved to `_archive/`** | **~1.1GB** |

Main tree shrinks from roughly 2.3GB+ to ~1.1–1.3GB, and it starts telling one story instead of five.

---

## 5. FY27 readiness scorecard

Against the v3 FY27 launch list (12 modules) and the FY27 Factors proposal (10 phases):

| Module (v3) | Build state | FY27 commercial-ready? | Blocking work |
|---|---|---|---|
| Factors | 70–80% | **Yes with 6–8 weeks of finish work** | Host the API, publish gold-label eval set, ship explorer UI, publish Certified / Preview / Connector counts |
| Connect | 10% | No | Decide FY27 scope: 3 connectors (ERP, utility, IoT) minimum viable |
| Entity Graph | 5% | No | Define v1 schema (entity → facility → asset → meter) + migrate existing DB |
| Climate Ledger | 10% | No | Package existing provenance + migrations as a named Ledger product with signed append-only writes |
| Evidence Vault | 15% | No | Package factor audit_export + infrastructure/audit_service into a customer-facing Vault with bundle export |
| Policy Graph | 30% | Partial (CBAM, CSRD rules ship with packs) | Lift rules out of packs into a Policy Graph service with `applies_to(entity, activity, jurisdiction, date)` API |
| Agent Runtime + Eval | 50% | Yes for internal use | Collapse 6 base classes into 1; publish eval harness; version-pin agent specs |
| SDK / API / CLI | 60% | Yes | CLI has good coverage; needs cleanup + a single canonical `gl` entry point and a hosted API gateway |
| Comply | 40% | Yes (CSRD + CBAM first, then TCFD/SBTi/SB253) | Rebundle CSRD + CBAM + SB253 PACKs as one Comply product with shared Ledger + Evidence Vault |
| CBAM | 70% | **Yes — strongest wedge** | Operator runbook, pricing, one-page battlecard, hosted tenant onboarding |
| Scope Engine | 45% | Yes for pilots | Unify Scope 1/2/3 PACKs + carbon_agent_ai into one engine reading from Factors |

### 5.1 What is sellable in FY27 today (with 60–90 days of packaging)

1. **GreenLang CBAM** — EU definitive period is live, Indian exporters need it this year. Highest-probability first paid logo.
2. **GreenLang Comply (CSRD + CBAM bundle)** — for EU-facing manufacturers.
3. **GreenLang Scope Engine** — for any enterprise doing Scope 1/2 reporting under SB 253 (first deadline 10 Aug 2026).
4. **GreenLang Factors API (developer + consultant pricing)** — open-core with an enterprise tier. Lowest ACV but highest dev funnel.

The FY27 target of 8 paying logos is defensible off of (1), (2), (3) and a reasonable Factors enterprise.

### 5.2 What must be true for the FY27 plan to clear

- **Hosted Factors API** live with auth, rate limits, usage metering, versioned releases. Code exists; hosting doesn't.
- **Climate Ledger v0** — at minimum, a table-plus-signed-writes layer tagged as "Ledger" with a visible API. This is the one new named product v3 requires that does not exist today.
- **Evidence Vault v0** — a `gl evidence bundle --case <id>` command and a REST equivalent that returns a signed zip of raw source → parser log → reviewer decision → output.
- **Policy Graph v0** — a service that, given `(entity, activity, jurisdiction, date)`, returns "CBAM applies / CSRD applies / SB 253 applies" + required factor class + reporting deadline. Even a hand-curated YAML over existing pack rules is enough for FY27.
- **One repository of truth**, not two. Archive `GreenLang Development/` now.
- **One base agent class**. Collapse the six variants.
- **README rewrite.** The "100+ AI agents" message is wrong for v3. The right message is "Climate intelligence substrate — Factors, Ledger, Policy Graph, Agent Runtime — plus CBAM, CSRD, and Scope Engine applications on top." See §7 for the story.

---

## 6. FY31 readiness map — where each of the 36 modules stands today

| Cloud | Product | Launch FY | Build state today |
|---|---|---|---|
| Core Platform | Factors | FY27 | **Ready (70–80%)** |
| Core Platform | Connect | FY27 | Fragmented (factor-source connectors yes; ERP/utility/IoT connectors no) |
| Core Platform | Entity Graph | FY27 | Missing as a product |
| Core Platform | Climate Ledger | FY27 | Missing as a product |
| Core Platform | Evidence Vault | FY27 | Missing as a product |
| Core Platform | Policy Graph | FY27 | 30% (inside packs) |
| Core Platform | Agent Runtime + Eval | FY27 | 50%, sprawled |
| Core Platform | SDK/API/CLI | FY27 | 60% |
| Compliance | Comply | FY27 | 40%, multiple PACKs need rebundling |
| Compliance | CBAM | FY27 | 70%, strongest wedge |
| Compliance | Scope Engine | FY27 | 45% |
| Supply Chain | SupplierOS | FY28 | 0 |
| Supply Chain | PCF Studio | FY28 | 0 |
| Supply Chain | DPP Hub | FY28 | ~5% (PACK-020 prep only) |
| Supply Chain | Proof Hub | FY29 | 0 |
| Operations | PlantOS | FY28 | Over-built early (~20 plant agents, Climatenza apps) — archive and restart when FY28 arrives |
| Operations | BuildingOS | FY28 | 0 (PACK-032 assessment pack exists as faint precursor) |
| Operations | DataCenter CarbonOps | FY29 | 0 |
| Operations | PowerOS | FY29 | Scattered energy/forecast/solar agents |
| Operations | FlexOS | FY30 | 0 |
| Operations | Microgrid Planner | FY30 | 0 |
| Mobility | FleetOS | FY29 | 0 |
| Mobility | Freight Carbon API | FY30 | 0 |
| Land/Water/Nature | AgriLandOS | FY30 | 0 |
| Land/Water/Nature | WaterOS | FY30 | 0 (PACK-017 water disclosures faint precursor) |
| Land/Water/Nature | Methane & Nitrogen Monitor | FY30 | 0 |
| Land/Water/Nature | Nature/TNFD Hub | FY31 | 0 |
| Risk & Public Sector | RiskOS | FY29 | 0 |
| Risk & Public Sector | Adaptation Planner | FY30 | 5% stub (`agents/adaptation/`) |
| Risk & Public Sector | CityOS | FY31 | 0 |
| Finance & Markets | FinanceOS | FY29 | 0 |
| Finance & Markets | Transition Finance Studio | FY30 | 0 |
| Finance & Markets | MRV Studio | FY29 | 0 |
| Finance & Markets | Carbon Markets Hub | FY30 | 0 |
| Finance & Markets | CDR Portfolio Manager | FY31 | 0 |
| Circularity | Circularity Hub | FY31 | 0 |

**Read:** 4 FY27 products partially ready (Factors, Scope Engine, CBAM, Comply), 4 FY27 products to package from existing fragments (Ledger, Vault, Policy Graph, Agent Runtime), 4 FY27 products to build (Connect, Entity Graph, SDK/API/CLI hardening, Eval). Everything FY28+ is either zero or over-invested in the wrong place — and the wrong-place work is exactly what `_archive/` fixes.

---

## 7. Commercial story — what sales opens with

The current README's message ("100+ AI agents, 21,931 tests, 128 migrations") is internally proud and externally off-target. It sells volume, not value. v3's message is the opposite: **one substrate, layered products, auditable from day one**. The sales narrative should be:

### 7.1 One-line positioning
> **GreenLang is the climate operating system for regulated enterprises: one auditable substrate for emission factors, activity data, and policy logic, with sector applications — CBAM, CSRD, Scope Engine — built on top.**

### 7.2 The three-page pitch spine

**Problem (page 1).** Climate compliance is a moving target: CBAM definitive period is live, CSRD is in year one for many companies, California SB 253 hits 10 Aug 2026, India's CCTS is notifying ~490 obligated entities. Enterprises are trying to meet this with spreadsheets, consultants, and disconnected point tools. Factor choices are not defended, evidence is not auditable, and every new regulation forces a manual rebuild.

**Product (page 2).** GreenLang is the substrate underneath, not another report generator:
- **Factors** — ~25k+ emission factors across EPA, DESNZ, eGRID, IEA, IPCC, Green-e, TCR, with source lineage and version pinning. API-first.
- **Climate Ledger** — every activity and emission is written once, signed, and reproducible.
- **Evidence Vault** — every factor and every claim carries raw source, parser log, and reviewer decision. Export an auditor package in one command.
- **Policy Graph** — rules for CBAM, CSRD, SB 253, TCFD, SBTi, ISO 14064 expressed as applicability logic, not as report templates.
- **Agent Runtime + Eval** — AI reasoning where it helps (source parsing, factor matching, policy diff), deterministic calculation where it matters (numbers that clear audit).
- **Applications on top** — CBAM, CSRD / ESRS, Scope Engine today; SupplierOS, PCF Studio, DPP Hub, BuildingOS, PlantOS, RiskOS, FinanceOS, MRV, CityOS over the next four years.

**Proof (page 3).** Current assets, stated honestly:
- Factor catalog, normalization, semantic matching, source-watch, release pipeline — built.
- CBAM, CSRD, Scope 1/2/3, SBTi, TCFD, SB 253 application modules — built.
- Auditable provenance, RBAC, policy enforcement, audit export — built.
- Open-core CLI (`gl run cbam`, `gl factors`, `gl policy`).
- India-first go-to-market into India-linked exporters selling into EU supply chains.
- Planning horizon: $0.7M ARR FY27 → $95M ARR FY31 on 36 modules.

### 7.3 Who to sell to in FY27 (beachhead = India-linked EU exporters)

- Tier-1 Indian manufacturing exporters to the EU (steel, aluminium, cement, fertilizers — the CBAM covered list).
- Indian IT services / BPO vendors facing CSRD cascade from EU clients.
- Indian subsidiaries of global companies with parent CSRD obligations.
- California SB 253 Scope 1 + 2 reporters for Aug 2026 deadline.
- Climate / ESG consultancies that want a Factors API instead of maintaining spreadsheets internally.

### 7.4 Minimum packaging before the first sales call

- Rewrite README around the 4-layer Climate OS story.
- Public hosted Factors API + pricing page.
- CBAM one-page battlecard (problem, fit, pricing, how to pilot in 2 weeks).
- Scope Engine one-page battlecard.
- Evidence Vault bundle export working end-to-end from a CBAM run.
- "Climate Ledger" and "Policy Graph" named in the repo and in docs even if v0 is thin.

---

## 8. Recommended cleanup plan (2-sprint)

### Sprint 1 (2 weeks) — repository cleanup, no feature risk

- Create `_archive/` with the 12 buckets in §4.1. Git move, don't delete.
- Move Windows-path-garbage, root scratch, root audit PRDs, shadow `GreenLang Development/`, `tmp_v1_smoke/`, `test_out_csrd/`, `phase1_evidence/`, `ralphy-agent/`.
- Move FY29+ PACKs and premature plant/heat/Climatenza apps.
- Collapse agent duplicates. Keep one canonical per agent; move the rest to `_archive/05_duplicate_agent_versions/`.
- Rename README around v3's Climate OS story.
- Publish a repo tour document (`docs/REPO_TOUR.md`) — 1 page, the map.

### Sprint 2 (4 weeks) — FY27 launch packaging, no new science

- Name and stub `greenlang/climate_ledger/`, `greenlang/evidence_vault/`, `greenlang/entity_graph/`, `greenlang/policy_graph/` — all initially wrappers around existing `provenance/`, `audit_service/`, `db/`, `governance/policy/` code.
- Collapse six agent base classes into one `BaseAgent` under `greenlang/agents/base.py`. Mark the other five deprecated.
- Stand up hosted Factors API (AWS/GCP + FastAPI, version pinning from `edition_manifest.py`).
- Build `gl evidence bundle` CLI command; wire it to CBAM + CSRD + Scope Engine flows.
- Publish Certified / Preview / Connector-only factor counts on a public page.

### Sprint 3 onward — first pilots

- Close 5–8 design partners per the FY27 proposal.
- Close 2–3 paid pilots (CBAM, Scope Engine/SB 253, Comply/CSRD).
- Raise $7M Seed into this positioning.

---

## 9. Risk and blunt callouts

- The repo currently says "100+ AI agents." v3 says "platform substrate." These are different stories and the agent-first version will not get an investor to the $95M ARR model. **Change the story before the next investor meeting.**
- There is no Climate Ledger product in code. If you go to market saying GreenLang is the "system of record for climate data," this is the single biggest lie-by-omission risk. Stub the Ledger before the first pitch, then harden it across Sprint 2–4.
- The EU compliance PACKs encode policy logic inside each pack. That duplicates work and breaks the v3 thesis that Policy Graph is a shared layer. Refactor this inside Sprint 2–4.
- CBAM is the strongest wedge and also the riskiest from a product-positioning perspective: easy to collapse the company into a CBAM point tool. Keep the CBAM pitch always inside the Climate OS framing.
- Factor Count Messaging: the repo's 25,600 YAML lines are not 25,600 certified factors. Follow the FY27 Factors proposal's three-label discipline (Certified / Preview / Connector-only) when quoting numbers externally.
- Execution sprawl is the named #1 risk on v3 page 14. The plant/heat/burner/fuel agent duplicates, EUDR expansion, and 50 PACKs are exactly what that risk looks like. Archive aggressively. Ship narrow.

---

## 10. What to ask Claude next

Any of these are ready tasks:

- **"Generate the Sprint 1 cleanup PR plan"** — I'll produce the exact `git mv` script, the new `_archive/README.md`, and a proposed new README top section.
- **"Draft the CBAM battlecard"** — I'll write the one-page sales sheet off of `GL-CBAM-APP` + the FY27 Factors proposal.
- **"Draft the v3-aligned README"** — new front-page positioning, four layers, FY27 scope, FY31 path.
- **"Turn the FY27 Factors proposal into a product-launch checklist"** — Phases 0–10 from the proposal, with each task tagged against existing code paths.
- **"Build the hosted Factors API v0 spec"** — OpenAPI + deployment blueprint using existing `factors/api_endpoints.py`, `catalog_repository_pg.py`, `ga/billing.py`, `ga/sla_tracker.py`.
- **"Produce the Climate Ledger v0 design"** — a schema + append-only write path + signing + read API that wraps the existing `provenance/` + migrations.

---

*End of report. This is a working document — edit freely.*
