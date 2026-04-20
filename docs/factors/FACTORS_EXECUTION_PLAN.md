# GreenLang Factors — 100 % Execution Plan

> **Status (2026-04-20):** Phase F1 shipped (canonical-v2 schema + non-negotiables enforcement + 30 tests green).
> **Source of truth:** CTO product outlook (2026-04-20) + audit reports in `docs/factors/FACTORS_CTO_GAP_PLAN.md`.
> **This doc:** every remaining item, phase by phase, at implementation granularity. Each item has an owner module, acceptance gate, and test count.

---

## Phase F2 — Method Pack Library *(P0, ~2 wks)*

**Goal:** the product becomes commercial rather than academic. Callers resolve factors via a *method profile* (e.g. `corporate_scope2_location_based`), not by raw catalog queries.

### F2.0 — Registry + base classes

| File | Purpose |
|---|---|
| `greenlang/factors/method_packs/__init__.py` | Public surface (`MethodPack`, `get_pack`, `list_packs`, `register_pack`) |
| `greenlang/factors/method_packs/base.py` | Abstract `MethodPack` dataclass; `SelectionRule`, `BoundaryRule`, `FallbackStep`, `DeprecationRule` |
| `greenlang/factors/method_packs/registry.py` | Process-wide registry keyed by `MethodProfile` enum |
| `tests/factors/method_packs/test_base.py` | Registry + composition tests (≥ 8) |

### F2.1 — Corporate Inventory Pack

| File | What ships |
|---|---|
| `greenlang/factors/method_packs/corporate.py` | `CorporateScope1Pack`, `CorporateScope2LocationPack`, `CorporateScope2MarketPack`, `CorporateScope3Pack` |
| Rules | GHG Protocol Scope 1/2/3 inclusion/exclusion, biogenic treatment (fossil-only for Scope 1; biogenic reported separately), AR6-100 GWP basis, operational-control default |
| Region hierarchy | facility → utility → country → GLOBAL |
| Tests | ≥ 15 (inclusion rules, GWP selection, biogenic handling, fallback cascade, deprecation warnings) |

### F2.2 — Electricity Pack

| File | What ships |
|---|---|
| `greenlang/factors/method_packs/electricity.py` | `ElectricityLocationPack`, `ElectricityMarketPack`, `ElectricitySupplierPack`, `ElectricityResidualMixPack` |
| Rules | Scope 2 location vs market split; market-based RECs/GO stacking; AIB residual mix fallback for EU; T&D loss inclusion per region |
| Parsers | `greenlang/factors/ingestion/parsers/india_cea.py` (India CEA CO2 baseline database v20+) + `greenlang/factors/ingestion/parsers/aib_residual_mix.py` (AIB European residual mix 2023/2024) |
| Tests | ≥ 12 |

### F2.3 — EU Policy Pack

| File | What ships |
|---|---|
| `greenlang/factors/method_packs/eu_policy.py` | `CBAMPack` (embedded emissions for covered goods), `DPPPack` (Digital Product Passport shape stub) |
| Rules | CBAM default-values selector by CN code; verification status gate (only `EXTERNAL_VERIFIED` or `REGULATOR_APPROVED` factors can back a CBAM declaration); DPP-ready product data structure (PACT-aligned) |
| Tests | ≥ 10 |

### F2.4 — Product Carbon Pack (stub)

| File | What ships |
|---|---|
| `greenlang/factors/method_packs/product_carbon.py` | `ProductCarbonPack` implementing ISO 14067 + GHG Protocol Product Standard shape |
| Boundary rules | Cradle-to-gate vs cradle-to-grave, PACT data-exchange object layout, recycled-content + allocation flags |
| Tests | ≥ 8 (schema-only pending full LCI data in F9) |

### F2.5 — Freight Pack (stub)

| File | What ships |
|---|---|
| `greenlang/factors/method_packs/freight.py` | `FreightISO14083Pack` with mode-specific (road/sea/air/rail), WTW/TTW, lane-based + payload-based calculations |
| GLEC compliance | Chain-calculation structure (first-leg + last-leg + modal mix) |
| Tests | ≥ 10 |

### F2.6 — Land/Removals + Finance Proxy (FY28-ready stubs)

| File | What ships |
|---|---|
| `greenlang/factors/method_packs/land_removals.py` | `LSRPack` — GHG Protocol Land Sector and Removals Standard hooks (biochar, reforestation, soil carbon, BECCS); permanence + reversal risk flags |
| `greenlang/factors/method_packs/finance_proxy.py` | `PCAFPack` — attribution factors for listed equity, corporate bonds, mortgages, project finance; data-quality score 1-5 |
| Tests | ≥ 6 each (shape smoke only; full pack lands in F9) |

---

## Phase F3 — Resolution Engine + Explain endpoint *(P0, ~2 wks)*

**Goal:** non-negotiable #3 ("never hide fallback logic") + the brain of the product. Every resolved factor comes with full derivation.

### F3.0 — Resolution engine core

| File | Purpose |
|---|---|
| `greenlang/factors/resolution/engine.py` | `ResolutionEngine` class with `resolve(request) → ResolvedFactor`; implements 7-step cascade (customer override → supplier → facility → utility/grid → country/sector → method-pack fallback → global default) |
| `greenlang/factors/resolution/request.py` | `ResolutionRequest` Pydantic model with required `method_profile` (enforces non-negotiable #6) |
| `greenlang/factors/resolution/result.py` | `ResolvedFactor` carrying chosen factor + alternates considered + tie-break reasons + assumptions + deprecation status + gas breakdown + CO2e basis + uncertainty band |
| `greenlang/factors/resolution/tiebreak.py` | Scoring: geography match, time match, technology match, unit compatibility, methodology compatibility, source authority, verification status, uncertainty, recency, license availability |

### F3.1 — Wire tenant overlay (step 1)

`tenant_overlay.py` already exists but is disconnected. Phase F3.1 plugs it into step 1 of the cascade so `ResolutionEngine.resolve()` first checks the per-tenant vault for an active override by `(factor_id, activity_schema)`.

### F3.2 — Explain endpoint

| Surface | Artefact |
|---|---|
| REST | `GET /api/v1/factors/{factor_id}/explain?method_profile=...&activity=...` — returns the full `ResolvedFactor` payload |
| SDK | `FactorsClient.explain(factor_id, method_profile, activity)` — Python + TypeScript |
| CLI | `gl factors resolve --profile corporate_scope2_location_based --activity "..." --explain` |

### F3.3 — Tests

- `tests/factors/resolution/test_engine.py` — 7-step cascade, each step; customer override > supplier > facility > utility > country > method-pack > global default.
- `tests/factors/resolution/test_explain.py` — verifies alternates list, tie-break reasons, assumptions, deprecation status, gas breakdown, license-class homogeneity.
- `tests/factors/resolution/test_method_profile_enforcement.py` — non-negotiable #6: calling `resolve()` without `method_profile` raises.

Target: ≥ 30 tests.

---

## Phase F4 — Mapping Layer taxonomies *(P1, ~2 wks)*

**Goal:** make the factor layer usable when customers don't start with perfect factor-ready labels.

| Module | Scope |
|---|---|
| `greenlang/factors/mapping/fuels.py` | Canonical fuel taxonomy + synonym map (150+ entries) |
| `greenlang/factors/mapping/transport.py` | Vehicle class × mode × payload × distance taxonomy |
| `greenlang/factors/mapping/materials.py` | Steel, aluminium, cement, plastics, concrete, glass, paper, chemicals |
| `greenlang/factors/mapping/waste.py` | Landfill / incineration / composting / recycling / AD |
| `greenlang/factors/mapping/electricity_market.py` | Supplier category × certificate type × balancing area |
| `greenlang/factors/mapping/classifications.py` | NAICS ↔ ISIC ↔ HS/CN ↔ GICS cross-map |
| `greenlang/factors/mapping/spend.py` | Spend category → factor family resolver |
| `routes/factors.py` | `POST /api/v1/factors/mapping/resolve` endpoint |

Target: 7 modules, ≥ 40 tests.

---

## Phase F5 — Unit / Chemistry Engine hardening *(P1, ~1.5 wks)*

| Module | Purpose |
|---|---|
| `greenlang/data/density_converter.py` | Mass ↔ volume via temperature/pressure-aware density tables (25+ fuels + 20+ materials) |
| `greenlang/data/oxidation.py` | IPCC oxidation factor logic (coal 0.98, oil 0.99, gas 0.995 — tier-specific) |
| `greenlang/data/moisture.py` | Wet/dry basis conversion for biomass + coal |
| `greenlang/data/biogenic_split.py` | Fossil/biogenic share calculator (B-class blends, waste fractions) |
| `greenlang/factors/ontology/unit_graph.py` | Directed conversion graph replacing flat lookups |

Target: 5 modules, ≥ 50 tests (chemistry is error-prone).

---

## Phase F6 — Provenance & Governance hardening *(P1, ~1.5 wks)*

| Capability | File |
|---|---|
| Per-factor version chain | `greenlang/factors/quality/versioning.py` — immutable linked list (`previous_version` ref + chain hash) |
| Immutability guard | Postgres trigger in `V442__factors_version_immutability.sql` + app-level assertion |
| Impact simulator | `greenlang/factors/quality/impact_simulator.py` — "what breaks if we replace pack X?" returns affected computations + customers |
| Per-factor rollback | `greenlang/factors/quality/rollback.py` — restore a specific factor row at a prior version |
| Migration notes + deprecation messages | Schema fields + surfaced in `/explain` |
| Customer webhook registration | `POST /api/v1/factors/webhooks` + DB-backed subscription store |
| Signed result receipts | `greenlang/factors/signing.py` — SHA-256 + optional Ed25519 over normalized response |

Target: ≥ 25 tests.

---

## Phase F7 — Operator UI build-out *(P2, ~3 wks)*

All under `frontend/src/pages/`:

| Page | Backing endpoint |
|---|---|
| `FactorsSourceConsole.tsx` | `/api/v1/factors/sources` (list + drill-in) |
| `FactorsMappingWorkbench.tsx` | F4 mapping resolve endpoint |
| `FactorsQADashboard.tsx` | `/factors/quality/review_queue` |
| `FactorsDiffViewer.tsx` | `/factors/{id}/diff?left=v1&right=v2` |
| `FactorsApprovalQueue.tsx` | approval workflow API |
| `FactorsOverrideManager.tsx` | tenant_overlay CRUD |
| `FactorsImpactSimulator.tsx` | F6 simulator endpoint |

Target: 7 pages + nav entries + 15 Vitest + 7 Playwright E2E.

---

## Phase F8 — Premium Pack SKUs + Consulting tier *(P2, ~1.5 wks)*

| Item | File |
|---|---|
| `Tier.CONSULTING` enum + limits (multi-tenant, white-label) | `tier_enforcement.py` |
| Pack-level entitlement table | new migration V443 + `greenlang/factors/entitlements.py` |
| Per-pack license SKU enforcement at API layer | middleware extension |
| OEM redistribution rights flag | schema field + enforcement |

Target: ≥ 18 tests.

---

## Phase F9 — Remaining source parsers + packs *(P2, ~2 wks)*

| Parser | Feeds |
|---|---|
| `pact_product_data.py` | Product Carbon Pack |
| `ec3_epd.py` | Product Carbon Pack (building materials) |
| `freight_lanes.py` | Freight Pack (mode × lane × payload table) |
| `pcaf_proxies.py` | Finance Proxy Pack |
| `lsr_removals.py` | Land-Removals Pack |
| `waste_treatment.py` | Corporate Pack (Cat 5) |

Target: 6 parsers, ≥ 30 tests.

---

## Phase F10 — Polish & 100 % coverage *(P3, rolling)*

- `factor_family` populated for 100 % of existing factor YAML rows (migration script).
- Raw vs normalized record distinction materialized (`raw_record_ref` populated on every certified factor).
- Publisher + publication_date + validity_period on every source entry.
- GICS + BICS mappings added.
- Deprecation cleanup: any `factor_status = "deprecated"` without a `replacement_factor_id` → fail CI.

---

## Exit bar — product is shippable when

- [ ] All 28 canonical-record fields populated for at least 1 CBAM factor + 1 Scope 2 electricity factor + 1 freight factor (Phase F2 completion gate).
- [ ] All 7 method packs callable via `MethodProfile` enum (Phase F2).
- [ ] `POST /api/v1/factors/resolve` + `GET /api/v1/factors/{id}/explain` return the full 7-step derivation for any of the method profiles (Phase F3).
- [ ] CTO non-negotiables 1–7 all ✅ in code + tests (Phase F6).
- [ ] 5 operator pages pass Playwright smoke + accessibility audit (Phase F7).
- [ ] Premium Pack SKU gating works end-to-end for at least one licensed pack (Phase F8).
- [ ] Gold-eval pass rate ≥ 90 % on resolved factors (Phase F3+F4 combined).
- [ ] Hosted API deploys to staging via `helm install` with the extended schema (tracked in `docs/factors/hosted_api.md`).

---

## Execution strategy for THIS session

Phase F2 (method packs) is the **commercial lever** per the CTO — "makes the product commercial rather than academic". Phase F3 (resolution + explain) is the **proof lever** — non-negotiable #3 lands here.

I'll execute F2 (registry + base + Corporate + Electricity + EU Policy packs + India CEA parser + AIB residual-mix parser) and the core of F3 (ResolutionEngine + 7-step cascade + explain endpoint + SDK + CLI) in this session. F4–F10 will be follow-ups.

Budget: high-leverage commits only. No scope creep. Tests gating every module.
