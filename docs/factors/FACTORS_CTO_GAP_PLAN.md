# GreenLang Factors — CTO Gap Plan

**Source:** CTO product outlook (2026-04-20) + 4 parallel codebase audits.
**Cumulative coverage today:** ~46 % of canonical-record fields, 6/15 factor families, 0/7 method packs, 0/7 steps of the resolution engine, 5/6 non-negotiables partially honoured.

---

## 0. Scoring summary (from audit)

| Area | Present | Partial | Missing |
|---|---|---|---|
| Canonical factor record (28 fields) | 13 | 6 | 9 |
| Source registry (14 fields) | 4 | 1 | 9 |
| Factor families (15) | 6 | 5 | 4 |
| Source authority parsers (11) | 5 | 2 | 4 |
| Method Pack Library (7) | 0 | 2 (Electricity, EU-Policy partial) | 5 |
| Resolution Engine (7-step order) | 0 | 3 (some steps faint) | 4 |
| Explain endpoint | 0 | 0 | 1 |
| Mapping Layer taxonomies (8) | 0 | 3 | 5 |
| Unit/Chemistry (10) | 4 | 4 | 2 |
| Quality/Uncertainty/Review (12) | 6 | 1 | 5 |
| Provenance/Governance (12) | 3 | 3 | 6 |
| Developer surface (10) | 7 | 1 | 2 |
| Operator UI (8) | 1 | 1 | 6 |
| Packaging tiers (5) | 3 | 0 | 2 |

**Non-negotiables:**

| # | Non-negotiable | Status |
|---|---|---|
| 1 | Never store only CO2e — gas vectors first | ✅ Done (`GHGVectors`) |
| 2 | Never overwrite a factor — version everything | ⚠️ Partial (edition-level, not per-factor) |
| 3 | Never hide fallback logic — always explain | ❌ Missing (no explain endpoint) |
| 4 | Never mix licensing classes | ⚠️ Partial (per-entry, not response-level) |
| 5 | Never ship without validity + source version | ⚠️ Partial (fields exist; enforcement unclear) |
| 6 | Never let policy workflows call raw factors | ⚠️ Partial (no method-profile guard) |
| 7 | Never blur open-core into enterprise | ✅ Done (tier enforcement) |

---

## Phased to-do list (F1 → F10)

The plan is deliberately ordered so that the **non-negotiables and FY27 v1 launch scope** (electricity IN/EU/UK/US, fuel combustion, refrigerants, road/sea/air freight, purchased-goods proxies, GHG Protocol corporate + Scope 2 + Scope 3, CBAM) land first. Sectors that FY27 doesn't sell (agrifood, finance proxies, LCI premium) are deferred.

### **PHASE F1 — Canonical Record + Non-Negotiables** *(P0, ~1 week)*

Backward-compatible schema extensions to `EmissionFactorRecord` + enforcement.

| Item | File |
|---|---|
| Add `factor_family` enum (15 families) | `greenlang/data/emission_factor_record.py` |
| Add `factor_name` (display string) | same |
| Add `method_profile` field (FK to method pack) | same |
| Add `factor_version` separate from `source_version` | same |
| Add `formula_type` enum (direct_factor / combustion / lca / spend_proxy / transport_chain / ...) | same |
| Add structured `jurisdiction{country, region, grid_region}` object | same |
| Add `activity_schema{category, sub_category, classification_codes}` | same |
| Add `explainability{assumptions, fallback_rank}` | same |
| Add `parameters{electricity_basis, residual_mix_applicable, supplier_specific, transmission_loss_included}` | same |
| Add `primary_data_flag` (primary vs secondary) | same |
| Add `verification{status, verified_by, verified_at}` | same |
| Add `uncertainty_distribution` enum (log_normal / triangular / uniform / normal / unknown) | same |
| Add `change_reason`, `changed_by`, `next_review_date` | same |
| Add `raw_record_ref` field (link to pre-normalized source record) | same |
| Source registry: add `publisher`, `publication_date`, `dataset_version`, `validity_period`, `source_type`, `verification_status`, `change_log`, `legal_notes` | `greenlang/factors/source_registry.py`, `source_registry.yaml` |
| Enforce license-class non-mixing at response level | `greenlang/factors/tier_enforcement.py` |

### **PHASE F2 — Method Pack Library** *(P0, ~2 weeks)*

Make the product commercial, not academic.

| Item | File |
|---|---|
| F2.1 Method Pack Registry (`greenlang/factors/method_packs/`) | new |
| F2.2 Corporate Inventory Pack (GHG Protocol Scope 1/2/3) | new |
| F2.3 Electricity Pack (location / market / supplier-specific / residual-mix) | new |
| F2.4 EU Policy Pack (CBAM selectors + DPP structures) | new |
| F2.5 Product Carbon Pack stub (ISO 14067 + PACT shape) | new |
| F2.6 Freight Pack stub (ISO 14083 + GLEC-aligned) | new |
| F2.7 Land/Removals + PCAF Pack stubs (FY28-ready shells) | new |
| Each pack: factor-selection rules, boundary rules, gas-to-CO2e basis, biogenic treatment, market-instrument treatment, region hierarchy, deprecation policy | per-pack YAML |
| India CEA parser (`greenlang/factors/ingestion/parsers/india_cea.py`) | new |
| AIB residual mix parser (`.../parsers/aib_residual_mix.py`) | new |

### **PHASE F3 — Resolution Engine + Explain** *(P0, ~2 weeks)*

The brain of the product + non-negotiable #3.

| Item | File |
|---|---|
| F3.1 `ResolutionEngine.resolve()` implementing the 7-step selection order | `greenlang/factors/resolution/engine.py` (new) |
| F3.2 Wire `tenant_overlay.py` into step 1 (customer override) | integrate with engine |
| F3.3 Supplier-specific + facility-specific resolution keys | new |
| F3.4 Utility/tariff + grid-subregion keys (eGRID extension) | new |
| F3.5 Method-pack fallback (calls F2 packs) | integrate |
| F3.6 Return alternates + tie-break reasons + assumptions + deprecation status | new |
| F3.7 `/api/v1/factors/{id}/explain` endpoint + SDK method | `routes/factors.py`, `sdk/__init__.py` |
| F3.8 `gl factors resolve` CLI command | `cli/cmd_factors.py` (or extend existing) |

### **PHASE F4 — Mapping Layer** *(P1, ~2 weeks)*

Make the factor layer addressable without perfect labels.

| Item | File |
|---|---|
| Fuel type taxonomy | `greenlang/factors/mapping/fuels.py` |
| Transport mode taxonomy | `.../mapping/transport.py` |
| Material / product taxonomy | `.../mapping/materials.py` |
| Waste route taxonomy | `.../mapping/waste.py` |
| Electricity market taxonomy | `.../mapping/electricity_market.py` |
| NAICS / ISIC / HS / CN code mappers | `.../mapping/classifications.py` |
| Spend category mapper | `.../mapping/spend.py` |
| Mapping workbench REST endpoints | `routes/factors.py` |

### **PHASE F5 — Unit / Chemistry Engine hardening** *(P1, ~1.5 weeks)*

| Item | File |
|---|---|
| Density lookup table + mass ↔ volume converter | `greenlang/data/density_converter.py` |
| Oxidation adjustment engine (IPCC oxidation factor logic) | `.../oxidation.py` |
| Moisture adjustment engine (wet vs dry basis) | `.../moisture.py` |
| Fossil / biogenic split calculator | `.../biogenic_split.py` |
| Numerator/denominator unit graph (upgrade from flat tables) | refactor `factors/ontology/units.py` |
| Test matrix: all 5 converters × 10 fuels | new tests |

### **PHASE F6 — Provenance & Governance hardening** *(P1, ~1.5 weeks)*

| Item | File |
|---|---|
| Per-factor version chain (not just edition) | `greenlang/factors/quality/versioning.py` (new) |
| Immutable-on-change guard (DB trigger + app-level assertion) | new |
| Impact simulator — "what breaks if we replace pack X" | `.../impact_simulator.py` |
| Per-factor rollback | `.../rollback.py` |
| Migration notes + deprecation-message fields | schema + API |
| Customer-facing webhook registration endpoint | `routes/factors.py` |
| Signed result receipts (Ed25519 optional, SHA-256 baseline) | `.../signing.py` |

### **PHASE F7 — Operator UI build-out** *(P2, ~3 weeks)*

| Page | File |
|---|---|
| Source ingestion console | `frontend/src/pages/FactorsSourceConsole.tsx` |
| Mapping workbench | `frontend/src/pages/FactorsMappingWorkbench.tsx` |
| QA dashboard | `frontend/src/pages/FactorsQADashboard.tsx` |
| Diff viewer (factor-to-factor) | `frontend/src/pages/FactorsDiffViewer.tsx` |
| Approval workflow UI | `frontend/src/pages/FactorsApprovalQueue.tsx` |
| Customer override manager | `frontend/src/pages/FactorsOverrideManager.tsx` |
| Impact simulator UI | `frontend/src/pages/FactorsImpactSimulator.tsx` |

### **PHASE F8 — Premium Pack SKUs + Consulting tier** *(P2, ~1.5 weeks)*

| Item | File |
|---|---|
| Add `Tier.CONSULTING` (white-label, multi-tenant) | `tier_enforcement.py` |
| Premium Pack entitlements table + enforcement | new |
| Per-pack license SKU enforcement at API layer | `routes/factors.py` |
| Signed release receipts (cryptographic) | `signing.py` |
| OEM redistribution rights flag | schema + enforcement |

### **PHASE F9 — Remaining source parsers + packs** *(P2, ~2 weeks)*

| Parser | Pack |
|---|---|
| PACT product-data exchange | Product Carbon Pack |
| EC3 / EPD embodied carbon | Product Carbon Pack |
| Freight lane factors | Freight Pack |
| PCAF proxy factors | Finance Pack |
| LSR land-use / removals | Land Pack |
| Waste treatment factors | Corporate Pack |

### **PHASE F10 — Polish & full coverage** *(P3, rolling)*

- Factor family populated for 100 % of existing records.
- Raw vs normalized record distinction.
- Source type enum + publisher + publication_date on all entries.
- Classification mapping expansion (GICS, BICS, industry codes).

---

## Immediate execution plan (this session)

I'm starting Phase F1 — canonical-record extensions + non-negotiable enforcement — because:

1. Every downstream phase depends on the schema being complete (method-pack → needs `method_profile`; resolution → needs `fallback_rank` + `assumptions`; explain → needs `explainability`).
2. The changes are backward-compatible additions to existing Pydantic / dataclass models.
3. Enforcement of "never ship without validity + source version" and "never mix licensing classes" unblocks customer-facing claims.

After Phase F1 lands, I'll wait for your green-light before Phase F2.
