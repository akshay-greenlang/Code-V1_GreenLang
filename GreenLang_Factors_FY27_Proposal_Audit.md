# GreenLang Factors FY27 — Proposal Audit & Gap-Close To-Do List

**Audit date:** 23 April 2026
**Audited document:** `GreenLang_Factors_FY27_Product_Development_Proposal.pdf` (21 pages, dated 16 April 2026)
**Audited against:** CTO brief on what GreenLang Factors must be (the five jobs, seven method packs, nine components, canonical factor record, licensing tiers, commercial packaging, and seven non-negotiables)

---

## 1. Bottom-line verdict

**Completion estimate: ~70–75%.**

The proposal captures the correct *philosophy* — operating layer, open-core, four-layer architecture, phased build, coverage labels, policy separated from factors. A CTO would nod at the posture.

But seven specific, **shippable** pillars the brief names explicitly are either missing or mentioned in one line without being specified. An engineer cannot build to this document and honor your non-negotiables.

The proposal is **brief-shaped**, not **spec-shaped**. It needs a spec appendix.

---

## 2. What the proposal gets right (keep as-is)

| Area | Status | Notes |
|---|---|---|
| Positioning as "operating layer, not database" | ✅ Strong | Section 1, Section 3, Section 18 |
| Three-target framing (cataloged / QA-certified / production-usable) | ✅ Strong | Section 1 |
| Four-layer architecture (core / connector / provenance / matching) | ✅ Correct | Section 4 |
| Coverage labels: Certified / Preview / Connector-only | ✅ Excellent | Section 5 |
| Release cadence separation (not weekly for everything) | ✅ Correct | Section 7 |
| 10 backend services listed | ✅ Directionally correct | Section 8 |
| Hybrid AI (deterministic serving, AI in curation only) | ✅ Correct | Section 9 |
| 52-week phased plan (Phases 0–10) | ✅ Solid | Section 10 |
| 30/60/90-day plan | ✅ Aligns with brief | Section 15 |
| KPIs, risks, team | ✅ Adequate | Sections 12–14 |
| Launch scope discipline (narrow GA, defer LCA/MRV) | ✅ Matches the brief | Section 17 |

Keep all of the above. The critique below is about what is **missing from** the document, not about reshaping what's there.

---

## 3. Seven critical gaps (ranked)

### GAP 1 — Method Pack Library is missing as a first-class product concept ⚠️ SEVERE

The brief explicitly names **seven** method packs that must ship inside Factors:

1. Corporate Inventory Pack (GHG Protocol Corporate + Scope 2 + Scope 3)
2. Product Carbon Pack (GHG Protocol Product + ISO 14067 + PACT)
3. Freight Pack (ISO 14083 + GLEC)
4. Electricity Pack (location-based / market-based / supplier-specific / residual-mix)
5. Land & Removals Pack (GHG Protocol LSR)
6. Finance Proxy Pack (PCAF)
7. EU Policy Pack (CBAM selectors + DPP structures)

The proposal mentions a "Policy and Method Store" service (one bullet in Section 8) and a "Policy/method" layer in the four-layer model, but it **never names the seven packs, never lists their contents, and never treats a method pack as a versioned, shippable artifact**. Without this, the product cannot answer "resolve factor for 12,500 kWh India FY27 corporate inventory location-based" differently from "resolve factor for 1 tonne hot-rolled steel CBAM profile EU import." Same activity, different method — different correct answer. That's the whole product.

**What a method pack must contain (per the brief):**
factor selection rules, boundary rules, inclusion/exclusion rules, gas-to-CO2e basis, biogenic carbon treatment, market-instrument treatment, region hierarchy, fallback logic, reporting labels, audit text templates, deprecation policy.

### GAP 2 — Resolution Engine is underspecified ⚠️ SEVERE

The brief specifies a **precise selection order** and **tie-breaking logic**. The proposal has "Search and Matching Service" (Section 8) and "hybrid retrieval" (Section 9) but doesn't codify:

- Selection order: customer override → supplier-specific → facility/asset → utility/tariff/grid-subregion → country/sector → method-pack fallback → global default
- Tie-breakers: geography, time, technology, unit compatibility, methodology compatibility, source authority, verification, uncertainty, recency, license availability
- Required output shape: chosen factor + alternates considered + why this won + source/version + quality score + uncertainty band + gas breakdown + CO2e basis + assumptions + deprecation status

The current proposal treats this as "search plus reranking." It is not a search product — it is a **deterministic resolver with explanation**. That distinction is the moat.

### GAP 3 — Unit & Chemistry Engine is essentially missing ⚠️ SEVERE

One bullet in Phase 2 ("unit ontology and conversion engine; include energy, mass, distance, volume, currency, and passenger/freight units"). That's it.

The brief is explicit and non-negotiable:

- Numerator/denominator unit graph
- SI + commercial unit conversion
- LHV vs HHV handling
- Density / moisture / oxidation adjustments
- Fossil vs biogenic carbon split
- **Gas-level storage (CO2, CH4, N2O, HFC, PFC, SF6) — CO2e is derived, not stored**
- Multiple GWP-set support (AR4, AR5, AR6, AR6+)

The proposal's canonical schema lists `gas_breakdown, co2e_total, gwp_basis, gwp_source` in a single bullet. That is not a gas-level model. If you ship this as-is, you will violate the non-negotiable "never store only CO2e."

### GAP 4 — Canonical Factor Record is a field list, not a schema ⚠️ HIGH

Section 8 lists ~9 bullets of fields. The brief includes a **full JSON example with ~50 fields, structured blocks, and a parameters section per category**. Missing from the proposal:

- Structured `jurisdiction{}`, `numerator{co2,ch4,n2o,co2e,unit}`, `denominator{value,unit}`, `parameters{}`, `quality{temporal,geographic,technology,verification,completeness}`, `lineage{ingested_at, ingested_by, approved_by, change_reason}`, `licensing{redistribution_class, customer_entitlement_required}`, `explainability{assumptions, fallback_rank}`
- `formula_type` (direct_factor, stoichiometric, composite)
- `scope_applicability` array
- `electricity_basis`, `residual_mix_applicable`, `supplier_specific`, `transmission_loss_included`, `biogenic_share`
- Separate `factor_version` vs `source_version` (currently conflated)

### GAP 5 — Category-specific parameter groups are missing ⚠️ HIGH

The brief lists parameter groups for **seven** categories: Combustion, Electricity, Transport, Materials/Products, Refrigerants, Land & Removals, Finance Proxies. None of these are enumerated in the proposal. Without them, parser specs (Phase 1) and canonical schema (Phase 2) can't be written correctly.

Example — the proposal never mentions:
- Combustion: fuel code, LHV/HHV, density, oxidation factor, fossil/biogenic carbon share, sulfur/moisture/ash
- Transport: mode, vehicle class, payload basis, distance basis, empty-running, utilization, refrigerated flag, WTW/TTW/tank-to-wheel labels
- Refrigerants: gas code, leakage basis, recharge, recovery/destruction treatment, GWP-set mapping

### GAP 6 — Commercial packaging is hand-waved ⚠️ HIGH

Section 10 Phase 10 says "create pricing tiers and contract terms." That's one bullet. The brief names **five explicit tiers** and a **price shape**:

| Tier | Price model | Included |
|---|---|---|
| Community / Open-Core | Free | Schema, SDK/CLI, limited public pack, sandbox API, docs |
| Developer Pro | Usage-based | Production API, rate limits, batch, version pinning, basic support |
| Consulting / Platform | Annual + usage | Multi-client workspaces, white-label, client override vaults, audit exports, premium packs, partner support |
| Enterprise | High-ACV | SSO/SCIM, VPC/private deploy, private registry, approval workflows, customer-specific factors, signed releases, SLA |
| Premium Data Packs | Separate SKUs | Electricity Premium, Freight Premium, PCF/LCI Premium, EPD/Construction Premium, Agrifood/Land Premium, Finance Proxy Premium, CBAM/EU Policy Premium |

And the brief is explicit on price shape: **do not price by number of factors.** Price by API calls, batch volume, pack entitlements, private-registry usage, tenants, OEM rights, SLA level. The proposal does not say any of this.

### GAP 7 — Licensing tiers are not treated as first-class product classes ⚠️ HIGH

Schema mentions `license_class, redistribution_rights`. That is field-level. The brief requires a **product-level separation** of four classes:

1. Open public factors
2. Licensed embedded factors
3. Customer-private factors
4. OEM redistribution rights

These need separate storage paths, separate API behavior, separate release bundles, and separate contract language. The proposal conflates this into "Connector layer" (Section 4), which is closer but still not enough.

---

## 4. Medium gaps

### Source pack segmentation is partial
Sources are listed (Section 6) but not organized into the four pack families the brief names: **Public/default**, **Product/LCI premium**, **Freight**, **Finance & Land**. Also missing by name: IPCC EFDB, India CEA CO2 baseline database, AIB residual mix, EC3 (EPD), PCAF.

### CBAM and DPP as specific features
The proposal flags "CBAM definitive period" as a market signal (Section 2) but does not commit to shipping CBAM selector logic or DPP-ready product data structures in FY27. The brief says these are required.

### Provenance & Governance is mentioned but not designed
The brief requires immutable versioning with **who changed it, why, what changed, affected calculations, impacted customers/inventories, rollback option, migration notes** per change. The proposal has "Release Manager" (one service bullet) and `lineage path` (one field). That is not a governance layer.

### Operator/admin surfaces are thin
Listed in Section 8 under "Admin Console." Missing from the brief's explicit list:
- Mapping workbench (not the same as Parser Service)
- Diff viewer (factor-level diffs, not just source document diffs)
- Customer override manager
- **Impact simulator** — "what breaks if we replace UK-2025 road freight factor pack?" This is called out explicitly in the brief and completely absent from the proposal

### Developer surfaces are incomplete
Listed: REST API, docs, version pinning, usage logging. Missing from the brief:
- SDKs (plural — not just one)
- CLI
- Webhooks on factor changes
- Signed result receipts
- Dedicated `/explain` endpoint (implied but not listed)
- GraphQL (brief mentions REST/GraphQL; proposal mentions REST only)

### Quality / uncertainty / review engine is one service bullet
The brief specifies a **composite Factor Quality Score (0–100) plus component scores** and fields for review owner, approval state, next review date, uncertainty type/range, primary vs secondary data flag. The proposal has "Quality Engine" as one of 10 services with no spec.

### Multi-GWP-set support is missing
The brief: "Support for multiple GWP sets matters because IPCC AR6 updated emissions metric values, while many reporting systems still operate with older bases." The proposal mentions `gwp_basis` as a single field, no multi-set derivation.

---

## 5. Minor gaps / polish

- **Seven non-negotiables** aren't codified as a principles page. The proposal has one "Critical product principle" box (Section 4). The brief has seven. Put them on a page of their own.
- **Signed releases / signed responses** (for enterprise tier) not mentioned.
- **Sub-tenant entitlements** for OEM embed not mentioned.
- **White-label branding** not mentioned.
- **Consultant client-specific factor vault** referenced only vaguely.
- **Methodology review workflow** mentioned in Phase 5 but not specified (who reviews, what triggers review, SLA on review).
- **Annual/versioned audit bundle** as a named deliverable not explicit.
- **Evaluation harness / gold-labeled set** exists (Phase 6, 300–500 cases) but the brief implies a richer harness across multiple method profiles.

---

## 6. Gap-close to-do list (buildable, with IDs)

Add these as a new appendix to the PDF, or as a companion spec document. IDs continue your existing workstream ID scheme (P, G, D, Q, M, A, U, C).

### New workstream: Method Packs (MP)
- **MP1** — Write Method Pack specification template (factor selection rules, boundary, inclusion/exclusion, gas→CO2e basis, biogenic, market instruments, region hierarchy, fallback, labels, audit text, deprecation policy)
- **MP2** — Ship v0.1 Corporate Inventory Pack (GHG Protocol Corporate + Scope 2 + Scope 3)
- **MP3** — Ship v0.1 Electricity Pack (location / market / supplier-specific / residual-mix)
- **MP4** — Ship v0.1 Freight Pack (ISO 14083 + GLEC profiles)
- **MP5** — Ship v0.1 EU Policy Pack (CBAM selectors + DPP-ready product structures)
- **MP6** — Ship v0.2 Product Carbon Pack (GHG Protocol Product + ISO 14067 + PACT) — FY27 late if capacity
- **MP7** — Ship v0.2 Land & Removals Pack (GHG Protocol LSR) — FY27 late or FY28
- **MP8** — Ship v0.2 Finance Proxy Pack (PCAF) — FY27 late or FY28
- **MP9** — Method-pack registry API (resolve factor *under* a method profile, not raw)
- **MP10** — Method-pack versioning & deprecation rules

### New workstream: Resolution Engine (R)
- **R1** — Write formal Resolver specification (selection order + tie-breaker rules)
- **R2** — Build Resolver service separate from Search
- **R3** — Resolver output contract: chosen + alternates + rationale + source/version + quality + uncertainty + gas breakdown + CO2e basis + assumptions + deprecation status
- **R4** — `/resolve` and `/explain` API endpoints with version-pinning
- **R5** — Resolver evaluation harness (per method profile, per category)
- **R6** — Customer override hook in resolver chain
- **R7** — License-aware resolver (won't return a factor the caller isn't entitled to)

### New workstream: Unit & Chemistry Engine (UC)
- **UC1** — Numerator/denominator unit graph
- **UC2** — SI + commercial unit conversion library with tests
- **UC3** — LHV/HHV handling + density/moisture/oxidation adjustments
- **UC4** — Fossil vs biogenic carbon split
- **UC5** — **Gas-level storage model (CO2, CH4, N2O, HFC, PFC, SF6) — CO2e is never stored, only derived**
- **UC6** — Multi-GWP-set support (AR4, AR5, AR6, AR6+) with derivation on read
- **UC7** — Chemistry validator (stoichiometric sanity checks)

### Expand Canonical Schema (D7–D12 new)
- **D7** — Ship full JSON schema matching brief's example (structured blocks: `jurisdiction`, `numerator`, `denominator`, `parameters`, `quality`, `lineage`, `licensing`, `explainability`)
- **D8** — Add `formula_type`, `scope_applicability`, `electricity_basis`, `residual_mix_applicable`, `supplier_specific`, `transmission_loss_included`, `biogenic_share`
- **D9** — Separate `factor_version` from `source_version` semantics
- **D10** — Category-specific parameter schemas (combustion, electricity, transport, materials, refrigerants, land, finance)
- **D11** — Quality sub-scores (temporal, geographic, technology, verification, completeness) + composite 0–100
- **D12** — Source object schema (authority, title, publisher, jurisdiction, dataset version, pub date, validity period, ingest date, source type, redistribution class, verification status, citation, change log, legal notes)

### New workstream: Commercial Packaging (C7–C12 new)
- **C7** — Define and publish the five-tier model (Community, Developer Pro, Consulting/Platform, Enterprise, Premium Data Packs)
- **C8** — Define price shape: API calls, batch volume, pack entitlements, private-registry usage, tenants, OEM rights, SLA. **Do not price by factor count.**
- **C9** — Define the seven Premium Data Pack SKUs (Electricity, Freight, PCF/LCI, Construction/EPD, Agrifood/Land, Finance Proxy, CBAM/EU Policy)
- **C10** — OEM / white-label contract terms
- **C11** — Enterprise features: SSO/SCIM, VPC/private deploy, private factor registry, approval workflows, customer-specific factors, signed releases, SLA
- **C12** — Consultant features: client-specific factor vault, override workflows, audit bundle export, factor comparison, methodology notes export

### New workstream: Licensing & Data-Class Separation (L)
- **L1** — Formalize four data classes as first-class product concept: Open / Licensed-Embedded / Customer-Private / OEM-Redistributable
- **L2** — Separate storage paths per class (physical segregation, not just a flag)
- **L3** — Entitlement service: which caller can see / resolve / export which class
- **L4** — Contract templates per class (legal workstream)
- **L5** — Release bundles segregated by class (a public release cannot accidentally include licensed data)

### Expand Source Catalog (G7–G10 new)
- **G7** — Organize sources into four pack families (Public/default, Product/LCI premium, Freight, Finance & Land)
- **G8** — Add IPCC EFDB, India CEA CO2 baseline DB, AIB residual mix to the default pack
- **G9** — Add EC3 (EPD) and ecoinvent connector to LCI premium pack
- **G10** — Add PCAF to Finance pack

### Expand Admin / Operator Surfaces (new O workstream)
- **O1** — Mapping workbench (distinct from Parser Service)
- **O2** — Factor-level diff viewer (not just source document diff)
- **O3** — Customer override manager UI
- **O4** — **Impact simulator** ("what breaks if we replace UK-2025 road freight factor pack?")
- **O5** — Methodology review workflow UI with SLAs and owner assignment
- **O6** — Annual audit bundle exporter

### Expand Developer Surfaces (A7–A12 new)
- **A7** — SDKs (Python, Node/TS at minimum; Java nice-to-have)
- **A8** — CLI for factor search / resolve / pin / export
- **A9** — Webhooks on factor / method-pack / source changes
- **A10** — Signed result receipts (enterprise tier)
- **A11** — Dedicated `/explain` endpoint
- **A12** — GraphQL endpoint for flexible querying (optional FY27; committed FY28)

### Codify Non-Negotiables (new N workstream — put on one page in the PDF)
- **N1** — Never store only CO2e. Store gas components; derive CO2e by selected GWP set.
- **N2** — Never overwrite a factor. Version everything.
- **N3** — Never hide fallback logic. Every resolution returns the rationale.
- **N4** — Never mix licensing classes. Physical segregation, not just flags.
- **N5** — Never ship a factor without validity dates and source version.
- **N6** — Policy workflows never call raw factors. They call a method profile.
- **N7** — Keep open-core boundaries clear. Community features never depend on Enterprise-only infra.

### Add Governance Layer (new GOV workstream)
- **GOV1** — Immutable change log per factor (who / why / what / affected calcs / impacted customers / rollback / migration notes)
- **GOV2** — Approval workflow with segregation of duties (ingester ≠ approver ≠ releaser)
- **GOV3** — Deprecation notices with N-day advance warning to API consumers
- **GOV4** — Reproducibility manifests per release tag (re-run an inventory filed on 2026-11-15 and get the same numbers)
- **GOV5** — Factor Foundation steward model (stub; align with the "GreenLang Foundation" concept in the business plan)

---

## 7. Suggested structure for a revised proposal

I would insert **three new sections** into the PDF between the current Section 8 (Product architecture) and Section 9 (AI/model strategy):

- **Section 8.1 — Method Pack Library** (the 7 packs, contents of each, launch sequence)
- **Section 8.2 — Resolution Engine specification** (selection order, tie-breakers, output contract)
- **Section 8.3 — Unit & Chemistry Engine** (unit graph, gas-level storage, multi-GWP)

And **one new section** after the current Section 16 (Detailed workstream checklist):

- **Section 16.1 — New workstreams**: MP, R, UC, L, O, N, GOV (the to-do list above)

Replace the one "Critical product principle" callout in Section 4 with a full-page **Section 0 — Non-Negotiables** listing the seven N-rules.

Expand Section 6 (sources) into a table by **pack family** rather than a flat list.

Rewrite Phase 10 (Weeks 39–52) to name the five commercial tiers and the seven Premium Data Pack SKUs explicitly, instead of "create pricing tiers."

---

## 8. What "100%" would look like

A future version of this proposal is 100% when an engineer who has never met you can read the PDF alone and:

1. Build a parser for a new source without asking which fields are required
2. Implement the resolver and pass the gold-labeled eval set on the first try
3. Ship a factor that stores CO2, CH4, N2O separately and derives CO2e under AR5 or AR6 on read
4. Respond to an enterprise RFP with a quote that matches your actual pricing model
5. Explain to a legal reviewer which of four licensing classes a given factor belongs to, and prove the storage is segregated
6. Show a customer an "impact simulator" preview before a pack upgrade
7. Demonstrate that the seven method packs return different (correct) answers for the same activity

Today's PDF gets you to ~2 of those 7. Closing the to-do list above gets you to 7 of 7.

---

## 9. Priority order for closing gaps

If you have to sequence the fix work inside Q1 FY27:

**Week 1–2 (blocking):** N1–N7 non-negotiables, D7–D12 canonical schema v0.2, MP1 method-pack template
**Week 3–4:** MP2 & MP3 (Corporate + Electricity packs — you cannot ship a GA without these), UC1–UC6 unit & chemistry engine
**Week 5–6:** R1–R4 resolver spec + resolver service, L1–L3 licensing segregation
**Week 7–8:** C7–C9 commercial packaging, MP4 & MP5 (Freight + EU Policy)
**Week 9–12:** O1–O6 operator surfaces, A7–A12 developer surfaces, GOV1–GOV4 governance
**FY27 H2:** MP6, MP7, MP8 (PCF, Land, Finance packs), A12 GraphQL, GOV5 Foundation

This preserves your existing Section 10 phasing and just *sharpens* the work inside each phase.

---

*End of audit. If you want, I can convert this to-do list directly into ClickUp/Jira/Linear tickets, or rewrite the PDF's Sections 8, 10, 16, and 17 to fold these additions in cleanly.*
