# GreenLang Factors FY27 - CTO Master To-Do List

**Prepared for:** CTO review  
**Prepared on:** 23 Apr 2026  
**Workspace:** `C:\Users\aksha\Code-V1_GreenLang`  
**Purpose:** Convert the GreenLang Factors proposal audit into a buildable, launch-grade execution plan.

---

## 0. Executive Verdict

GreenLang Factors should be completed as:

**Factor Registry + Resolution Engine + Method Packs + Governance + API/OEM Platform.**

This document treats the current repo as already partially built. Several assets exist under `greenlang/factors/`, `tests/factors/`, `frontend/packages/factors-explorer/`, `frontend/packages/factors-ops-console/`, `.github/workflows/`, and `docs/api/`. The remaining work is therefore not only "write code"; it is also:

- freeze the product contract,
- prove every CTO non-negotiable with tests,
- harden API/runtime behavior,
- wire operator/developer surfaces,
- create commercial packaging,
- cut a signed v1 Certified edition,
- deploy a design-partner-ready hosted product.

**Definition of 100% complete:** a developer, consultant, platform partner, enterprise methodology lead, and legal reviewer can all use the product without private founder explanation.

---

## 1. Priority Model

| Priority | Meaning | Ship rule |
|---|---|---|
| P0 | Launch blocker | Cannot start external design-partner pilot without this |
| P1 | Beta/commercial blocker | Must complete before paid beta or enterprise RFP |
| P2 | FY27 scale item | Needed for full FY27 roadmap, not day-one narrow launch |
| P3 | FY27 H2 / FY28 item | Explicitly deferred unless design partner demands it |

## 2. Work State Tags

| Tag | Meaning |
|---|---|
| Build | Capability appears missing or incomplete |
| Harden | Capability exists but needs production behavior, tests, docs, or enforcement |
| Verify | Capability appears present; prove with tests, fixtures, demo, and release evidence |
| Commercial | Pricing, packaging, legal, support, or sales enablement |
| Ops | Deployment, monitoring, support, security, or SRE |

## 3. Suggested Owners

| Owner | Scope |
|---|---|
| Product Lead | Scope, personas, launch gates, prioritization |
| Methodology Lead | Standards, method packs, QA review, audit text |
| Data Platform Engineer | Source ingestion, schema, catalog, lineage, quality |
| Backend/API Engineer | Resolver, API, auth, entitlements, billing, signing |
| Frontend Engineer | Explorer, operator console, pricing/developer surfaces |
| DevRel / Technical Writer | Docs, SDK examples, quickstarts, changelog |
| SRE / Platform Engineer | Deployment, CI/CD, monitoring, backup, DR |
| Legal / Commercial | Licensing classes, data rights, contracts, OEM rights |
| QA Lead | Gold-set eval, launch coverage, regression gates |

---

## 4. CTO Non-Negotiables - Must Become Release Gates

These are not principles for a slide. They must be executable launch checks.

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| N1 | P0 | Verify | Prove GreenLang never stores only CO2e as the sole canonical value. Store gas components and derive CO2e by selected GWP set. | Data Platform + QA | Test fails if a certified factor lacks gas-level fields or only carries aggregate CO2e. |
| N2 | P0 | Verify | Prove no factor can be overwritten in place. All numeric, boundary, source, method, license, and quality changes create immutable versions. | Backend + QA | Mutation test attempts overwrite and receives error; new version creates changelog. |
| N3 | P0 | Harden | Make fallback logic visible in every resolver output. | Backend | `/resolve` and `/explain` always return chosen factor, alternates, fallback rank, and rationale. |
| N4 | P0 | Harden | Physically and logically separate Open, Licensed Embedded, Customer Private, and OEM Redistributable data classes. | Backend + Legal | Access tests prove each class has distinct storage/entitlement/export behavior. |
| N5 | P0 | Verify | Block release of any factor without source version, validity window, jurisdiction, unit basis, and status. | QA | Release signoff fails on missing fields. |
| N6 | P0 | Harden | Ensure policy workflows call method profiles, not raw factors. | Backend + Methodology | Guard test fails when CBAM/Comply/Scope Engine directly calls raw factor lookup. |
| N7 | P0 | Commercial | Keep open-core and enterprise boundaries explicit. | Product + Legal | Public docs and contracts define what is free, paid, licensed, private, and OEM. |

---

## 5. Product Contract and Scope Freeze

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| PC1 | P0 | Build | Write a one-page GreenLang Factors product charter. | Product | Approved charter names buyers, v1 scope, non-scope, revenue model, and launch metrics. |
| PC2 | P0 | Build | Freeze FY27 narrow launch scope: electricity, combustion fuels, refrigerants, freight, selected materials, corporate/Scope 2/Scope 3/CBAM profiles. | Product + Methodology | Scope table approved by CTO and methodology lead. |
| PC3 | P0 | Build | Define the minimum v1 user journeys for developers, consultants, platforms, and methodology operators. | Product | 12-20 testable journeys attached to roadmap. |
| PC4 | P0 | Build | Convert proposal gaps into an engineering backlog using this document's IDs. | Product | Jira/Linear/ClickUp tickets exist with owners, priorities, and acceptance criteria. |
| PC5 | P1 | Build | Define explicit non-scope for v1: full LCA suite, full MRV/removals platform, full finance portfolio analytics, unrestricted licensed-data redistribution. | Product | Non-scope appears in docs and sales collateral. |
| PC6 | P1 | Build | Create launch decision record: "Factors is a resolver system, not a factor CSV." | Product | Architecture Decision Record checked into `docs/adr/`. |
| PC7 | P1 | Build | Establish weekly launch review with CTO: P0 progress, gate status, defects, design-partner feedback. | Product | Weekly dashboard or written status template exists. |

---

## 6. Canonical Data Model and Factor Registry

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| FR1 | P0 | Verify | Confirm canonical factor schema contains structured blocks: `jurisdiction`, `activity_schema`, `numerator`, `denominator`, `parameters`, `quality`, `lineage`, `licensing`, `explainability`. | Data Platform | Schema test validates all required blocks for certified factors. |
| FR2 | P0 | Verify | Separate `factor_id`, `factor_family`, `factor_version`, `source_id`, `source_version`, `release_version`, and `method_profile`. | Data Platform | Schema migration and tests prove each field has independent semantics. |
| FR3 | P0 | Verify | Make `valid_from` and `valid_to` mandatory for certified factors. | Data Platform + QA | Release signoff fails if missing or invalid. |
| FR4 | P0 | Verify | Require gas breakdown fields for CO2, CH4, N2O, and extensible fluorinated gases where relevant. | Data Platform | Certified factor validation fails if gas structure is absent. |
| FR5 | P0 | Harden | Define `formula_type`: direct factor, stoichiometric, composite, proxy, supplier-specific, residual-mix, lifecycle-derived. | Methodology + Data Platform | Schema enum and docs exist. |
| FR6 | P0 | Harden | Add category-specific parameter schemas for combustion, electricity, transport, materials/products, refrigerants, land/removals, and finance proxies. | Data Platform | JSON/YAML schema files and fixtures exist for each category. |
| FR7 | P0 | Verify | Store raw source records separately from GreenLang-normalized records. | Data Platform | Every normalized factor links to immutable raw artifact + source row or source object. |
| FR8 | P0 | Verify | Ensure source catalog object contains authority, title, publisher, jurisdiction, dataset version, publication date, validity period, ingestion date, source type, redistribution class, verification status, citation, legal notes. | Data Platform + Legal | Source schema validation and source inventory export pass. |
| FR9 | P1 | Harden | Add classification mappings to every applicable factor: activity category, fuel code, product/material taxonomy, transport mode, waste route, spend category, trade/industry code where applicable. | Data Platform | Classification coverage dashboard shows percent complete by factor family. |
| FR10 | P1 | Harden | Implement replacement pointers for deprecated factors. | Backend + Data Platform | Deprecated factor API response points to replacement when available. |
| FR11 | P1 | Build | Create registry-level data dictionary for every field. | DevRel + Data Platform | Public docs include field descriptions, examples, nullability, and business meaning. |
| FR12 | P1 | Harden | Add schema compatibility policy for v1.x releases. | Backend | Semver policy defines breaking vs non-breaking field changes. |
| FR13 | P1 | Verify | Confirm catalog supports Certified, Preview, Connector-only, Deprecated, Superseded, Private statuses. | Data Platform | Status enum, tests, and UI filters exist. |
| FR14 | P1 | Build | Create sample canonical records for India electricity, UK residual mix, diesel combustion, R-134a refrigerant, road freight, hot-rolled steel proxy, CBAM steel selector. | Methodology + DevRel | Examples appear in docs and tests. |
| FR15 | P2 | Build | Create factor-family registry so new categories can be added without schema drift. | Data Platform | New family can be registered with parameter schema and resolver rules. |

---

## 7. Unit and Chemistry Engine

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| UC1 | P0 | Verify | Confirm numerator/denominator unit graph exists and supports mass, energy, volume, distance, area, currency, passenger, freight, and product units. | Data Platform + QA | Unit graph tests cover all v1 families. |
| UC2 | P0 | Verify | Confirm SI and commercial unit conversion with dimensional checks. | Data Platform | Invalid conversion, such as kg to kWh without bridge parameter, fails. |
| UC3 | P0 | Verify | Confirm gas-level storage and CO2e derivation from selected GWP set. | Data Platform | Same gas components can be returned under AR4, AR5, AR6. |
| UC4 | P0 | Harden | Add explicit GWP sets required for v1: IPCC AR4 100, AR5 100, AR6 100, and extension slot for future sets. | Methodology + Data Platform | Test resolves same refrigerant under each GWP set. |
| UC5 | P0 | Harden | Validate LHV/HHV handling for combustion fuels. | Methodology | Tests show factor changes or warnings when energy basis differs. |
| UC6 | P0 | Harden | Validate density conversions for mass-volume fuel conversions. | Data Platform | Diesel/LPG examples convert correctly with source density assumptions. |
| UC7 | P0 | Harden | Implement oxidation factor support for combustion. | Methodology | Combustion fixture includes oxidation assumption in explain payload. |
| UC8 | P1 | Harden | Implement moisture adjustment where needed for biomass/solid fuel factors. | Methodology | Moisture-adjusted fixture includes assumption and uncertainty. |
| UC9 | P1 | Harden | Implement fossil vs biogenic carbon split. | Methodology | Resolver output labels fossil CO2, biogenic CO2, CH4, N2O separately where relevant. |
| UC10 | P1 | Verify | Validate refrigerant gas-code mapping and GWP-set mapping. | QA | Refrigerant tests cover common gases and blends. |
| UC11 | P1 | Build | Add chemistry sanity validator for impossible stoichiometric or unit combinations. | Data Platform | Validator catches invalid carbon content/fuel conversions. |
| UC12 | P2 | Build | Add uncertainty propagation for composite factors. | Data Platform + Methodology | Composite factor returns uncertainty band derived from inputs. |

---

## 8. Method Pack Library

Every method pack must be a versioned, shippable artifact, not just code branches.

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| MP1 | P0 | Verify | Confirm method pack registry exists and names all seven packs: Corporate Inventory, Product Carbon, Freight, Electricity, Land/Removals, Finance Proxy, EU Policy. | Methodology + Backend | Registry endpoint lists all packs with version/status. |
| MP2 | P0 | Build | Write Method Pack Specification Template. | Methodology | Template includes selection rules, boundaries, inclusions/exclusions, gas-to-CO2e basis, biogenic treatment, market instruments, region hierarchy, fallback logic, audit text, labels, deprecation policy. |
| MP3 | P0 | Harden | Corporate Inventory Pack v0.1: GHG Protocol Corporate + Scope 2 + Scope 3 alignment. | Methodology | Pack resolves Scope 1 fuel, Scope 2 electricity, and Scope 3 freight/material examples. |
| MP4 | P0 | Harden | Electricity Pack v0.1: location-based, market-based, supplier-specific, residual-mix logic. | Methodology + Backend | India location-based, EU residual mix, US eGRID, supplier-specific examples pass. |
| MP5 | P0 | Harden | EU Policy Pack v0.1: CBAM selectors and DPP-ready product structures. | Methodology | CBAM steel/aluminium/cement examples return method-correct factor choices. |
| MP6 | P1 | Harden | Freight Pack v0.1: ISO 14083 + GLEC-aligned WTW/TTW transport chain logic. | Methodology | Road/sea/air freight examples include mode, payload, utilization, WTW/TTW labels. |
| MP7 | P1 | Harden | Product Carbon Pack v0.2: GHG Protocol Product + ISO 14067 + PACT object compatibility. | Methodology | PACT-compatible payload example validates. |
| MP8 | P2 | Harden | Land & Removals Pack v0.2: GHG Protocol LSR-aligned profile. | Methodology | Land-use/removals example returns permanence/reversal/biogenic treatment fields. |
| MP9 | P2 | Harden | Finance Proxy Pack v0.2: PCAF-aligned proxy factors. | Methodology | Finance asset-class example returns proxy confidence and intensity basis. |
| MP10 | P0 | Harden | Method-pack versioning and deprecation. | Backend | API can pin `method_pack_version` and reproduce old decision. |
| MP11 | P0 | Harden | Method packs must call resolver, not bypass resolver. | Backend | Tests prove all pack examples go through resolver chain. |
| MP12 | P1 | Build | Add method-pack release notes and changelog. | DevRel + Methodology | Each method pack release includes changes, impact, and migration notes. |
| MP13 | P1 | Build | Create audit text templates per method pack. | Methodology | Explain payload includes method-specific audit paragraph. |
| MP14 | P1 | Build | Add method-pack coverage dashboard. | Frontend + Backend | UI shows pack status, version, supported families, unsupported cases. |
| MP15 | P1 | Build | Add method-pack conformance tests. | QA | Each pack has canonical examples and expected selected factors. |

---

## 9. Resolution Engine

The resolver is the product brain. Search is only an input.

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| R1 | P0 | Verify | Confirm resolver service is separate from search/matching service. | Backend | Code path shows deterministic resolver after candidate generation. |
| R2 | P0 | Verify | Implement and document selection order: customer override, supplier/manufacturer factor, facility/asset factor, utility/tariff/grid subregion, country/sector factor, method-pack fallback, global default. | Backend + Methodology | Resolver test asserts ordered fallback behavior. |
| R3 | P0 | Verify | Implement tie-breakers: geography, time, technology, unit, method compatibility, source authority, verification, uncertainty, recency, license availability. | Backend | Tie-break fixture proves deterministic winner. |
| R4 | P0 | Harden | Resolver output contract must include chosen factor, alternates, rationale, source/version, quality, uncertainty, gas breakdown, CO2e basis, assumptions, deprecation status. | Backend | OpenAPI contract and unit tests enforce output shape. |
| R5 | P0 | Harden | Make `/resolve` accept activity amount, unit, geography, time, method profile, basis, category, supplier, product, and caller entitlement context. | Backend | End-to-end API tests for electricity, fuel, freight, material, refrigerant. |
| R6 | P0 | Harden | Make `/explain` first-class and callable by factor ID, resolve request ID, or signed receipt ID. | Backend | API docs and tests for all three explain modes. |
| R7 | P0 | Harden | Resolver must be license-aware. | Backend + Legal | Caller without entitlement never receives restricted factor value. |
| R8 | P0 | Harden | Resolver must be version-pinned by factor release and method pack version. | Backend | Same request returns same result when pinned to prior edition. |
| R9 | P0 | Harden | Customer override hook must be explicit and auditable. | Backend | Override appears in lineage, explain output, and audit log. |
| R10 | P1 | Build | Supplier/manufacturer-specific factor priority path. | Backend | Supplier factor beats generic default when entitlement and validity match. |
| R11 | P1 | Build | Facility/asset-specific factor priority path. | Backend | Facility factor beats country default. |
| R12 | P1 | Build | Utility/tariff/grid-subregion matching for electricity. | Backend | eGRID/subregion/residual-mix examples pass. |
| R13 | P1 | Build | Bulk resolver endpoint for CSV/JSON batch jobs. | Backend | Batch job resolves 10k rows with receipt manifest and error file. |
| R14 | P1 | Harden | Add resolver latency targets: p95 under 500ms for single resolve on warm cache, p95 under agreed SLA for batch. | Backend + SRE | Load test report committed. |
| R15 | P1 | Build | Add "alternates considered" scoring transparency. | Backend | Explain output shows at least top 3 alternates with reason lost. |
| R16 | P1 | Build | Add deprecation warning behavior. | Backend | Deprecated chosen factor returns warning and replacement pointer. |
| R17 | P1 | Build | Add "cannot resolve safely" behavior. | Backend | Resolver returns structured error instead of unsafe global default when method requires specificity. |
| R18 | P2 | Build | GraphQL query layer for flexible factor exploration. | Backend | GraphQL schema and tests exist; optional for day-one launch. |

---

## 10. Mapping Layer and Taxonomies

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| ML1 | P0 | Verify | Confirm mapping layer is distinct from parser service and resolver. | Data Platform | Architecture docs identify mapping workbench/service separately. |
| ML2 | P0 | Verify | Fuel taxonomy maps common names, aliases, and source labels to canonical fuel codes. | Data Platform | Diesel/natural gas/LPG/coal examples normalize. |
| ML3 | P0 | Harden | Electricity market taxonomy supports country, grid region, subregion, supplier, residual mix, certificate context. | Data Platform | Electricity mapping tests pass. |
| ML4 | P0 | Harden | Transport taxonomy supports mode, vehicle/vessel/aircraft class, payload basis, distance basis, WTW/TTW labels. | Data Platform | Freight examples normalize into resolver-ready payloads. |
| ML5 | P1 | Harden | Product/material taxonomy supports steel, aluminium, cement, fertilizer, plastics, paper, glass, chemicals. | Data Platform | Purchased-goods proxy examples map to factor families. |
| ML6 | P1 | Build | Waste route taxonomy supports landfill, incineration, recycling, composting, wastewater starter set. | Data Platform | Waste examples map and resolve where v1 data exists. |
| ML7 | P1 | Build | Refrigerant taxonomy supports ASHRAE-style gas/blend codes and aliases. | Data Platform | R-134a/R134a/HFC-134a normalize to same gas code. |
| ML8 | P1 | Build | Supplier category and spend category taxonomy for Scope 3 screening. | Data Platform + Methodology | Spend-based examples map to proxy factors with confidence. |
| ML9 | P1 | Build | Trade code mapping for CBAM/DPP flows. | Data Platform | CN/HS code examples map to CBAM selector categories. |
| ML10 | P1 | Build | Industry code mapping: NAICS, NACE, ISIC crosswalk. | Data Platform | Crosswalk tests pass for representative sectors. |
| ML11 | P1 | Build | Mapping workbench UI for human review and correction. | Frontend + Data Platform | Operator can approve/reject mappings and create audit trail. |
| ML12 | P2 | Build | Active-learning loop from failed searches and overrides. | Data Platform + ML | Failed resolve cases create review queue and improve mapping eval set. |

---

## 11. Source Catalog, Ingestion, and Source Packs

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| S1 | P0 | Verify | Confirm source registry exists with rights, cadence, owner, parser type, legal status. | Data Platform + Legal | Source inventory export reviewed by legal. |
| S2 | P0 | Verify | Default public pack includes EPA GHG Hub, EPA eGRID, UK DESNZ, GHG Protocol objects, IPCC defaults/EFDB coverage, India CEA, AIB residual mix or equivalent residual-mix source. | Data Platform + Methodology | Coverage matrix lists source, status, parser, license, update cadence. |
| S3 | P0 | Harden | Ensure every source has raw artifact storage with checksum and ingestion timestamp. | Data Platform | Raw-to-normalized trace exists for certified sources. |
| S4 | P0 | Harden | Parser specs for all v1 launch sources. | Data Platform | Each parser spec includes source format, fields, transformation rules, tests. |
| S5 | P0 | Harden | Source diffing and source-watch jobs for official pages/files. | Data Platform + SRE | Watch pipeline creates ticket/change event when source changes. |
| S6 | P0 | Harden | Legal rights matrix before any source becomes Certified/public. | Legal + Product | Certification workflow blocks missing rights signoff. |
| S7 | P1 | Harden | Product/LCI premium pack: ecoinvent connector, EC3/EPD support, PACT data objects. | Data Platform + Legal | Premium pack is entitlement-gated and docs disclose license limits. |
| S8 | P1 | Harden | Freight premium pack: lane logic, mode factors, payload/utilization, WTW/TTW. | Data Platform + Methodology | Freight pack test cases pass. |
| S9 | P2 | Harden | Finance pack: PCAF proxies and proxy confidence model. | Data Platform + Methodology | Finance examples validate; commercial status can remain Preview. |
| S10 | P2 | Harden | Land/removals pack: LSR data objects and assumptions. | Data Platform + Methodology | LSR examples validate; commercial status can remain Preview. |
| S11 | P1 | Build | Source catalog UI with source status, parser health, license class, current edition, last watch, next review. | Frontend | Operator can inspect source health and legal status. |
| S12 | P1 | Build | Source ingestion console for reruns, failures, diffs, parser logs. | Frontend + Data Platform | Operator can rerun parser and see diff output. |
| S13 | P1 | Harden | Release cadence per source, not universal weekly refresh. | Product + Data Platform | Source registry has watch cadence and publish behavior per source. |
| S14 | P1 | Build | Source citation generator with required attribution text. | DevRel + Legal | API and audit export include citation text. |
| S15 | P2 | Build | Partner/customer-provided source ingestion path. | Backend + Legal | Customer private upload remains private and traceable. |

---

## 12. Quality, Uncertainty, and Review Engine

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| Q1 | P0 | Verify | Quality score model includes temporal, geographic, technology, verification, and completeness component scores. | Methodology + QA | Certified factor includes components and composite 0-100 score. |
| Q2 | P0 | Harden | Define Factor Quality Score calculation and thresholds for Certified vs Preview. | Methodology | Written scoring spec and tests. |
| Q3 | P0 | Harden | Add uncertainty type and uncertainty range fields. | Data Platform | Schema and fixtures include quantitative or qualitative uncertainty. |
| Q4 | P0 | Harden | Approval state, review owner, next review date required for Certified factors. | Methodology + QA | Release signoff blocks missing review metadata. |
| Q5 | P0 | Harden | Human methodology review workflow for factors needing approval. | Frontend + Backend | Review queue supports assign, approve, reject, request changes. |
| Q6 | P0 | Verify | Validation rules cover invalid units, missing lineage, invalid dates, duplicate candidates, outliers, broken geography, license gaps. | QA | Validator test suite green. |
| Q7 | P1 | Build | QA dashboard by family/source/status/quality score. | Frontend + QA | Dashboard shows counts and drilldown. |
| Q8 | P1 | Build | Drift analysis vs previous release. | Data Platform | QA dashboard flags material numeric changes. |
| Q9 | P1 | Build | Cross-source reconciliation checks. | Data Platform + Methodology | Conflicting factors create review issues. |
| Q10 | P1 | Build | Annual audit bundle exporter. | Backend + DevRel | Export includes raw evidence, parser logs, normalized factor, QA result, reviewer, release manifest. |
| Q11 | P1 | Harden | Gold-label evaluation set across method profiles and families. | QA | CI fails if precision/recall drops below threshold. |
| Q12 | P2 | Build | Public benchmark report for matching/resolution quality. | DevRel + QA | Published report shows eval set size, precision, recall, known limitations. |

---

## 13. Provenance, Governance, and Release Management

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| GOV1 | P0 | Verify | Immutable change log per factor includes who, why, what changed, source, affected calculations, rollback/migration notes. | Backend + QA | Change log fixture proves all fields. |
| GOV2 | P0 | Harden | Segregation of duties: ingester cannot be sole approver/releaser. | Backend + Product | Approval workflow enforces role separation. |
| GOV3 | P0 | Harden | Release manager creates signed edition manifest. | Backend + SRE | `v1.0 Certified` manifest has hash and signature. |
| GOV4 | P0 | Harden | Reproducibility manifests allow re-running a filed inventory with same factors/method packs. | Backend | Re-run test returns identical result under pinned edition. |
| GOV5 | P0 | Build | Cut first narrow Certified edition. | Product + Methodology + Backend | Edition tag exists, signed, documented, and deployed. |
| GOV6 | P1 | Build | Factor-level diff viewer. | Frontend + Backend | Operator can compare factor versions side-by-side. |
| GOV7 | P1 | Build | Impact simulator: "what breaks if we replace this factor/method pack/source?" | Backend + Frontend | Simulator lists impacted tenants, inventories, receipts, methods. |
| GOV8 | P1 | Build | Deprecation notices with lead time and webhook. | Backend + DevRel | API consumer receives deprecation webhook and docs explain timeline. |
| GOV9 | P1 | Build | Rollback workflow with audit trail. | Backend + SRE | Test release can be rolled back and logged. |
| GOV10 | P1 | Build | Release notes generator from source/method/factor changes. | Backend + DevRel | Release notes are produced for edition cut. |
| GOV11 | P2 | Build | GreenLang Foundation governance stub for schemas, factor governance, OSS repositories, community standards. | Product + Legal | Governance note exists; not launch-blocking. |

---

## 14. Licensing, Entitlements, and Data-Class Separation

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| L1 | P0 | Harden | Formalize four data classes: Open, Licensed Embedded, Customer Private, OEM Redistributable. | Legal + Product | Data-class policy approved. |
| L2 | P0 | Harden | Separate storage locations/namespaces by data class. | Backend + SRE | Storage review proves no mixed public/licensed/private bucket/table. |
| L3 | P0 | Harden | Entitlement service decides search visibility, resolve permission, export permission, and redistribution permission. | Backend | Entitlement tests cover all classes and plans. |
| L4 | P0 | Harden | Public API must not leak restricted factor value, source detail, or derived data to unauthorized caller. | Backend + Security | Negative tests prove redaction/403 behavior. |
| L5 | P0 | Harden | Customer-private factor vault. | Backend + Frontend | Tenant A cannot see Tenant B private factor; override appears only for tenant. |
| L6 | P1 | Build | OEM redistribution rights model. | Legal + Backend | OEM tenant can redistribute only allowed classes to sub-tenants. |
| L7 | P1 | Build | Contract templates by data class. | Legal | Standard clauses exist for public, licensed, private, OEM. |
| L8 | P1 | Build | Release bundles segregated by data class. | Backend | Public release cannot include licensed/customer-private factors. |
| L9 | P1 | Build | License scanner before certification. | Data Platform + Legal | CI or release check flags unknown/restricted licenses. |
| L10 | P2 | Build | Licensed connector audit log. | Backend + Legal | Connector usage logs support source-license audit. |

---

## 15. API, SDK, CLI, Batch, Webhooks

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| API1 | P0 | Harden | Stable REST API under `/v1`. | Backend | OpenAPI spec and contract tests pass. |
| API2 | P0 | Harden | `/search`, `/factors/{id}`, `/resolve`, `/explain`, `/releases`, `/lineage`, `/sources`, `/method-packs`, `/quality`, `/batch` endpoints. | Backend | API docs include examples and tests. |
| API3 | P0 | Harden | Every factor-returning endpoint supports edition pinning. | Backend | `X-GL-Edition` and query pinning tests pass. |
| API4 | P0 | Harden | API auth: keys/OAuth, tenant context, plan context. | Backend + Security | Unauthorized and cross-tenant tests pass. |
| API5 | P0 | Harden | Rate limiting by plan. | Backend + SRE | Community, Developer Pro, Platform, Enterprise plan tests pass. |
| API6 | P0 | Harden | Usage metering for billing. | Backend | Metered events include caller, endpoint, edition, pack, units, status. |
| API7 | P0 | Harden | Signed result receipts enforced by middleware. | Backend + Security | Every resolve/explain response has verifiable signature. |
| API8 | P0 | Verify | Python SDK supports auth, search, resolve, explain, edition pinning, receipt verification. | DevRel + Backend | SDK integration test uses hosted/staging API. |
| API9 | P0 | Verify | TypeScript SDK supports auth, search, resolve, explain, edition pinning, receipt verification. | DevRel + Backend | SDK integration test uses hosted/staging API. |
| API10 | P1 | Harden | CLI supports factor search, resolve, explain, pin, export, edition show, source status. | DevRel + Backend | CLI quickstart works end to end. |
| API11 | P1 | Build | Batch API for CSV/JSON jobs. | Backend | 10k-row batch returns result file, error file, receipt manifest. |
| API12 | P1 | Build | Webhooks for factor, method-pack, source, deprecation, release changes. | Backend | Test receiver gets signed webhook. |
| API13 | P1 | Build | Hosted explain logs for paid plans. | Backend | User can retrieve prior resolve request and explain payload. |
| API14 | P1 | Build | SDK release automation to PyPI/npm. | SRE + DevRel | CI can publish tagged SDK release. |
| API15 | P2 | Build | GraphQL API for flexible querying. | Backend | Not day-one blocker unless requested by design partner. |
| API16 | P2 | Build | Java SDK. | DevRel | Optional unless platform partner requires it. |

---

## 16. Operator Surfaces

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| O1 | P0 | Harden | Factor explorer for internal/operator use. | Frontend | Search/filter factors by family, geography, method, source, status. |
| O2 | P0 | Harden | Source ingestion console. | Frontend + Data Platform | Operator sees parser health, diffs, failures, reruns. |
| O3 | P0 | Harden | QA dashboard. | Frontend + QA | Shows quality score distribution, validation failures, review queue. |
| O4 | P0 | Harden | Approval workflow UI. | Frontend + Methodology | Methodology lead can approve/reject factor changes. |
| O5 | P1 | Harden | Mapping workbench. | Frontend + Data Platform | Operator can review and approve raw-label-to-taxonomy mappings. |
| O6 | P1 | Harden | Factor diff viewer. | Frontend + Backend | Version-to-version diff visible. |
| O7 | P1 | Harden | Customer override manager. | Frontend + Backend | Override can be created, approved, audited, and resolved. |
| O8 | P1 | Harden | Impact simulator. | Frontend + Backend | Operator previews impacted customers/inventories before release. |
| O9 | P1 | Build | Method-pack manager UI. | Frontend + Methodology | Pack version/status/deprecation visible. |
| O10 | P1 | Build | Entitlements/admin console for plans, packs, private factors, OEM sub-tenants. | Frontend + Backend | Admin can grant/revoke pack and data-class access. |
| O11 | P1 | Build | Audit bundle export UI. | Frontend + Backend | Operator exports evidence package from UI. |
| O12 | P2 | Build | Public coverage dashboard. | Frontend | Certified/Preview/Connector-only counts visible externally. |

---

## 17. Developer Portal and Public Documentation

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| DX1 | P0 | Build | Developer portal landing page focused on API use, not marketing fluff. | DevRel + Frontend | New developer can find quickstart immediately. |
| DX2 | P0 | Build | Quickstart: get API key, install SDK, resolve India electricity, inspect explain payload. | DevRel | Quickstart completes in under 10 minutes. |
| DX3 | P0 | Build | API reference from OpenAPI with examples. | DevRel + Backend | `/resolve` and `/explain` examples render correctly. |
| DX4 | P0 | Build | Concepts docs: factor, source, method pack, edition, license class, quality score, signed receipt. | DevRel | Docs reviewed by product/methodology/backend. |
| DX5 | P0 | Build | Canonical factor schema documentation with complete JSON example. | DevRel + Data Platform | Docs include required fields and category-specific parameters. |
| DX6 | P1 | Build | Method pack documentation for all seven packs. | Methodology + DevRel | Each pack has support status and examples. |
| DX7 | P1 | Build | SDK docs for Python and TypeScript. | DevRel | Install and usage examples tested. |
| DX8 | P1 | Build | CLI docs. | DevRel | CLI examples tested. |
| DX9 | P1 | Build | Licensing docs for open vs licensed vs private vs OEM. | Legal + DevRel | Docs avoid overpromising redistribution. |
| DX10 | P1 | Build | Changelog and deprecation feed. | DevRel + Backend | Public changelog generated from release manager. |
| DX11 | P1 | Build | Error-code documentation. | DevRel + Backend | Resolver, auth, entitlement, license, validation errors documented. |
| DX12 | P2 | Build | Public examples repo with notebooks and CSV batch examples. | DevRel | Examples run in CI. |

---

## 18. Commercial Packaging and Pricing

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| C1 | P0 | Commercial | Define five-tier packaging: Community, Developer Pro, Consulting/Platform, Enterprise, Premium Data Packs. | Product + Commercial | Pricing/package memo approved by CTO/founder. |
| C2 | P0 | Commercial | Define price drivers: API calls, batch volume, pack entitlements, private registry usage, tenants, OEM rights, SLA/support. Do not price by factor count. | Product + Commercial | Pricing page and contract model reflect this. |
| C3 | P0 | Commercial | Define Community/Open-Core boundary: schema, SDK/CLI, limited public pack, sandbox API, docs, examples. | Product + Legal | Free tier docs do not imply enterprise entitlements. |
| C4 | P0 | Commercial | Define Developer Pro: production API, rate limits, batch, version pinning, basic support, hosted explain logs. | Product | SKU exists in billing config. |
| C5 | P1 | Commercial | Define Consulting/Platform plan: multi-client workspaces, white-label, client override vaults, audit exports, premium packs, partner support. | Product + Commercial | Sales deck and SKU exist. |
| C6 | P1 | Commercial | Define Enterprise: SSO/SCIM, VPC/private deploy, private registry, approval workflows, customer-specific factors, signed releases, SLA. | Product + Commercial | Enterprise checklist and contract exhibit exist. |
| C7 | P1 | Commercial | Define Premium Data Pack SKUs: Electricity, Freight, PCF/LCI, Construction/EPD, Agrifood/Land, Finance Proxy, CBAM/EU Policy. | Product + Methodology | SKU table and entitlement mapping exist. |
| C8 | P1 | Build | Implement billing SKUs and plan limits in code. | Backend | Billing tests cover package entitlements. |
| C9 | P1 | Build | Pricing page with plan comparison and self-serve Developer Pro checkout. | Frontend + Commercial | Test checkout creates tenant + API key. |
| C10 | P1 | Commercial | OEM/white-label terms. | Legal + Commercial | Standard OEM order form and redistribution clauses exist. |
| C11 | P1 | Commercial | Consultant sales one-pager. | Commercial + DevRel | PDF/page explains client vaults, overrides, audit bundles. |
| C12 | P1 | Commercial | Enterprise RFP response pack. | Commercial + Security | Security, SLA, data rights, auditability, deployment options documented. |
| C13 | P2 | Commercial | Licensed pack pass-through + margin model. | Commercial + Legal | Source cost and GreenLang margin model approved. |

---

## 19. Deployment, SRE, Observability, and Support

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| DEP1 | P0 | Ops | Deploy staging API for design partners. | SRE + Backend | Staging `/v1/health` returns current edition. |
| DEP2 | P0 | Ops | Deploy production API behind HTTPS, WAF/API gateway, rate limits, auth. | SRE | Production smoke test passes. |
| DEP3 | P0 | Ops | Provision Postgres/pgvector for catalog/search and Redis for cache. | SRE + Backend | Migration and cache health checks pass. |
| DEP4 | P0 | Ops | Configure release environments: dev, staging, prod. | SRE | Promotion path documented and tested. |
| DEP5 | P0 | Ops | Metrics: p50/p95/p99 latency, error rate, resolution success, fallback rank, entitlement denials, billing events. | SRE | Grafana/Prometheus dashboard live. |
| DEP6 | P0 | Ops | Alerts for p95 latency, error spikes, failed source watch, failed parser, failed release signoff, billing failures. | SRE | Alert simulation works. |
| DEP7 | P0 | Ops | Backup/restore test for catalog and raw artifacts. | SRE | Restore drill documented. |
| DEP8 | P0 | Ops | Incident runbook for wrong factor returned, source license issue, API outage, bad release, signed receipt verification failure. | SRE + Product | Runbook approved. |
| DEP9 | P1 | Ops | SLA tracker for paid plans. | SRE + Commercial | SLA report can be generated for tenant. |
| DEP10 | P1 | Ops | Status page for Factors API. | SRE | Public/internal status page exists. |
| DEP11 | P1 | Ops | Blue/green or canary release for API. | SRE | Rollback tested. |
| DEP12 | P1 | Ops | Load test with realistic resolve/search/batch mix. | SRE + QA | Load report meets launch thresholds. |
| DEP13 | P2 | Ops | VPC/private deployment template. | SRE | Enterprise deployment pattern documented. |

---

## 20. Security, Compliance, and Enterprise Readiness

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| SEC1 | P0 | Harden | API key security: hashing, rotation, revocation, scoped keys. | Backend + Security | Security tests pass. |
| SEC2 | P0 | Harden | Tenant isolation for catalog, private factors, overrides, receipts, logs. | Backend + Security | Cross-tenant access tests fail closed. |
| SEC3 | P0 | Harden | Signed receipts use key rotation and published verification keys. | Backend + Security | Offline verifier handles current and prior keys. |
| SEC4 | P0 | Harden | Audit log for admin, operator, release, entitlement, override actions. | Backend | Audit trail export works. |
| SEC5 | P1 | Build | SSO/SAML/OIDC for Enterprise. | Backend + SRE | Enterprise test tenant logs in through SSO. |
| SEC6 | P1 | Build | SCIM provisioning for Enterprise. | Backend | SCIM create/update/deactivate tests pass. |
| SEC7 | P1 | Build | Data retention policy for logs, receipts, customer-private factors, raw artifacts. | Legal + SRE | Retention config and policy exist. |
| SEC8 | P1 | Build | SOC2-ready control evidence for Factors launch. | Security + SRE | Control checklist and evidence folder exist. |
| SEC9 | P1 | Build | DPA/security questionnaire pack. | Legal + Security | Sales can answer enterprise security reviews. |
| SEC10 | P2 | Build | Private deployment hardening guide. | SRE + Security | Required for high-ACV enterprise only. |

---

## 21. Testing, Evaluation, and Launch QA

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| T1 | P0 | Verify | Run full `tests/factors` suite and fix failures. | QA + Backend | Green CI run captured. |
| T2 | P0 | Verify | Run launch v1 coverage tests for combustion, electricity, freight, materials, method profiles, refrigerants. | QA | Coverage CI green. |
| T3 | P0 | Verify | Gold-set evaluation has 300-500 real-world activity descriptions. | QA + Methodology | Dataset count and coverage report. |
| T4 | P0 | Harden | CI gate for resolver precision/recall. | QA | PR fails if quality drops below threshold. |
| T5 | P0 | Harden | OpenAPI contract tests. | Backend + QA | Contract tests green. |
| T6 | P0 | Harden | Explain contract tests: no factor result without explain unless explicitly compact. | QA | Test fails on missing explain. |
| T7 | P0 | Harden | Entitlement negative tests for licensed/private/OEM classes. | QA + Backend | Unauthorized access denied. |
| T8 | P0 | Harden | Signed receipt verification tests. | QA + Security | Saved response verifies offline. |
| T9 | P0 | Harden | Reproducibility tests for pinned editions. | QA | Same pinned input returns same output. |
| T10 | P1 | Harden | API load tests for search, resolve, explain, batch. | QA + SRE | Performance report meets launch SLA. |
| T11 | P1 | Build | Frontend E2E tests for explorer, QA dashboard, approval queue, impact simulator, pricing checkout. | Frontend + QA | Playwright tests green. |
| T12 | P1 | Build | Source parser regression tests for all v1 sources. | Data Platform + QA | Parser tests green. |
| T13 | P1 | Build | Method-pack conformance test matrix. | Methodology + QA | Each pack has expected selection cases. |
| T14 | P1 | Build | Security test pass: auth, tenant isolation, secrets, API abuse. | Security | Security report has no P0/P1 issues. |
| T15 | P2 | Build | Chaos/failure tests for source watch, cache outage, DB failover. | SRE | Failure mode runbook validated. |

---

## 22. Design Partners and Pilot Readiness

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| DP1 | P0 | Commercial | Identify 5-8 design partners: developer/platform, consultant, India-linked exporter, auditor/verifier, ERP/integrator. | Product + Commercial | Named list with sponsor, use case, status. |
| DP2 | P0 | Build | Pilot onboarding checklist. | Product | Checklist covers NDA/MSA, API key, workspace, use case, support channel, success metrics. |
| DP3 | P0 | Build | First pilot use cases: India electricity Scope 2, fuel combustion Scope 1, CBAM material selector, freight lane, refrigerant. | Product + Methodology | Each pilot has test payload and expected business outcome. |
| DP4 | P1 | Commercial | Design partner agreement template. | Legal | Template includes data rights, feedback rights, SLA limits, licensed-data restrictions. |
| DP5 | P1 | Build | Support runbook and escalation process. | Product + SRE | Slack/email/support workflow defined. |
| DP6 | P1 | Build | Pilot analytics: API usage, failed resolves, fallback ranks, support tickets, time-to-factor. | Backend + Product | Weekly pilot dashboard exists. |
| DP7 | P1 | Build | Feedback-to-backlog workflow. | Product | Failed/ambiguous cases become triaged tickets. |
| DP8 | P1 | Commercial | Pilot conversion plan to paid Developer Pro, Platform, or Enterprise. | Commercial | Pricing path and renewal date defined per partner. |
| DP9 | P2 | Commercial | Public reference case study after successful pilot. | Commercial | Only after partner approval. |

---

## 23. Launch Documentation for CTO, Legal, Sales, and Engineering

| ID | Priority | State | Task | Owner | Acceptance proof |
|---|---:|---|---|---|---|
| DOC1 | P0 | Build | CTO architecture deck: registry, resolver, method packs, governance, API, licensing. | Product + Backend | CTO can present architecture without oral caveats. |
| DOC2 | P0 | Build | Engineering runbook: local dev, tests, source ingestion, release, rollback. | Backend + SRE | New engineer can run tests and cut dev edition. |
| DOC3 | P0 | Build | Methodology manual: supported standards, pack rules, fallback policy, quality scoring. | Methodology | Reviewed by methodology lead. |
| DOC4 | P0 | Build | Legal source-rights binder. | Legal | Every v1 source has license notes and redistribution class. |
| DOC5 | P1 | Commercial | Sales one-pager for Factors API. | Commercial + DevRel | Usable in outbound and investor/data-partner meetings. |
| DOC6 | P1 | Commercial | Consultant/auditor one-pager. | Commercial | Emphasizes audit bundles, overrides, methodology notes. |
| DOC7 | P1 | Commercial | Platform/OEM one-pager. | Commercial | Emphasizes white label, sub-tenants, signed receipts, redistribution controls. |
| DOC8 | P1 | Build | RFP/security answer bank. | Commercial + Security | Security and procurement answers ready. |
| DOC9 | P1 | Build | Public changelog page. | DevRel | First v1 edition changelog published. |
| DOC10 | P2 | Build | Public roadmap with open-core/commercial boundaries. | Product | Avoids community confusion. |

---

## 24. 12-Week Execution Sequence

### Weeks 1-2: Product Contract and Release Gates

| ID | Task |
|---|---|
| W1-1 | Approve product charter, launch scope, non-scope, buyer journeys. |
| W1-2 | Convert N1-N7 into release-blocking automated/manual gates. |
| W1-3 | Freeze canonical schema v1 and source object schema. |
| W1-4 | Confirm method pack registry and write Method Pack Specification Template. |
| W1-5 | Audit legal rights matrix for all v1 sources. |
| W1-6 | Run full `tests/factors` suite and record baseline failures. |
| W1-7 | Assign owners for every P0/P1 task in this document. |

### Weeks 3-4: Resolver, Schema, Method Packs

| ID | Task |
|---|---|
| W3-1 | Harden `/resolve` and `/explain` output contracts. |
| W3-2 | Confirm resolver fallback order and tie-breakers with method lead. |
| W3-3 | Harden Corporate Inventory, Electricity, and EU Policy packs. |
| W3-4 | Finish gas-level/GWP/unit/LHV/HHV release tests. |
| W3-5 | Add entitlement checks to every factor-returning path. |
| W3-6 | Add launch coverage tests for v1 use cases. |

### Weeks 5-6: API, Entitlements, Release, QA

| ID | Task |
|---|---|
| W5-1 | Stabilize REST `/v1` API and OpenAPI contract. |
| W5-2 | Enforce signed receipts middleware. |
| W5-3 | Enforce edition pinning. |
| W5-4 | Cut staging v1 Certified edition. |
| W5-5 | Build QA dashboard and review workflow enough for methodology signoff. |
| W5-6 | Complete gold-set CI gate. |

### Weeks 7-8: Deployment and Developer Experience

| ID | Task |
|---|---|
| W7-1 | Deploy staging API with Postgres/Redis/metrics. |
| W7-2 | Publish developer quickstart and API docs. |
| W7-3 | Validate Python and TypeScript SDKs against staging. |
| W7-4 | Build CLI quickstart. |
| W7-5 | Run load tests and fix p95 latency issues. |
| W7-6 | Publish initial coverage dashboard internally. |

### Weeks 9-10: Commercial Surface and Operator Console

| ID | Task |
|---|---|
| W9-1 | Finalize five-tier packaging and pack SKUs. |
| W9-2 | Wire billing/plan limits/usage metering. |
| W9-3 | Launch pricing page or private pricing deck for pilots. |
| W9-4 | Wire factor explorer, source console, approval queue, diff viewer, impact simulator. |
| W9-5 | Create enterprise RFP/security pack. |
| W9-6 | Prepare design partner onboarding. |

### Weeks 11-12: Pilot Launch

| ID | Task |
|---|---|
| W11-1 | Cut production `v1.0 Certified` edition. |
| W11-2 | Deploy production API. |
| W11-3 | Onboard first 2-3 design partners. |
| W11-4 | Run live pilot scenarios and collect failed resolves. |
| W11-5 | Produce first weekly pilot report. |
| W11-6 | CTO go/no-go review for wider beta. |

---

## 25. P0 Launch Checklist

The CTO should not approve external pilot until all items below are done.

| Gate | Requirement |
|---|---|
| GATE-1 | Canonical schema v1 is frozen, tested, and documented. |
| GATE-2 | N1-N7 non-negotiables have tests or manual signoff evidence. |
| GATE-3 | `/v1/resolve` and `/v1/explain` return full explain payloads. |
| GATE-4 | Resolver is method-pack-aware, version-pinned, entitlement-aware. |
| GATE-5 | Corporate Inventory, Electricity, and EU Policy packs are v0.1-ready. |
| GATE-6 | Unit/chemistry engine supports gas-level storage and AR4/AR5/AR6 derivation. |
| GATE-7 | Launch source pack has legal rights classification and source/version lineage. |
| GATE-8 | Gold-set and launch coverage tests are green. |
| GATE-9 | Signed v1 Certified edition exists. |
| GATE-10 | Staging API is deployed with auth, rate limits, metering, signed receipts, metrics. |
| GATE-11 | Developer quickstart works with Python or TypeScript SDK. |
| GATE-12 | Entitlement tests prove licensed/private/OEM data does not leak. |
| GATE-13 | Operator can approve/reject factor changes and inspect lineage. |
| GATE-14 | Incident, rollback, and wrong-factor runbooks exist. |
| GATE-15 | First design partners have scoped use cases and onboarding docs. |

---

## 26. P1 Paid Beta Checklist

The CTO should not approve paid beta until all items below are done.

| Gate | Requirement |
|---|---|
| BETA-1 | Production API deployed with monitoring and status page. |
| BETA-2 | Pricing/packaging approved and implemented in billing config. |
| BETA-3 | Developer Pro checkout or private invoicing path exists. |
| BETA-4 | Python and TypeScript SDKs are versioned and documented. |
| BETA-5 | Batch API exists and produces receipt manifest. |
| BETA-6 | Webhooks exist for release/deprecation changes. |
| BETA-7 | QA dashboard, mapping workbench, diff viewer, and impact simulator usable by operator. |
| BETA-8 | Audit bundle exporter works. |
| BETA-9 | Freight Pack v0.1 is ready. |
| BETA-10 | Consultant and platform sales one-pagers are ready. |
| BETA-11 | Enterprise security/RFP pack is ready. |
| BETA-12 | Pilot analytics dashboard exists. |

---

## 27. FY27 H2 / Scale Checklist

| ID | Priority | Task |
|---|---:|---|
| H2-1 | P2 | Product Carbon Pack v0.2 with ISO 14067 / PACT compatibility. |
| H2-2 | P2 | Product/LCI Premium Pack with licensed connector governance. |
| H2-3 | P2 | Land & Removals Pack v0.2. |
| H2-4 | P2 | Finance Proxy Pack v0.2. |
| H2-5 | P2 | GraphQL API if platform partners request it. |
| H2-6 | P2 | Java SDK if enterprise/platform partner requests it. |
| H2-7 | P2 | VPC/private deployment template. |
| H2-8 | P2 | Public benchmark report and examples repo. |
| H2-9 | P2 | Formal GreenLang Foundation governance memo. |
| H2-10 | P2 | Wider source-pack expansion beyond narrow v1. |

---

## 28. Deferred Unless Explicitly Required

| Item | Reason |
|---|---|
| Full LCA suite | Belongs to PCF Studio/DPP Hub, not Factors v1. |
| Full MRV/removals workflow | Land/removals factors can exist, but MRV product is separate. |
| Full finance portfolio analytics | Finance proxy factors can exist, but FinanceOS is later. |
| Real-time grid optimization | Factors can connect live grid signals, but optimization belongs elsewhere. |
| Unlimited licensed LCI redistribution | Legally unsafe without source contracts. |
| Custom consulting logic inside public core | Creates maintenance debt and open-core confusion. |
| Sector-specific industrial process models | Belong in Scope Engine/PlantOS modules, not canonical Factors v1. |

---

## 29. CTO Dashboard View

Use this table in weekly CTO review.

| Area | Owner | Target status by Week 2 | Week 6 | Week 12 |
|---|---|---|---|---|
| Product scope | Product | Frozen | No scope creep | Pilot-ready |
| Non-negotiables | CTO + QA | Gates defined | Tests green | Release enforced |
| Canonical schema | Data Platform | v1 frozen | Docs/examples ready | Stable |
| Sources/legal | Data + Legal | Rights matrix complete | Certified source pack ready | Watch cycle live |
| Method packs | Methodology | Template + top 3 specs | Corporate/Electricity/EU ready | Freight ready |
| Resolver | Backend | Spec approved | API contract green | Production live |
| Unit/chemistry | Data | Test baseline | Required conversions green | Stable |
| Quality/review | QA | Scoring spec | Dashboard/review queue | Audit exports |
| Governance/release | Backend | Edition policy | Staging edition | Signed production edition |
| Licensing/entitlements | Backend + Legal | Policy approved | Tests green | Production enforced |
| API/SDK/CLI | Backend + DevRel | Contract draft | Staging API + SDK | Production API |
| Operator UI | Frontend | Wireframe/current-state audit | Core screens wired | Pilot usable |
| Developer portal | DevRel | Outline | Quickstart ready | Public/private portal live |
| Commercial | Product + Sales | Packaging memo | SKU/billing | Paid beta ready |
| Deployment/SRE | SRE | Plan | Staging | Production |
| Security | Security | Threat model | Controls tested | RFP-ready |
| Design partners | Product + Sales | 5-8 named | 2-3 ready | Live pilots |

---

## 30. Final Definition of Done

GreenLang Factors is 100% complete for FY27 launch when:

1. A developer can sign up, get an API key, install an SDK, call `/resolve`, and verify a signed receipt.
2. A consultant can resolve client activity data, apply overrides, export an audit bundle, and explain every factor choice.
3. A platform/OEM partner can use entitlements, sub-tenants, signed responses, and redistribution controls.
4. An enterprise methodology team can review, approve, deprecate, simulate impact, and cut a new factor release.
5. Legal can prove each factor belongs to Open, Licensed Embedded, Customer Private, or OEM Redistributable class.
6. The CTO can re-run an old inventory under a pinned edition and get the same answer.
7. The product can answer the canonical demo request:

```text
Resolve factor for 12,500 kWh, India, FY2027,
corporate inventory, Scope 2, location-based.
```

and return:

- chosen factor,
- alternates considered,
- why it won,
- source and source version,
- factor version,
- method pack and version,
- validity window,
- gas breakdown,
- GWP basis,
- quality score,
- uncertainty,
- licensing class,
- assumptions,
- fallback rank,
- signed receipt.

If any of those are missing, GreenLang Factors is not 100% complete.

