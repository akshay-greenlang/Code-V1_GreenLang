# PRD — GreenLang Factors (FY27 v1)

**Product:** GreenLang Factors — Layer 1 of Climate OS v3
**Release:** FY27 v1.0 (fiscal year 2027, first commercial cut)
**Version:** 1.0
**Date:** 2026-04-22
**Status:** Draft for founder + CTO sign-off
**Owners:** Founder (product), CTO (architecture), Head of Engineering (delivery), GTM Lead (pricing + pilot)
**Scope decision (founder, 2026-04-22):** v1 ships the FULL surface — all 7 method packs, all jurisdictions, every UI, every Premium Data Pack SKU. Execution-sprawl risk is managed via thin vertical slices: the earliest slices ship **Certified**, later slices ship **Preview**, none are excluded from v1 scope.
**Ground truth:** the repo walk in `FY27_vs_Reality_Analysis.md` (19 Apr 2026) and the CTO Canonical Factor Record spec (20 Apr 2026). This PRD does not invent; it packages what exists and names the gaps.

---

## 1. Executive summary

### 1.1 Problem

Every climate, ESG, and compliance product today re-builds the same bottom layer from scratch: a factor catalog, a matching engine, a source-watch pipeline, a licensing model, and a methodology map. That work is duplicated across ERP vendors, carbon-accounting platforms, PCF tools, consultancies, and individual enterprise methodology teams. When a regulator moves (CBAM Q-reports, SB 253, CSRD/ESRS, EUDR, CCTS), every one of these teams re-does the factor work in parallel. The numbers they produce are not defensible to auditors because the provenance, version, methodology, and fallback logic are not captured in a first-class record.

The market wants a **canonical climate reference system** — versioned factors with source lineage, a deterministic resolution cascade, a published methodology per regulation, licensed redistribution classes, and an explainability contract — that every downstream system can call instead of re-building.

### 1.2 Product

**GreenLang Factors** is the first product shipping in FY27 from the Climate OS v3 four-layer architecture. It is Layer 1 of the substrate. Every other GreenLang product in FY27 (CBAM, Comply/CSRD, Scope Engine) and every future product (SupplierOS, PCF Studio, DPP Hub, FinanceOS, AgriLandOS) calls Factors first. Factors is also sold standalone to three external buyer personas: developers (open-core + usage pricing), consultants and boutique ESG firms (per-client override vaults, audit bundles), and platform / OEM partners (white-label embed, signed responses).

Product pillars (1:1 with existing code):
- **Factor Registry** — ~25 600 lines of YAML + SQLite catalog across EPA, DESNZ, eGRID, IEA, IPCC, Green-e, TCR, CBAM, AIB, India-CEA, Japan-METI, Australia-NGA, EC3-EPD, PACT, PCAF, LSR, freight lanes, waste treatment.
- **Resolution Engine** — 7-step cascade (tenant override → supplier → facility → tariff/grid → country/sector → method-pack default → global default), with tiebreak, unit graph, explainability.
- **Method-Pack Library** — 7 profiles (corporate, electricity, freight, eu_policy, land_removals, product_carbon, finance_proxy) registered in `greenlang/factors/method_packs/`.
- **Source Catalog** — source registry with license class, watch cadence, legal sign-off, jurisdiction, dataset version, validity period.
- **Developer Platform** — REST API, Python + TypeScript SDKs, CLI, developer portal, Factor Explorer UI, Operator Console.
- **Commercial Layer** — tiered auth (Community / Developer Pro / Consulting / Enterprise), 8 Premium Data Pack SKUs, Stripe metering, signed receipts, tenant overlays, per-pack entitlements, SLA tracker.

### 1.3 Positioning line

> **GreenLang Factors is the canonical climate reference system every downstream product calls first.**

### 1.4 Open-core motion

| Layer | License | What it contains |
|---|---|---|
| Open core | Apache-2.0 | Catalog schema, Canonical Factor Record v1.0, Resolution Engine, 7 method packs, SDKs, CLI, ~60% of the catalog sourced from Open-license jurisdictions (EPA, DESNZ, eGRID, IPCC Tier-1, Green-e, TCR, India-CEA, Japan-METI, Australia-NGA, waste_treatment, IPCC defaults). |
| Commercial SaaS | Proprietary | Hosted API (auth, rate limits, ETags, signed receipts), Operator Console, developer portal analytics, Tenant Overlay, Approval Workflows, Impact Simulator, Audit Bundle export. |
| Premium Data Packs | Licensed or OEM | Electricity Premium, Freight Premium, Product-Carbon/LCI Premium (ecoinvent-tier), EPD Premium (EC3-tier), Agrifood Premium, Finance Proxy Premium, CBAM/EU-Policy Premium, Land Premium. |
| Customer-private | Not redistributable | Tenant overlays, private factor packs, internal supplier factors. |

### 1.5 FY27 target

| Metric | FY27 target |
|---|---|
| ARR (end of year) | $0.7M |
| Recognized revenue | $0.3M |
| Paying logos | 8 |
| Developer signups (free + paid) | 4 000 |
| Community contributors | 80 |
| Certified factors (public dashboard) | 1 500 |
| Preview factors (public dashboard) | 2 000 |
| Connector-only factors (public dashboard) | 3 000+ |
| Top-1 resolve accuracy on gold set | ≥ 85% |
| p95 `/factors/resolve` latency | < 200 ms |
| API availability | 99.9% |

---

## 2. User stories (4 personas × ~10 stories)

### 2.1 Developer persona
**Profile:** Single engineer at a climate startup, or the in-house software engineer at a climate consultancy. Builds internal tools, CBAM calculators, Scope 1/2 dashboards, or PCF prototypes. Wants deterministic, versioned, explainable factors accessible via API and SDK. Not an expert in methodology and will not read 30-page standards documents. Buys Developer Pro at $49–$499/mo on self-service.

| # | Story | Acceptance criteria |
|---|---|---|
| D-1 | As a developer, I want to resolve a factor for `(activity="diesel combustion", jurisdiction="US", date="2026-06-01", method_profile="corporate_scope1")` in one API call so that I don't hand-pick from a list. | `GET /v1/factors/resolve` returns one `ResolvedFactor` with `factor_id`, `factor_version`, `edition_id`, gas vector + derived CO2e, `fallback_rank` (1–7), `explanation`, `signed_receipt`. p95 < 200 ms. Request is idempotent per `(request_hash, edition_id)`. |
| D-2 | As a developer, I want to pin my application to a specific factor edition so that my numbers are reproducible when the catalog updates next quarter. | Every response includes `X-GreenLang-Edition: <edition_id>`. Passing the same header on request locks resolution to that edition. Edition IDs are immutable and signed by `edition_manifest.py`. Deprecated editions still resolve with a `Warning` header. |
| D-3 | As a developer, I want a Python SDK that mirrors the REST API so that I don't hand-roll HTTP. | `pip install greenlang-factors`; `from greenlang.factors import Client; client.resolve(...)`. Types match the Canonical Factor Record. SDK lives in `greenlang/factors/sdk/python/`. |
| D-4 | As a developer, I want a TypeScript SDK for the same so that my Next.js dashboard can call Factors. | `npm install @greenlang/factors`; lives in `greenlang/factors/sdk/ts/`. Includes browser + Node builds. |
| D-5 | As a developer, I want to see why the engine picked this factor over the alternatives so that I can explain the number to my auditor. | `GET /v1/factors/{id}/explain?request_id=...` returns the 7-step cascade log, alternate candidates considered, tiebreak reasons, unit-graph conversions, and the authoring methodology note. |
| D-6 | As a developer, I want to page through the catalog for a faceted UI I'm building. | `POST /v1/factors/search` accepts filters (`jurisdiction`, `activity_category`, `method_profile`, `source_id`, `factor_status`) + cursor pagination + sort. Returns `results`, `next_cursor`, `total_count`. |
| D-7 | As a developer, I want to batch-resolve 10 000 activity rows in one round-trip so that my quarterly inventory doesn't hit 10 000 HTTP calls. | `POST /v1/factors/batch` accepts ≤ 10 000 activities, returns one row per input with the same resolution object as D-1. Async variant returns a `job_id` with `GET /v1/factors/batch/{job_id}` polling. |
| D-8 | As a developer, I want a free Community tier so that I can prototype before I ask my CFO for budget. | Community tier: 1 000 API calls/month, Certified factors only, no Premium Data Packs, attribution required. Enforced in `tier_enforcement.py`. |
| D-9 | As a developer, I want Prometheus-style metrics for my own traffic so that I can monitor my quota burn. | Every response includes `X-GreenLang-Quota-Remaining`, `X-GreenLang-Quota-Reset`. Developer portal exposes a per-key usage chart. |
| D-10 | As a developer, I want to subscribe to webhooks when a factor I pinned is deprecated so that I find out before my next audit. | `POST /v1/webhooks/subscribe` with event types `factor.deprecated`, `factor.updated`, `edition.published`, etc. HMAC-signed delivery, exponential backoff. |

### 2.2 Consultant persona
**Profile:** Partner at an ESG boutique (5–50 consultants) or the climate practice lead at a Big-4. Delivers carbon inventories, CSRD disclosures, CBAM filings, SBTi targets to mid-market and enterprise clients. Needs per-client override vaults, audit bundles, methodology notes, workbook export, and the ability to hand an auditor a one-click proof package. Buys Consulting tier at ~$25k–$75k/yr plus per-client usage.

| # | Story | Acceptance criteria |
|---|---|---|
| C-1 | As a consultant, I want a separate overlay vault per client so that Acme Corp's supplier factors never leak into Globex's inventory. | `tenant_overlay.py` supports multi-tenant isolation: a consulting account can create N `sub_tenants`, each with an independent overlay table. Resolution cascade step 1 reads `sub_tenant_id` first. Verified by a cross-tenant read test. |
| C-2 | As a consultant, I want to export a signed audit bundle for a single client run that includes every factor used, source PDF URI, parser log, reviewer decision, and the exact 7-step cascade for each resolution. | `POST /v1/audit-bundle` with filter `{sub_tenant_id, date_range, case_id}` returns a zipped bundle (JSON + PDFs + signed SHA-256 chain). `build_audit_bundle()` in `api_endpoints.py` already generates per-factor bundles — v1 adds a per-run aggregator. |
| C-3 | As a consultant, I want to export the resolved factors for a client inventory as an Excel workbook so that I can hand it to a client CFO. | `GET /v1/factors/export?format=xlsx&...` returns an Excel with one row per resolution, columns for factor_id, version, source, license class, fallback_rank, explanation. Enforces tier and pack entitlements. |
| C-4 | As a consultant, I want to see the methodology note that applies to a given method_profile so that I can paste it verbatim into my report. | `GET /v1/method-packs/{profile}` returns the full methodology text, the regulatory references (e.g., "GHG Protocol Corporate Standard §4"), and the selection rules used by the engine. |
| C-5 | As a consultant, I want to diff two editions for a client so that I can show them exactly what changed between last quarter's inventory and this quarter's. | `GET /v1/factors/{id}/diff?from=<edition>&to=<edition>` returns field-by-field diffs. `GET /v1/editions/diff?from=...&to=...` returns the catalog-level diff: added / removed / changed factor_ids. Implemented in `service.py :: compare_editions()`. |
| C-6 | As a consultant, I want to override a single factor for a single client with their supplier-specific value, date-bound, with a note and a file attachment. | UI + REST: `POST /v1/overlays` with `factor_id`, `override_value`, `valid_from`, `valid_to`, `source`, `notes`, `attachment_url`. Appears as fallback_rank=1 in that tenant's resolutions. Versioned; no overwrite. |
| C-7 | As a consultant, I want to batch-upload a client's activity file (CSV/XLSX) and get resolved factors back with explanations so that I can finish a CBAM filing in hours not weeks. | Operator Console upload widget → staging job → `POST /v1/factors/batch` → downloadable results + audit bundle. |
| C-8 | As a consultant, I want to be able to show a client the Certified vs Preview label on every factor so that I can flag which numbers need additional review. | Every `ResolvedFactor` includes `factor_status: certified | preview | connector_only | deprecated`. Factor Explorer UI uses three-label color coding. |
| C-9 | As a consultant, I want SSO with my firm's IdP (Okta, Azure AD) so that I don't manage per-consultant credentials. | SAML 2.0 + OIDC supported via existing `greenlang.integration.api.dependencies`. Consulting tier includes this. |
| C-10 | As a consultant, I want a consulting-branded Factor Explorer URL I can share with clients so that they see my firm, not GreenLang. | Consulting tier includes subdomain white-label (`factors.acmeclimate.greenlang.io`) with custom logo + colors. Mechanism: `tier_enforcement.py` + frontend tenant theme. |

### 2.3 Platform persona
**Profile:** Product team at an ERP vendor (SAP, Oracle), a carbon-accounting platform (Persefoni, Watershed), a PCF tool, or a supplier-engagement SaaS. Wants to **embed** Factors in their product, with white-label branding, signed responses (so their customers' auditors can verify independently), redistribution controls, and sub-tenant entitlements. Buys Platform/OEM at ~$150k–$500k/yr + volume.

| # | Story | Acceptance criteria |
|---|---|---|
| P-1 | As a platform PM, I want to embed Factors API behind my own API gateway with my domain and my branding so that my customers see one product, not two. | OEM mode: white-label API domain (`api.yourvendor.com → api.greenlang.io/v1/...`), custom HTTP header `X-Powered-By: YourVendor`, custom error messages. Delivered via `middleware/` + deployment config. |
| P-2 | As a platform PM, I want every response to carry a GreenLang signature so that my customers' auditors can verify the response independently of me. | `signing.py` already emits signed receipts (Ed25519). v1 exposes the public key at `GET /v1/.well-known/greenlang-signing-key`. Verification SDK + docs shipped. |
| P-3 | As a platform PM, I want to provision sub-tenants programmatically so that each of my enterprise customers has isolated usage, overrides, and entitlements. | `POST /v1/platform/sub-tenants` creates a sub-tenant with its own API keys, overlay vault, pack entitlements, and quota. Parent account sees aggregate usage; sub-tenants see only their own. |
| P-4 | As a platform PM, I want to control which Premium Data Packs are redistributable vs internal-only vs forbidden for each of my sub-tenants. | `entitlements.py :: OEMRights` tri-state (`forbidden | internal_only | redistributable`) persisted per `(sub_tenant_id, pack_sku)`. Resolution engine checks before returning. |
| P-5 | As a platform PM, I want volume-tiered pricing so that my unit economics scale past 10M calls/mo. | Pricing page publishes tiers: 0–1M / 1–10M / 10–100M / 100M+ calls/mo with per-call rates. Stripe metering via `billing/aggregator.py` + `billing/usage_sink.py`. |
| P-6 | As a platform PM, I want a 99.95% SLA with financial credits so that I can put my reputation on this. | Enterprise/OEM tier: 99.95% monthly uptime SLO, 10% credit on miss, 25% credit on major miss. `observability/sla.py` + `ga/sla_tracker.py` produce the monthly credit report. |
| P-7 | As a platform PM, I want to embed the Factor Explorer as an iframe inside my product so that my customers don't leave. | Factor Explorer ships a `/embed` route with `postMessage` bridge, iframe-safe CSP, and theme props. OEM-only. |
| P-8 | As a platform PM, I want a webhooks channel for source-watch events so that I can proactively notify my customers when a factor they care about changes. | `/v1/webhooks/subscribe` with event filter `{tenant_scope: sub_tenants}`. Delivered with sub_tenant_id so my router can fan out. |
| P-9 | As a platform PM, I want a regional deployment option (EU data residency) so that my EU customers don't trigger cross-border issues. | Enterprise/OEM tier: deploy-in-region option (EU-FRA, US-IAD, IN-HYD). DB pinned to region; metering rolled up cross-region. |
| P-10 | As a platform PM, I want a staging tier mirroring production catalog so that I can integration-test pack rollouts before promoting. | `X-GreenLang-Environment: staging` header + separate key prefix (`gl_fac_stg_...`). Staging refreshes within 24h of production edition publish. |

### 2.4 Enterprise methodology lead persona
**Profile:** Climate controller or senior methodology engineer at a Fortune-500 reporter (e.g., a CSRD-obligated Indian cement exporter, a CBAM-obligated steel manufacturer). Runs an internal compliance or decarbonization program. Needs approval workflows, segregation of duties, private factor packs, scenario testing, impact diffing, deprecation notices, policy-linked selectors, and an answer to "which number goes in the filing?" that clears Big-4 audit. Buys Enterprise at ~$100k–$300k ACV plus private-registry and OEM addons.

| # | Story | Acceptance criteria |
|---|---|---|
| E-1 | As a methodology lead, I want a 4-eyes approval workflow before a new internal factor is published to my company's private registry so that a single engineer cannot ship a number into the CSRD filing. | Override creation → review queue → 2+ approvers with separate roles → publish. Implemented via `quality/review_queue.py` + `quality/release_signoff.py` + `quality/review_workflow.py`. Dual-control enforcement verified. |
| E-2 | As a methodology lead, I want segregation of duties between "author" and "approver" so that the same person cannot both write and publish a factor. | RBAC: `overlay_author`, `overlay_approver`, `overlay_publisher` roles. A user may hold at most one of `author` and `approver` for a given factor. Enforced in the approval workflow. |
| E-3 | As a methodology lead, I want a private factor pack scoped to my company that every downstream product (CBAM, CSRD, Scope Engine) picks up automatically. | Private pack = Tenant Overlay + pack manifest. `resolution/engine.py` step-1 reads it automatically. Audit log: every private-pack write is append-only with actor + reason. |
| E-4 | As a methodology lead, I want to simulate the emissions impact of switching from factor edition N to N+1 before I approve the switch for the company. | `quality/impact_simulator.py` takes `(tenant_id, from_edition, to_edition, activity_sample)` → returns delta totals per scope / site / product. Exposed at `POST /v1/editions/simulate`. |
| E-5 | As a methodology lead, I want to be notified 90 days before a factor I depend on is deprecated so that I can re-approve the alternative. | Webhooks `factor.deprecated` with `deprecation_date`. Factor Explorer + Operator Console show deprecation countdown. Source-watch pipeline (`watch/release_orchestrator.py`) emits the event. |
| E-6 | As a methodology lead, I want policy-linked selectors — "for CBAM, use the method_pack that CBAM regulator approves" — not raw factor IDs in our calculations, so that a regulation change automatically re-routes the selection. | Calculation code calls `client.resolve(method_profile="eu_cbam", activity=...)`, NOT `client.get_factor(factor_id="...")`. The method pack (`method_packs/eu_policy.py`) maps CBAM rules to factor selectors. Enforced by CTO non-negotiable #6 — `validate_non_negotiables()` rejects records that lack `method_profile` when used in a regulated framework. |
| E-7 | As a methodology lead, I want a single "where was this number from" panel for any emissions figure in any downstream product so that I can trace my CSRD E1 disclosure to a factor ID, source URI, parser log, reviewer decision. | `GET /v1/factors/{id}/explain` returns the full chain: source_id + publication + citation + parser run id + QA decision + approvers. Audit bundle export (`POST /v1/audit-bundle`) packages it. |
| E-8 | As a methodology lead, I want SSO + SCIM + VPC-peering so that my security team will accept GreenLang. | Enterprise tier: SAML SSO, SCIM 2.0 user provisioning, VPC peering option, customer-managed keys (CMK) for overlay encryption at rest. |
| E-9 | As a methodology lead, I want full CRUD logs on every override, with immutable history so that when my auditor asks "who changed this, when, why" I have a single screen to answer. | `tenant_overlay.py` already writes append-only overlay records. v1 adds an `audit log` view in Operator Console with filter by `actor`, `factor_id`, `date_range`. |
| E-10 | As a methodology lead, I want to gate which company sites / business units can read which Premium Data Packs so that licensed data doesn't leak beyond the team that paid for it. | `entitlements.py` per-pack entitlements scoped by `sub_tenant_id` and `user_group`. Resolution filters candidates before returning. |
| E-11 | As a methodology lead, I want the system to refuse to return a factor if its validity window does not cover the requested date, instead of silently falling back. | Resolution engine respects `valid_from`/`valid_to` hard boundaries before fallback logic kicks in. If no valid factor exists in the 7-step cascade, returns `404 NO_VALID_FACTOR` with the explanation of what was considered. |
| E-12 | As a methodology lead, I want the engine to refuse to mix licensed classes in one response so that our licensed LCI data never ends up in a public export. | `enforce_license_class_homogeneity()` already enforces CTO non-negotiable #4 at response time. v1 exposes it in the public API with an explicit `redistribution_class` field on every record. |

**Total: 42 user stories across 4 personas.**

---

## 3. Feature list — one row per feature

Status key: **BUILT** = code is in the repo and usable today, **PARTIAL** = code exists but needs packaging or hardening, **MISSING** = must be built for v1. The "Owner" column is a role (PM = product, FE = frontend, BE = backend, DE = data engineering, ML = matching/ML, Sec = security, GTM = go-to-market, Ops = DevOps/SRE).

### 3.1 Core registry and resolution

| # | Feature | Module path (under `greenlang/factors/`) | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-001 | Canonical Factor Record v1.0 schema | `../data/canonical_v2.py` + `../data/emission_factor_record.py` | BUILT | CTO non-negotiables #1, #2, #4, #5, #6 enforced by `validate_non_negotiables()`. New fields ship optional. | BE |
| F-002 | 7-step Resolution Engine | `resolution/engine.py` | BUILT | All 7 steps wired; tests cover each step + tiebreak + unit-graph conversion. p95 < 200 ms against SQLite catalog. | BE |
| F-003 | Resolution request / result schema | `resolution/request.py`, `resolution/result.py` | BUILT | ResolvedFactor carries `fallback_rank`, `explanation`, `gas_vectors`, `alternates`. | BE |
| F-004 | Tiebreak rules | `resolution/tiebreak.py` | BUILT | Specificity → verification → recency → geographical fit → methodology distance. | BE |
| F-005 | Method-pack library (7 profiles) | `method_packs/{corporate,electricity,freight,eu_policy,land_removals,product_carbon,finance_proxy}.py` | BUILT | All 7 packs registered via `method_packs/registry.py`; each exposes `SelectionRule` + methodology note. | BE |
| F-006 | Method-pack base class | `method_packs/base.py` | BUILT | Pack contract: `applies_to()`, `select_rule()`, `methodology_notes()`. | BE |
| F-007 | LCA variants (product carbon) | `method_packs/product_lca_variants.py` | BUILT | ISO 14067, GHG Product Standard, PACT pathways. | BE |
| F-008 | Unit graph | `ontology/unit_graph.py`, `ontology/units.py` | BUILT | Deterministic conversions with error on unknown path. | BE |
| F-009 | Chemistry ontology | `ontology/chemistry.py` | BUILT | Fuel properties, carbon content, oxidation factors. | BE |
| F-010 | GWP sets | `ontology/gwp_sets.py` | BUILT | AR6, AR5, AR4, SAR — 100yr + 20yr. Derive CO2e at response time. | BE |
| F-011 | Heating values | `ontology/heating_values.py` | BUILT | Per-fuel LHV/HHV with jurisdiction overrides. | BE |
| F-012 | Geography ontology | `ontology/geography.py` | BUILT | ISO 3166-1/2 + grid sub-regions (eGRID, NERC, CEA). | BE |
| F-013 | Methodology ontology | `ontology/methodology.py` | BUILT | IPCC tiers, corporate standard refs, ISO references. | BE |
| F-014 | Edition manifest | `edition_manifest.py` | BUILT | Immutable, signed, includes factor_count + per-source hashes + deprecations + policy_rule_refs. | BE |
| F-015 | Source registry | `source_registry.py` + `data/source_registry.yaml` | BUILT | 19 sources live; rights-matrix model covers license_class + redistribution + cadence + legal sign-off. | DE |
| F-016 | Tenant overlay (customer-private) | `tenant_overlay.py` | BUILT | Multi-tenant isolation; date-bound; append-only; wired as cascade step 1. | BE |
| F-017 | Non-negotiable validators | `data/canonical_v2.py :: validate_non_negotiables()` + `enforce_license_class_homogeneity()` | BUILT | Ran at repository write time + response build time. CI linter hooked. | BE |

### 3.2 Data ingestion and source management

| # | Feature | Module path | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-018 | Fetcher harness (pull from upstream) | `ingestion/fetchers.py`, `ingestion/tabular_fetchers.py` | BUILT | 19 sources wired; robots.txt respected; ETags stored. | DE |
| F-019 | Parser harness (normalize to canonical) | `ingestion/parser_harness.py` + `ingestion/parsers/*.py` | BUILT | 19 parsers: EPA, DESNZ, eGRID, IPCC, Green-e, TCR, GHG Protocol, CBAM, AIB, India-CEA, Japan-METI, Australia-NGA, EC3-EPD, PACT, PCAF, LSR, freight-lanes, waste, Green-e-residual. | DE |
| F-020 | Bulk ingest orchestrator | `ingestion/bulk_ingest.py` | BUILT | Parallel source ingest, dead-letter queue, retry with jitter. | DE |
| F-021 | Source artifact storage | `ingestion/artifacts.py` | BUILT | Content-addressed; SHA-256 verified; S3-backed in prod. | DE |
| F-022 | Ingestion metadata DB | `ingestion/sqlite_metadata.py` | BUILT | Tracks every fetch + parse + QA run. | DE |
| F-023 | Synthetic-data generator (for tests) | `ingestion/synthetic_data.py` | BUILT | Used in gold-eval + fixtures. | DE |
| F-024 | Source-watch scheduler | `watch/scheduler.py`, `watch/source_watch.py` | BUILT | Per-source cadence — quarterly for EPA/DESNZ/eGRID, monthly for IEA, ad-hoc for regulators. | DE |
| F-025 | Change detector + classifier | `watch/change_detector.py`, `watch/change_classification.py` | BUILT | Classifies as editorial / factor-value / methodology / breaking. | DE |
| F-026 | Doc diff | `watch/doc_diff.py` | BUILT | Source-side changelog diff + PDF text diff where available. | DE |
| F-027 | Release orchestrator | `watch/release_orchestrator.py` | BUILT | Releases new edition only after QA + approval gates. | DE |
| F-028 | Rollback pipeline | `watch/rollback_edition.py`, `watch/rollback_cli.py` | BUILT | `gl factors rollback --edition <id>` reverts; existing resolutions pinned to prior edition stay safe. | DE |
| F-029 | Changelog drafter | `watch/changelog_draft.py`, `watch/cross_edition_changelog.py` | BUILT | Generates human-readable release notes per edition. | DE |
| F-030 | Regulatory-event watch | `watch/regulatory_events.py` | BUILT | CBAM, CSRD, SB 253 notification hooks. | DE |
| F-031 | Watch status API | `watch/status_api.py` | BUILT | `GET /v1/sources/watch/status` returns per-source health + next run. | BE |

### 3.3 Matching and mapping

| # | Feature | Module path | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-032 | Semantic matching pipeline | `matching/pipeline.py` | BUILT | Embedding + BM25 hybrid → LLM rerank → deterministic filter. | ML |
| F-033 | Embedding model adapter | `matching/embedding.py` | BUILT | MPNet default; MiniLM fallback; configurable. | ML |
| F-034 | pgvector index | `matching/pgvector_index.py` | BUILT | HNSW; per-tenant namespace; updated on edition publish. | ML |
| F-035 | Semantic index abstraction | `matching/semantic_index.py` | BUILT | SQLite / FAISS / pgvector pluggable. | ML |
| F-036 | LLM reranker | `matching/llm_rerank.py` | BUILT | Gated behind feature flag + budget guard; returns explanation. | ML |
| F-037 | Match-suggestion agent | `matching/suggestion_agent.py` | BUILT | Powers "did you mean" in Operator Console + API. | ML |
| F-038 | Matching evaluation harness | `matching/evaluation.py` | BUILT | Computes top-1 / top-5 / MRR. | ML |
| F-039 | Gold eval set (300–500 activity descriptions) | `matching/evaluation.py` fixtures + new test assets under `tests/factors/gold/` | PARTIAL | Harness exists; curated gold set with auditor sign-off is MISSING. v1 ships 500 items across 7 method packs. | ML + DE |
| F-040 | Mapping library (crosswalks) | `mapping/{base,biogenic_sources,circular_economy,classifications,electricity_market,fuels,industry_codes,land_use,materials,regulatory_frameworks,spend,transport,waste}.py` | BUILT | 13 mapping modules: NAICS→IPCC, CN→CBAM, ISIC→method pack, spend→factor family, transport mode normalization. | BE |
| F-041 | Mapping CLI + API | Integrated in `cli.py` + `api_endpoints.py` | PARTIAL | Library is BUILT; REST surface at `GET /v1/mappings/{classification_system}` is MISSING for v1. | BE |

### 3.4 Quality, QA, and governance

| # | Feature | Module path | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-042 | Validators (schema, unit, range, chemistry) | `quality/validators.py` | BUILT | Runs in ingestion + CI. | DE |
| F-043 | Dedupe engine | `quality/dedup_engine.py`, `dedupe_rules.py` | BUILT | Near-duplicate detection across sources. | DE |
| F-044 | Cross-source consensus | `quality/cross_source.py`, `quality/consensus.py` | BUILT | Compares EPA vs DESNZ vs IPCC for the same factor family; flags drift. | DE |
| F-045 | License scanner | `quality/license_scanner.py` | BUILT | Ensures no OEM-redistributable factor leaks into Open. | Sec |
| F-046 | Review queue | `quality/review_queue.py` | BUILT | Inbox for reviewers; filter by priority + source. | BE |
| F-047 | Review workflow | `quality/review_workflow.py` | BUILT | Two-reviewer pattern; status: pending → approved → published. | BE |
| F-048 | Release sign-off | `quality/release_signoff.py` | BUILT | Gate before edition publish; records actor + reason. | BE |
| F-049 | Audit export (per-factor) | `quality/audit_export.py` + `api_endpoints.py :: build_audit_bundle()` | BUILT | Per-factor bundle. v1 adds per-run aggregation (see F-068). | BE |
| F-050 | Promotion (Preview→Certified) | `quality/promotion.py` | BUILT | Moves factors across three-label coverage states. | BE |
| F-051 | Batch QA runner | `quality/batch_qa.py` | BUILT | Runs validators + dedupe + cross-source on a candidate edition. | DE |
| F-052 | Impact simulator | `quality/impact_simulator.py` | BUILT | Simulates delta before adoption. Exposed in Operator Console. | BE |
| F-053 | Rollback | `quality/rollback.py` | BUILT | Reverts a single factor's promotion. | BE |
| F-054 | Escalation | `quality/escalation.py` | BUILT | Surfaces stalled reviews to the QA lead. | BE |
| F-055 | SLA tracker | `quality/sla.py` + `ga/sla_tracker.py` | BUILT | QA-throughput + resolution-latency SLA reports. | BE |
| F-056 | Versioning (no overwrite) | `quality/versioning.py` + `data/canonical_v2.py :: ChangeLogEntry` | BUILT | Every factor change creates a new `factor_version`; CTO non-negotiable #2. | BE |
| F-057 | Approval gate | `approval_gate.py` | BUILT | CI / pre-publish gate; refuses to publish if any QA errors are open. | BE |
| F-058 | Regulatory tagger | `regulatory_tagger.py` | BUILT | Tags factors with `CBAM`, `CSRD`, `SB253`, `PACT`, `EUDR` applicability. | DE |
| F-059 | Policy mapping | `policy_mapping.py` | PARTIAL | Maps method profiles to regulations; v1 unification with Policy Graph stub is MISSING (separate product, but stub lands here). | BE |

### 3.5 API, auth, billing, middleware

| # | Feature | Module path | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-060 | FastAPI routes (`/factors/*`, `/sources/*`, etc.) | `api_endpoints.py` + route module (not yet wired to FastAPI router) | PARTIAL | Pure-logic functions for audit-bundle / bulk-export / diff / search-v2 exist. FastAPI router wiring is MISSING for v1. | BE |
| F-061 | Auth (JWT + API key) | `api_auth.py` | BUILT | Dual auth supported. SSO/OIDC via `greenlang.integration.api.dependencies`. | Sec |
| F-062 | Tier enforcement (Community/Pro/Consulting/Enterprise/Internal) | `tier_enforcement.py` | BUILT | Per-tier visibility flags and export limits. | Sec |
| F-063 | Pack entitlements (8 Premium SKUs) | `entitlements.py` | BUILT | Per-tenant, per-pack allow/deny; OEM rights tri-state. | BE |
| F-064 | Rate limiter | `middleware/rate_limiter.py` | BUILT | Token-bucket per key; tier-driven quotas. | Sec |
| F-065 | Auth + metering middleware | `middleware/auth_metering.py` | BUILT | Every request passes through auth → metering → router. | Sec |
| F-066 | Signed receipts | `signing.py` | BUILT | SHA-256 HMAC + Ed25519; receipt ships in header or body. | Sec |
| F-067 | Metering | `metering.py` + `billing/metering.py` | BUILT | Per-call, per-batch, per-export, per-pack metering events. | BE |
| F-068 | Audit bundle (per-run) | `api_endpoints.py :: build_audit_bundle()` + new per-run aggregator | PARTIAL | v1 adds the run-level aggregator + ZIP export endpoint `POST /v1/audit-bundle`. | BE |
| F-069 | Bulk export (streaming) | `api_endpoints.py :: bulk_export_factors()` | BUILT | CSV/JSON/XLSX streaming with tier-aware row limits. | BE |
| F-070 | Factor diff (field-by-field) | `api_endpoints.py :: diff_factors()` (existing) + `service.py :: compare_editions()` | BUILT | `GET /v1/factors/{id}/diff?from=<e>&to=<e>`. | BE |
| F-071 | Search v2 (POST body + sort + pagination) | `api_endpoints.py` + search_cache | BUILT | Facet filter + cursor pagination + ETag. | BE |
| F-072 | ETag / Cache-Control | `api_endpoints.py` helpers + `search_cache.py` | BUILT | 304 responses on matching `If-None-Match`. | BE |
| F-073 | Index manager (search) | `index_manager.py` | BUILT | Maintains pgvector + full-text indexes on edition publish. | BE |
| F-074 | Redis cache | `cache_redis.py` | BUILT | Hot-path cache for resolve + search. | BE |
| F-075 | Performance optimizer | `performance_optimizer.py` | BUILT | Query plan + cache-hit tuning. | BE |
| F-076 | OpenAPI spec | `api_endpoints.py` schemas + new `openapi.yaml` | MISSING | v1 publishes a frozen OpenAPI 3.1 file at `/v1/openapi.json`. | BE + PM |
| F-077 | Webhooks (11 event types) | `webhooks.py` | BUILT | HMAC-signed; retry with backoff; 11 event types already defined. | BE |
| F-078 | Batch jobs | `batch_jobs.py` | BUILT | Async large batches; polling endpoint. | BE |
| F-079 | Backfill | `backfill.py` | BUILT | Fills historical factors into new editions. | DE |
| F-080 | Notifications (in-product) | `notifications/` | BUILT | Used by Operator Console. | BE |

### 3.6 Billing + GA scaffolding

| # | Feature | Module path | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-081 | Stripe provider | `billing/stripe_provider.py` | BUILT | Subscription + metered products wired. | GTM + BE |
| F-082 | Usage aggregator | `billing/aggregator.py` | BUILT | Rolls metering events to Stripe usage records. | BE |
| F-083 | Usage sink | `billing/usage_sink.py` | BUILT | Persists events; replay-safe. | BE |
| F-084 | Webhook handler (Stripe) | `billing/webhook_handler.py` | BUILT | `checkout.session.completed` → activate entitlement. | BE |
| F-085 | SKU catalog (4 tiers × 8 packs + OEM) | `billing/` + new `ga/sku_catalog.py` | MISSING | v1 publishes a Stripe product list; see §7. | GTM |
| F-086 | GA billing integration | `ga/billing.py` | BUILT | Ties metering → Stripe → customer invoices. | BE |
| F-087 | GA readiness checks | `ga/readiness.py` | BUILT | Pre-launch gate: auth + metering + SLA + receipts all green. | Ops |

### 3.7 Observability, SDKs, CLI

| # | Feature | Module path | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-088 | Prometheus metrics | `observability/prometheus.py`, `observability/prometheus_exporter.py` | BUILT | Per-endpoint + per-tenant counters; histograms for latency. | Ops |
| F-089 | Health endpoints | `observability/health.py` | BUILT | `/v1/health`, `/v1/ready`. | Ops |
| F-090 | SLA observability | `observability/sla.py` | BUILT | Monthly SLO + credit report. | Ops |
| F-091 | Python SDK | `sdk/python/` | BUILT | `pip install greenlang-factors`; mirrors REST. | BE |
| F-092 | TypeScript SDK | `sdk/ts/` | BUILT | `npm install @greenlang/factors`. | FE |
| F-093 | CLI (`gl factors`) | `cli.py` + `__main__.py` | BUILT | `inventory`, `manifest`, `ingest-builtin`, `watch dry-run`, `validate-registry`, `rollback`. v1 adds `resolve`, `explain`, `audit-bundle`. | BE |
| F-094 | Catalog repository (SQLite) | `catalog_repository.py` | BUILT | In-memory + SQLite variants. | BE |
| F-095 | Catalog repository (Postgres) | `catalog_repository_pg.py` | BUILT | Used in hosted prod; keeps parity with SQLite surface. | BE |
| F-096 | ETL | `etl/` | BUILT | Ingest-to-catalog pipeline. | DE |
| F-097 | Bounded contexts (DDD partitioning) | `bounded_contexts/` | BUILT | Ingestion / Catalog / Watch / QA are isolated modules. | BE |
| F-098 | Security hardening | `security/` | BUILT | Key rotation, input sanitization, audit log. | Sec |
| F-099 | Artifacts store | `artifacts/` | BUILT | Signed artifact store for editions. | DE |
| F-100 | Data package | `data/` (within factors) | BUILT | SQLite seed + source_registry.yaml + fixtures. | DE |
| F-101 | Inventory / coverage matrix | `inventory.py` | BUILT | `gl factors inventory` emits the three-label counts. | BE |

### 3.8 Pilot + onboarding

| # | Feature | Module path | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-102 | Pilot provisioner | `pilot/provisioner.py` | BUILT | One-command pilot tenant setup. | Ops |
| F-103 | Pilot registry | `pilot/registry.py` | BUILT | Tracks 5–8 FY27 design partners. | GTM |
| F-104 | Pilot telemetry | `pilot/telemetry.py` | BUILT | Pilot-scoped metrics + feedback loop. | Ops |
| F-105 | Pilot feedback capture | `pilot/feedback.py` | BUILT | Structured feedback → product backlog. | PM |
| F-106 | Onboarding health check | `onboarding/health_check.py` | BUILT | First-run sanity check. | Ops |
| F-107 | Onboarding partner setup | `onboarding/partner_setup.py` | BUILT | Automates OEM partner onboarding. | Ops |
| F-108 | Onboarding sample queries | `onboarding/sample_queries.py` | BUILT | Ships with developer portal. | PM |

### 3.9 UI surfaces (frontend; all MISSING)

| # | Feature | Module path | State | v1 acceptance | Owner |
|---|---|---|---|---|---|
| F-109 | Factor Explorer (public) | new `web/factor-explorer/` | MISSING | Search, detail, explain, source catalog, method-pack browser, three-label dashboard. See §8.1. | FE |
| F-110 | Operator Console (internal) | new `web/operator-console/` | MISSING | Ingestion, mapping workbench, QA dashboard, diff viewer, approval workflow, override manager, impact simulator. See §8.2. | FE |
| F-111 | Developer portal | new `web/dev-portal/` | MISSING | Docs, API keys, usage dashboard, webhook config, SDK downloads, Stripe-managed billing. | FE + PM |
| F-112 | Pricing page | new `web/marketing/pricing/` | MISSING | Publishes four tiers + eight Premium packs + enterprise add-ons. | GTM |

### 3.10 MISSING items summary

For v1, the MISSING work (everything else is BUILT or PARTIAL → hardening):

- F-039 Gold eval set (300–500 items) — ML + DE, week 3–4.
- F-041 Mapping REST surface — BE, week 5.
- F-059 Policy mapping stub (Policy Graph v0) — BE, week 7.
- F-060 FastAPI router wiring (route modules exist; glue to actual FastAPI app is partial) — BE, week 3.
- F-068 Per-run audit-bundle aggregator — BE, week 5.
- F-076 Frozen OpenAPI 3.1 spec — BE + PM, week 2.
- F-085 SKU catalog (Stripe) — GTM, week 7.
- F-109 Factor Explorer UI — FE, weeks 5–6.
- F-110 Operator Console UI — FE, weeks 9–10.
- F-111 Developer portal — FE + PM, week 5.
- F-112 Pricing page — GTM, week 7.

Everything else is BUILT and needs hardening only.

---

## 4. Data model

All schemas here are **v1.0 frozen** for the FY27 launch. Any additions must land as optional fields (backward compatible). Removal is forbidden — use deprecation.

### 4.1 Canonical Factor Record v1.0

Source of truth: `greenlang/data/emission_factor_record.py` (baseline fields) + `greenlang/data/canonical_v2.py` (CTO extensions).

```yaml
# Canonical Factor Record v1.0
factor_id:            string     # globally unique (e.g., "EF:US:diesel:2026Q1:v3")
factor_family:        enum       # FactorFamily (15 values): emissions, energy_conversion,
                                 # carbon_content, oxidation, heating_value, density,
                                 # refrigerant_gwp, grid_intensity, residual_mix,
                                 # transport_lane, material_embodied, waste_treatment,
                                 # land_use_removals, finance_proxy, classification_mapping
method_profile:       enum       # MethodProfile: corporate_scope1/2_location/2_market/3,
                                 # product_carbon, freight_iso_14083, land_removals,
                                 # finance_proxy, eu_cbam, eu_dpp
source_id:            string     # FK → SourceRegistryEntry.source_id
source_release:       string     # e.g., "EPA-2024-Q4", "DESNZ-2025"
factor_version:       string     # semver; immutable once published
jurisdiction:         Jurisdiction
                                 #   country: ISO 3166-1 alpha-2
                                 #   region:  ISO 3166-2 (e.g., "US-CA")
                                 #   grid_region: eGRID/NERC/CEA subregion
valid_from:           date       # REQUIRED — non-negotiable #5
valid_to:             date?      # null = open-ended
activity_schema:      ActivitySchema
                                 #   category:           e.g., "purchased_electricity"
                                 #   sub_category:       e.g., "grid_average"
                                 #   classification_codes: ["NAICS:221112","CN:7208","ISIC:D351"]
numerator:
  unit:               string     # e.g., "kg_CO2e"
  gas_breakdown:      GHGVectors # CO2, CH4, N2O, SF6, NF3, HFC_xxx, PFC_xxx, biogenic_CO2
  derived_co2e:       float      # computed, never stored alone (non-negotiable #1)
denominator:
  unit:               string     # e.g., "kWh", "tonne", "USD", "tonne.km"
gwp_set:              enum       # IPCC_AR6_100 (default), IPCC_AR6_20, IPCC_AR5_100, ...
formula_type:         enum       # FormulaType: direct_factor, combustion, lca, spend_proxy,
                                 # transport_chain, residual_mix, carbon_budget, custom
parameters:           FactorParameters
                                 #   scope_applicability: ["scope_1","scope_2_location"]
                                 #   electricity_basis:   location_based | market_based | ...
                                 #   residual_mix_applicable: bool
                                 #   supplier_specific:   bool
                                 #   transmission_loss_included: bool
                                 #   biogenic_share:     0.0–1.0
                                 #   uncertainty_low:    float
                                 #   uncertainty_high:   float
quality:
  temporal:           int(1-5)   # 5 = factor year = activity year
  geographical:       int(1-5)
  technological:      int(1-5)
  representativeness: int(1-5)
  methodological:     int(1-5)
  composite:          float      # weighted average; derived
lineage:
  provenance:         SourceProvenance   # source_org, publication, year, methodology,
                                         # version, citation, source_url
  raw_record:         RawRecordRef       # raw_record_id, raw_payload_hash, format, storage_uri
  change_log:         [ChangeLogEntry]   # append-only; non-negotiable #2
licensing:
  license:            string     # SPDX identifier (e.g., "CC0-1.0", "CC-BY-4.0", "Proprietary")
  redistribution_class: enum     # RedistributionClass: open | restricted | licensed |
                                 # customer_private | oem_redistributable
  redistribution_allowed: bool
  commercial_use_allowed: bool
  attribution_required: bool
  attribution_text:   string
explainability:
  assumptions:        [string]   # human-readable assumptions used in derivation
  fallback_rank:      int(1-7)   # 1 = tenant override, 7 = global default
  rationale:          string     # one-sentence natural language
verification:
  status:             enum       # unverified | internal_review | external_verified | regulator_approved
  verified_by:        string?
  verified_at:        datetime?
  verification_reference: string?
# Additional v1 fields
compliance_frameworks: [string]  # ["CBAM","CSRD","SB253","PACT","EUDR"] — used by regulatory_tagger
factor_status:        enum       # certified | preview | connector_only | deprecated
content_hash:         string     # SHA-256 of canonical JSON — stable across serializations
primary_data_flag:    enum       # primary | primary_modeled | secondary | proxy
uncertainty_95ci:     float      # fractional (e.g., 0.05 = ±5%)
uncertainty_distribution: enum   # unknown | normal | log_normal | triangular | uniform | beta_pert
```

**Enforcement.** `validate_non_negotiables(record)` is called at every write. A record cannot be persisted if:
1. It has no gas vectors (non-negotiable #1 — only CO2e is forbidden).
2. It has no `factor_version` when status is `certified` or `deprecated` (non-negotiable #2).
3. It has no `valid_from` or no `provenance`+`source_release` pair (non-negotiable #5).
4. It covers a regulated framework (CBAM/CSRD/SB253/PACT) without a `method_profile` (non-negotiable #6).

`enforce_license_class_homogeneity(records)` is called at every response build — the API cannot mix redistribution classes in a single response (non-negotiable #4).

### 4.2 Source Record v1.0

Source: `greenlang/factors/source_registry.py :: SourceRegistryEntry`.

```yaml
source_id:            string     # stable slug, e.g., "epa_ghg_hub"
display_name:         string
publisher:            string     # "EPA", "DESNZ", "IEA"
jurisdiction:         string     # "US", "EU", "UK", "Global"
dataset_version:      string     # "2024-Q4"
publication_date:     date
validity_period:      string     # ISO 8601 interval "2024-01-01/2024-12-31"
source_type:          enum       # government | standard_setter | industry_body |
                                 # licensed_commercial | customer_provided | academic
license_class:        enum       # open | restricted | licensed | customer_private | oem_redistributable
redistribution_allowed: bool
derivative_works_allowed: bool
commercial_use_allowed: bool
attribution_required: bool
attribution_text:     string
citation_text:        string
connector_only:       bool       # if true, factors are not redistributable — API returns on demand only
cadence:              enum       # annual | quarterly | monthly | ad_hoc | on_notification
watch:
  mechanism:          enum       # rss | scrape | file_diff | regulator_notification | none
  url:                string
  file_type:          enum?      # pdf | csv | xml | xlsx
approval_required_for_certified: bool
legal_signoff_artifact: string?  # URI to signed counsel memo
legal_signoff_version: string?
ingestion_date:       datetime
verification_status:  enum
change_log_uri:       string
legal_notes:          string
```

### 4.3 Method Pack schema

Source: `greenlang/factors/method_packs/base.py`.

```yaml
pack_id:              string     # "corporate_scope1", "eu_cbam", "freight_iso_14083", ...
pack_name:            string
method_profile:       enum       # MethodProfile
version:              string     # semver
regulatory_references: [string]  # ["GHG-Protocol:Corporate-Standard:2004+Revised-2015", ...]
selection_rules:      [SelectionRule]
                                 # ordered list; first match wins within a cascade step
scope_applicability:  [string]   # ["scope_1","scope_3.cat4"]
methodology_notes:    markdown   # full human-readable methodology, paste-safe into reports
unit_requirements:    dict       # required numerator / denominator units
default_gwp_set:      enum
```

### 4.4 Mapping schema

Source: `greenlang/factors/mapping/base.py`.

```yaml
mapping_id:           string
from_system:          string     # "NAICS", "CN", "ISIC", "UNSPSC", "GLEIF"
from_code:            string
to_system:            string     # "greenlang.factor_family" | "greenlang.method_profile"
to_code:              string
confidence:           float      # 0–1
curator:              string
curated_at:           datetime
rationale:            string
```

### 4.5 Edition schema

Source: `greenlang/factors/edition_manifest.py :: EditionManifest`.

```yaml
edition_id:           string     # e.g., "greenlang-factors-2026Q2-certified"
status:               enum       # stable | pending | retired
created_at:           datetime   # UTC ISO-8601
factor_count:         int
aggregate_content_hash: string   # SHA-256 over sorted factor content_hashes
per_source_hashes:    dict       # source_id → SHA-256 of that source's subset
deprecations:         [string]   # factor_ids deprecated in this edition
changelog:            [string]   # human-readable bullets
policy_rule_refs:     [string]   # Policy-Graph rule refs bound to this edition
manifest_fingerprint: string     # deterministic hash excluding created_at
signature:            string     # Ed25519 signature by GreenLang signing key
```

### 4.6 Entitlement schema

Source: `greenlang/factors/entitlements.py`.

```yaml
entitlement_id:       string
tenant_id:            string
sub_tenant_id:        string?    # for Consulting/OEM multi-tenant
pack_sku:             enum       # PackSKU: electricity_premium, freight_premium,
                                 # product_carbon_premium, epd_premium, agrifood_premium,
                                 # finance_premium, cbam_premium, land_premium
effective_from:       date
effective_to:         date?
seat_cap:             int?       # null = unlimited
volume_cap:           int?       # monthly call cap
oem_rights:           enum       # forbidden | internal_only | redistributable
status:               enum       # active | suspended | expired
created_by:           string
created_at:           datetime
```

### 4.7 Overlay schema

Source: `greenlang/factors/tenant_overlay.py :: TenantOverlay`.

```yaml
overlay_id:           string
tenant_id:            string
sub_tenant_id:        string?
factor_id:            string
override_value:       float
override_unit:        string
gas_breakdown:        GHGVectors?  # optional, for multi-gas overrides
valid_from:           date
valid_to:             date?
source:               string     # "supplier_audit", "internal_energy_audit", ...
notes:                string
attachment_uri:       string?
approver_ids:         [string]   # dual control
created_by:           string
created_at:           datetime
updated_at:           datetime
active:               bool
```

### 4.8 Resolution Request / Resolved Factor

Source: `greenlang/factors/resolution/request.py`, `greenlang/factors/resolution/result.py`.

```yaml
# ResolutionRequest
tenant_id:            string
sub_tenant_id:        string?
method_profile:       enum       # REQUIRED — non-negotiable #6
activity:
  category:           string
  sub_category:       string?
  classification_codes: [string]?
  description_text:   string?    # free text for semantic matching
jurisdiction:         Jurisdiction
date:                 date       # resolution date, used for valid_from/valid_to
edition_id:           string?    # null = latest stable
request_id:           string     # idempotency key

# ResolvedFactor
factor_id:            string
factor_version:       string
edition_id:           string
factor:               CanonicalFactorRecord
gas_breakdown:        GHGVectors
derived_co2e:         float
denominator_unit:     string
fallback_rank:        int(1-7)
uncertainty_band:     UncertaintyBand    # {low, high, distribution}
alternates:           [AlternateCandidate]  # scored-but-not-selected candidates
tiebreak_reasons:     [string]
explanation:          Explainability
signed_receipt:       Receipt
warnings:             [string]   # e.g., deprecated source, widening uncertainty
```

---

## 5. REST API contract — OpenAPI v1

### 5.1 Conventions

- Base URL: `https://api.greenlang.io/v1/` (prod), `https://api-staging.greenlang.io/v1/` (staging).
- Auth: `Authorization: Bearer <JWT>` or `X-API-Key: gl_fac_{env}_{rand}`. JWT validated via `greenlang.integration.api.dependencies`; API key validated via `api_auth.py`.
- Content: `application/json; charset=utf-8`. Bulk export also supports `application/x-ndjson`, `text/csv`, `application/vnd.ms-excel`.
- Versioning: URL-prefixed `/v1/`. Breaking changes ship at `/v2/`.
- Edition pinning: clients send `X-GreenLang-Edition: <edition_id>` to lock resolution. Server echoes the same header on every response.
- Idempotency: `Idempotency-Key` header supported on POST endpoints. TTL 24h.
- Pagination: cursor-based (`cursor`, `next_cursor`, `limit` default 50 max 500).
- Rate limits: Community 60 rpm + 1 000/month, Pro 600 rpm + 100 000/month, Consulting 3 000 rpm + 1M/month, Enterprise negotiated. Enforced in `middleware/rate_limiter.py`. Responses carry `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.
- Caching: `ETag` + `Cache-Control: private, max-age=<tier>`. `If-None-Match` returns 304. Implemented in `api_endpoints.py` + `search_cache.py`.
- Signed receipts: every 2xx response carries `X-GreenLang-Receipt: <base64>` (body also includes `signed_receipt` for easy handling).
- Quotas: `X-GreenLang-Quota-Remaining`, `X-GreenLang-Quota-Reset` on every response.

### 5.2 Endpoints

Pseudo-OpenAPI; see `/v1/openapi.json` (F-076, v1 deliverable) for the full machine-readable file.

#### 5.2.1 Resolution

```
GET /v1/factors/resolve
  query: method_profile (required), category (required), sub_category, classification_codes,
         country, region, grid_region, date (default today), edition_id (optional)
  headers: Authorization, X-GreenLang-Edition (optional)
  200: ResolvedFactor
  400: BAD_REQUEST (missing method_profile, invalid date)
  404: NO_VALID_FACTOR (no candidate survives 7-step cascade)
  409: LICENSE_CLASS_CONFLICT (non-negotiable #4 triggered)
  429: RATE_LIMITED
```

```
POST /v1/factors/batch
  body: { activities: [ResolutionRequest], edition_id, mode: sync|async }
  sync (≤ 1 000 items): 200 { results: [ResolvedFactor] }
  async (≤ 100 000 items): 202 { job_id }

GET /v1/factors/batch/{job_id}
  200: { status: queued|running|complete|failed, progress, download_url }
```

#### 5.2.2 Catalog

```
GET /v1/factors/{factor_id}
  query: edition_id (optional)
  200: CanonicalFactorRecord
  404: NOT_FOUND

GET /v1/factors/{factor_id}/explain
  query: edition_id (optional), request_id (optional — ties to a prior resolve call)
  200: { fallback_rank, cascade_steps_considered, alternates_scored,
         unit_conversions, methodology_note, licensing_check, non_negotiables_checked }

GET /v1/factors/{factor_id}/diff
  query: from (edition), to (edition)
  200: { added_fields, removed_fields, changed_fields: [{path, old, new}] }

POST /v1/factors/search
  body: { query?: string, filters: { method_profile, jurisdiction, activity_category,
          source_id, factor_status, compliance_frameworks }, sort, cursor, limit }
  200: { results, total_count, next_cursor }
```

#### 5.2.3 Editions

```
GET /v1/factors/editions
  query: status (stable|pending|retired), limit, cursor
  200: { editions: [EditionManifest], next_cursor }

GET /v1/factors/editions/{edition_id}
  200: EditionManifest (with signature)

GET /v1/factors/editions/diff
  query: from, to
  200: { added_factor_ids, removed_factor_ids, changed_factor_ids, unchanged_count }

POST /v1/factors/editions/simulate  (Enterprise tier)
  body: { from_edition, to_edition, activity_sample, tenant_scope }
  200: { delta_totals: {scope: delta_tco2e}, per_factor: [...] }
```

#### 5.2.4 Sources

```
GET /v1/sources
  query: publisher, jurisdiction, license_class, cadence
  200: { sources: [SourceRegistryEntry] }

GET /v1/sources/{source_id}
  200: SourceRegistryEntry + recent_artifacts

GET /v1/sources/watch/status
  200: { per_source: [{source_id, last_run, next_run, last_change_kind, health}] }
```

#### 5.2.5 Method packs

```
GET /v1/method-packs
  200: { packs: [{pack_id, method_profile, version, regulatory_references, summary}] }

GET /v1/method-packs/{pack_id}
  200: MethodPack (full, including methodology_notes markdown)

GET /v1/method-packs/{pack_id}/rules
  200: { selection_rules: [...] }
```

#### 5.2.6 Mappings (F-041)

```
GET /v1/mappings/{classification_system}
  query: code, to_system
  200: { mappings: [{from_code, to_code, confidence, rationale}] }
```

#### 5.2.7 Overlays (tenant-private)

```
POST /v1/overlays          -> create (requires overlay_author role)
GET  /v1/overlays/{id}
GET  /v1/overlays          -> list (tenant-scoped)
PATCH /v1/overlays/{id}/approve  -> requires overlay_approver role (different user from author)
PATCH /v1/overlays/{id}/publish  -> requires overlay_publisher role
DELETE /v1/overlays/{id}   -> soft delete (active=false); no hard delete, ever
```

#### 5.2.8 Audit bundle

```
POST /v1/audit-bundle
  body: { scope: factor_id | edition_id | run_id,
          tenant_id, sub_tenant_id?, date_range?, case_id? }
  202: { job_id, download_url (when ready) }
  Returns a signed ZIP: normalized factors (JSON), raw source artifacts (PDF/CSV),
  parser logs, QA decisions, approver signatures, SHA-256 chain.
```

#### 5.2.9 Export

```
GET /v1/factors/export
  query: format (csv|jsonl|xlsx), edition_id, filters...
  200: streaming file (tier-aware row limits from TierVisibility.max_export_rows)
  403: LICENSE_EXPORT_DENIED (pack not entitled for redistribution)
```

#### 5.2.10 Entitlements + usage

```
GET /v1/entitlements
  200: { packs: [{pack_sku, status, effective_from, effective_to, oem_rights}] }

GET /v1/usage
  query: period (day|month|year)
  200: { api_calls, batch_items, export_rows, per_pack_breakdown }
```

#### 5.2.11 Webhooks

```
POST /v1/webhooks/subscribe
  body: { url, events: [...], secret, sub_tenant_id? }
  201: { webhook_id, signing_key }

GET /v1/webhooks
DELETE /v1/webhooks/{id}
GET /v1/webhooks/{id}/deliveries  (last 100 deliveries with status)
```

#### 5.2.12 Platform (Consulting + OEM)

```
POST /v1/platform/sub-tenants
GET  /v1/platform/sub-tenants
PATCH /v1/platform/sub-tenants/{id}/entitlements
GET  /v1/platform/sub-tenants/{id}/usage
```

#### 5.2.13 Meta

```
GET /v1/health                    -> { status: ok }
GET /v1/ready                     -> { db: ok, cache: ok, signing_key: ok }
GET /v1/openapi.json              -> frozen OpenAPI 3.1 spec
GET /v1/.well-known/greenlang-signing-key  -> public Ed25519 key (PEM)
```

### 5.3 Error codes

All errors return `{ error: { code, message, trace_id, docs_url } }`.

| Code | HTTP | Meaning |
|---|---|---|
| `BAD_REQUEST` | 400 | Invalid input |
| `UNAUTHENTICATED` | 401 | No valid credentials |
| `FORBIDDEN_TIER` | 403 | Tier too low |
| `FORBIDDEN_PACK_ENTITLEMENT` | 403 | Pack not entitled |
| `LICENSE_EXPORT_DENIED` | 403 | Connector-only cannot export |
| `NOT_FOUND` | 404 | Resource absent |
| `NO_VALID_FACTOR` | 404 | Resolution failed all 7 steps |
| `LICENSE_CLASS_CONFLICT` | 409 | Non-negotiable #4 |
| `NON_NEGOTIABLE_VIOLATION` | 422 | Write rejected by validator |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Wrapped server error |
| `UPSTREAM_UNAVAILABLE` | 502 | Catalog or cache down |
| `DEGRADED` | 503 | Serving stale edition |

---

## 6. Factor-pack roadmap

### 6.1 SKU boundaries

The Community tier + open sources cover ~60% of typical corporate Scope 1–2 reporting for the G20 markets. Premium packs fill the rest.

| Pack SKU | Contents | Based on | License class | Redistribution | Who buys | FY27 readiness slice |
|---|---|---|---|---|---|---|
| `electricity_premium` | Residual mix, hourly grid, market-based supplier factors, regional trackers (AIB, Green-e, IEA, Ember) | `ingestion/parsers/{aib_residual_mix, green_e, green_e_residual, india_cea, japan_meti_residual, australia_nga_residual}.py` + Electricity Maps connector | Licensed (partial) + Open | Licensed subset: attribution-only; Open subset: redistributable | CBAM filers, CSRD E1 filers, RE100 reporters | **Certified, week 3–4** |
| `freight_premium` | ISO 14083 lane factors, GLEC default factors, mode-chain models | `ingestion/parsers/freight_lanes.py` + `method_packs/freight.py` | Restricted | Attribution required | Shippers, 3PL, CBAM transport chain | **Certified, week 11–12** |
| `product_carbon_premium` | LCI dataset (ecoinvent-tier), PACT exchange factors | `method_packs/product_carbon.py` + `ingestion/parsers/pact_product_data.py` + new ecoinvent connector | Licensed (commercial) | Internal-use only; OEM needs separate redistribution addon | PCF tool vendors, DPP Hub partners | **Preview** |
| `epd_premium` | Construction EPDs (EC3-tier) | `ingestion/parsers/ec3_epd.py` | Restricted | Attribution required | Construction, real-estate, BuildingOS precursor | **Preview** |
| `agrifood_premium` | Agricultural defaults, food LCA, livestock emission factors, fertilizer N2O | `ingestion/parsers/ipcc_defaults.py` (agri subset) + new FAOSTAT connector + new agri-LCA connector | Open + Restricted | Mixed (separate responses per class) | Agrifood manufacturers, retailers, EUDR filers | **Preview** |
| `finance_premium` | PCAF asset-class proxy factors | `ingestion/parsers/pcaf_proxies.py` + `method_packs/finance_proxy.py` | Restricted | Attribution required | Banks, asset managers, FinanceOS precursor | **Preview** |
| `cbam_premium` | CBAM default values, CN-code to factor mapping, quarterly Q-report factor sets | `ingestion/parsers/cbam_full.py` + `method_packs/eu_policy.py` | Open (regulator-published) + Licensed (industry data) | Mixed | CBAM-obligated importers/exporters | **Certified, week 11–12** |
| `land_premium` | LSR removals, biogenic carbon, forest land-use | `ingestion/parsers/lsr_removals.py` + `method_packs/land_removals.py` | Open + Restricted | Attribution required | Nature/TNFD precursor, CDR, agrifood | **Preview** |

### 6.2 v1 release sequence

The release gate per slice: an edition passes the QA suite (`quality/batch_qa.py`), the gold eval set scores ≥ 85% top-1, `validate_non_negotiables` is clean, license counsel signs off, and the edition manifest is signed. Only then does the slice ship **Certified**. All other packs ship **Preview** (visible, queryable, not recommended for filings) in v1.

| Sequence | Week | Slice | Pack | Status after slice | Gate |
|---|---|---|---|---|---|
| 1 | 3–4 | Electricity | `electricity_premium` | Certified | Gold set ≥ 85%, 5 grid sub-regions, 3 residual mixes live |
| 2 | 5–6 | Stationary combustion | `community` (open) | Certified | EPA/DESNZ/IPCC Tier-1 defaults, 40 fuels |
| 3 | 7–8 | Scope 3 Category 1–4 (spend + supplier) | `community` (open) + partial `product_carbon_premium` | Preview | Spend-based + average-data methods only |
| 4 | 9–10 | Scope 3 combustion downstream + refrigerants | `community` (open) | Certified | GWP sets locked AR6 default |
| 5 | 11–12 | Freight + CBAM | `freight_premium`, `cbam_premium` | Certified | ISO 14083 lane factors live; CBAM Q-report mapping signed off |
| 6 | Post-v1 (FY27 Q3) | EPD (construction) | `epd_premium` | Preview → Certified on Q3 release | EC3 license negotiated |
| 7 | Post-v1 (FY27 Q3) | Agrifood + Land | `agrifood_premium`, `land_premium` | Preview | FAOSTAT + LSR data wired |
| 8 | Post-v1 (FY27 Q4) | Finance proxy | `finance_premium` | Preview → Certified on Q4 release | PCAF license negotiated |
| 9 | Post-v1 (FY27 Q4) | Product-carbon (ecoinvent-tier) | `product_carbon_premium` | Preview | Ecoinvent OEM arrangement concluded |

### 6.3 Licensing and redistribution rules per pack

- **Open packs** (CC0, CC-BY, US government public-domain): redistributable, commercial use allowed, attribution required where license says so. Example: EPA GHG Hub.
- **Restricted packs**: attribution required, redistribution allowed only with attribution + no derivative aggregation. Example: IEA tables, DESNZ conversion factors.
- **Licensed packs**: commercial license required; internal-use default; derivative works must keep source attribution intact. Example: ecoinvent-tier LCI, EC3 EPD library.
- **Customer-private packs**: tenant overlays and company-internal factor libraries. Never redistributable. Never leak outside `tenant_id` scope.
- **OEM-redistributable packs**: explicit OEM addon; platform/OEM tier can embed and serve to their own customers; license terms chained.

The API **never** mixes these in a single response (CTO non-negotiable #4, enforced by `enforce_license_class_homogeneity()` in `data/canonical_v2.py`). If a query spans classes, the response is split by `redistribution_class` across multiple sub-responses (or the caller supplies a `redistribution_class` filter).

---

## 7. Pricing architecture

### 7.1 Principle

**Per the CTO non-negotiable in the spec, pricing is NOT by factor count.** Counting factors penalizes the customer for getting more coverage, which is exactly the outcome we want. Pricing is by: API calls, batch volume, factor-pack entitlements, private registry usage, tenant count, OEM rights, and SLA / support level.

### 7.2 Tier matrix

| Dimension | Community | Developer Pro | Consulting / Platform | Enterprise |
|---|---|---|---|---|
| **Price** | Free | $49 / $199 / $499 / mo (3 sub-tiers) | $25k–$75k / yr + usage | $100k–$300k / yr ACV |
| **API calls / month** | 1 000 | 25 000 / 100 000 / 500 000 | 1M / 5M | 10M+ (negotiated) |
| **Batch items / month** | 1 000 | 25k / 100k / 500k | 2M | 10M+ |
| **Factors visibility** | Certified only | Certified + Preview | Certified + Preview | All incl. Connector-only |
| **Premium pack entitlement** | None | Per-pack addon ($99–$999/mo) | 3 packs included + more addon | All 8 packs available via pack-addon |
| **Private registry (overlay) entries** | 0 | 50 per project | Multi-client sub-tenants | Unlimited + approval workflow + SCIM |
| **Audit bundle export** | No | No | Yes (per-run) | Yes (per-run + signed) |
| **Bulk export** | No | Yes (up to 5 000 rows) | Yes (up to 50 000 rows) | Yes (up to 1M rows; higher by quote) |
| **Signed receipts** | Yes (HMAC) | Yes (HMAC + Ed25519 opt-in) | Yes (Ed25519) | Yes (Ed25519 + customer-managed keys) |
| **SLA** | None | 99.5% | 99.9% (5% monthly credit on miss) | 99.95% (10%/25% credits) |
| **Support** | Community forum | Email, business hours | Email + Slack, 1 business day | Named CSM, 4h P1, 24×7 |
| **SSO / SCIM** | No | No | Optional ($500/mo) | Included |
| **VPC peering / data residency** | No | No | No | Yes |
| **OEM white-label** | No | No | Yes (included in Platform SKU) | Yes (addon $50k+) |
| **Contract** | Click-through | Self-service | Annual MSA | Annual MSA + DPA |

### 7.3 Premium pack SKUs (add to any tier)

| Pack SKU | Pro addon | Consulting bundle | Enterprise addon |
|---|---|---|---|
| `electricity_premium` | $99/mo | Included (1 of 3) | $12k/yr |
| `freight_premium` | $199/mo | Include option | $18k/yr |
| `product_carbon_premium` | $499/mo | Include option | $40k/yr (+ ecoinvent license chain) |
| `epd_premium` | $199/mo | Include option | $18k/yr |
| `agrifood_premium` | $199/mo | Include option | $24k/yr |
| `finance_premium` | $299/mo | Include option | $36k/yr (+ PCAF license chain) |
| `cbam_premium` | $299/mo | Included (1 of 3) | $36k/yr |
| `land_premium` | $149/mo | Include option | $18k/yr |

### 7.4 Competitive defense

| Competitor | Price reference | GreenLang position |
|---|---|---|
| ecoinvent | ~$10k / seat for LCI | We don't sell raw ecoinvent; we sell **resolution on top** of licensed LCI; customers bring their own ecoinvent license or buy the `product_carbon_premium` with an OEM arrangement. |
| Climatiq | $0.01–$0.10 / call | We match at Pro tier (~$0.005–$0.02 effective per call at 500k calls/mo) and beat on method-pack breadth + explainability. |
| Watershed | $100k+ ACV (bundled with app) | Our Factors ACV is $100k–$300k; we are the substrate, not the app. Watershed-tier customers buy Enterprise with 2–3 Premium packs. |
| Persefoni | $75k+ ACV (bundled with app) | Same framing. |
| Climate TRACE, Open Supply Hub, OpenLCA open tools | Free / open | Our Community tier matches; we layer Certified, Explain, Editions, and SLA on top. |

### 7.5 Billing and metering implementation

- Stripe products + prices provisioned via `billing/stripe_provider.py` + a new `ga/sku_catalog.py` (F-085, v1 deliverable).
- Metering events (per call, per batch item, per export row, per pack access) pushed via `billing/metering.py` + `metering.py` through `billing/aggregator.py` into Stripe.
- Replay-safe usage sink (`billing/usage_sink.py`) backs every event; Stripe webhook handler (`billing/webhook_handler.py`) activates pack entitlements on checkout.session.completed.
- Enterprise invoicing is manual (via `ga/billing.py`) for custom ACV contracts.

---

## 8. UI surface

v1 ships two web surfaces and a developer portal. All three are MISSING as code today; v1 builds them against the existing REST API. These sections prescribe the component tree, routes, and API wiring — not pixel-perfect designs.

### 8.1 Factor Explorer (public, `factors.greenlang.io`)

**Purpose.** A public browsable catalog for developers, consultants, and enterprise evaluators. Free to use (rate-limited).

**Routes.**

- `/` → Search page.
- `/factors` → Faceted search results (powered by `POST /v1/factors/search`).
- `/factors/:id` → Factor detail page.
- `/factors/:id/explain` → 7-step cascade viewer.
- `/sources` → Source catalog list.
- `/sources/:id` → Source detail (license, cadence, last update, artifacts).
- `/method-packs` → Method-pack browser.
- `/method-packs/:id` → Full methodology note + selection rules.
- `/dashboard` → Three-label dashboard (Certified / Preview / Connector-only counts per source, per method pack, per jurisdiction).
- `/editions` → Edition timeline + changelog.
- `/editions/:id/diff/:other` → Edition diff viewer.
- `/embed/*` → Iframe-safe variants (OEM only, auth-gated).

**Component tree.**

```
<App>
  <Shell>
    <Header tier={tierBadge} />
    <Sidebar>
      <FilterBar (jurisdiction, method_profile, activity_category, source, status) />
    </Sidebar>
    <Main>
      <Routes>
        <SearchPage>              api → POST /v1/factors/search
          <ResultsGrid />
          <CursorPager />
        <FactorDetailPage>        api → GET /v1/factors/{id} + /explain + /diff
          <FactorHeader/><FactorStatusBadge/><ProvenancePanel/>
          <GasVectorTable/><UncertaintyChart/>
          <CascadeTimeline/>      ← 7-step viewer
          <AlternatesTable/>
          <MethodologyNote/>
          <LicenseBadge/>
          <CopyAsCurl/>
        <SourceCatalog>           api → GET /v1/sources
        <SourceDetailPage>        api → GET /v1/sources/{id}
        <MethodPackBrowser>       api → GET /v1/method-packs
        <ThreeLabelDashboard>     api → service.py :: status_summary()
          <CertifiedCountTile/><PreviewCountTile/><ConnectorOnlyCountTile/>
          <BySourceBreakdown/><ByMethodProfileBreakdown/>
        <EditionsTimeline>        api → GET /v1/factors/editions
        <EditionDiffViewer>       api → GET /v1/factors/editions/diff
      </Routes>
    </Main>
    <Footer />
  </Shell>
</App>
```

**Stack.** Next.js 14 (App Router) + TypeScript + Tailwind + shadcn/ui. Uses the TypeScript SDK (`greenlang/factors/sdk/ts`) under the hood.

### 8.2 Operator Console (internal, `ops.greenlang.io`)

**Purpose.** Internal tool for the GreenLang data-engineering team and for Enterprise methodology leads (a lite variant). Ingestion, mapping, QA, diff, approval, override management, impact simulation.

**Routes.**

- `/` → Dashboard (active ingests, open QA queue, pending approvals, SLA health).
- `/ingestion` → Ingestion console.
  - `/ingestion/:source_id` → Per-source run history, last artifact, parser log, dead-letter queue.
- `/mapping` → Mapping workbench (NAICS/CN/ISIC ↔ factor family / method profile).
- `/qa` → QA dashboard (validators, cross-source, license scanner, consensus).
  - `/qa/queue` → Review queue.
  - `/qa/factor/:id/review` → Per-factor review with approve/reject.
- `/diff` → Edition + factor diff viewer.
- `/approvals` → Approval workflow board (author → reviewer → publisher, 4-eyes).
- `/overrides` → Tenant overlay manager (Enterprise only).
  - `/overrides/new` → Create overlay form (attachment upload, reason field).
- `/simulator` → Impact simulator (pick from_edition + to_edition + activity sample).
- `/webhooks` → Subscription + delivery viewer.
- `/sla` → SLA monthly credit report.

**Component tree.**

```
<App>
  <Shell>
    <TopNav tier="internal|enterprise" />
    <SideNav>
      <NavItem to="/ingestion" /> ...
    </SideNav>
    <Main>
      <Routes>
        <Dashboard>
          <ActiveIngestsTile/><OpenQATile/><PendingApprovalsTile/><SLAHealthTile/>
        <IngestionConsole>            api → watch/status_api + ingestion/sqlite_metadata
          <SourceTable/>
          <SourceRunLog/><ParserLogViewer/>
          <DeadLetterQueue/>
          <RetryButton/>
        <MappingWorkbench>            api → /v1/mappings + mapping/*.py libraries
          <CrosswalkTable/>
          <CurateMappingModal/>       ← create / edit a mapping
        <QADashboard>                 api → quality/* + release_signoff
          <ValidatorFailureList/>
          <CrossSourceDriftTable/>
          <LicenseScannerAlerts/>
        <ReviewQueue>                 api → /v1/qa/queue
        <FactorReviewPage>            api → /v1/factors/{id} + approve/reject hooks
          <FactorSideBySide/>         ← proposed vs current
          <AssumptionsEditor/>
          <ApproveRejectForm/>
        <DiffViewer>                  api → /v1/factors/{id}/diff + /v1/editions/diff
        <ApprovalBoard>               api → quality/review_workflow
        <OverrideManager>             api → /v1/overlays
          <OverlayTable/>
          <CreateOverlayForm/>
          <ApproveOverlayModal/>
        <ImpactSimulator>             api → POST /v1/editions/simulate
          <EditionPicker/><SamplePicker/><DeltaChart/><PerFactorTable/>
        <WebhookViewer>               api → /v1/webhooks
        <SLACreditReport>             api → ga/sla_tracker
      </Routes>
    </Main>
  </Shell>
</App>
```

**Stack.** Same as Factor Explorer. Both share a shared component library under `web/packages/ui-common/`.

### 8.3 Developer portal (`developers.greenlang.io`)

**Routes.**

- `/` → Landing + value prop.
- `/docs` → OpenAPI-generated reference (from `/v1/openapi.json`).
- `/sdk` → SDK docs (Python + TypeScript) + sample apps.
- `/keys` → API key management (create, rotate, delete).
- `/usage` → Per-key usage dashboard (from `GET /v1/usage`).
- `/webhooks` → Webhook config + delivery log.
- `/billing` → Stripe customer portal link.
- `/quickstarts` → Onboarding sample queries (from `onboarding/sample_queries.py`).

---

## 9. Non-negotiables (per CTO)

These are the rules that v1 encodes, enforces, and never relaxes. Each one maps to specific code.

1. **Never store only CO2e.** Store gas components (CO2, CH4, N2O, SF6, NF3, HFC_xxx, PFC_xxx, biogenic_CO2) and derive CO2e at response time using the requested `gwp_set`. Enforced by `validate_non_negotiables(record)` rejecting records with no `vectors`. Field: `numerator.gas_breakdown` in the Canonical Factor Record.
2. **Never overwrite a factor.** Every change creates a new `factor_version`. Previous versions remain queryable. The `ChangeLogEntry` list on the record is append-only. Editions are immutable once published (signed manifest in `edition_manifest.py`). Rollback means pointing at a previous edition, not mutating an existing one.
3. **Never hide fallback logic.** Every resolved factor carries `fallback_rank` (1–7) and an `Explainability` block. `GET /factors/{id}/explain` is a first-class primitive — the cascade is visible in the API, the SDK, and the Factor Explorer.
4. **Never mix licensing classes.** `enforce_license_class_homogeneity(records)` runs at every response build. Open + Licensed + Customer-private never appear in the same response. Callers who query across classes get multiple responses or an explicit `redistribution_class` filter.
5. **Never ship a factor without `valid_from` / `valid_to` and source version.** `validate_non_negotiables(record)` rejects any record missing `valid_from` or missing `provenance` + `source_release`. Resolution respects validity windows before fallback kicks in.
6. **Policy workflows never call raw factors.** They call `method_profile`s. Certified records in regulated frameworks (CBAM, CSRD, SB253, PACT) must carry a `method_profile`. Enforced in `validate_non_negotiables()` and in the resolution request schema (`method_profile` is REQUIRED, never optional).
7. **Open-core vs Enterprise boundaries are explicit and stable.** `tier_enforcement.py :: TierVisibility` is the single source of truth. Community cannot see Preview or Connector-only; Pro cannot see Connector-only; Consulting gets export but not connector-only; Enterprise gets everything. Source code that belongs to the open core lives under Apache-2.0; commercial code lives under a proprietary license and is explicitly separated in the build.

---

## 10. FY27 90-day launch plan (week-by-week)

Assumption: launch date is week 12 (end of Q1 FY27). Team: 1 PM (you), 1 CTO (architecture + sign-off), 3 BE, 1 DE, 1 ML, 1 FE, 1 Sec, 1 Ops, 1 GTM. Every week ends with a demo + retro.

### Week 1 — Commit

- PRD sign-off by founder + CTO.
- OpenAPI 3.1 spec **frozen** (F-076). Any change after this needs a documented version bump.
- Canonical Factor Record v1.0 schema **frozen**.
- Repo cleanup Sprint 1 from `FY27_vs_Reality_Analysis.md §8` begins in parallel (non-blocking).
- GTM starts pricing-page copy + 8 pilot prospect list.

### Week 2 — Schema + contract

- Freeze Source / Method-Pack / Mapping / Edition / Entitlement / Overlay schemas.
- Publish `/v1/openapi.json` from the frozen spec.
- CI gate: any PR touching `greenlang/data/canonical_v2.py` or `greenlang/factors/resolution/*` must cite the PRD schema section.
- Dev-portal landing skeleton (Next.js) stood up.

### Week 3 — Hosted API (staging)

- Deploy FastAPI app on staging (AWS EKS or GCP GKE — decide by end of week 2).
- Wire the existing `api_endpoints.py` pure-logic functions into a FastAPI router under `greenlang/factors/api/` (F-060 remediation).
- Connect `catalog_repository_pg.py` to a managed Postgres (RDS/Cloud SQL) in staging.
- Stand up Redis (`cache_redis.py`) + pgvector index.
- Auth + metering middleware live; API keys issued for the team.
- First Certified edition cut for the **Electricity slice** (Sequence 1).

### Week 4 — Electricity Certified

- Finalize 5 grid sub-regions + 3 residual mixes for electricity_premium.
- Gold eval set (F-039): 100 electricity items curated with auditor-style ground truth; top-1 ≥ 85% required to ship Certified.
- Release sign-off via `quality/release_signoff.py`. Signed manifest published.
- Public staging URL hit with internal load test: 1 000 rps sustained, p95 < 200 ms.

### Week 5 — Factor Explorer + developer portal v1

- Factor Explorer UI (F-109): Search, Detail, Source Catalog, Method-Pack Browser live.
- Developer portal (F-111): docs from OpenAPI, API key UI, usage chart, webhook UI.
- Per-run audit-bundle aggregator (F-068) ships behind a feature flag.
- Mapping REST surface (F-041) ships.

### Week 6 — Factor Explorer finishing + Stationary combustion Certified

- Explainability page (`/factors/:id/explain`) live in Factor Explorer with 7-step cascade timeline.
- Three-label dashboard page live.
- Edition diff viewer live.
- Stationary combustion slice Certified (Sequence 2): 40 fuels, EPA + DESNZ + IPCC Tier-1 defaults.
- First external beta developer signup gate opens (waitlisted; no payments yet).

### Week 7 — Pricing + SKUs + Stripe

- SKU catalog (F-085) frozen: 4 tiers × 3 sub-tiers (Pro) + 8 Premium packs + OEM addons.
- Stripe products/prices provisioned. Test-mode end-to-end checkout + metering verified.
- Pricing page (F-112) live.
- Policy mapping stub (F-059): a thin `policy_mapping.py` exposes `applies_to(entity, activity, jurisdiction, date)` over CBAM + CSRD + SB253 — seeds Policy Graph v0.

### Week 8 — Stripe live + Scope 3 Preview

- Stripe live-mode flip. First paid API key issued internally.
- Scope 3 Cat 1–4 slice ships Preview (Sequence 3): spend-based + average-data methods.
- Webhooks (F-077) end-to-end tested with a partner sandbox.
- Signed receipts (F-066) enforced on every 2xx response in middleware.

### Week 9 — Operator Console v1

- Operator Console (F-110) ships: Ingestion, Mapping Workbench, QA Dashboard, Diff Viewer.
- Gold eval set grown to 500 items across 7 method packs.
- Scope 3 Cat 5+6+7+8 slice prep.

### Week 10 — Scope 3 downstream Certified + Approval workflow

- Scope 3 downstream + refrigerants Certified (Sequence 4).
- Operator Console Approval Workflow + Override Manager + Impact Simulator live.
- 4-eyes enforcement verified: same user cannot author + approve.
- Second Certified edition cut and published.

### Week 11 — Freight + CBAM

- Freight slice Certified (Sequence 5 part 1): ISO 14083 lane factors, GLEC defaults, mode-chain model.
- CBAM slice Certified (Sequence 5 part 2): CN-code mapping signed off, Q-report factor set published.
- 5 pilot design partners onboarded in staging.

### Week 12 — Launch

- Public launch: `factors.greenlang.io` live, `api.greenlang.io/v1/` live, `developers.greenlang.io` live, `ops.greenlang.io` live (internal only).
- Two Certified editions available (Electricity + Combustion); 3 Preview editions (Scope 3 CBAM/Freight Certified in current edition).
- 8 paying logos target: 3 CBAM/Comply design partners converted from pilots, 2 Consulting/Platform, 3 Developer Pro self-service (ad-funnel + content).
- KPIs dashboard (§11) live in Operator Console.

### Post-launch (FY27 Q2 — Q4)

- FY27 Q2: EC3 + FAOSTAT + LSR wire-up; ship Agrifood + Land + EPD Preview.
- FY27 Q3: PCAF license signed; Finance proxy Certified.
- FY27 Q4: Ecoinvent OEM negotiated; Product-carbon Preview → Certified.

---

## 11. KPIs

Per the CTO — **three separate factor counts, never one vanity number:**

1. **Factors cataloged** — raw records present in the catalog, regardless of quality. Baseline today: ~25 600 lines of YAML across 19 sources. Target by EOFY27: 50 000+.
2. **Factors QA-certified** — records that passed the full QA suite (`quality/batch_qa.py`), have a signed release (`quality/release_signoff.py`), and ship under `factor_status=certified`. Target by EOFY27: 1 500.
3. **Factors usable-through-API-in-production** — records hit by a paying customer in the last 30 days. Target by EOFY27: 500 distinct factors actively resolved.

Publicly disclosed on the three-label dashboard (§8.1 `/dashboard`). The gap between (1) and (2) and between (2) and (3) is the truth about coverage.

### 11.1 Product KPIs

| KPI | Target | Measured via |
|---|---|---|
| Top-1 resolve accuracy on gold eval set | ≥ 85% to promote a slice to Certified | `matching/evaluation.py` on CI |
| p95 `/factors/resolve` latency | < 200 ms | Prometheus (`observability/prometheus.py`) |
| p99 `/factors/resolve` latency | < 500 ms | Prometheus |
| API availability (monthly) | 99.9% (Pro), 99.95% (Enterprise) | `ga/sla_tracker.py` |
| Batch job completion (10k items) | p95 < 60 s | Prometheus |
| Webhook delivery success | ≥ 99% first-attempt, 100% within 24h | `webhooks.py` delivery log |
| Non-negotiable violations in prod | 0 | CI + prod validators |

### 11.2 Commercial KPIs

| KPI | Target EOFY27 |
|---|---|
| Paying logos | 8 |
| ARR | $0.7M |
| Recognized revenue | $0.3M |
| Developer signups (free + paid) | 4 000 |
| Community contributors (≥ 1 merged PR) | 80 |
| API calls (monthly, all tiers) | 25M |
| NPS from pilot partners | ≥ 40 |
| Gross retention (logo) | ≥ 90% |

---

## 12. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation (tied to code) |
|---|---|---|---|---|
| R-1 | **Execution sprawl** (named #1 risk in v3 p. 14) — v1 "full scope" of 7 packs + 8 SKUs + 3 UIs + 4 tiers overwhelms 12 people. | High | Critical | Thin vertical slices (§10). Only 2 Certified slices required for week-12 launch (Electricity + Combustion); everything else ships Preview. Cleanup Sprint 1 runs in parallel per `FY27_vs_Reality_Analysis §8`. Weekly slice demo = explicit scope gate. |
| R-2 | **Methodology drift** — an agent, a consultant, or a regulator interprets a method pack differently than our code. | Medium | High | Methodology notes (`method_packs/*.py :: methodology_notes()`) are shipped in API and UI, versioned per pack. Every Certified edition gets legal + methodology counsel sign-off via `quality/release_signoff.py`. Gold eval set includes adversarial items (where default factor vs method-profile-specific factor differ). |
| R-3 | **Source license revocation** — EPA, IEA, DESNZ, ecoinvent can change terms. | Medium | High | Source registry tracks `legal_signoff_artifact` + `legal_signoff_version` per source. `connectors/license_manager.py` + `quality/license_scanner.py` run on every edition. `connectors/audit_log.py` logs every access. We hold a quarterly counsel review (process, not code). Connector-only mode for IEA/ecoinvent isolates exposure. |
| R-4 | **Regulatory change** — CBAM Q-report format or ESRS datapoint list changes mid-year. | Medium | Medium | `watch/regulatory_events.py` monitors regulator publications. `policy_mapping.py` and method packs are versioned separately from factors — swap a method pack without re-QAing the catalog. Edition simulator (`quality/impact_simulator.py`) quantifies impact before adoption. |
| R-5 | **Open-core vs Enterprise blur** — contributors accidentally land proprietary code in Apache-2.0 tree (or vice versa). | Medium | Medium | Build system splits the two trees; CI lint enforces that `greenlang/factors/` is importable without commercial code present. License headers verified on every file. Pack redistribution class is a required field (`RedistributionClass`). |
| R-6 | **Single-source dependency failure** — a parser breaks because upstream publishes a non-standard file. | High | Medium | `ingestion/parser_harness.py` + dead-letter queue + `backfill.py`. Source-watch classifies breaks as `breaking | factor-value | editorial`. Alert fires within 1 hour (`observability/` + `watch/status_api.py`). A stale source falls back to the last known good edition. |
| R-7 | **Gold eval set rot** — the set we ship at week 4 does not reflect real customer activity descriptions by week 12. | Medium | Medium | Gold set v1 is a floor, not a ceiling. Pilot partners contribute real anonymized activity descriptions into a `tests/factors/gold/` expansion. CI blocks edition publish if top-1 ≤ 85% on v1 set OR if drift > 2% appears on the 30-day rolling partner set. |
| R-8 | **Pricing race to the bottom** — Climatiq etc. undercut Pro tier on per-call price. | Medium | Medium | We do not compete on per-call price; we compete on explainability, Certified labels, method packs, audit bundles, and OEM rights. Pro tier is loss-leader; Enterprise + Platform are the margin. Every `/factors/resolve` response carries `X-GreenLang-Receipt` + `fallback_rank` + `explain` link — differentiators Climatiq does not have. |
| R-9 | **Hosted Ops costs blow out** — Postgres + pgvector + Redis + signing infra at 25M calls/mo is real money. | Medium | Low | Redis cache (`cache_redis.py`) + ETag (`api_endpoints.py` helpers) + Performance optimizer (`performance_optimizer.py`) bring read cost per call to ~0.2ms of Postgres. Batch jobs get a separate cold-path pool. Infra cost target: < 20% of ARR at launch, < 10% at $10M ARR. |
| R-10 | **Non-negotiables break real customer workflows** — e.g., a customer wants only CO2e rollups and is annoyed by gas vectors. | Low | Medium | API surfaces both: response always has gas vectors (non-negotiable #1), but also has a `derived_co2e` convenience field. Docs and SDK make the one-field usage easy. The non-negotiable is about **storage** and **derivation**, not about **display**. |
| R-11 | **Auditor rejection** — a Big-4 auditor rejects a GreenLang-backed number because the `Explainability` rationale is not deep enough. | Medium | High | Audit bundle export (`POST /v1/audit-bundle`) ships the full verification chain (raw source + parser log + reviewer decision + signed receipt). Design-partner pilots each close with a dry-run audit. Known auditor firms (Big-4 climate practice leads) are invited to review the bundle format at week 8. |

---

## Appendix A — Module coverage matrix (feature → code path)

This is a one-page sanity check that v1 features map to existing code. `*` = partial / needs finishing. `^` = new for v1.

```
API router wiring              → greenlang/factors/api_endpoints.py *
Auth                           → greenlang/factors/api_auth.py
Auth + metering middleware     → greenlang/factors/middleware/auth_metering.py
Rate limiter                   → greenlang/factors/middleware/rate_limiter.py
Billing — aggregator           → greenlang/factors/billing/aggregator.py
Billing — metering             → greenlang/factors/billing/metering.py
Billing — Stripe provider      → greenlang/factors/billing/stripe_provider.py
Billing — usage sink           → greenlang/factors/billing/usage_sink.py
Billing — webhook handler      → greenlang/factors/billing/webhook_handler.py
Canonical record v1.0          → greenlang/data/canonical_v2.py + emission_factor_record.py
Catalog repo (SQLite)          → greenlang/factors/catalog_repository.py
Catalog repo (Postgres)        → greenlang/factors/catalog_repository_pg.py
CLI                            → greenlang/factors/cli.py + __main__.py
Connectors (ecoinvent, IEA,    → greenlang/factors/connectors/*
  Electricity Maps, license
  manager, audit log)
Edition manifest               → greenlang/factors/edition_manifest.py
Entitlements (8 packs, OEM)    → greenlang/factors/entitlements.py
ETL                            → greenlang/factors/etl/
GA billing / readiness / SLA   → greenlang/factors/ga/*
Gold eval set                  → tests/factors/gold/ ^
Index manager                  → greenlang/factors/index_manager.py
Ingestion — artifacts          → greenlang/factors/ingestion/artifacts.py
Ingestion — bulk               → greenlang/factors/ingestion/bulk_ingest.py
Ingestion — fetchers           → greenlang/factors/ingestion/fetchers.py
Ingestion — parsers (19)       → greenlang/factors/ingestion/parsers/*.py
Ingestion — parser harness     → greenlang/factors/ingestion/parser_harness.py
Ingestion — synthetic data     → greenlang/factors/ingestion/synthetic_data.py
Inventory / coverage matrix    → greenlang/factors/inventory.py
Mapping (13 crosswalks)        → greenlang/factors/mapping/*.py
Mapping REST                   → greenlang/factors/api/mapping.py ^
Matching — embedding           → greenlang/factors/matching/embedding.py
Matching — eval harness        → greenlang/factors/matching/evaluation.py
Matching — LLM rerank          → greenlang/factors/matching/llm_rerank.py
Matching — pgvector            → greenlang/factors/matching/pgvector_index.py
Matching — pipeline            → greenlang/factors/matching/pipeline.py
Matching — suggestion agent    → greenlang/factors/matching/suggestion_agent.py
Method packs (7)               → greenlang/factors/method_packs/*.py
Metering                       → greenlang/factors/metering.py
Non-negotiable validators      → greenlang/data/canonical_v2.py :: validate_non_negotiables
Observability — health         → greenlang/factors/observability/health.py
Observability — Prometheus     → greenlang/factors/observability/prometheus*.py
Observability — SLA            → greenlang/factors/observability/sla.py
Onboarding (health / partner / → greenlang/factors/onboarding/*
  sample queries)
Ontology — chemistry           → greenlang/factors/ontology/chemistry.py
Ontology — geography           → greenlang/factors/ontology/geography.py
Ontology — GWP sets            → greenlang/factors/ontology/gwp_sets.py
Ontology — heating values      → greenlang/factors/ontology/heating_values.py
Ontology — methodology         → greenlang/factors/ontology/methodology.py
Ontology — unit graph          → greenlang/factors/ontology/unit_graph.py + units.py
OpenAPI spec                   → greenlang/factors/api/openapi.yaml ^
Performance optimizer          → greenlang/factors/performance_optimizer.py
Pilot (provisioner/registry/   → greenlang/factors/pilot/*
  telemetry/feedback)
Policy mapping                 → greenlang/factors/policy_mapping.py *
Quality — audit export         → greenlang/factors/quality/audit_export.py
Quality — batch QA             → greenlang/factors/quality/batch_qa.py
Quality — consensus            → greenlang/factors/quality/consensus.py
Quality — cross-source         → greenlang/factors/quality/cross_source.py
Quality — dedupe               → greenlang/factors/quality/dedup_engine.py
Quality — escalation           → greenlang/factors/quality/escalation.py
Quality — impact simulator     → greenlang/factors/quality/impact_simulator.py
Quality — license scanner      → greenlang/factors/quality/license_scanner.py
Quality — promotion            → greenlang/factors/quality/promotion.py
Quality — release sign-off     → greenlang/factors/quality/release_signoff.py
Quality — review queue         → greenlang/factors/quality/review_queue.py
Quality — review workflow      → greenlang/factors/quality/review_workflow.py
Quality — rollback             → greenlang/factors/quality/rollback.py
Quality — SLA                  → greenlang/factors/quality/sla.py
Quality — validators           → greenlang/factors/quality/validators.py
Quality — versioning           → greenlang/factors/quality/versioning.py
Redis cache                    → greenlang/factors/cache_redis.py
Regulatory tagger              → greenlang/factors/regulatory_tagger.py
Resolution — engine            → greenlang/factors/resolution/engine.py
Resolution — request           → greenlang/factors/resolution/request.py
Resolution — result            → greenlang/factors/resolution/result.py
Resolution — tiebreak          → greenlang/factors/resolution/tiebreak.py
Search cache                   → greenlang/factors/search_cache.py
Security                       → greenlang/factors/security/
Service (compare/status)       → greenlang/factors/service.py
Signing (HMAC + Ed25519)       → greenlang/factors/signing.py
Source registry                → greenlang/factors/source_registry.py
SDK — Python                   → greenlang/factors/sdk/python/
SDK — TypeScript               → greenlang/factors/sdk/ts/
Tenant overlay                 → greenlang/factors/tenant_overlay.py
Tier enforcement               → greenlang/factors/tier_enforcement.py
UI — Factor Explorer           → web/factor-explorer/ ^
UI — Operator Console          → web/operator-console/ ^
UI — Developer portal          → web/dev-portal/ ^
UI — Pricing                   → web/marketing/pricing/ ^
Watch — change classification  → greenlang/factors/watch/change_classification.py
Watch — change detector        → greenlang/factors/watch/change_detector.py
Watch — changelog draft        → greenlang/factors/watch/changelog_draft.py
Watch — cross-edition          → greenlang/factors/watch/cross_edition_changelog.py
Watch — doc diff               → greenlang/factors/watch/doc_diff.py
Watch — regulatory events      → greenlang/factors/watch/regulatory_events.py
Watch — release orchestrator   → greenlang/factors/watch/release_orchestrator.py
Watch — rollback CLI           → greenlang/factors/watch/rollback_cli.py
Watch — rollback edition       → greenlang/factors/watch/rollback_edition.py
Watch — scheduler              → greenlang/factors/watch/scheduler.py
Watch — source watch           → greenlang/factors/watch/source_watch.py
Watch — status API             → greenlang/factors/watch/status_api.py
Webhooks                       → greenlang/factors/webhooks.py
```

Legend: `*` = existing code needs finishing for v1; `^` = new for v1; everything else = BUILT.

---

## Appendix B — Sign-off

| Role | Name | Sign-off date | Signature |
|---|---|---|---|
| Founder | | | |
| CTO | | | |
| Head of Engineering | | | |
| GTM Lead | | | |
| Security Lead | | | |

---

*End of PRD.*
