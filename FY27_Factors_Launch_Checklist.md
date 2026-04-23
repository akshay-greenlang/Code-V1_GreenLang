# GreenLang Factors — FY27 Launch Checklist

**Prepared for:** Akshay
**Prepared on:** 23 Apr 2026
**Basis:** `GreenLang_Factors_FY27_Product_Development_Proposal.pdf` (16 Apr 2026), CTO spec (13 Apr 2026), `FY27_vs_Reality_Analysis.md` (19 Apr 2026), and code-walk on 23 Apr 2026.
**Goal:** Close the last 10–15% so GreenLang Factors is commercially shippable to a developer, consultant, or enterprise buyer in 6–8 weeks. Every task is tagged against existing code paths. No green-field.

---

## 0. How to read this document

Each task has:

- **What** — the deliverable, stated as a done/not-done check.
- **Why** — which CTO non-negotiable or FY27 proposal clause it satisfies.
- **Where** — the file paths that already exist in `Code-V1_GreenLang/`.
- **Effort** — rough engineering days.
- **Owner** — placeholder for assignment.
- **Acceptance** — the visible proof the task is closed.

Tasks are sequenced into three tracks that run in parallel:

- Track A — Hosted API + version pinning (deployment).
- Track B — Publisher-grade proof (gold-label eval, three-label dashboard, v1.0 release cut).
- Track C — Commercial surface (pricing page, developer portal, OEM, signed receipts enforcement).

An 8-week Gantt is in §7.

---

## 1. Non-negotiables check (CTO spec § "Never rules")

Before any launch task, confirm the six non-negotiables hold. These are not new work — they are audits of what already exists.

| # | Non-negotiable | Current state | File to verify |
|---|---|---|---|
| 1 | Never store only CO2e. Store gas components; derive CO2e by GWP set. | Holds. `numerator` is per-gas (`co2`, `ch4`, `n2o`, `co2e`) and `gwp_set` is required on every record. | `greenlang/factors/ontology/chemistry.py`, `greenlang/factors/ontology/gwp_sets.py` |
| 2 | Never overwrite a factor. Version everything. | Holds. | `greenlang/factors/quality/versioning.py`, `greenlang/factors/watch/rollback_edition.py` |
| 3 | Never hide fallback logic. Always explain. | Holds in engine; not enforced everywhere in API. See Track A-4. | `greenlang/factors/resolution/engine.py::build_factor_explain`, `api_endpoints.py::explain` |
| 4 | Never mix licensing classes (open / licensed / customer-private). | Holds at record level. Needs runtime gate on API (see Track A-5). | `greenlang/factors/source_registry.py`, `greenlang/factors/entitlements.py`, `greenlang/factors/tier_enforcement.py` |
| 5 | Never ship a factor without validity dates and source version. | Holds in schema. CI guard missing. See Track B-2. | `greenlang/factors/quality/validators.py` |
| 6 | Policy workflows must call method profiles, not raw factors. | Partially holds. Compliance PACKs still call factor APIs directly in places. See Track B-6. | `greenlang/factors/method_packs/registry.py`, `greenlang/factors/policy_mapping.py` |

**Audit task (0.5 day):** one engineer runs this table end-to-end on a checklist PR before any launch task ships. Anything that does not hold becomes a P0 blocker.

---

## 2. Track A — Hosted Factors API v0 (deployment)

This is the single biggest gap. The engine exists; no hosted endpoint exists that a design partner can call.

### A-1. Stand up `api.greenlang.ai` with FastAPI + Gunicorn behind AWS ALB

- **What:** Public HTTPS endpoint serving `greenlang.factors.api_endpoints` under a versioned path (`/v1/...`). TLS, health checks, readiness probes, blue/green slots.
- **Why:** FY27 proposal § 3 ("Public + enterprise API with version pinning"); CTO spec § 9 "Developer surface — REST / GraphQL API".
- **Where:** `greenlang/factors/api_endpoints.py` (routes already written), `deployment/Dockerfile.api` (image exists), `deployment/docker-compose-unified.yml` (compose already present), `deployment/database/` (Postgres setup present).
- **Effort:** 4 days.
- **Owner:** _TBA (platform eng)._
- **Acceptance:** `curl https://api.greenlang.ai/v1/health` returns 200 with the running edition ID. Prometheus metrics scraped by `greenlang/factors/observability/prometheus_exporter.py`.

### A-2. Version-pinning via edition manifest

- **What:** Every API response includes the resolved `edition_id` and `factor_version`. Clients can pin to an edition via `X-GL-Edition` header or `?edition=` query.
- **Why:** CTO non-negotiable "Never overwrite a factor"; FY27 proposal § 3.
- **Where:** `greenlang/factors/edition_manifest.py` (manifest object exists), needs middleware plumbing in `greenlang/factors/middleware/auth_metering.py`.
- **Effort:** 2 days.
- **Acceptance:** A call against `edition=2026-04-v1.0` returns byte-identical factors three months later.

### A-3. Auth, rate limits, metering enforced at middleware

- **What:** API key or OAuth2 required on every non-public endpoint. Rate limit by plan (Community / Developer Pro / Enterprise). Every call metered to `usage_sink`.
- **Why:** FY27 proposal § 3 ("enterprise API"); CTO spec commercial packaging.
- **Where:** `greenlang/factors/middleware/auth_metering.py`, `greenlang/factors/middleware/rate_limiter.py`, `greenlang/factors/security/api_key_manager.py`, `greenlang/factors/billing/metering.py`, `greenlang/factors/billing/usage_sink.py`. All files exist; they need to be wired into the FastAPI app lifespan.
- **Effort:** 3 days.
- **Acceptance:** Community-tier key throttled at 60 req/min; Developer Pro key at 1,000 req/min; Enterprise key unlimited. Every request appears in `usage_sink` within 5 seconds.

### A-4. `/explain` is a first-class primitive

- **What:** `GET /v1/factors/{id}/explain` and `POST /v1/resolve` both return the full explain payload: chosen factor, alternates considered, why this one won, source + version, quality score, uncertainty band, gas breakdown, CO2e basis, assumptions, deprecation status. No endpoint that returns a factor returns only a number.
- **Why:** CTO non-negotiable "Never hide fallback logic"; CTO spec § 5 "The output should never just be a number."
- **Where:** `greenlang/factors/api_endpoints.py::build_factor_explain` exists (verified lines 650–663). Missing: a contract test that asserts every factor response includes an `explain` object. Default behaviour is "always include"; a `?compact=true` flag is the only way to suppress it.
- **Effort:** 1.5 days.
- **Acceptance:** OpenAPI contract test `tests/factors/test_openapi_explain_contract.py` passes; fails if any response omits the `explain` object without `compact=true`.

### A-5. Licensing-class enforcement on every call

- **What:** A request for a factor whose `licensing.redistribution_class` is `licensed` or `customer-private` returns 402 (Payment Required) / 403 (Forbidden) unless the caller's entitlement grants access. An OEM caller can only see factors under their OEM redistribution grant.
- **Why:** CTO non-negotiable "Never mix licensing classes"; spec § "Licensing rule".
- **Where:** `greenlang/factors/entitlements.py`, `greenlang/factors/tier_enforcement.py`, `greenlang/factors/tenant_overlay.py` — all present; need a licensing guard decorator on every factor-returning route in `api_endpoints.py`.
- **Effort:** 2 days.
- **Acceptance:** Negative test: a Developer Pro key attempting to resolve an ecoinvent-class factor returns 403 with a clear `licensing_gap` error code and an upgrade link.

### A-6. Signed receipts enforced at middleware (not optional)

- **What:** Every API response carries an `X-GL-Signature` header. The signature covers edition ID, factor IDs, timestamps, caller ID. Clients can verify offline using the published JWK set.
- **Why:** CTO spec § "Developer surface — signed result receipts"; FY27 proposal § 7 ("auditability is the enterprise moat").
- **Where:** `greenlang/factors/signing.py` (exists), `greenlang/factors/middleware/signed_receipts.py` (file present but needs to be registered in the FastAPI middleware stack in `api_endpoints.py`).
- **Effort:** 1.5 days.
- **Acceptance:** Offline verify tool (shipped in `greenlang/factors/sdk/python`) can independently validate a signed response from a saved HTTP log. Contract test in `tests/factors/test_signed_receipts_e2e.py`.

### A-7. Hosted Postgres + Redis + Prometheus + Grafana + Sentry

- **What:** Managed Postgres (RDS or Aurora) for `catalog_repository_pg`, Redis for `cache_redis` and `search_cache`, Prometheus + Grafana dashboards for SLA, Sentry for errors.
- **Why:** FY27 proposal § 6 ("Enterprise support layer with SLAs"); CTO spec § "Developer surface — hosted explain logs".
- **Where:** `greenlang/factors/catalog_repository_pg.py`, `greenlang/factors/cache_redis.py`, `greenlang/factors/search_cache.py`, `greenlang/factors/observability/prometheus.py`, `greenlang/factors/observability/sla.py`, `greenlang/factors/ga/sla_tracker.py`.
- **Effort:** 3 days (largely IaC).
- **Acceptance:** Grafana dashboard at `grafana.greenlang.ai/factors-sla` shows p50/p95/p99, error rate, and edition-hit rate. SLO alert fires on p95 > 500ms.

### A-8. OpenAPI 3.1 spec published at `api.greenlang.ai/openapi.json`

- **What:** Versioned OpenAPI spec with examples for every endpoint. Swagger UI at `/docs`. Redoc at `/redoc`.
- **Where:** OpenAPI schema is auto-generated by FastAPI; needs review + examples. `tests/factors/test_openapi_contract.py` exists — extend it.
- **Effort:** 1 day.
- **Acceptance:** Spec validates against Spectral ruleset; examples render in both Swagger and Redoc.

**Track A total: ~18 engineering days. Two engineers over 3 calendar weeks.**

---

## 3. Track B — Publisher-grade proof

This track is what lets you quote numbers externally without lying. It is also what separates GreenLang from "a CSV of emissions factors."

### B-1. Public gold-label evaluation set (300–500 activity descriptions)

- **What:** A checked-in JSON/YAML set of ≥300 representative activity strings with their ground-truth factor selections, jurisdictions, method profiles, and expected fallback ranks. CI runs the matching pipeline against this set on every PR and fails if precision@1 drops below a threshold.
- **Why:** FY27 proposal § 7 ("Target KPIs: factors cataloged vs factors QA-certified vs factors usable-through-API in production"). You cannot defend a "Certified" count without this.
- **Where:** `greenlang/factors/matching/evaluation.py` (harness exists), `greenlang/factors/matching/pipeline.py`. Missing: `greenlang/factors/data/gold_set/` directory with the labeled cases, and a GitHub Actions job in `.github/workflows/factors_gold_eval.yml`.
- **Effort:** 5 days (data curation is the real cost; code is 1 day).
- **Acceptance:** CI fails on a PR that degrades precision@1 by more than 2 points. Gold set is published at `data.greenlang.ai/gold-set-v1.0.json`.

### B-2. Three-label dashboard (Certified / Preview / Connector-only)

- **What:** A public page at `greenlang.ai/factors/coverage` that shows three live counts per source and per factor family. Refreshed nightly from the catalog.
- **Why:** FY27 proposal § 7 explicit requirement.
- **Where:** Label fields already exist on the factor record schema. Missing: aggregation query in `greenlang/factors/ga/readiness.py` (file exists; add a `label_counts()` method), a Next.js page in `frontend/src/pages/FactorsCatalogStatus.tsx` (page file already exists — wire it to the readiness endpoint).
- **Effort:** 3 days.
- **Acceptance:** The public page shows three numbers per family (e.g., Electricity — Certified 412, Preview 1,203, Connector-only 5,412) and the numbers match `SELECT label, COUNT(*) FROM factors GROUP BY label`.

### B-3. v1.0 Certified edition cut

- **What:** `release_signoff.py` run against the current catalog. Produces an immutable edition manifest, signed, tagged `2026-05-v1.0`. This is the edition the hosted API ships with on day one.
- **Why:** FY27 proposal § 10 ("launch in narrow initial form"); CTO non-negotiable "Never ship a factor without validity dates and source version."
- **Where:** `greenlang/factors/quality/release_signoff.py`, `greenlang/factors/edition_manifest.py`, `greenlang/factors/signing.py`. The tooling runs; the cut has not been executed.
- **Effort:** 2 days (execution + sign-off meetings).
- **Acceptance:** `gl factors edition show 2026-05-v1.0` returns a signed manifest. Manifest hash is published in the release notes and pinned to the v1.0 git tag.

### B-4. Narrow-ship v1 coverage validated end-to-end

- **What:** For each spec § "FY27 launch scope" family, a pass/fail matrix that shows at least the minimum coverage resolves cleanly through the full cascade and explain path:
  - Electricity — India, EU/UK, US (location-based, market-based, residual-mix).
  - Fuel combustion — natural gas, diesel, coal, LPG across India, EU, US.
  - Refrigerants — R-22, R-32, R-134a, R-410A, R-404A with AR6 GWP.
  - Freight — road, sea, air with WTW/TTW labels.
  - Purchased goods proxies — steel, aluminium, cement, fertilizer, plastics, paper.
  - Method profiles — corporate (GHG Protocol), Scope 2 location + market, Scope 3 cat. 1 + 4, CBAM selectors.
- **Where:** `tests/factors/` has 92 files; add one integration file per family: `tests/factors/test_launch_v1_coverage_{family}.py`. Each test resolves a representative activity and asserts that `explain` includes source, version, quality, and chosen fallback rank.
- **Effort:** 4 days.
- **Acceptance:** GitHub Actions matrix job `factors-launch-v1-coverage` is green on every PR.

### B-5. Factor Quality Score dashboard (0–100) live per factor family

- **What:** Composite FQS rendered in the `FactorsQADashboard.tsx` page. Drill-down to component scores: temporal, geographic, technological, verification, completeness.
- **Where:** `greenlang/factors/quality/composite_fqs.py` (exists), `frontend/src/pages/FactorsQADashboard.tsx` (page exists — wire the data).
- **Effort:** 2 days.
- **Acceptance:** QA Dashboard shows a per-family distribution. You can drill down to a single factor and see all five component scores.

### B-6. Policy workflows call method profiles, not raw factors

- **What:** The CBAM, CSRD, SB 253 PACKs call `resolve(method_profile=...)`, not `get_factor(id=...)`. This is the CTO's sixth non-negotiable.
- **Where:** `greenlang/factors/policy_mapping.py` (exists), `greenlang/factors/method_packs/registry.py`. Review touch points in `applications/GL-CBAM-APP/` and `applications/GL-CSRD-APP/`. Grep for direct catalog calls and refactor.
- **Effort:** 3 days.
- **Acceptance:** A grep guard test `tests/factors/test_no_direct_catalog_calls_in_packs.py` fails the build if any pack calls `catalog_repository.get(...)` directly.

**Track B total: ~19 engineering days. One engineer + 0.5 methodology lead over 4 calendar weeks.**

---

## 4. Track C — Commercial surface

This is what turns the engine into something a buyer can swipe a credit card against.

### C-1. Pricing page + live Stripe SKUs

- **What:** A `greenlang.ai/pricing` page showing four SKUs: Community (free), Developer Pro (usage), Consulting / Platform (annual + usage), Enterprise (ACV). Self-serve checkout for Community + Developer Pro.
- **Why:** CTO spec § "Commercial packaging — This packaging is the most sensible interpretation of the plan's API + enterprise support, usage/data revenue logic, and open-core motion."
- **Where:** `greenlang/factors/billing/skus.py` (SKU model exists), `greenlang/factors/billing/stripe_provider.py` (integration stub exists), `greenlang/factors/billing/aggregator.py`, `greenlang/factors/billing/webhook_handler.py`. Frontend: new `frontend/src/pages/PricingPage.tsx`.
- **Effort:** 4 days.
- **Acceptance:** A new signup through `greenlang.ai/pricing` creates a Stripe customer, generates an API key, returns a Quickstart page, and appears as a tenant in the admin console.

### C-2. Developer portal

- **What:** A hosted doc site at `developers.greenlang.ai` with: Quickstart, Concepts (factor record, editions, licensing classes), API reference (from OpenAPI), SDK install (Python + TypeScript), gold-set page, changelog.
- **Where:** `greenlang/factors/sdk/python`, `greenlang/factors/sdk/ts`, `greenlang/factors/sdk/README.md`, `greenlang/factors/sdk/CHANGELOG.md`, `greenlang/factors/sdk/RELEASE_NOTES_v1.1.0.md`. Docusaurus or Mintlify site, source-of-truth in `docs/factors-portal/`.
- **Effort:** 4 days.
- **Acceptance:** `pip install greenlang-factors` and `npm install @greenlang/factors` both work. A Quickstart that calls `/v1/resolve` completes in under 5 minutes end-to-end.

### C-3. SDK v1.0 release in both languages

- **What:** Pin SDK versions at 1.0.0, publish to PyPI and npm. The SDK contains: typed client, offline explain verifier (from Track A-6), edition pinning helper, retries, rate-limit-aware backoff.
- **Where:** `greenlang/factors/sdk/python/pyproject.toml`, `greenlang/factors/sdk/ts/` (verify package.json exists).
- **Effort:** 2 days.
- **Acceptance:** Both packages visible on PyPI / npm. Install, authenticate, resolve, explain — end-to-end example in README completes.

### C-4. Operator console (tenant-private, internal)

- **What:** The operator surfaces listed in the CTO spec § 9 — factor explorer, source ingestion console, mapping workbench, QA dashboard, diff viewer, approval workflow, customer override manager, impact simulator. These pages already exist in `frontend/src/pages/` but need to be wired to live API data and gated behind admin auth.
- **Where:** `FactorsExplorer.tsx`, `FactorsSourceConsole.tsx`, `FactorsMappingWorkbench.tsx`, `FactorsQADashboard.tsx`, `FactorsDiffViewer.tsx`, `FactorsApprovalQueue.tsx`, `FactorsOverrideManager.tsx`, `FactorsImpactSimulator.tsx`. Backends: `greenlang/factors/quality/review_queue.py`, `greenlang/factors/quality/impact_simulator.py`, `greenlang/factors/watch/doc_diff.py`.
- **Effort:** 5 days (wiring + auth).
- **Acceptance:** A methodology lead can log in, propose a factor change, run the impact simulator, see the diff, approve it, and see the new version cut into a Preview edition — all through the UI.

### C-5. OEM white-label onboarding path

- **What:** A flow where a third-party platform (a consultancy, an ERP integrator) can sign up as an OEM, configure their sub-tenants, brand the UI, and redistribute factors under their license grant. This matches CTO spec § "For platforms — OEM embed, white-label branding, sub-tenant entitlements, signed responses, rate-limited bulk resolution, redistribution controls."
- **Where:** `greenlang/factors/tenant_overlay.py`, `greenlang/factors/entitlements.py`, `greenlang/factors/onboarding/partner_setup.py`. The objects exist; the onboarding flow and branding config do not.
- **Effort:** 5 days.
- **Acceptance:** A dummy OEM account can provision two sub-tenants, apply a logo + color scheme, and the resulting API responses include the OEM's branding metadata. Redistribution to sub-tenants is gated by license class.

### C-6. CBAM battlecard + Factors one-pager

- **What:** Two PDFs. The CBAM battlecard positions the hosted Factors API as the factor backbone for a CBAM pilot. The Factors one-pager is the standalone developer / consultant pitch.
- **Where:** Owned by commercial / founder, not engineering. `docs/` has historical material that can be repurposed.
- **Effort:** 2 days (content), 1 day (design).
- **Acceptance:** Both PDFs live on the website and in a shared Drive folder.

**Track C total: ~23 engineering days + 3 commercial days. Two engineers + founder over 3 calendar weeks.**

---

## 5. Out of scope for v1.0 (intentionally)

These are in the spec but not on the launch path. They are parked, not dropped.

| Deferred | Why | When |
|---|---|---|
| GraphQL API | REST + SDK covers the v1 developer need. GraphQL ships when a design partner explicitly asks. | FY27 Q3 |
| Full Product Carbon / LCI Premium pack (ecoinvent, EPD licensed) | License negotiations and legal review take longer than 8 weeks. | FY27 Q4 |
| Finance Proxy Pack commercial SKU | Factors exists; PCAF-aligned SKU depends on FY29 FinanceOS pull. | FY29 |
| Land / Removals Pack commercial SKU | Same. Spec says "exist even if not fully commercialized on day one." | FY28 |
| Policy Graph integration as a dedicated service | Track B-6 covers the non-negotiable. Full Policy Graph service is a separate FY27 core-platform deliverable. | FY27 Q2 |
| GreenLang Foundation (standards body) | The plan calls for it; not a launch blocker. | FY27 Q3 |

---

## 6. Launch gate — the single yes/no

A release is "launch-ready" when the following 10 gates all pass. This is the number the CTO should check on the go/no-go call.

1. `curl https://api.greenlang.ai/v1/health` returns 200 and edition `2026-05-v1.0`.
2. `POST /v1/resolve` for India electricity, FY2027, corporate inventory, location-based returns a factor with full explain payload in under 500ms (p95).
3. CI on `main` is green, including the Track B-4 launch-v1 coverage matrix and the Track B-1 gold-set job.
4. Three-label dashboard at `greenlang.ai/factors/coverage` is live and matches the catalog.
5. Pricing page at `greenlang.ai/pricing` is live; a test card completes Developer Pro checkout.
6. SDK v1.0.0 is on PyPI and npm.
7. Offline signed-receipt verifier validates a saved API response against the published JWK set.
8. A Developer Pro key hits a 403 when requesting a licensed factor it does not own.
9. Operator console impact simulator successfully proposes → diffs → approves → promotes a test factor change.
10. CBAM battlecard + Factors one-pager are published on the website.

If all 10 pass, the release is cut. If any fail, that task is the P0 for the next sprint.

---

## 7. 8-week sequence

```
Week 1  ▮ A-1 hosted API stood up               ▮ B-1 gold-set v0 (200 cases)  ▮ C-3 SDK pin + publish prep
Week 2  ▮ A-3 auth/rate/metering                ▮ B-2 three-label dashboard    ▮ C-6 battlecard/one-pager
Week 3  ▮ A-5 licensing enforcement             ▮ B-4 coverage tests           ▮ C-2 dev portal v0
Week 4  ▮ A-6 signed receipts enforced          ▮ B-5 FQS dashboard            ▮ C-1 pricing page + Stripe
Week 5  ▮ A-2 version pinning finalized         ▮ B-6 policy-profile guard     ▮ C-4 operator console wiring
Week 6  ▮ A-4 explain primitive contract test   ▮ B-1 gold-set v1 (350 cases)  ▮ C-4 impact simulator live
Week 7  ▮ A-7 hosted infra SLO dashboards       ▮ B-3 v1.0 edition cut         ▮ C-5 OEM onboarding path
Week 8  ▮ A-8 OpenAPI published                 ▮ Launch gate runs (§6)        ▮ C-3 SDK v1.0 release
```

Weeks 1–4 are the "code" half. Weeks 5–8 are the "cut and ship" half. The first three paid pilot conversations (CBAM first) should open in Week 6 and close in Week 10.

---

## 8. Team and cost

- **2 platform engineers** full-time for 8 weeks.
- **1 methodology lead** at 50% for Weeks 1–7 (mostly on Track B).
- **1 frontend engineer** for Weeks 3–8 (Tracks B + C).
- **1 founder** at 20% for Track C commercial work (Weeks 2–8).

Total engineering cost ≈ 20 person-weeks. Matches the FY27 proposal's "6–8 weeks of finish work" called out in the Reality Analysis.

---

## 9. Risk and the "don't do this" list

- **Do not** try to land all seven method-pack commercial SKUs in v1.0. v1.0 is Corporate + Electricity + Freight + CBAM selectors. The rest exist as code, not as SKUs.
- **Do not** launch without the three-label dashboard. A marketing page that says "25,000+ factors" without Certified / Preview / Connector counts will attract the one auditor question you cannot answer.
- **Do not** let the CBAM pitch collapse into a CBAM point tool. The Factors API is the substrate; CBAM is one application on top.
- **Do not** expose licensed factors (ecoinvent, EC3) without explicit entitlement. The redistribution class field is there for a reason, and the commercial risk of getting this wrong is higher than any SaaS downside.
- **Do not** skip the signed-receipt middleware. Auditability is the enterprise moat, and a receipt that has to be added later is a receipt that will not be trusted.
- **Do not** start Track D (anything not in this checklist). Execution sprawl is named the #1 risk on v3 page 14 for a reason.

---

## 10. The single answer to "is this done?"

**Not yet. It will be in 8 weeks if this checklist runs and no new scope is added.**

The engine is ~90% of the CTO blueprint. The commercial productisation is ~60% of what a buyer can swipe against. This document closes the second number to 100% without adding any new science.

---

*End of checklist. Update the gate counters in §6 each sprint to track drift.*
