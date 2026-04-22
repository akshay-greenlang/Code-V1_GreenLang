# GreenLang Factors API — Security Hardening Checklist

**Status**: pre-deployment gate for v1 Certified public launch.
**Owners**: platform security lead + Factors engineering lead.
**Last verified**: 2026-04-22.

This document is the go/no-go checklist the Factors API passes through before the first paying customer receives a signed response. Every control cites the GreenLang component (SEC-*, INFRA-*, OBS-*) that implements it and the code path that invokes it. Anything with a `TODO` is a blocker; anything with `VERIFIED` is production-gated.

---

## 1. Non-negotiables from the CTO spec (2026-04-22)

These map 1:1 to spec rules and must each clear green before public launch.

| # | Rule | Control | Verification |
|---|---|---|---|
| 1 | Never store only CO2e — always store gas components + derive | `greenlang/data/emission_factor_record.py::GHGVectors` + `GWPValues.calculate_co2e()` | JSON-Schema invariant in `config/schemas/factor_record_v1.schema.json`; CI gate `scripts/check_factor_invariants.py` |
| 2 | Never overwrite a factor — version everything | `edition_manifest.py` + `quality/versioning.py` + `watch/rollback_edition.py`; append-only in `catalog_repository_pg.py` | DB migration constraint: `factor_version NOT NULL`, unique `(factor_id, factor_version)` |
| 3 | Never hide fallback logic — every response explains | `resolution/engine.py` 7-step cascade + `resolution/result.py::ResolvedFactor.explain()` + `/explain` endpoint | OpenAPI contract mandates `alternates` + `step_label` on every explain response |
| 4 | Never mix licensing classes — open vs restricted vs licensed vs customer_private | `connectors/license_manager.py` + `quality/license_scanner.py` + tier_enforcement at response boundary | 451 `Unavailable For Legal Reasons` on any violation; CI e2e test forbids mixing |
| 5 | Never ship a factor without valid-from/valid-to + source version | JSON-Schema `required: [valid_from, valid_to, source_version]`; validators reject missing | Release signoff checklist S3 |
| 6 | Policy workflows must call method profiles, not raw factors | Only `resolve()` + `explain()` endpoints accept `method_profile`; raw factor lookup requires explicit pin | Code review rule: grep for `get_factor(edition, factor_id)` outside repo/tests is flagged |
| 7 | Keep open-core / enterprise boundaries clear | `tenant_overlay.py` + `entitlements.py` + `tier_enforcement.py`; SKU catalog at `greenlang/factors/billing/skus.py` | Stripe products prefixed `prod_factors_*`; tier visibility matrix in `docs/product/PRD-FY27-Factors.md` §7 |

---

## 2. AuthN / AuthZ

| Control | Component | Status | Notes |
|---|---|---|---|
| JWT signing secret rotation (30-day) | SEC-001 + Vault path `secret/factors/{{stage}}/jwt` | VERIFIED | `GL_JWT_SECRET` + legacy `JWT_SECRET` alias both in ExternalSecret per gotcha in deploy runbook |
| API key format enforcement (`gl_*` prefix, ≥32 chars) | `greenlang/integration/api/main.py::get_current_user` | VERIFIED | Keys issued via ops console; rotated per customer on request |
| Per-tenant isolation | `tenant_overlay.py` + DB row-level security | VERIFIED | No cross-tenant factor leak: `SELECT` always filters on `tenant_id` column |
| Role-based access (SEC-002) | `/api/v1/ops/*` routes gated on roles: admin, methodology_lead, data_curator, reviewer, release_manager, legal, support, viewer | VERIFIED | Segregation-of-duties: author ≠ approver on release signoff (`quality/review_workflow.py`) |
| Rate limiting per tier | Kong `rate-limiting-advanced` + app middleware `middleware/rate_limiter.py` | VERIFIED | Community 60/min, Pro 600/min, Enterprise 6000/min; 429 Retry-After honored |
| OEM sub-tenant enforcement | `tenant_overlay.py::OEMSubTenant` + signed receipts carry `tenant_id` | VERIFIED | Tested in `tests/factors/entitlements/` |

---

## 3. Transport + payload integrity

| Control | Component | Status |
|---|---|---|
| TLS 1.3 only (SEC-004) | Kong ingress + INFRA-006; HSTS `max-age=63072000; includeSubDomains` | VERIFIED |
| Certificate pinning for the SDK clients | Python + TS SDK bundle the GreenLang CA cert | TODO — bundle in next SDK release |
| Signed response receipts (SEC-006 + `signing.py`) | FastAPI middleware `middleware/signed_receipts.py`; HMAC-SHA256 for community/pro, Ed25519 for consulting/platform/enterprise | VERIFIED — installed on `/api/v1/factors` in `greenlang/integration/api/main.py` |
| Edition pinning in signed payload | Receipt `signed_over.edition_id` pulled from `X-GreenLang-Edition` response header | VERIFIED |
| Constant-time signature verification | `signing.py::verify_sha256_hmac` uses `hmac.compare_digest` | VERIFIED |
| Body-hash in receipt (SHA-256 of bytes client sees) | `middleware/signed_receipts.py::_body_hash` | VERIFIED |

---

## 4. Data at rest

| Control | Component | Status |
|---|---|---|
| AES-256 at rest (SEC-003) | INFRA-002 managed RDS + INFRA-004 S3 SSE-KMS | VERIFIED |
| Per-tenant encryption for private registries | Vault transit backend keyed by `tenant_id` | TODO — Enterprise customers only, ship in Week 14 per deploy runbook |
| PII scan on ingested source artifacts (SEC-011) | `greenlang/security/pii_scanner.py` run on every parser_log + raw-source write | VERIFIED |
| Backups + point-in-time recovery | RDS 7-day PITR + daily snapshots retained 30 days | VERIFIED |
| Customer data deletion (GDPR Art 17) | DELETE on `tenants` cascades to tenant_overlay + entitlements + audit_log retention 7y per SEC-005 policy | VERIFIED |

---

## 5. Licensing-class firewall

The #1 commercial risk. Every response must honor redistribution rules.

| Control | Component | Status |
|---|---|---|
| Response boundary license check | `connectors/license_manager.py::check_redistribution(factor, caller_entitlement)` called in every search / resolve / bulk-export path | VERIFIED |
| 451 on license violation with upgrade pointer | OpenAPI `factors-v1.yaml` declares 451; route handlers in `integration/api/routes/factors.py` | VERIFIED |
| Connector-only enforcement | `source_registry.yaml` `connector_only: true` flag short-circuits public bulk export at `service.py::export_factors` | VERIFIED |
| OEM redistribution rights | `entitlements.py::OEMEntitlement` + signed receipt carries `redistribution_class` for audit | VERIFIED |
| Ecoinvent / EC3 / Electricity Maps / IEA licensed connectors | All flagged `connector_only: true` in `source_registry.yaml` — values never in public bulk export | VERIFIED |
| CEDA + EXIOBASE + GLEC + PACT Pathfinder + PCAF + EC3 EPD licensed-class | Added 2026-04-22 with correct `license_class` per source registry | VERIFIED |

---

## 6. Ingestion + supply chain

| Control | Component | Status |
|---|---|---|
| Source-side change detection (SEC-008) | `watch/source_watch.py` + `change_detector.py` + `doc_diff.py` | VERIFIED |
| Parser output schema validation | `quality/validators.py::validate_canonical_row` enforces schema on every ingest | VERIFIED |
| Cross-source consistency check | `quality/cross_source.py` — deviations >5% trigger review | VERIFIED |
| Dedup at ingest | `quality/dedup_engine.py` | VERIFIED |
| Release signoff (9-step S1-S9) | `quality/release_signoff.py::release_signoff_checklist` | VERIFIED — S7+S8 raised from recommended to required for v1 per edition cutlist §11 |
| SBOM + image signing (cosign) | GHA workflow `.github/workflows/factors-api-deploy.yml` produces SPDX + signs with cosign | VERIFIED |
| Base image digest pin | `Dockerfile.factors-service` pins `python:3.11-slim-bookworm` by digest (update required per release) | VERIFIED — gotcha #3 in deploy runbook |
| Supply-chain scan (SEC-007) | GHA `lint-test` job runs `pip-audit` + `trivy image` | VERIFIED |

---

## 7. Observability + incident response

| Control | Component | Status |
|---|---|---|
| Prometheus metrics per route (OBS-001) | `observability/prometheus.py` + `prometheus_exporter.py`; Kong `prometheus` plugin | VERIFIED |
| SLOs (OBS-005) | p95 resolve <200ms, p95 explain <500ms, 99.9% availability; error budget 43min/30d | VERIFIED — `PrometheusRule` alerts in `deployment/k8s/factors/base/prometheusrule.yaml` |
| Centralized logs to Loki (INFRA-009) | All containers log JSON; Promtail sidecar ships to Loki | VERIFIED |
| Distributed tracing (OBS-003) | OTel collector sidecar on every pod; traces to Tempo | VERIFIED |
| Audit trail (SEC-005) | Every auth, every factor mutation, every override write, every release signoff writes immutable audit log entry; retained 7y | VERIFIED |
| Alert routing | OBS-004 alertmanager → PagerDuty for critical, Slack for warnings | VERIFIED |
| Incident response matrix | `docs/deployment/FACTORS-API-DEPLOY.md` §incident-response — 5 scenarios with 3-step remediation | VERIFIED |

---

## 8. Tenant + OEM boundaries

| Control | Component | Status |
|---|---|---|
| Per-tenant factor override isolation | `tenant_overlay.py`; every read scopes to `(tenant_id, factor_id)` | VERIFIED |
| OEM white-label enforcement | `tenant_overlay.py::BrandingConfig` controls response envelope customization | VERIFIED |
| Cross-tenant leak prevention | Integration test `tests/factors/entitlements/test_cross_tenant.py` asserts 0 leak on every route | VERIFIED |
| Redistribution class never downgraded in OEM sub-calls | Signed receipt carries original class + propagated to sub-tenant | VERIFIED |
| Customer-specific override never surfaced in explain alternates | `resolution/engine.py::_build_alternates` strips tenant_overlay candidates from alternate list | TODO — add regression test next sprint |

---

## 9. Pre-launch gate — green light criteria

All ☑ required for the v1 Certified public launch.

- [x] JSON-Schema v1.0 frozen + released (`config/schemas/factor_record_v1.schema.json`)
- [x] OpenAPI v1 contract frozen (`docs/api/factors-v1.yaml`)
- [x] Signed-receipts middleware installed on `/api/v1/factors` in prod FastAPI app
- [x] Composite FQS 0-100 surface exposed via `/api/v1/factors/{id}/quality`
- [x] All 7 method packs registered + 27 source registry rows validated
- [x] K8s manifests + Kong config + GHA deploy workflow in `deployment/k8s/factors/`
- [x] Reporting labels for CSRD_E1 / CA_SB253 / UK_SECR / India_BRSR on corporate packs
- [ ] First Certified edition promotion cut through `release_signoff.py` (Task #10 — blockers cleared, awaiting execution)
- [ ] Gold-label eval set ≥85% top-1 on the promoted slice (Task #6 — in flight)
- [ ] Stripe SKU catalog provisioned in live mode (Task #7 — in flight)
- [ ] `kubectl kustomize overlays/staging` dry-run green
- [ ] `tests/factors/smoke/test_staging.py` passes against staging URL
- [ ] SEC-010 security operations automation playbook reviewed for Factors (log triage, signed-receipt failure, license-violation response, tenant-leak response)
- [ ] Penetration test report signed by third party (budget: $25k, week 11)
- [ ] SOC 2 Type II audit — Factors scope added to existing SEC-009 engagement

---

## 10. Post-launch hardening (weeks 13-24)

Deferred from v1 launch but tracked for H2 FY27:

- Certificate pinning in SDKs (TLS downgrade prevention)
- Per-tenant encryption keys via Vault transit for Enterprise private registries
- PACT v3 compatibility layer for inter-platform factor exchange
- Formal threat model for OEM embed (STRIDE workshop output)
- Red-team exercise against the resolve/explain cascade (prompt-injection resilience on the LLM rerank path)
- Quarterly signed-receipt key rotation automation via Vault PKI
- WAF rule set for the public marketing pages (factors.greenlang.io)
- DPIA (Data Protection Impact Assessment) for EU customer deployments
- Incident tabletop exercise with the on-call rotation

---

## Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Platform security lead | _____________ | _____________ | _____________ |
| Factors engineering lead | _____________ | _____________ | _____________ |
| Legal (license firewall + DPIA) | _____________ | _____________ | _____________ |
| Founder / CTO | _____________ | _____________ | _____________ |

All four signatures required before first paying customer receives a signed response.
