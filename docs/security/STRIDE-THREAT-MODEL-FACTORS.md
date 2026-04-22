# STRIDE Threat Model — GreenLang Factors API

**Status**: v1.0 (post-launch hardening deliverable)
**Scope**: `/api/v1/factors/*` family, the resolution cascade, the LLM
rerank path, tenant overlays, premium-pack entitlements, the signed-receipt
middleware, the licensed-connector firewall, and all upstream + downstream
trust boundaries listed in §2 below.
**Owners**: Platform Security Lead, Factors Engineering Lead
**Review cadence**: annual + on every material change to authentication,
resolution cascade, LLM path, or licensed-connector set.
**Last reviewed**: 2026-04-22

This document closes the "Formal threat model for OEM embed (STRIDE
workshop output)" bullet in §10 of `FACTORS_API_HARDENING.md` and the
companion "Red-team exercise against the resolve/explain cascade" item.
Every threat row cites an existing component path; anything that needs
new code is carried forward to the Prioritized Remediation List in §5.

---

## 1. Trust zones + assumptions

| Zone | Operator | Trust level | Notes |
|---|---|---|---|
| Z0 Public internet | anyone | untrusted | Marketing site `factors.greenlang.io`; `status/summary` + `watch/status` routes are intentionally unauthenticated (public trust signal per `integration/api/routes/factors.py::status_summary`) |
| Z1 Kong API gateway | GreenLang SRE | semi-trusted | INFRA-006; TLS 1.3 terminator, rate limiter, OIDC/JWT verifier, WAF hook |
| Z2 FastAPI ingress | GreenLang SRE | trusted | `greenlang/integration/api/main.py`; signed-receipt middleware runs here |
| Z3 Resolution engine | GreenLang SRE | trusted | `greenlang/factors/resolution/engine.py` 7-step cascade |
| Z4 Tenant overlay store | per-tenant isolation | trusted w.r.t. tenant | `tenant_overlay.py` + Postgres row-level security |
| Z5 Catalog repo (Postgres + pgvector) | GreenLang SRE | trusted | INFRA-002 + INFRA-005 |
| Z6 LLM rerank provider (Anthropic/OpenAI) | 3rd party | untrusted (output) | `matching/llm_rerank.py`; output is filtered before it reaches the caller |
| Z7 HashiCorp Vault | GreenLang SRE | trusted | SEC-006; transit backend keyed per tenant |
| Z8 Stripe (billing) | 3rd party | semi-trusted | `greenlang/factors/billing/*`; webhooks verified with HMAC |
| Z9 OEM sub-tenants | per-OEM isolation | trusted w.r.t. OEM | `tenant_overlay.py::BrandingConfig` + OEM signing key |

Assumptions stated up-front (disprove these and the threat model collapses):

1. TLS 1.3 is enforced at the Kong ingress; plaintext HTTP is rejected at
   the load balancer (SEC-004).
2. The JWT signing key is rotated every 30 days via ExternalSecret
   pointing at `secret/factors/{stage}/jwt` (SEC-001 + Vault gotcha #3
   in the deploy runbook).
3. Postgres row-level security on `factors`, `tenant_overlays`, and
   `factor_pack_entitlements` is enabled and the application role cannot
   `SET ROLE` to bypass it.
4. All container images are pinned by digest and signed with cosign
   (§6 of the hardening checklist).
5. The JSON-Schema v1.0 contract at
   `config/schemas/factor_record_v1.schema.json` is frozen — any new
   field ships as an optional extension, never a required mutation.

---

## 2. System diagram + trust boundaries

Data flows (left to right = request, right to left = response):

```
 [Z0 PUBLIC INTERNET]
    |
    | (TLS 1.3 only, HSTS)
    v
 [Z1 KONG INGRESS] ──(rate-limit, WAF, JWT verify, tenant claim)──> [Z2 FASTAPI]
                                                                       |
                                                                       | (signed-receipt mw)
                                                                       v
  +----------------------------------+   +---------------------+   +-----------------+
  | Z3 RESOLUTION ENGINE (7 steps)  |<->| Z4 TENANT OVERLAY   |<->| Z7 VAULT        |
  |  1. strict match                |   |   (per-tenant rows, |   |  (transit per-  |
  |  2. method-profile rules        |   |    RLS-scoped read) |   |   tenant; JWT   |
  |  3. geography cascade           |   +---------------------+   |   secret)       |
  |  4. time-window interpolation   |            ^                +-----------------+
  |  5. activity-tag semantic match |            |
  |  6. LLM rerank (Enterprise)     |────────────+
  |  7. fallback with confidence    |            |
  +----------------------------------+            |
           |                ^                      |
           |                |                      v
           v                |              +----------------------+
    +--------------+  +--------------+     | Z5 CATALOG REPO      |
    | Z6 ANTHROPIC |  | Z6 OPENAI    |     |   (Postgres +        |
    |   or         |  |              |     |    pgvector)         |
    +--------------+  +--------------+     +----------------------+
           ^                ^                       ^
           |  (outbound     |                       |
           |   only, no     |                       |
           |   webhook-in)  |                       |
           +----------------+                       |
                                                    |
                                              [Z8 STRIPE webhook]
                                                    v
                                         [BILLING LEDGER +
                                          factor_pack_entitlements]

            [Z9 OEM sub-tenant]  ──────────────────── (read-only subset,
                                                       enforced by
                                                       tenant_overlay::OEMSubTenant)
```

Trust boundaries that MUST be crossed with an identity + signature:

* `Z0 -> Z1`  : TLS 1.3 + Kong JWT verify
* `Z1 -> Z2`  : trusted JWT claim propagation + signed header from Kong
* `Z2 -> Z3`  : tenant-id + tier from request context
* `Z3 -> Z4/Z5`: per-tenant RLS-scoped SQL
* `Z3 -> Z6`  : outbound-only; stubbed provider in tests, prompt is
                built from caller-supplied `activity_text` and is
                sanitised through `_parse_rerank_response` on return
* `Z2 -> Z8`  : Stripe webhook verified via HMAC; one-way inbound only

---

## 3. STRIDE threats — per boundary

Format for every row:
**[S|T|R|I|D|E]** — scenario → likelihood / impact → mitigation (component:path) → residual risk.

### 3.1 Z0 → Z1 (public internet → Kong)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 01 | S | Stolen/replayed JWT used from a different IP | M / H | Kong OIDC plugin + short-lived JWT (15 min) + refresh rotation (SEC-001); per-tenant rate limiter in `middleware/rate_limiter.py` | L — refresh chain can be traced via audit log |
| 02 | S | TLS downgrade via cipher-suite negotiation | L / H | TLS 1.3 only at Kong ingress (SEC-004); HSTS `max-age=63072000; includeSubDomains`; certificate pinning in SDKs (open TODO in §3 hardening checklist) | M — until cert pinning ships in the SDK (Week 13) |
| 03 | T | Tampered request body after JWT verify | L / H | Signed-receipt mw computes SHA-256 over response body; client verifies with stored public key (`signing.py::verify_sha256_hmac`, constant-time compare) | L |
| 04 | R | Unauthenticated caller on `status/summary` cannot be attributed | H / L | Intentional — this route is a public trust signal; Kong still logs source IP + UA | Accept; documented in §2 of this doc |
| 05 | I | API-key leak via HTTP Referer on redirect | L / M | Keys only accepted in `Authorization: Bearer` header, never in URL params (`api_auth.py::get_current_user`) | L |
| 06 | D | L7 flood against `/search` | H / H | Kong `rate-limiting-advanced` (60/600/6000 per tier) + WAF rule `RateLimit2000Per5MinIp` (new, see `deployment/waf/factors-marketing.yaml`) | M — scale horizontally on spike |
| 07 | E | Forged `X-Tenant-Id` header bypassing Kong | L / H | Kong strips client-supplied `X-Tenant-Id`; FastAPI reads tenant from verified JWT claim only (`dependencies.py::get_current_user`) | L |

### 3.2 Z1 → Z2 (Kong → FastAPI)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 08 | S | Bypass Kong by hitting pod-ip directly inside the cluster | L / H | K8s NetworkPolicy — only `ingress-nginx` and `kong` pods can reach factors-api pods on 8000; mesh mTLS (Linkerd) enforces peer identity | L |
| 09 | T | Kong-injected header tampering (plugin bug) | L / M | Signed-receipts middleware verifies nothing about Kong; it re-signs FastAPI's own response. Defense-in-depth: audit log stamps `route + tenant_id + request_id` (SEC-005) | L |
| 10 | R | Kong audit log rotation loses 10-min window | L / M | Loki long-term retention (INFRA-009); immutable audit bucket S3 object-lock (SEC-005) | L |
| 11 | I | Kong admin API exposed on public IP | L / C | Admin API bound to `127.0.0.1:8001`; K8s LoadBalancer `internal` annotation + NetworkPolicy deny-by-default | L |
| 12 | D | Kong upstream-connection exhaustion | M / H | Circuit breaker config in Kong (`failures=5, timeout=30s`); autoscaling HPA on factors-api Deployment | M |
| 13 | E | Plugin CVE lets an anon caller skip JWT check | L / C | Kong image digest-pinned in `deployment/k8s/kong/base/kustomization.yaml`; GHA `trivy image` run on every build (SEC-007) | L |

### 3.3 Z2 → Z3 (FastAPI → Resolution engine)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 14 | T | `X-Factors-Edition` header pins to an unpublished edition | L / M | `service.py::resolve_edition_id` only accepts `stable` or `pending` editions and compares against the manifest table; unknown edition ⇒ 400 | L |
| 15 | T | Edition pin spoofed in signed receipt | L / H | Receipt's `signed_over.edition_id` pulled from `X-GreenLang-Edition` RESPONSE header, which is server-set, not client-controlled (`middleware/signed_receipts.py`) | L |
| 16 | T | Factor record tampered in transit | L / H | Content-hash in receipt (`_body_hash` = SHA-256 of bytes); Ed25519 for Enterprise, HMAC-SHA256 for Pro/Community | L |
| 17 | R | Fallback step not logged in explain output | L / H | Non-negotiable #3 in hardening checklist §1 — `/explain` always returns `step_label` + `fallback_rank` + `why_chosen` + `alternates`. JSON-Schema invariant gate in CI (`scripts/check_factor_invariants.py`) | L |
| 18 | R | Audit log entry omitted on factor mutation | L / H | Append-only audit table, write-ahead of the mutation (SEC-005); integration test `test_api_auth.py` asserts every mutating route writes an entry | L |
| 19 | I | `alternates` list leaks another tenant's overlay | M / H | `resolution/engine.py::_build_alternates` strips tenant_overlay candidates from the alternate list; regression test `tests/factors/security/test_cross_tenant_leak.py` (new) | L |
| 20 | I | Error message reveals internal factor_id namespace | L / M | `ErrorResponse` schema omits stack traces in prod (DEBUG=0); 404 messages never echo server-side paths | L |
| 21 | D | Resolution cascade exhaustion — craft activity with no-match to force 7-step fallback + LLM rerank on every hit | M / H | Per-tier rate limit enforced BEFORE cascade starts; LLM rerank gated by `rate_limit_rpm=10` (`llm_rerank.py`); method-profile caching via Redis INFRA-003 | M — add a cascade-depth budget (open task in §5 below) |
| 22 | E | Raw-factor lookup bypasses method-profile selection rules | L / H | Non-negotiable #6: only `resolve()` + `explain()` accept `method_profile`; grep-based code-review rule flags any new `get_factor(edition, factor_id)` call outside repo tests | L |

### 3.4 Z3 → Z4 (Resolution engine → tenant overlay store)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 23 | S | Customer writes a factor_id collision to hide catalog default | L / M | `tenant_overlay.py::create_overlay` is author-only; catalog default is authoritative when `is_valid_on` is false; collision detection in review workflow | L |
| 24 | T | Overlay `override_value` edited out-of-band via direct DB access | L / H | DB credentials provisioned only to the migration runner; `tenant_overlay.py` writes are audited; Vault + RLS on application role prevent UPDATE on other tenants | L |
| 25 | R | Overlay update without actor attribution | L / M | `update_overlay(updated_by=...)` is a required kwarg at callers; audit trail in `tenant_overlay_audit` table stores `actor` and is PITR-backed | L |
| 26 | I | Cross-tenant leak via SQL forgetting `tenant_id` filter | H (when bug) / C | PG row-level security is the primary control; `TenantOverlayRegistry._get_overlay` adds a defensive filter; new regression test `test_cross_tenant_leak.py` covers every route | L |
| 27 | I | Overlay metadata (notes, source) surfaced in public search results | M / M | `search_factors_v2` uses `merge_search_results` which strips `_overlay_*` keys before returning; CI contract test pins the response schema | L |
| 28 | D | Overlay write storm inflates audit log disk | L / M | Write-rate quota per tenant (enterprise default: 1000 overlays/day) in overlay service; S3 archival of old audit rows after 90 d (SEC-005) | L |
| 29 | E | Reviewer role grants itself overlay-write rights | L / H | SoD enforced in `review_workflow.py` — author ≠ approver; RBAC (SEC-002) rejects `overlay_write` role grant without methodology_lead co-sign | L |

### 3.5 Z3 → Z5 (Resolution engine → catalog repo)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 30 | T | Factor version overwrite (spec non-negotiable #2) | L / C | `factor_version NOT NULL` unique `(factor_id, factor_version)` constraint; append-only via `catalog_repository_pg.insert_factors`; CI gate `scripts/check_factor_invariants.py` | L |
| 31 | T | Content-hash drift between ingested parser output and stored record | L / H | `quality/validators.py::validate_canonical_row` recomputes hash on ingest; mismatch rejects the row | L |
| 32 | R | Factor re-ingested without `source_version` | L / H | JSON-Schema `required: [valid_from, valid_to, source_version]`; ingest rejects missing | L |
| 33 | I | pgvector query returns a vector whose factor row is in another tenant's private namespace | L / H | Search queries always join on `factors.tenant_id = :caller` for private factors; public catalog sits in `tenant_id = 'public'` | L |
| 34 | D | Expensive vector ANN query DoS | M / H | HNSW index tuning per env (dev `m=8`, prod `m=24`); per-query timeout at 500 ms; Redis result cache in `search_cache.py` | M |
| 35 | E | SQL injection via activity-tag string | L / C | `catalog_repository_pg.py` uses `psycopg` parameterised queries throughout; never concatenates strings into SQL; CI `pylint-strict` + `bandit` pass | L |

### 3.6 Z3 → Z6 (Resolution engine → LLM rerank provider)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 36 | S | Provider impersonation (DNS hijack) | L / H | Outbound via pinned provider base URL (`api.anthropic.com`, `api.openai.com`); egress NetworkPolicy allows only those two hosts + Stripe | L |
| 37 | T | Model injects a new `factor_id` into the rerank | M / H | `_parse_rerank_response` intersects LLM output with input candidates; regression test `test_llm_rerank_injection.py` covers 9 hostile model responses | L |
| 38 | R | Rerank call not attributed to caller | L / M | Prometheus histogram `factors_rerank_latency_seconds{tenant,model}` emitted per call (`observability/prometheus.py`) | L |
| 39 | I | Activity-text exfil via prompt injection ("print your system prompt") | M / M | Reranker returns only the ORIGINAL candidate dict references, never the LLM's response text (`llm_rerank.py::_parse_rerank_response`) | L |
| 40 | I | Training-data leak through model (cross-tenant) | L / M | Anthropic + OpenAI both assert zero-retention on API tier (contract clause in vendor due-diligence record); activity_text only, no factor values in prompt | M — revisit when a new vendor is added |
| 41 | D | LLM rerank cost amplification via expensive activity_text | H / M | Per-Enterprise-tier `rate_limit_rpm=10`; rerank truncated to `max_candidates=20`; Anthropic max_tokens=2048 cap | M |
| 42 | E | Tool-use hijack ("call file_read on /etc/shadow") | L / C | Reranker prompt opts out of tool use; Anthropic SDK call uses `messages.create` with no `tools` parameter; provider has no filesystem access | L |

### 3.7 Z2 → Z7 (FastAPI → Vault)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 43 | S | Pod AppRole credentials stolen and replayed | L / H | AppRole secret-id TTL 30 d, role-id TTL unlimited but scoped to namespace; audit path tied to K8s ServiceAccount (SEC-006) | L |
| 44 | T | Transit cipher swapped without re-encrypt | L / H | Key rotation policy `min_decryption_version` pinned at current minus 1; CI gate asserts rotation occurred in last 90 d | L |
| 45 | R | Vault audit device silenced by operator error | L / M | Audit-device health probe in OBS-001; PagerDuty alert if `vault.audit.log_request` metric is zero for >5 min | L |
| 46 | I | Vault token cached by FastAPI logged in stack trace | L / M | No plaintext token in logs — `signing.py` masks secrets in the logging filter (INFRA-009 sanitiser) | L |
| 47 | D | Vault API unavailability blocks every signed response | L / H | Local cache of the public signing key for 60 s; fallback to HMAC-SHA256 if transit unavailable (Community/Pro only) | L |
| 48 | E | Role-confusion: factors pod reads billing KV path | L / H | Vault policy `factors-read-only` whitelists `secret/factors/*` only; CI integration test rejects drift | L |

### 3.8 Z2 → Z8 (FastAPI → Stripe)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 49 | S | Webhook forged by attacker | L / H | Stripe signature verified via `Stripe-Signature` header + `STRIPE_WEBHOOK_SECRET` (`billing/webhooks.py`); clock skew tolerance 300 s | L |
| 50 | T | Replayed webhook escalates entitlement | L / H | `webhook_events.event_id` UNIQUE constraint; idempotent handler skips duplicates | L |
| 51 | R | Stripe event processed twice after retry | L / M | Same idempotency guarantee; audit log entry per unique event | L |
| 52 | I | PII leaked in Stripe event payload (customer email) | M / M | Events stored in `webhook_events` table with PII masked via `security/pii_scanner.py` on ingest (SEC-011) | L |
| 53 | D | Stripe outage blocks entitlement upgrades | M / L | Cached entitlement in Redis for 10 min; degraded-mode documented in deploy runbook | Accept |
| 54 | E | Stripe customer upgraded itself via promo-code hack | L / M | Promo codes issued via ops console only; webhook handler treats promo as ledger event, not command | L |

### 3.9 Z2 → Z9 (FastAPI → OEM sub-tenant)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 55 | S | OEM signs receipts on behalf of downstream customer they don't own | L / H | OEM signing key scoped to `oem_tenant_id` in Vault transit backend; signed receipts carry both `oem_tenant_id` and `sub_tenant_id` and are verified by the SDK | L |
| 56 | T | OEM injects their branding into a non-OEM-tier response | L / M | `BrandingConfig` only applied when `tenant.has_oem_rights` — enforced at response-serialisation layer | L |
| 57 | R | OEM cannot attribute a factor to the originating sub-tenant | L / M | Signed receipt includes `redistribution_class` and a nested `sub_tenant_id`; retained in OEM's audit-bundle export | L |
| 58 | I | Sub-tenant overlays leak to sibling OEM sub-tenants | L / H | `OEMSubTenant` enforces `sub_tenant_id` scoping on every read; same RLS story as Z4; regression covered in §8 integration test | L |
| 59 | D | OEM calling pattern inflates cost attribution | L / M | Per-OEM metering via `metering.py`; monthly reconciliation report in billing ledger | L |
| 60 | E | OEM sub-key reused across OEMs | L / H | Vault PKI issues short-lived OEM sub-keys (TTL 24 h); automated rotation via quarterly job (open in §10 of hardening checklist) | M — quarterly rotation is not yet automated |

---

## 4. Cross-cutting threats (not tied to a single boundary)

| ID | Category | Scenario | L / I | Mitigation | Residual |
|---|---|---|---|---|---|
| 61 | S | JWT issuer confusion (accepting a token from a dev tenant in prod) | L / H | `iss` + `aud` claims checked in `api_auth.py::get_current_user`; separate signing keys per stage in Vault | L |
| 62 | T | Release-signoff checklist forged | L / H | Two-signature requirement (`release_signoff.py::release_signoff_checklist`); S7+S8 promoted from recommended to required for v1 | L |
| 63 | R | Factor deprecation without changelog | L / M | `edition_manifest.py` requires `changelog` field on promote; JSON-Schema invariant | L |
| 64 | I | PII inside a parser_log written to the raw-source bucket | M / M | `security/pii_scanner.py` runs on every parser_log + raw-source write; matches + redacts (SEC-011) | L |
| 65 | I | Licensed-connector value leaks to a non-entitled tenant via explain alternates | M / C | `integration/api/routes/factors.py::explain_factor` filters alternates by `caller_entitlement` class; regression test `test_connector_only_451.py` | L |
| 66 | D | Third-party dependency in SBOM flagged with critical CVE | M / H | GHA `pip-audit` + `trivy image` fail build on CVSS ≥7.0 (SEC-007); cosign signing ensures image provenance | L |
| 67 | E | Segregation-of-Duties bypass — author self-approves a promotion | L / H | `review_workflow.py` rejects when `author_user_id == approver_user_id`; RBAC (SEC-002) enforces `methodology_lead` vs `release_manager` | L |
| 68 | E | Legal role escalates to release_manager via transitive group membership | L / H | Group-closure check in SEC-002 role resolver prevents cycles; quarterly access review | L |

---

## 5. Prioritized remediation list (top 10)

Ranked by `likelihood × impact`, biased toward `I`/`E` over `D`:

| # | Threat IDs | Action | Owner | ETA |
|---|---|---|---|---|
| 1 | 26 | Ship the new cross-tenant leak regression test `tests/factors/security/test_cross_tenant_leak.py` and wire it into the release-signoff gate | Factors Eng | week 13 |
| 2 | 21 | Add a cascade-depth budget (max 7 steps + 2 s wall-clock) and a Prometheus counter `factors_resolution_budget_exceeded_total` | Factors Eng | week 14 |
| 3 | 37, 39, 42 | Ship the LLM-rerank red-team suite `tests/factors/security/test_llm_rerank_injection.py` and add a WAF rule that blocks the canonical jailbreak strings in `activity_text` | Platform Sec | week 13 |
| 4 | 02 | Bundle the GreenLang CA certificate in the Python + TS SDKs (TODO in §3 of hardening checklist) | Factors Eng | week 15 |
| 5 | 65 | Add an OpenAPI contract test asserting 451 on every bulk-export request against a `connector_only:true` source with a non-entitled caller (covered by `test_connector_only_451.py`) | Factors Eng | week 13 |
| 6 | 60 | Automate quarterly OEM-sub-key rotation via Vault PKI scheduled job | Platform Sec | week 16 |
| 7 | 58 | Add an OEM sub-tenant leak integration test that issues a JWT for OEM-A-sub-1 and asserts OEM-A-sub-2 data is never returned | Factors Eng | week 14 |
| 8 | 41 | Move LLM rerank rate-limit enforcement from in-process (`_check_rate_limit`) to Redis token bucket so horizontal replicas share the budget | Factors Eng | week 14 |
| 9 | 40 | Renegotiate Anthropic + OpenAI vendor agreements to confirm zero-retention clause is bound to the GreenLang org-id used in production | Legal + CTO | week 15 |
| 10 | 64 | Extend PII scanner to cover the IEA + ecoinvent raw-source artifacts (currently only parsers are scanned) | Platform Sec | week 16 |

---

## 6. Appendix A — Threat severity keys

| Likelihood | Definition |
|---|---|
| H | Expected at least monthly in the wild (>= 1 attempt/month) |
| M | Possible — known technique, requires moderate attacker investment |
| L | Unlikely — requires chained bug or insider access |

| Impact | Definition |
|---|---|
| C | Catastrophic — multi-tenant data leak, license-class breach, SoC 2 failing finding |
| H | High — single-tenant data leak, customer-visible incident, money loss > $100k |
| M | Medium — degraded service, recoverable within 1 hour |
| L | Low — internal-only or cosmetic |

## 7. Appendix B — Component path cheatsheet

The threat table references these source files:

* `greenlang/integration/api/main.py` — FastAPI app factory, middleware chain
* `greenlang/integration/api/routes/factors.py` — route handlers for the `/api/v1/factors/*` family
* `greenlang/integration/api/dependencies.py` — `get_current_user`, `get_factor_service`
* `greenlang/factors/api_auth.py` — JWT/API-key verification (SEC-001)
* `greenlang/factors/middleware/signed_receipts.py` — response-signing middleware (SEC-006)
* `greenlang/factors/middleware/rate_limiter.py` — per-tier rate enforcement
* `greenlang/factors/resolution/engine.py` — 7-step cascade
* `greenlang/factors/resolution/result.py` — ResolvedFactor.explain()
* `greenlang/factors/matching/llm_rerank.py` — LLM rerank path (enterprise-only)
* `greenlang/factors/tenant_overlay.py` — per-tenant overlay store (F064)
* `greenlang/factors/entitlements.py` — premium-pack entitlement ledger (F8)
* `greenlang/factors/tier_enforcement.py` — tier → visibility mapping
* `greenlang/factors/connectors/license_manager.py` — license-class firewall (§5 of hardening checklist)
* `greenlang/factors/data/source_registry.yaml` — the authoritative source list
* `greenlang/factors/signing.py` — Ed25519 + HMAC-SHA256 helpers (`verify_sha256_hmac` uses `hmac.compare_digest`)
* `greenlang/factors/quality/review_workflow.py` — SoD: author ≠ approver
* `greenlang/security/pii_scanner.py` — PII redaction (SEC-011)
* `greenlang/factors/observability/prometheus.py` — OBS-001 metrics
* `config/schemas/factor_record_v1.schema.json` — frozen v1 JSON-Schema
* `scripts/check_factor_invariants.py` — CI gate for the non-negotiables

---

## 8. Change log

| Date | Author | Note |
|---|---|---|
| 2026-04-22 | Platform Sec + Factors Eng | Initial v1.0; covers launch scope |
