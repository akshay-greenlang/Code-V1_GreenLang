# GreenLang Factors — FY27 Deployment Architecture

**Document owner:** GL-AppArchitect
**Audience:** DevOps, SRE, Platform Engineering, Product Ops
**Status:** Implementation-ready, FY27 launch blueprint
**Last updated:** 2026-04-22
**Code basis:** `greenlang/factors/` at commit `f88439d0` (master)
**Reality basis:** `FY27_vs_Reality_Analysis.md` sections 3 and 5 — build is 85–92% complete; gap is hosting + UIs + release-cut + three-label dashboard + signed-receipt enforcement
**Scope:** Full v1 — all 7 method packs, all jurisdictions, all UIs, shipped in 7 vertical slices

---

## 0. What exists today (grounding)

Top-level `greenlang/factors/` inventory (verified 2026-04-22):

```
__init__.py                      ingestion/         (fetchers, parser_harness, normalizer, parsers/)
__main__.py                      inventory.py
api_auth.py                      mapping/           (classifications, fuels, materials, transport, waste, spend, …)
api_endpoints.py                 matching/          (embedding, pgvector_index, llm_rerank, pipeline, suggestion_agent)
approval_gate.py                 method_packs/      (corporate, electricity, freight, eu_policy, land_removals, product_carbon, finance_proxy, product_lca_variants, registry)
artifacts/                       metering.py
backfill.py                      middleware/        (auth_metering, rate_limiter)
batch_jobs.py                    notifications/     (webhook_notifier)
billing/                         (aggregator, metering, stripe_provider, usage_sink, webhook_handler)
bounded_contexts/                observability/     (prometheus, prometheus_exporter, health, sla)
cache_redis.py                   onboarding/        (health_check, partner_setup, sample_queries)
catalog_repository.py            ontology/          (unit_graph, chemistry, gwp_sets, heating_values, geography, methodology)
catalog_repository_pg.py         performance_optimizer.py
cli.py                           pilot/             (provisioner, telemetry, feedback, registry)
connectors/                      (ecoinvent, electricity_maps, iea, license_manager, audit_log, registry)
data/                            policy_mapping.py
dedupe_rules.py                  quality/           (validators, dedup_engine, cross_source, license_scanner, review_queue, release_signoff, audit_export, impact_simulator, versioning, rollback, promotion, escalation, sla, consensus)
edition_manifest.py              regulatory_tagger.py
entitlements.py                  resolution/        (engine [7-step cascade], request, result, tiebreak)
etl/                             (ingest, normalize, qa)
ga/                              (billing, readiness, sla_tracker)
index_manager.py                 sdk/               (python/, ts/)
ingestion/                       search_cache.py
inventory.py                     security/          (api_key_manager, audit, input_validation)
method_packs/                    service.py
...                              signing.py         (SHA-256 HMAC + Ed25519, key_id rotation, Vault-backed private key)
                                 source_registry.py
                                 tenant_overlay.py
                                 tier_enforcement.py
                                 watch/             (source_watch, change_detector, change_classification, doc_diff, release_orchestrator, rollback_edition, scheduler, changelog_draft, cross_edition_changelog, regulatory_events, status_api, rollback_cli)
                                 webhooks.py
```

Key anchor files referenced throughout this document:

- 7-step cascade: `greenlang/factors/resolution/engine.py` (lines 1–80 confirm spec match)
- REST surface: `greenlang/factors/api_endpoints.py` (F032–F036 primitives)
- Runtime wiring: `greenlang/factors/service.py` (FactorCatalogService)
- Postgres + pgvector repo: `greenlang/factors/catalog_repository_pg.py`
- SQLite repo (local/dev): `greenlang/factors/catalog_repository.py`
- Redis cache: `greenlang/factors/cache_redis.py`, `greenlang/factors/search_cache.py`
- Signing: `greenlang/factors/signing.py`
- Release pipeline: `greenlang/factors/watch/release_orchestrator.py`, `watch/rollback_edition.py`
- Billing: `greenlang/factors/billing/stripe_provider.py`, `billing/aggregator.py`, `billing/usage_sink.py`
- SDKs: `greenlang/factors/sdk/python/`, `greenlang/factors/sdk/ts/`

Per `FY27_vs_Reality_Analysis.md` sections 3.1 and 5: Factors itself is Ready; the **gap is packaging** — hosted deployment, public factor explorer UI, operator console, three-label dashboard, signed-receipts enforced at middleware, `/explain` as first-class primitive, and v1.0 Certified edition cut. This document closes those gaps.

---

## 1. Context diagram

Factors is a hosted **API + UI + SDK** product. It is consumed by six distinct caller classes.

```
                                    +--------------------------------+
                                    |    PUBLIC FACTORS EXPLORER     |
                                    |    (Next.js on Vercel)         |
                                    |  - search, browse, explain     |
                                    |  - three-label dashboard       |
                                    +---------------+----------------+
                                                    |
                                                    v
      +---------------------+          +------------+-------------+          +----------------------+
      |  Developer (SDK)    |          |                          |          | Consultant / Analyst |
      |  python / ts SDK    +--------->|     KONG API GATEWAY     |<---------+ (Factor Explorer UI) |
      |  factors/sdk/       |          |     (INFRA-006)          |          |                      |
      +---------------------+          |                          |          +----------------------+
                                       |  - JWT auth (SEC-001)    |
      +---------------------+          |  - rate limiting         |          +----------------------+
      |  Platform OEM       +--------->|  - tier enforcement      |<---------+ Enterprise Methodology|
      |  tenant_overlay.py  |          |  - metering capture      |          |  Console (SAML SSO)  |
      |  entitlements.py    |          |                          |          +----------------------+
      +---------------------+          +-------------+------------+
                                                     |
      +---------------------+                        v
      |  Internal Operator  +---.          +---------+----------+
      |  Console (SAML SSO) +----o         |   FACTORS SERVICE  |
      |  ingestion + QA +   |    o-------->|   FastAPI + uvicorn|
      |  mapping + release  |   /          |  api_endpoints.py  |
      +---------------------+  /           |  service.py        |
                              /            |  7-step engine     |
      +---------------------+/             +---------+----------+
      |  Downstream GL apps |                        |
      |  - GL-CBAM-APP      +----SDK--HTTP---------->+
      |  - GL-CSRD-APP      |                        |
      |  - Scope Engine     |                        |
      |  - PCF Studio (FY28)|                        |
      |  - DPP Hub (FY28)   |                        |
      +---------------------+                        v
                                            +-------+-------+
                                            |  PG + Redis + |
                                            |  S3 + Stripe  |
                                            +---------------+
```

**Callers:**

| Caller | Interface | Primary endpoints | Auth | License tier |
|---|---|---|---|---|
| Developer (SDK) | `factors/sdk/python/client.py`, `factors/sdk/ts/src/` | `/resolve`, `/search`, `/explain`, `/editions` | JWT + API key | Free / Dev / Pro |
| Consultant UI | Public Factor Explorer (Next.js) | `/search`, `/factors/{id}`, `/explain`, `/editions/diff` | Anon + optional account | Free / Pro |
| Platform OEM | HTTP + `tenant_overlay` headers | `/resolve`, `/bulk-resolve`, `/overrides`, `/receipts` | OEM signed JWT | OEM tier |
| Enterprise Methodology Console | Browser (SAML SSO) | `/editions/pin`, `/overrides`, `/audit-bundle`, `/impact-simulate` | SAML / OIDC | Enterprise |
| Internal Operator Console | Browser (internal SSO) | `/admin/ingest`, `/admin/review`, `/admin/promote`, `/admin/rollback` | Okta / SAML | Internal-only RBAC |
| Downstream GL apps (CBAM, CSRD, Scope Engine, PCF Studio, DPP Hub) | python SDK over HTTP | `/resolve`, `/audit-bundle`, `/receipts` | Service-account JWT | Enterprise |

Every response from any path crosses the **signed-receipt boundary** (`greenlang/factors/signing.py`) and the **license-class gate** (`greenlang/factors/connectors/license_manager.py` + `quality/license_scanner.py`). No calculation path uses an LLM — LLM is confined to matching/rerank (`matching/llm_rerank.py`) and suggestion (`matching/suggestion_agent.py`), both of which produce *candidates*, never *values*.

---

## 2. Component diagram

```
 +---------------------------------------------------------------------------------------+
 |                                  INGRESS (public)                                     |
 |         ALB (AWS NLB/ALB) --> Kong API Gateway (INFRA-006) --> FastAPI pods           |
 +---------------------------------------------------------------------------------------+
                                         |
  +--------------------------+-----------+----------------+------------------------------+
  |                          |                            |                              |
  v                          v                            v                              v
+----------+        +----------------+          +------------------+          +------------------+
| FastAPI  |        | Operator SPA   |          | Public Explorer  |          | Webhooks         |
| service  |        | (React + SSO)  |          | (Next.js)        |          | billing/Stripe   |
| pods x N |        | served by same |          | CDN (Vercel /    |          | webhooks.py      |
|          |        | ingress, /ops  |          |  Cloudflare)     |          | billing/         |
|          |        |                |          |                  |          |  webhook_handler |
+----+-----+        +-------+--------+          +---------+--------+          +---------+--------+
     |                      |                             |                             |
     | uvicorn workers      | static assets               | read-only /search,          |
     | api_endpoints.py     | (S3+CloudFront)             | /explain, /editions         |
     | service.py           |                             |                             |
     v                      v                             v                             v
+-------------------------------------------------------------------------------+
|                         INTERNAL SERVICE MESH (EKS, INFRA-001)                |
|                                                                               |
|  +------------------+   +--------------------+   +---------------------+     |
|  | ResolutionEngine |   | FactorCatalog      |   | SignerService       |     |
|  | resolution/      |   | Service            |   | signing.py          |     |
|  | engine.py        |   | service.py         |   |  SHA256 HMAC /      |     |
|  |  - 7-step cascade|   |  catalog_repo_pg.py|   |  Ed25519 (Vault kid)|     |
|  |  - tiebreak.py   |   +----------+---------+   +----------+----------+     |
|  +---------+--------+              |                        |                |
|            |                       |                        |                |
|  +---------+--------+   +----------+---------+   +----------+----------+     |
|  | MethodPack       |   | Cache Layer        |   | License Gate        |     |
|  | method_packs/    |   | cache_redis.py     |   | connectors/         |     |
|  | registry.py      |   | search_cache.py    |   |  license_manager.py |     |
|  +------------------+   | (INFRA-003 Redis)  |   | quality/            |     |
|                         +--------------------+   |  license_scanner.py |     |
|  +------------------+                            +---------------------+     |
|  | Matching         |   +--------------------+   +---------------------+     |
|  | matching/        |   | Metering + Billing |   | Tenant Overlay      |     |
|  |  pgvector_index  |   | middleware/        |   | tenant_overlay.py   |     |
|  |  embedding       |   |  auth_metering.py  |   | entitlements.py     |     |
|  |  llm_rerank      |   | billing/aggregator |   | tier_enforcement.py |     |
|  |  suggestion_agent|   | billing/usage_sink |   |                     |     |
|  +------------------+   +----------+---------+   +---------------------+     |
|                                    |                                         |
+------------------------------------+-----------------------------------------+
                                     |
              +----------------------+--------------------------+
              |                      |                          |
              v                      v                          v
   +----------+-----------+  +-------+--------+       +---------+----------+
   | PostgreSQL 14+       |  | Redis 7        |       | S3 artifact store  |
   | (RDS + TimescaleDB)  |  | cluster mode   |       | raw sources, parser|
   |  INFRA-002           |  | INFRA-003      |       | logs, audit zips,  |
   | + pgvector INFRA-005 |  | cache-hit      |       | signed editions    |
   |  - factor_editions   |  | target >80%    |       | INFRA-004          |
   |  - factor_records    |  |                |       |                    |
   |  - sources, reviews  |  |                |       |                    |
   |  - embeddings (HNSW) |  |                |       |                    |
   +----------------------+  +----------------+       +--------------------+

   +---------------------------+      +-----------------------+
   | Stripe (external SaaS)    |      | OpenAI / Anthropic    |
   | billing/stripe_provider.py|      | matching/embedding.py |
   |  - meter ingest           |      | matching/llm_rerank.py|
   |  - invoice webhook        |      | (keys in Vault)       |
   +---------------------------+      +-----------------------+

   Observability (sidecar / daemonset):
   - OpenTelemetry collector (OBS-003)  --> Prometheus (OBS-001) + Grafana (OBS-002)
   - promtail --> Loki (INFRA-009)
   - Alertmanager (OBS-004) --> PagerDuty, Slack
   - SLO / error-budget burn alerts (OBS-005)

   Secrets:
   - Vault (SEC-006) — Ed25519 private key, DB creds, Stripe keys, OpenAI/Anthropic keys
   - K8s Secrets sync via External Secrets Operator
```

**Components map to code:**

| Component | Source of truth | Notes |
|---|---|---|
| Kong gateway | INFRA-006 (existing) | JWT verify, rate limit, per-plan quota, circuit breaker |
| FastAPI service | `factors/api_endpoints.py` + `factors/service.py` + `factors/cli.py` (`serve` cmd) | uvicorn workers behind gunicorn supervisor |
| 7-step cascade | `factors/resolution/engine.py` | Deterministic — no LLM |
| Method packs | `factors/method_packs/*.py` | Registered at boot via `method_packs/registry.py` |
| PG repo | `factors/catalog_repository_pg.py` | Pools via psycopg3; pgvector for embedding search |
| Cache | `factors/cache_redis.py`, `search_cache.py` | Key: `(edition_id, factor_id, unit)`; TTL 1h read, 24h search |
| Signing | `factors/signing.py` | Ed25519 in prod; HMAC for dev |
| Metering | `factors/middleware/auth_metering.py`, `factors/metering.py` | Every request → usage_sink → Stripe |
| Tenant overlay | `factors/tenant_overlay.py` | Resolved as step 1 of 7-step cascade |
| Entitlements | `factors/entitlements.py`, `factors/tier_enforcement.py` | Per-plan feature gates at router layer |
| Source watch | `factors/watch/source_watch.py` | CronJob → `change_detector` → `release_orchestrator` |
| Release cut | `factors/watch/release_orchestrator.py` + `quality/release_signoff.py` | Human gate via Operator Console |
| Rollback | `factors/watch/rollback_edition.py`, `watch/rollback_cli.py` | Zero-downtime edition pin swap |

---

## 3. Deployment topology

### 3.1 Multi-region strategy

| Region | Purpose | Residency |
|---|---|---|
| `us-east-1` (primary) | US workloads, global CDN origin | US customers, global open-core |
| `eu-west-1` (secondary) | EU customers (CBAM, CSRD) requiring EU-only data residency | Data never leaves EU |
| `ap-south-1` (future, FY28) | India beachhead | Per India DPDP compliance |

Each region runs an independent EKS cluster (INFRA-001) with its own PostgreSQL (multi-AZ), Redis, and S3 bucket. **Editions are replicated** US↔EU via a signed pull (the receiving region verifies the Ed25519 edition manifest signature before admitting). Customer override tables are **not** replicated cross-region — they stay pinned to the customer's home region.

### 3.2 VPC layout (per region)

```
VPC 10.x.0.0/16
  public subnets (3 AZ):  NAT GW, ALB, Kong gateway (edge)
  private app subnets:    EKS node groups (FastAPI, Operator SPA, Workers)
  private data subnets:   RDS PostgreSQL, ElastiCache Redis, NOT internet-routable
  private egress subnets: Vault cluster, OTel collectors
```

Security groups: public → Kong (443 only); Kong → FastAPI (internal port); FastAPI → PG/Redis/S3 VPC endpoint; Vault is reachable only from EKS node SGs via mTLS. No public IPs on any database or cache.

### 3.3 Kubernetes workloads

| Deployment | Replicas (prod baseline) | Resources | HPA target | Node pool |
|---|---|---|---|---|
| `factors-api` (FastAPI) | 6 | 2 vCPU / 4 GiB | CPU 60%, min 6, max 60 | app-general |
| `factors-operator-spa` | 2 | 0.5 vCPU / 1 GiB | — | app-general |
| `factors-matcher` (embedding + rerank workers) | 3 | 4 vCPU / 8 GiB | queue depth | app-cpu-high |
| `factors-watch-scheduler` | 1 (singleton) | 0.5 vCPU / 1 GiB | — | app-general |
| `factors-ingestion-workers` | 4 | 2 vCPU / 8 GiB | SQS queue depth | app-cpu-high |
| `factors-release-orchestrator` | 1 (singleton) | 0.5 vCPU / 1 GiB | — | app-general |
| `factors-signer` (Ed25519 signer) | 3 | 0.5 vCPU / 1 GiB | — | app-general |
| `otel-collector` daemonset | 1 per node | 0.2 vCPU / 512 MiB | — | all |
| `promtail` daemonset | 1 per node | 0.2 vCPU / 256 MiB | — | all |

All pods: `readinessProbe` → `/healthz` (from `factors/observability/health.py`); `livenessProbe` → same with different timing. PDB: min available 50%.

### 3.4 Blue/green cut-over + edition pinning

**Rolling pod updates** handle code deploys. **Edition cut-overs** are different — they change the default `edition_id` customers resolve against. Flow:

1. Operator in the Operator Console triggers `release_orchestrator.cut_edition(candidate_edition_id)`.
2. Orchestrator calls `quality/release_signoff.py` — human sign-off gate, requires N approvers per method pack (configurable).
3. Orchestrator writes new `EditionManifest` to `edition_manifest.py` with `status=promoted`.
4. A feature-flag (INFRA-008) named `factors.default_edition_v1_next` flips atomically. Clients passing explicit `edition_id` are unaffected.
5. Cache invalidation pushed via Redis pub/sub (channel `factors:edition:bump`) — `cache_redis.py` listens and clears the `default` keyspace.
6. **Rollback**: `watch/rollback_edition.py` flips the flag back. Guaranteed zero-downtime because editions are content-addressed (SHA-256) and both editions remain resident in PG.

Rollback CLI: `python -m greenlang.factors.watch.rollback_cli --to <edition_id>` (implemented in `watch/rollback_cli.py`).

---

## 4. Data flows

### 4.1 Ingestion flow (source → promoted edition)

```
  +-------------+     +------------------+     +-------------+     +----------------+
  | source_watch|---->|  fetchers        |---->|  parser     |---->|  normalizer    |
  | watch/      |     |  ingestion/      |     |  harness    |     |  ingestion/    |
  | source_watch|     |   fetchers.py    |     |  ingestion/ |     |  normalizer.py |
  +-------------+     +------------------+     +-------------+     +----------------+
                                                                          |
                                                                          v
  +------------------+    +------------------+    +----------------+    +---------------+
  | release_signoff  |<---| review_queue     |<---| cross_source   |<---| validators    |
  | quality/         |    | quality/         |    | quality/       |    | quality/      |
  | release_signoff  |    | review_queue.py  |    | cross_source   |    | validators.py |
  +------+-----------+    +------------------+    +----------------+    +-------+-------+
         |                                                                      |
         v                                                                      |
  +------+-----------+    +------------------+    +----------------+    +-------+-------+
  | promotion        |<---| approval_gate    |<---| dedup_engine   |<---| license_      |
  | quality/         |    | approval_gate.py |    | quality/       |    | scanner       |
  | promotion.py     |    |                  |    | dedup_engine.py|    | quality/      |
  +------+-----------+    +------------------+    +----------------+    +---------------+
         |
         v
  +------+-------------+
  |  EditionManifest   |
  |  edition_manifest  |
  |  status=promoted   |
  |  signed Ed25519    |
  +--------------------+
```

Raw source artifacts, parser logs, and reviewer decisions are stored in S3 under `s3://gl-factors-{env}-{region}/raw/{source_id}/{fetch_ts}/`. Metadata indexes live in PG.

### 4.2 Resolve flow (client request → signed response)

```
  Client (SDK) --> Kong gateway ----+
                                    |  JWT verify, rate limit, quota, tier check
                                    v
                           +--------+--------+
                           | FastAPI         |
                           | api_endpoints.py|
                           +--------+--------+
                                    |
                                    v
                           +--------+--------+      Cache hit? return early
                           | cache_redis.py  |<--+
                           +--------+--------+   |
                                    | miss       |
                                    v            |
                  +-----------------+-----------------+
                  | ResolutionEngine.resolve()        |
                  | resolution/engine.py              |
                  |                                   |
                  | Step 1: tenant_overlay_reader     |<-- tenant_overlay.py
                  | Step 2: supplier/manufacturer     |
                  | Step 3: facility/asset            |
                  | Step 4: utility/tariff/subregion  |
                  | Step 5: country/sector average    |
                  | Step 6: method-pack default       |<-- method_packs/registry.py
                  | Step 7: global default            |
                  |                                   |
                  | Tiebreak per step: tiebreak.py    |
                  |                                   |
                  | Unit conversion via unit_graph    |<-- ontology/unit_graph.py
                  +-----------------+-----------------+
                                    |
                                    v
                           +--------+--------+
                           | catalog_repo_pg |  (pooled psycopg3, HNSW for semantic search)
                           +--------+--------+
                                    |
                                    v
                           +--------+--------+
                           | license_gate    |  quality/license_scanner.py + connectors/license_manager.py
                           | reject if       |  (e.g. ecoinvent redistribute=false on open tier)
                           | license-class   |
                           | forbids caller  |
                           +--------+--------+
                                    |
                                    v
                           +--------+--------+
                           | signing.py      |  Ed25519 sign payload_sha256
                           | produce receipt |
                           +--------+--------+
                                    |
                                    v
                           +--------+--------+
                           | auth_metering   |  middleware/auth_metering.py
                           | record usage    |  --> billing/usage_sink.py --> Stripe
                           +--------+--------+
                                    |
                                    v
                              Client response
                              + signed receipt (header + body)
```

**Latency budget** (P95, warm cache):
- Kong: 5 ms
- FastAPI ingress + auth: 10 ms
- Cache hit path: 5 ms
- Resolution (miss path): 80 ms (includes 1 PG query with prepared statement + license gate)
- Signing (Ed25519): 2 ms
- Metering: 3 ms
- **Target P95: 120 ms (cache hit), 180 ms (cache miss)**

### 4.3 Explain flow (`/explain/{factor_id}`)

```
Client --> Kong --> FastAPI --> repo.get_factor() --> build_explain_payload():
  - 7-step cascade trace (which step fired, which rule filtered, which tiebreak ranked)
  - provenance: source_org, publication, year, methodology, citation URL
  - license_info (flag if caller cannot redistribute)
  - DQS quality scores (completeness, reliability, temporal, geographic, technology)
  - unit conversion trace via ontology/unit_graph.py
  - alternate candidates (top 3) with tiebreak reasons
  - edition_id + content_hash
--> signed --> return
```

Built from `factors/api_endpoints.py::build_audit_bundle` (lines 33–80) adapted to a streaming-safe explain shape. UI consumes this verbatim.

### 4.4 Audit bundle export (`/audit-bundle/{case_id}`)

```
Enterprise customer --> POST /audit-bundle  {case_id, date_range, factor_ids[]}
   --> FastAPI spawns async job (batch_jobs.py) --> worker:
        for each factor:
           get_factor() from catalog_repository_pg
           fetch raw source artifact from S3 (using source_registry.py ref)
           fetch parser log from S3
           fetch reviewer decision from PG (quality/review_queue.py)
           build bundle dict per factors/api_endpoints.py::build_audit_bundle
        package as ZIP with signed manifest
        upload to customer-specific S3 prefix (tenant-scoped via tenant_overlay.py)
        signed URL (TTL 24h) returned via webhook + polling endpoint
   --> bundle entries include SHA-256 content_hash verification chain
```

Implementation leverages `quality/audit_export.py` plus `batch_jobs.py`. Bundle is Ed25519-signed so auditors can verify offline.

### 4.5 Change detection flow (source diff → changelog)

```
CronJob (per source, cadence per source_registry.py) -->
  source_watch.fetch_head() -->
  change_detector.diff(prev_hash, new_hash) -->
     if changed:
        doc_diff.generate() (field-level diff)
        change_classification.classify() (major/minor/editorial/deprecation)
        regulatory_events.tag() (e.g., "CBAM annex update")
        changelog_draft.render()
        cross_edition_changelog.append()
        open review in review_queue (operator console)
     --> orchestrator:
        if auto_promote_allowed and classification=editorial:
           release_orchestrator.auto_cut()
        else:
           notify ops via notifications/webhook_notifier.py
```

`watch/scheduler.py` is a singleton cron scheduler. `watch/status_api.py` exposes the change pipeline state to the operator console.

---

## 5. Security architecture

| Concern | Control | Source |
|---|---|---|
| Edge auth | JWT with rotating signing keys | SEC-001 + Kong |
| API keys | Per-tenant bearer keys, hashed at rest | `factors/security/api_key_manager.py` |
| Authorization | RBAC with plan-tier checks | SEC-002 + `factors/tier_enforcement.py`, `factors/entitlements.py` |
| Transport | TLS 1.3 everywhere; mTLS for service-to-service inside mesh | SEC-004 |
| Audit logging | Every resolve, explain, admin action logged | SEC-005 + `factors/security/audit.py` |
| Secrets | Ed25519 signing key, DB creds, Stripe, OpenAI, Anthropic | SEC-006 (Vault) |
| PII scan on sources | Source artifacts scanned before promotion | SEC-011 + `factors/security/input_validation.py` |
| Signed responses | Ed25519 receipt on every response | `factors/signing.py` — enforced at middleware (new in v1) |
| License-class gate | Block redistribution when license forbids | `factors/connectors/license_manager.py` + `factors/quality/license_scanner.py` |
| Rate limiting | Per plan, per endpoint | `factors/middleware/rate_limiter.py` + Kong |
| Input validation | Pydantic v2 on every request model | `factors/security/input_validation.py` |

**Signed-receipt enforcement (v1 change):** Today `signing.py` exists but is opt-in. v1 enforces signing as middleware. Every 2xx response from `/resolve`, `/bulk-resolve`, `/explain`, `/audit-bundle` carries:
- `X-GL-Signature` header (base64 signature)
- `X-GL-Key-Id` header (rotating kid)
- `X-GL-Algorithm` header (`ed25519` in prod, `sha256-hmac` in dev)
- Body includes `_receipt` object per `signing.py` format

Customers treat the receipt as their evidence artifact. Key rotation: kid rotates every 90 days; old kids remain verifiable for 2 years.

**License-class gate** (new enforcement at response boundary in v1): Before returning any factor, the response middleware calls `license_manager.can_redistribute(factor, caller_tier)`. If the factor is a licensed connector response (e.g. ecoinvent) and caller is below OEM tier, the response is replaced with `{"license_error": "factor_requires_licensed_tier", "upgrade_url": "..."}` and logged. This prevents accidental leakage of licensed content through the open API.

---

## 6. SDK + OEM architecture

### 6.1 Python SDK (`greenlang/factors/sdk/python/`)

Files: `client.py`, `auth.py`, `transport.py` (retry, backoff, connection pool), `models.py` (Pydantic mirrors of server models), `pagination.py`, `webhooks.py`, `errors.py`, `cli.py`, `py.typed`.

Packaged as `greenlang-factors` on PyPI (separate from the greenlang monorepo package so SDK consumers don't pull the whole platform).

```python
from greenlang_factors import FactorsClient
c = FactorsClient(api_key="...", edition_id="v1.0-certified")
f = c.resolve(activity="diesel combustion, on-road, HGV", unit="kg_co2e/L")
print(f.value, f.receipt.verify())
```

### 6.2 TypeScript SDK (`greenlang/factors/sdk/ts/`)

Files: `index.ts`, `src/`, `package.json`, dual-build (`tsconfig.cjs.json` + `tsconfig.esm.json`), Jest tests. Shipped as `@greenlang/factors` on npm. Mirrors the Python surface.

### 6.3 OEM tenant overlay architecture

OEMs (platforms that embed Factors) use:
- `factors/tenant_overlay.py` — per-tenant factor overrides become step 1 of the cascade.
- `factors/entitlements.py` — which method packs + sources + jurisdictions this OEM's customers can access.
- `factors/tier_enforcement.py` — rate limits, max bulk-resolve batch size, audit bundle export allowed?
- White-label branding config: tenant can supply logo, color, domain; Explorer UI reads via `/tenants/{id}/branding`.
- Redistribution controls: `license_manager.py` per-tenant policy layer.
- Signed receipts: OEM customers get signed receipts signed by a **sub-kid** tied to the OEM tenant, so OEM customers' auditors can verify against GL root + OEM intermediate.

**OEM data flow:** OEM platform embeds SDK or iframe, passes tenant JWT with `tenant_id` claim. FastAPI resolves tenant at request boundary; tenant_overlay reads overrides from `factor_overrides` table keyed by `tenant_id`; response is signed with the OEM sub-key.

### 6.4 Enterprise methodology console (Enterprise customer UI)

Distinct from Operator Console (internal). Enterprise customers use a self-service UI:
- Pin editions per business unit
- Manage customer overrides (with audit trail)
- Trigger impact simulations via `factors/quality/impact_simulator.py` (before pinning a new edition, show the delta on their book of activities)
- Export audit bundles
- View SLA dashboard (via `ga/sla_tracker.py`)

---

## 7. UI architecture

### 7.1 Public Factor Explorer (Next.js 14, read-only)

**Hosting:** Vercel (primary) or Cloudflare Pages. Static generation where possible; API routes are thin proxies to the Factors REST API.

**Key pages:**

| Route | Function | Server call |
|---|---|---|
| `/` | landing, three-label dashboard (Certified count, Preview count, Connector-only count) | `GET /v1/catalog/stats` |
| `/search` | full-text + semantic search | `POST /v1/search` (F035 with sort + pagination) |
| `/f/{factor_id}` | factor detail | `GET /v1/factors/{id}` |
| `/f/{factor_id}/explain` | explain visualization: 7-step cascade trace | `GET /v1/explain/{id}` |
| `/editions` | edition list + changelog | `GET /v1/editions` |
| `/editions/{a}/diff/{b}` | edition diff viewer | `GET /v1/editions/diff` (F034) |
| `/sources` | source registry view | `GET /v1/sources` |
| `/method-packs` | 7 method pack overviews | `GET /v1/method-packs` |
| `/docs` | developer portal, OpenAPI-rendered | static |
| `/playground` | live SDK playground | browser SDK |

**Search back-end:** pgvector-backed semantic search + PG full-text hybrid via RRF. Served by `matching/pgvector_index.py` + `search_cache.py` (24h TTL).

**Auth:** Optional. Anonymous users get read-only access to Certified + Preview factors; Connector-only content returns license teaser ("sign in with an eligible plan to unlock"). Account creation via OAuth (GitHub, Google).

### 7.2 Internal Operator Console (React SPA, SSO-gated)

**Hosting:** Served by `factors-operator-spa` Deployment inside EKS (not public CDN). Route `/ops/*` on the same Kong gateway, gated by SAML/OIDC SSO (Okta).

**Screens:**

| Screen | Backs onto |
|---|---|
| Source ingestion console | `ingestion/fetchers.py`, `ingestion/parser_harness.py`, status via `watch/status_api.py` |
| Mapping workbench | `matching/suggestion_agent.py`, `matching/pipeline.py`, human accept/reject |
| QA dashboard | `quality/validators.py`, `quality/cross_source.py`, `quality/sla.py` |
| Review queue | `quality/review_queue.py`, `quality/review_workflow.py` |
| Diff viewer (edition→edition, source→source) | `watch/doc_diff.py`, `watch/cross_edition_changelog.py` |
| Approval workflow | `approval_gate.py`, `quality/release_signoff.py`, `quality/escalation.py` |
| Customer override manager | `tenant_overlay.py` CRUD + audit |
| Impact simulator | `quality/impact_simulator.py` — "if we promote edition X, N customers' resolutions change" |
| Release cut-over | `watch/release_orchestrator.py` + feature-flag flip |
| Rollback | `watch/rollback_edition.py`, `watch/rollback_cli.py` |
| Connector health | `connectors/audit_log.py`, `connectors/registry.py`, `connectors/metrics.py` |
| Billing & SLA | `ga/billing.py`, `ga/sla_tracker.py`, `ga/readiness.py` |

**RBAC**: operator, reviewer, releaser, admin — enforced via SEC-002 backed by Okta groups.

---

## 8. Slice rollout plan — 7 vertical slices

Each slice ships a method-pack end-to-end: sources → parsers → ontology → mapping → matching gold set → resolution → gold-eval CI gate → release-signed edition → SDK examples → explorer landing page → downstream-app integration test. Slices share the same infra; slice merges add packs and gold sets, never re-architect.

### Slice E — Electricity (weeks 1–4, launch wedge)

- **Scope:** Scope 2 location-based + market-based, grid subregions, residual mix, certified renewables, hourly carbon (via Electricity Maps connector)
- **Source packs promoted:** eGRID, IEA, AIB, India-CEA, Japan-METI, Australia-NGA, Electricity Maps (licensed), Green-e, Green-e residual, TCR
- **Method packs live:** `method_packs/electricity.py`
- **Gold-eval coverage:** 200 activity descriptions → expected factor IDs; P@1 ≥ 0.85 gate in CI
- **GTM hook:** unlocks GL-CBAM-APP electricity inputs + GL-SB253-APP + GL-CSRD-APP E1 climate
- **Acceptance gate:** E1 edition cut and signed; 500 rps sustained resolve; P95 < 150 ms; zero license-class leaks in 1000-request probe

### Slice C — Combustion (weeks 3–6)

- **Scope:** Scope 1 stationary + mobile combustion, fuel LHV/HHV switch, CO2/CH4/N2O gas breakdown
- **Source packs:** EPA GHG Hub, DESNZ, IPCC 2006 + 2019 refinement
- **Method packs live:** `method_packs/corporate.py`
- **Gold-eval:** 150 activities; P@1 ≥ 0.90
- **GTM hook:** unlocks GL-CBAM-APP combustion, GL-SCOPE-1-2-APP, GL-GHG-APP
- **Acceptance gate:** parity with DESNZ published tables within 0.5% across 500 sampled factors

### Slice F — Freight (weeks 5–8)

- **Scope:** Freight lanes, modal mix, backhauling, distance-or-weight, ISO 14083
- **Source packs:** GLEC Framework, GHG-Protocol, DESNZ freight, regional lane averages
- **Method packs live:** `method_packs/freight.py`
- **Gold-eval:** 150 activities
- **GTM hook:** unlocks Scope 3 Cat 4 + 9, PCF Studio (FY28 preview), future FleetOS
- **Acceptance gate:** 3 mode-mix scenarios reproduce GLEC worked examples

### Slice M — Material / CBAM (weeks 7–10)

- **Scope:** CBAM-covered sectors (steel, aluminium, cement, fertilizers, electricity, hydrogen), CBAM default values, installation-level declarations
- **Source packs:** CBAM EU regulation parser, EPD databases (EC3-EPD, PACT), PCAF
- **Method packs live:** `method_packs/eu_policy.py`, `method_packs/product_carbon.py`
- **Gold-eval:** 250 activities (mostly CBAM goods)
- **GTM hook:** unlocks GL-CBAM-APP end-to-end (the FY27 largest wedge)
- **Acceptance gate:** CBAM quarterly report reconstruction from 3 customer pilots matches EC-tested reference within rounding

### Slice L — Land / Removals (weeks 9–12)

- **Scope:** LULUCF, biogenic CO2, removals, afforestation, soil carbon
- **Source packs:** IPCC AFOLU, LSR, land-use databases
- **Method packs live:** `method_packs/land_removals.py`
- **Gold-eval:** 100 activities
- **GTM hook:** unlocks CSRD E1 E3 E4, future AgriLandOS (FY30)
- **Acceptance gate:** biogenic flag + GWP set propagation verified

### Slice P — Product Carbon (weeks 11–14)

- **Scope:** PCF/LCA variants (cradle-to-gate, cradle-to-grave), allocation rules, ISO 14067
- **Source packs:** ecoinvent (licensed), EC3-EPD, PACT, product_lca_variants.py
- **Method packs live:** `method_packs/product_carbon.py`, `method_packs/product_lca_variants.py`
- **Gold-eval:** 200 activities
- **GTM hook:** unlocks PCF Studio (FY28), DPP Hub (FY28) preview, GL-CSRD-APP E5 circular
- **Acceptance gate:** ecoinvent license-class gate tested; no leakage to non-licensed callers

### Slice FP — Finance Proxy (weeks 13–16)

- **Scope:** PCAF asset-class proxies, spend-based factors (EEIO), industry averages
- **Source packs:** PCAF, EEIO-US, Exiobase-lite, industry codes mapping
- **Method packs live:** `method_packs/finance_proxy.py`, `mapping/spend.py`, `mapping/industry_codes.py`
- **Gold-eval:** 150 activities
- **GTM hook:** unlocks Scope 3 Cat 15 (Investments), future FinanceOS (FY29)
- **Acceptance gate:** PCAF data-quality-score propagates to resolved factor

**Slice acceptance gate (shared):** every slice must pass (a) CI gold-eval threshold, (b) P95 < 180 ms resolve, (c) signed-receipt verification round-trip in SDK test, (d) license-class gate probe, (e) operator console approves an edition-cut end-to-end, (f) changelog auto-published to Explorer UI.

---

## 9. Infrastructure reuse

Factors does not build new infra. It consumes the existing GreenLang stack (per MEMORY.md, status dates 2026-02-06 through 2026-02-07):

| Component | Reused for | Factors-specific config |
|---|---|---|
| INFRA-001 EKS | Compute for all factors workloads | Dedicated namespace `gl-factors`; node selector for matcher pods |
| INFRA-002 PostgreSQL + TimescaleDB | `catalog_repository_pg.py`; TimescaleDB not required for Factors core, but usage metering uses hypertables | Logical DB `factors`; extensions: `pgvector`, `pg_trgm`, `btree_gin` |
| INFRA-003 Redis | `cache_redis.py`, `search_cache.py` | 2 logical DBs: `0` for read cache, `1` for search cache |
| INFRA-004 S3 | Raw source artifacts, parser logs, audit bundles, signed edition manifests | Buckets: `gl-factors-{env}-{region}-raw`, `-bundles`, `-editions` |
| INFRA-005 pgvector | Semantic search index (`matching/pgvector_index.py`) | HNSW (m=24 prod, m=16 staging, m=8 dev) on 768-d MPNet embeddings |
| INFRA-006 Kong | API gateway | Plugins: JWT, rate-limiting, request-transformer (injects tenant claim), prometheus |
| INFRA-007 CI/CD | Build, test, deploy pipelines | Gold-eval gate is a required check per slice; release cut is a manual workflow |
| INFRA-008 Feature flags | `factors.default_edition_v1_next`, per-tenant `edition_pin_override`, canary rollouts | LaunchDarkly-style SDK in FastAPI middleware |
| INFRA-009 Loki | Structured logs (shipped via promtail daemonset, already configured) | Label set: `app=factors,component=<api|matcher|watch|ingestion>` |
| OBS-001 Prometheus | Metrics from `factors/observability/prometheus_exporter.py` | Scrape target: `/metrics` on each pod |
| OBS-002 Grafana | Dashboards | Factors dashboards: Resolve latency, cache hit rate, 7-step distribution, license-gate blocks, signing errors, edition cut events, source-watch health |
| OBS-003 OpenTelemetry | Traces via OTel SDK wired into FastAPI + psycopg3 + httpx | Sampling: 10% prod, 100% below 1k rps |
| OBS-004 Alerting | Alertmanager → PagerDuty + Slack | Alerts listed in §12 |
| OBS-005 SLO/SLI | Error-budget dashboards | SLOs listed in §12 |
| SEC-001 JWT | Edge auth | Kong JWT plugin + FastAPI verifier |
| SEC-002 RBAC | Plan-tier and operator-console roles | Backed by Okta groups; per-endpoint enforcement via `factors/tier_enforcement.py` |
| SEC-003 Encryption at rest | RDS + S3 + Redis encryption on | KMS-managed keys per region |
| SEC-004 TLS 1.3 | All ingress and intra-mesh | Cert-manager + Let's Encrypt for public; internal CA for mesh |
| SEC-005 Audit logging | Every resolve, admin action, edition cut | Streamed to central audit store + Loki |
| SEC-006 Vault | Ed25519 private key, DB creds, Stripe keys, OpenAI/Anthropic keys | External Secrets Operator syncs into K8s Secrets |
| SEC-007 Security scanning | Container scan, IaC scan, SAST | GitHub Actions integration |
| SEC-008 Security policies | Published docs referenced from developer portal | Link from Explorer UI footer |
| SEC-009 SOC 2 Type II | Audit program inherits | Factors included in the SOC 2 scope boundary |
| SEC-010 SecOps automation | Runbooks for key rotation, cert renewal, DR drills | Factors runbooks in `docs/runbooks/factors/` |
| SEC-011 PII detection | Scans every incoming source artifact before promotion | Runs inside `ingestion/parser_harness.py` step |

Net new infra required for Factors v1: **zero**. All components above are listed as BUILT/PRODUCTION READY per MEMORY.md.

---

## 10. Cost model

Rough monthly run-rate across four load tiers. Assumes US-East primary, EU-West secondary, steady-state.

**Per-request workload assumption:**
- Average request mix: 70% resolve (cache hit 75%), 20% search, 8% explain, 2% audit-bundle
- Per request: 0.5 PG query (weighted by cache miss), 0.4 Redis ops, 50 bytes logged, 200 bytes metered, 1 Ed25519 sign
- Embedding refresh: weekly batch on promoted editions, not per-request

### 10.1 Cost breakdown

| Cost line | Driver | 1k calls/day | 10k calls/day | 100k calls/day | 1M calls/day |
|---|---|---|---|---|---|
| EKS compute (FastAPI + matcher + workers) | Pod CPU/mem | $600 | $900 | $2,200 | $12,000 |
| RDS PostgreSQL (multi-AZ, m6g.large → xlarge → 2xlarge) | DB size + IOPS | $400 | $550 | $1,800 | $8,500 |
| ElastiCache Redis (cache.m6g.large cluster) | Memory + ops | $180 | $250 | $700 | $3,200 |
| S3 storage (raw + bundles + editions) | GB-month | $60 | $90 | $300 | $1,400 |
| S3 requests (GET/PUT) | Op volume | $10 | $40 | $220 | $1,500 |
| Egress (public Explorer + SDK downloads + API responses) | GB out | $40 | $200 | $1,400 | $9,000 |
| NAT gateway + VPC endpoints | Fixed + data | $120 | $150 | $280 | $900 |
| ALB + Kong | Fixed + request units | $80 | $110 | $240 | $900 |
| CloudFront / Vercel (Explorer UI) | Hits + egress | $20 | $60 | $280 | $1,400 |
| OpenAI embeddings (matching + weekly refresh) | $0.00002 per 1k tokens, avg 200 tok/activity | $5 | $20 | $180 | $1,500 |
| Anthropic rerank (top-k=20, P@1 refresh + suggestion_agent) | $3/M input + $15/M output tokens, used only in matching pipeline and suggestion agent | $15 | $90 | $700 | $5,500 |
| Stripe fees (on usage-billed portion) | 0.5% platform fee on metered revenue | $5 | $50 | $500 | $5,000 |
| Vault + External Secrets | Fixed | $90 | $90 | $90 | $90 |
| Observability (Prometheus storage, Loki storage, Grafana Cloud if applicable) | Cardinality + ingest | $150 | $250 | $650 | $2,400 |
| Cross-region replication (US↔EU edition sync) | Data transfer | $40 | $60 | $180 | $900 |
| Cert-manager, cert-issuance | — | $0 | $0 | $0 | $0 |
| **Total monthly (single region US)** | | **$1,815** | **$2,910** | **$9,740** | **$54,190** |
| **Total monthly (US + EU, add ~80% of above)** | | **~$3,270** | **~$5,240** | **~$17,530** | **~$97,540** |

### 10.2 Cost shape observations

- Compute scales roughly linearly above 100k/day; below that, we pay for redundancy floor (minimum 3 replicas, multi-AZ DB).
- Anthropic rerank cost is the largest variable — used only in the matching pipeline (`matching/llm_rerank.py`), not on resolve path. Cache the top-20 reranks per normalized activity string for 90 days (`search_cache.py`) — this reduces rerank cost by ~70% at steady state.
- OpenAI embedding cost is negligible until 1M/day because we embed once per factor at promotion time, not per request.
- Egress is the second-largest variable above 1M/day — mitigate with CloudFront in front of `/search` (24h TTL on warm hashes) and aggressive response compression.
- Stripe fees are pass-through against metered revenue; they only bite when the customer pays.
- **Break-even vs. 70% gross margin** needs ARPA ≥ ~$350/month at 100k calls/day; EU + US combined flips that to ~$600/month. Enterprise plans ($2k–$10k/month) clear easily. Developer/free plans are subsidized by Pro+ tiers.

### 10.3 Unit economics watchpoints

| Watch | Signal | Mitigation |
|---|---|---|
| Cache hit rate drops below 75% | Cache size too small or hot-set diversity up | Increase Redis memory; push warm top-10k into L1 in-proc cache (`performance_optimizer.py`) |
| Rerank cost per 100k > $1,000 | Cache poisoning or cold start | Pre-warm rerank cache after each edition cut |
| Egress per 100k > $2,000 | Explorer SSR hydration leak or SDK retry storm | Enable response compression; adjust SDK retry backoff in `transport.py` |
| PG IOPS hot | Missing index or N+1 in resolve path | Prepared-statement cache; verify cascade query plan |

---

## 11. API surface (v1)

Factors v1 freezes the following endpoints under `/v1/`. Exact request/response shapes track Pydantic models already defined in `factors/api_endpoints.py` + `factors/resolution/request.py` + `factors/resolution/result.py`.

| Method | Path | Purpose | Source file |
|---|---|---|---|
| POST | `/v1/resolve` | Single factor resolution | `api_endpoints.py` + `resolution/engine.py` |
| POST | `/v1/bulk-resolve` | Batch resolution (max 1000 per request) | `batch_jobs.py` |
| POST | `/v1/search` | Hybrid semantic + full-text search with sort + pagination | `api_endpoints.py` F035 |
| GET | `/v1/factors/{id}` | Single factor detail | `catalog_repository_pg.py` |
| GET | `/v1/explain/{id}` | 7-step trace, provenance, DQS, alternates | `api_endpoints.py` F032-adapted |
| GET | `/v1/editions` | List editions | `edition_manifest.py` |
| GET | `/v1/editions/{id}` | Edition manifest (signed) | `edition_manifest.py` |
| GET | `/v1/editions/diff` | Edition-to-edition diff (F034) | `api_endpoints.py` + `service.py::compare_editions` |
| GET | `/v1/sources` | Source registry listing | `source_registry.py` |
| GET | `/v1/method-packs` | Method pack overviews | `method_packs/registry.py` |
| POST | `/v1/audit-bundle` | Async audit bundle export | `api_endpoints.py::build_audit_bundle` |
| GET | `/v1/audit-bundle/{job_id}` | Poll + download URL | `batch_jobs.py` |
| POST | `/v1/overrides` | Customer override CRUD | `tenant_overlay.py` |
| GET | `/v1/catalog/stats` | Three-label dashboard counts | `inventory.py::write_coverage_matrix` |
| POST | `/v1/impact-simulate` | Delta preview before edition pin | `quality/impact_simulator.py` |
| POST | `/v1/receipts/verify` | Verify a receipt we issued | `signing.py` |
| GET | `/ops/*` | Operator console APIs (SSO-gated) | Various admin routes |

All responses carry signed receipts. OpenAPI spec auto-generated at `/v1/openapi.json` and rendered at `/docs`.

---

## 12. Monitoring, SLOs, and alerts

### 12.1 SLOs (committed to customers on Enterprise plan)

| Service | SLI | SLO | Error budget (30d) |
|---|---|---|---|
| Resolve | Availability | 99.9% | 43 min |
| Resolve | Latency P95 | < 200 ms | 5% slow |
| Search | Latency P95 | < 400 ms | 5% slow |
| Audit bundle | Time to deliver | < 10 min | 5% slow |
| Edition promotion | No data-integrity regression | 100% | 0 |

### 12.2 Metrics (Prometheus, from `factors/observability/prometheus_exporter.py`)

- `factors_resolve_total{edition_id, step_fired, method_pack, tier, outcome}` — counter
- `factors_resolve_latency_seconds{endpoint}` — histogram
- `factors_cache_hit_total{layer, outcome}` — counter
- `factors_license_gate_block_total{reason, tier}` — counter
- `factors_signing_error_total{algorithm, reason}` — counter
- `factors_edition_cut_total{status}` — counter
- `factors_source_watch_failure_total{source_id, reason}` — counter
- `factors_rerank_cost_dollars_total` — counter (derived from upstream provider API)
- `factors_dqs_distribution{bucket}` — gauge (histogram of DQS scores over promoted factors)

### 12.3 Alerts (OBS-004)

| Alert | Trigger | Severity | Route |
|---|---|---|---|
| `FactorsResolveP95High` | P95 > 300ms for 10m | warning | Slack #gl-factors |
| `FactorsResolveErrorBudgetFastBurn` | 2% budget in 1h | critical | PagerDuty primary |
| `FactorsLicenseGateSpike` | block rate > 5% of requests | warning | Slack + SecOps |
| `FactorsSigningFailure` | any signing error | critical | PagerDuty primary |
| `FactorsSourceWatchStalled` | source > 2x cadence without update | warning | Slack #gl-factors-ops |
| `FactorsEditionCutStuck` | orchestrator stuck > 30m | critical | PagerDuty primary + ops lead |
| `FactorsCacheHitRateLow` | < 60% over 30m | warning | Slack |
| `FactorsReRankCostSpike` | daily spend > 3x 7d baseline | warning | Slack + finance |

---

## 13. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM creeps into calculation path | Zero-hallucination guarantee broken | Architectural rule enforced in CI: `resolution/engine.py` imports blocked from calling any LLM provider; `matching/` is allowed to, but output must be a *candidate list*, not a value |
| Licensed connector content leaks through open API | Legal exposure (ecoinvent, IEA) | License-class gate enforced at middleware; probe test in CI; audit of every resolve with license_class != `open` |
| Edition cut breaks downstream apps | CBAM/CSRD runs start returning different numbers silently | Edition pin is explicit per downstream app; impact_simulator.py surfaces deltas before promotion; customers notified via webhooks |
| Rerank cost runs away | Unit-economics break | Pre-cache top-20 reranks at edition cut; daily budget alarm; circuit-breaker fallback to deterministic-only match |
| pgvector HNSW quality regresses after edition cut | Matching P@1 drops silently | Gold-eval gate in CI blocks promotion on P@1 regression > 2% |
| EU data residency violation | Legal exposure | Per-region independent DBs; no cross-region replication of customer override tables; residency test in CI |
| Key rotation breaks old receipts | Customer auditors can't verify old data | Old kids remain verifiable for 2 years via Vault historical key vault; customers cache kid→pubkey mapping |
| Operator Console compromise | Unauthorized edition cut | Two-person signoff at `quality/release_signoff.py`; SAML + hardware key required; every action audit-logged |

---

## 14. Timeline (FY27 engineering estimate)

Assumption: full-time team of 4 backend + 2 frontend + 1 devops + 1 PM, building on top of the 85–92% complete codebase.

| Phase | Weeks | Output |
|---|---|---|
| Phase 0 — deployment foundation | 1–2 | EKS namespace, Terraform for Factors-specific infra, Vault paths, Kong routes, staging cluster live |
| Phase 1 — Electricity slice live | 1–4 | End-to-end E slice in prod; public Explorer launched read-only; SDK published |
| Phase 2 — Combustion slice live | 3–6 | C slice; Operator Console MVP; enterprise SSO |
| Phase 3 — Freight slice live | 5–8 | F slice; audit-bundle export GA; webhooks GA |
| Phase 4 — Material / CBAM slice live | 7–10 | M slice; GL-CBAM-APP uses hosted Factors; first paid pilot |
| Phase 5 — Land/Removals slice live | 9–12 | L slice; CSRD E1–E5 unlocked |
| Phase 6 — Product Carbon slice live | 11–14 | P slice; ecoinvent tier live; PCF Studio (FY28) preview wiring |
| Phase 7 — Finance Proxy slice live | 13–16 | FP slice; PCAF live; v1.0 Certified edition cut |
| Phase 8 — Hardening + SOC 2 inclusion | 15–18 | Penetration test, SOC 2 audit evidence, v1.0 GA announcement |

Phases 1–7 overlap by design; slices are additive, not sequential. v1.0 GA at week 18.

---

## 15. What is explicitly out of scope for v1

- New sources beyond the 19 already parsed. (Additions come post-v1 via normal source-watch + slice expansion.)
- Real-time (sub-second) factor streaming. Factors are edition-bound; the streaming primitive is edition_cut, not per-factor push.
- Customer-authored method packs. v1 ships the 7 packs under `method_packs/`. OEMs customize via overrides (`tenant_overlay.py`), not method-pack forks.
- FY28+ products: SupplierOS, PCF Studio, DPP Hub, PlantOS, BuildingOS. These consume Factors v1 but are separate applications.
- On-prem / VPC deployment for customers. v1 is multi-tenant SaaS only. On-prem appears on the FY28 roadmap.

---

## 16. Acceptance criteria — "v1 is done"

- All 7 slices promoted; v1.0 Certified edition cut via `release_signoff.py` with two-person approval
- Three-label dashboard (Certified / Preview / Connector-only counts) live on Explorer home
- Public Explorer live with search, explain, editions diff, method-pack docs
- Operator Console live behind SSO with ingest / review / promote / rollback flows
- Python + TypeScript SDKs published (`greenlang-factors` on PyPI, `@greenlang/factors` on npm)
- Stripe SKUs live for Free, Dev, Pro, Enterprise, OEM
- Signed receipts enforced at middleware; SDKs include receipt verification
- License-class gate enforced at response boundary; probe test green
- P95 resolve latency < 200ms under 500 rps sustained load
- Gold-eval gate (P@1 ≥ 0.85) passing in CI for every slice
- SOC 2 Type II scope boundary includes Factors
- Five design-partner pilots live (per FY27 business plan)
- Two of (CBAM, Scope Engine, CSRD) paid pilots resolving factors against hosted v1.0

---

**End of document.**
