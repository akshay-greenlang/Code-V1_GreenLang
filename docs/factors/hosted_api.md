# GreenLang Factors — Hosted API

> **What this is.** The operator guide for standing up the GreenLang Factors API as a hosted SaaS. Covers deployment, auth, billing, SLAs, and support.
> **Who it's for.** DevOps engineers operating a GreenLang tenant or our own SaaS control plane.

---

## 1. Architecture at a glance

```
               ┌───────────────────────────────────────┐
   clients ──▶ │  Ingress (NGINX / API Gateway)        │
               │  TLS 1.3, rate limits, DDoS           │
               └───────────────┬───────────────────────┘
                               │
                  ┌────────────┴───────────┐
                  │  FastAPI API pods      │  HPA 2-20 replicas
                  │  greenlang-factors-api │
                  └─┬──────────────────────┘
                    │
      ┌─────────────┼─────────────┬──────────────┐
      ▼             ▼             ▼              ▼
   Postgres     pgvector       Redis          Worker pods
   (factors +   (embeddings)   (rate limits   (ingestion +
    ledger +                    + tier cache)   QA +
    vault +                                     watch pipeline)
    entity +
    policy +
    billing)
```

**Deployable artefact.** `deployment/helm/greenlang-factors/` — a Helm chart with:

| Template | Purpose |
|---|---|
| `deployment-api.yaml` | FastAPI service pods |
| `deployment-worker.yaml` | Background ingestion + QA + watch workers |
| `migration-job.yaml` | Runs the Postgres migration chain (V001 → V441) |
| `service.yaml` | ClusterIP for the API pods |
| `ingress.yaml` | External TLS + path routing |
| `hpa.yaml` | Horizontal pod autoscaler (2–20 API pods, 1–5 workers) |
| `pdb.yaml` | PodDisruptionBudget (min 1 API, min 1 worker) |
| `servicemonitor.yaml` + `prometheusrule.yaml` | Prometheus scraping + alert rules |
| `networkpolicy.yaml` | Default-deny with allow-lists for ingress + DB + Redis |
| `external-secrets.yaml` | HashiCorp Vault / AWS Secrets Manager integration |
| `smoke-test-job.yaml` | Post-deploy verification |

## 2. Environments

Three values files ship with the chart:

| File | Purpose | Notable overrides |
|---|---|---|
| `values.yaml` | Base defaults | Image tag, replica counts, resources |
| `values-dev.yaml` | Dev cluster | 1 API, 1 worker, relaxed resources, SQLite fallback |
| `values-staging.yaml` | Staging cluster | 2 API, 1 worker, real Postgres, staging DNS |
| `values-prod.yaml` | Production | 4+ API, 2+ worker, Postgres HA, PagerDuty, SLO 99.9 % |

All deploys use the same chart + different values file.

## 3. Deploy (staging walk-through)

Prereqs: kubectl access to the target cluster, Helm 3.13+, a Postgres 16 instance, a Redis 7 instance, and DNS delegation for `factors-staging.greenlang.io`.

```bash
# 1. Pull chart + set release name.
cd deployment/helm/greenlang-factors
RELEASE=greenlang-factors
NS=greenlang-factors-staging

# 2. Create namespace + bootstrap secrets.
kubectl create namespace $NS
kubectl -n $NS create secret generic factors-db \
  --from-literal=dsn="postgresql://..." \
  --from-literal=username="factors" \
  --from-literal=password="..."
kubectl -n $NS create secret generic factors-jwt \
  --from-literal=jwt-secret="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"

# 3. Run the migration chain (Flyway-style job).
helm install $RELEASE-migrate . \
  -f values-staging.yaml \
  --namespace $NS \
  --set migration.enabled=true \
  --set deployment.api.replicas=0 \
  --set deployment.worker.replicas=0
kubectl -n $NS wait --for=condition=complete --timeout=5m \
  job/$RELEASE-migrate-db

# 4. Full deploy.
helm upgrade --install $RELEASE . \
  -f values-staging.yaml \
  --namespace $NS \
  --wait --timeout 10m

# 5. Run smoke tests.
helm test $RELEASE --namespace $NS --timeout 5m
```

## 4. Authentication

Two authentication paths are accepted. See `greenlang/factors/api_auth.py`.

### 4.1 JWT Bearer (production default)

- Token signed with HS256 using `JWT_SECRET` (env var, 32+ chars, never `changeme`).
- Expected claims: `sub`, `tier` (`community`/`pro`/`enterprise`/`internal`), `tenant_id`, optional `roles`, `permissions`.
- Token TTL: 24 hours (refresh by re-issuing).

Example request:

```bash
curl -H "Authorization: Bearer $JWT" \
  https://factors.greenlang.io/api/v1/factors/search?q=diesel
```

### 4.2 API Key (developer tier + self-serve)

Keyring sources (either or both):

- **Env var** `GL_FACTORS_API_KEYS` — JSON blob listing `{key_id, key, tier, tenant_id, user_id, active}`.
- **Env var** `GL_FACTORS_API_KEY_FILE` — path to a JSON file of the same shape. Preferred for production (ConfigMap / external-secrets).

Example request:

```bash
curl -H "X-API-Key: gl_live_sk_xxx" \
  https://factors.greenlang.io/api/v1/factors/search?q=diesel
```

### 4.3 Installing the middleware

```python
from fastapi import FastAPI
from greenlang.factors.middleware.auth_metering import install_factors_middleware

app = FastAPI()
install_factors_middleware(app, protected_prefix="/api/v1/factors")
# Register the Factors router after installing the middleware.
from greenlang.integration.api.routes.factors import router as factors_router
app.include_router(factors_router)
```

The middleware attaches `request.state.user` + `request.state.tier`, enforces the tier gate, and emits a credit-weighted usage event per request.

### 4.4 Tier → endpoint matrix

| Endpoint | Community | Pro | Enterprise |
|---|---|---|---|
| `GET /api/v1/factors` | ✅ | ✅ | ✅ |
| `GET /api/v1/factors/search` | ✅ | ✅ | ✅ |
| `POST /api/v1/factors/search/v2` | ✅ | ✅ | ✅ |
| `GET /api/v1/factors/search/facets` | ✅ | ✅ | ✅ |
| `GET /api/v1/factors/coverage` | ✅ | ✅ | ✅ |
| `GET /api/v1/factors/{id}` | ✅ | ✅ | ✅ |
| `POST /api/v1/factors/match` | ❌ | ✅ | ✅ |
| `GET /api/v1/factors/export` | ❌ | ✅ | ✅ |
| `GET /api/v1/factors/{id}/diff` | ❌ | ✅ | ✅ |
| `GET /api/v1/factors/{id}/audit-bundle` | ❌ | ❌ | ✅ |

## 5. Billing + usage metering

### 5.1 Credit table

| Endpoint | Credits |
|---|---|
| `search`, `search/v2`, `search/facets`, `coverage` | **1** |
| `match` | **2** |
| `diff` | **2** |
| `export` | **1 per 100 rows** (min 1) |
| `audit-bundle` (Enterprise only) | **5** |
| everything else | **1** |

Implementation: `greenlang/factors/billing/metering.py::credits_for`.

### 5.2 Persistence

Set `GL_FACTORS_USAGE_SQLITE=/var/lib/greenlang/usage.sqlite` (or mount an EBS volume). Two tables are written on every request:

- `api_usage_events` — legacy hit log.
- `api_usage_credits` — credit-weighted log with `(tier, endpoint, method, user_id, tenant_id, api_key_id, credits, row_count, status_code, recorded_at)`.

For production, an aggregator job (`greenlang/factors/billing/aggregator.py`) rolls credits into monthly invoices and pushes events to the Stripe webhook (`webhook_handler.py`).

### 5.3 Operations

- **Dashboards.** Prometheus metrics via `observability/prometheus.py`; ship alongside Grafana dashboards from `deployment/monitoring/grafana-dashboards/factors-*.json`.
- **Alerting.** `prometheusrule.yaml` covers p95 latency > 500 ms, error rate > 1 %, credit-ingest lag > 5 min.
- **PagerDuty.** Integrates via the central alertmanager config (see `deployment/helm/monitoring/`).

## 6. SLA tiers

| Tier | Monthly uptime | Response time (p95) | Support |
|---|---|---|---|
| Developer (free) | Best-effort | No guarantee | Community (GitHub issues) |
| Startup | 99.5 % | 500 ms | Email, next business day |
| Enterprise | 99.9 % (prod), 99.5 % (staging) | 250 ms | Dedicated TAM, 1-hour response for P1 |

Credits on downtime: Enterprise gets 10 % service credit per 0.1 % below 99.9 %.

## 7. Migration chain

All database work is captured as numbered Flyway-style migrations under `deployment/database/migrations/sql/`. The chain includes (latest additions):

| Migration | Purpose |
|---|---|
| V426–V436 | Factors catalog + search indexes + GA billing / SLA |
| V437 | Scope Engine computations |
| V438 | Comply Hub jobs |
| **V439** | **Climate Ledger** (Phase 2.1) |
| **V440** | **Evidence Vault** (Phase 2.2) |
| **V441** | **Entity Graph** (Phase 2.3) |

Every migration includes append-only triggers where applicable so a compromised app cannot mutate history.

## 8. Runbook

See `deployment/helm/greenlang-factors/RUNBOOK.md` for incident response, rollback steps, and on-call playbooks.

## 9. References

- `docs/CLI_REFERENCE.md` — `gl factors *` surface.
- `docs/REPO_TOUR.md` — L1 Factors module mapping.
- `docs/sales/FACTORS_API_BATTLECARD.md` — commercial positioning.
- `greenlang/factors/sdk/README.md` — Python + TypeScript client libraries.
- `docs/factors/sdk.md` — SDK install + usage.

---

*Last updated: 2026-04-20. Source: FY27_vs_Reality_Analysis.md Phase 4.3; Helm chart at `deployment/helm/greenlang-factors/`.*
