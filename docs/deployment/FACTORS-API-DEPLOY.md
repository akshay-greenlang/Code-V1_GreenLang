# Factors API — Deploy Runbook

Owner: `factors-team` &nbsp;•&nbsp; On-call: PagerDuty service `greenlang-factors`
&nbsp;•&nbsp; Slack: `#factors-alerts`

This runbook is written for a senior engineer with **zero prior GreenLang
context**. Five-minute-to-staging is the goal.

---

## 1. Quickstart — Staging in 5 commands

```bash
# 1. AWS + kubectl context
aws eks update-kubeconfig --name greenlang-staging --region us-east-1

# 2. Apply (Kustomize renders the full base + staging overlay)
kubectl apply -k deployment/k8s/factors/overlays/staging

# 3. Apply Kong declarative config (one-time per cluster)
kubectl apply -n greenlang-factors -f deployment/k8s/factors/kong/factors-api.yaml

# 4. Wait for rollout
kubectl -n greenlang-factors rollout status deploy/factors-api --timeout=10m

# 5. Smoke (inside the cluster)
kubectl -n greenlang-factors run smoke-$RANDOM --rm -i --restart=Never \
  --image=curlimages/curl:8.7.1 -- \
  sh -c 'curl -fsS http://factors-api.greenlang-factors.svc.cluster.local:8080/api/v1/health'
```

If all five succeed, staging is live at `https://staging.greenlang.io`.

---

## 2. Prerequisites

| Item | How to obtain | Why |
|------|---------------|-----|
| AWS IAM creds (role `greenlang-factors-staging-deployer`) | `aws sso login --profile greenlang-staging` | EKS access + IRSA. |
| Kubeconfig pointing at `greenlang-staging` EKS cluster | `aws eks update-kubeconfig --name greenlang-staging` | Apply manifests. |
| Vault token (reader on `secret/factors/staging/*`) | `vault login -method=oidc` | Secret provisioning only. |
| Kong Admin API access | VPN + `kubectl port-forward svc/kong-admin 8001:8001 -n kong` | Inspect / debug plugins. |
| Stripe live-mode keys in Vault | See §4; ops populates these | Billing + webhooks. |
| GitHub PAT with `write:packages` | Only needed for manual image build; CI has its own | Push to GHCR. |
| `kubectl`, `kustomize`, `aws` CLIs installed | `brew install kubectl kustomize awscli` | Duh. |

GreenLang infra reuse (do NOT recreate):

- **INFRA-001** EKS cluster — Terraform in `deployment/terraform/modules/eks/`
- **INFRA-002** RDS PostgreSQL + TimescaleDB + pgvector
- **INFRA-003** Redis (ElastiCache) cluster
- **INFRA-006** Kong API gateway
- **SEC-001/002/006** JWT + RBAC + Vault
- **OBS-001..005** Prometheus / Grafana / OTel / alerting / SLO dashboards
- **INFRA-009** Loki log aggregation

---

## 3. What this deploy contains

```
deployment/k8s/factors/
├── base/
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── serviceaccount.yaml           # IRSA — patched per-env
│   ├── configmap.yaml                # non-secret env
│   ├── externalsecrets.yaml          # SEC-006 Vault wiring
│   ├── deployment.yaml               # factors-api (3 replicas, uid 10001)
│   ├── deployment-worker.yaml        # 4 workers: ingest / embed / dedupe / watch
│   ├── service.yaml                  # ClusterIP 8080 + 9090
│   ├── hpa.yaml                      # 3..10 on CPU 70% + factors_request_rate
│   ├── pdb.yaml                      # minAvailable 2
│   ├── networkpolicy.yaml            # default-deny + allowlist
│   ├── servicemonitor.yaml           # OBS-001 Prometheus scrape
│   ├── ingressroute.yaml             # Kong + fallback Ingress
│   └── prometheusrule.yaml           # OBS-004 alerts
├── kong/factors-api.yaml             # Kong plugins + consumer groups
└── overlays/
    ├── staging/                      # 1 replica, staging.greenlang.io
    └── prod/                         # 3..10 replicas, factors.greenlang.io
```

---

## 4. Secret provisioning (Vault)

All secrets live at `secret/factors/{staging,prod}/*` in Vault (SEC-006).
External Secrets Operator syncs them into the `factors-api-secrets` k8s
Secret on a 5-minute refresh.

| Vault path | Keys | Env var consumed by `main.py` | Rotation | Rotator |
|------------|------|-------------------------------|----------|---------|
| `secret/factors/{stage}/jwt` | `secret` | `GL_JWT_SECRET` (+ alias `JWT_SECRET`) | 90 d | security-ops |
| `secret/factors/{stage}/api_keys` | `csv` | `GL_API_KEYS` | 90 d | factors-team |
| `secret/factors/{stage}/signing` | `hmac_secret`, `ed25519_private_key` | `GL_FACTORS_SIGNING_SECRET`, `GL_FACTORS_ED25519_PRIVATE_KEY` | 90 d (HMAC), 180 d (Ed25519) | factors-team + coordinate with verifier clients |
| `secret/factors/{stage}/postgres` | `dsn` | `DATABASE_URL` | 30 d (vault-agent auto) | infra-ops |
| `secret/factors/{stage}/redis` | `url` | `REDIS_URL` | 30 d | infra-ops |
| `secret/factors/{stage}/vendor` | `openai_api_key`, `anthropic_api_key` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | 180 d | factors-team |
| `secret/factors/{stage}/stripe` | `api_key`, `webhook_secret` | `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET` | 180 d (or on Stripe rotation) | billing-ops |

Writing a secret (one-time):

```bash
vault kv put secret/factors/staging/jwt secret="$(openssl rand -hex 64)"
vault kv put secret/factors/staging/signing \
  hmac_secret="$(openssl rand -hex 64)" \
  ed25519_private_key="$(cat factors-ed25519.priv)"
vault kv put secret/factors/staging/vendor \
  openai_api_key="sk-..." \
  anthropic_api_key="sk-ant-..."
vault kv put secret/factors/staging/stripe \
  api_key="sk_test_..." \
  webhook_secret="whsec_..."
```

Verify sync:

```bash
kubectl -n greenlang-factors get externalsecret factors-api-secrets
kubectl -n greenlang-factors get secret factors-api-secrets -o json \
  | jq '.data | keys'
```

---

## 5. Database migration (Alembic)

Migrations live at `deployment/database/migrations/` and are **shared** with
the rest of GreenLang. The factors API only depends on `factors.*` schemas
which are created by migrations V326-V395 (per-pack table).

```bash
# From any pod with DATABASE_URL and alembic installed
kubectl -n greenlang-factors exec -it deploy/factors-api -- \
  alembic -c deployment/database/migrations/alembic.ini upgrade head
```

**Rule:** migrations are applied BEFORE the new image rolls out. The CI
`deploy-staging` job already enforces this ordering when the migration
revision is committed alongside application code.

Rollback:

```bash
alembic -c deployment/database/migrations/alembic.ini downgrade -1
```

---

## 6. First-time staging deploy (fresh cluster)

```bash
# 1. Create the namespace (idempotent; Kustomize also creates it)
kubectl create ns greenlang-factors --dry-run=client -o yaml | kubectl apply -f -

# 2. Ensure External Secrets Operator + Kong KIC are installed
kubectl get deploy -A | egrep 'external-secrets|kong'

# 3. Populate Vault (see §4) — REQUIRED before apply, or pods CrashLoop on missing env.

# 4. Kustomize dry-run
kubectl kustomize deployment/k8s/factors/overlays/staging \
  | kubectl apply --dry-run=client -f -

# 5. Apply
kubectl apply -k deployment/k8s/factors/overlays/staging
kubectl apply -n greenlang-factors -f deployment/k8s/factors/kong/factors-api.yaml

# 6. Migrate
kubectl -n greenlang-factors exec -it deploy/factors-api -- \
  alembic -c deployment/database/migrations/alembic.ini upgrade head

# 7. Wait
kubectl -n greenlang-factors rollout status deploy/factors-api --timeout=10m

# 8. External smoke
GL_FACTORS_STAGING_URL=https://staging.greenlang.io \
GL_FACTORS_STAGING_JWT=$(vault kv get -field=test_token secret/factors/staging/smoke) \
  pytest tests/factors/smoke/test_staging.py -v
```

---

## 7. Prod deploy sequence (2-approver gate)

1. Open the Actions tab and run `factors-api-deploy` workflow with
   `stage=prod`.
2. Workflow stops at `gate-prod` — GitHub environment `factors-prod-gate`
   requires **2 approvers** (configured in repo settings → Environments).
3. After approval, `deploy-prod` runs:
   - `kubectl apply -k deployment/k8s/factors/overlays/prod`
   - Waits for rollout.
   - Runs `tests/factors/smoke/test_staging.py` against `factors.greenlang.io`.
4. On success, Slack `#factors-alerts` gets a `:rocket:` notification.
5. On failure, `rollback-on-failure` runs `kubectl rollout undo` and posts
   `:rotating_light:` to Slack.

Never bypass the gate. If you must emergency-deploy, use:

```bash
# Last-resort manual rollback
kubectl -n greenlang-factors rollout undo deploy/factors-api
```

---

## 8. Smoke tests — what must pass

`tests/factors/smoke/test_staging.py` must return green on all of:

- `GET /api/v1/health` → 200 + `status=healthy`
- `X-Request-Id` echoed by Kong correlation-id plugin
- `GET /api/v1/factors` without auth → 401/403
- `GET /api/v1/factors?limit=5` with JWT → 200 + `X-Factors-Edition` header
- `GET /api/v1/stats/coverage` with JWT → 200 + `X-Factors-Edition`
- `GET /api/v1/factors/{id}` → 200 + `data_quality` payload + signed-receipt header
- Rate limit: burst of 80 `/api/v1/health` → at least one 429 (if using
  community-tier smoke key)
- Security headers: `Strict-Transport-Security`, `X-Content-Type-Options`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` all present

---

## 9. Canary / blue-green pattern

The factors API uses **edition rollback** as the first line of defence
before touching pod images. A bad edition can be reverted in seconds:

```bash
# List editions
python -m greenlang.factors.cli editions

# Roll back to the previous edition
python -m greenlang.factors.cli rollback --edition-id 2026.03.1
```

For image-level canary, label a single replica:

```bash
# 1. Scale up by 1 with the new image
kubectl -n greenlang-factors set image deploy/factors-api \
  factors-api=ghcr.io/greenlang/factors-api:sha-<new> --record
kubectl -n greenlang-factors scale deploy/factors-api --replicas=4

# 2. Watch latency + 5xx on Grafana `factors-api / overview` dashboard.
#    If golden, let HPA settle back to steady state. If red, rollback:
kubectl -n greenlang-factors rollout undo deploy/factors-api
```

Blue-green: duplicate the deployment with `name: factors-api-green`,
point the Service selector at `app.kubernetes.io/version=green`, flip
back if SLOs break.

---

## 10. Incident response matrix

### Incident: p95 high <a id="incident-p95-high"></a>

Alert: `FactorsAPIHighLatencyP95` (> 500ms for 5m).

1. Check `factors-api / latency` Grafana panel (OBS-002). If the spike is
   localized to a pod, `kubectl delete pod <pod>` to cycle it.
2. Check downstream: `aws rds describe-db-clusters --query 'DBClusters[*].{id:DBClusterIdentifier,cpu:DBClusterInstanceClass}'`
   and the `cloudwatch:pgvector-latency` dashboard — pgvector HNSW is the
   most common offender.
3. If latency is caused by OpenAI/Anthropic, temporarily disable the
   rerank path: `kubectl -n greenlang-factors set env deploy/factors-api
   GL_FACTORS_RERANK_ENABLED=false` and notify `#factors-alerts`.

### Incident: 5xx high <a id="incident-5xx-high"></a>

Alert: `FactorsAPIHighErrorRate` (> 1% for 5m).

1. `kubectl -n greenlang-factors logs -l app.kubernetes.io/name=factors-api --tail=200 | grep ERROR`
2. If the latest deploy is the cause, `kubectl rollout undo deploy/factors-api`.
3. If DB is unreachable, verify RDS + security-group + ExternalSecret sync.

### Incident: signing broken <a id="incident-signing-broken"></a>

Alert: `FactorsSigningFailures` (> 0 for 5m).

1. **Halt all deploys.** Signed receipts are audit-critical.
2. Check `GL_FACTORS_SIGNING_SECRET` and `GL_FACTORS_ED25519_PRIVATE_KEY`
   are set: `kubectl -n greenlang-factors get secret factors-api-secrets
   -o json | jq '.data | keys'`
3. If Vault sync lagged, force refresh:
   `kubectl -n greenlang-factors annotate externalsecret factors-api-secrets \
     force-sync=$(date +%s) --overwrite`
4. Page security on-call.

### Incident: source-license violation <a id="incident-license-violation"></a>

Alert: `FactorsSourceLicenseViolation`.

1. Freeze ingestion: `kubectl -n greenlang-factors scale deploy/factors-worker-ingest --replicas=0`
2. Identify the offending source: `kubectl logs -l greenlang.io/worker-mode=ingest --tail=500 | grep license_violation`
3. Page `#legal-oncall` — DO NOT publish the edition.

### Incident: tenant leak <a id="incident-tenant-leak"></a>

Alert: `FactorsTenantLeakSuspected`.

1. **Immediately** rotate the JWT signing key (`vault kv put secret/factors/prod/jwt secret=$(openssl rand -hex 64)`).
2. Force pod restart: `kubectl -n greenlang-factors rollout restart deploy/factors-api`.
3. Extract the offending request IDs from Loki and open a P0 security
   incident.

### Incident: replicas low <a id="incident-replicas-low"></a>

Alert: `FactorsAPIReplicasBelowPDB`.

1. `kubectl -n greenlang-factors get pods -o wide` — look for Pending / CrashLoop.
2. If Pending, cluster is out of capacity. Check Karpenter logs.
3. If CrashLoop, `kubectl logs <pod> --previous`.

---

## 11. On-call rotation

PagerDuty schedule: **[greenlang-factors](https://greenlang.pagerduty.com/schedules#factors)**
(URL placeholder — update once the PD service is created).

Primary: factors-team weekly rotation.
Escalation: platform-sre after 15m.

---

## 12. Known gotchas

1. **Dockerfile CMD is currently wrong.** `deployment/docker/Dockerfile.factors-service`
   line 111 runs `uvicorn greenlang.factors.api:app`, but that module
   does not exist. The real app is `greenlang.integration.api.main:app`.
   The `deployment.yaml` in this directory overrides the CMD with the
   correct target via `command:` + `args:`. **Fix the Dockerfile before
   the next major release so the override can be deleted.**

2. **JWT_SECRET env-var alias.** The factors API reads `GL_JWT_SECRET`
   directly (`os.getenv("GL_JWT_SECRET")`). Some SEC-001 middleware
   modules still read `JWT_SECRET`. The `factors-api-secrets` ExternalSecret
   populates both keys with the same value; do NOT remove the alias until
   every middleware consumer is audited.

3. **Worker watch vs CronJob.** `deployment-worker.yaml` ships a long-running
   `factors-worker-watch` Deployment AND the existing `factors-watch-daily`
   CronJob at `deployment/k8s/factors-watch-cronjob.yaml` runs. The watcher
   is idempotent by design; the cron acts as a belt-and-braces catch-up
   for sources on annual/quarterly cadence.

4. **Kong Rate-Limit Advanced requires Redis.** The `factors-rate-limit`
   plugin depends on `redis-master.redis.svc.cluster.local`. If Redis is
   down, rate-limiting fails *open*. Alert `RedisDown` (OBS-004) must be
   wired to the same PagerDuty service.

5. **Kustomize `replicas` + HPA conflict.** The Kustomize `replicas:`
   stanza sets the Deployment's initial replica count; the HPA
   re-reconciles within 30s. Don't be surprised if `kubectl get deploy`
   shows a different count than the overlay.

6. **Staging uses `letsencrypt-staging` issuer** — browsers will warn on
   cert trust. Curl works with `-k`; production uses `letsencrypt-prod`.

7. **IRSA role ARNs** in `patch-serviceaccount.yaml` currently point at
   `123456789012`. Replace with your actual AWS account ID before first
   apply. Terraform module at `deployment/terraform/modules/iam/factors/`
   emits the correct ARN.

---

## 13. Deployment sanity checklist (every deploy)

- [ ] `kubectl kustomize deployment/k8s/factors/overlays/<stage> | kubectl apply --dry-run=client -f -` passes
- [ ] Vault has all keys under `secret/factors/<stage>/*`
- [ ] External Secret has `SecretSyncedReady=True`
- [ ] Alembic migrations are at head
- [ ] Smoke tests pass against the target URL
- [ ] Grafana `factors-api / overview` dashboard shows steady traffic
- [ ] No firing alerts in AlertManager
