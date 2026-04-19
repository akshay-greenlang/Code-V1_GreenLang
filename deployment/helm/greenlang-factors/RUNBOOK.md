# greenlang-factors — Operator Runbook

## Deploy

```bash
# dev
helm upgrade --install greenlang-factors deployment/helm/greenlang-factors \
  -f deployment/helm/greenlang-factors/values-dev.yaml \
  --namespace greenlang-factors --create-namespace

# staging (prod secrets via ExternalSecrets)
helm upgrade --install greenlang-factors deployment/helm/greenlang-factors \
  -f deployment/helm/greenlang-factors/values-staging.yaml \
  --namespace greenlang-factors --create-namespace

# prod
helm upgrade --install greenlang-factors deployment/helm/greenlang-factors \
  -f deployment/helm/greenlang-factors/values-prod.yaml \
  --namespace greenlang-factors
```

Hooks fire in this order:
1. `pre-install`/`pre-upgrade`: Flyway migration Job (`migration-job.yaml`)
2. Core resources applied
3. `post-install`/`post-upgrade`: smoke test Job (`smoke-test-job.yaml`)

If the smoke Job fails, Helm rolls back automatically.

## Rollback

```bash
helm rollback greenlang-factors <revision>
kubectl -n greenlang-factors rollout status deploy greenlang-factors-api
```

`helm history greenlang-factors` lists revisions.

## Tier limit tuning

Tier request-per-day limits live in `values*.yaml` under `ingress.rateLimits`.
Applied at Kong via `KongPlugin/rate-limiting-advanced`. Change + `helm upgrade`
causes Kong to pick up new config in ~30s (no pod restarts).

## Secret rotation

ExternalSecrets refresh on `refreshInterval` (default 1h) or immediately via:

```bash
kubectl annotate externalsecret greenlang-factors-secrets \
  force-sync=$(date +%s) --overwrite
```

Vault paths (default):
- `greenlang/factors/database_url`
- `greenlang/factors/redis_url`
- `greenlang/factors/jwt_signing_key`
- `greenlang/factors/s3_access_key`
- `greenlang/factors/s3_secret_key`

## Embedding re-backfill

After a bulk source ingest, embeddings need regeneration:

```bash
# Scale up embed workers
kubectl -n greenlang-factors scale deploy \
  greenlang-factors-worker-embed --replicas=8

# Or temporarily via values:
helm upgrade greenlang-factors ... --set workers.embed.replicaCount=8

# Monitor progress
kubectl -n greenlang-factors logs -l app.kubernetes.io/component=worker-embed -f
```

## SLO alerts (PrometheusRule)

See `templates/prometheusrule.yaml`. Alertmanager routes `severity=page` to
PagerDuty (OBS-004). On page:

| Alert | Likely Cause | Immediate action |
|-------|--------------|------------------|
| FactorsHighErrorRate | PG pool exhausted, catalog mismatch | Check `factors_db_connections`, roll pods |
| FactorsSearchLatencyP99 | HNSW rebuild, cache miss storm | Check `factors_cache_hit_rate`, tune `ef_search` |
| FactorsTier429Spike | DDoS or legit burst | Coordinate with customer success; check Kong logs |
| FactorsCacheHitRateLow | Redis outage or cache stampede | Verify Redis pods, warm cache |

## Incident paging

- OBS-004 Alertmanager -> PagerDuty service `greenlang-factors-oncall`
- Logs: Loki `{app="greenlang-factors"}` (OBS-009 stack)
- Traces: Jaeger `greenlang-factors` service (OBS-003)
- Dashboards: Grafana folder "Factors"

## Disaster recovery

- Database: V426-V438 migrations in `deployment/database/migrations/sql/`.
  Backup cadence per INFRA-002 (PITR enabled, 7-day RPO 15s).
- S3 ingestion artifacts: bucket versioning on, cross-region replication
  to backup region.
- Embeddings are reproducible from the catalog + model weights; backups not
  strictly required (regenerate via `WORKER_MODE=embed` workers).

## Capacity planning (prod)

| Metric | Target | Alert threshold |
|--------|--------|-----------------|
| API replicas | 3-10 (HPA) | <2 sustained 5m |
| Search P99 latency | <200ms | >500ms 10m |
| Cache hit rate | >90% | <80% 30m |
| 429 rate | <0.1% | >2% 5m |
| DB connections | <60% pool | >80% 10m |
