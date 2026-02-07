# Normalizer Service Down

## Alert

**Alert Name:** `NormalizerServiceDown`

**Severity:** Critical

**Threshold:** `absent(up{job="normalizer-service"} == 1) or sum(up{job="normalizer-service"}) == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Unit & Reference Normalizer service (AGENT-FOUND-003) are running. The Normalizer Service is the centralized unit conversion and entity resolution gateway responsible for:

1. **Unit conversion across 10 physical dimensions** (mass, energy, volume, area, distance, emissions, currency, time, temperature, pressure) using Decimal-precision arithmetic with deterministic rounding (ROUND_HALF_UP)
2. **GHG conversion with Global Warming Potentials** (IPCC AR4, AR5, AR6) converting between greenhouse gases (CO2, CH4, N2O) and CO2-equivalent values
3. **Entity resolution for fuels, materials, and processes** using vocabulary lookup tables with fuzzy matching fallback and confidence scoring
4. **Dimensional analysis** to prevent invalid cross-dimension conversions (e.g., blocking kg-to-kWh)
5. **Provenance tracking** with SHA-256 hash chains for every conversion operation, ensuring complete audit trails
6. **Currency conversion** with date-specific exchange rates and triangulation through USD
7. **Tenant-specific custom conversion factors** and vocabulary mappings
8. **Exposing REST API endpoints** for unit conversion (`POST /v1/normalizer/convert`), GHG conversion (`POST /v1/normalizer/ghg`), entity resolution (`POST /v1/normalizer/resolve`), batch operations (`POST /v1/normalizer/batch`), and vocabulary management
9. **Emitting Prometheus metrics** for conversion throughput, entity resolution confidence, cache hit rates, vocabulary size, and latency percentiles

When the Normalizer Service is down, all upstream services that depend on unit conversion or entity resolution will be unable to process data. This includes every GreenLang application pipeline -- CSRD, CBAM, VCCI, SB253, EUDR, and Taxonomy. Emissions calculation agents, intake agents, reporting agents, and any service that calls the normalizer API will receive errors or timeouts. Downstream calculations that depend on normalized values will stall, and compliance reports cannot be generated.

**Note:** Vocabulary tables and conversion factor definitions are stored in PostgreSQL and are not affected by a service outage. Once the service recovers, vocabulary caches will need to be repopulated from the database, but the warm-up scheduler will automatically reload popular vocabularies.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | All unit conversions blocked; users cannot submit or process emissions data |
| **Data Impact** | High | Raw data cannot be normalized; calculations using unnormalized values would produce incorrect results |
| **SLA Impact** | Critical | Conversion latency SLA violated (P95 targets: <10ms single, <50ms batch, <100ms entity resolution) |
| **Revenue Impact** | High | Customer-facing compliance workflows cannot produce normalized calculations or reports |
| **Compliance Impact** | Critical | Emissions reports cannot be generated without unit normalization; regulatory submission deadlines at risk |
| **Downstream Impact** | Critical | All 6 GreenLang application pipelines (CSRD, CBAM, VCCI, SB253, EUDR, Taxonomy) depend on normalizer; cascading failures across calculation and reporting agents |

---

## Symptoms

- `up{job="normalizer-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=normalizer-service`
- `glnorm_conversions_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Upstream agents report "normalizer service unavailable" or "conversion timeout" errors in logs
- `glnorm_vocabulary_size` gauge drops to 0 (cache lost on restart)
- Grafana Normalizer Service dashboard shows "No Data" or stale timestamps
- Batch conversion endpoint `POST /v1/normalizer/batch` returns errors
- Emissions calculation agents log "unable to convert units" or "normalizer connection refused"

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List normalizer service pods
kubectl get pods -n greenlang -l app=normalizer-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=normalizer-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to normalizer service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=normalizer-service | tail -30

# Check deployment status
kubectl describe deployment normalizer-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment normalizer-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=normalizer-service

# Check for rollout issues
kubectl rollout status deployment/normalizer-service -n greenlang

# Check HPA status
kubectl get hpa -n greenlang -l app=normalizer-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=normalizer-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=normalizer-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for normalizer-specific errors
kubectl logs -n greenlang -l app=normalizer-service --tail=500 \
  | grep -i "vocabulary\|convert\|normalize\|dimension\|gwp\|provenance"

# Look for database and Redis connection errors
kubectl logs -n greenlang -l app=normalizer-service --tail=500 \
  | grep -i "database\|postgres\|redis\|connection\|pool"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of normalizer service pods
kubectl top pods -n greenlang -l app=normalizer-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=normalizer-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes

# Check if resource quota is exhausted
kubectl describe resourcequota -n greenlang
```

### Step 5: Check Database Connectivity

The normalizer service uses PostgreSQL for vocabulary tables, conversion factor storage, and provenance records.

```bash
# Verify PostgreSQL connectivity from the pod network
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check database connection pool status in logs
kubectl logs -n greenlang -l app=normalizer-service --tail=200 \
  | grep -i "pool\|connection\|database\|postgres"

# Check if the normalizer database tables exist
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE 'normalizer_%';"
```

### Step 6: Check Vocabulary Loading

```bash
# Check if vocabulary loading succeeded during startup
kubectl logs -n greenlang -l app=normalizer-service --tail=500 \
  | grep -i "vocabulary\|vocab\|loaded\|loading\|fuel\|material\|unit"

# Check vocabulary table row counts in the database
kubectl run pg-vocab --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT 'fuel_vocabulary' as table_name, count(*) FROM normalizer_fuel_vocabulary
   UNION ALL SELECT 'material_vocabulary', count(*) FROM normalizer_material_vocabulary
   UNION ALL SELECT 'unit_definitions', count(*) FROM normalizer_unit_definitions
   UNION ALL SELECT 'custom_factors', count(*) FROM normalizer_custom_factors;"
```

### Step 7: Check Redis Cache Connectivity

```bash
# Verify Redis connectivity (used for conversion result caching)
kubectl run redis-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-redis.redis.svc.cluster.local 6379'

# Check Redis key count for normalizer cache
kubectl run redis-check --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 KEYS 'gl:norm:*' | head -20

# Check Redis memory usage
kubectl run redis-mem --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 INFO memory | head -10
```

### Step 8: Check ConfigMap and Secrets

```bash
# Verify the normalizer service ConfigMap exists and is valid
kubectl get configmap normalizer-service-config -n greenlang
kubectl get configmap normalizer-service-config -n greenlang -o yaml | head -50

# Verify secrets exist (DB credentials, API keys)
kubectl get secret normalizer-service-secrets -n greenlang

# Check ESO sync status
kubectl get externalsecrets -n greenlang | grep normalizer
```

### Step 9: Check Health Endpoint Directly

```bash
# Port-forward to the normalizer service (if at least one pod exists)
kubectl port-forward -n greenlang svc/normalizer-service 8080:8080

# Test the health endpoint
curl -s http://localhost:8080/health | python3 -m json.tool

# Test the metrics endpoint
curl -s http://localhost:8080/metrics | python3 -m json.tool
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137, `restartCount` is incrementing.

**Cause:** The normalizer service is consuming more memory than its configured limit. This typically occurs due to large vocabulary tables loaded into memory, oversized batch conversion requests, or excessive in-memory cache growth from high-cardinality conversion factor lookups.

**Resolution:**

1. **Confirm the OOM cause:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. **Check the current memory limit and usage pattern:**

```bash
# Current memory limits
kubectl get deployment normalizer-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].resources}'

# Check Prometheus for memory trend before OOM
# PromQL: container_memory_working_set_bytes{namespace="greenlang", pod=~"normalizer-service.*"}
```

3. **Immediate mitigation -- increase memory limits:**

```bash
kubectl patch deployment normalizer-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "normalizer-service",
            "resources": {
              "limits": {
                "cpu": "2",
                "memory": "2Gi"
              },
              "requests": {
                "cpu": "500m",
                "memory": "1Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

4. **If caused by excessive vocabulary cache size, reduce cache limits:**

```bash
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_VOCABULARY_CACHE_MAX_MB=128 \
  GL_NORM_CONVERSION_CACHE_MAX_ENTRIES=10000
```

5. **Verify pods restart successfully:**

```bash
kubectl rollout status deployment/normalizer-service -n greenlang
kubectl get pods -n greenlang -l app=normalizer-service
```

### Scenario 2: CrashLoopBackOff -- Vocabulary Loading Failure

**Symptoms:** Pod status shows CrashLoopBackOff, logs show "vocabulary loading failed" or "unable to initialize conversion tables".

**Cause:** The service cannot load its vocabulary tables from PostgreSQL during startup. This blocks initialization because the normalizer requires at least the base unit definitions to function.

**Resolution:**

1. **Check the crash reason:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl logs -n greenlang <pod-name> --previous --tail=100
```

2. **Verify database connectivity and vocabulary tables:**

```bash
# Check if PostgreSQL is reachable
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Verify vocabulary tables exist and have data
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name, (SELECT count(*) FROM information_schema.columns WHERE table_name = t.table_name) as columns
   FROM information_schema.tables t WHERE table_schema='public' AND table_name LIKE 'normalizer_%';"
```

3. **If vocabulary tables are missing, run database migrations:**

```bash
# Check current migration version
kubectl run pg-migration --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description FROM flyway_schema_history ORDER BY installed_rank DESC LIMIT 5;"

# Apply missing migrations (V023 for normalizer service)
kubectl run flyway-migrate --rm -it \
  --image=greenlang/flyway-migrations:latest -n greenlang --restart=Never -- \
  flyway migrate
```

4. **Restart the deployment after fixing:**

```bash
kubectl rollout restart deployment/normalizer-service -n greenlang
kubectl rollout status deployment/normalizer-service -n greenlang
```

### Scenario 3: CrashLoopBackOff -- Configuration Error

**Symptoms:** Pod status shows CrashLoopBackOff, logs show "configuration error", "missing required environment variable", or "invalid GWP source".

**Cause:** Application startup failure due to missing or invalid configuration -- typically missing database URL, invalid GWP version specification, or incorrect vocabulary source path.

**Resolution:**

1. **Check configuration-related errors:**

```bash
kubectl logs -n greenlang <pod-name> --previous | grep -i "config\|env\|validation\|parse\|gwp\|dimension"
```

2. **Verify ConfigMap values:**

```bash
kubectl get configmap normalizer-service-config -n greenlang -o yaml
```

3. **Key configuration variables to verify:**

```bash
# Essential environment variables
kubectl get deployment normalizer-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

Required variables:
- `GL_NORM_DATABASE_URL` -- PostgreSQL connection string
- `GL_NORM_REDIS_URL` -- Redis connection string (optional, degrades gracefully)
- `GL_NORM_GWP_DEFAULT_SOURCE` -- Default GWP table (AR4, AR5, AR6)
- `GL_NORM_DEFAULT_PRECISION` -- Default decimal precision (typically 6)
- `GL_NORM_STRICT_MODE` -- Whether to enforce strict dimensional analysis

4. **Fix the configuration and restart:**

```bash
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_GWP_DEFAULT_SOURCE=AR6 \
  GL_NORM_DEFAULT_PRECISION=6 \
  GL_NORM_STRICT_MODE=true

kubectl rollout restart deployment/normalizer-service -n greenlang
```

### Scenario 4: Database Connection Pool Exhaustion

**Symptoms:** Service was running but health check fails, logs show "connection pool exhausted", "too many connections", or "timeout waiting for available connection".

**Cause:** All database connections in the pool are in use. This can happen during high-volume batch operations or when a downstream database issue causes connections to hang.

**Resolution:**

1. **Check database connection pool status:**

```bash
kubectl logs -n greenlang -l app=normalizer-service --tail=500 \
  | grep -i "pool\|connection\|exhausted\|timeout\|waiting"
```

2. **Check active connections in PostgreSQL:**

```bash
kubectl run pg-conns --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT count(*), state, application_name FROM pg_stat_activity
   WHERE application_name LIKE '%normalizer%'
   GROUP BY state, application_name;"
```

3. **Increase the connection pool size:**

```bash
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_DB_POOL_MIN=5 \
  GL_NORM_DB_POOL_MAX=20 \
  GL_NORM_DB_POOL_TIMEOUT=30

kubectl rollout restart deployment/normalizer-service -n greenlang
```

4. **If connections are stuck, restart the service:**

```bash
kubectl rollout restart deployment/normalizer-service -n greenlang
kubectl rollout status deployment/normalizer-service -n greenlang
```

### Scenario 5: ImagePullBackOff

**Symptoms:** Pod status shows ImagePullBackOff or ErrImagePull.

**Cause:** Container image not found, registry authentication failure, or tag mismatch.

**Resolution:**

1. **Check the image and pull errors:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A5 "Events"
```

2. **Verify image exists in registry:**

```bash
# Check current image tag
kubectl get deployment normalizer-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'

# Verify the image exists
aws ecr describe-images --repository-name greenlang/normalizer-service --image-ids imageTag=latest
```

3. **Fix image tag if needed:**

```bash
kubectl set image deployment/normalizer-service \
  normalizer-service=greenlang/normalizer-service:<correct-tag> -n greenlang
```

### Scenario 6: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns, no infrastructure issues found.

**Cause:** A code regression introduced in the latest deployment.

**Resolution:**

1. **Check recent deployment history:**

```bash
kubectl rollout history deployment/normalizer-service -n greenlang
```

2. **Rollback to the previous version:**

```bash
kubectl rollout undo deployment/normalizer-service -n greenlang
kubectl rollout status deployment/normalizer-service -n greenlang
```

3. **Verify the rollback resolved the issue:**

```bash
kubectl get pods -n greenlang -l app=normalizer-service
kubectl port-forward -n greenlang svc/normalizer-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

4. **Create a bug report** with the logs from the failed deployment for the development team.

---

## Post-Incident Steps

After the normalizer service is restored, perform the following verification and cleanup steps.

### Step 1: Verify Service Health

```bash
# 1. Check all pods are running and ready
kubectl get pods -n greenlang -l app=normalizer-service

# 2. Check the health endpoint
kubectl port-forward -n greenlang svc/normalizer-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# 3. Verify the normalizer service is being scraped
up{job="normalizer-service"} == 1

# 4. Verify conversion metrics are incrementing (if requests are coming in)
increase(glnorm_conversions_total[5m])

# 5. Verify vocabulary cache is populated
glnorm_vocabulary_size > 0
```

### Step 3: Verify Vocabulary Cache is Loaded

After a restart, the vocabulary cache is repopulated from PostgreSQL. The warm-up scheduler should automatically reload popular vocabularies.

```bash
# Check vocabulary cache status via metrics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Vocabulary size: {data.get('vocabulary_size', 0)}\")
print(f\"Fuel entries: {data.get('fuel_vocab_size', 0)}\")
print(f\"Material entries: {data.get('material_vocab_size', 0)}\")
print(f\"Unit definitions: {data.get('unit_definitions_count', 0)}\")
print(f\"Cache hit rate: {data.get('cache_hit_rate', 0):.1f}%\")
"
```

### Step 4: Verify Upstream Services Have Recovered

```bash
# Check emissions calculation agents for normalizer errors
kubectl logs -n greenlang -l app.kubernetes.io/component=calculation-agent --tail=100 \
  | grep -i "normalizer\|convert\|unit\|dimension"

# Check intake agents for recovery
kubectl logs -n greenlang -l app.kubernetes.io/component=intake-agent --tail=100 \
  | grep -i "normalizer\|normalize\|unit"
```

### Step 5: Run a Smoke Test Conversion

```bash
# Test a basic unit conversion
curl -X POST http://localhost:8080/v1/normalizer/convert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "value": 1000,
    "from_unit": "kg",
    "to_unit": "tonnes",
    "precision": 6
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert data.get('converted_value') == 1.0, f'Unexpected result: {data}'
print(f'Conversion OK: 1000 kg = {data[\"converted_value\"]} tonnes')
print(f'Provenance hash: {data.get(\"provenance_hash\", \"N/A\")[:16]}...')
"

# Test a GHG conversion
curl -X POST http://localhost:8080/v1/normalizer/ghg \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "value": 1000,
    "from_unit": "kgCH4",
    "to_unit": "tCO2e",
    "gwp_source": "AR6"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'GHG conversion OK: 1000 kgCH4 = {data[\"converted_value\"]} tCO2e (GWP={data[\"gwp_applied\"]})')
"
```

### Step 6: Review Metrics for Root Cause

```promql
# Check for memory spikes before the outage
container_memory_working_set_bytes{namespace="greenlang", pod=~"normalizer-service.*"}

# Check for CPU throttling
sum(rate(container_cpu_cfs_throttled_seconds_total{namespace="greenlang", pod=~"normalizer-service.*"}[5m]))

# Check conversion throughput pattern (was there a spike?)
rate(glnorm_conversions_total[5m])

# Check vocabulary cache size trend (drop to 0 indicates restart)
glnorm_vocabulary_size

# Check database connection pool utilization
glnorm_db_pool_active / glnorm_db_pool_max
```

---

## Interim Mitigation

While the Normalizer Service is being restored:

1. **Vocabulary data is safe.** Vocabularies and conversion factors are stored in PostgreSQL, which persists independently. All definitions are intact.

2. **No new conversions can complete.** Applications that depend on the normalizer will receive errors. Upstream applications should implement retry logic with exponential backoff.

3. **Provenance chain is preserved.** All previously generated provenance hashes are stored in the database and are unaffected by the outage. New provenance records will resume once the service recovers.

4. **Manual conversion is not recommended.** While manual unit conversions are mathematically possible, they bypass the provenance tracking system and create audit trail gaps. Only use manual conversion as a last resort for critical regulatory deadlines.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#data-ops` -- data pipeline impact
   - `#platform-oncall` -- engineering response
   - `#compliance-ops` -- regulatory deadline impact

6. **Monitor the outage duration.** If the normalizer service is down for more than 15 minutes, escalate per the escalation path below. Emissions calculations and compliance reports cannot proceed without unit normalization.

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Normalizer service down, no active data pipelines impacted | On-call engineer | Immediate (<5 min) |
| L2 | Normalizer service down > 15 minutes, data pipelines blocked | Platform team lead + #normalizer-oncall | 15 minutes |
| L3 | Normalizer service down > 30 minutes, regulatory deadline at risk | Platform team + compliance team + CTO notification | Immediate |
| L4 | Normalizer service down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Normalizer Service Health (`/d/normalizer-service-health`)
- **Dashboard:** Unit Conversion Overview (`/d/normalizer-conversion-overview`)
- **Alert:** `NormalizerServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="normalizer-service"}` (should always be >= 2)
  - `glnorm_conversions_total` rate (should be non-zero during business hours)
  - `glnorm_conversions_failed` rate (should be low relative to total)
  - `glnorm_vocabulary_size` (should be stable; drop to 0 indicates restart)
  - `glnorm_entity_resolution_avg_confidence` (should be >0.8 in steady state)
  - `glnorm_conversion_latency_p95_ms` (should stay below SLA targets)
  - `glnorm_db_pool_active / glnorm_db_pool_max` (should stay below 80%)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales based on conversion throughput** and CPU utilization
4. **Vocabulary cache is sized for all active vocabularies** (default: 256 MB)
5. **Database connection pool** is sized for expected concurrency (default: min 5, max 20)
6. **Batch operation limits** are enforced (default: 10,000 items per batch)

### Configuration Best Practices

- Set `GL_NORM_VOCABULARY_CACHE_MAX_MB` to at most 40% of the container memory limit
- Set `GL_NORM_DB_POOL_MAX` based on expected concurrency (2x expected peak concurrent requests)
- Set `GL_NORM_BATCH_MAX_ITEMS` conservatively; start at 10,000 and adjust based on load testing
- Set `GL_NORM_GWP_DEFAULT_SOURCE` explicitly (do not rely on defaults) and pin it per reporting period
- Use ESO for secrets rotation (DB credentials, API keys)
- Test configuration changes in staging before production

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `NormalizerServiceDown` | Critical | This alert -- no normalizer service pods running |
| `NormalizerConversionAccuracyDrift` | Warning | Conversion factor discrepancies detected |
| `NormalizerEntityResolutionLowConfidence` | Warning | Average entity resolution confidence < 0.7 |
| `NormalizerHighLatency` | Warning | P95 conversion latency exceeds SLA targets |
| `NormalizerHighErrorRate` | Warning | >10% conversion errors over 5 minutes |

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any P1 normalizer service incident
- **Related alerts:** `NormalizerConversionAccuracyDrift`, `NormalizerEntityResolutionLowConfidence`, `NormalizerHighLatency`
- **Related dashboards:** Normalizer Service Health, Unit Conversion Overview
- **Related runbooks:** [Conversion Accuracy Drift](./conversion-accuracy-drift.md), [Entity Resolution Low Confidence](./entity-resolution-low-confidence.md), [Normalizer High Latency](./normalizer-high-latency.md)
