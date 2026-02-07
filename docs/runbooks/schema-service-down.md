# Schema Service Down

## Alert

**Alert Name:** `SchemaServiceDown`

**Severity:** Critical

**Threshold:** `absent(up{job="schema-service"} == 1) or sum(up{job="schema-service"}) == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Schema Compiler & Validator service (AGENT-FOUND-002) are running. The Schema Service is the centralized validation gateway responsible for:

1. **Validating climate data payloads** against JSON Schema (Draft 2020-12) with GreenLang extensions ($unit, $dimension, $rules, $aliases, $deprecated)
2. **Compiling schemas to Intermediate Representation (IR)** with property flattening, constraint indexing, regex safety analysis, and SHA-256 provenance hashing
3. **Caching compiled schema IRs** via an LRU cache with TTL-based expiration and background warm-up scheduling
4. **Serving the schema registry** backed by Git, providing versioned schema lookup and resolution
5. **Exposing REST API endpoints** for single validation (`POST /v1/schema/validate`), batch validation (`POST /v1/schema/validate/batch`), schema compilation (`POST /v1/schema/compile`), version listing, and schema retrieval
6. **Emitting Prometheus metrics** for validation throughput, error rates, cache hit/miss ratios, and latency percentiles
7. **Performing ReDoS safety analysis** on regex patterns in schemas to prevent denial-of-service via catastrophic backtracking

When the Schema Service is down, all upstream services that depend on payload validation will be unable to validate incoming data. This includes every GreenLang application pipeline -- CSRD, CBAM, VCCI, SB253, EUDR, and Taxonomy. Intake agents, validation agents, and any service that calls the schema validation API will receive errors or timeouts. Invalid data may pass through pipelines unchecked if upstream services lack fallback validation, creating compliance risk.

**Note:** Schema definitions stored in the Git-backed registry are not affected by a service outage. The registry data persists independently. Once the service recovers, all cached IRs will need to be recompiled (cache is in-memory and lost on restart), but the warm-up scheduler will automatically repopulate popular schemas.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | All data validation is blocked; users cannot submit payloads for processing |
| **Data Impact** | High | Unvalidated data may enter pipelines if upstream services lack fallback; data quality degradation |
| **SLA Impact** | Critical | Validation latency SLA violated (P95 targets: <25ms small, <150ms medium, <500ms large payloads) |
| **Revenue Impact** | High | Customer-facing compliance workflows cannot process new data submissions |
| **Compliance Impact** | Critical | Payloads accepted without validation may not meet CSRD, CBAM, or other regulatory data quality requirements |
| **Downstream Impact** | High | All GreenLang applications depend on schema validation; cascading failures across intake and validation agents |

---

## Symptoms

- `up{job="schema-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=schema-service`
- `glschema_validations_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Upstream agents report "schema service unavailable" or "validation timeout" errors in logs
- `glschema_cache_size` gauge drops to 0 (cache lost on restart)
- Grafana Schema Service dashboard shows "No Data" or stale timestamps
- Batch validation endpoint `POST /v1/schema/validate/batch` returns errors

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List schema service pods
kubectl get pods -n greenlang -l app=schema-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=schema-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to schema service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=schema-service | tail -30

# Check deployment status
kubectl describe deployment schema-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment schema-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=schema-service

# Check for rollout issues
kubectl rollout status deployment/schema-service -n greenlang

# Check HPA status
kubectl get hpa -n greenlang -l app=schema-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=schema-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for schema-specific errors
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "schema\|compile\|validate\|cache\|registry\|redis"

# Look for database and Redis connection errors
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "database\|postgres\|redis\|connection\|pool"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of schema service pods
kubectl top pods -n greenlang -l app=schema-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=schema-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes

# Check if resource quota is exhausted
kubectl describe resourcequota -n greenlang
```

### Step 5: Check Redis Connectivity

The schema service uses Redis for IR cache storage in production (in-memory LRU cache with optional Redis backend).

```bash
# Verify Redis connectivity
kubectl run redis-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-redis.redis.svc.cluster.local 6379'

# Check Redis key count for schema cache
kubectl run redis-check --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 KEYS 'gl:schema:ir:*' | head -20

# Check Redis memory usage
kubectl run redis-mem --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 INFO memory | head -10
```

### Step 6: Check Schema Registry (Git Backend)

```bash
# Check if the schema registry Git repository is accessible
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry status 2>/dev/null || echo "Git registry not accessible"

# Check registry volume mount
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Volumes"

# Check PVC for schema registry
kubectl get pvc -n greenlang | grep schema
```

### Step 7: Check ConfigMap and Secrets

```bash
# Verify the schema service ConfigMap exists and is valid
kubectl get configmap schema-service-config -n greenlang
kubectl get configmap schema-service-config -n greenlang -o yaml | head -50

# Verify secrets exist (API keys, Redis URL, registry credentials)
kubectl get secret schema-service-secrets -n greenlang

# Check ESO sync status
kubectl get externalsecrets -n greenlang | grep schema
```

### Step 8: Check Health Endpoint Directly

```bash
# Port-forward to the schema service (if at least one pod exists)
kubectl port-forward -n greenlang svc/schema-service 8080:8080

# Test the health endpoint
curl -s http://localhost:8080/health | python3 -m json.tool

# Test the metrics endpoint
curl -s http://localhost:8080/metrics | python3 -m json.tool
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137, `restartCount` is incrementing.

**Cause:** The schema service is consuming more memory than its configured limit. This typically occurs due to large numbers of cached schema IRs, extremely large schemas being compiled, or high-volume batch validation requests consuming memory.

**Resolution:**

1. **Confirm the OOM cause:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. **Check the current memory limit and usage pattern:**

```bash
# Current memory limits
kubectl get deployment schema-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].resources}'

# Check Prometheus for memory trend before OOM
# PromQL: container_memory_working_set_bytes{namespace="greenlang", pod=~"schema-service.*"}
```

3. **Immediate mitigation -- increase memory limits:**

```bash
kubectl patch deployment schema-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "schema-service",
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

4. **If caused by excessive cache size, reduce the IR cache max memory:**

```bash
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_IR_CACHE_MAX_MEMORY_MB=128 GL_SCHEMA_CACHE_MAX_SIZE=500
```

5. **Verify pods restart successfully:**

```bash
kubectl rollout status deployment/schema-service -n greenlang
kubectl get pods -n greenlang -l app=schema-service
```

### Scenario 2: CrashLoopBackOff (Application Error)

**Symptoms:** Pod status shows CrashLoopBackOff, container exits with non-zero code (not 137).

**Cause:** Application startup failure due to configuration error, missing secrets, Git registry initialization failure, or code bug.

**Resolution:**

1. **Check the crash reason:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl logs -n greenlang <pod-name> --previous --tail=100
```

2. **If caused by missing configuration:**

```bash
# Check ConfigMap values
kubectl get configmap schema-service-config -n greenlang -o yaml

# Look for config parsing errors in logs
kubectl logs -n greenlang <pod-name> --previous | grep -i "config\|env\|validation\|parse"
```

3. **If caused by missing secrets:**

```bash
# Verify secrets exist
kubectl get secret schema-service-secrets -n greenlang

# Check ESO sync status
kubectl describe externalsecret schema-service-secrets -n greenlang

# If secrets are missing, check SSM parameters
aws ssm get-parameters-by-path --path "/gl/prod/schema-service/" --query "Parameters[*].Name"
```

4. **If caused by Git registry initialization failure:**

```bash
# Check if the Git registry PVC exists and is bound
kubectl get pvc -n greenlang | grep schema-registry

# Check if the Git repository is valid
kubectl run git-check --rm -it --image=alpine/git:latest -n greenlang --restart=Never -- \
  git -C /data/schema-registry log --oneline -5
```

5. **Restart the deployment after fixing:**

```bash
kubectl rollout restart deployment/schema-service -n greenlang
kubectl rollout status deployment/schema-service -n greenlang
```

### Scenario 3: ImagePullBackOff

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
kubectl get deployment schema-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'

# Verify the image exists
aws ecr describe-images --repository-name greenlang/schema-service --image-ids imageTag=latest
```

3. **Fix image tag if needed:**

```bash
kubectl set image deployment/schema-service \
  schema-service=greenlang/schema-service:<correct-tag> -n greenlang
```

### Scenario 4: Liveness Probe Failure

**Symptoms:** Pod is running but not Ready, restarting due to liveness probe failure, health endpoint returning errors.

**Cause:** The `/health` endpoint is not responding within the probe timeout. This can be caused by the async event loop being blocked, a long-running schema compilation monopolizing the CPU, or Redis connectivity issues degrading the health check.

**Resolution:**

1. **Check liveness probe configuration and failure events:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Liveness"
kubectl get events -n greenlang --field-selector reason=Unhealthy --sort-by='.lastTimestamp'
```

2. **Test the health endpoint manually:**

```bash
kubectl port-forward -n greenlang svc/schema-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

3. **Check for blocked compilation operations:**

```bash
# Look for long-running or stuck compilation
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "compile\|timeout\|blocked\|slow\|regex\|redos"
```

4. **Restart the deployment:**

```bash
kubectl rollout restart deployment/schema-service -n greenlang
kubectl rollout status deployment/schema-service -n greenlang
```

### Scenario 5: Redis Connectivity Failure

**Symptoms:** Logs show "connection refused" to Redis, cache operations failing, all validations performing full compilation (no cache hits).

**Cause:** Redis cluster is down or network policy is blocking traffic.

**Resolution:**

1. **Verify Redis is running:**

```bash
kubectl get pods -n redis -l app=redis
kubectl get svc -n redis | grep redis
```

2. **Test connectivity:**

```bash
kubectl run redis-test --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 PING
```

3. **Note:** The schema service uses an in-memory LRU cache as the primary cache layer. Redis is optional. If Redis is unavailable, the service should degrade gracefully to in-memory-only caching. If the service is crashing instead of degrading, this indicates a bug. Restart and monitor:

```bash
kubectl rollout restart deployment/schema-service -n greenlang
```

### Scenario 6: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns, no infrastructure issues found.

**Cause:** A code regression introduced in the latest deployment.

**Resolution:**

1. **Check recent deployment history:**

```bash
kubectl rollout history deployment/schema-service -n greenlang
```

2. **Rollback to the previous version:**

```bash
kubectl rollout undo deployment/schema-service -n greenlang
kubectl rollout status deployment/schema-service -n greenlang
```

3. **Verify the rollback resolved the issue:**

```bash
kubectl get pods -n greenlang -l app=schema-service
curl -s http://localhost:8080/health | python3 -m json.tool
```

4. **Create a bug report** with the logs from the failed deployment for the development team.

---

## Post-Incident Steps

After the schema service is restored, perform the following verification and cleanup steps.

### Step 1: Verify Service Health

```bash
# 1. Check all pods are running and ready
kubectl get pods -n greenlang -l app=schema-service

# 2. Check the health endpoint
kubectl port-forward -n greenlang svc/schema-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# 3. Verify the schema service is being scraped
up{job="schema-service"} == 1

# 4. Verify validation metrics are incrementing (if requests are coming in)
increase(glschema_validations_total[5m])

# 5. Verify cache is being populated
glschema_cache_size > 0
```

### Step 3: Verify IR Cache is Warming Up

After a restart, the IR cache is empty. The warm-up scheduler should automatically repopulate popular schemas.

```bash
# Check cache size via metrics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Cache size: {data.get('cache_size', 0)}\")
print(f\"Cache hits: {data.get('cache_hits', 0)}\")
print(f\"Cache misses: {data.get('cache_misses', 0)}\")
"
```

### Step 4: Verify Upstream Services Have Recovered

```bash
# Check orchestrator and agent logs for schema validation errors
kubectl logs -n greenlang -l app=orchestrator-service --tail=100 \
  | grep -i "schema\|validation\|validate"

# Check intake agents for recovery
kubectl logs -n greenlang -l app.kubernetes.io/component=intake-agent --tail=100 \
  | grep -i "schema\|validation"
```

### Step 5: Review Metrics for Root Cause

```promql
# Check for memory spikes before the outage
container_memory_working_set_bytes{namespace="greenlang", pod=~"schema-service.*"}

# Check for CPU throttling
sum(rate(container_cpu_cfs_throttled_seconds_total{namespace="greenlang", pod=~"schema-service.*"}[5m]))

# Check validation throughput pattern (was there a spike?)
rate(glschema_validations_total[5m])

# Check cache hit rate trend (low hit rate = high compilation load)
glschema_cache_hits / (glschema_cache_hits + glschema_cache_misses)
```

---

## Interim Mitigation

While the Schema Service is being restored:

1. **Cached IRs are lost.** The in-memory IR cache is cleared on restart. The warm-up scheduler will repopulate popular schemas within minutes of recovery. The first few validation requests after restart will be slower due to cache misses.

2. **No new validations can complete.** Applications that depend on the schema service will receive errors. Upstream applications should implement retry logic with exponential backoff.

3. **Schema registry data is safe.** The Git-backed schema registry persists independently of the service process. All schema versions and definitions are intact.

4. **Manual validation is not possible.** The schema service is the only validation path. There is no manual fallback for payload validation.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#data-ops` -- data pipeline impact
   - `#platform-oncall` -- engineering response

6. **Monitor the outage duration.** If the schema service is down for more than 15 minutes, escalate per the escalation path below. Data quality may be compromised for any payloads accepted without validation during the outage.

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Schema service down, no active data pipelines impacted | On-call engineer | Immediate (<5 min) |
| L2 | Schema service down > 15 minutes, data pipelines blocked | Platform team lead + #schema-oncall | 15 minutes |
| L3 | Schema service down > 30 minutes, regulatory deadline at risk | Platform team + compliance team + CTO notification | Immediate |
| L4 | Schema service down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Schema Service Health (`/d/schema-service-health`)
- **Dashboard:** Schema Validation Overview (`/d/schema-validation-overview`)
- **Alert:** `SchemaServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="schema-service"}` (should always be >= 2)
  - `glschema_validations_total` rate (should be non-zero during business hours)
  - `glschema_validations_failed` rate (should be low relative to total)
  - `glschema_cache_size` (should stabilize after warm-up; drop to 0 indicates restart)
  - `glschema_cache_hits / (glschema_cache_hits + glschema_cache_misses)` (target >80%)
  - `glschema_validation_time_p95_ms` (should stay below SLA targets)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales based on validation throughput** and CPU utilization
4. **IR cache is sized for all active schemas** (default: 1000 entries, 256 MB)
5. **Batch validation limits** are enforced (default: 1000 items per batch, 10 MB total)

### Configuration Best Practices

- Set `GL_SCHEMA_CACHE_MAX_SIZE` based on the number of active schemas in your registry
- Set `GL_SCHEMA_IR_CACHE_MAX_MEMORY_MB` to at most 50% of the container memory limit
- Set `GL_SCHEMA_MAX_BATCH_ITEMS` conservatively; start at 1000 and adjust based on load testing
- Use ESO for secrets rotation (API keys, Redis URL, registry credentials)
- Test configuration changes in staging before production

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `SchemaServiceDown` | Critical | This alert -- no schema service pods running |
| `SchemaHighValidationErrorRate` | Warning | >25% validation errors over 5 minutes |
| `SchemaCompilationFailure` | Warning | Schema compilation errors occurring |
| `SchemaCacheHitRateLow` | Warning | Cache hit rate <50% over 10 minutes |
| `SchemaHighLatency` | Warning | P95 validation latency exceeds SLA targets |

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any P1 schema service incident
- **Related alerts:** `SchemaHighValidationErrorRate`, `SchemaCompilationFailure`, `SchemaCacheHitRateLow`
- **Related dashboards:** Schema Service Health, Schema Validation Overview
- **Related runbooks:** [High Validation Errors](./high-validation-errors.md), [Schema Cache Corruption](./schema-cache-corruption.md), [Compilation Timeout](./compilation-timeout.md)
