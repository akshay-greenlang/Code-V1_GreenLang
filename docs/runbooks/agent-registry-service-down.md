# Agent Registry & Service Catalog Service Down

## Alert

**Alert Name:** `AgentRegistryServiceDown`

**Severity:** Critical

**Threshold:** `up{job="agent-registry-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Agent Registry & Service Catalog (AGENT-FOUND-007) are running. The Agent Registry & Service Catalog is the centralized agent metadata management and discovery service for all GreenLang Climate OS agents. It is responsible for:

1. **Agent registration and metadata management** -- Every agent in the GreenLang platform is registered with its metadata including execution layer, execution mode, idempotency, determinism, GLIP version, and checkpointing support
2. **Version tracking** -- Each agent can have multiple versions with resource profiles, container specifications, legacy HTTP configurations, tags, sectors, and provenance hashes
3. **Capability definitions** -- Agent capabilities are registered with input/output types, categories, and parameter constraints, enabling capability-based agent discovery
4. **Dependency graph management** -- Agent-to-agent dependencies are tracked with version constraints, optional flags, and reasons, enabling the orchestrator to construct valid DAGs
5. **Dependency cycle detection** -- The registry detects circular dependencies during registration and resolution to prevent infinite loops in DAG construction
6. **Health monitoring** -- Periodic health checks (liveness, readiness, startup, deep) are performed on registered agents, with results stored in a TimescaleDB hypertable
7. **Service catalog publishing** -- A public-facing directory of available agents with display names, summaries, categories, and publication status
8. **Hot reload** -- Registry data is periodically refreshed from the database and cached in Redis for fast lookups
9. **Registry audit trail** -- All registry operations (registrations, version publishes, capability changes, dependency modifications) are logged in a TimescaleDB hypertable
10. **Emitting Prometheus metrics** (12+ metrics under the `gl_agent_registry_*` prefix) for monitoring agent counts, health status, query performance, dependency resolution, cache performance, and service health

When the Agent Registry is down:
- **The orchestrator cannot discover agents** and will be unable to construct or execute DAGs
- **Dependency resolution fails** and new pipeline submissions will be rejected
- **Health monitoring stops** and degraded agents will not be detected
- **Service catalog queries return errors** and agent discovery UIs will be unavailable

**Note:** Agent registrations, versions, capabilities, dependencies, health check data, and audit events are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full registry will be immediately available. No data is lost during an outage.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Agent discovery unavailable; new pipeline submissions may be rejected |
| **Data Impact** | Medium | No new health checks recorded; audit trail gap during outage |
| **SLA Impact** | High | Registry query latency SLA violated (P99 targets: <50ms single lookup) |
| **Revenue Impact** | High | DAG execution blocked if orchestrator cannot resolve agent fleet |
| **Compliance Impact** | Medium | Registry audit trail requirements violated during outage |
| **Downstream Impact** | Critical | Orchestrator (GL-FOUND-X-001) depends on registry for agent discovery; all DAG pipelines affected |

---

## Symptoms

- `up{job="agent-registry-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=agent-registry-service`
- `gl_agent_registry_agents_total` metric is 0 or absent
- `gl_agent_registry_queries_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Orchestrator logs show "registry unavailable" or "agent discovery timeout" errors
- Grafana Agent Registry dashboard shows "No Data" or stale timestamps
- New DAG submissions fail with "cannot resolve agent" errors

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List agent registry service pods
kubectl get pods -n greenlang -l app=agent-registry-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=agent-registry-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to agent registry service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=agent-registry-service | tail -30

# Check deployment status
kubectl describe deployment agent-registry-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment agent-registry-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=agent-registry-service

# Check for rollout issues
kubectl rollout status deployment/agent-registry-service -n greenlang

# Check HPA status (scales 2-10 replicas)
kubectl get hpa -n greenlang -l app=agent-registry-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=agent-registry-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=agent-registry-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for registry-specific errors
kubectl logs -n greenlang -l app=agent-registry-service --tail=500 \
  | grep -i "registry\|agent\|version\|capability\|dependency\|health\|catalog\|cache"

# Look for database connection errors
kubectl logs -n greenlang -l app=agent-registry-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of agent registry service pods
kubectl top pods -n greenlang -l app=agent-registry-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=agent-registry-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes
```

### Step 5: Check Database Connectivity

```bash
# Verify PostgreSQL connectivity
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check database connection pool status in logs
kubectl logs -n greenlang -l app=agent-registry-service --tail=200 \
  | grep -i "pool\|connection\|database\|postgres"

# Check if the agent_registry_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='agent_registry_service'
   ORDER BY table_name;"
```

### Step 6: Check ConfigMap and Secrets

```bash
# Verify the agent registry service ConfigMap exists and is valid
kubectl get configmap agent-registry-service-config -n greenlang
kubectl get configmap agent-registry-service-config -n greenlang -o yaml | head -50

# Verify secrets exist
kubectl get secret agent-registry-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment agent-registry-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 7: Check Network Policies

```bash
# Check network policies affecting the agent registry service
kubectl get networkpolicy -n greenlang | grep agent-registry

# Verify the agent registry service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify upstream services can reach the agent registry service
kubectl run net-test-2 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv agent-registry-service.greenlang.svc.cluster.local 8080'
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137.

**Resolution:**

1. Confirm the OOM cause:
```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. Increase memory limits:
```bash
kubectl patch deployment agent-registry-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "agent-registry-service",
            "resources": {
              "limits": {
                "cpu": "1",
                "memory": "1Gi"
              },
              "requests": {
                "cpu": "250m",
                "memory": "512Mi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

3. Verify pods restart successfully:
```bash
kubectl rollout status deployment/agent-registry-service -n greenlang
kubectl get pods -n greenlang -l app=agent-registry-service
```

### Scenario 2: CrashLoopBackOff -- Database Migration Failure

**Symptoms:** Pod status shows CrashLoopBackOff, init container logs show migration errors.

**Resolution:**

1. Check init container logs:
```bash
kubectl logs -n greenlang <pod-name> -c check-db-migration --tail=100
```

2. Verify database schema:
```bash
kubectl run pg-migration --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description, success FROM flyway_schema_history
   ORDER BY installed_rank DESC LIMIT 5;"
```

3. Restart the deployment after fixing:
```bash
kubectl rollout restart deployment/agent-registry-service -n greenlang
kubectl rollout status deployment/agent-registry-service -n greenlang
```

### Scenario 3: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/agent-registry-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/agent-registry-service -n greenlang
kubectl rollout status deployment/agent-registry-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=agent-registry-service
kubectl port-forward -n greenlang svc/agent-registry-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=agent-registry-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/agent-registry-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the agent registry service is being scraped
up{job="agent-registry-service"} == 1

# Verify agent count metric is populated
gl_agent_registry_agents_total > 0

# Verify query metrics are incrementing
increase(gl_agent_registry_queries_total[5m])
```

### Step 3: Verify Registry Data is Loaded

```bash
# Check registered agents via API
curl -s http://localhost:8080/v1/agents?limit=10 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check service catalog entries
curl -s http://localhost:8080/v1/catalog?limit=10 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Verify Dependency Resolution Works

```bash
# Run dependency resolution test
curl -s http://localhost:8080/v1/agents/GL-FOUND-X-001/dependencies \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

---

## Interim Mitigation

While the Agent Registry & Service Catalog is being restored:

1. **Agent registration data is safe.** All agents, versions, capabilities, dependencies, health checks, and audit events are stored in PostgreSQL with TimescaleDB. The database persists independently.

2. **Orchestrator behavior depends on caching.** If the orchestrator has a cached copy of the agent registry, it may continue to function with stale data until the cache expires.

3. **Health monitoring is suspended.** Agent health status will not be updated during the outage. Unhealthy agents may not be detected.

4. **New agent registrations are blocked.** Any agent registration, version publish, or capability update will fail during the outage.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-foundation` -- engineering response
   - `#platform-oncall` -- on-call engineer
   - `#data-pipeline-ops` -- DAG execution impact

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Agent registry service down, orchestrator using cached data | On-call engineer | Immediate (<5 min) |
| L2 | Agent registry service down > 15 minutes, DAG submissions failing | Platform team lead + #platform-foundation | 15 minutes |
| L3 | Agent registry service down > 30 minutes, audit gap, multiple pipelines blocked | Platform team + data pipeline team + CTO notification | Immediate |
| L4 | Agent registry service down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Agent Registry & Service Catalog Health (`/d/agent-registry-service`)
- **Alert:** `AgentRegistryServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="agent-registry-service"}` (should always be >= 2)
  - `gl_agent_registry_agents_total` (should be stable; drop to 0 indicates data loss)
  - `gl_agent_registry_queries_total` rate (should be non-zero during business hours)
  - `gl_agent_registry_health_checks_total` rate (should be non-zero)
  - `gl_agent_registry_dependency_cycles_total` (should be 0)
  - `gl_agent_registry_query_duration_seconds` p99 (should stay below 50ms)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 85%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 10 replicas** based on CPU and memory utilization
4. **Database connection pool** is sized for expected concurrency (default: min 2, max 10)

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `AgentRegistryServiceDown` | Critical | This alert -- no agent registry service pods running |
| `AgentRegistryHighUnhealthyRate` | Warning | >20% of registered agents are unhealthy |
| `AgentRegistryHighUnhealthyRateCritical` | Critical | >50% of registered agents are unhealthy |
| `AgentRegistryNoRegisteredAgents` | Critical | Registry reports zero registered agents |
| `AgentRegistryDependencyCycleDetected` | Critical | Circular dependency detected in agent graph |
| `AgentRegistryAuditGap` | Critical | Registry operations without audit entries |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Foundation Team
- **Review cadence:** Quarterly or after any P1 agent registry service incident
- **Related runbooks:** [Agent Registry Health Monitoring](./agent-registry-health-monitoring.md)
