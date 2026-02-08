# Agent Registry Health Monitoring

## Alert

**Alert Names:** `AgentRegistryHighUnhealthyRate`, `AgentRegistryHighUnhealthyRateCritical`, `AgentRegistryHealthCheckFailure`

**Severities:** Warning (>20% unhealthy), Critical (>50% unhealthy), Warning (health check failures)

**Thresholds:**
- Warning: `gl_agent_registry_health_status{status="unhealthy"} / gl_agent_registry_agents_total > 0.20` for 5 minutes
- Critical: `gl_agent_registry_health_status{status="unhealthy"} / gl_agent_registry_agents_total > 0.50` for 5 minutes
- Warning: `sum(rate(gl_agent_registry_health_checks_total{status="unhealthy"}[5m])) > 1` for 10 minutes

---

## Description

These alerts fire when a significant proportion of agents registered in the Agent Registry & Service Catalog (AGENT-FOUND-007) are reporting unhealthy status, or when health check probes are consistently failing. The registry performs periodic health checks on all registered agents to track their operational status and availability.

Health check statuses:
- **healthy** -- Agent is running and responding normally
- **unhealthy** -- Agent is not responding or returning error responses
- **degraded** -- Agent is responding but with elevated latency or partial functionality
- **unknown** -- Health check timed out or returned an unexpected response
- **starting** -- Agent is in startup phase and not yet accepting traffic

When a high percentage of agents are unhealthy:
- **The orchestrator may fail to execute DAGs** because required agents are unavailable
- **Calculation pipelines are disrupted** because dependent agents cannot process requests
- **Service catalog shows inaccurate status** leading to confusion about agent availability
- **SLO error budgets are consumed** as unhealthy agents cannot serve requests

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Pipelines using unhealthy agents will fail or produce incomplete results |
| **Data Impact** | Low | Agent data is preserved; only real-time processing is affected |
| **SLA Impact** | High | Agent availability SLA targets violated |
| **Revenue Impact** | Medium | Customer workflows dependent on unhealthy agents are blocked |
| **Compliance Impact** | Low | Health monitoring data continues to be recorded |
| **Downstream Impact** | High | Orchestrator must reroute around unhealthy agents; cascading failures possible |

---

## Symptoms

- `gl_agent_registry_health_status{status="unhealthy"}` gauge is elevated
- Health check results in Grafana show increasing failure rate
- Orchestrator logs show "agent unavailable" or "health check failed" warnings
- DAG execution latency increases due to unhealthy agent retries
- Service catalog shows agents with "unhealthy" or "degraded" status
- Health check latency (p99) is elevated above normal baselines

---

## Diagnostic Steps

### Step 1: Identify Unhealthy Agents

```bash
# Query the registry API for unhealthy agents
curl -s http://localhost:8080/v1/agents?health_status=unhealthy \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check health check metrics by agent
kubectl port-forward -n greenlang svc/agent-registry-service 8080:8080
curl -s http://localhost:8080/metrics | grep gl_agent_registry_health
```

### Step 2: Check Individual Agent Pod Status

```bash
# For each unhealthy agent, check its pod status
# Example for orchestrator (GL-FOUND-X-001):
kubectl get pods -n greenlang -l app=orchestrator-service
kubectl get pods -n greenlang -l app=schema-compiler-service
kubectl get pods -n greenlang -l app=normalizer-service
kubectl get pods -n greenlang -l app=assumptions-service
kubectl get pods -n greenlang -l app=citations-service
kubectl get pods -n greenlang -l app=access-guard-service
kubectl get pods -n greenlang -l app=agent-registry-service

# Check for CrashLoopBackOff or other failure states
kubectl get pods -n greenlang --field-selector status.phase!=Running
```

### Step 3: Check Health Check Probe Configuration

```bash
# Review health check configuration
kubectl get configmap agent-registry-service-config -n greenlang -o yaml \
  | grep -A10 "health_monitoring"

# Check health check timeout and interval settings
kubectl logs -n greenlang -l app=agent-registry-service --tail=200 \
  | grep -i "health.check\|probe\|timeout\|unreachable"
```

### Step 4: Check Network Connectivity Between Registry and Agents

```bash
# Test connectivity from registry pod to each agent service
REGISTRY_POD=$(kubectl get pods -n greenlang -l app=agent-registry-service -o jsonpath='{.items[0].metadata.name}')

# Test orchestrator
kubectl exec -n greenlang "$REGISTRY_POD" -- nc -zv orchestrator-service.greenlang.svc.cluster.local 8080

# Test schema compiler
kubectl exec -n greenlang "$REGISTRY_POD" -- nc -zv schema-compiler-service.greenlang.svc.cluster.local 8080

# Check network policies that may be blocking health probes
kubectl get networkpolicy -n greenlang | grep -E "agent-registry|default-deny"
```

### Step 5: Check Agent-Specific Logs

```bash
# Get logs from an unhealthy agent (replace with actual agent service name)
kubectl logs -n greenlang -l app=<unhealthy-agent-service> --tail=200

# Look for common failure patterns
kubectl logs -n greenlang -l app=<unhealthy-agent-service> --tail=500 \
  | grep -i "error\|fatal\|panic\|oom\|timeout\|connection refused"
```

### Step 6: Check Shared Infrastructure

```bash
# Check PostgreSQL status (shared dependency for all agents)
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check Redis status (shared cache)
kubectl run redis-test --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local ping

# Check node resource pressure
kubectl top nodes
kubectl describe nodes | grep -A5 "Conditions"
```

---

## Resolution Steps

### Scenario 1: Single Agent Unhealthy -- Agent-Specific Issue

**Symptoms:** One specific agent is unhealthy; others are healthy.

**Resolution:**

1. Identify the unhealthy agent from the registry API or dashboard

2. Check the agent's pod status and logs:
```bash
kubectl get pods -n greenlang -l app=<agent-service-name>
kubectl logs -n greenlang -l app=<agent-service-name> --tail=200
kubectl describe pod -n greenlang <pod-name>
```

3. Restart the affected agent:
```bash
kubectl rollout restart deployment/<agent-service-name> -n greenlang
kubectl rollout status deployment/<agent-service-name> -n greenlang
```

4. Verify health recovery:
```bash
curl -s http://localhost:8080/v1/agents/<agent-id>/health \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Scenario 2: Multiple Agents Unhealthy -- Infrastructure Issue

**Symptoms:** Many agents are unhealthy simultaneously; shared infrastructure may be the root cause.

**Resolution:**

1. Check shared infrastructure (database, Redis, network):
```bash
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

kubectl run redis-test --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local ping
```

2. Check for node-level issues:
```bash
kubectl top nodes
kubectl get nodes -o wide
kubectl describe nodes | grep -B5 -A5 "Pressure\|NotReady"
```

3. If database is the issue, check database pod status:
```bash
kubectl get pods -n database
kubectl logs -n database -l app=postgresql --tail=200
```

4. If Redis is the issue:
```bash
kubectl get pods -n redis
kubectl logs -n redis -l app=redis --tail=200
```

### Scenario 3: Health Check Probe Misconfiguration

**Symptoms:** Agents are actually running fine but health checks are reporting failures due to misconfigured timeouts or wrong endpoints.

**Resolution:**

1. Review health check configuration:
```bash
kubectl get configmap agent-registry-service-config -n greenlang -o yaml
```

2. Increase health check timeout if needed:
```bash
kubectl edit configmap agent-registry-service-config -n greenlang
# Update GL_AGENT_REGISTRY_HEALTH_CHECK_TIMEOUT_MS to a higher value (e.g., 10000)
```

3. Restart the registry to pick up configuration changes:
```bash
kubectl rollout restart deployment/agent-registry-service -n greenlang
```

### Scenario 4: Dependency Cycle Detected

**Symptoms:** `AgentRegistryDependencyCycleDetected` alert is firing; dependency resolution is failing.

**Resolution:**

1. Check which dependencies are involved in the cycle:
```bash
curl -s http://localhost:8080/v1/dependencies/cycles \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. Review the dependency graph:
```bash
curl -s http://localhost:8080/v1/dependencies/graph \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

3. Remove or mark as optional the dependency that creates the cycle:
```bash
curl -s -X DELETE http://localhost:8080/v1/dependencies/<dependency-id> \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

4. Verify cycle is resolved:
```bash
curl -s http://localhost:8080/v1/dependencies/validate \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify All Agents Are Healthy

```bash
# Check overall health status
curl -s http://localhost:8080/v1/agents/health/summary \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Verify Prometheus metrics
curl -s http://localhost:8080/metrics | grep gl_agent_registry_health_status
```

### Step 2: Verify Dependency Resolution

```bash
# Run a full dependency resolution for the orchestrator
curl -s http://localhost:8080/v1/agents/GL-FOUND-X-001/dependencies/resolve \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 3: Review Health Check History

```promql
# Check health check success rate over the last hour
sum(rate(gl_agent_registry_health_checks_total{status="healthy"}[1h]))
/ sum(rate(gl_agent_registry_health_checks_total[1h]))

# Check health check latency trend
histogram_quantile(0.99, sum(rate(gl_agent_registry_health_check_duration_ms_bucket[1h])) by (le))
```

---

## Interim Mitigation

While agent health issues are being resolved:

1. **The orchestrator can skip unhealthy agents.** If an unhealthy agent has an optional dependency, the orchestrator can proceed without it.

2. **Cached registry data remains available.** Even if health status is stale, agent metadata and capabilities are still accessible from cache.

3. **Manual agent status override.** If health checks are misconfigured, an admin can manually set agent status:
```bash
curl -s -X PUT http://localhost:8080/v1/agents/<agent-id>/health \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "healthy", "reason": "manual override during incident"}'
```

4. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-foundation` -- engineering response
   - `#data-pipeline-ops` -- pipeline execution impact

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Single agent unhealthy, non-critical path | On-call engineer | 15 minutes |
| L2 | Multiple agents unhealthy (>20%), pipelines impacted | Platform team lead + #platform-foundation | 10 minutes |
| L3 | Majority agents unhealthy (>50%), infrastructure failure suspected | Platform team + infrastructure team | Immediate |
| L4 | All agents unhealthy, complete pipeline outage | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Agent Registry & Service Catalog Health (`/d/agent-registry-service`)
- **Alerts:** `AgentRegistryHighUnhealthyRate`, `AgentRegistryHealthCheckFailure`, `AgentRegistryDependencyCycleDetected`
- **Key metrics to watch:**
  - `gl_agent_registry_health_status` by status (healthy/unhealthy/degraded)
  - `gl_agent_registry_health_checks_total` rate by status
  - `gl_agent_registry_health_check_duration_ms` p99
  - `gl_agent_registry_dependency_cycles_total` (should be 0)
  - `gl_agent_registry_dependency_failures_total` rate

### Health Check Configuration

1. **Default health check interval:** 60 seconds
2. **Default health check timeout:** 5000ms
3. **Health check retention:** 30 days in TimescaleDB hypertable
4. **Probe types:** liveness, readiness, startup, deep

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `AgentRegistryHighUnhealthyRate` | Warning | >20% of registered agents are unhealthy |
| `AgentRegistryHighUnhealthyRateCritical` | Critical | >50% of registered agents are unhealthy |
| `AgentRegistryHealthCheckFailure` | Warning | Health checks consistently failing |
| `AgentRegistryDependencyCycleDetected` | Critical | Circular dependency detected |
| `AgentRegistryMissingDependency` | Warning | Unresolved agent dependencies |
| `AgentRegistryServiceDown` | Critical | Registry service itself is down |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Foundation Team
- **Review cadence:** Quarterly or after any P1 agent health monitoring incident
- **Related runbooks:** [Agent Registry Service Down](./agent-registry-service-down.md)
