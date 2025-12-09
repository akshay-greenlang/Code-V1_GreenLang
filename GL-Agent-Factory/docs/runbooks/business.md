# Business Metrics Runbook

This runbook covers alerts related to business logic, agent functionality, and operational metrics in the GreenLang platform.

---

## Table of Contents

- [LowCalculationThroughput](#lowcalculationthroughput)
- [AgentRegistryEmpty](#agentregistryempty)
- [NoActiveAgents](#noactiveagents)
- [EFCacheMissRateHigh](#efcachemissratehigh)

---

## LowCalculationThroughput

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | LowCalculationThroughput |
| **Severity** | Warning |
| **Team** | Backend |
| **Evaluation Interval** | 60s |
| **For Duration** | 15m |
| **Threshold** | <1 calculation per second |

**PromQL Expression:**

```promql
sum(rate(gl_calculations_total{status="success"}[10m])) < 1
```

### Description

This alert fires when the system is processing fewer than 1 successful calculation per second for 15 minutes. This may indicate a processing bottleneck, infrastructure issue, or simply low demand.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Variable | May indicate blocked processing |
| **Data Impact** | Medium | Calculations may be queued/delayed |
| **SLA Impact** | Medium | May indicate service degradation |
| **Revenue Impact** | Medium | Delayed customer deliverables |

### Diagnostic Steps

1. **Check if low throughput is expected**

   ```bash
   # Compare to historical baseline
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total{status='success'}[10m]offset1d))" | jq .

   # Check time of day - business hours vs. off-hours
   date

   # Check if weekend/holiday
   ```

2. **Check queue depth**

   ```bash
   # Pending calculations
   curl -s "http://prometheus:9090/api/v1/query?query=gl_pending_calculations" | jq .

   # Queue size in message broker
   kubectl exec -n greenlang deploy/rabbitmq -- rabbitmqctl list_queues
   ```

3. **Check calculation failure rate**

   ```bash
   # Failed vs successful calculations
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total[10m]))by(status)" | jq .

   # If many failures, see error-rates runbook
   ```

4. **Check agent health**

   ```bash
   # Agent status
   curl -s "http://prometheus:9090/api/v1/query?query=gl_active_agents" | jq .

   # Agent-specific throughput
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total{status='success'}[10m]))by(agent)" | jq .
   ```

5. **Check upstream dependencies**

   ```bash
   # API request rate
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total[10m]))" | jq .

   # If API requests are also low, issue is upstream
   ```

6. **Check worker pod status**

   ```bash
   # Worker pods running
   kubectl get pods -n greenlang -l role=worker

   # Check for pending/failed pods
   kubectl get pods -n greenlang -l role=worker --field-selector=status.phase!=Running
   ```

### Resolution Steps

#### Scenario 1: Workers not processing (stuck)

```bash
# 1. Check worker logs for errors
kubectl logs -n greenlang -l role=worker --tail=100 | grep -i error

# 2. Check for blocked workers
kubectl logs -n greenlang -l role=worker | grep -E "waiting|blocked|timeout"

# 3. Restart workers
kubectl rollout restart deployment -n greenlang calculation-workers

# 4. Verify processing resumes
watch "curl -s 'http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total{status=\"success\"}[5m]))' | jq '.data.result[0].value[1]'"
```

#### Scenario 2: Queue consumer issues

```bash
# 1. Check queue consumer status
kubectl exec -n greenlang deploy/rabbitmq -- \
  rabbitmqctl list_consumers

# 2. Check queue depth
kubectl exec -n greenlang deploy/rabbitmq -- \
  rabbitmqctl list_queues name messages consumers

# 3. If messages building up but no consumers
kubectl rollout restart deployment -n greenlang queue-consumer

# 4. If queue is empty, check for upstream issue
# No calculations being submitted
```

#### Scenario 3: Dependency blocking processing

```bash
# 1. Check database connectivity
kubectl exec -n greenlang deploy/calculation-workers -- \
  pg_isready -h postgres.greenlang.svc.cluster.local

# 2. Check Redis connectivity
kubectl exec -n greenlang deploy/calculation-workers -- \
  redis-cli -h redis.greenlang.svc.cluster.local ping

# 3. Check emission factor service
curl http://ef-service.greenlang.svc.cluster.local/health

# 4. If dependency down, resolve per appropriate runbook
```

#### Scenario 4: Scale up workers

```bash
# 1. Check current worker count
kubectl get deployment -n greenlang calculation-workers \
  -o jsonpath='{.spec.replicas}'

# 2. Scale up
kubectl scale deployment -n greenlang calculation-workers --replicas=10

# 3. Monitor throughput increase
watch "curl -s 'http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total{status=\"success\"}[5m]))' | jq '.data.result[0].value[1]'"
```

#### Scenario 5: Low traffic is expected

```bash
# 1. Verify this is normal (off-hours, weekend, etc.)
# 2. If expected low traffic, acknowledge alert
# 3. Consider adjusting alert thresholds for different time windows

# Example: Different thresholds for business hours vs. off-hours
# This would require modifying alert rules
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If processing blocked |
| L3 | Customer Success | #customer-success Slack | If customer impact confirmed |

---

## AgentRegistryEmpty

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | AgentRegistryEmpty |
| **Severity** | Critical |
| **Team** | Backend |
| **Evaluation Interval** | 60s |
| **For Duration** | 5m |
| **Threshold** | 0 agents |

**PromQL Expression:**

```promql
gl_registry_agents_count == 0
```

### Description

This alert fires when no agents are registered in the system. This is a critical condition that prevents ALL calculations from being processed.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Critical | ALL calculations will fail |
| **Data Impact** | High | No processing possible |
| **SLA Impact** | Critical | Complete service outage |
| **Revenue Impact** | Critical | Total service disruption |

### Diagnostic Steps

1. **Verify registry state**

   ```bash
   # Query registry directly
   curl http://registry.greenlang.svc.cluster.local/api/v1/agents

   # Check registry service health
   kubectl get pods -n greenlang -l app=agent-registry
   kubectl logs -n greenlang -l app=agent-registry --tail=100
   ```

2. **Check agent registration process**

   ```bash
   # Check agent pods
   kubectl get pods -n greenlang -l type=agent

   # Check agent startup logs
   kubectl logs -n greenlang -l type=agent --tail=100 | grep -i "register"
   ```

3. **Check registry database**

   ```bash
   # Direct database query
   kubectl exec -n greenlang deploy/postgres -- \
     psql -c "SELECT * FROM agent_registry;"
   ```

4. **Check recent deployments**

   ```bash
   # Registry deployment history
   kubectl rollout history deployment -n greenlang agent-registry

   # Agent deployment history
   kubectl rollout history deployment -n greenlang -l type=agent
   ```

### Resolution Steps

#### Scenario 1: Registry service down

```bash
# 1. Check registry pod status
kubectl get pods -n greenlang -l app=agent-registry

# 2. If not running, check events
kubectl describe pod -n greenlang -l app=agent-registry

# 3. Restart registry
kubectl rollout restart deployment -n greenlang agent-registry

# 4. Wait for registry to be ready
kubectl rollout status deployment -n greenlang agent-registry

# 5. Trigger agent re-registration
kubectl rollout restart deployment -n greenlang -l type=agent
```

#### Scenario 2: Registry database corrupted/empty

```bash
# 1. Check database table
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT count(*) FROM agent_registry;"

# 2. If empty, check for database issues
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "\\dt" # List all tables

# 3. Restore from backup if necessary
kubectl exec -n greenlang deploy/postgres -- \
  pg_restore -d greenlang /backups/latest.dump

# 4. Or re-initialize agents
kubectl rollout restart deployment -n greenlang -l type=agent
```

#### Scenario 3: Agent registration failing

```bash
# 1. Check agent logs for registration errors
kubectl logs -n greenlang -l type=agent | grep -i "register\|error"

# 2. Check network connectivity to registry
kubectl exec -n greenlang deploy/gl-001-carbon-emissions -- \
  curl -v http://registry.greenlang.svc.cluster.local/health

# 3. Check DNS resolution
kubectl exec -n greenlang deploy/gl-001-carbon-emissions -- \
  nslookup registry.greenlang.svc.cluster.local

# 4. Fix network/DNS issues and restart agents
```

#### Scenario 4: Configuration issue preventing registration

```bash
# 1. Check agent configuration
kubectl get configmap -n greenlang agent-config -o yaml

# 2. Verify registry URL is correct
kubectl get configmap -n greenlang agent-config \
  -o jsonpath='{.data.REGISTRY_URL}'

# 3. Update configuration if incorrect
kubectl patch configmap -n greenlang agent-config \
  -p '{"data":{"REGISTRY_URL":"http://registry.greenlang.svc.cluster.local"}}'

# 4. Restart agents to pick up new config
kubectl rollout restart deployment -n greenlang -l type=agent
```

### Post-Resolution

1. **Verify agents registered**

   ```bash
   curl http://registry.greenlang.svc.cluster.local/api/v1/agents | jq .
   ```

2. **Verify calculations processing**

   ```bash
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total{status='success'}[5m]))" | jq .
   ```

3. **Check for queued calculations** that need processing

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Immediate response |
| L2 | Backend Team | Direct page | If not resolved in 5 minutes |
| L3 | Engineering Manager | Phone call | This is a P1 incident |

---

## NoActiveAgents

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | NoActiveAgents |
| **Severity** | Critical |
| **Team** | Backend |
| **Evaluation Interval** | 60s |
| **For Duration** | 10m |
| **Threshold** | 0 active agents |

**PromQL Expression:**

```promql
gl_active_agents == 0
```

### Description

This alert fires when no agents are actively processing requests. This differs from AgentRegistryEmpty - agents may be registered but not actively running or healthy.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Critical | No calculations can be processed |
| **Data Impact** | High | Requests will queue indefinitely |
| **SLA Impact** | Critical | Complete processing outage |
| **Revenue Impact** | Critical | Total service disruption |

### Diagnostic Steps

1. **Check agent pod status**

   ```bash
   # All agent pods
   kubectl get pods -n greenlang -l type=agent

   # Non-running agent pods
   kubectl get pods -n greenlang -l type=agent --field-selector=status.phase!=Running
   ```

2. **Check agent health endpoints**

   ```bash
   # Test individual agent health
   for agent in gl-001 gl-002 gl-003; do
     echo "$agent: $(kubectl exec -n greenlang deploy/$agent -- curl -s localhost:8000/health)"
   done
   ```

3. **Check agent readiness**

   ```bash
   # Readiness probe status
   kubectl describe pods -n greenlang -l type=agent | grep -A5 "Readiness:"
   ```

4. **Compare registered vs active**

   ```bash
   # Registered agents
   curl http://registry.greenlang.svc.cluster.local/api/v1/agents | jq length

   # Active agents (from metrics)
   curl -s "http://prometheus:9090/api/v1/query?query=gl_active_agents" | jq .
   ```

### Resolution Steps

#### Scenario 1: Agent pods crashing

```bash
# 1. Check crash reason
kubectl describe pods -n greenlang -l type=agent | grep -A10 "State:"

# 2. Check logs for crash reason
kubectl logs -n greenlang -l type=agent --previous --tail=100

# 3. If OOMKilled, increase memory
kubectl patch deployment -n greenlang <agent-name> \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"agent","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# 4. Restart agents
kubectl rollout restart deployment -n greenlang -l type=agent
```

#### Scenario 2: Agent readiness probe failing

```bash
# 1. Check readiness probe configuration
kubectl get deployment -n greenlang <agent> -o yaml | grep -A10 readinessProbe

# 2. Test readiness endpoint manually
kubectl exec -n greenlang deploy/<agent> -- curl localhost:8000/health/ready

# 3. Check what's making agent not ready
kubectl logs -n greenlang -l type=agent | grep -i "ready\|health"

# 4. If dependency issue, resolve dependency
# 5. If probe too strict, adjust probe settings
kubectl patch deployment -n greenlang <agent> \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"agent","readinessProbe":{"initialDelaySeconds":30,"periodSeconds":10,"failureThreshold":5}}]}}}}'
```

#### Scenario 3: Agents unable to connect to dependencies

```bash
# 1. Check database connectivity
kubectl exec -n greenlang deploy/<agent> -- \
  pg_isready -h postgres.greenlang.svc.cluster.local

# 2. Check Redis connectivity
kubectl exec -n greenlang deploy/<agent> -- \
  redis-cli -h redis.greenlang.svc.cluster.local ping

# 3. Check registry connectivity
kubectl exec -n greenlang deploy/<agent> -- \
  curl http://registry.greenlang.svc.cluster.local/health

# 4. Resolve dependency issues per appropriate runbook
```

#### Scenario 4: All agents scaled to zero

```bash
# 1. Check deployment replicas
kubectl get deployments -n greenlang -l type=agent

# 2. Scale up agents
kubectl scale deployment -n greenlang -l type=agent --replicas=2

# 3. Check HPA settings
kubectl get hpa -n greenlang

# 4. If HPA scaled down incorrectly, adjust min replicas
kubectl patch hpa -n greenlang <agent>-hpa -p '{"spec":{"minReplicas":2}}'
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Immediate response |
| L2 | Backend Team | Direct page | If not resolved in 5 minutes |
| L3 | Engineering Manager | Phone call | This is a P1 incident |

---

## EFCacheMissRateHigh

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | EFCacheMissRateHigh |
| **Severity** | Warning |
| **Team** | Data |
| **Evaluation Interval** | 60s |
| **For Duration** | 15m |
| **Threshold** | >50% miss rate |

**PromQL Expression:**

```promql
(
  sum(rate(gl_ef_lookups_total{cache="miss"}[5m]))
  /
  sum(rate(gl_ef_lookups_total[5m]))
) * 100 > 50
```

### Description

This alert fires when more than 50% of emission factor lookups are cache misses. High cache miss rates increase latency and load on the EF database.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Slower calculations due to DB lookups |
| **Data Impact** | None | Data integrity not affected |
| **SLA Impact** | Medium | Contributes to calculation latency |
| **Revenue Impact** | Low | Performance degradation |

### Diagnostic Steps

1. **Check current cache stats**

   ```bash
   # Cache hit rate
   curl -s "http://prometheus:9090/api/v1/query?query=(sum(rate(gl_ef_lookups_total{cache='hit'}[5m]))/sum(rate(gl_ef_lookups_total[5m])))*100" | jq .

   # Miss rate by EF source
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_ef_lookups_total{cache='miss'}[5m]))by(source)" | jq .
   ```

2. **Check Redis cache status**

   ```bash
   # EF cache key count
   kubectl exec -n greenlang deploy/redis -- \
     redis-cli --scan --pattern 'ef:*' | wc -l

   # Cache memory usage
   kubectl exec -n greenlang deploy/redis -- redis-cli info memory
   ```

3. **Check if new EF types being requested**

   ```bash
   # Recent lookups by type
   kubectl logs -n greenlang -l app=ef-service --tail=500 | \
     grep "lookup" | jq -r '.ef_type' | sort | uniq -c | sort -rn
   ```

4. **Check cache TTL settings**

   ```bash
   # Sample key TTL
   kubectl exec -n greenlang deploy/redis -- \
     redis-cli ttl "ef:ecoinvent:electricity:US"
   ```

5. **Check if cache was flushed**

   ```bash
   # Recent Redis commands
   kubectl exec -n greenlang deploy/redis -- \
     redis-cli monitor | head -100 | grep -i "flush\|del"
   ```

### Resolution Steps

#### Scenario 1: Cache was flushed (needs warming)

```bash
# 1. Warm the cache with common EF lookups
kubectl exec -n greenlang deploy/ef-service -- \
  python -c "from app.cache import warm_cache; warm_cache()"

# 2. Or trigger bulk lookup
kubectl exec -n greenlang deploy/ef-service -- \
  python scripts/warm_ef_cache.py

# 3. Monitor cache hit rate recovery
watch "curl -s 'http://prometheus:9090/api/v1/query?query=(sum(rate(gl_ef_lookups_total{cache=\"hit\"}[5m]))/sum(rate(gl_ef_lookups_total[5m])))*100' | jq '.data.result[0].value[1]'"
```

#### Scenario 2: Cache TTL too short

```bash
# 1. Check current TTL setting
kubectl get configmap -n greenlang ef-service-config \
  -o jsonpath='{.data.EF_CACHE_TTL}'

# 2. Increase cache TTL
kubectl patch configmap -n greenlang ef-service-config \
  -p '{"data":{"EF_CACHE_TTL":"86400"}}'

# 3. Restart EF service to pick up new config
kubectl rollout restart deployment -n greenlang ef-service
```

#### Scenario 3: New EF types not in cache

```bash
# 1. Identify new EF types being requested
kubectl logs -n greenlang -l app=ef-service | \
  grep "cache miss" | jq -r '.ef_code' | sort | uniq -c | sort -rn | head -20

# 2. Add new EF types to cache warming script
# Update scripts/warm_ef_cache.py with new EF codes

# 3. Run cache warming
kubectl exec -n greenlang deploy/ef-service -- \
  python scripts/warm_ef_cache.py

# 4. Consider implementing write-through caching
```

#### Scenario 4: Redis memory full (evicting EF keys)

```bash
# 1. Check Redis memory and evictions
kubectl exec -n greenlang deploy/redis -- redis-cli info memory
kubectl exec -n greenlang deploy/redis -- redis-cli info stats | grep evicted

# 2. If EF keys being evicted, increase Redis memory
kubectl patch deployment -n greenlang redis \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"redis","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# 3. Or use separate Redis instance for EF cache
kubectl apply -f k8s/redis/ef-cache-redis.yaml

# 4. Update EF service to use dedicated cache
kubectl set env deployment/ef-service -n greenlang \
  EF_CACHE_REDIS_HOST=ef-cache-redis.greenlang.svc.cluster.local
```

#### Scenario 5: Implement background cache refresh

```bash
# 1. Enable background refresh
kubectl set env deployment/ef-service -n greenlang \
  BACKGROUND_CACHE_REFRESH=true \
  CACHE_REFRESH_INTERVAL=3600

# 2. This keeps cache warm proactively
# Reduces miss rate during normal operation

# 3. Restart to apply
kubectl rollout restart deployment -n greenlang ef-service
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Data Team | #data-oncall Slack | If cache strategy issue |
| L3 | Backend Team | #backend-oncall Slack | If code changes needed |

---

## Quick Reference Card

| Alert | Severity | Threshold | First Check | Quick Fix |
|-------|----------|-----------|-------------|-----------|
| LowCalculationThroughput | Warning | <1/s | Queue depth | Restart workers |
| AgentRegistryEmpty | Critical | 0 agents | Registry service | Restart registry |
| NoActiveAgents | Critical | 0 active | Agent pod status | Restart agents |
| EFCacheMissRateHigh | Warning | >50% miss | Redis cache stats | Warm cache |

## Business Health Dashboard

Key metrics to monitor for business health:

```promql
# Calculation throughput
sum(rate(gl_calculations_total{status="success"}[5m]))

# Active agents
gl_active_agents

# Queue depth
gl_pending_calculations

# Cache hit rate
sum(rate(gl_ef_lookups_total{cache="hit"}[5m])) / sum(rate(gl_ef_lookups_total[5m])) * 100

# Error rate
sum(rate(gl_errors_total[5m])) / sum(rate(gl_calculations_total[5m])) * 100
```
