# Latency Runbook

This runbook covers alerts related to response times, latency percentiles, and performance degradation in the GreenLang platform.

---

## Table of Contents

- [HighP99Latency](#highp99latency)
- [HighP95Latency](#highp95latency)
- [APIResponseTimeSlow](#apiresponsetime slow)
- [EFLookupSlow](#eflookup slow)

---

## HighP99Latency

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | HighP99Latency |
| **Severity** | Critical |
| **Team** | Backend |
| **Evaluation Interval** | 30s |
| **For Duration** | 5m |
| **SLA Threshold** | 1 second |

**PromQL Expression:**

```promql
histogram_quantile(0.99,
  sum(rate(gl_calculation_duration_seconds_bucket[5m])) by (agent, le)
) > 1
```

### Description

This alert fires when the 99th percentile (P99) latency for agent calculations exceeds 1 second. P99 means 99% of requests complete within this time - only 1% are slower. This is a critical SLA metric.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | 1% of users experiencing >1s response times |
| **Data Impact** | Low | Data integrity not affected |
| **SLA Impact** | Critical | Direct SLA violation (P99 < 1s commitment) |
| **Revenue Impact** | Medium | Poor user experience, potential churn |

### Diagnostic Steps

1. **Identify latency distribution**

   ```bash
   # Get full latency percentiles
   for p in 50 75 90 95 99; do
     echo "P$p: $(curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.$p,sum(rate(gl_calculation_duration_seconds_bucket{agent='{{ $labels.agent }}'}[5m]))by(le))" | jq -r '.data.result[0].value[1]')s"
   done
   ```

2. **Check when latency increased**

   ```bash
   # Query latency over time
   curl -s "http://prometheus:9090/api/v1/query_range?query=histogram_quantile(0.99,sum(rate(gl_calculation_duration_seconds_bucket{agent='{{ $labels.agent }}'}[5m]))by(le))&start=$(date -d '2 hours ago' +%s)&end=$(date +%s)&step=60" | jq '.data.result[0].values | .[-20:]'
   ```

3. **Identify slow calculation types**

   ```bash
   # Latency by calculation type
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_calculation_duration_seconds_bucket{agent='{{ $labels.agent }}'}[5m]))by(calculation_type,le))" | jq .
   ```

4. **Check resource utilization**

   ```bash
   # CPU usage
   kubectl top pods -n greenlang -l agent={{ $labels.agent }}

   # Memory usage
   kubectl exec -n greenlang deploy/{{ $labels.agent }} -- \
     cat /sys/fs/cgroup/memory/memory.usage_in_bytes

   # Check for CPU throttling
   kubectl exec -n greenlang deploy/{{ $labels.agent }} -- \
     cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled
   ```

5. **Check database query times**

   ```bash
   # Query database latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(pg_query_duration_seconds_bucket[5m]))by(le))" | jq .

   # Check slow query log
   kubectl logs -n greenlang -l app=postgres --tail=100 | grep "duration:"
   ```

6. **Check external dependency latency**

   ```bash
   # EF lookup latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_ef_lookup_duration_seconds_bucket[5m]))by(le))" | jq .

   # External API latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(http_client_request_duration_seconds_bucket[5m]))by(host,le))" | jq .
   ```

7. **Check request queue depth**

   ```bash
   # Pending requests
   curl -s "http://prometheus:9090/api/v1/query?query=gl_pending_calculations{agent='{{ $labels.agent }}'}" | jq .
   ```

### Resolution Steps

#### Scenario 1: High CPU causing slowdown

```bash
# 1. Check current CPU usage
kubectl top pods -n greenlang -l agent={{ $labels.agent }}

# 2. Scale horizontally to distribute load
kubectl scale deployment -n greenlang {{ $labels.agent }} --replicas=5

# 3. Wait for new pods to be ready
kubectl rollout status deployment -n greenlang {{ $labels.agent }}

# 4. Verify latency improvement
watch "curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_calculation_duration_seconds_bucket{agent=\"{{ $labels.agent }}\"}[5m]))by(le))' | jq '.data.result[0].value[1]'"
```

#### Scenario 2: Database queries slow

```bash
# 1. Check for long-running queries
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query
           FROM pg_stat_activity
           WHERE state != 'idle'
           ORDER BY duration DESC LIMIT 10;"

# 2. Check for missing indexes
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT schemaname, relname, seq_scan, idx_scan
           FROM pg_stat_user_tables
           WHERE seq_scan > idx_scan
           ORDER BY seq_scan DESC LIMIT 10;"

# 3. Kill long-running queries if safe
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
           WHERE duration > interval '5 minutes' AND query NOT LIKE '%pg_stat%';"

# 4. Add missing indexes (coordinate with DBA)
# 5. Consider read replicas for read-heavy workloads
```

#### Scenario 3: External API latency

```bash
# 1. Identify slow external call
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(http_client_request_duration_seconds_bucket[5m]))by(host,le))" | jq .

# 2. Enable caching for external calls
kubectl set env deployment/{{ $labels.agent }} -n greenlang \
  EXTERNAL_API_CACHE_TTL=3600

# 3. Increase timeout and implement retry with backoff
kubectl set env deployment/{{ $labels.agent }} -n greenlang \
  EXTERNAL_API_TIMEOUT=10s \
  EXTERNAL_API_RETRY_COUNT=3

# 4. Consider async processing for non-critical external calls
```

#### Scenario 4: Memory pressure causing GC pauses

```bash
# 1. Check GC metrics (Python)
kubectl exec -n greenlang deploy/{{ $labels.agent }} -- \
  python -c "import gc; print(gc.get_stats())"

# 2. Increase memory limits
kubectl patch deployment -n greenlang {{ $labels.agent }} \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"4Gi"},"requests":{"memory":"2Gi"}}}]}}}}'

# 3. Tune GC settings
kubectl set env deployment/{{ $labels.agent }} -n greenlang \
  PYTHONMALLOC=malloc \
  MALLOC_TRIM_THRESHOLD_=100000
```

#### Scenario 5: Traffic spike

```bash
# 1. Check current request rate
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total{agent='{{ $labels.agent }}'}[5m]))" | jq .

# 2. Compare to normal rate
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total{agent='{{ $labels.agent }}'}[5m]offset1d))" | jq .

# 3. Scale to handle load
# Calculate needed replicas: current_rate / baseline_rate * current_replicas
kubectl scale deployment -n greenlang {{ $labels.agent }} --replicas=10

# 4. Enable HPA if not already
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ $labels.agent }}-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ $labels.agent }}
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
EOF
```

### Post-Resolution

1. **Verify P99 recovery**

   ```bash
   # Monitor for 15 minutes
   watch -n 30 "curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_calculation_duration_seconds_bucket{agent=\"{{ $labels.agent }}\"}[5m]))by(le))' | jq '.data.result[0].value[1]'"
   ```

2. **Check SLA compliance dashboard**

3. **Document optimization actions** for capacity planning

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If not resolved in 15 minutes |
| L3 | Database Team | #database-oncall Slack | If DB-related |
| L4 | Architecture Team | #architecture Slack | If systemic issue |

### Related Dashboards

- [Agent Performance Dashboard](https://grafana.greenlang.io/d/agent-performance)
- [Latency Breakdown Dashboard](https://grafana.greenlang.io/d/latency-breakdown)

---

## HighP95Latency

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | HighP95Latency |
| **Severity** | Warning |
| **Team** | Backend |
| **Evaluation Interval** | 30s |
| **For Duration** | 10m |
| **Threshold** | 500ms |

**PromQL Expression:**

```promql
histogram_quantile(0.95,
  sum(rate(gl_calculation_duration_seconds_bucket[5m])) by (agent, le)
) > 0.5
```

### Description

This alert fires when P95 latency exceeds 500ms for 10 minutes. This is a warning that latency is trending upward and may soon breach P99 SLA thresholds.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | 5% of users experiencing degraded performance |
| **Data Impact** | None | Data integrity not affected |
| **SLA Impact** | Low | Warning indicator, not yet SLA breach |
| **Revenue Impact** | Low | Early warning for potential issues |

### Diagnostic Steps

1. **Compare to P99 latency**

   ```bash
   # Check if P99 is also elevated
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_calculation_duration_seconds_bucket{agent='{{ $labels.agent }}'}[5m]))by(le))" | jq .
   ```

2. **Check latency trend**

   ```bash
   # Is it getting worse?
   curl -s "http://prometheus:9090/api/v1/query_range?query=histogram_quantile(0.95,sum(rate(gl_calculation_duration_seconds_bucket{agent='{{ $labels.agent }}'}[5m]))by(le))&start=$(date -d '1 hour ago' +%s)&end=$(date +%s)&step=60" | jq '.data.result[0].values | .[-10:]'
   ```

3. **Identify slow operations**

   Follow same diagnostic steps as [HighP99Latency](#diagnostic-steps)

### Resolution Steps

Since this is a warning alert, focus on prevention:

1. **Monitor closely** - This often precedes P99 breach

2. **Proactive scaling**

   ```bash
   # Add capacity before P99 breach
   kubectl scale deployment -n greenlang {{ $labels.agent }} --replicas=4
   ```

3. **Investigate root cause** while not yet critical

4. **Plan capacity** if this is becoming a pattern

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If trending toward P99 breach |

---

## APIResponseTimeSlow

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | APIResponseTimeSlow |
| **Severity** | Warning |
| **Team** | Backend |
| **Evaluation Interval** | 30s |
| **For Duration** | 5m |
| **Threshold** | 2 seconds |

**PromQL Expression:**

```promql
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket{handler!~"/metrics|/health"}[5m])) by (handler, le)
) > 2
```

### Description

This alert fires when API endpoint P95 response time exceeds 2 seconds. This excludes health and metrics endpoints which are expected to be fast.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Slow API responses for 5% of requests |
| **Data Impact** | None | Data integrity not affected |
| **SLA Impact** | Medium | Contributes to overall latency SLA |
| **Revenue Impact** | Medium | Degraded API consumer experience |

### Diagnostic Steps

1. **Identify the slow endpoint**

   ```bash
   # Get P95 by handler
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(http_request_duration_seconds_bucket{handler!~'/metrics|/health'}[5m]))by(handler,le))" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse | .[0:5]'
   ```

2. **Analyze request breakdown**

   ```bash
   # Check specific endpoint metrics
   curl -s "http://prometheus:9090/api/v1/query?query=http_request_duration_seconds_sum{handler='{{ $labels.handler }}'}/http_request_duration_seconds_count{handler='{{ $labels.handler }}'}" | jq .
   ```

3. **Check downstream service latency**

   ```bash
   # If endpoint calls other services
   kubectl logs -n greenlang -l app=api-gateway --tail=500 | \
     grep "{{ $labels.handler }}" | grep -E "duration|latency"
   ```

4. **Review endpoint implementation**

   ```bash
   # Check for N+1 queries or inefficient code
   kubectl logs -n greenlang -l app=api-gateway --tail=200 | \
     grep "{{ $labels.handler }}" | grep "query"
   ```

### Resolution Steps

#### Scenario 1: Database-heavy endpoint

```bash
# 1. Check query count per request
kubectl logs -n greenlang -l app=api-gateway | \
  grep "{{ $labels.handler }}" | grep -c "SELECT"

# 2. If N+1 query pattern, fix in code (batch queries)
# 3. Add caching for frequently accessed data
kubectl set env deployment/api-gateway -n greenlang \
  CACHE_{{ endpoint }}_TTL=300
```

#### Scenario 2: Large payload serialization

```bash
# 1. Check response size
kubectl logs -n greenlang -l app=api-gateway | \
  grep "{{ $labels.handler }}" | jq -r '.response_size' | sort -n | tail

# 2. Implement pagination if returning large lists
# 3. Enable response compression
kubectl set env deployment/api-gateway -n greenlang \
  ENABLE_GZIP=true
```

#### Scenario 3: External API dependency

```bash
# 1. Check external call latency
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(http_client_request_duration_seconds_bucket[5m]))by(host,le))" | jq .

# 2. Implement caching
# 3. Consider async processing
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If affecting customer experience |

---

## EFLookupSlow

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | EFLookupSlow |
| **Severity** | Warning |
| **Team** | Data |
| **Evaluation Interval** | 30s |
| **For Duration** | 10m |
| **Threshold** | 500ms |

**PromQL Expression:**

```promql
histogram_quantile(0.95,
  sum(rate(gl_ef_lookup_duration_seconds_bucket[5m])) by (le)
) > 0.5
```

### Description

This alert fires when emission factor lookups P95 latency exceeds 500ms. Slow EF lookups directly impact calculation latency since most calculations require multiple EF lookups.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Slower calculations across all agents |
| **Data Impact** | None | Data integrity not affected |
| **SLA Impact** | Medium | Contributes to calculation latency |
| **Revenue Impact** | Low | Background service, less visible |

### Diagnostic Steps

1. **Check EF service health**

   ```bash
   # Service status
   kubectl get pods -n greenlang -l app=ef-service

   # Health endpoint
   kubectl exec -n greenlang deploy/ef-service -- curl localhost:8000/health
   ```

2. **Check cache performance**

   ```bash
   # Cache hit rate
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_ef_lookups_total{cache='hit'}[5m]))/sum(rate(gl_ef_lookups_total[5m]))" | jq .

   # If low hit rate, cache is not effective
   ```

3. **Check database query performance**

   ```bash
   # EF database query latency
   kubectl logs -n greenlang -l app=ef-service --tail=200 | \
     grep "query_duration" | tail -20
   ```

4. **Check external API latency**

   ```bash
   # If fetching from external source
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(gl_ef_external_lookup_duration_seconds_bucket[5m]))by(source,le))" | jq .
   ```

### Resolution Steps

#### Scenario 1: Low cache hit rate

```bash
# 1. Check cache size
kubectl exec -n greenlang deploy/redis -- redis-cli info memory

# 2. Increase cache TTL
kubectl set env deployment/ef-service -n greenlang \
  EF_CACHE_TTL=86400

# 3. Pre-warm cache with common EF lookups
kubectl exec -n greenlang deploy/ef-service -- \
  python -c "from app.cache import warm_cache; warm_cache()"
```

#### Scenario 2: Database performance

```bash
# 1. Check for missing indexes
kubectl exec -n greenlang deploy/ef-postgres -- \
  psql -c "EXPLAIN ANALYZE SELECT * FROM emission_factors WHERE code = 'sample';"

# 2. Add appropriate indexes
kubectl exec -n greenlang deploy/ef-postgres -- \
  psql -c "CREATE INDEX CONCURRENTLY idx_ef_code ON emission_factors(code);"

# 3. Vacuum and analyze tables
kubectl exec -n greenlang deploy/ef-postgres -- \
  psql -c "VACUUM ANALYZE emission_factors;"
```

#### Scenario 3: External API slow

```bash
# 1. Enable aggressive caching
kubectl set env deployment/ef-service -n greenlang \
  EXTERNAL_CACHE_TTL=604800

# 2. Implement background refresh
kubectl set env deployment/ef-service -n greenlang \
  BACKGROUND_REFRESH_ENABLED=true

# 3. Consider bulk import instead of on-demand lookups
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Data Team | #data-oncall Slack | If EF service issues |
| L3 | Database Team | #database-oncall Slack | If DB performance issue |

---

## Quick Reference Card

| Alert | Severity | Threshold | First Check | Quick Fix |
|-------|----------|-----------|-------------|-----------|
| HighP99Latency | Critical | >1s | Check CPU/DB latency | Scale horizontally |
| HighP95Latency | Warning | >500ms | Check trend | Proactive scaling |
| APIResponseTimeSlow | Warning | >2s | Identify slow endpoint | Add caching |
| EFLookupSlow | Warning | >500ms | Check cache hit rate | Warm cache |

## Performance Optimization Checklist

When investigating latency issues, check these in order:

1. **Horizontal scaling** - Is the service under-provisioned?
2. **Database queries** - N+1 queries, missing indexes, slow queries?
3. **External dependencies** - Slow external APIs?
4. **Caching** - Is caching enabled and effective?
5. **Resource limits** - CPU throttling, memory pressure?
6. **Code efficiency** - Inefficient algorithms, large payloads?
7. **Network** - Cross-region calls, DNS resolution?
