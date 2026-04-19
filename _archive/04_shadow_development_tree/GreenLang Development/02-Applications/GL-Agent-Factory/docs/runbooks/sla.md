# SLA Compliance Runbook

This runbook covers alerts related to Service Level Agreement (SLA) compliance in the GreenLang platform. These alerts are the most critical as they directly indicate contractual obligations at risk.

---

## Table of Contents

- [SLAAvailabilityViolation](#slaavailabilityviolation)
- [SLALatencyViolation](#slalatencyviolation)
- [ErrorBudgetBurning](#errorbudgetburning)

---

## GreenLang SLA Commitments

Before responding to SLA alerts, understand our commitments:

| SLA Metric | Target | Measurement Window |
|------------|--------|-------------------|
| **Availability** | 99.9% | Monthly |
| **P99 Latency** | <1 second | Per calculation |
| **Error Rate** | <1% | Monthly |
| **Data Accuracy** | >99.5% | Per calculation |

**Error Budget Calculation:**

- Monthly allowance: 0.1% of requests can fail (availability)
- For 1M requests/month: 1,000 errors allowed
- Daily budget: ~33 errors/day

---

## SLAAvailabilityViolation

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | SLAAvailabilityViolation |
| **Severity** | Critical |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 5m |
| **SLA Target** | 99.9% |

**PromQL Expression:**

```promql
(
  1 - (
    sum(rate(http_requests_total{status=~"5.."}[1h]))
    /
    sum(rate(http_requests_total[1h]))
  )
) * 100 < 99.9
```

### Description

This alert fires when system availability drops below 99.9% over a 1-hour window. This is a CRITICAL SLA violation that may trigger contractual penalties and requires immediate executive notification.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Critical | Service unavailable for users |
| **Data Impact** | Medium | Requests failing |
| **SLA Impact** | Critical | Direct SLA breach |
| **Revenue Impact** | Critical | Contractual penalties may apply |
| **Reputation** | High | Customer trust affected |

### Immediate Actions

**This is a P1 Incident - Follow Incident Response Procedure**

1. **Declare Incident**

   ```bash
   # Create incident in PagerDuty
   # Notify incident commander
   # Start incident bridge call
   ```

2. **Notify Stakeholders**

   - Engineering Manager
   - VP Engineering
   - Customer Success (for proactive customer communication)

3. **Start Incident Timeline** - Document all actions with timestamps

### Diagnostic Steps

1. **Determine current availability**

   ```bash
   # Current 1-hour availability
   curl -s "http://prometheus:9090/api/v1/query?query=(1-(sum(rate(http_requests_total{status=~'5..'}[1h]))/sum(rate(http_requests_total[1h]))))*100" | jq .

   # Current 24-hour availability
   curl -s "http://prometheus:9090/api/v1/query?query=(1-(sum(rate(http_requests_total{status=~'5..'}[24h]))/sum(rate(http_requests_total[24h]))))*100" | jq .
   ```

2. **Identify error source**

   ```bash
   # Errors by endpoint
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total{status=~'5..'}[1h]))by(handler)" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse'

   # Errors by status code
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total{status=~'5..'}[1h]))by(status)" | jq .
   ```

3. **Identify when degradation started**

   ```bash
   # Availability over time
   curl -s "http://prometheus:9090/api/v1/query_range?query=(1-(sum(rate(http_requests_total{status=~'5..'}[5m]))/sum(rate(http_requests_total[5m]))))*100&start=$(date -d '4 hours ago' +%s)&end=$(date +%s)&step=60" | jq '.data.result[0].values | map(select(.[1] | tonumber < 99.9))'
   ```

4. **Check for correlated issues**

   ```bash
   # Service health
   curl -s "http://prometheus:9090/api/v1/query?query=up{job=~'greenlang.*'}" | jq .

   # Pod status
   kubectl get pods -n greenlang --field-selector=status.phase!=Running

   # Recent events
   kubectl get events -n greenlang --sort-by='.lastTimestamp' | tail -30
   ```

5. **Check recent changes**

   ```bash
   # Deployment history
   kubectl rollout history deployment -n greenlang --all-namespaces

   # Config changes
   kubectl get configmaps -n greenlang -o yaml | md5sum
   ```

### Resolution Steps

#### Scenario 1: Single service causing errors

```bash
# 1. Identify the failing service
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total{status=~'5..'}[15m]))by(job)" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse | .[0]'

# 2. Check service logs
kubectl logs -n greenlang -l app=<service> --tail=200 | grep -i error

# 3. Rollback if recent deployment
kubectl rollout undo deployment -n greenlang <service>

# 4. Scale up for resilience
kubectl scale deployment -n greenlang <service> --replicas=5

# 5. Monitor recovery
watch "curl -s 'http://prometheus:9090/api/v1/query?query=(1-(sum(rate(http_requests_total{status=~\"5..\"}[5m]))/sum(rate(http_requests_total[5m]))))*100' | jq '.data.result[0].value[1]'"
```

#### Scenario 2: Database causing errors

```bash
# 1. Check database health
kubectl exec -n greenlang deploy/postgres -- pg_isready
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 2. Check for blocking queries
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pid, now() - query_start AS duration, query
           FROM pg_stat_activity WHERE state = 'active'
           ORDER BY duration DESC LIMIT 5;"

# 3. Kill blocking queries if safe
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
           WHERE state = 'active' AND query_start < now() - interval '5 minutes';"

# 4. If database overloaded, reduce connection limits temporarily
kubectl scale deployment -n greenlang -l role=worker --replicas=1
```

#### Scenario 3: Infrastructure failure

```bash
# 1. Check node health
kubectl get nodes
kubectl describe nodes | grep -A10 "Conditions:"

# 2. Check for node pressure
kubectl describe nodes | grep -E "MemoryPressure|DiskPressure|PIDPressure"

# 3. Drain unhealthy nodes
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# 4. Add emergency capacity
# AWS:
aws autoscaling set-desired-capacity --auto-scaling-group-name greenlang-nodes --desired-capacity 20

# 5. Verify pods rescheduled
kubectl get pods -n greenlang -o wide
```

#### Scenario 4: Enable degraded mode

```bash
# 1. If full recovery not possible, enable degraded mode
# Disable non-critical features

# 2. Enable circuit breakers
kubectl set env deployment -n greenlang --all \
  CIRCUIT_BREAKER_ENABLED=true \
  CIRCUIT_BREAKER_THRESHOLD=5

# 3. Enable rate limiting
kubectl apply -f k8s/emergency/rate-limit-policy.yaml

# 4. Communicate to customers
# "We are experiencing degraded performance. Some features temporarily unavailable."
```

### Post-Incident

1. **Document SLA impact**

   ```bash
   # Calculate total downtime
   # Calculate error count during incident
   # Determine if SLA credit needed
   ```

2. **Customer communication**

   - Status page update
   - Direct communication to affected customers
   - SLA credit calculation if applicable

3. **Post-incident review**

   - Root cause analysis
   - Timeline of events
   - Prevention measures
   - Runbook updates

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Immediate - P1 |
| L2 | Engineering Manager | Direct page | Immediately |
| L3 | VP Engineering | Phone call | If >15 min without resolution |
| L4 | CEO | Phone call | If customer facing >30 min |

---

## SLALatencyViolation

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | SLALatencyViolation |
| **Severity** | Critical |
| **Team** | Backend |
| **Evaluation Interval** | 60s |
| **For Duration** | 5m |
| **SLA Target** | P99 < 1 second |

**PromQL Expression:**

```promql
histogram_quantile(0.99,
  sum(rate(gl_calculation_duration_seconds_bucket[1h])) by (le)
) > 1
```

### Description

This alert fires when P99 calculation latency exceeds 1 second over a 1-hour window. This indicates that more than 1% of calculations are taking longer than our SLA commitment.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | Slow response times for users |
| **Data Impact** | Low | Data integrity not affected |
| **SLA Impact** | Critical | Direct latency SLA breach |
| **Revenue Impact** | Medium | Poor user experience |

### Diagnostic Steps

1. **Determine current latency**

   ```bash
   # Current P99 latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_calculation_duration_seconds_bucket[1h]))by(le))" | jq .

   # Latency by percentile
   for p in 50 75 90 95 99; do
     echo "P$p: $(curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.$p,sum(rate(gl_calculation_duration_seconds_bucket[1h]))by(le))" | jq -r '.data.result[0].value[1]')s"
   done
   ```

2. **Identify slow agents**

   ```bash
   # Latency by agent
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_calculation_duration_seconds_bucket[1h]))by(agent,le))" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse'
   ```

3. **Check for bottleneck**

   ```bash
   # Database latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(pg_query_duration_seconds_bucket[1h]))by(le))" | jq .

   # EF lookup latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_ef_lookup_duration_seconds_bucket[1h]))by(le))" | jq .

   # External API latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(http_client_request_duration_seconds_bucket[1h]))by(host,le))" | jq .
   ```

4. **Check resource utilization**

   ```bash
   # CPU usage
   kubectl top pods -n greenlang --sort-by=cpu | head -10

   # Memory usage
   kubectl top pods -n greenlang --sort-by=memory | head -10

   # Check for throttling
   kubectl get pods -n greenlang -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].state}{"\n"}{end}'
   ```

### Resolution Steps

#### Scenario 1: CPU-bound slowdown

```bash
# 1. Scale horizontally
kubectl scale deployment -n greenlang -l type=agent --replicas=10

# 2. Or increase CPU limits
kubectl patch deployment -n greenlang <agent> \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"agent","resources":{"limits":{"cpu":"4"}}}]}}}}'

# 3. Enable HPA if not already
kubectl apply -f k8s/hpa/agent-hpa.yaml
```

#### Scenario 2: Database slowdown

```bash
# 1. Kill slow queries
kubectl exec -n greenlang deploy/postgres -- \
  psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
           WHERE state = 'active' AND query_start < now() - interval '10 seconds';"

# 2. Add read replicas if available
kubectl scale statefulset -n greenlang postgres-replica --replicas=3

# 3. Route read queries to replicas
kubectl set env deployment -n greenlang -l type=agent \
  DATABASE_READ_HOST=postgres-replica.greenlang.svc.cluster.local
```

#### Scenario 3: External dependency slow

```bash
# 1. Enable aggressive caching
kubectl set env deployment -n greenlang -l type=agent \
  EXTERNAL_CACHE_TTL=3600 \
  EXTERNAL_CACHE_ENABLED=true

# 2. Increase timeouts but fail faster
kubectl set env deployment -n greenlang -l type=agent \
  EXTERNAL_TIMEOUT=5s \
  EXTERNAL_RETRY_COUNT=1

# 3. Enable fallback to cached data
kubectl set env deployment -n greenlang -l type=agent \
  FALLBACK_TO_CACHE=true
```

#### Scenario 4: Traffic spike

```bash
# 1. Check current load
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total[5m]))" | jq .

# 2. Scale to handle load
kubectl scale deployment -n greenlang -l type=agent --replicas=20

# 3. Enable request queuing with backpressure
kubectl set env deployment -n greenlang api-gateway \
  MAX_CONCURRENT_REQUESTS=1000 \
  REQUEST_QUEUE_SIZE=5000
```

### Post-Resolution

1. **Verify latency recovery**

   ```bash
   # Monitor P99 for 30 minutes
   watch -n 60 "curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(gl_calculation_duration_seconds_bucket[5m]))by(le))' | jq '.data.result[0].value[1]'"
   ```

2. **Calculate SLA impact**

   ```bash
   # Percentage of requests exceeding SLA during incident
   # Duration of violation
   ```

3. **Capacity planning** - Review if current capacity is sufficient

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | Direct page | If not resolved in 10 minutes |
| L3 | Engineering Manager | Phone call | If SLA breach continuing >20 min |

---

## ErrorBudgetBurning

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | ErrorBudgetBurning |
| **Severity** | Warning |
| **Team** | Backend |
| **Evaluation Interval** | 60s |
| **For Duration** | 30m |
| **Threshold** | >0.5% error rate (burning 50%+ of budget) |

**PromQL Expression:**

```promql
(
  sum(rate(gl_errors_total[1h]))
  /
  sum(rate(gl_calculations_total[1h]))
) * 100 > 0.5
```

### Description

This alert fires when the error rate exceeds 0.5%, indicating we are consuming error budget faster than sustainable. At this rate, we will exhaust our monthly 1% error budget before month end.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Some requests failing |
| **Data Impact** | Low | Failed requests can retry |
| **SLA Impact** | High | On track to breach SLA |
| **Revenue Impact** | Medium | Risk of SLA penalty |

### Error Budget Calculation

```bash
# Monthly error budget: 1% of requests
# Current month day: $(date +%d)
# Days in month: 30
# Budget consumed should be < (day/30) * 100%

# Example: Day 15 should have <50% budget consumed
# If error rate is 0.5%, we're burning 50% of daily budget each hour
```

### Diagnostic Steps

1. **Calculate remaining error budget**

   ```bash
   # Current error rate
   curl -s "http://prometheus:9090/api/v1/query?query=(sum(rate(gl_errors_total[24h]))/sum(rate(gl_calculations_total[24h])))*100" | jq .

   # Monthly budget remaining
   # (This requires custom calculation based on month start)
   ```

2. **Identify error sources**

   ```bash
   # Errors by agent
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_errors_total[1h]))by(agent)" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse'

   # Errors by type
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_errors_total[1h]))by(error_type)" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse'
   ```

3. **Check error trend**

   ```bash
   # Is error rate increasing or stable?
   curl -s "http://prometheus:9090/api/v1/query_range?query=(sum(rate(gl_errors_total[5m]))/sum(rate(gl_calculations_total[5m])))*100&start=$(date -d '6 hours ago' +%s)&end=$(date +%s)&step=300" | jq '.data.result[0].values'
   ```

4. **Identify root cause**

   Follow diagnostic steps from [HighErrorRate](./error-rates.md#higherrorrate)

### Resolution Steps

#### Scenario 1: Specific agent causing errors

```bash
# 1. Identify agent
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_errors_total[1h]))by(agent)" | jq .

# 2. Check agent logs
kubectl logs -n greenlang -l agent=<agent> --tail=200 | grep -i error

# 3. Rollback agent if recent deployment
kubectl rollout undo deployment -n greenlang <agent>

# 4. Or fix the issue and redeploy
```

#### Scenario 2: Data quality issues

```bash
# 1. Identify validation errors
kubectl logs -n greenlang -l type=agent | \
  grep "validation_error" | jq -r '.field' | sort | uniq -c

# 2. Contact affected customers
# 3. Improve data validation feedback
# 4. Consider lenient mode temporarily
kubectl set env deployment -n greenlang -l type=agent \
  VALIDATION_MODE=lenient
```

#### Scenario 3: Intermittent dependency failures

```bash
# 1. Enable retry with backoff
kubectl set env deployment -n greenlang -l type=agent \
  RETRY_ENABLED=true \
  RETRY_MAX_ATTEMPTS=3 \
  RETRY_BACKOFF_MS=1000

# 2. Enable circuit breaker
kubectl set env deployment -n greenlang -l type=agent \
  CIRCUIT_BREAKER_ENABLED=true

# 3. Implement fallback behavior where possible
```

#### Scenario 4: Freeze non-critical changes

```bash
# 1. If error budget nearly exhausted, freeze deployments
# 2. Only allow critical bug fixes
# 3. Increase monitoring sensitivity
# 4. Plan for next month's budget
```

### Error Budget Management

**Proactive Measures:**

1. **Weekly error budget review**

   ```bash
   # Track weekly consumption
   # Alert at 50% and 75% of monthly budget
   ```

2. **Feature freeze triggers**

   - 75% budget consumed before day 20: Freeze non-critical changes
   - 90% budget consumed: Critical fixes only

3. **Budget allocation by team**

   - Platform: 30%
   - Backend: 40%
   - Data: 20%
   - External: 10%

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If error rate not decreasing |
| L3 | Engineering Manager | Slack DM | If >75% monthly budget consumed |

---

## SLA Compliance Dashboard

### Key Metrics to Monitor

```promql
# Real-time availability
(1 - (sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])))) * 100

# Hourly availability
(1 - (sum(rate(http_requests_total{status=~"5.."}[1h])) / sum(rate(http_requests_total[1h])))) * 100

# P99 latency (real-time)
histogram_quantile(0.99, sum(rate(gl_calculation_duration_seconds_bucket[5m])) by (le))

# Error rate
(sum(rate(gl_errors_total[5m])) / sum(rate(gl_calculations_total[5m]))) * 100

# Error budget remaining (requires recording rule)
# sla:error_budget_remaining:ratio
```

### Monthly SLA Report Query

```promql
# Monthly availability
(1 - (sum(increase(http_requests_total{status=~"5.."}[30d])) / sum(increase(http_requests_total[30d])))) * 100

# Monthly P99 latency
histogram_quantile(0.99, sum(increase(gl_calculation_duration_seconds_bucket[30d])) by (le))

# Monthly error rate
(sum(increase(gl_errors_total[30d])) / sum(increase(gl_calculations_total[30d]))) * 100
```

---

## Quick Reference Card

| Alert | Severity | SLA Target | Response Time | Escalation |
|-------|----------|------------|---------------|------------|
| SLAAvailabilityViolation | Critical | 99.9% | Immediate | VP Engineering |
| SLALatencyViolation | Critical | P99 <1s | 5 minutes | Eng Manager |
| ErrorBudgetBurning | Warning | <1% errors | 15 minutes | Backend Lead |

## SLA Incident Response Checklist

1. [ ] Acknowledge alert
2. [ ] Assess current SLA status
3. [ ] Identify root cause
4. [ ] Implement mitigation
5. [ ] Notify stakeholders
6. [ ] Monitor recovery
7. [ ] Document timeline
8. [ ] Calculate SLA impact
9. [ ] Customer communication (if needed)
10. [ ] Post-incident review
