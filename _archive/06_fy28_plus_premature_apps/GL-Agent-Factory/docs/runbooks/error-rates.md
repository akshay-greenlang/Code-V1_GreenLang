# Error Rates Runbook

This runbook covers alerts related to error rates, calculation failures, and HTTP errors in the GreenLang platform.

---

## Table of Contents

- [HighErrorRate](#higherrorrate)
- [CalculationFailureSpike](#calculationfailurespike)
- [HighHTTP5xxRate](#highhttp5xxrate)
- [EFLookupFailures](#eflookupfailures)

---

## HighErrorRate

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | HighErrorRate |
| **Severity** | Critical |
| **Team** | Backend |
| **Evaluation Interval** | 30s |
| **For Duration** | 5m |
| **SLA Threshold** | 1% |

**PromQL Expression:**

```promql
(
  sum(rate(gl_errors_total[5m])) by (agent)
  /
  sum(rate(gl_calculations_total[5m])) by (agent)
) * 100 > 1
```

### Description

This alert fires when an agent's error rate exceeds 1% over a 5-minute window. This is a critical SLA metric - our commitment is to maintain error rates below 1% for all calculation operations.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | Calculations failing for 1+ in 100 requests |
| **Data Impact** | Medium | Failed calculations need to be retried |
| **SLA Impact** | Critical | Direct SLA violation at >1% error rate |
| **Revenue Impact** | High | Failed calculations delay customer reporting |

### Diagnostic Steps

1. **Identify the error type**

   ```bash
   # Query error breakdown by type
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_errors_total[5m]))by(agent,error_type)" | jq .

   # Common error types:
   # - validation_error: Input data issues
   # - calculation_error: Processing failures
   # - dependency_error: External service failures
   # - timeout_error: Operation timeouts
   ```

2. **Check error logs**

   ```bash
   # Get recent errors for the affected agent
   kubectl logs -n greenlang -l agent={{ $labels.agent }} --tail=500 | \
     grep -i "error\|exception\|failed" | tail -50

   # Search for specific error patterns
   kubectl logs -n greenlang -l agent={{ $labels.agent }} --tail=1000 | \
     grep -B5 -A10 "Traceback"
   ```

3. **Review recent deployments**

   ```bash
   # Check deployment history
   kubectl rollout history deployment -n greenlang {{ $labels.agent }}

   # Get current image version
   kubectl get deployment -n greenlang {{ $labels.agent }} \
     -o jsonpath='{.spec.template.spec.containers[0].image}'
   ```

4. **Check input data quality**

   ```bash
   # Query validation error rate
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_validation_errors_total{agent='{{ $labels.agent }}'}[5m]))by(field)" | jq .
   ```

5. **Verify external dependencies**

   ```bash
   # Check emission factor service
   curl -s "http://prometheus:9090/api/v1/query?query=probe_success{job='ef-service'}" | jq .

   # Check database health
   kubectl exec -n greenlang deploy/{{ $labels.agent }} -- \
     pg_isready -h postgres.greenlang.svc.cluster.local
   ```

6. **Correlate with load**

   ```bash
   # Check request rate
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_calculations_total{agent='{{ $labels.agent }}'}[5m]))" | jq .

   # Check if spike correlates with traffic increase
   curl -s "http://prometheus:9090/api/v1/query_range?query=sum(rate(gl_calculations_total{agent='{{ $labels.agent }}'}[5m]))&start=$(date -d '1 hour ago' +%s)&end=$(date +%s)&step=60" | jq .
   ```

### Resolution Steps

#### Scenario 1: Validation errors from bad input data

```bash
# 1. Identify the problematic input fields
kubectl logs -n greenlang -l agent={{ $labels.agent }} | \
  grep "validation_error" | jq -r '.field' | sort | uniq -c | sort -rn

# 2. Check if specific customer affected
kubectl logs -n greenlang -l agent={{ $labels.agent }} | \
  grep "validation_error" | jq -r '.customer_id' | sort | uniq -c

# 3. Contact customer if specific customer has data issues
# 4. If schema changed, update validation rules

# 5. Add temporary lenient validation (if appropriate)
kubectl set env deployment/{{ $labels.agent }} -n greenlang \
  VALIDATION_MODE=lenient
```

#### Scenario 2: Recent deployment caused regression

```bash
# 1. Check when errors started
curl -s "http://prometheus:9090/api/v1/query_range?query=sum(rate(gl_errors_total{agent='{{ $labels.agent }}'}[5m]))&start=$(date -d '2 hours ago' +%s)&end=$(date +%s)&step=60" | jq .

# 2. If correlated with deployment, rollback
kubectl rollout undo deployment -n greenlang {{ $labels.agent }}

# 3. Verify rollback completed
kubectl rollout status deployment -n greenlang {{ $labels.agent }}

# 4. Monitor error rate after rollback
watch "curl -s 'http://prometheus:9090/api/v1/query?query=sum(rate(gl_errors_total{agent=\"{{ $labels.agent }}\"}[5m]))/sum(rate(gl_calculations_total{agent=\"{{ $labels.agent }}\"}[5m]))*100' | jq '.data.result[0].value[1]'"
```

#### Scenario 3: External dependency failures

```bash
# 1. Identify failing dependency
kubectl logs -n greenlang -l agent={{ $labels.agent }} | \
  grep "dependency_error" | jq -r '.dependency' | sort | uniq -c

# 2. Check dependency health
# For emission factor service:
kubectl get pods -n greenlang -l app=ef-service
curl http://ef-service.greenlang.svc.cluster.local/health

# For database:
kubectl exec -n greenlang deploy/postgres -- pg_isready

# 3. If dependency down, escalate to appropriate team
# 4. Enable circuit breaker if available
kubectl set env deployment/{{ $labels.agent }} -n greenlang \
  CIRCUIT_BREAKER_ENABLED=true
```

#### Scenario 4: Resource exhaustion causing errors

```bash
# 1. Check pod resources
kubectl top pods -n greenlang -l agent={{ $labels.agent }}

# 2. If OOM or CPU throttling
kubectl describe pod -n greenlang -l agent={{ $labels.agent }} | grep -A5 "State:"

# 3. Scale horizontally
kubectl scale deployment -n greenlang {{ $labels.agent }} --replicas=5

# 4. Or increase resources
kubectl patch deployment -n greenlang {{ $labels.agent }} \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"requests":{"cpu":"500m","memory":"1Gi"},"limits":{"cpu":"2","memory":"4Gi"}}}]}}}}'
```

### Post-Resolution

1. **Verify error rate recovery**

   ```bash
   # Monitor error rate for 15 minutes
   watch -n 30 "curl -s 'http://prometheus:9090/api/v1/query?query=(sum(rate(gl_errors_total{agent=\"{{ $labels.agent }}\"}[5m]))/sum(rate(gl_calculations_total{agent=\"{{ $labels.agent }}\"}[5m])))*100' | jq '.data.result[0].value[1]'"
   ```

2. **Check for retry queue**

   ```bash
   # Check if failed calculations need retry
   kubectl exec -n greenlang deploy/{{ $labels.agent }} -- \
     python -c "from app.queue import get_retry_queue_size; print(get_retry_queue_size())"
   ```

3. **Document root cause** for post-incident review

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If not resolved in 15 minutes |
| L3 | Backend Lead | Direct page | If SLA breach continuing >30 min |
| L4 | Engineering Director | Phone call | If customer escalation |

### Related Dashboards

- [Agent Error Dashboard](https://grafana.greenlang.io/d/agent-errors)
- [SLA Compliance Dashboard](https://grafana.greenlang.io/d/sla-compliance)

---

## CalculationFailureSpike

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | CalculationFailureSpike |
| **Severity** | Warning |
| **Team** | Backend |
| **Evaluation Interval** | 30s |
| **For Duration** | 5m |
| **Threshold** | >50 failures in 10 minutes |

**PromQL Expression:**

```promql
increase(gl_calculations_total{status="failed"}[10m]) > 50
```

### Description

This alert fires when there is a sudden spike in calculation failures - more than 50 failures within a 10-minute window. This is a leading indicator that may precede a full error rate breach.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Some calculations failing |
| **Data Impact** | Low | Failed calculations can be retried |
| **SLA Impact** | Medium | May escalate to SLA violation |
| **Revenue Impact** | Medium | Delayed reports for affected customers |

### Diagnostic Steps

1. **Identify failure pattern**

   ```bash
   # Get failure count by agent
   curl -s "http://prometheus:9090/api/v1/query?query=increase(gl_calculations_total{status='failed'}[10m])" | jq .

   # Get failure reasons
   kubectl logs -n greenlang -l agent={{ $labels.agent }} --tail=500 | \
     grep "status.*failed" | jq -r '.reason' | sort | uniq -c
   ```

2. **Check if specific calculation type**

   ```bash
   # Query by calculation type
   curl -s "http://prometheus:9090/api/v1/query?query=increase(gl_calculations_total{status='failed',agent='{{ $labels.agent }}'}[10m])by(calculation_type)" | jq .
   ```

3. **Review recent changes**

   ```bash
   # Check deployment timestamps
   kubectl rollout history deployment -n greenlang {{ $labels.agent }}

   # Check config changes
   kubectl get configmap -n greenlang {{ $labels.agent }}-config -o yaml
   ```

4. **Check upstream data sources**

   ```bash
   # Verify emission factor database status
   kubectl exec -n greenlang deploy/ef-service -- \
     curl localhost:8000/health/detailed
   ```

### Resolution Steps

#### Scenario 1: Spike from batch processing

```bash
# 1. Check if large batch job running
kubectl logs -n greenlang -l agent={{ $labels.agent }} | \
  grep "batch_id" | tail -10

# 2. If batch has bad data, pause the batch
kubectl exec -n greenlang deploy/{{ $labels.agent }} -- \
  python -c "from app.batch import pause_batch; pause_batch('<batch_id>')"

# 3. Investigate batch data quality
# 4. Resume or retry after fixing data
```

#### Scenario 2: Calculation logic error

```bash
# 1. Identify specific calculation failing
kubectl logs -n greenlang -l agent={{ $labels.agent }} | \
  grep "calculation_error" | jq -r '.calculation_id' | head -5

# 2. Replay calculation in debug mode
kubectl exec -n greenlang deploy/{{ $labels.agent }} -- \
  python -c "from app.debug import replay_calculation; replay_calculation('<calc_id>', debug=True)"

# 3. If bug found, hotfix or rollback
```

#### Scenario 3: Data quality issue

```bash
# 1. Check validation statistics
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_validation_errors_total[10m]))by(field)" | jq .

# 2. Enable detailed validation logging
kubectl set env deployment/{{ $labels.agent }} -n greenlang \
  LOG_VALIDATION_DETAILS=true

# 3. Identify and contact affected customer
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If spike continues >15 min |
| L3 | Data Team | #data-oncall Slack | If data quality issue |

---

## HighHTTP5xxRate

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | HighHTTP5xxRate |
| **Severity** | Critical |
| **Team** | Backend |
| **Evaluation Interval** | 30s |
| **For Duration** | 5m |
| **Threshold** | >1% |

**PromQL Expression:**

```promql
(
  sum(rate(http_requests_total{status=~"5.."}[5m])) by (handler)
  /
  sum(rate(http_requests_total[5m])) by (handler)
) * 100 > 1
```

### Description

This alert fires when any HTTP endpoint has a 5xx error rate exceeding 1%. HTTP 5xx errors indicate server-side failures that users cannot resolve themselves.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | API consumers receiving server errors |
| **Data Impact** | Low | Requests not processed |
| **SLA Impact** | Critical | Directly impacts availability SLA |
| **Revenue Impact** | High | Integration failures for customers |

### Diagnostic Steps

1. **Identify the affected endpoint**

   ```bash
   # Get 5xx rate by handler
   curl -s "http://prometheus:9090/api/v1/query?query=(sum(rate(http_requests_total{status=~'5..'}[5m]))by(handler)/sum(rate(http_requests_total[5m]))by(handler))*100" | jq .
   ```

2. **Get specific error codes**

   ```bash
   # Breakdown by status code
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total{status=~'5..',handler='{{ $labels.handler }}'}[5m]))by(status)" | jq .

   # 500 = Internal Server Error
   # 502 = Bad Gateway
   # 503 = Service Unavailable
   # 504 = Gateway Timeout
   ```

3. **Review error logs**

   ```bash
   # Get 5xx error logs
   kubectl logs -n greenlang -l app=api-gateway --tail=500 | \
     grep -E '"status":\s*5[0-9]{2}' | tail -20

   # Get stack traces
   kubectl logs -n greenlang -l app=api-gateway --tail=1000 | \
     grep -B5 -A15 "Internal Server Error"
   ```

4. **Check upstream services**

   ```bash
   # If 502/504, check upstream service health
   kubectl get pods -n greenlang -l app={{ upstream_service }}

   # Check upstream response times
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(http_request_duration_seconds_bucket{handler='{{ $labels.handler }}'}[5m]))by(le))" | jq .
   ```

5. **Check for resource issues**

   ```bash
   # Check pod CPU/memory
   kubectl top pods -n greenlang -l app=api-gateway

   # Check for OOM kills
   kubectl get events -n greenlang --field-selector reason=OOMKilled
   ```

### Resolution Steps

#### Scenario 1: 500 Internal Server Error (Application Bug)

```bash
# 1. Get the stack trace
kubectl logs -n greenlang -l app=api-gateway --tail=1000 | \
  grep -A20 "Traceback"

# 2. If known bug, deploy hotfix
# 3. If unknown, rollback to previous version
kubectl rollout undo deployment -n greenlang api-gateway

# 4. Verify rollback
kubectl rollout status deployment -n greenlang api-gateway
```

#### Scenario 2: 502 Bad Gateway (Upstream Service Down)

```bash
# 1. Identify upstream service
kubectl logs -n greenlang -l app=api-gateway | \
  grep "502" | jq -r '.upstream'

# 2. Check upstream health
kubectl get pods -n greenlang -l app={{ upstream }}
kubectl describe pod -n greenlang -l app={{ upstream }}

# 3. Restart upstream if stuck
kubectl rollout restart deployment -n greenlang {{ upstream }}

# 4. If upstream cannot recover, enable circuit breaker
```

#### Scenario 3: 503 Service Unavailable (Overloaded)

```bash
# 1. Check request rate
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total{handler='{{ $labels.handler }}'}[1m]))" | jq .

# 2. Scale up
kubectl scale deployment -n greenlang api-gateway --replicas=10

# 3. Enable rate limiting if under attack
kubectl apply -f k8s/rate-limit-policy.yaml

# 4. Check HPA status
kubectl get hpa -n greenlang api-gateway
```

#### Scenario 4: 504 Gateway Timeout (Slow Upstream)

```bash
# 1. Check upstream latency
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(http_request_duration_seconds_bucket[5m]))by(handler,le))" | jq .

# 2. Increase timeout if appropriate
kubectl set env deployment/api-gateway -n greenlang \
  UPSTREAM_TIMEOUT=60s

# 3. Investigate slow upstream
# See latency runbook for detailed steps
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If not resolved in 10 minutes |
| L3 | Platform Team | #platform-oncall Slack | If infrastructure issue |
| L4 | Engineering Director | Phone call | If customer-facing >15 min |

---

## EFLookupFailures

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | EFLookupFailures |
| **Severity** | Warning |
| **Team** | Data |
| **Evaluation Interval** | 30s |
| **For Duration** | 5m |
| **Threshold** | >20 failures in 15 minutes |

**PromQL Expression:**

```promql
increase(gl_ef_lookups_total{status="failed"}[15m]) > 20
```

### Description

This alert fires when emission factor (EF) lookups are failing at an elevated rate. EF lookups are critical for carbon calculations - failures here will cascade into calculation failures.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Calculations requiring EF data will fail |
| **Data Impact** | Medium | Calculations cannot complete without EF |
| **SLA Impact** | Medium | Contributes to overall error rate |
| **Revenue Impact** | Medium | Delayed carbon calculations |

### Diagnostic Steps

1. **Identify failure source**

   ```bash
   # Get failures by source
   curl -s "http://prometheus:9090/api/v1/query?query=increase(gl_ef_lookups_total{status='failed'}[15m])by(source)" | jq .

   # Sources: ecoinvent, exiobase, custom, cache
   ```

2. **Check EF service health**

   ```bash
   # Service status
   kubectl get pods -n greenlang -l app=ef-service

   # Health check
   kubectl exec -n greenlang deploy/ef-service -- curl localhost:8000/health

   # Check logs
   kubectl logs -n greenlang -l app=ef-service --tail=100
   ```

3. **Verify external data source connectivity**

   ```bash
   # Check ecoinvent API
   kubectl exec -n greenlang deploy/ef-service -- \
     curl -v https://api.ecoinvent.org/health

   # Check database connection
   kubectl exec -n greenlang deploy/ef-service -- \
     pg_isready -h ef-postgres.greenlang.svc.cluster.local
   ```

4. **Check cache status**

   ```bash
   # Redis cache health
   kubectl exec -n greenlang deploy/redis -- redis-cli ping

   # Cache hit rate
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(gl_ef_lookups_total{cache='hit'}[15m]))/sum(rate(gl_ef_lookups_total[15m]))" | jq .
   ```

### Resolution Steps

#### Scenario 1: External API unavailable

```bash
# 1. Confirm external API is down
curl -v https://api.ecoinvent.org/health

# 2. Enable fallback to cache-only mode
kubectl set env deployment/ef-service -n greenlang \
  EF_FALLBACK_MODE=cache_only

# 3. Monitor cache hit rate
watch "curl -s 'http://prometheus:9090/api/v1/query?query=sum(rate(gl_ef_lookups_total{cache=\"hit\"}[5m]))/sum(rate(gl_ef_lookups_total[5m]))' | jq '.data.result[0].value[1]'"

# 4. Contact external provider if extended outage
```

#### Scenario 2: Database connection issues

```bash
# 1. Check database status
kubectl get pods -n greenlang -l app=ef-postgres

# 2. Check connection pool
curl -s "http://prometheus:9090/api/v1/query?query=pg_stat_activity_count{datname='emission_factors'}" | jq .

# 3. Restart connection pool
kubectl rollout restart deployment -n greenlang ef-service

# 4. If database down, escalate to DBA
```

#### Scenario 3: Cache failures

```bash
# 1. Check Redis status
kubectl get pods -n greenlang -l app=redis

# 2. Check Redis memory
kubectl exec -n greenlang deploy/redis -- redis-cli info memory

# 3. If memory full, flush stale keys
kubectl exec -n greenlang deploy/redis -- redis-cli --scan --pattern 'ef:*:stale:*' | \
  xargs redis-cli del

# 4. Increase Redis memory if needed
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Data Team | #data-oncall Slack | If EF service issues |
| L3 | External Vendor | Vendor support | If external API down |

---

## Quick Reference Card

| Alert | Severity | First Check | Quick Fix |
|-------|----------|-------------|-----------|
| HighErrorRate | Critical | Check error types in logs | Rollback if deployment-related |
| CalculationFailureSpike | Warning | Check batch jobs and data | Pause problematic batch |
| HighHTTP5xxRate | Critical | Check specific error codes | Scale or rollback |
| EFLookupFailures | Warning | Check EF service and cache | Enable cache-only mode |
