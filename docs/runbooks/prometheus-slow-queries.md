# Prometheus Slow Queries

## Alert

**Alert Name:** `PrometheusSlowQueries`

**Severity:** Warning

**Threshold:** `histogram_quantile(0.99, rate(prometheus_engine_query_duration_seconds_bucket[5m])) > 30`

**Duration:** 5 minutes

---

## Description

This alert fires when Prometheus query execution time exceeds 30 seconds at the 99th percentile. Slow queries can cause:

1. **Dashboard timeouts** - Grafana panels fail to load
2. **Alert evaluation delays** - Rules not evaluated on time
3. **API request failures** - External integrations timing out
4. **User experience degradation** - Slow troubleshooting

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Dashboards slow or failing |
| **Data Impact** | Low | Data still collected, just slow to query |
| **SLA Impact** | Medium | Alert delays may cause SLA breaches |
| **Revenue Impact** | Low | Indirect impact through slower incident response |

---

## Symptoms

- Grafana dashboards timing out
- Recording rules not updating
- Alert manager not receiving alerts
- Prometheus UI queries taking >30 seconds
- API clients timing out

---

## Diagnostic Steps

### Step 1: Check Query Performance Metrics

```promql
# Query duration distribution
histogram_quantile(0.99, rate(prometheus_engine_query_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(prometheus_engine_query_duration_seconds_bucket[5m]))
histogram_quantile(0.50, rate(prometheus_engine_query_duration_seconds_bucket[5m]))

# Concurrent queries
prometheus_engine_queries

# Query queue length
prometheus_engine_queries_concurrent_max - prometheus_engine_queries
```

### Step 2: Identify Slow Queries

```bash
# Check Prometheus logs for slow queries
kubectl logs -n monitoring prometheus-server-0 | grep -i "slow"

# Check for query timeout errors
kubectl logs -n monitoring prometheus-server-0 | grep -i "timeout"
```

### Step 3: Check Recording Rule Evaluation

```promql
# Recording rule evaluation duration
prometheus_rule_group_duration_seconds

# Rules with slow evaluation
topk(10, prometheus_rule_evaluation_duration_seconds)

# Failed rule evaluations
increase(prometheus_rule_evaluation_failures_total[1h])
```

### Step 4: Check TSDB Performance

```promql
# Head block size (affects query speed)
prometheus_tsdb_head_series

# Block query latency
prometheus_tsdb_data_replay_duration_seconds

# Compaction status
prometheus_tsdb_compactions_total
prometheus_tsdb_compaction_duration_seconds
```

### Step 5: Check Resource Usage

```bash
# Check Prometheus CPU usage
kubectl top pods -n monitoring -l app.kubernetes.io/name=prometheus

# Check disk I/O
kubectl exec -n monitoring prometheus-server-0 -- iostat -x 1 5
```

### Step 6: Review Recent Query Load

```promql
# Query rate
rate(prometheus_engine_query_duration_seconds_count[5m])

# Queries by type
rate(prometheus_engine_query_duration_seconds_count[5m]) by (slice)
```

---

## Resolution Steps

### Scenario 1: Complex PromQL Queries

**Symptoms:** Specific dashboards or rules are slow

**Resolution:**

1. **Identify the slow queries from logs:**

```bash
kubectl logs -n monitoring prometheus-server-0 --since=1h | grep "slow query" | head -20
```

2. **Analyze and optimize the query:**

```promql
# SLOW: High cardinality aggregation without time range
sum by (pod) (rate(http_requests_total[5m]))

# FAST: Limit label dimensions
sum by (service) (rate(http_requests_total[5m]))

# SLOW: Regex matching all series
{__name__=~".+"}

# FAST: Specific metric names
http_requests_total or node_cpu_seconds_total
```

3. **Create recording rules for frequently used queries:**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: recording-rules
  namespace: monitoring
spec:
  groups:
    - name: performance.rules
      interval: 1m
      rules:
        # Pre-compute expensive aggregations
        - record: job:http_requests:rate5m
          expr: sum(rate(http_requests_total[5m])) by (job)

        - record: job:http_latency:p99
          expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job))
```

4. **Apply recording rules:**

```bash
kubectl apply -f recording-rules.yaml
```

### Scenario 2: High Cardinality Causing Slow Queries

**Symptoms:** All queries are slow, high series count

**Resolution:**

1. **Check current cardinality:**

```promql
prometheus_tsdb_head_series
```

2. **Identify high-cardinality metrics:**

```promql
topk(10, count by (__name__)({__name__=~".+"}))
```

3. **Add relabel configs to drop high-cardinality labels:**

```yaml
scrape_configs:
  - job_name: 'high-cardinality-job'
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'metric_with_uuid_labels'
        action: drop
      - source_labels: [trace_id]
        action: labeldrop
```

4. **Restart Prometheus to apply:**

```bash
kubectl rollout restart statefulset -n monitoring prometheus-server
```

### Scenario 3: Insufficient Resources

**Symptoms:** High CPU/memory usage on Prometheus

**Resolution:**

1. **Increase Prometheus resources:**

```yaml
prometheus:
  prometheusSpec:
    resources:
      requests:
        cpu: 1000m
        memory: 4Gi
      limits:
        cpu: 4000m
        memory: 16Gi
```

2. **Apply changes:**

```bash
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring -f values.yaml
```

### Scenario 4: Slow Storage I/O

**Symptoms:** High disk I/O wait, slow TSDB operations

**Resolution:**

1. **Check storage class performance:**

```bash
kubectl get pvc -n monitoring

# Verify storage class
kubectl get storageclass
```

2. **Migrate to faster storage (gp3):**

```yaml
prometheus:
  prometheusSpec:
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3-high-iops
          resources:
            requests:
              storage: 100Gi
```

3. **If using EBS, increase IOPS:**

```bash
aws ec2 modify-volume --volume-id vol-xxx --iops 10000
```

### Scenario 5: Recording Rules Need Optimization

**Symptoms:** Rule evaluation taking too long

**Resolution:**

1. **Identify slow rules:**

```promql
topk(10, prometheus_rule_group_duration_seconds)
```

2. **Increase rule evaluation interval for non-critical rules:**

```yaml
groups:
  - name: slow.rules
    interval: 5m  # Increase from 1m
    rules:
      - record: expensive:metric
        expr: ...
```

3. **Split large rule groups:**

```yaml
# Instead of one large group
groups:
  - name: all.rules
    rules:
      - ... # 50 rules

# Split into smaller groups
groups:
  - name: api.rules
    rules:
      - ... # 10 rules
  - name: agent.rules
    rules:
      - ... # 10 rules
```

### Scenario 6: Query Frontend Caching

**Symptoms:** Same queries repeatedly hitting Prometheus

**Resolution:**

1. **Deploy Thanos Query Frontend with caching:**

```yaml
# deployment/helm/thanos/values.yaml
queryFrontend:
  enabled: true
  config: |-
    type: IN-MEMORY
    config:
      max_size: 512MB
      max_size_items: 1000
      validity: 5m
```

2. **Route Grafana through Query Frontend:**

```yaml
# Grafana datasource
datasources:
  - name: Thanos
    type: prometheus
    url: http://thanos-query-frontend.monitoring.svc:9090
```

---

## Grafana Dashboard Optimization

### Dashboard Best Practices

1. **Use recording rules for complex queries:**

```promql
# In dashboard, use pre-computed metric
job:http_requests:rate5m

# Instead of
sum(rate(http_requests_total[5m])) by (job)
```

2. **Reduce time range for initial load:**
   - Default to 1h instead of 24h
   - Use "last 6h" for most dashboards

3. **Limit label selectors:**

```promql
# Specific labels
http_requests_total{job="greenlang-api", namespace="greenlang"}

# Not open-ended
http_requests_total{job=~".+"}
```

4. **Use `$__rate_interval` instead of fixed intervals:**

```promql
rate(http_requests_total[$__rate_interval])
```

---

## Emergency Actions

### If Queries Are Completely Timing Out

1. **Increase query timeout temporarily:**

```bash
# Edit Prometheus ConfigMap
kubectl edit configmap -n monitoring prometheus-server

# Add under global:
global:
  query_log_file: /prometheus/query.log
  # Increase timeout
```

2. **Kill expensive queries:**

```bash
# There's no direct way to kill queries, restart Prometheus
kubectl rollout restart statefulset -n monitoring prometheus-server
```

---

## Escalation Path

| Level | Condition | Contact |
|-------|-----------|---------|
| L1 | P99 query time >30s | On-call engineer |
| L2 | Queries timing out, dashboards unusable | Platform team lead |
| L3 | Prometheus unresponsive | Platform team + SRE |

---

## Prevention

1. **Review queries before production:**
   - Test query performance in staging
   - Use `EXPLAIN` in Prometheus UI (if available)

2. **Monitor query performance trends:**

```promql
# Alert on query performance degradation
histogram_quantile(0.95, rate(prometheus_engine_query_duration_seconds_bucket[5m])) > 10
```

3. **Regular recording rule optimization:**
   - Review rule evaluation times monthly
   - Pre-compute dashboard queries

4. **Capacity planning:**
   - Monitor TSDB head series growth
   - Plan for query load increases

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Prometheus Performance | https://grafana.greenlang.io/d/prometheus-performance |
| Query Analytics | https://grafana.greenlang.io/d/query-analytics |

---

## Related Alerts

- `PrometheusHighMemoryUsage`
- `PrometheusRuleEvaluationSlow`
- `ThanosQuerySlowQueries`

---

## References

- [PromQL Performance Tips](https://prometheus.io/docs/prometheus/latest/querying/performance/)
- [Recording Rules Best Practices](https://prometheus.io/docs/practices/rules/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)
