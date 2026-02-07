# Prometheus High Memory Usage

## Alert

**Alert Name:** `PrometheusHighMemoryUsage`

**Severity:** Warning

**Threshold:** `process_resident_memory_bytes{job="prometheus"} / container_spec_memory_limit_bytes{container="prometheus"} > 0.8`

**Duration:** 5 minutes

---

## Description

This alert fires when Prometheus memory usage exceeds 80% of its configured limit. High memory usage typically indicates:

1. **High cardinality metrics** - Too many unique label combinations
2. **Large number of active time series** - More series than expected
3. **Long retention period** - Keeping too much data locally
4. **Complex queries** - Recording rules or queries consuming memory

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Slow queries, potential data gaps if OOMKilled |
| **Data Impact** | High | Risk of data loss if Prometheus restarts |
| **SLA Impact** | Medium | Alert delivery may be delayed |
| **Revenue Impact** | Low | Indirect impact through monitoring gaps |

---

## Symptoms

- Prometheus memory usage exceeds 80% of container limit
- Query response times increasing
- Grafana dashboards loading slowly
- `promtool` commands timing out
- Potential OOMKill events in pod events

---

## Diagnostic Steps

### Step 1: Check Current Memory Usage

```bash
# Get Prometheus pod memory usage
kubectl top pods -n monitoring -l app.kubernetes.io/name=prometheus

# Check memory limits
kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[*].spec.containers[*].resources.limits.memory}'
```

### Step 2: Query Prometheus for Memory Metrics

```promql
# Current resident memory
process_resident_memory_bytes{job="prometheus"}

# Memory usage as percentage of limit
process_resident_memory_bytes{job="prometheus"} /
container_spec_memory_limit_bytes{container="prometheus"} * 100

# Go heap memory
go_memstats_heap_inuse_bytes{job="prometheus"}
```

### Step 3: Check Cardinality (Primary Cause)

```promql
# Total active time series (high cardinality = high memory)
prometheus_tsdb_head_series

# Top 10 metrics by cardinality
topk(10, count by (__name__)({__name__=~".+"}))

# Top 10 label combinations by cardinality
topk(10, count by (__name__, job)({__name__=~".+"}))

# Labels with high cardinality
count by (__name__)(count by (__name__, instance)({__name__=~".+"})) > 100
```

### Step 4: Check Sample Ingestion Rate

```promql
# Samples ingested per second
rate(prometheus_tsdb_head_samples_appended_total[5m])

# Samples scraped per job
rate(scrape_samples_scraped[5m])

# Samples dropped due to high cardinality
rate(prometheus_target_scrapes_sample_out_of_bounds_total[5m])
```

### Step 5: Check for Memory Leaks

```promql
# Memory growth over time
increase(process_resident_memory_bytes{job="prometheus"}[1h])

# Heap objects
go_memstats_heap_objects{job="prometheus"}

# GC frequency
rate(go_gc_duration_seconds_count{job="prometheus"}[5m])
```

### Step 6: Check Pod Events

```bash
# Check for OOMKill events
kubectl describe pods -n monitoring -l app.kubernetes.io/name=prometheus | grep -A 5 "State:"

# Check pod events
kubectl get events -n monitoring --field-selector involvedObject.kind=Pod | grep prometheus
```

---

## Resolution Steps

### Scenario 1: High Cardinality Metrics

**Symptoms:** `prometheus_tsdb_head_series` is very high (>1M series)

**Resolution:**

1. **Identify high-cardinality metrics:**

```promql
# Find metrics with many unique label combinations
topk(20, count by (__name__)({__name__=~".+"}))
```

2. **Add relabeling rules to drop high-cardinality labels:**

```yaml
# Add to Prometheus scrape config
scrape_configs:
  - job_name: 'high-cardinality-job'
    relabel_configs:
      # Drop specific labels that cause high cardinality
      - source_labels: [__name__]
        regex: 'problematic_metric_.*'
        action: drop
    metric_relabel_configs:
      # Drop labels that create too many unique combinations
      - source_labels: [request_id]
        action: labeldrop
      - source_labels: [trace_id]
        action: labeldrop
```

3. **Apply configuration:**

```bash
# Reload Prometheus config (if using hot reload)
curl -X POST http://prometheus.monitoring.svc:9090/-/reload

# Or restart Prometheus
kubectl rollout restart statefulset -n monitoring prometheus-server
```

### Scenario 2: Too Many Targets

**Symptoms:** High number of scrape targets

**Resolution:**

1. **Check target count:**

```promql
# Count active targets
count(up)

# Targets per job
count by (job)(up)
```

2. **Reduce scrape frequency for non-critical targets:**

```yaml
scrape_configs:
  - job_name: 'low-priority-metrics'
    scrape_interval: 60s  # Increase from 15s
    scrape_timeout: 30s
```

3. **Use federation for large deployments:**

```yaml
# On central Prometheus, federate from edge Prometheus
scrape_configs:
  - job_name: 'federate'
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job="greenlang-api"}'
        - '{job="greenlang-agents"}'
    static_configs:
      - targets:
        - 'prometheus-edge-1:9090'
        - 'prometheus-edge-2:9090'
```

### Scenario 3: Need More Memory

**Symptoms:** Cardinality is reasonable but memory is still high

**Resolution:**

1. **Increase memory limits in Helm values:**

```yaml
# deployment/helm/prometheus-stack/values.yaml
prometheus:
  prometheusSpec:
    resources:
      requests:
        memory: 4Gi
      limits:
        memory: 12Gi  # Increase from 8Gi
```

2. **Apply changes:**

```bash
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  -f values.yaml
```

### Scenario 4: Reduce Local Retention

**Symptoms:** Old data consuming memory

**Resolution:**

1. **Reduce retention period:**

```yaml
prometheus:
  prometheusSpec:
    retention: 3d           # Reduce from 7d
    retentionSize: 30GB     # Reduce from 50GB
```

2. **Apply and verify:**

```bash
# Apply Helm changes
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  -f values.yaml

# Verify retention
kubectl exec -n monitoring prometheus-server-0 -- promtool tsdb list
```

### Scenario 5: Add Recording Rules

**Symptoms:** Complex queries consuming memory

**Resolution:**

1. **Create recording rules for frequently used queries:**

```yaml
# deployment/kubernetes/monitoring/prometheus-rules/recording-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: recording-rules
  namespace: monitoring
spec:
  groups:
    - name: greenlang.recording
      interval: 1m
      rules:
        # Pre-aggregate high-cardinality metrics
        - record: gl:api_request_rate:5m
          expr: sum(rate(gl_api_requests_total[5m])) by (service, method, status_code)

        - record: gl:api_latency_p99:5m
          expr: histogram_quantile(0.99, sum(rate(gl_api_request_duration_seconds_bucket[5m])) by (le, service))
```

2. **Apply recording rules:**

```bash
kubectl apply -f deployment/kubernetes/monitoring/prometheus-rules/recording-rules.yaml
```

---

## Emergency Actions

### If Prometheus is OOMKilled

1. **Scale down to single replica temporarily:**

```bash
kubectl scale statefulset -n monitoring prometheus-server --replicas=1
```

2. **Increase memory limits immediately:**

```bash
kubectl patch statefulset -n monitoring prometheus-server -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "prometheus",
            "resources": {
              "limits": {
                "memory": "16Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

3. **Restart with new limits:**

```bash
kubectl rollout restart statefulset -n monitoring prometheus-server
```

---

## Escalation Path

| Level | Condition | Contact |
|-------|-----------|---------|
| L1 | Memory >80% | On-call engineer |
| L2 | Memory >90% or OOMKill | Platform team lead |
| L3 | Repeated OOMKills, data loss | Platform team + SRE |

---

## Prevention

1. **Monitor cardinality trends:**

```promql
# Alert on rapid cardinality growth
increase(prometheus_tsdb_head_series[1h]) > 100000
```

2. **Set cardinality limits per job:**

```yaml
scrape_configs:
  - job_name: 'greenlang-api'
    sample_limit: 10000  # Max samples per scrape
    target_limit: 100    # Max targets
```

3. **Review new metrics before deployment:**
   - Check for high-cardinality labels (UUIDs, timestamps, IPs)
   - Ensure proper label naming conventions
   - Test with `promtool check rules` and `promtool tsdb analyze`

4. **Regular capacity planning:**
   - Monitor `prometheus_tsdb_head_series` weekly
   - Plan for 20% growth buffer
   - Document expected cardinality per service

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Prometheus Health | https://grafana.greenlang.io/d/prometheus-health |
| Cardinality Explorer | https://grafana.greenlang.io/d/cardinality-explorer |
| TSDB Status | https://grafana.greenlang.io/d/tsdb-status |

---

## Related Alerts

- `PrometheusStorageAlmostFull`
- `PrometheusTSDBCompactionsFailed`
- `PrometheusRuleEvaluationFailures`

---

## References

- [Prometheus Storage Documentation](https://prometheus.io/docs/prometheus/latest/storage/)
- [Cardinality Best Practices](https://www.robustperception.io/cardinality-is-key/)
- [GreenLang Metrics Guide](../development/metrics-guide.md)
