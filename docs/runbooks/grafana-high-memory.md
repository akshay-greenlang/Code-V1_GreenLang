# Grafana High Memory Usage

## Alert

**Alert Name:** `GrafanaHighMemory`

**Severity:** Warning

**Threshold:** `process_resident_memory_bytes{job="grafana"} > 1.5e9`

**Duration:** 10 minutes

**Related Alert:** `GrafanaDown` (fires if Grafana is OOMKilled and becomes unreachable)

---

## Description

This alert fires when Grafana memory usage exceeds 1.5GB (75% of the 2Gi production limit) for more than 10 minutes. Sustained high memory usage indicates that Grafana is at risk of being OOMKilled by the Kubernetes container runtime. Common causes include:

1. **Too many concurrent dashboard viewers** -- Each active session holds dashboard state in memory
2. **Complex queries with large result sets** -- PromQL/LogQL queries returning millions of data points
3. **Plugin memory leaks** -- Third-party plugins may not release memory properly
4. **Dashboard rendering backlog** -- Image rendering (Chromium) consumes significant memory
5. **Excessive dashboard count** -- Loading and indexing hundreds of dashboards

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Slow dashboard rendering, potential OOMKill causing outage |
| **Alerting Impact** | High | OOMKill interrupts Grafana-managed alert evaluation |
| **Data Impact** | None | Underlying data (Prometheus, Loki) is unaffected |
| **SLA Impact** | Medium | Risk of observability SLA breach if OOMKill occurs |
| **Revenue Impact** | Low | Indirect impact through degraded incident response |

---

## Symptoms

- Grafana UI feels sluggish, dashboards take longer to render
- Browser console shows timeout errors when loading panels
- Prometheus metric `process_resident_memory_bytes{job="grafana"}` exceeds 1.5GB
- Pod events may show previous OOMKill restarts
- Grafana API responses become slow (>2s latency)
- Image rendering requests fail or timeout

---

## Diagnostic Steps

### Step 1: Check Current Memory Usage

```bash
# Get Grafana pod memory usage
kubectl top pods -n monitoring -l app.kubernetes.io/name=grafana

# Check memory limits configured on the pod
kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].resources.limits.memory}{"\n"}{end}'

# Check for recent OOMKill events
kubectl describe pods -n monitoring -l app.kubernetes.io/name=grafana | grep -A 5 "Last State:"
```

### Step 2: Query Grafana Memory Metrics

```promql
# Current resident memory (bytes)
process_resident_memory_bytes{job="grafana"}

# Memory as percentage of container limit
process_resident_memory_bytes{job="grafana"}
/ on(pod) kube_pod_container_resource_limits{resource="memory", container="grafana"}
* 100

# Go heap memory in use
go_memstats_heap_inuse_bytes{job="grafana"}

# Go heap objects (indicates object churn)
go_memstats_heap_objects{job="grafana"}

# Memory growth rate over the last hour
deriv(process_resident_memory_bytes{job="grafana"}[1h])
```

### Step 3: Check Concurrent Users and Sessions

```promql
# Active sessions / concurrent users
grafana_stat_total_users

# HTTP request rate (proxy for active usage)
rate(grafana_http_request_total[5m])

# Active dashboard viewers (approximate from API calls)
rate(grafana_api_dashboard_get_milliseconds_count[5m])
```

```bash
# Check active WebSocket connections (live dashboards)
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:3000/api/admin/stats | python3 -m json.tool
```

### Step 4: Check Dashboard Count and Complexity

```promql
# Total number of dashboards
grafana_stat_total_dashboards

# Total number of alert rules
grafana_stat_total_alert_rules

# Dashboard load time distribution (slow dashboards use more memory)
histogram_quantile(0.99, rate(grafana_api_dashboard_get_milliseconds_bucket[5m]))
```

```bash
# List dashboards sorted by panel count (requires API token)
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -H "Authorization: Bearer $GRAFANA_API_TOKEN" \
     http://localhost:3000/api/search?type=dash-db | python3 -m json.tool | head -100
```

### Step 5: Check Plugin Memory Usage

```bash
# List installed plugins
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- grafana cli plugins ls

# Check plugin process memory (if running as separate processes)
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- ps aux --sort=-%mem | head -20
```

### Step 6: Check Rendering Service

```promql
# Rendering request rate
rate(grafana_rendering_request_total[5m])

# Active rendering requests
grafana_rendering_queue_size

# Rendering errors (failed renders hold memory)
rate(grafana_rendering_request_total{status="failure"}[5m])
```

### Step 7: Check Go Runtime Metrics

```promql
# GC pause duration (high GC pressure = memory pressure)
rate(go_gc_duration_seconds_sum{job="grafana"}[5m])
/ rate(go_gc_duration_seconds_count{job="grafana"}[5m])

# Number of goroutines (leak indicator)
go_goroutines{job="grafana"}

# Goroutine growth rate
deriv(go_goroutines{job="grafana"}[1h])
```

---

## Resolution Steps

### Scenario 1: Too Many Concurrent Dashboard Viewers

**Symptoms:** High `grafana_http_request_total` rate, many active sessions.

**Resolution:**

1. **Reduce auto-refresh intervals on heavy dashboards:**

```bash
# Identify dashboards with aggressive refresh intervals
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:3000/api/search?type=dash-db \
  | python3 -c "
import json, sys
dashboards = json.load(sys.stdin)
for d in dashboards[:20]:
    print(f'{d[\"uid\"]}: {d[\"title\"]}')"
```

2. **Set minimum refresh interval in grafana.ini:**

```yaml
# Update Helm values
grafana:
  grafana.ini:
    dashboards:
      min_refresh_interval: 30s  # Prevent <30s auto-refresh
```

3. **Apply the configuration change:**

```bash
helm upgrade grafana deployment/helm/grafana/ \
  -n monitoring \
  -f deployment/helm/grafana/values-prod.yaml
```

### Scenario 2: Complex Queries with Large Result Sets

**Symptoms:** High P99 dashboard load times, `go_memstats_heap_inuse_bytes` spikes during query execution.

**Resolution:**

1. **Identify slow dashboards:**

```promql
# P99 dashboard load time
histogram_quantile(0.99,
  rate(grafana_api_dashboard_get_milliseconds_bucket[5m])
)
```

2. **Optimize slow dashboards:**
   - Reduce time range (from 30d to 7d default)
   - Add `step` parameter to reduce data point density
   - Replace `rate()` over long ranges with recording rules
   - Use `max_data_points` in panel query options
   - Split multi-query panels into separate panels

3. **Add query caching:**

```yaml
# Enable query caching in grafana.ini
grafana:
  grafana.ini:
    caching:
      enabled: true
      backend: redis
      ttl: 300  # 5 minutes
```

### Scenario 3: Plugin Memory Leak

**Symptoms:** Memory grows steadily over time without correlation to usage patterns. `go_goroutines` may also be growing.

**Resolution:**

1. **Identify the leaking plugin:**

```bash
# Check plugin versions
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- grafana cli plugins ls
```

2. **Update or disable the plugin:**

```bash
# Update all plugins
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- grafana cli plugins update-all

# Or disable a specific plugin
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- grafana cli plugins remove <plugin-id>
```

3. **Restart Grafana to reclaim memory:**

```bash
kubectl rollout restart deployment -n monitoring grafana
```

### Scenario 4: Dashboard Rendering Backlog

**Symptoms:** `grafana_rendering_queue_size` is high, rendering failures detected.

**Resolution:**

1. **Limit concurrent rendering:**

```yaml
# Update Helm values
grafana:
  grafana.ini:
    rendering:
      concurrent_render_request_limit: 5  # Reduce from default 30
```

2. **Use external rendering service (offload memory):**

```yaml
# Deploy Grafana Image Renderer as a separate pod
grafana:
  grafana.ini:
    rendering:
      server_url: http://grafana-image-renderer.monitoring.svc:8081/render
      callback_url: http://grafana.monitoring.svc:3000/
```

### Scenario 5: Increase Memory Limits (Short-term)

**Symptoms:** All other optimizations are in place but memory is genuinely needed.

**Resolution:**

1. **Increase memory limits immediately:**

```bash
kubectl patch deployment -n monitoring grafana -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "grafana",
            "resources": {
              "requests": {
                "memory": "2Gi"
              },
              "limits": {
                "memory": "3Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

2. **Update Helm values for persistence:**

```yaml
# deployment/helm/grafana/values-prod.yaml
grafana:
  resources:
    requests:
      memory: 2Gi
      cpu: 500m
    limits:
      memory: 3Gi
      cpu: 2000m
```

3. **Apply via Helm:**

```bash
helm upgrade grafana deployment/helm/grafana/ \
  -n monitoring \
  -f deployment/helm/grafana/values-prod.yaml
```

---

## Emergency Actions

### If Grafana is OOMKilled and restarting in a loop

1. **Increase memory limit immediately to stop the crash loop:**

```bash
kubectl patch deployment -n monitoring grafana -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "grafana",
            "resources": {
              "limits": {
                "memory": "4Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

2. **Watch the rollout:**

```bash
kubectl rollout status deployment -n monitoring grafana --timeout=300s
```

3. **Verify health:**

```bash
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:3000/api/health
```

4. **Use direct Prometheus UI** while Grafana recovers:

```bash
kubectl port-forward -n monitoring svc/gl-prometheus-server 9090:9090
```

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Memory >1.5GB (75% of limit) | On-call engineer | 15 minutes |
| L2 | Memory >1.8GB or OOMKill once | Platform team lead | 15 minutes |
| L3 | Repeated OOMKills, Grafana outage | Platform team + SRE | 30 minutes |
| L4 | Memory leak confirmed, requires code fix | Platform team + Grafana Labs support | 1 hour |

---

## Prevention Measures

1. **Memory alerting at 75%:** GrafanaHighMemory alert fires at 1.5GB (75% of 2Gi limit), giving time to act before OOMKill
2. **Minimum refresh interval:** Set `min_refresh_interval: 30s` to prevent aggressive auto-refresh
3. **Dashboard panel limits:** Enforce maximum 25 panels per dashboard via team guidelines
4. **Query caching:** Enable Redis-backed query cache with 5-minute TTL
5. **External rendering:** Offload image rendering to a separate pod to isolate memory usage
6. **Dashboard audits:** Monthly review of dashboard count, panel count, and query complexity
7. **Plugin updates:** Keep plugins updated to latest versions; prefer official Grafana plugins
8. **Resource monitoring:** Track `process_resident_memory_bytes` and `go_memstats_heap_inuse_bytes` trends weekly

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Grafana Health | https://grafana.greenlang.io/d/grafana-health |
| Grafana Usage | https://grafana.greenlang.io/d/grafana-usage |
| Kubernetes Cluster | https://grafana.greenlang.io/d/kubernetes-cluster |

---

## Related Alerts

- `GrafanaDown` -- Server completely unreachable (fires if OOMKilled)
- `GrafanaBackendDBDown` -- PostgreSQL backend connectivity
- `GrafanaCacheHitRateLow` -- Cache efficiency (may correlate with memory pressure)
- `GrafanaDashboardLoadSlow` -- Dashboard performance (memory pressure causes slow loads)
- `GrafanaTooManyDashboards` -- Dashboard sprawl contributes to memory usage

---

## References

- [Grafana Resource Usage Guide](https://grafana.com/docs/grafana/latest/setup-grafana/configure-grafana/#resource-usage)
- [Grafana Performance Best Practices](https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/)
- [GreenLang Monitoring Architecture](../architecture/prometheus-stack.md)
- [Grafana Image Rendering Setup](https://grafana.com/docs/grafana/latest/setup-grafana/image-rendering/)
