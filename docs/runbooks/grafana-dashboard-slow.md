# Grafana Dashboard Load Slow

## Alert

**Alert Name:** `GrafanaDashboardLoadSlow`

**Severity:** Warning

**Threshold:** `histogram_quantile(0.95, rate(grafana_api_dashboard_get_milliseconds_bucket[5m])) > 3000`

**Duration:** 5 minutes

**Related Alerts:** `GrafanaHighMemory` (memory pressure can cause slow loads), `GrafanaDataSourceUnreachable` (datasource errors cause panel timeouts)

---

## Description

This alert fires when the 95th percentile dashboard load time exceeds 3 seconds for more than 5 minutes. Slow dashboard loads directly impact the incident response workflow -- engineers rely on dashboards for real-time situational awareness during outages. Common causes include:

1. **Complex PromQL/LogQL queries** -- Queries scanning large time ranges or with high cardinality
2. **Too many panels per dashboard** -- Each panel fires a separate query to the datasource
3. **Slow datasource responses** -- Prometheus, Loki, or PostgreSQL responding slowly
4. **High concurrent user load** -- Many engineers viewing dashboards simultaneously
5. **Large query result sets** -- Queries returning millions of data points without aggregation

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Engineers wait >3s per dashboard during incident response |
| **Alerting Impact** | Low | Alert evaluation is separate from dashboard rendering |
| **Data Impact** | None | Underlying data is unaffected |
| **SLA Impact** | Medium | Degrades time-to-insight during incidents |
| **Revenue Impact** | Medium | Slower incident response extends MTTR |

---

## Symptoms

- Dashboards display loading spinners for more than 3 seconds
- Panel queries show "Timeout" or take excessively long
- Users report "Dashboard is slow" in Slack
- Grafana API `/api/dashboards/uid/:uid` response time is elevated
- Browser network tab shows long-running HTTP requests to Grafana API

---

## Diagnostic Steps

### Step 1: Identify the Overall Dashboard Load Time

```promql
# P50, P95, P99 dashboard load times
histogram_quantile(0.50, rate(grafana_api_dashboard_get_milliseconds_bucket[5m]))
histogram_quantile(0.95, rate(grafana_api_dashboard_get_milliseconds_bucket[5m]))
histogram_quantile(0.99, rate(grafana_api_dashboard_get_milliseconds_bucket[5m]))

# Dashboard request rate
rate(grafana_api_dashboard_get_milliseconds_count[5m])
```

### Step 2: Identify Slow Datasource Queries

```promql
# Datasource request duration by datasource type
histogram_quantile(0.95,
  rate(grafana_datasource_request_duration_seconds_bucket[5m])
) > 0

# Datasource request rate and errors
rate(grafana_datasource_request_total[5m])
rate(grafana_datasource_request_total{status="error"}[5m])

# Proxy request duration (queries proxied through Grafana)
histogram_quantile(0.95,
  rate(grafana_proxy_request_duration_seconds_bucket[5m])
)
```

### Step 3: Check Prometheus Query Performance

```promql
# Prometheus query duration (if Prometheus is the slow datasource)
histogram_quantile(0.95,
  rate(prometheus_engine_query_duration_seconds_bucket[5m])
)

# Prometheus query concurrency
prometheus_engine_queries

# Prometheus query samples loaded
prometheus_engine_query_samples_total
```

```bash
# Check Prometheus resource usage
kubectl top pods -n monitoring -l app.kubernetes.io/name=prometheus
```

### Step 4: Check Grafana Server Resources

```bash
# Check Grafana pod CPU and memory
kubectl top pods -n monitoring -l app.kubernetes.io/name=grafana

# Check if resource limits are being hit
kubectl describe pods -n monitoring -l app.kubernetes.io/name=grafana \
  | grep -A 10 "Limits:"
```

```promql
# Grafana CPU usage
rate(process_cpu_seconds_total{job="grafana"}[5m])

# Grafana memory usage
process_resident_memory_bytes{job="grafana"}

# Grafana goroutine count (high count = request queuing)
go_goroutines{job="grafana"}
```

### Step 5: Check Database Performance

```promql
# Grafana database query performance
grafana_database_conn_open
grafana_database_conn_max

# Database connection pool utilization
grafana_database_conn_open / grafana_database_conn_max
```

```bash
# Check PostgreSQL backend from within the cluster
kubectl run -n monitoring --rm -it --restart=Never pg-test \
  --image=postgres:15 -- \
  pg_isready -h grafana-db.monitoring.svc -p 5432
```

### Step 6: Check HTTP Request Distribution

```promql
# Request duration by handler (find slow handlers)
histogram_quantile(0.95,
  sum by (handler, le) (rate(grafana_http_request_duration_seconds_bucket[5m]))
)

# Request rate by handler (find high-traffic handlers)
sum by (handler) (rate(grafana_http_request_total[5m]))

# 5xx errors by handler
sum by (handler) (rate(grafana_http_request_total{status_code=~"5.."}[5m]))
```

### Step 7: Check Network Connectivity

```bash
# Test connectivity from Grafana to Prometheus
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -o /dev/null -w "%{time_total}s" \
     http://gl-prometheus-server.monitoring.svc:9090/api/v1/status/runtimeinfo

# Test connectivity to Loki
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -o /dev/null -w "%{time_total}s" \
     http://loki.monitoring.svc:3100/ready

# Check network policy rules
kubectl get networkpolicy -n monitoring | grep grafana
```

---

## Resolution Steps

### Scenario 1: Complex PromQL Queries

**Symptoms:** `prometheus_engine_query_duration_seconds` P95 is high, specific dashboards are slow.

**Resolution:**

1. **Identify the slow queries in Prometheus:**

```promql
# Slow query log (if enabled)
# Check Prometheus logs for slow queries
```

```bash
kubectl logs -n monitoring -l app.kubernetes.io/name=prometheus --tail=200 \
  | grep "slow query"
```

2. **Optimize the queries:**
   - Replace `rate(metric[30d])` with recording rules for long ranges
   - Use `sum by (relevant_labels)` instead of `sum by (all_labels)`
   - Avoid `{__name__=~".*"}` regex matches
   - Replace `count(metric) > 0` with `metric > 0` where possible
   - Use `max_over_time` instead of `rate` for gauge metrics

3. **Create recording rules for expensive queries:**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: grafana-recording-rules
  namespace: monitoring
spec:
  groups:
    - name: grafana.recording
      interval: 1m
      rules:
        # Pre-aggregate frequently used expensive queries
        - record: gl:api_request_rate:5m
          expr: sum(rate(gl_api_requests_total[5m])) by (service, method, status_code)

        - record: gl:api_latency_p95:5m
          expr: |
            histogram_quantile(0.95,
              sum(rate(gl_api_request_duration_seconds_bucket[5m])) by (le, service)
            )

        - record: gl:emissions_processing_rate:5m
          expr: sum(rate(gl_emissions_processed_total[5m])) by (agent_type, status)
```

```bash
kubectl apply -f deployment/kubernetes/monitoring/prometheus-rules/grafana-recording-rules.yaml
```

### Scenario 2: Too Many Panels Per Dashboard

**Symptoms:** Specific dashboards with 30+ panels are slow, while other dashboards load quickly.

**Resolution:**

1. **Identify dashboards with many panels:**

```bash
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -H "Authorization: Bearer $GRAFANA_API_TOKEN" \
     http://localhost:3000/api/search?type=dash-db \
  | python3 -c "
import json, sys
dbs = json.load(sys.stdin)
for d in dbs:
    uid = d.get('uid', 'unknown')
    title = d.get('title', 'unknown')
    print(f'{uid}: {title}')"
```

2. **Dashboard optimization best practices:**
   - Limit dashboards to 20-25 panels maximum
   - Split large dashboards into linked sub-dashboards
   - Use dashboard rows with collapsed sections (lazy loading)
   - Set default time range to 6h or 12h instead of 24h or 7d
   - Use `$__interval` variable instead of hardcoded step intervals
   - Collapse rows that are not immediately needed

3. **Enable lazy loading of panels:**

```yaml
# Update grafana.ini via Helm values
grafana:
  grafana.ini:
    panels:
      enable_alpha: false
    dashboards:
      default_home_dashboard_path: ""
```

### Scenario 3: Datasource Responding Slowly

**Symptoms:** `grafana_datasource_request_duration_seconds` P95 is high for a specific datasource.

**Resolution:**

1. **Check the datasource health:**

```bash
# For Prometheus datasource
kubectl top pods -n monitoring -l app.kubernetes.io/name=prometheus

# For Loki datasource
kubectl top pods -n monitoring -l app.kubernetes.io/name=loki

# For PostgreSQL datasource
kubectl exec -n monitoring --rm -it --restart=Never pg-perf \
  --image=postgres:15 -- \
  psql -h grafana-db.monitoring.svc -U grafana -c "SELECT count(*) FROM pg_stat_activity;"
```

2. **Increase datasource query timeout:**

```yaml
# In Grafana datasource provisioning
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://gl-prometheus-server.monitoring.svc:9090
    jsonData:
      timeInterval: 15s
      queryTimeout: 60s  # Increase from default 30s
      httpMethod: POST   # POST is more efficient for large queries
```

3. **Add query caching at the Grafana level:**

```yaml
grafana:
  grafana.ini:
    caching:
      enabled: true
      backend: redis
      ttl: 300
    redis:
      addr: redis.monitoring.svc:6379
```

### Scenario 4: High Concurrent User Load

**Symptoms:** Dashboard load times correlate with high `grafana_http_request_total` rate.

**Resolution:**

1. **Scale Grafana horizontally:**

```bash
kubectl scale deployment -n monitoring grafana --replicas=5
```

2. **Update HPA for automatic scaling:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: grafana-hpa
  namespace: monitoring
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: grafana
  minReplicas: 3
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
```

3. **Enable sticky sessions on the Ingress:**

```yaml
# Ingress annotation for session affinity
nginx.ingress.kubernetes.io/affinity: "cookie"
nginx.ingress.kubernetes.io/session-cookie-name: "grafana-sticky"
nginx.ingress.kubernetes.io/session-cookie-max-age: "3600"
```

### Scenario 5: Network Latency Between Grafana and Datasource

**Symptoms:** `curl` from Grafana pod to datasource shows high `time_total`.

**Resolution:**

1. **Check DNS resolution time:**

```bash
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- nslookup gl-prometheus-server.monitoring.svc
```

2. **Check network policies are not blocking:**

```bash
kubectl get networkpolicy -n monitoring
kubectl describe networkpolicy -n monitoring grafana-network-policy
```

3. **Verify Grafana and datasources are in the same availability zone:**

```bash
kubectl get pods -n monitoring -o wide | grep -E "grafana|prometheus"
```

---

## Emergency Actions

If dashboards are critically slow during an active incident:

1. **Use direct Prometheus UI** for metric queries:

```bash
kubectl port-forward -n monitoring svc/gl-prometheus-server 9090:9090
```

2. **Use Grafana Explore** instead of dashboards (simpler rendering, single query):

Navigate to `https://grafana.greenlang.io/explore`

3. **Scale Grafana replicas immediately:**

```bash
kubectl scale deployment -n monitoring grafana --replicas=5
```

4. **Query Loki directly** via `logcli`:

```bash
# Port-forward Loki
kubectl port-forward -n monitoring svc/loki 3100:3100

# Query logs directly
logcli query '{namespace="greenlang"}' --limit=100
```

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | P95 >3s for 5 min | On-call engineer | 15 minutes |
| L2 | P95 >10s or user complaints | Platform team lead | 15 minutes |
| L3 | All dashboards unusable | Platform team + SRE | 30 minutes |
| L4 | Datasource infrastructure issue | Platform team + DBA/SRE | 30 minutes |

---

## Prevention Measures

1. **Dashboard design guidelines:** Maximum 25 panels, use rows with lazy loading, default 6h time range
2. **Query optimization:** Use recording rules for expensive queries; avoid scanning >7d in real-time
3. **Query caching:** Enable Redis-backed query cache with 5-minute TTL for non-real-time dashboards
4. **Performance testing:** Load-test dashboards with k6 browser module before promoting to production
5. **Auto-refresh limits:** Set `min_refresh_interval: 30s` to prevent dashboard polling storms
6. **Dashboard review process:** Require team lead review for dashboards with >20 panels or >10 queries
7. **Datasource health monitoring:** Alert on datasource latency before it impacts dashboard performance
8. **Horizontal scaling:** Configure HPA to scale Grafana pods based on CPU and memory pressure

---

## Dashboard Optimization Checklist

When investigating a slow dashboard, verify the following:

- [ ] Time range is set to 6h or 12h (not 7d or 30d)
- [ ] Panel count is below 25
- [ ] No `{__name__=~".*"}` regex matchers in queries
- [ ] Recording rules exist for queries scanning >1h of data
- [ ] `$__interval` variable is used instead of hardcoded step
- [ ] Collapsed rows are used for below-the-fold panels
- [ ] Template variables do not query all values on load
- [ ] No `topk()` or `bottomk()` over high-cardinality metrics without filters
- [ ] `max_data_points` is configured in panel query options
- [ ] Dashboard uses `POST` method for Prometheus datasource

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Grafana Health | https://grafana.greenlang.io/d/grafana-health |
| Grafana Usage | https://grafana.greenlang.io/d/grafana-usage |
| Prometheus Health | https://grafana.greenlang.io/d/prometheus-health |

---

## Related Alerts

- `GrafanaHighMemory` -- Memory pressure causes slow query processing
- `GrafanaAPIErrors` -- 5xx errors may correlate with slow dashboard loads
- `GrafanaDBConnectionPoolExhausted` -- Database pool saturation slows API
- `GrafanaDataSourceUnreachable` -- Datasource failures cause panel timeouts
- `GrafanaCacheHitRateLow` -- Low cache efficiency increases datasource load

---

## References

- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/)
- [PromQL Performance Tips](https://prometheus.io/docs/prometheus/latest/querying/basics/#gotchas)
- [Grafana Query Caching](https://grafana.com/docs/grafana/latest/administration/data-source-management/#query-caching)
- [GreenLang Monitoring Architecture](../architecture/prometheus-stack.md)
- [GreenLang Metrics Guide](../development/metrics-guide.md)
