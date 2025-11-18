# GL-002 BoilerEfficiencyOptimizer - Monitoring Guide

## Overview

This document provides comprehensive monitoring setup, configuration, and operational guidance for GL-002 BoilerEfficiencyOptimizer. The monitoring infrastructure provides production-grade observability with Prometheus metrics, Grafana dashboards, and intelligent alerting.

## Table of Contents

1. [Architecture](#architecture)
2. [Metrics](#metrics)
3. [Dashboards](#dashboards)
4. [Alerting](#alerting)
5. [Setup Instructions](#setup-instructions)
6. [Troubleshooting](#troubleshooting)
7. [Runbooks](#runbooks)

---

## Architecture

### Monitoring Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    GL-002 Application                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FastAPI App (Port 8000)                             │  │
│  │  └─ /metrics endpoint (Prometheus format)            │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ Scrape (30s interval)
                             │
                    ┌────────▼────────┐
                    │   Prometheus    │
                    │  (Time Series   │
                    │    Database)    │
                    └────────┬────────┘
                             │
                  ┌──────────┼──────────┐
                  │          │          │
         ┌────────▼───┐  ┌──▼──────┐  ┌▼───────────┐
         │  Grafana   │  │ Alert   │  │ Long-term  │
         │ Dashboards │  │ Manager │  │  Storage   │
         └────────────┘  └─────────┘  └────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              ┌─────▼──┐  ┌───▼────┐  ┌▼────────┐
              │ Slack  │  │ Email  │  │PagerDuty│
              └────────┘  └────────┘  └─────────┘
```

### Components

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and grouping
- **ServiceMonitor**: Kubernetes Prometheus Operator integration

---

## Metrics

### Metric Categories

#### 1. HTTP Request Metrics

```promql
# Total requests
gl_002_http_requests_total{method, endpoint, status}

# Request duration (histogram)
gl_002_http_request_duration_seconds{method, endpoint}

# Request/response size
gl_002_http_request_size_bytes{method, endpoint}
gl_002_http_response_size_bytes{method, endpoint}
```

**Use Cases:**
- Monitor API performance
- Track request throughput
- Identify slow endpoints

#### 2. Optimization Metrics

```promql
# Optimization requests
gl_002_optimization_requests_total{strategy, status}

# Optimization duration (histogram)
gl_002_optimization_duration_seconds{strategy}

# Efficiency improvement (histogram)
gl_002_optimization_efficiency_improvement_percent{strategy}

# Cost savings (histogram)
gl_002_optimization_cost_savings_usd_per_hour{fuel_type}

# Emissions reduction (gauge)
gl_002_optimization_emissions_reduction_kg_hr{boiler_id}
```

**Use Cases:**
- Calculate ROI
- Track efficiency gains
- Measure carbon impact

#### 3. Boiler Operating Metrics

```promql
# Efficiency
gl_002_boiler_efficiency_percent{boiler_id, fuel_type}

# Flow rates
gl_002_boiler_steam_flow_kg_hr{boiler_id}
gl_002_boiler_fuel_flow_kg_hr{boiler_id, fuel_type}

# Temperature & pressure
gl_002_boiler_combustion_temperature_c{boiler_id}
gl_002_boiler_pressure_bar{boiler_id}

# Load
gl_002_boiler_load_percent{boiler_id}
gl_002_boiler_excess_air_percent{boiler_id}
```

**Use Cases:**
- Real-time boiler monitoring
- Performance trending
- Anomaly detection

#### 4. Emissions Metrics

```promql
# Emissions concentrations
gl_002_emissions_co2_kg_hr{boiler_id}
gl_002_emissions_nox_ppm{boiler_id}
gl_002_emissions_co_ppm{boiler_id}
gl_002_emissions_so2_ppm{boiler_id}

# Compliance
gl_002_emissions_compliance_violations_total{boiler_id, pollutant, limit_type}
gl_002_emissions_compliance_status{boiler_id}  # 1=compliant, 0=violation
```

**Use Cases:**
- Regulatory compliance monitoring
- Emissions trending
- Violation alerting

#### 5. Cache Metrics

```promql
# Hit/miss tracking
gl_002_cache_hits_total{cache_key_pattern}
gl_002_cache_misses_total{cache_key_pattern}
gl_002_cache_evictions_total{cache_key_pattern}

# Cache hit rate calculation
100 * sum(rate(gl_002_cache_hits_total[5m]))
/ (sum(rate(gl_002_cache_hits_total[5m])) + sum(rate(gl_002_cache_misses_total[5m])))
```

**Use Cases:**
- Optimize cache configuration
- Identify cache bottlenecks
- Performance tuning

#### 6. Database Metrics

```promql
# Connection pool
gl_002_db_connection_pool_size

# Query performance (histogram)
gl_002_db_query_duration_seconds{query_type}

# Errors
gl_002_db_query_errors_total{query_type, error_type}
```

**Use Cases:**
- Database performance monitoring
- Connection pool tuning
- Query optimization

#### 7. System Metrics

```promql
# Uptime
gl_002_system_uptime_seconds

# Memory usage
gl_002_system_memory_usage_bytes{type}  # type: rss, vms, heap

# CPU usage
gl_002_system_cpu_usage_percent

# Disk usage
gl_002_system_disk_usage_bytes{mount_point}
```

**Use Cases:**
- Resource utilization tracking
- Capacity planning
- Performance monitoring

#### 8. Business Metrics

```promql
# Annual projections
gl_002_optimization_annual_savings_usd{boiler_id}
gl_002_optimization_annual_emissions_reduction_tons{boiler_id}

# Payback period
gl_002_optimization_payback_period_months{recommendation_type}
```

**Use Cases:**
- Executive reporting
- ROI calculation
- Business case validation

---

## Dashboards

### 1. Executive Dashboard (KPIs & ROI)

**Location:** `monitoring/grafana/executive_dashboard.json`

**Panels:**
- Annual Cost Savings (USD)
- Annual CO2 Reduction (Tons)
- Average Efficiency Improvement (%)
- Optimization Success Rate (%)
- Cost Savings Trend (24h)
- Emissions Reduction Trend (24h)
- Boiler Efficiency by Plant
- Emissions Compliance Status
- ROI Metrics Summary Table
- Optimization Requests by Strategy
- Carbon Impact Over Time

**Refresh Rate:** 30s

**Target Audience:** Executives, Product Managers, Business Analysts

**Access Dashboard:**
```bash
https://grafana.greenlang.io/d/gl-002-executive
```

### 2. Operations Dashboard (Health & Performance)

**Location:** `monitoring/grafana/operations_dashboard.json`

**Panels:**
- System Uptime
- HTTP Request Rate
- Error Rate (%)
- P95 Latency
- Request Latency Distribution (p50, p95, p99)
- Request Rate by Endpoint
- CPU Usage
- Memory Usage
- Database Query Performance
- Database Connection Pool
- External API Latency
- Error Breakdown
- Optimization Duration
- Active Boilers Status Table

**Refresh Rate:** 10s

**Target Audience:** DevOps Engineers, SREs, Operations

**Access Dashboard:**
```bash
https://grafana.greenlang.io/d/gl-002-operations
```

### 3. Agent Dashboard (Tools & Cache Performance)

**Location:** `monitoring/grafana/agent_dashboard.json`

**Panels:**
- Cache Hit Rate (%)
- Tool Execution Rate
- Tool Success Rate (%)
- Cache Evictions
- Cache Performance Over Time
- Tool Execution Latency by Strategy
- Cache Hit Rate by Pattern
- Tool Execution Count by Strategy
- Tool Failure Rate by Strategy
- Optimization Efficiency Improvement Distribution
- Tool Execution Statistics Table
- Cache Eviction Rate Over Time
- Boiler Operations Heatmap

**Refresh Rate:** 10s

**Target Audience:** Developers, ML Engineers, Performance Engineers

**Access Dashboard:**
```bash
https://grafana.greenlang.io/d/gl-002-agent
```

### 4. Quality Dashboard (Determinism & Accuracy)

**Location:** `monitoring/grafana/quality_dashboard.json`

**Panels:**
- Determinism Score (%)
- Efficiency Accuracy (vs Target)
- Emissions Compliance Rate (%)
- Calculation Errors (24h)
- Efficiency Prediction vs Actual
- Optimization Consistency (Same Input)
- Emissions Compliance Violations
- Boiler Efficiency Distribution
- Optimization Repeatability Score
- Tool Execution Variance
- Accuracy Metrics by Boiler Table
- Cost Savings Accuracy vs Prediction
- Quality Metrics Summary

**Refresh Rate:** 30s

**Target Audience:** Quality Engineers, Data Scientists, Compliance Officers

**Access Dashboard:**
```bash
https://grafana.greenlang.io/d/gl-002-quality
```

---

## Alerting

### Alert Severity Levels

| Severity | Response Time | Escalation | Examples |
|----------|--------------|------------|----------|
| **CRITICAL** | Immediate | PagerDuty | Agent down, high error rate, compliance violation |
| **WARNING** | 15 minutes | Slack | Performance degradation, low cache hit rate |
| **INFO** | 1 hour | Email | Low cost savings, business metric alerts |

### Critical Alerts

#### 1. GL002AgentUnavailable
```yaml
Condition: up{job="gl-002-boiler-optimizer"} == 0
Duration: 1 minute
Action: Immediate investigation, check pod logs, restart if needed
```

#### 2. GL002HighErrorRate
```yaml
Condition: Error rate > 5%
Duration: 2 minutes
Action: Check application logs, investigate root cause
```

#### 3. GL002DeterminismFailure
```yaml
Condition: Optimization failure rate > 1%
Duration: 5 minutes
Action: Review calculation logic, check input data quality
```

#### 4. GL002EmissionsComplianceViolation
```yaml
Condition: Compliance violation detected
Duration: 1 minute
Action: Immediate notification to compliance team, regulatory reporting
```

#### 5. GL002DatabaseConnectionFailure
```yaml
Condition: DB connection pool == 0
Duration: 1 minute
Action: Check database connectivity, scale connection pool
```

#### 6. GL002HighMemoryUsage
```yaml
Condition: Memory > 4GB
Duration: 5 minutes
Action: Check for memory leaks, restart pod, scale horizontally
```

#### 7. GL002OptimizationTimeout
```yaml
Condition: p95 latency > 10s
Duration: 5 minutes
Action: Investigate slow calculations, optimize algorithms
```

### Warning Alerts

#### 1. GL002PerformanceDegradation
```yaml
Condition: Latency increased > 10% vs baseline
Duration: 5 minutes
Action: Review recent changes, check resource utilization
```

#### 2. GL002LowCacheHitRate
```yaml
Condition: Cache hit rate < 70%
Duration: 10 minutes
Action: Review cache configuration, increase TTL or size
```

#### 3. GL002HighCacheEvictionRate
```yaml
Condition: Evictions > 1/sec
Duration: 5 minutes
Action: Increase cache size, review eviction policy
```

#### 4. GL002EfficiencyBelowTarget
```yaml
Condition: Boiler efficiency < 80%
Duration: 15 minutes
Action: Inspect boiler operations, recommend maintenance
```

### Alert Configuration

**File:** `monitoring/alerts/prometheus_rules.yaml`

**Deployment:**
```bash
kubectl apply -f monitoring/alerts/prometheus_rules.yaml
```

**Verify Alerts:**
```bash
# Check if alerts are loaded
kubectl get prometheusrule -n greenlang

# View alert status in Prometheus
curl http://prometheus:9090/api/v1/rules
```

---

## Setup Instructions

### Prerequisites

- Kubernetes cluster (v1.24+)
- Prometheus Operator installed
- Grafana installed
- kubectl access to cluster

### Step 1: Deploy Application with Metrics

Ensure your application exposes metrics on `/metrics` endpoint:

```python
# In your FastAPI app
from prometheus_client import make_asgi_app

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Step 2: Deploy ServiceMonitor

```bash
# Apply ServiceMonitor for automatic discovery
kubectl apply -f deployment/servicemonitor.yaml

# Verify ServiceMonitor is created
kubectl get servicemonitor -n greenlang gl-002-boiler-optimizer
```

### Step 3: Deploy Alerting Rules

```bash
# Apply Prometheus alerting rules
kubectl apply -f monitoring/alerts/prometheus_rules.yaml

# Verify rules are loaded
kubectl get prometheusrule -n greenlang gl-002-alerts
```

### Step 4: Import Grafana Dashboards

**Option A: Manual Import**
1. Open Grafana UI
2. Navigate to Dashboards > Import
3. Upload JSON files from `monitoring/grafana/`
4. Select Prometheus datasource
5. Click Import

**Option B: Automated Import**
```bash
# Create ConfigMap with dashboards
kubectl create configmap gl-002-grafana-dashboards \
  --from-file=monitoring/grafana/ \
  -n greenlang \
  --dry-run=client -o yaml | kubectl apply -f -

# Label ConfigMap for auto-discovery
kubectl label configmap gl-002-grafana-dashboards \
  grafana_dashboard=1 -n greenlang
```

### Step 5: Configure AlertManager

```yaml
# alertmanager-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-config
  namespace: greenlang
type: Opaque
stringData:
  alertmanager.yaml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'
      pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

    route:
      group_by: ['alertname', 'severity']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'default'
      routes:
        - match:
            severity: critical
          receiver: 'pagerduty'
          continue: true

        - match:
            severity: warning
          receiver: 'slack'

        - match:
            severity: info
          receiver: 'email'

    receivers:
      - name: 'default'
        slack_configs:
          - channel: '#greenlang-alerts'
            title: 'GL-002 Alert: {{ .GroupLabels.alertname }}'
            text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

      - name: 'pagerduty'
        pagerduty_configs:
          - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
            description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

      - name: 'slack'
        slack_configs:
          - channel: '#greenlang-ops'
            title: 'GL-002 Warning: {{ .GroupLabels.alertname }}'
            text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

      - name: 'email'
        email_configs:
          - to: 'team@greenlang.io'
            from: 'alerts@greenlang.io'
            smarthost: 'smtp.gmail.com:587'
            auth_username: 'alerts@greenlang.io'
            auth_password: 'YOUR_EMAIL_PASSWORD'
```

Apply configuration:
```bash
kubectl apply -f alertmanager-config.yaml
```

### Step 6: Verify Monitoring Stack

```bash
# Check if metrics are being scraped
kubectl port-forward -n greenlang svc/prometheus 9090:9090

# Open browser: http://localhost:9090/targets
# Verify gl-002-boiler-optimizer target is UP

# Check Grafana dashboards
kubectl port-forward -n greenlang svc/grafana 3000:3000

# Open browser: http://localhost:3000
# Login and verify dashboards are imported
```

---

## Troubleshooting

### Metrics Not Appearing

**Symptom:** Prometheus not scraping metrics

**Debug Steps:**
```bash
# 1. Check if /metrics endpoint is accessible
kubectl port-forward -n greenlang pod/gl-002-xxx 8000:8000
curl http://localhost:8000/metrics

# 2. Verify ServiceMonitor is created
kubectl get servicemonitor -n greenlang

# 3. Check Prometheus targets
kubectl port-forward -n greenlang svc/prometheus 9090:9090
# Visit http://localhost:9090/targets

# 4. Check Prometheus logs
kubectl logs -n greenlang -l app=prometheus -f

# 5. Verify service labels match ServiceMonitor selector
kubectl get svc -n greenlang gl-002-metrics -o yaml
```

**Resolution:**
- Ensure service has correct labels
- Verify port name matches ServiceMonitor
- Check network policies allow Prometheus to scrape

### Dashboard Not Loading Data

**Symptom:** Grafana dashboard shows "No Data"

**Debug Steps:**
```bash
# 1. Test query directly in Prometheus
kubectl port-forward -n greenlang svc/prometheus 9090:9090
# Visit http://localhost:9090/graph
# Try query: gl_002_http_requests_total

# 2. Check Grafana datasource configuration
# Grafana UI > Configuration > Data Sources > Prometheus
# Click "Test" button

# 3. Verify metric names match dashboard queries
curl http://localhost:8000/metrics | grep gl_002

# 4. Check time range in dashboard
# Ensure time range overlaps with data availability
```

**Resolution:**
- Verify Prometheus datasource URL in Grafana
- Check metric names match exactly
- Adjust time range to when data exists

### Alerts Not Firing

**Symptom:** Expected alerts not triggering

**Debug Steps:**
```bash
# 1. Check if PrometheusRule is loaded
kubectl get prometheusrule -n greenlang

# 2. View rule status in Prometheus
kubectl port-forward -n greenlang svc/prometheus 9090:9090
# Visit http://localhost:9090/rules

# 3. Test alert query manually
# Copy alert expression and run in Prometheus UI

# 4. Check AlertManager logs
kubectl logs -n greenlang -l app=alertmanager -f

# 5. Verify AlertManager configuration
kubectl get secret alertmanager-config -n greenlang -o yaml
```

**Resolution:**
- Ensure PrometheusRule has correct labels
- Verify alert expressions return data
- Check AlertManager routing configuration

### High Cardinality Issues

**Symptom:** Prometheus consuming excessive memory/disk

**Debug Steps:**
```bash
# 1. Identify high-cardinality metrics
kubectl port-forward -n greenlang svc/prometheus 9090:9090

# Run in Prometheus UI:
topk(10, count by (__name__)({__name__=~".+"}))

# 2. Check metric label cardinality
count by (__name__, boiler_id)({__name__=~"gl_002_.*"})

# 3. Review metric relabeling rules in ServiceMonitor
kubectl get servicemonitor gl-002-boiler-optimizer -o yaml
```

**Resolution:**
- Drop unnecessary labels in relabeling
- Aggregate high-cardinality metrics
- Implement metric retention policies

---

## Runbooks

### Runbook: Agent Unavailable

**Alert:** GL002AgentUnavailable

**Severity:** CRITICAL

**Steps:**
1. **Check pod status**
   ```bash
   kubectl get pods -n greenlang -l app=gl-002-boiler-optimizer
   kubectl describe pod <pod-name> -n greenlang
   ```

2. **Check logs**
   ```bash
   kubectl logs -n greenlang <pod-name> --tail=100
   kubectl logs -n greenlang <pod-name> --previous  # If pod restarted
   ```

3. **Check resource limits**
   ```bash
   kubectl top pod <pod-name> -n greenlang
   ```

4. **Restart pod if needed**
   ```bash
   kubectl delete pod <pod-name> -n greenlang
   ```

5. **Scale deployment if issue persists**
   ```bash
   kubectl scale deployment gl-002-boiler-optimizer --replicas=3 -n greenlang
   ```

### Runbook: High Error Rate

**Alert:** GL002HighErrorRate

**Severity:** CRITICAL

**Steps:**
1. **Identify error types**
   ```promql
   sum(rate(gl_002_http_requests_total{status="error"}[5m])) by (endpoint)
   ```

2. **Check application logs**
   ```bash
   kubectl logs -n greenlang -l app=gl-002-boiler-optimizer --tail=500 | grep ERROR
   ```

3. **Review recent deployments**
   ```bash
   kubectl rollout history deployment/gl-002-boiler-optimizer -n greenlang
   ```

4. **Rollback if recent deployment caused issues**
   ```bash
   kubectl rollout undo deployment/gl-002-boiler-optimizer -n greenlang
   ```

5. **Check external dependencies**
   ```bash
   # Test database connectivity
   kubectl run -it --rm debug --image=postgres:14 --restart=Never -- \
     psql -h <db-host> -U <user> -d <database> -c "SELECT 1"

   # Test Redis connectivity
   kubectl run -it --rm debug --image=redis:7 --restart=Never -- \
     redis-cli -h <redis-host> PING
   ```

### Runbook: Low Cache Hit Rate

**Alert:** GL002LowCacheHitRate

**Severity:** WARNING

**Steps:**
1. **Analyze cache patterns**
   ```promql
   100 * sum(rate(gl_002_cache_hits_total[5m])) by (cache_key_pattern)
   / (sum(rate(gl_002_cache_hits_total[5m])) by (cache_key_pattern)
   + sum(rate(gl_002_cache_misses_total[5m])) by (cache_key_pattern))
   ```

2. **Check cache eviction rate**
   ```promql
   sum(rate(gl_002_cache_evictions_total[5m]))
   ```

3. **Increase cache size (if evictions are high)**
   - Edit `config.py` and increase `cache_max_size`
   - Redeploy application

4. **Increase TTL (if data is stable)**
   - Edit `config.py` and increase `cache_ttl_seconds`
   - Redeploy application

5. **Monitor after changes**
   - Watch cache hit rate in Agent Dashboard

### Runbook: Compliance Violation

**Alert:** GL002EmissionsComplianceViolation

**Severity:** CRITICAL

**Steps:**
1. **Identify violated pollutant**
   ```promql
   gl_002_emissions_compliance_violations_total{boiler_id="xxx"}
   ```

2. **Check current emissions levels**
   ```promql
   gl_002_emissions_nox_ppm{boiler_id="xxx"}
   gl_002_emissions_co_ppm{boiler_id="xxx"}
   gl_002_emissions_so2_ppm{boiler_id="xxx"}
   ```

3. **Notify compliance team**
   - Send email to compliance@greenlang.io
   - Include boiler ID, pollutant, concentration

4. **Adjust optimization strategy**
   - Prioritize emissions reduction over efficiency
   - Reduce load on non-compliant boiler

5. **Document incident**
   - Create incident report
   - Record remediation actions

---

## Performance Tuning

### Prometheus Retention

```yaml
# Adjust retention period
prometheus:
  retention: 30d
  retentionSize: 50GB

# Configure remote write for long-term storage
prometheus:
  remoteWrite:
    - url: https://prometheus-remote-storage.greenlang.io/api/v1/write
      queueConfig:
        capacity: 10000
        maxShards: 50
```

### Grafana Performance

```yaml
# Enable query caching
grafana:
  datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus:9090
      jsonData:
        cacheLevel: 'High'
        timeInterval: '30s'
```

### Metric Optimization

```python
# Use metric relabeling to reduce cardinality
metric_relabel_configs:
  # Drop high-cardinality labels
  - source_labels: [__name__]
    regex: 'gl_002_http_request_size_bytes'
    action: drop

  # Aggregate by hour instead of per-request
  - source_labels: [timestamp]
    target_label: hour
    replacement: '${1}:00:00'
```

---

## Integration Examples

### Python Integration

```python
from monitoring.metrics import MetricsCollector

# Update boiler metrics
MetricsCollector.update_boiler_metrics(
    boiler_id="BOILER-001",
    metrics={
        "efficiency_percent": 87.5,
        "steam_flow_kg_hr": 15000,
        "fuel_flow_kg_hr": 1200,
        "combustion_temperature_c": 1150,
        "excess_air_percent": 15.2,
        "pressure_bar": 45,
        "load_percent": 82,
        "fuel_type": "natural_gas"
    }
)

# Record optimization result
MetricsCollector.record_optimization_result(
    boiler_id="BOILER-001",
    fuel_type="natural_gas",
    strategy="fuel_efficiency",
    result={
        "improvement_percent": 3.2,
        "cost_savings_usd_hr": 125.50,
        "emissions_reduction_kg_hr": 45.2,
        "annual_savings_usd": 1099380,
        "annual_emissions_reduction_tons": 396
    }
)
```

### API Integration

```bash
# Query metrics via Prometheus API
curl http://prometheus:9090/api/v1/query \
  -d 'query=gl_002_boiler_efficiency_percent{boiler_id="BOILER-001"}'

# Range query
curl http://prometheus:9090/api/v1/query_range \
  -d 'query=gl_002_optimization_cost_savings_usd_per_hour_sum' \
  -d 'start=2025-11-17T00:00:00Z' \
  -d 'end=2025-11-17T23:59:59Z' \
  -d 'step=1h'
```

---

## Support

### Resources

- **Documentation:** https://docs.greenlang.io/gl-002
- **Runbooks:** https://runbooks.greenlang.io/gl-002
- **Dashboard Links:** https://grafana.greenlang.io/dashboards/gl-002

### Contact

- **Operations Team:** ops@greenlang.io
- **On-Call:** PagerDuty - GL-002 Escalation Policy
- **Slack:** #greenlang-gl-002

### SLA

| Metric | Target | Current |
|--------|--------|---------|
| Availability | 99.9% | [Check Dashboard] |
| P95 Latency | < 2s | [Check Dashboard] |
| Error Rate | < 0.1% | [Check Dashboard] |
| MTTR | < 15min | [Check Incidents] |

---

**Last Updated:** 2025-11-17
**Version:** 1.0.0
**Maintained By:** GreenLang DevOps Team
