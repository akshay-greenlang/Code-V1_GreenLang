# GL-003 SteamSystemAnalyzer - Monitoring Guide

## Overview

This document provides comprehensive monitoring setup, configuration, and operational guidance for GL-003 SteamSystemAnalyzer. The monitoring infrastructure provides production-grade observability with Prometheus metrics, Grafana dashboards, and intelligent alerting.

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
│                    GL-003 Application                       │
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
gl_003_http_requests_total{method, endpoint, status}

# Request duration (histogram)
gl_003_http_request_duration_seconds{method, endpoint}

# Request/response size
gl_003_http_request_size_bytes{method, endpoint}
gl_003_http_response_size_bytes{method, endpoint}
```

**Use Cases:**
- Monitor API performance
- Track request throughput
- Identify slow endpoints

#### 2. Steam System Operating Metrics

```promql
# Pressure & temperature
gl_003_steam_pressure_bar{system_id, location, steam_type}
gl_003_steam_temperature_c{system_id, location, steam_type}

# Flow rates
gl_003_steam_flow_rate_kg_hr{system_id, location}
gl_003_condensate_return_rate_kg_hr{system_id, location}

# Efficiency
gl_003_distribution_efficiency_percent{system_id}
gl_003_condensate_return_percent{system_id}
gl_003_steam_quality_percent{system_id, location}
```

**Use Cases:**
- Real-time steam system monitoring
- Performance trending
- Efficiency tracking

#### 3. Leak Detection Metrics

```promql
# Active leaks
gl_003_active_leaks_count{system_id, severity}

# Leak impact
gl_003_leak_cost_impact_usd_hr{system_id}
gl_003_leak_loss_rate_kg_hr{system_id, leak_id}

# Detection accuracy
gl_003_leak_detection_confidence_percent{system_id, leak_id}
```

**Use Cases:**
- Proactive leak monitoring
- Cost impact assessment
- Maintenance prioritization

#### 4. Steam Trap Metrics

```promql
# Trap counts
gl_003_steam_trap_operational_count{system_id, trap_type}
gl_003_steam_trap_failed_count{system_id, trap_type, failure_mode}

# Performance
gl_003_steam_trap_performance_score_percent{system_id, trap_id}
gl_003_steam_trap_losses_kg_hr{system_id, trap_id}
```

**Use Cases:**
- Trap maintenance scheduling
- Failure pattern analysis
- Loss quantification

#### 5. Distribution Efficiency Metrics

```promql
# Efficiency tracking
gl_003_distribution_efficiency_percent{system_id}
gl_003_pipe_heat_loss_kw{system_id, pipe_segment}
gl_003_insulation_effectiveness_percent{system_id, pipe_segment}
gl_003_pressure_drop_bar{system_id, segment}
```

**Use Cases:**
- Identify inefficient segments
- Insulation assessment
- Pressure management

#### 6. Business Metrics

```promql
# Annual projections
gl_003_analysis_annual_savings_usd{system_id}
gl_003_analysis_annual_energy_savings_mwh{system_id}

# Cost tracking
gl_003_steam_cost_per_ton_usd{system_id}
gl_003_analysis_payback_period_months{recommendation_type}
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
- Annual Energy Savings (MWh)
- Average Distribution Efficiency (%)
- Analysis Success Rate (%)
- Cost Savings Trend (24h)
- Energy Savings Trend (24h)
- System Efficiency by Location
- Active Leaks by Severity
- Steam Trap Failure Rate
- ROI Metrics Summary Table

**Refresh Rate:** 30s

**Target Audience:** Executives, Product Managers, Business Analysts

### 2. Operations Dashboard (Health & Performance)

**Location:** `monitoring/grafana/operations_dashboard.json`

**Panels:**
- System Uptime
- HTTP Request Rate
- Error Rate (%)
- P95 Latency
- Request Latency Distribution
- CPU/Memory Usage
- Database Performance
- External API Latency
- Active Systems Status Table

**Refresh Rate:** 10s

**Target Audience:** DevOps Engineers, SREs, Operations

### 3. Agent Dashboard (Steam System Performance)

**Location:** `monitoring/grafana/agent_dashboard.json`

**Panels:**
- Steam Pressure by System
- Steam Flow Rates
- Condensate Return Efficiency
- Distribution Efficiency
- Active Leaks Map
- Steam Trap Status
- Heat Loss by Segment
- Cache Performance

**Refresh Rate:** 10s

**Target Audience:** Plant Engineers, Operators, Performance Engineers

---

## Alerting

### Alert Severity Levels

| Severity | Response Time | Escalation | Examples |
|----------|--------------|------------|----------|
| **CRITICAL** | Immediate | PagerDuty | Agent down, critical leaks, massive trap failure |
| **WARNING** | 15 minutes | Slack | Low efficiency, moderate leaks, high heat loss |
| **INFO** | 1 hour | Email | Low savings, business metric alerts |

### Critical Alerts

#### 1. GL003AgentUnavailable
```yaml
Condition: up{job="gl-003-steam-analyzer"} == 0
Duration: 1 minute
Action: Immediate investigation, check pod logs, restart if needed
```

#### 2. GL003CriticalSteamLeak
```yaml
Condition: Active critical leaks > 0
Duration: 1 minute
Action: Isolate leak, notify plant management, dispatch maintenance
```

#### 3. GL003SteamTrapMassiveFailure
```yaml
Condition: Trap failure rate > 30%
Duration: 5 minutes
Action: Inspect trap system, schedule immediate maintenance
```

#### 4. GL003LowDistributionEfficiency
```yaml
Condition: Efficiency < 70%
Duration: 15 minutes
Action: Investigate heat losses, check for leaks, inspect insulation
```

### Alert Configuration

**File:** `monitoring/alerts/prometheus_rules.yaml`

**Deployment:**
```bash
kubectl apply -f monitoring/alerts/prometheus_rules.yaml
kubectl apply -f monitoring/alerts/determinism_alerts.yml
```

---

## Setup Instructions

### Prerequisites

- Kubernetes cluster (v1.24+)
- Prometheus Operator installed
- Grafana installed
- kubectl access to cluster

### Step 1: Deploy Application with Metrics

```python
# In your FastAPI app
from prometheus_client import make_asgi_app

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Step 2: Deploy ServiceMonitor

```bash
kubectl apply -f ../deployment/servicemonitor.yaml
kubectl get servicemonitor -n greenlang gl-003-steam-analyzer
```

### Step 3: Deploy Alerting Rules

```bash
kubectl apply -f monitoring/alerts/prometheus_rules.yaml
kubectl get prometheusrule -n greenlang gl-003-alerts
```

### Step 4: Import Grafana Dashboards

```bash
# Create ConfigMap with dashboards
kubectl create configmap gl-003-grafana-dashboards \
  --from-file=monitoring/grafana/ \
  -n greenlang

# Label for auto-discovery
kubectl label configmap gl-003-grafana-dashboards \
  grafana_dashboard=1 -n greenlang
```

### Step 5: Verify Monitoring Stack

```bash
# Check metrics endpoint
kubectl port-forward -n greenlang svc/gl-003-metrics 8000:8000
curl http://localhost:8000/metrics | grep gl_003

# Check Prometheus targets
kubectl port-forward -n greenlang svc/prometheus 9090:9090
# Visit http://localhost:9090/targets

# Check Grafana dashboards
kubectl port-forward -n greenlang svc/grafana 3000:3000
# Visit http://localhost:3000
```

---

## Troubleshooting

### Metrics Not Appearing

**Debug Steps:**
```bash
# 1. Check /metrics endpoint
kubectl port-forward -n greenlang pod/gl-003-xxx 8000:8000
curl http://localhost:8000/metrics

# 2. Verify ServiceMonitor
kubectl get servicemonitor -n greenlang

# 3. Check Prometheus targets
# Visit http://prometheus:9090/targets

# 4. Check service labels
kubectl get svc -n greenlang gl-003-metrics -o yaml
```

### Dashboard Not Loading Data

**Debug Steps:**
```bash
# 1. Test query in Prometheus
# Visit http://prometheus:9090/graph
# Try: gl_003_steam_pressure_bar

# 2. Check Grafana datasource
# Grafana UI > Configuration > Data Sources > Prometheus

# 3. Verify metric names
curl http://localhost:8000/metrics | grep gl_003
```

### Alerts Not Firing

**Debug Steps:**
```bash
# 1. Check PrometheusRule
kubectl get prometheusrule -n greenlang

# 2. View rules in Prometheus
# Visit http://prometheus:9090/rules

# 3. Test alert query manually
# Copy expression and run in Prometheus UI

# 4. Check AlertManager logs
kubectl logs -n greenlang -l app=alertmanager -f
```

---

## Runbooks

### Runbook: Critical Steam Leak

**Alert:** GL003CriticalSteamLeak

**Severity:** CRITICAL

**Steps:**
1. **Identify leak location**
   ```promql
   gl_003_active_leaks_count{severity="critical"}
   gl_003_leak_loss_rate_kg_hr
   ```

2. **Assess cost impact**
   ```promql
   sum(gl_003_leak_cost_impact_usd_hr) by (system_id)
   ```

3. **Isolate affected segment**
   - Close isolation valves
   - Redirect steam flow if possible

4. **Dispatch maintenance**
   - Create work order
   - Notify plant management
   - Schedule emergency repair

5. **Monitor resolution**
   - Track leak count reduction
   - Verify cost impact decrease

### Runbook: Low Distribution Efficiency

**Alert:** GL003LowDistributionEfficiency

**Severity:** CRITICAL/WARNING

**Steps:**
1. **Check efficiency metrics**
   ```promql
   gl_003_distribution_efficiency_percent
   gl_003_distribution_losses_percent
   ```

2. **Identify loss sources**
   ```promql
   sum(gl_003_pipe_heat_loss_kw) by (pipe_segment)
   sum(gl_003_active_leaks_count) by (severity)
   sum(gl_003_steam_trap_failed_count) by (failure_mode)
   ```

3. **Investigate root causes**
   - Check insulation effectiveness
   - Inspect for leaks
   - Review steam trap status
   - Analyze pressure drops

4. **Implement corrective actions**
   - Repair/replace failed steam traps
   - Fix identified leaks
   - Upgrade insulation where needed
   - Optimize pressure management

5. **Verify improvement**
   - Monitor efficiency trend
   - Track energy savings

---

## Integration Examples

### Python Integration

```python
from monitoring.metrics import MetricsCollector
from monitoring.metrics_integration import MetricsIntegration

# Initialize metrics
metrics = MetricsIntegration()

# Update steam system metrics
MetricsCollector.update_steam_system_metrics(
    system_id="STEAM-001",
    metrics={
        "pressure_bar": 10.5,
        "temperature_c": 184,
        "flow_rate_kg_hr": 5000,
        "condensate_return_rate_kg_hr": 4250,
        "condensate_return_percent": 85,
        "steam_quality_percent": 98,
        "distribution_efficiency_percent": 92,
        "location": "main_header",
        "steam_type": "high_pressure"
    }
)

# Update leak metrics
MetricsCollector.update_leak_metrics(
    system_id="STEAM-001",
    leaks={
        "active_leaks": {
            "minor": 2,
            "moderate": 1,
            "major": 0,
            "critical": 0
        },
        "total_leak_cost_usd_hr": 25.50
    }
)

# Update steam trap metrics
MetricsCollector.update_steam_trap_metrics(
    system_id="STEAM-001",
    traps={
        "operational_count": {
            "thermostatic": 45,
            "thermodynamic": 32,
            "inverted_bucket": 18
        },
        "failed_count": {
            "thermostatic": {"blowing": 2, "plugged": 1},
            "thermodynamic": {"blowing": 1, "plugged": 0}
        }
    }
)

# Record analysis result
MetricsCollector.record_analysis_result(
    system_id="STEAM-001",
    steam_type="high_pressure",
    analysis_type="distribution_efficiency",
    result={
        "improvement_percent": 4.5,
        "cost_savings_usd_hr": 85.50,
        "energy_savings_mwh": 12.3,
        "annual_savings_usd": 749580,
        "annual_energy_savings_mwh": 107784
    }
)
```

---

## Support

### Resources

- **Documentation:** https://docs.greenlang.io/gl-003
- **Runbooks:** https://runbooks.greenlang.io/gl-003
- **Dashboard Links:** https://grafana.greenlang.io/dashboards/gl-003

### Contact

- **Operations Team:** ops@greenlang.io
- **On-Call:** PagerDuty - GL-003 Escalation Policy
- **Slack:** #greenlang-gl-003

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
