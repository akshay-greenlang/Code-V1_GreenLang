# GL-002 Monitoring Infrastructure

Production-grade monitoring, observability, and alerting for GL-002 BoilerEfficiencyOptimizer.

## Quick Start

### 1. Deploy Monitoring Stack

```bash
# Deploy ServiceMonitor for Prometheus discovery
kubectl apply -f ../deployment/servicemonitor.yaml

# Deploy alerting rules
kubectl apply -f alerts/prometheus_rules.yaml

# Verify deployment
kubectl get servicemonitor,prometheusrule -n greenlang
```

### 2. Import Grafana Dashboards

```bash
# Executive Dashboard (KPIs & ROI)
grafana-cli dashboard import grafana/executive_dashboard.json

# Operations Dashboard (Health & Performance)
grafana-cli dashboard import grafana/operations_dashboard.json

# Agent Dashboard (Tools & Cache)
grafana-cli dashboard import grafana/agent_dashboard.json

# Quality Dashboard (Determinism & Accuracy)
grafana-cli dashboard import grafana/quality_dashboard.json
```

### 3. Access Dashboards

- **Executive:** https://grafana.greenlang.io/d/gl-002-executive
- **Operations:** https://grafana.greenlang.io/d/gl-002-operations
- **Agent:** https://grafana.greenlang.io/d/gl-002-agent
- **Quality:** https://grafana.greenlang.io/d/gl-002-quality

## Directory Structure

```
monitoring/
├── README.md                           # This file
├── MONITORING.md                       # Comprehensive monitoring guide
├── metrics.py                          # Prometheus metrics definitions
├── health_checks.py                    # Health check endpoints
├── grafana/                            # Grafana dashboards
│   ├── executive_dashboard.json        # Executive KPIs & ROI
│   ├── operations_dashboard.json       # Health & Performance
│   ├── agent_dashboard.json            # Tools & Cache
│   └── quality_dashboard.json          # Determinism & Accuracy
└── alerts/                             # Prometheus alerting rules
    └── prometheus_rules.yaml           # Alert definitions
```

## Metrics Categories

### HTTP & API Metrics
- Request rate, latency, errors
- Request/response sizes
- Endpoint performance

### Optimization Metrics
- Optimization requests by strategy
- Duration, efficiency improvements
- Cost savings, emissions reduction

### Boiler Operating Metrics
- Efficiency, flow rates
- Temperature, pressure, load
- Excess air, combustion metrics

### Emissions Metrics
- CO2, NOx, CO, SO2 levels
- Compliance status
- Violation tracking

### Cache Metrics
- Hit rate, miss rate
- Eviction rate
- Performance by pattern

### Database Metrics
- Connection pool size
- Query latency
- Error rates

### System Metrics
- CPU, memory, disk usage
- Uptime
- Resource utilization

### Business Metrics
- Annual cost savings
- Annual emissions reduction
- ROI, payback period

## Alert Severity

| Level | Response | Examples |
|-------|----------|----------|
| **CRITICAL** | Immediate | Agent down, high error rate, compliance violation |
| **WARNING** | 15 min | Performance degradation, low cache hit |
| **INFO** | 1 hour | Business metrics below target |

## Key Metrics

### Cache Hit Rate
```promql
100 * sum(rate(gl_002_cache_hits_total[5m]))
/ (sum(rate(gl_002_cache_hits_total[5m])) + sum(rate(gl_002_cache_misses_total[5m])))
```
**Target:** > 85%

### Error Rate
```promql
100 * sum(rate(gl_002_http_requests_total{status="error"}[5m]))
/ sum(rate(gl_002_http_requests_total[5m]))
```
**Target:** < 0.1%

### P95 Latency
```promql
histogram_quantile(0.95, sum(rate(gl_002_http_request_duration_seconds_bucket[5m])) by (le))
```
**Target:** < 2s

### Efficiency
```promql
avg(gl_002_boiler_efficiency_percent)
```
**Target:** > 85%

## Integration Example

```python
from monitoring.metrics import MetricsCollector

# Update boiler metrics
MetricsCollector.update_boiler_metrics(
    boiler_id="BOILER-001",
    metrics={
        "efficiency_percent": 87.5,
        "steam_flow_kg_hr": 15000,
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
        "cost_savings_usd_hr": 125.50
    }
)
```

## Troubleshooting

### Metrics Not Appearing
1. Check `/metrics` endpoint: `curl http://localhost:8000/metrics`
2. Verify ServiceMonitor: `kubectl get servicemonitor -n greenlang`
3. Check Prometheus targets: http://prometheus:9090/targets

### Dashboard No Data
1. Test query in Prometheus UI
2. Verify datasource in Grafana
3. Check time range

### Alerts Not Firing
1. Check PrometheusRule: `kubectl get prometheusrule -n greenlang`
2. View rules in Prometheus: http://prometheus:9090/rules
3. Verify AlertManager config

## Documentation

- **Full Guide:** [MONITORING.md](MONITORING.md)
- **Metrics Reference:** [metrics.py](metrics.py)
- **Health Checks:** [health_checks.py](health_checks.py)
- **Alert Rules:** [alerts/prometheus_rules.yaml](alerts/prometheus_rules.yaml)

## Support

- **Slack:** #greenlang-gl-002
- **Email:** ops@greenlang.io
- **PagerDuty:** GL-002 Escalation Policy

## SLOs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Availability | 99.9% | 30-day rolling |
| P95 Latency | < 2s | 1-hour window |
| Error Rate | < 0.1% | 5-minute window |
| MTTR | < 15min | Per incident |
