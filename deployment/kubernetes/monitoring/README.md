# GreenLang Agent Factory - Monitoring Stack

Production-grade monitoring and observability for the GreenLang Agent Factory using Prometheus, Grafana, and Alertmanager.

## Overview

This monitoring stack provides:

- **Prometheus**: Metrics collection and storage (100GB persistent volume, 15s scrape interval)
- **Grafana**: Visualization dashboards
- **Alertmanager**: Alert routing to Slack/PagerDuty
- **ServiceMonitors**: Automatic agent discovery and scraping

## Components

### 1. Prometheus Operator

Installed via Helm using `kube-prometheus-stack` chart:

```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  -f prometheus-values.yaml \
  --namespace monitoring \
  --create-namespace
```

### 2. ServiceMonitors

Three agents are monitored:

| Agent | ServiceMonitor | Namespace | Port | Path |
|-------|---------------|-----------|------|------|
| Fuel Analyzer | `servicemonitor-fuel-analyzer.yaml` | greenlang, gl-fuel | 8000 | /metrics |
| CBAM Importer | `servicemonitor-cbam.yaml` | gl-cbam | 8000 | /metrics |
| Building Energy | `servicemonitor-building-energy.yaml` | greenlang, gl-building | 8000 | /metrics |

### 3. Grafana Dashboards

Three dashboards are provided:

1. **Agent Factory Overview** (`agent-factory-overview.json`)
   - Total requests per agent
   - Error rate
   - P95 latency
   - Cache hit rate

2. **Agent Health** (`agent-health.json`)
   - Per-agent request rate
   - Per-agent error rate
   - Per-tool usage
   - Database query latency

3. **Infrastructure** (`infrastructure.json`)
   - PostgreSQL connections
   - Redis memory usage
   - K8s pod status

### 4. Alerting Rules

Three alert rules configured in `prometheus-rules.yaml`:

| Alert | Severity | Condition | Duration |
|-------|----------|-----------|----------|
| AgentHighErrorRate | critical | error_rate > 1% | 5 min |
| AgentHighLatency | warning | p95_latency > 500ms | 5 min |
| AgentPodNotReady | warning | replicas < desired | 5 min |

## Agent Metrics

Each agent must expose the following metrics on port 8000 at `/metrics`:

```python
# Required metrics (from greenlang.monitoring)
agent_requests_total{agent_id, status, method}        # Counter
agent_request_duration_seconds{agent_id, method}      # Histogram
agent_calculations_total{agent_id, tool_name, status} # Counter
agent_cache_hits_total{agent_id, cache_tier}          # Counter
```

### Integration Example

```python
from greenlang.monitoring import StandardAgentMetrics

# Initialize metrics for your agent
metrics = StandardAgentMetrics(
    agent_id="GL-001",
    agent_name="FuelAnalyzer",
    codename="FUELWATCH",
    version="1.0.0",
    domain="emissions"
)

# Track requests
with metrics.track_request("POST", "/api/v1/calculate"):
    result = calculate_emissions(input_data)

# Track calculations
with metrics.track_calculation("calculate_emissions"):
    emissions = run_calculation()

# Track cache
metrics.record_cache_operation("redis", hit=True)
```

## Installation

### Quick Install

```bash
# Make script executable
chmod +x install.sh

# Run installation
./install.sh

# Or dry-run first
./install.sh --dry-run
```

### Manual Installation

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Add Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 3. Create Grafana secret
kubectl create secret generic grafana-admin \
  --from-literal=admin-user=admin \
  --from-literal=admin-password=YOUR_SECURE_PASSWORD \
  --namespace monitoring

# 4. Install Prometheus Stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  -f prometheus-values.yaml \
  --namespace monitoring \
  --create-namespace

# 5. Apply ServiceMonitors
kubectl apply -f servicemonitor-fuel-analyzer.yaml
kubectl apply -f servicemonitor-cbam.yaml
kubectl apply -f servicemonitor-building-energy.yaml

# 6. Apply alerting rules
kubectl apply -f prometheus-rules.yaml

# 7. Apply dashboards
kubectl create configmap grafana-dashboard-overview \
  --from-file=dashboards/agent-factory-overview.json \
  --namespace monitoring
kubectl label configmap grafana-dashboard-overview grafana_dashboard=1 -n monitoring
# Repeat for other dashboards
```

## Access

### Port Forwarding (Development)

```bash
# Prometheus
kubectl port-forward svc/prometheus-prometheus -n monitoring 9090:9090

# Grafana
kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80

# Alertmanager
kubectl port-forward svc/prometheus-alertmanager -n monitoring 9093:9093
```

### Ingress (Production)

Configure DNS for:
- `prometheus.greenlang.ai` -> Prometheus UI
- `grafana.greenlang.ai` -> Grafana UI
- `alertmanager.greenlang.ai` -> Alertmanager UI

## Slack Integration

Configure Alertmanager to send alerts to Slack:

1. Create a Slack webhook URL
2. Set the environment variable or secret:

```bash
kubectl create secret generic alertmanager-slack \
  --from-literal=webhook-url=https://hooks.slack.com/services/... \
  --namespace monitoring
```

3. Update `prometheus-values.yaml` with the webhook reference

## Troubleshooting

### Check Prometheus targets

```bash
kubectl port-forward svc/prometheus-prometheus -n monitoring 9090:9090
# Open http://localhost:9090/targets
```

### Check ServiceMonitor discovery

```bash
kubectl get servicemonitors -n monitoring
kubectl describe servicemonitor fuel-analyzer-agent -n monitoring
```

### Check alerting rules

```bash
kubectl get prometheusrules -n monitoring
kubectl describe prometheusrule greenlang-agent-rules -n monitoring
```

### View Prometheus logs

```bash
kubectl logs -l app.kubernetes.io/name=prometheus -n monitoring -f
```

## File Structure

```
k8s/monitoring/
├── README.md                           # This file
├── install.sh                          # Installation script
├── namespace.yaml                      # Namespace and resource quotas
├── prometheus-values.yaml              # Helm values for kube-prometheus-stack
├── prometheus-rules.yaml               # PrometheusRule CRD with alerting rules
├── servicemonitor-fuel-analyzer.yaml   # ServiceMonitor for Fuel Analyzer
├── servicemonitor-cbam.yaml            # ServiceMonitor for CBAM Importer
├── servicemonitor-building-energy.yaml # ServiceMonitor for Building Energy
└── dashboards/
    ├── agent-factory-overview.json     # Overview dashboard
    ├── agent-health.json               # Agent health dashboard
    └── infrastructure.json             # Infrastructure dashboard
```

## Success Criteria

- [x] Prometheus scraping all 3 agents
- [x] 3 Grafana dashboards functional
- [x] Metrics visible in Grafana
- [x] 3 alert rules configured
- [x] Alertmanager sending to Slack

## Related Documentation

- [Prometheus Operator](https://prometheus-operator.dev/)
- [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)

---

Created: 2025-12-03
Team: Monitoring & Observability
