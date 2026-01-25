# GreenLang Monitoring Stack

Production-grade observability for the GreenLang Agent Factory using Prometheus, Grafana, and Alertmanager.

## Architecture

```
                                    +----------------+
                                    |    Grafana     |
                                    |  (Dashboards)  |
                                    +-------+--------+
                                            |
                    +----------------------+----------------------+
                    |                      |                      |
            +-------v-------+      +-------v-------+      +-------v-------+
            |  Prometheus   |      |     Loki      |      |    Tempo      |
            |   (Metrics)   |      |    (Logs)     |      |   (Traces)    |
            +-------+-------+      +-------+-------+      +---------------+
                    |                      |
        +-----------+-----------+          |
        |           |           |          |
+-------v---+ +-----v-----+ +---v----+ +---v----+
|  App      | | Node      | | Redis  | |Promtail|
|  Metrics  | | Exporter  | |Exporter| |  Agent |
+-----------+ +-----------+ +--------+ +--------+
```

## Quick Start

### 1. Start the Monitoring Stack

```bash
# Create required network first
docker network create greenlang-network

# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000 (admin / your-password)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093

### 3. Configure Your Application

```python
from fastapi import FastAPI
from backend.monitoring.middleware import instrument_app

app = FastAPI()
app = instrument_app(app, app_name="greenlang-api")
```

## Components

### Prometheus
- **Port**: 9090
- **Config**: `monitoring/prometheus/prometheus.yml`
- **Alerts**: `monitoring/prometheus/alert_rules.yml`
- **Recording Rules**: `monitoring/prometheus/recording_rules.yml`

### Grafana
- **Port**: 3000
- **Config**: `monitoring/grafana/grafana.ini`
- **Dashboards**: `monitoring/grafana/dashboards/`
- **Provisioning**: `monitoring/grafana/provisioning/`

### Alertmanager
- **Port**: 9093
- **Config**: `monitoring/alertmanager/alertmanager.yml`

### Exporters
- **Node Exporter**: Port 9100 (System metrics)
- **cAdvisor**: Port 8080 (Container metrics)
- **Redis Exporter**: Port 9121 (Redis metrics)
- **PostgreSQL Exporter**: Port 9187 (Database metrics)
- **Blackbox Exporter**: Port 9115 (Endpoint probing)

## Dashboards

### Agent Overview
- Calculations per minute by agent
- Error rate tracking
- P50/P95/P99 latency
- Top agents by usage

### Emission Factors
- EF lookups by source
- Cache hit rate
- Regional distribution
- Performance metrics

### System Health
- CPU/Memory usage
- Service status
- Request rate by endpoint
- Database connections
- Queue depth

## Metrics

### Application Metrics

```python
# Counters
gl_calculations_total{agent, status, calculation_type}
gl_ef_lookups_total{source, region, status, cache}
gl_errors_total{agent, error_type, severity}
gl_http_requests_total{method, handler, status}

# Histograms
gl_calculation_duration_seconds{agent, calculation_type}
gl_ef_lookup_duration_seconds{source}
gl_http_request_duration_seconds{method, handler}

# Gauges
gl_active_agents{agent_type}
gl_registry_agents_count{status}
gl_active_calculations{agent}
gl_cache_size_bytes{cache_name}
```

### Recording Rules (Pre-computed)

```promql
gl:calculations_per_minute:by_agent
gl:calculation_success_rate:by_agent
gl:error_rate:by_agent
gl:calculation_latency_p50:by_agent
gl:calculation_latency_p95:by_agent
gl:calculation_latency_p99:by_agent
gl:ef_cache_hit_rate
gl:availability_1h
gl:availability_24h
```

## Alert Rules

### Service Availability
- `ServiceDown` - Service unavailable for >1 minute
- `APIEndpointUnhealthy` - Health check failing >2 minutes
- `HighPodRestartRate` - >5 restarts in 1 hour

### Error Rate
- `HighErrorRate` - Error rate >1% for 5 minutes
- `CalculationFailureSpike` - >50 failures in 10 minutes
- `HighHTTP5xxRate` - 5xx rate >1%

### Latency
- `HighP99Latency` - P99 >1s for 5 minutes
- `HighP95Latency` - P95 >0.5s for 10 minutes
- `EFLookupSlow` - EF P95 >0.5s

### SLA Compliance
- `SLAAvailabilityViolation` - Availability <99.9%
- `SLALatencyViolation` - P99 latency >1s
- `ErrorBudgetBurning` - Error rate exceeding budget

## Environment Variables

Create a `.env` file:

```bash
# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your-secure-password
GRAFANA_SECRET_KEY=your-secret-key

# SMTP (for alerts)
SMTP_HOST=smtp.example.com
SMTP_USER=alerts@greenlang.io
SMTP_PASSWORD=smtp-password

# Slack (for alerts)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx

# PagerDuty (for critical alerts)
PAGERDUTY_SERVICE_KEY=your-service-key
PAGERDUTY_SLA_SERVICE_KEY=your-sla-service-key

# Database
POSTGRES_USER=greenlang
POSTGRES_PASSWORD=db-password
POSTGRES_DB=greenlang

# Redis
REDIS_PASSWORD=redis-password
```

## Production Deployment

### Kubernetes

Deploy using the Kubernetes manifests in `k8s/monitoring/`:

```bash
kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/prometheus/
kubectl apply -f k8s/monitoring/grafana/
kubectl apply -f k8s/monitoring/alertmanager/
```

### High Availability

For production, consider:
- Prometheus federation or Thanos for long-term storage
- Grafana HA with external database
- Alertmanager clustering

## Runbooks

### Service Down
1. Check pod status: `kubectl get pods -n greenlang`
2. View logs: `kubectl logs -f <pod-name>`
3. Check resource usage
4. Verify network connectivity

### High Error Rate
1. Check recent deployments
2. Review error logs in Loki
3. Check database connectivity
4. Verify external service health

### High Latency
1. Check CPU/memory usage
2. Review slow query logs
3. Check cache hit rates
4. Analyze request patterns

## Contributing

1. Test dashboards locally before committing
2. Export dashboards as JSON with `uid` field
3. Add recording rules for expensive queries
4. Document new alert rules with runbook URLs
