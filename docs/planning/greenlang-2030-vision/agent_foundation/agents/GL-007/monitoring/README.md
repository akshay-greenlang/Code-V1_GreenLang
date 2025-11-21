# GL-007 FurnacePerformanceMonitor - Monitoring Documentation

## Overview

Comprehensive monitoring and observability infrastructure for GL-007 FurnacePerformanceMonitor agent, providing real-time visibility into furnace operations, system health, and business metrics.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GL-007 Agent                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Metrics    │  │    Logs      │  │   Traces     │      │
│  │ (Prometheus) │  │    (JSON)    │  │ (OpenTelemetry)│    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌─────────┐        ┌─────────┐       ┌─────────┐
    │Prometheus│        │  Loki/  │       │ Jaeger/ │
    │         │        │   ELK   │       │ Zipkin  │
    └────┬────┘        └────┬────┘       └────┬────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ▼
                      ┌──────────┐
                      │ Grafana  │
                      │Dashboards│
                      └──────────┘
```

## Components

### 1. Metrics (metrics.py)

Prometheus metrics covering:

#### HTTP Metrics (4 metrics)
- `gl_007_http_requests_total` - Total HTTP requests
- `gl_007_http_request_duration_seconds` - Request latency
- `gl_007_http_request_size_bytes` - Request body size
- `gl_007_http_response_size_bytes` - Response body size

#### Agent Health Metrics (3 metrics)
- `gl_007_agent_health_status` - Agent health status
- `gl_007_furnace_monitoring_active` - Active furnaces monitored
- `gl_007_calculation_duration_seconds` - Calculation execution time

#### Furnace Operating Metrics (8 metrics)
- `gl_007_furnace_thermal_efficiency_percent` - Thermal efficiency
- `gl_007_furnace_fuel_consumption_kg_hr` - Fuel consumption rate
- `gl_007_furnace_temperature_celsius` - Temperature by zone/sensor
- `gl_007_furnace_pressure_bar` - Furnace pressure
- `gl_007_furnace_draft_pa` - Furnace draft
- `gl_007_furnace_oxygen_level_percent` - O2 in flue gas
- `gl_007_furnace_production_rate_tons_hr` - Production rate
- `gl_007_furnace_specific_energy_consumption_kwh_ton` - Specific energy consumption

#### Thermal Performance Metrics (5 metrics)
- `gl_007_heat_recovery_efficiency_percent` - Heat recovery efficiency
- `gl_007_heat_loss_rate_kw` - Heat loss by type
- `gl_007_flame_temperature_celsius` - Flame temperature
- `gl_007_wall_temperature_celsius` - Wall temperature
- `gl_007_refractory_temperature_celsius` - Refractory temperature

#### Combustion Metrics (4 metrics)
- `gl_007_air_fuel_ratio` - Air-fuel ratio
- `gl_007_excess_air_percent` - Excess air percentage
- `gl_007_combustion_efficiency_percent` - Combustion efficiency
- `gl_007_flue_gas_temperature_celsius` - Flue gas temperature

#### Maintenance Metrics (5 metrics)
- `gl_007_maintenance_alerts_total` - Maintenance alerts count
- `gl_007_maintenance_prediction_confidence` - Prediction confidence
- `gl_007_component_remaining_life_hours` - Component remaining life
- `gl_007_refractory_degradation_rate_mm_day` - Refractory degradation
- `gl_007_burner_performance_index` - Burner performance index

#### ML/Prediction Metrics (5 metrics)
- `gl_007_prediction_accuracy` - Model accuracy
- `gl_007_model_inference_duration_seconds` - Inference time
- `gl_007_model_training_duration_seconds` - Training time
- `gl_007_model_prediction_error` - Prediction error (RMSE)
- `gl_007_anomaly_detection_alerts_total` - Anomaly alerts

#### SCADA Integration Metrics (5 metrics)
- `gl_007_scada_connection_status` - Connection status
- `gl_007_scada_data_points_received_total` - Data points received
- `gl_007_scada_data_latency_seconds` - Data latency
- `gl_007_scada_polling_errors_total` - Polling errors
- `gl_007_scada_tag_quality` - Tag data quality

#### Business Metrics (6 metrics)
- `gl_007_energy_cost_savings_usd_hr` - Energy cost savings
- `gl_007_annual_energy_savings_usd` - Annual energy savings projection
- `gl_007_carbon_emissions_reduction_kg_hr` - Carbon reduction rate
- `gl_007_annual_carbon_reduction_tons` - Annual carbon reduction
- `gl_007_maintenance_cost_avoidance_usd_total` - Maintenance cost avoided
- `gl_007_production_uptime_percent` - Production uptime

**Total: 45+ metrics**

### 2. Health Checks (health_checks.py)

Kubernetes-compatible health probes:

#### Liveness Probe
- **Endpoint**: `/api/v1/health`
- **Purpose**: Determines if agent should be restarted
- **Checks**: Application running, basic functionality

#### Readiness Probe
- **Endpoint**: `/api/v1/ready`
- **Purpose**: Determines if agent can accept traffic
- **Checks**:
  - Database connectivity
  - Cache availability
  - SCADA connection
  - Time-series database
  - ML models loaded
  - Startup completion

#### Startup Probe
- **Endpoint**: `/api/v1/startup`
- **Purpose**: Initial startup validation
- **Grace Period**: 30 seconds (for ML model loading)

### 3. Alerts (alerts/prometheus_rules.yaml)

18 alert rules across 5 categories:

#### Critical Alerts (10 rules)
1. `GL007AgentUnavailable` - Agent down >1min
2. `GL007HighErrorRate` - Error rate >5%
3. `GL007SCADAConnectionDown` - SCADA disconnected
4. `GL007FurnaceTemperatureAnomaly` - Temperature deviation >15%
5. `GL007LowThermalEfficiency` - Efficiency <70%
6. `GL007DatabaseConnectionFailure` - DB pool exhausted
7. `GL007HighMemoryUsage` - Memory >6GB
8. `GL007RefractoryDegradationCritical` - Degradation >5mm/day
9. `GL007MaintenancePredictionHighConfidence` - Failure predicted <7 days
10. `GL007CalculationTimeout` - Calculation p95 >5s

#### Warning Alerts (14 rules)
1. `GL007PerformanceDegradation` - Latency increase >15%
2. `GL007LowCacheHitRate` - Cache hit rate <75%
3. `GL007EfficiencyBelowTarget` - Efficiency <80%
4. `GL007HighFuelConsumption` - Fuel consumption +20%
5. `GL007SCADADataLatency` - SCADA latency >5s
6. `GL007SCADATagQualityDegraded` - Tag quality <95%
7. `GL007HighDatabaseLatency` - DB query p95 >500ms
8. `GL007ExternalAPILatency` - API latency >5s
9. `GL007HighCPUUsage` - CPU >80%
10. `GL007PredictionAccuracyDrop` - Model accuracy <85%
11. `GL007ProductionCorrelationBreak` - Correlation <0.7
12. `GL007SensorMalfunction` - Sensor stuck reading
13. `GL007BurnerPerformanceDegraded` - Burner index <75
14. `GL007HighHeatLoss` - Heat loss >15%

#### Business Alerts (3 rules)
1. `GL007LowEnergySavings` - Annual savings <$100k
2. `GL007LowCarbonReduction` - Annual CO2 reduction <500 tons
3. `GL007LowProductionUptime` - Uptime <95%

#### SLO Alerts (3 rules)
1. `GL007SLOAvailabilityViolation` - 30-day availability <99.9%
2. `GL007SLOLatencyViolation` - p95 latency >3s
3. `GL007SLOErrorRateBudgetExhausted` - Error rate >0.1%

#### Quality Alerts (2 rules)
1. `GL007DeterminismFailure` - Determinism score <100%
2. `GL007CalculationInconsistency` - High variance detected

### 4. Grafana Dashboards

#### Agent Dashboard (agent_dashboard.json)
- Agent health and availability
- HTTP request metrics (rate, latency, errors)
- System resources (CPU, memory, disk)
- Cache performance
- Database latency
- SCADA connectivity
- Calculation performance
- Determinism tracking

#### Furnace Operations Dashboard (furnace_operations_dashboard.json)
- Thermal efficiency by furnace
- Fuel consumption tracking
- Temperature profiling (20+ zones/sensors)
- Production rate and SEC
- Heat recovery and losses
- Combustion metrics
- Refractory health
- Burner performance
- Maintenance predictions
- Component remaining life
- Production uptime

#### Executive Dashboard (executive_dashboard.json)
- Fleet-wide efficiency
- Annual energy savings projection
- Annual carbon reduction
- Fleet availability
- ROI analysis
- Cost savings trends
- Maintenance cost avoidance
- ML model accuracy
- Compliance metrics
- KPI summary

### 5. Structured Logging (logging_config.py)

Features:
- **JSON Format**: ELK/Loki compatible
- **Correlation IDs**: Request tracing
- **Context Variables**: Furnace ID, User ID tracking
- **Log Rotation**: 100MB files, 10 backups
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Middleware**: Automatic correlation ID injection

Usage:
```python
from monitoring.logging_config import setup_logging, get_logger, LogContext

# Setup
setup_logging_for_environment('production')

# Get logger
logger = get_logger(__name__)

# Log with context
with LogContext(correlation_id='req-123', furnace_id='F-001'):
    logger.info("Processing furnace data")
```

### 6. Distributed Tracing (tracing_config.py)

Features:
- **OpenTelemetry**: Industry standard
- **Jaeger Export**: Production tracing backend
- **OTLP Support**: Cloud-native observability
- **Automatic Instrumentation**: HTTP, logging
- **Custom Spans**: Furnace-specific operations

Usage:
```python
from monitoring.tracing_config import traced, TracingContext

# Decorator
@traced("calculate_efficiency")
async def calculate_efficiency(furnace_id: str):
    return efficiency

# Context manager
with TracingContext("scada_read", furnace_id="F-001"):
    data = read_scada()
```

## Quick Start

### 1. Install Dependencies

```bash
pip install prometheus-client opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-jaeger opentelemetry-instrumentation
```

### 2. Configure Environment

```bash
export ENVIRONMENT=production
export JAEGER_ENDPOINT=jaeger-collector:6831
export OTLP_ENDPOINT=otel-collector:4317
```

### 3. Initialize Monitoring

```python
from monitoring.logging_config import setup_logging_for_environment
from monitoring.tracing_config import setup_tracing_for_environment
from monitoring.metrics import MetricsCollector

# Setup logging
setup_logging_for_environment('production')

# Setup tracing
tracer = setup_tracing_for_environment('production')

# Initialize metrics collector
metrics = MetricsCollector()
```

### 4. Deploy Prometheus Rules

```bash
kubectl apply -f monitoring/alerts/prometheus_rules.yaml
```

### 5. Import Grafana Dashboards

```bash
# Import via Grafana UI or API
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/agent_dashboard.json
```

## Monitoring Best Practices

### 1. Metrics Collection
- Use labels sparingly (high cardinality = high memory)
- Prefer histograms for latency (p50, p95, p99)
- Use counters for events, gauges for states
- Set appropriate bucket sizes for histograms

### 2. Alert Configuration
- Set appropriate thresholds based on baseline
- Use `for` clause to avoid alert flapping
- Include runbook URLs in all alerts
- Test alerts in staging before production

### 3. Dashboard Design
- Group related metrics together
- Use appropriate visualization types
- Set reasonable time ranges
- Include template variables for filtering

### 4. Log Management
- Use structured logging (JSON)
- Include correlation IDs
- Set appropriate log levels
- Implement log rotation
- Avoid logging sensitive data

### 5. Tracing Strategy
- Sample in production (10-50%)
- Trace critical paths
- Include business context in spans
- Set span attributes for filtering

## Troubleshooting

### High Memory Usage
1. Check Prometheus metric cardinality
2. Review cache size configuration
3. Analyze memory leaks with profiling
4. Consider horizontal scaling

### Missing Metrics
1. Verify Prometheus scrape configuration
2. Check network connectivity
3. Review metric registration
4. Validate label consistency

### Alert Fatigue
1. Adjust alert thresholds
2. Increase `for` duration
3. Use alert grouping
4. Implement maintenance windows

### Dashboard Performance
1. Reduce time range
2. Use recording rules for expensive queries
3. Optimize PromQL queries
4. Consider dashboard caching

## Integration Points

### Kubernetes
```yaml
# Deployment with monitoring annotations
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"

# Health probes
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /api/v1/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5

startupProbe:
  httpGet:
    path: /api/v1/startup
    port: 8000
  failureThreshold: 30
  periodSeconds: 10
```

### FastAPI Application
```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from monitoring.logging_config import LoggingMiddleware
from monitoring.tracing_config import TracingMiddleware

app = FastAPI()

# Add monitoring middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

## Support

For questions or issues:
- **Documentation**: https://docs.greenlang.io/gl-007/monitoring
- **Runbooks**: See ALERT_RUNBOOK.md
- **Team**: greenlang-ops@greenlang.io
- **Slack**: #gl-007-monitoring

## Version History

- **1.0.0** (2025-01-19): Initial comprehensive monitoring setup
  - 45+ Prometheus metrics
  - 18 alert rules
  - 3 Grafana dashboards
  - Structured logging
  - Distributed tracing
  - Health checks
