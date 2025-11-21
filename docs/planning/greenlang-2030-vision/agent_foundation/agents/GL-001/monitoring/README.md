# GL-001 ProcessHeatOrchestrator - Monitoring Infrastructure

## Overview

Comprehensive monitoring and observability infrastructure for GL-001 ProcessHeatOrchestrator, the master orchestrator managing 99 sub-agents and multi-plant industrial process heat operations.

## Quick Start

```bash
# 1. Deploy ServiceMonitor for Prometheus scraping
kubectl apply -f ../deployment/servicemonitor.yaml

# 2. Deploy alerting rules
kubectl apply -f alerts/prometheus_rules.yaml

# 3. Import Grafana dashboards
kubectl create configmap gl-001-grafana-dashboards \
  --from-file=grafana/ \
  -n greenlang

# 4. Verify metrics endpoint
kubectl port-forward -n greenlang svc/gl-001-orchestrator 8000:8000
curl http://localhost:8000/metrics | grep gl_001
```

## Architecture

### Monitoring Stack Components

```
┌─────────────────────────────────────────────────────────────┐
│                  GL-001 Master Orchestrator                  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   metrics.py │  │ health.py    │  │ determinism  │     │
│  │  (120+ metrics)│  │ checks      │  │ validator    │     │
│  └───────┬──────┘  └──────┬───────┘  └──────┬───────┘     │
│          │                 │                  │              │
└──────────┼─────────────────┼──────────────────┼──────────────┘
           │                 │                  │
           ├─────────────────┴──────────────────┤
           │                                    │
    ┌──────▼──────┐                    ┌────────▼────────┐
    │  Prometheus │                    │  Grafana        │
    │  - Scraping │                    │  - Dashboards   │
    │  - Alerting │                    │  - Visualization │
    └──────┬──────┘                    └─────────────────┘
           │
    ┌──────▼──────┐
    │ AlertManager│
    │ - PagerDuty │
    │ - Slack     │
    │ - Email     │
    └─────────────┘
```

## Components

### 1. metrics.py (1000+ lines, 120+ metrics)

Comprehensive Prometheus metrics covering:

#### Master Orchestrator Metrics
- `gl_001_orchestrator_health_status` - Health status (1=healthy, 0=unhealthy)
- `gl_001_orchestrator_uptime_seconds` - Uptime tracking
- `gl_001_orchestration_requests_total` - Request counter by type and status
- `gl_001_orchestration_duration_seconds` - Latency histogram
- `gl_001_orchestrator_state` - Current state (READY, EXECUTING, ERROR, etc.)

#### Multi-Plant Coordination Metrics
- `gl_001_active_plants_count` - Number of active plants
- `gl_001_plant_health_status` - Per-plant health
- `gl_001_plant_thermal_efficiency_percent` - Plant-level efficiency
- `gl_001_plant_heat_generation_mw` - Heat generation per plant
- `gl_001_plant_heat_demand_mw` - Heat demand per plant
- `gl_001_plant_heat_losses_mw` - Heat losses by type
- `gl_001_cross_plant_heat_transfer_mw` - Inter-plant heat transfer

#### Sub-Agent Coordination Metrics (GL-002 to GL-100)
- `gl_001_active_subagents_count` - Active agent count by category
- `gl_001_subagent_health_status` - Per-agent health
- `gl_001_subagent_response_time_seconds` - Agent response latency
- `gl_001_subagent_task_assignments_total` - Task delegation counter
- `gl_001_subagent_task_completion_total` - Task completion tracking
- `gl_001_subagent_coordination_failures` - Coordination failure counter
- `gl_001_subagent_message_queue_depth` - Message backlog per agent

#### SCADA Integration Metrics (Per Plant)
- `gl_001_scada_connection_status` - Connection health per plant
- `gl_001_scada_data_points_received_total` - Data ingestion rate
- `gl_001_scada_data_latency_seconds` - Data freshness
- `gl_001_scada_data_quality_percent` - Data quality score
- `gl_001_scada_tags_monitored` - Tag count per plant
- `gl_001_scada_alarms_active` - Active alarm count by severity

#### ERP Integration Metrics
- `gl_001_erp_connection_status` - ERP connectivity
- `gl_001_erp_data_sync_total` - Sync operation counter
- `gl_001_erp_data_sync_duration_seconds` - Sync latency
- `gl_001_erp_production_schedule_updates` - Schedule update tracking
- `gl_001_erp_cost_data_updates` - Cost data sync tracking

#### Thermal Efficiency Metrics
- `gl_001_aggregate_thermal_efficiency_percent` - Enterprise-wide efficiency
- `gl_001_aggregate_heat_generation_mw` - Total heat generation
- `gl_001_plant_boiler_efficiency_percent` - Per-plant boiler efficiency
- `gl_001_plant_heat_recovery_efficiency_percent` - Heat recovery tracking
- `gl_001_efficiency_improvement_percent` - Optimization impact

#### Heat Distribution Metrics
- `gl_001_heat_distribution_optimization_score` - Distribution quality (0-100)
- `gl_001_heat_distribution_imbalance_mw` - Supply-demand mismatch
- `gl_001_heat_network_pressure_bar` - Network pressure by segment
- `gl_001_heat_network_temperature_c` - Network temperature
- `gl_001_heat_network_flow_rate_kg_hr` - Flow rates

#### Energy Balance Metrics
- `gl_001_energy_balance_closure_percent` - Balance accuracy
- `gl_001_energy_balance_error_mw` - Imbalance magnitude
- `gl_001_energy_input_mw` - Total energy input by type
- `gl_001_energy_output_mw` - Total energy output by type
- `gl_001_energy_storage_mwh` - Storage state

#### Emissions Compliance Metrics
- `gl_001_emissions_compliance_status` - Compliance per regulation
- `gl_001_emissions_co2_tons_hr` - CO2 emission rate
- `gl_001_emissions_co2_intensity_kg_mwh` - Carbon intensity
- `gl_001_emissions_compliance_violations` - Violation counter
- `gl_001_carbon_credit_balance_tons` - Carbon credits

#### Task Delegation Metrics
- `gl_001_tasks_delegated_total` - Task assignment counter
- `gl_001_tasks_completed_total` - Completion tracking
- `gl_001_task_queue_depth` - Queue backlog by priority
- `gl_001_task_execution_duration_seconds` - Task latency
- `gl_001_task_failure_rate_percent` - Failure rate by category

#### Performance Metrics
- `gl_001_calculation_cache_hit_rate_percent` - Cache efficiency
- `gl_001_calculation_duration_seconds` - Calculation latency
- `gl_001_message_bus_latency_seconds` - Inter-agent messaging latency
- `gl_001_system_memory_usage_bytes` - Memory consumption
- `gl_001_system_cpu_usage_percent` - CPU utilization

### 2. Prometheus Alert Rules (400+ lines)

#### CRITICAL Alerts (Immediate Response)
- **GL001MasterOrchestratorDown** - Master orchestrator unavailable
- **GL001AllSubAgentsFailed** - All 99 sub-agents failed
- **GL001MultiPlantHeatLoss** - Excessive heat losses (>100 MW)
- **GL001CriticalPlantFailure** - Plant shutdown
- **GL001SCADAConnectionLost** - SCADA connectivity lost
- **GL001ERPIntegrationFailure** - ERP integration down
- **GL001EmissionsComplianceViolation** - Regulatory violation

#### WARNING Alerts (Investigation Required)
- **GL001OrchestratorPerformanceDegradation** - >20% slowdown
- **GL001SubAgentDegradation** - >20% agents unavailable
- **GL001PlantEfficiencyDrop** - Efficiency <70%
- **GL001HighTaskQueueDepth** - Queue >100 tasks
- **GL001SCADADataQualityLow** - Data quality <90%
- **GL001LowCacheHitRate** - Cache efficiency <60%

#### BUSINESS Alerts (Strategic Monitoring)
- **GL001LowOptimizationSavings** - Annual savings <$500k
- **GL001LowEmissionsReduction** - CO2 reduction <1000 tons/year
- **GL001LowPlantCapacityUtilization** - Capacity <60%

#### SLO Alerts (Service Level Objectives)
- **GL001SLOAvailabilityViolation** - Availability <99.9%
- **GL001SLOLatencyViolation** - P95 latency >5s
- **GL001SLOErrorRateBudgetExhausted** - Error rate >0.1%

### 3. Grafana Dashboards

#### master_orchestrator_dashboard.json
**Primary operational view for DevOps team**

Panels:
- Orchestrator health status and uptime
- Request rate and latency (p50, p95, p99)
- Active orchestration sessions
- Error rate and error breakdown
- Resource utilization (CPU, memory, disk)

Key Queries:
```promql
# Orchestration request rate
rate(gl_001_orchestration_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(gl_001_orchestration_duration_seconds_bucket[5m]))

# Error rate
100 * sum(rate(gl_001_orchestration_requests_total{status="failure"}[5m]))
/ sum(rate(gl_001_orchestration_requests_total[5m]))
```

#### multi_plant_dashboard.json
**Multi-plant coordination overview**

Panels:
- Active plants map with health status
- Plant-level thermal efficiency heatmap
- Total heat generation vs demand
- Cross-plant heat transfer flows
- Plant capacity utilization
- Heat loss breakdown by plant

Key Queries:
```promql
# Plant health matrix
sum(gl_001_plant_health_status) by (plant_id, plant_name)

# Aggregate efficiency
gl_001_aggregate_thermal_efficiency_percent

# Heat losses by type
sum(gl_001_plant_heat_losses_mw) by (plant_id, loss_type)
```

#### subagent_coordination_dashboard.json
**Sub-agent monitoring (GL-002 to GL-100)**

Panels:
- Active sub-agents by category
- Agent health matrix (99 agents)
- Response time heatmap
- Task delegation rate
- Task completion rate and status
- Coordination failure trends
- Message queue depths

Key Queries:
```promql
# Sub-agent health matrix
gl_001_subagent_health_status

# Agent response times
histogram_quantile(0.95, rate(gl_001_subagent_response_time_seconds_bucket[5m]))

# Task success rate
100 * sum(rate(gl_001_tasks_completed_total{status="success"}[5m]))
/ sum(rate(gl_001_tasks_completed_total[5m]))
```

#### thermal_efficiency_dashboard.json
**Thermal performance analytics**

Panels:
- Enterprise-wide thermal efficiency trend
- Per-plant efficiency comparison
- Boiler efficiency distribution
- Heat recovery performance
- Distribution efficiency
- Efficiency improvement tracking
- Energy balance closure

Key Queries:
```promql
# Aggregate efficiency
gl_001_aggregate_thermal_efficiency_percent

# Plant efficiency ranking
topk(10, gl_001_plant_thermal_efficiency_percent)

# Efficiency improvements
rate(gl_001_efficiency_improvement_percent_sum[1h])
```

#### operations_dashboard.json
**24/7 operations team view**

Panels:
- System health overview
- Active alerts
- SCADA connection status matrix
- ERP integration health
- Data quality scores
- Task queue depths
- Recent errors and warnings
- Key performance indicators

## Integration Examples

### Python Integration

```python
from monitoring.metrics import MetricsCollector, track_orchestration_metrics

# Initialize metrics collector
metrics = MetricsCollector()

# Update orchestrator state
metrics.update_orchestrator_state(
    state="EXECUTING",
    is_healthy=True
)

# Update plant metrics
metrics.update_plant_metrics(
    plant_id="PLANT-001",
    plant_name="Petrochemical Complex A",
    metrics={
        "health_status": "healthy",
        "thermal_efficiency_percent": 87.5,
        "heat_generation_mw": 250.0,
        "heat_demand_mw": 245.0,
        "capacity_utilization_percent": 92.3,
        "location": "Houston, TX"
    }
)

# Update sub-agent metrics
metrics.update_subagent_metrics(
    agent_id="GL-002",
    agent_type="BoilerEfficiencyOptimizer",
    plant_id="PLANT-001",
    metrics={
        "health_status": "healthy",
        "message_queue_depth": 5
    }
)

# Record task delegation
metrics.record_task_delegation(
    task_category="efficiency_optimization",
    priority="high",
    agent_id="GL-002",
    task_type="optimize_boiler_combustion"
)

# Update SCADA integration metrics
metrics.update_scada_metrics(
    plant_id="PLANT-001",
    scada_system="Wonderware",
    metrics={
        "connection_status": "connected",
        "data_quality_percent": {
            "temperature": 98.5,
            "pressure": 99.2,
            "flow": 97.8
        },
        "tags_monitored": 1250,
        "active_alarms": {
            "low": 3,
            "medium": 1,
            "high": 0,
            "critical": 0
        }
    }
)

# Update emissions metrics
metrics.update_emissions_metrics(
    plant_id="PLANT-001",
    emissions={
        "co2_tons_hr": 45.2,
        "co2_intensity_kg_mwh": 180.8,
        "nox_kg_hr": 2.3,
        "compliance_status": {
            "EPA_MRR": "compliant",
            "EU_ETS": "compliant"
        }
    }
)

# Decorator for automatic tracking
@track_orchestration_metrics('multi_plant_optimization')
async def optimize_all_plants():
    # Your orchestration logic
    return {"status": "optimized"}
```

### Flask/FastAPI Integration

```python
from prometheus_client import generate_latest, REGISTRY
from monitoring.metrics import track_request_metrics

@app.route('/metrics')
def metrics():
    return generate_latest(REGISTRY)

@app.route('/api/v1/orchestrate', methods=['POST'])
@track_request_metrics('POST', '/api/v1/orchestrate')
async def orchestrate():
    # Your endpoint logic
    pass
```

## Alert Destinations

### Severity Routing

| Severity | Response Time | Channels | Examples |
|----------|---------------|----------|----------|
| CRITICAL | Immediate | PagerDuty + Slack #greenlang-critical | Master orchestrator down, All agents failed |
| WARNING | 15 minutes | Slack #greenlang-ops | Performance degradation, Low efficiency |
| INFO | 1 hour | Email team@greenlang.io | Low savings, Business metrics |

### PagerDuty Integration

```yaml
receivers:
  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<GL-001-SERVICE-KEY>'
        severity: 'critical'
        details:
          firing: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
          resolved: 'Alert resolved'
```

### Slack Integration

```yaml
receivers:
  - name: 'slack-ops'
    slack_configs:
      - api_url: '<SLACK-WEBHOOK-URL>'
        channel: '#greenlang-gl-001'
        title: 'GL-001 Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

## Dashboard Access

- **Executive Dashboard**: https://grafana.greenlang.io/d/gl-001-executive
- **Master Orchestrator**: https://grafana.greenlang.io/d/gl-001-master
- **Multi-Plant Operations**: https://grafana.greenlang.io/d/gl-001-multi-plant
- **Sub-Agent Coordination**: https://grafana.greenlang.io/d/gl-001-subagents
- **Thermal Efficiency**: https://grafana.greenlang.io/d/gl-001-thermal
- **Operations Dashboard**: https://grafana.greenlang.io/d/gl-001-operations

## Common PromQL Queries

### Orchestrator Health
```promql
# Orchestrator availability (30-day)
100 * avg_over_time(up{job="gl-001-orchestrator"}[30d])

# Current error rate
100 * sum(rate(gl_001_orchestration_requests_total{status="failure"}[5m]))
/ sum(rate(gl_001_orchestration_requests_total[5m]))

# P95 orchestration latency
histogram_quantile(0.95,
  sum(rate(gl_001_orchestration_duration_seconds_bucket[5m])) by (le, orchestration_type)
)
```

### Multi-Plant Performance
```promql
# Average plant efficiency
avg(gl_001_plant_thermal_efficiency_percent)

# Total heat generation across all plants
sum(gl_001_plant_heat_generation_mw)

# Plants with efficiency < 80%
count(gl_001_plant_thermal_efficiency_percent < 80)

# Heat loss breakdown
sum(gl_001_plant_heat_losses_mw) by (loss_type)
```

### Sub-Agent Coordination
```promql
# Sub-agent availability percentage
100 * (gl_001_active_subagents_count / 99)

# Average agent response time
avg(rate(gl_001_subagent_response_time_seconds_sum[5m])
  / rate(gl_001_subagent_response_time_seconds_count[5m]))

# Task completion rate
sum(rate(gl_001_tasks_completed_total[5m])) by (status)

# Agents with highest queue depth
topk(10, gl_001_subagent_message_queue_depth)
```

### SCADA Integration
```promql
# SCADA connection health
sum(gl_001_scada_connection_status) by (plant_id, scada_system)

# Data quality by plant
avg(gl_001_scada_data_quality_percent) by (plant_id)

# Data ingestion rate
sum(rate(gl_001_scada_data_points_received_total[5m])) by (plant_id)

# Active critical alarms
sum(gl_001_scada_alarms_active{severity="critical"})
```

### Business Metrics
```promql
# Total annual savings
sum(gl_001_optimization_annual_savings_usd)

# Total CO2 reduction
sum(gl_001_optimization_annual_emissions_reduction_tons)

# Plant ROI ranking
topk(10, gl_001_optimization_annual_savings_usd)

# Average capacity utilization
avg(gl_001_plant_capacity_utilization_percent)
```

## Service Level Objectives (SLOs)

### Availability SLO: 99.9%
```promql
# 30-day availability
100 * avg_over_time(up{job="gl-001-orchestrator"}[30d])

# Alert if < 99.9%
100 * avg_over_time(up{job="gl-001-orchestrator"}[30d]) < 99.9
```

### Latency SLO: P95 < 5 seconds
```promql
# P95 orchestration latency
histogram_quantile(0.95,
  sum(rate(gl_001_orchestration_duration_seconds_bucket[1h])) by (le)
)

# Alert if > 5s
histogram_quantile(0.95,
  sum(rate(gl_001_orchestration_duration_seconds_bucket[1h])) by (le)
) > 5
```

### Error Budget: 0.1% (30-day)
```promql
# 30-day error rate
100 * sum(increase(gl_001_orchestration_requests_total{status="failure"}[30d]))
/ sum(increase(gl_001_orchestration_requests_total[30d]))

# Alert if > 0.1%
100 * sum(increase(gl_001_orchestration_requests_total{status="failure"}[30d]))
/ sum(increase(gl_001_orchestration_requests_total[30d])) > 0.1
```

## Runbooks

### GL001MasterOrchestratorDown
https://runbooks.greenlang.io/gl-001-orchestrator-down

**Symptoms**: Master orchestrator unreachable, all plant operations halted

**Impact**: Complete system outage, 99 sub-agents orphaned

**Resolution**:
1. Check pod status: `kubectl get pods -n greenlang -l app=gl-001`
2. View logs: `kubectl logs -n greenlang <pod-name> --tail=100`
3. Restart if crashed: `kubectl rollout restart deployment/gl-001-orchestrator -n greenlang`
4. Escalate if persistent failures

### GL001AllSubAgentsFailed
https://runbooks.greenlang.io/gl-001-all-agents-failed

**Symptoms**: No sub-agents responding, coordination failures

**Impact**: Zero operational capacity

**Resolution**:
1. Check message bus health
2. Verify network connectivity to sub-agents
3. Check sub-agent deployment status
4. Review recent configuration changes
5. Rollback if recent deploy

## Troubleshooting

### High Memory Usage
```bash
# Check memory consumption
kubectl top pod -n greenlang -l app=gl-001

# View memory metrics
curl http://localhost:8000/metrics | grep memory_usage_bytes

# Check for memory leaks in cache
# Expected cache size: < 1000 entries
```

### High Task Queue Depth
```bash
# Check queue depths
curl http://localhost:8000/metrics | grep task_queue_depth

# Identify bottleneck agents
kubectl logs -n greenlang gl-001-xxx | grep "task_delegation"

# Consider horizontal scaling
kubectl scale deployment gl-001-orchestrator --replicas=3 -n greenlang
```

### SCADA Connection Issues
```bash
# Check SCADA connectivity per plant
curl http://localhost:8000/metrics | grep scada_connection_status

# Review SCADA integration logs
kubectl logs -n greenlang gl-001-xxx | grep "SCADA"

# Verify network rules allow SCADA ports
# Common SCADA ports: OPC UA (4840), Modbus (502), DNP3 (20000)
```

## Support

- **Documentation**: https://docs.greenlang.io/gl-001
- **Runbooks**: https://runbooks.greenlang.io/gl-001
- **On-Call**: PagerDuty - GL-001 Escalation Policy
- **Slack**: #greenlang-gl-001
- **Email**: ops@greenlang.io

## Monitoring Checklist

- [ ] Prometheus scraping GL-001 metrics every 15s
- [ ] AlertManager receiving alerts
- [ ] PagerDuty integration configured
- [ ] Slack notifications working
- [ ] Grafana dashboards imported
- [ ] SLO dashboards configured
- [ ] Runbooks accessible
- [ ] On-call rotation active
- [ ] Backup alerting channels configured
- [ ] Metrics retention: 30 days (Prometheus), 1 year (Thanos)

---

**Version**: 1.0.0
**Last Updated**: 2025-11-17
**Maintained By**: GreenLang DevOps Team
**Contact**: ops@greenlang.io
