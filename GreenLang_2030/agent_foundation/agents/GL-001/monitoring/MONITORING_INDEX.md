# GL-001 ProcessHeatOrchestrator - Monitoring Stack Index

## Overview

Complete monitoring and observability infrastructure for GL-001 Master Orchestrator managing 99 sub-agents and multi-plant industrial process heat operations.

**Total Coverage**: 120+ metrics, 40+ alerts, 5 dashboards, comprehensive runbooks

---

## File Inventory

### Core Monitoring Files

#### 1. metrics.py (1000+ lines)
**Location**: `monitoring/metrics.py`
**Purpose**: Prometheus metrics definitions and collectors
**Metrics Count**: 120+
**Categories**:
- Master orchestrator metrics (10 metrics)
- Multi-plant coordination (15 metrics)
- Sub-agent coordination (12 metrics)
- SCADA integration per plant (10 metrics)
- ERP integration (8 metrics)
- Thermal efficiency (12 metrics)
- Heat distribution (8 metrics)
- Energy balance (10 metrics)
- Emissions compliance (10 metrics)
- Task delegation (8 metrics)
- Performance metrics (10 metrics)
- Determinism tracking (6 metrics)
- Business metrics (8 metrics)
- System resources (5 metrics)

**Key Classes**:
- `MetricsCollector` - Main collector class with static methods
- Decorators: `@track_request_metrics`, `@track_orchestration_metrics`

**Integration**:
```python
from monitoring.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.update_orchestrator_state("EXECUTING", is_healthy=True)
metrics.update_plant_metrics("PLANT-001", "Houston Complex", {...})
```

#### 2. alerts/prometheus_rules.yaml (400+ lines)
**Location**: `monitoring/alerts/prometheus_rules.yaml`
**Purpose**: Prometheus alerting rules
**Alert Count**: 40+
**Severity Levels**:
- CRITICAL (12 alerts) - Immediate response required
- WARNING (15 alerts) - Investigation within 15 minutes
- BUSINESS (3 alerts) - Strategic monitoring
- SLO (4 alerts) - Service level objectives
- INTEGRATION (3 alerts) - Integration health
- COORDINATION (3 alerts) - Multi-agent coordination

**Critical Alerts**:
- GL001MasterOrchestratorDown
- GL001AllSubAgentsFailed
- GL001MultiPlantHeatLoss
- GL001CriticalPlantFailure
- GL001SCADAConnectionLost
- GL001ERPIntegrationFailure
- GL001EmissionsComplianceViolation

**SLO Alerts**:
- Availability: 99.9% (30-day)
- Latency: P95 < 5s
- Error Budget: < 0.1% (30-day)
- Sub-Agent Availability: > 95%

#### 3. README.md (comprehensive guide)
**Location**: `monitoring/README.md`
**Purpose**: Complete monitoring documentation
**Sections**:
- Quick start guide
- Architecture overview
- Component descriptions
- Integration examples (Python, Flask/FastAPI)
- Alert routing and destinations
- Common PromQL queries
- Service Level Objectives
- Runbook links
- Troubleshooting guide

**Key Content**:
- PagerDuty integration
- Slack notifications
- Grafana dashboard links
- PromQL query library
- Troubleshooting scenarios

#### 4. grafana/DASHBOARD_SPECIFICATIONS.md
**Location**: `monitoring/grafana/DASHBOARD_SPECIFICATIONS.md`
**Purpose**: Detailed dashboard specifications
**Dashboards**: 5 comprehensive dashboards
**Content**:
- Panel layouts and queries
- Visualization types
- Alert annotations
- Template variables
- Import instructions

**Dashboard List**:
1. Master Orchestrator Dashboard - DevOps operations
2. Multi-Plant Dashboard - Plant managers
3. Sub-Agent Coordination Dashboard - Agent monitoring (GL-002 to GL-100)
4. Thermal Efficiency Dashboard - Energy managers
5. Operations Dashboard - 24/7 NOC team

---

## Metrics Categories

### 1. Master Orchestrator Metrics (10 metrics)
```promql
gl_001_orchestrator_health_status
gl_001_orchestrator_uptime_seconds
gl_001_orchestration_requests_total{orchestration_type, status}
gl_001_orchestration_duration_seconds
gl_001_orchestration_complexity_score
gl_001_orchestrator_state{state_name}
```

### 2. Multi-Plant Coordination (15 metrics)
```promql
gl_001_active_plants_count
gl_001_plant_health_status{plant_id, plant_name, location}
gl_001_plant_thermal_efficiency_percent{plant_id, plant_name}
gl_001_plant_heat_generation_mw{plant_id, plant_name}
gl_001_plant_heat_demand_mw{plant_id, plant_name}
gl_001_plant_heat_losses_mw{plant_id, plant_name, loss_type}
gl_001_plant_capacity_utilization_percent{plant_id, plant_name}
gl_001_cross_plant_heat_transfer_mw{source_plant, destination_plant}
```

### 3. Sub-Agent Coordination (12 metrics)
```promql
gl_001_active_subagents_count{agent_category}
gl_001_subagent_health_status{agent_id, agent_type, plant_id}
gl_001_subagent_response_time_seconds{agent_id, agent_type}
gl_001_subagent_task_assignments_total{agent_id, task_type, priority}
gl_001_subagent_task_completion_total{agent_id, task_type, status}
gl_001_subagent_coordination_failures_total{agent_id, failure_type}
gl_001_subagent_message_queue_depth{agent_id}
gl_001_subagent_task_latency_seconds{agent_id, task_type}
```

### 4. SCADA Integration (10 metrics per plant)
```promql
gl_001_scada_connection_status{plant_id, scada_system}
gl_001_scada_data_points_received_total{plant_id, data_category}
gl_001_scada_data_latency_seconds{plant_id, scada_system}
gl_001_scada_data_quality_percent{plant_id, data_category}
gl_001_scada_integration_errors_total{plant_id, error_type}
gl_001_scada_tags_monitored{plant_id}
gl_001_scada_alarms_active{plant_id, severity}
```

### 5. Thermal Efficiency (12 metrics)
```promql
gl_001_aggregate_thermal_efficiency_percent
gl_001_aggregate_heat_generation_mw
gl_001_plant_boiler_efficiency_percent{plant_id, plant_name}
gl_001_plant_heat_recovery_efficiency_percent{plant_id, plant_name}
gl_001_plant_distribution_efficiency_percent{plant_id, plant_name}
gl_001_efficiency_improvement_percent{plant_id, optimization_type}
```

### 6. Emissions Compliance (10 metrics)
```promql
gl_001_emissions_compliance_status{plant_id, regulation}
gl_001_emissions_co2_tons_hr{plant_id}
gl_001_emissions_co2_intensity_kg_mwh{plant_id}
gl_001_emissions_nox_kg_hr{plant_id}
gl_001_emissions_compliance_violations_total{plant_id, pollutant, regulation}
gl_001_carbon_credit_balance_tons{plant_id, credit_type}
```

---

## Alert Summary

### Critical Alerts (P0 - Immediate Response)
| Alert Name | Condition | Duration | Impact |
|------------|-----------|----------|--------|
| GL001MasterOrchestratorDown | up == 0 | 1m | Total system outage |
| GL001AllSubAgentsFailed | active_subagents == 0 | 2m | Zero operational capacity |
| GL001MultiPlantHeatLoss | heat_losses > 100 MW | 5m | Massive energy waste |
| GL001EmissionsComplianceViolation | compliance == 0 | 1m | Regulatory violation |
| GL001SCADAConnectionLost | scada_status == 0 | 1m | Blind operation |
| GL001ERPIntegrationFailure | erp_status == 0 | 5m | Schedule disruption |

### Warning Alerts (P1 - 15 min response)
| Alert Name | Condition | Duration | Impact |
|------------|-----------|----------|--------|
| GL001OrchestratorPerformanceDegradation | Latency +20% | 10m | Degraded performance |
| GL001SubAgentDegradation | Availability < 80% | 5m | Reduced capacity |
| GL001PlantEfficiencyDrop | Efficiency < 70% | 15m | Energy waste |
| GL001HighTaskQueueDepth | Queue > 100 | 10m | Processing bottleneck |

### SLO Alerts
| SLO | Target | Alert Threshold | Duration |
|-----|--------|-----------------|----------|
| Availability | 99.9% | < 99.9% | 5m |
| P95 Latency | < 5s | > 5s | 15m |
| Error Budget | < 0.1% | > 0.1% | 1h |
| Sub-Agent Availability | > 95% | < 95% | 30m |

---

## Dashboard Summary

### 1. Master Orchestrator Dashboard
**Audience**: DevOps, SRE
**Refresh**: 10s
**Panels**: 15 panels in 4 rows
**Key Metrics**:
- Health status and uptime
- Request rate and latency (P50, P95, P99)
- Error rate and error breakdown
- CPU, Memory, Cache performance
- Active operations and queue depth

### 2. Multi-Plant Dashboard
**Audience**: Plant Managers, Operations
**Refresh**: 30s
**Panels**: 12 panels in 5 rows
**Key Metrics**:
- Active plants and fleet efficiency
- Plant health matrix
- Efficiency ranking and heatmap
- Heat generation vs demand
- Cross-plant heat transfer flows

### 3. Sub-Agent Coordination Dashboard
**Audience**: DevOps, Agent Developers
**Refresh**: 15s
**Panels**: 14 panels in 6 rows
**Key Metrics**:
- Agent fleet status (99 agents)
- Agent health matrix (10x10 grid)
- Response time heatmap
- Task delegation and completion
- Message queue depths
- Coordination failures

### 4. Thermal Efficiency Dashboard
**Audience**: Energy Managers
**Refresh**: 1m
**Panels**: 12 panels in 5 rows
**Key Metrics**:
- Fleet thermal efficiency gauge
- Plant efficiency comparison
- Boiler, recovery, distribution efficiency
- Optimization impact tracking
- Energy balance closure

### 5. Operations Dashboard
**Audience**: 24/7 NOC Team
**Refresh**: 15s
**Panels**: 12 panels in 6 rows
**Key Metrics**:
- System health overview
- Integration health (SCADA, ERP)
- Real-time heat generation
- Task queues and processing rate
- Recent errors and optimizations
- Key business metrics

---

## Integration Guide

### Quick Integration (3 steps)

#### Step 1: Import Metrics
```python
from monitoring.metrics import MetricsCollector

# Initialize
metrics = MetricsCollector()
```

#### Step 2: Update Metrics
```python
# Update orchestrator state
metrics.update_orchestrator_state("EXECUTING", is_healthy=True)

# Update plant metrics
metrics.update_plant_metrics(
    plant_id="PLANT-001",
    plant_name="Houston Complex",
    metrics={
        "health_status": "healthy",
        "thermal_efficiency_percent": 87.5,
        "heat_generation_mw": 250.0,
        "heat_demand_mw": 245.0,
        "capacity_utilization_percent": 92.3
    }
)

# Update sub-agent metrics
metrics.update_subagent_metrics(
    agent_id="GL-002",
    agent_type="BoilerEfficiencyOptimizer",
    plant_id="PLANT-001",
    metrics={"health_status": "healthy", "message_queue_depth": 5}
)
```

#### Step 3: Expose Metrics Endpoint
```python
from prometheus_client import generate_latest, REGISTRY
from flask import Flask

app = Flask(__name__)

@app.route('/metrics')
def metrics_endpoint():
    return generate_latest(REGISTRY)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### Decorator Usage
```python
from monitoring.metrics import track_orchestration_metrics

@track_orchestration_metrics('multi_plant_optimization')
async def optimize_all_plants():
    # Automatically tracks latency and status
    return {"status": "optimized"}
```

---

## PromQL Query Library

### Orchestrator Health
```promql
# Uptime in hours
gl_001_orchestrator_uptime_seconds / 3600

# Request rate
sum(rate(gl_001_orchestration_requests_total[5m])) by (orchestration_type)

# Error rate percentage
100 * sum(rate(gl_001_orchestration_requests_total{status="failure"}[5m]))
/ sum(rate(gl_001_orchestration_requests_total[5m]))

# P95 latency
histogram_quantile(0.95,
  sum(rate(gl_001_orchestration_duration_seconds_bucket[5m])) by (le)
)
```

### Multi-Plant Operations
```promql
# Fleet efficiency average
avg(gl_001_plant_thermal_efficiency_percent)

# Total heat generation
sum(gl_001_plant_heat_generation_mw)

# Plants below target efficiency
count(gl_001_plant_thermal_efficiency_percent < 85)

# Heat losses by type
sum(gl_001_plant_heat_losses_mw) by (loss_type)
```

### Sub-Agent Coordination
```promql
# Agent availability percentage
100 * (gl_001_active_subagents_count / 99)

# Average agent response time
avg(rate(gl_001_subagent_response_time_seconds_sum[5m])
  / rate(gl_001_subagent_response_time_seconds_count[5m]))

# Task success rate
100 * sum(rate(gl_001_tasks_completed_total{status="success"}[5m]))
/ sum(rate(gl_001_tasks_completed_total[5m]))

# Top 10 agents by queue depth
topk(10, gl_001_subagent_message_queue_depth)
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] Review metrics.py - verify all metrics are properly labeled
- [ ] Review prometheus_rules.yaml - validate alert thresholds
- [ ] Test alert routing (PagerDuty, Slack, Email)
- [ ] Import Grafana dashboards
- [ ] Configure dashboard permissions

### Deployment
- [ ] Deploy ServiceMonitor for Prometheus scraping
- [ ] Apply Prometheus rules to AlertManager
- [ ] Verify metrics endpoint is accessible
- [ ] Import all 5 Grafana dashboards
- [ ] Test alert firing and resolution

### Post-Deployment
- [ ] Verify Prometheus is scraping metrics (15s interval)
- [ ] Confirm alerts are evaluating correctly
- [ ] Test PagerDuty integration with test alert
- [ ] Verify Grafana dashboards render correctly
- [ ] Train operations team on dashboards
- [ ] Document runbook procedures
- [ ] Set up on-call rotation

---

## Support & Resources

### Documentation
- **Main README**: `monitoring/README.md`
- **Dashboard Specs**: `monitoring/grafana/DASHBOARD_SPECIFICATIONS.md`
- **This Index**: `monitoring/MONITORING_INDEX.md`

### External Resources
- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **PromQL Tutorial**: https://prometheus.io/docs/prometheus/latest/querying/basics/

### Contact
- **DevOps Team**: devops@greenlang.io
- **On-Call**: PagerDuty - GL-001 Escalation Policy
- **Slack**: #greenlang-gl-001
- **Runbooks**: https://runbooks.greenlang.io/gl-001

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-11-17 | Initial monitoring stack | GreenLang DevOps |

---

**Status**: Production Ready
**Last Updated**: 2025-11-17
**Maintained By**: GreenLang DevOps Team
